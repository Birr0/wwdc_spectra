import math 

# Need to make a torch model wrapper for spender
import torch
import torch.nn as nn
import lightning as L
import numpy as np

from torch.func import jvp
from torch.autograd import Function
import torch.nn.functional as F
from torch import Tensor
from wwdc_spectra.models.modules import get_conditional_len, BaseModel
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from timm.layers import trunc_normal_
from huggingface_hub import PyTorchModelHubMixin

class VelocityField(nn.Module, PyTorchModelHubMixin):
    def __init__(self, code_dim, hidden_dim, conditional_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.vf = nn.Sequential(  # vector field
            nn.Linear(code_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )

        self.FiLM_params = nn.Linear(conditional_dim, 2 * hidden_dim)

        self.null_y = nn.Embedding(
            num_embeddings=1,
            embedding_dim=conditional_dim,
        )

    def forward(self, t: Tensor, x_t: Tensor, y: Tensor):
        # flag here if you want conditional and unconditional output.
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])

        null_vector = self.null_y(
            torch.zeros(y.size(0), dtype=torch.long, device=x_t.device)
        )
        x_t = torch.cat([x_t, x_t], dim=0)
        t = torch.cat([t, t], dim=0).unsqueeze(-1)
        x_t = torch.cat([x_t, t], dim=1)

        y = torch.cat([y, null_vector])
        gamma, beta = self.FiLM_params(y).chunk(2, dim=1)
        for idx, layer in enumerate(self.vf):
            x_t = layer(x_t)
            if idx != len(self.vf) - 1 and isinstance(layer, nn.Linear):
                x_t = gamma * x_t + beta
        return x_t

class WrappedModel(nn.Module):
    """Wrapper around velocity model to inject month condition during inference.
    Implements classifier-free guidance according to the formula:
    u ← (1-w)*u_null + w*u_cond
    where:
    - u_null is the velocity with condition dropped
    - u_cond is the velocity with condition intact
    - w is the cfg_scale (default=1.0, which means no guidance)
    """

    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(self, x, t, **model_extras):
        """Forward pass with classifier-free guidance.

        Args:
            x: Input tensor (batch_size, ...)
            t: Time tensor (batch_size, ) or ()

        Returns:
            Predicted velocity with CFG applied if cfg_scale > 1.0
        """
        cfg_scale = model_extras["cfg_scale"]

        if "r" in model_extras:
            v = self.velocity_model(
                x_t=x, t=t, r=model_extras["r"], y=model_extras["y"]
            )

        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        v = self.velocity_model(x_t=x, t=t, y=model_extras["y"])
        v_cond, v_uncond = torch.chunk(v, chunks=2, dim=0)
        return (1 - cfg_scale) * v_uncond + cfg_scale * v_cond


class LightningFlowMatching(L.LightningModule):
    def __init__(
        self,
        base_model: BaseModel,
        lr,
        batch_size,
        code_dim,
        hidden_dim,
        catalog,
        n_steps=10,
        ckpt_path: str = None,
        method="midpoint"
    ):
        super().__init__()

        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.lr = lr

        # --- Models --- #
        self.vf = VelocityField(code_dim, hidden_dim, get_conditional_len(catalog))
        self.vf.apply(self._init_weights)
        self.base_model = base_model

        # --- Load Checkpoints --- #
        if ckpt_path:
            self.vf_state_dict = torch.load(ckpt_path)[
                "state_dict"
            ]  # map_location="cpu"
            self.load_state_dict(self.vf_state_dict, strict=False)
            print("✅ Loaded state dict from checkpoint.")
            self.wrapped_vf = WrappedModel(self.vf)
            # ODE solver hparams
            self.n_steps = n_steps
            self.solver = ODESolver(velocity_model=self.wrapped_vf)
            self.wrapped_vf = WrappedModel(self.vf)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.method = method
        self.step_size = 1./n_steps

    @property
    def T(self):
        return torch.tensor([1., 0.], device=self.device)
        #torch.linspace(1, 0, self.n_steps, device=self.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        params = list(self.vf.parameters())

        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )


    def base_step(self, batch, partition):
        X, y = batch

        z = self.base_model.encode(X)

        x_0 = torch.randn_like(z)
        t = torch.rand(z.shape[0], device=z.device)

        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=z)

        # flow matching l2 loss
        ut = self.vf(x_t=path_sample.x_t, y=y, t=path_sample.t)

        u_target = torch.cat([path_sample.dx_t, path_sample.dx_t], dim=0)
        loss = torch.pow(
            ut - u_target,
            2,
        ).mean()
        self.log(f"{partition}_loss", loss)

        return loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch, _batch_idx):
        return self.base_step(batch, "test")

    def predict_step(self, X, y, embed_opt=["cond"]):
        self.eval()
        with torch.no_grad():
            output = {}
            code = self.base_model.encode(X)
            if "orig" in embed_opt:
                output["orig"] = code
            
            # could reduce this to a single forward pass.
            if "cond" in embed_opt:
                output["cond"] = self.solver.sample(
                    x_init=code,
                    step_size=self.step_size,
                    y=y,
                    cfg_scale=1.0,
                    time_grid=self.T,
                    method=self.method,
                )

            if "uncond" in embed_opt:
                output["uncond"] = self.solver.sample(
                    x_init=code,
                    step_size=self.step_size,
                    y=y,
                    cfg_scale=0.0,
                    time_grid=self.T,
                    method=self.method,
                )
        return output

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Multiply gradient by -alpha
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class AdversarialRegressor(nn.Module):
    def __init__(self, code_dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
            nn.Sigmoid() # squash [0,1]
        )

    def forward(self, x):
        return self.net(x)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        #use_cfg_embedding = dropout_prob > 0
        #self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.embedding = nn.Linear(
            cond_dim, hidden_size
        )

    def forward(self, y):
        return self.embedding(y)


class AdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # We predict scale (gamma) and shift (beta)
        self.ada_lin = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3*hidden_dim,  bias=True)
        )

    def forward(self, x, cond_emb):
        # Generate params from conditioning
        gamma, beta, gate = self.ada_lin(cond_emb).chunk(3, dim=-1)
    
        # Apply normalization first, then scale and shift
        x = self.norm(x)
        x = x * (1 + gamma) + beta # (1 + gamma) helps initialization stay near identity
        return self.linear(x), gate # return gating mechanism

class MeanVelocityField(nn.Module, PyTorchModelHubMixin):
    def __init__(self, code_dim, hidden_dim, conditional_dim, n_hidden=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = 2

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.r_embedder = TimestepEmbedder(hidden_dim)

        self.input_proj = nn.Linear(code_dim, hidden_dim)

        self.act = nn.SiLU()
        self.ada_lns = nn.ModuleList(
            [AdaLN(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        )
        self.out_proj = nn.Linear(hidden_dim, code_dim) 

        self.cond_embed = ConditionEmbedder(
            conditional_dim, hidden_dim
        )
        self.null_y = nn.Embedding(
            num_embeddings=1,
            embedding_dim=conditional_dim,
        )
        # parameter for y embeddings

    def forward(self, x_t: Tensor, r: Tensor, t: Tensor, y: Tensor):
        t_embed = self.t_embedder(t).flatten(start_dim=1)
        r_embed = self.r_embedder(t - r).flatten(start_dim=1)
        y_embed = self.cond_embed(y)

        x = self.input_proj(x_t)
        c = t_embed + r_embed + y_embed

        for adaln, lin in zip(self.ada_lns, self.linears):
            identity = x
            modulated, gate = adaln(x, c) # replace with c here.
            x = self.act(lin(modulated))
            x = identity + gate*x
        return self.out_proj(x)

class LightningMeanFlowMatching(L.LightningModule):
    def __init__(
        self,
        base_model: BaseModel,
        lr,
        batch_size,
        code_dim,
        hidden_dim,
        catalog,
        n_steps=10,
        ckpt_path: str = None,
    ):
        super().__init__()

        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.lr = lr
        self.ratio_r_not_equal_t = 0.5  # add to config

        # --- Models --- #
        self.conditional_dim = get_conditional_len(catalog)
        self.vf = MeanVelocityField(code_dim, hidden_dim, self.conditional_dim)
        #self.adversary = AdversarialRegressor(code_dim, self.conditional_dim, hidden_dim)
        self._init_weights()
        self.base_model = base_model

        # --- Load Checkpoints --- #
        if ckpt_path:
            self.vf_state_dict = torch.load(ckpt_path)[
                "state_dict"
            ]  # map_location="cpu"
            self.load_state_dict(self.vf_state_dict, strict=False)
            print("✅ Loaded state dict from checkpoint.")
            self.wrapped_vf = WrappedModel(self.vf)
            # ODE solver hparams
            self.n_steps = n_steps
            self.solver = ODESolver(velocity_model=self.wrapped_vf)
            
        #self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.path_type = "linear"

        self.max_alpha = 1.0
        self.alpha_ramp_steps = 20000
        self.adv_start_step = 0 #5000
        self._lambda = 1.

    @property
    def T(self):
        return torch.linspace(1, 0, self.n_steps, device=self.device)

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.vf.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.vf.cond_embed.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.vf.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.vf.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.vf.r_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.vf.r_embedder.mlp[2].weight, std=0.02)
        
        for adaln in self.vf.ada_lns:
            # zero-out final layer of the AdaLN modulation
            # i.e. the gate so that initially we return the identity.
            nn.init.constant_(adaln.ada_lin[-1].weight, 0) 
            nn.init.constant_(adaln.ada_lin[-1].bias, 0)

        # Zero-out the output projection
        # This makes the initial velocity predictions zero
        nn.init.constant_(self.vf.out_proj.weight, 0)
        nn.init.constant_(self.vf.out_proj.bias, 0)

        '''
        for m in self.adversary.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        '''

    def configure_optimizers(self):
        params = list(self.vf.parameters()) #+ list(self.adversary.parameters())

        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )

    '''
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer, 
            gradient_clip_val=0.5, 
            gradient_clip_algorithm="norm"
        )
    '''
    
    '''
    def get_alpha(self):
        if self.global_step < self.alpha_ramp_steps:
            return self.max_alpha * (self.global_step / self.alpha_ramp_steps)
        return self.max_alpha
    '''
    
    def get_alpha(self):
        if self.global_step < self.adv_start_step:
            return 0.0
        s = self.global_step - self.adv_start_step
        if s < self.alpha_ramp_steps:
            return self.max_alpha * (s / self.alpha_ramp_steps)
        return self.max_alpha
        
    
    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def hsic(self, X, Y, sigma_x=-1, sigma_y=-1):
        """
        Calculates the Hilbert-Schmidt Independence Criterion (unbiased estimator).
        X: (Batch_size, Features_X)
        Y: (Batch_size, Features_Y)
        """
        X = X.flatten(start_dim=1)
        Y = Y.flatten(start_dim=1)
        b = X.shape[0]

        if b <= 3:
            raise ValueError(f"Batch size must be > 3 for unbiased HSIC, got {b}")

        # Helper to compute Gaussian RBF Kernel
        def compute_kernel(data, sigma):
            # Compute squared Euclidean distance matrix: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            dist_sq = torch.cdist(data, data, p=2)**2
            
            if sigma < 0:
                # Median trick: sigma^2 = median of squared distances
                v = torch.quantile(dist_sq, 0.5)
            else:
                v = sigma**2
            
            v = v.detach()
                
            # RBF Kernel and set diagonal to zero for unbiased estimator
            K = torch.exp(-0.5 * dist_sq / v)
            mask = 1.0 - torch.eye(b, device=data.device)
            return K * mask

        K = compute_kernel(X, sigma_x)
        L = compute_kernel(Y, sigma_y)

        # Core HSIC math (Song et al. 2012)
        KL = K @ L
        trace = KL.trace()
        second_term = K.sum() * L.sum() / ((b - 1) * (b - 2))
        third_term = KL.sum() / (b - 2)

        hsic_val = (trace + second_term - 2.0 * third_term) / (b * (b - 3))
        return torch.clamp(hsic_val, min=0.0)

    def base_step(self, batch, partition):
        # Fix the shape issues
        X, y = batch

        with torch.no_grad():
            z = self.base_model.encode(X)
    

        time_samples = torch.rand(
            z.size(0), 2, device=self.device
        )
        # sort t and r.
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0].view(-1, 1), sorted_samples[:, 1].view(-1, 1)
        fraction_equal = 1.0 - self.ratio_r_not_equal_t 
        equal_mask = torch.rand(z.size(0), 1, device=self.device) < fraction_equal
        r = torch.where(equal_mask, t, r)

        noises = torch.randn_like(z)
        
        # Calculate interpolation and z_t
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)
        alpha_r, sigma_r, d_alpha_r, d_sigma_r = self.interpolant(r)

        z_t = alpha_t * z + sigma_t * noises
        z_r = alpha_r * z + sigma_r * noises

        # Calculate instantaneous velocity v_t 
        v_t = d_alpha_t * z + d_sigma_t * noises

        u = self.vf(z_t, r, t, y)
        u_r = self.vf(z_r, t, r, y)
     
        def fn(x, cur_r, cur_t):
            return self.vf(x, cur_r, cur_t, y)
        
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
    
        _, dudt = jvp(
            fn, primals, tangents
        )

        u_target = v_t - (t - r)*dudt
        
        error = u - u_target.detach()
       

        loss_mean_ref = torch.mean((error**2))
        loss_sym = torch.mean(((t - r)*u - (r - t)*u_r)**2)
        z_base = z + self.vf(z, torch.ones_like(r), torch.zeros_like(t), y)

        # Adversarial set-up
        current_alpha = self.get_alpha()
        #z_reversed = grad_reverse(z_base, current_alpha)

        #y_hat = self.adversary(z_reversed)
        #loss_adv = F.mse_loss(y_hat, y)

        HSIC = self.hsic(z_base, y)

        total_loss = loss_mean_ref + current_alpha*loss_sym + current_alpha*HSIC
        self.log(f"{partition}_loss", loss_mean_ref)
        self.log(f"{partition}_hsic", HSIC)
        self.log(f"{partition}_step", self.global_step)
        self.log(f"{partition}_alpha", current_alpha)
        self.log(f"{partition}_sym_loss", loss_sym)
        self.log(f"{partition}_total_loss", total_loss)

        return total_loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch, _batch_idx):
        return self.base_step(batch, "test")

    def predict_step(self, X, y, embed_opt=["cond"]):
        self.eval()
        with torch.no_grad():
            output = {}
            code = self.base_model.encode(X)
            if "orig" in embed_opt:
                output["orig"] = code

            # could reduce this to a single forward pass.
            if "cond" in embed_opt:
                output["cond"] = self.solver.sample(
                    x_init=code,
                    step_size=None,
                    y=y,
                    cfg_scale=1.0,
                    time_grid=self.T,
                    method="midpoint",
                )

            if "uncond" in embed_opt:
                output["uncond"] = self.solver.sample(
                    x_init=code,
                    step_size=None,
                    y=y,
                    cfg_scale=0.0,
                    time_grid=self.T,
                    method="midpoint",
                )
        return output

class LightningAlphaFlowMatching(L.LightningModule):
    def __init__(
        self,
        base_model: BaseModel,
        lr,
        batch_size,
        code_dim,
        hidden_dim,
        catalog,
        n_steps=10,
        ckpt_path: str = None,
        k_s=0,
        k_e=10000,
        gamma=25,
        eta=5e-3,
        beta_start_step=10000,
        beta_ramp_steps=10000,
        max_beta=1.0

    ):
        super().__init__()

        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.lr = lr

        # --- Models --- #
        self.conditional_dim = get_conditional_len(catalog)
        self.vf = MeanVelocityField(code_dim, hidden_dim, self.conditional_dim)
        #self.adversary = AdversarialRegressor(code_dim, self.conditional_dim, hidden_dim)
        
        self.base_model = base_model

        # --- Load Checkpoints --- #
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint["state_dict"]
            self.load_state_dict(state_dict, strict=True)
            print("✅ Loaded state dict from checkpoint.")
            self.wrapped_vf = WrappedModel(self.vf)
            # ODE solver hparams
            self.n_steps = n_steps
            #self.solver = ODESolver(velocity_model=self.wrapped_vf)
        #self.path = AffineProbPath(scheduler=CondOTScheduler())
        else:
            self._init_weights()

        self.path_type = "linear"
        self._lambda = 1.

        self.sigmoid = nn.Sigmoid()

        # alpha schedule
        self.k_e = k_e # end schedule
        self.k_s = k_s # start schedule
        self.gamma = gamma # temperature parameter gamma=25 in paper
        self.eta = eta # clamping value

        # beta schedule
        self.max_beta = max_beta
        self.beta_start_step = beta_start_step
        self.beta_ramp_steps = beta_ramp_steps

    @property
    def T(self):
        return torch.linspace(1, 0, self.n_steps, device=self.device)

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.vf.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.vf.cond_embed.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.vf.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.vf.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.vf.r_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.vf.r_embedder.mlp[2].weight, std=0.02)
        
        for adaln in self.vf.ada_lns:
            # zero-out final layer of the AdaLN modulation
            # i.e. the gate so that initially we return the identity.
            nn.init.constant_(adaln.ada_lin[-1].weight, 0) 
            nn.init.constant_(adaln.ada_lin[-1].bias, 0)

        # Zero-out the output projection
        # This makes the initial velocity predictions zero
        nn.init.constant_(self.vf.out_proj.weight, 0)
        nn.init.constant_(self.vf.out_proj.bias, 0)

        '''
        for m in self.adversary.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        '''

    def configure_optimizers(self):
        params = list(self.vf.parameters()) #+ list(self.adversary.parameters())

        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )

    '''
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer, 
            gradient_clip_val=0.5, 
            gradient_clip_algorithm="norm"
        )
    '''

    def sample_alpha(self, k):
        # need to make this adjustable in the config.
        # ciriculium learning for alpha flow
        # k is current iteration 
        scale = 1/ (self.k_e - self.k_s)
        offset = - (self.k_e + self.k_s) / 2 / (self.k_e - self.k_s)
        x = torch.tensor((scale * k + offset) * self.gamma, device=self.device)
        alpha = 1 - self.sigmoid(x) 

        alpha = 1 if alpha > (1 - self.eta) else (0 if alpha < self.eta else alpha)
        return alpha

    def get_beta(self):
        # beta parameters for HSIC and symmetry.
        # config these parameters.

        if self.global_step < self.beta_start_step:
            return 0.0
        s = self.global_step - self.beta_start_step
        if s < self.beta_ramp_steps:
            return self.max_beta * (s / self.beta_ramp_steps)
        return self.max_beta
    
    def interpolant(self, t):
        """Define interpolation function"""
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def hsic(self, X, Y, sigma_x=-1, sigma_y=-1):
        """
        Calculates the Hilbert-Schmidt Independence Criterion (unbiased estimator).
        X: (Batch_size, Features_X)
        Y: (Batch_size, Features_Y)
        """
        X = X.flatten(start_dim=1)
        Y = Y.flatten(start_dim=1)
        b = X.shape[0]

        if b <= 3:
            raise ValueError(f"Batch size must be > 3 for unbiased HSIC, got {b}")

        # Helper to compute Gaussian RBF Kernel
        def compute_kernel(data, sigma):
            # Compute squared Euclidean distance matrix: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            dist_sq = torch.cdist(data, data, p=2)**2
            
            if sigma < 0:
                # Median trick: sigma^2 = median of squared distances
                v = torch.quantile(dist_sq, 0.5)
            else:
                v = sigma**2
            
            v = v.detach()
                
            # RBF Kernel and set diagonal to zero for unbiased estimator
            K = torch.exp(-0.5 * dist_sq / v)
            mask = 1.0 - torch.eye(b, device=data.device)
            return K * mask

        K = compute_kernel(X, sigma_x)
        L = compute_kernel(Y, sigma_y)

        # Core HSIC math (Song et al. 2012)
        KL = K @ L
        trace = KL.trace()
        second_term = K.sum() * L.sum() / ((b - 1) * (b - 2))
        third_term = KL.sum() / (b - 2)

        hsic_val = (trace + second_term - 2.0 * third_term) / (b * (b - 3))
        return torch.clamp(hsic_val, min=0.0)

    def base_step(self, batch, partition):
        # Fix the shape issues
        X, y = batch

        with torch.no_grad():
            z = self.base_model.encode(X)


        time_samples = torch.rand(
            z.size(0), 2, device=self.device
        )
        # sort t and r.
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0].view(-1, 1), sorted_samples[:, 1].view(-1, 1)

        alpha = self.sample_alpha(self.global_step)
        s = alpha * r + (1 - alpha) * t
        e = torch.randn_like(z)

        z_t = (1 - t) * z + t * e # t=0 is data
        z_r = (1 - r) * z + r * e # t=0 is data
        v = e - z

        primals = (z_t, r, t)
        tangents = (v, torch.zeros_like(r), torch.ones_like(t))

        def fn(x, cur_r, cur_t):
            return self.vf(x, cur_r, cur_t, y)

        u = fn(z_t, r, t)

        if alpha == 0:
            _, dudt = jvp(fn, primals, tangents)
            u_tgt = v - (t - r)* dudt
        else:
            z_s = z_t - (t - s)*v
            u_tgt = alpha * v + (1 - alpha) * fn(z_s, r, s)

        error = u - u_tgt.detach()
        alpha_loss = torch.mean((error**2))
        current_beta = self.get_beta()

        u_r = self.vf(z_r, t, r, y)
        loss_sym = torch.mean(((t - r)*u - (r - t)*u_r)**2)
    
        z_base = z - self.vf(z, torch.ones_like(r), torch.zeros_like(t), y)
        HSIC = self.hsic(z_base, y)

        total_loss = alpha_loss + current_beta*loss_sym + current_beta*HSIC

        #print(f"HSIC: {HSIC}")
        #print(f"Sym loss: {loss_sym}")

        self.log(f"{partition}_loss", alpha_loss)
        self.log(f"{partition}_step", self.global_step)
        self.log(f"{partition}_alpha", alpha)
        self.log(f"{partition}_total_loss", total_loss)
        self.log(f"{partition}_beta", current_beta)
        self.log(f"{partition}_loss_sym", loss_sym)
        self.log(f"{partition}_hsic", HSIC)
    
        return total_loss #, HSIC, alpha_loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch, _batch_idx):
        return self.base_step(batch, "test")

    def predict_step(self, X, y, embed_opt=["cond"]):
        self.eval()
        with torch.no_grad():
            output = {}
            s = self.base_model.encode(X)
            if "orig" in embed_opt:
                output["orig"] = s

            # Simple Euler scheme
            dt = 1.0 / self.n_steps
            s_curr = s.clone()  # Starting at Data (t=0)
            for i in range(self.n_steps):
                # Calculate current time (0.0, 0.05, 0.1, ..., 0.95)
                t_val = i * dt
                t = torch.ones(s_curr.shape[0], device=self.device) * t_val
                with torch.no_grad():
                    # Velocity pointing from Data toward Noise
                    v = self.vf(s_curr, t, t + dt, y)
                s_curr = s_curr + v * dt
            output["cond"] = s_curr
        return output

# --- Main Test Script ---
if __name__ == "__main__":
    # --- Mocks for Missing Components ---
    class PyTorchModelHubMixin: pass

    class BaseModel(nn.Module):
        def __init__(self, code_dim):
            super().__init__()
            self.code_dim = code_dim
        def encode(self, x):
            # Mock encoder: projects input to code_dim
            return torch.randn(x.size(0), self.code_dim)

    def get_conditional_len(catalog):
        return 16  # Mock length

    class AffineProbPath:
        def __init__(self, scheduler): pass

    class CondOTScheduler: pass

    # Mock weight init helper
    def trunc_normal_(tensor, std=0.02):
        nn.init.normal_(tensor, std=std)

    # 1. Hyperparameters
    CODE_DIM = 10
    HIDDEN_DIM = 64
    BATCH_SIZE = 512
    INPUT_FEATURES = 128
    CATALOG = {} # Mock catalog

    # 2. Setup Data
    # Mock X: (Batch, Features), Mock y: (Batch, Cond_Dim)
    mock_X = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    mock_y = torch.randn(BATCH_SIZE, get_conditional_len(CATALOG))
    batch = (mock_X, mock_y)

    # 3. Initialize Model
    base_model = BaseModel(code_dim=CODE_DIM)
    
    model = LightningAlphaFlowMatching( # LightningMeanFlowMatching(
        base_model=base_model,
        lr=1e-4,
        batch_size=BATCH_SIZE,
        code_dim=CODE_DIM,
        hidden_dim=HIDDEN_DIM,
        catalog=CATALOG
    )

    print("🚀 Initializing test run...")

    # 4. Run a Training Step
    try:
        loss = model.training_step(batch, 0)
        print(f"✅ Success! Training step loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Check VelocityField Output Shape
    # (Testing the forward pass manually)
    z = torch.randn(BATCH_SIZE, CODE_DIM)
    r = torch.rand(BATCH_SIZE, 1)
    t = torch.rand(BATCH_SIZE, 1)
    vf_out = model.vf(z, r, t, mock_y)
        
    print(f"📊 VelocityField Output Shape: {vf_out.shape} (Expected: [{BATCH_SIZE}, {CODE_DIM}])")

'''
vt = self.vf(x_t, t, r, y)
vt_err = torch.pow(
    vr + vt,
    2,
).mean()
''' # time reversability condition.

# need to add symmetric loss to force u(z_r, r, t) == u(z_t, t, r)
#loss = vr_err #+ vt_err

'''
x0_predict = z - self.vf(
    z, 
    torch.ones_like(t), 
    torch.zeros_like(r), 
    y
)

x0_y, x0_null = x0_predict.chunk(2, dim=1)

dist_y = torch.sum(x0_y**2, dim=-1)
dist_null = torch.sum(x0_null**2, dim=-1)

log_ratio = 0.5 * (dist_null - dist_y)

L_MI = 0.5*(dist_null - dist_y)
mi_loss = (L_MI**2).mean()
weight = torch.clamp(
    torch.tensor(self.global_step, device=L_MI.device) / 5000,
    max=1.0,
)

wlog_ratio = weight*mi_loss
loss = loss + wlog_ratio
'''

'''
MMD = self.MMD(
    x0_predict, 
    x_0,
    bandwidth=10
)

weight = torch.clamp(
    torch.tensor(self.global_step, device=MMD.device) / 5000,
    max=1.0,
)

# remove 100 and put into a lambda prefactor.
wMMD = weight*MMD
loss = loss + wMMD


null_vector = self.vf.null_y(
    torch.zeros(y.size(0), dtype=torch.long, device=y.device)
)
y = torch.cat([
    y, null_vector
])

    def MMD(self, x, y, bandwidth):
        """Emprical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx # Used for A in (1)
        dyy = ry.t() + ry - 2. * yy # Used for B in (1)
        dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        
        XX, YY, XY = (
            torch.zeros_like(xx),
            torch.zeros_like(yy),
            torch.zeros_like(xx)
        )
                
        XX += torch.exp(-0.5*dxx/bandwidth)
        YY += torch.exp(-0.5*dyy/bandwidth)
        XY += torch.exp(-0.5*dxy/bandwidth)
    
        return torch.mean(XX + YY - 2. * XY)

'''
