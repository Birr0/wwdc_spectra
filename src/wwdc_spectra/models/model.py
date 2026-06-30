import math 

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
        """
        Forward pass with classifier-free guidance.

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

class WrappedModel(nn.Module):
    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(self, x, t, **model_extras):
        cfg_scale = model_extras.get("cfg_scale", 1.0)
        y = model_extras.get("y")
        batch_size = x.shape[0]

        # Ensure t is the right shape for the concatenated batch
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        
        # If no guidance, just run conditional
        if cfg_scale == 1.0:
            return self.velocity_model(x_t=x, t=t, y=y)

        # 1. Create Null y for inference
        null_idx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        y_null = self.velocity_model.null_y(null_idx)

        # 2. Batch doubling
        x_double = torch.cat([x, x], dim=0)
        t_double = torch.cat([t, t], dim=0)
        y_double = torch.cat([y, y_null], dim=0)

        # 3. Predict velocities
        v_double = self.velocity_model(x_t=x_double, t=t_double, y=y_double)
        
        # 4. Chunk and Guide
        v_cond, v_uncond = torch.chunk(v_double, chunks=2, dim=0)
        
        # Apply CFG formula
        return v_uncond + cfg_scale * (v_cond - v_uncond)
        
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

class VelocityField(nn.Module, PyTorchModelHubMixin):
    def __init__(self, code_dim, hidden_dim, conditional_dim, n_hidden=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = 1

        self.t_embedder = TimestepEmbedder(hidden_dim)
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

    def forward(self, x_t: Tensor, t: Tensor, y: Tensor):
        t_embed = self.t_embedder(t).flatten(start_dim=1)
        y_embed = self.cond_embed(y)

        x = self.input_proj(x_t)
        c = t_embed + y_embed

        for adaln, lin in zip(self.ada_lns, self.linears):
            identity = x
            modulated, gate = adaln(x, c) # replace with c here.
            x = self.act(lin(modulated))
            x = identity + gate*x
        return self.out_proj(x)


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

        x_1 = self.base_model.encode(X)
        batch_size = x_1.shape[0]

        # 1. Sample the path as usual
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=x_1.device)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # 2. Construct the Null y
        # Fetch the learned null vector and expand it to match the batch size
        null_idx = torch.zeros(batch_size, dtype=torch.long, device=y.device)
        y_null = self.vf.null_y(null_idx) # Result shape: [batch_size, conditional_dim]

        # 3. Double the tensors (Batch Doubling)
        # x_t and t are simply duplicated
        x_t_double = torch.cat([path_sample.x_t, path_sample.x_t], dim=0)
        t_double = torch.cat([path_sample.t, path_sample.t], dim=0)
        
        # y is paired with y_null
        y_double = torch.cat([y, y_null], dim=0)

        # 4. Forward pass through the Velocity Field
        # ut will have shape [2 * batch_size, code_dim]
        ut = self.vf(x_t=x_t_double, t=t_double, y=y_double)

        # 5. Calculate loss against the doubled target
        u_target_double = torch.cat([path_sample.dx_t, path_sample.dx_t], dim=0)
        
        loss = torch.pow(ut - u_target_double, 2).mean()

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

class LightningParamInference(L.LightningModule):
    def __init__(
        self,
        lr,
        batch_size,
        num_params,
        hidden_dim,
        conditional_dim,
        catalog,
        n_steps=10,
        ckpt_path: str = None,

    ):
        super().__init__()

        self.num_params = num_params
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr

        # --- Models --- #
        self.vf = VelocityField(num_params, hidden_dim, conditional_dim)
        self.vf.apply(self._init_weights)

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

    @property
    def T(self):
        return torch.linspace(1, 0, self.n_steps, device=self.device)

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
        y, params, _ = batch
        batch_size = y.shape[0]

        x_0 = torch.randn_like(params)
        t = torch.rand(params.shape[0], device=params.device)

        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=params)

        ut = self.vf(x_t=path_sample.x_t, t=t, y=y)
        loss = torch.pow(ut - path_sample.dx_t, 2).mean()

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

# --- Main Test Script ---
if __name__ == "__main__":
    '''# 1. Hyperparameters
    CODE_DIM = 10
    HIDDEN_DIM = 128
    BATCH_SIZE = 512
    INPUT_FEATURES = 128
    CATALOG = {
        "catalog_name": "hsc_embeddings",
        "fp": None,
        "join_method": "concat",
        "variables": {
            "m": {
                "name": "mass",
                "size": 1,
                "processing_fn": None
            },
            "sfr": {
                "name": "star formation rate",
                "size": 1,
                "processing_fn": None
            },
            "A_v": {
                "name": "Dust attenuation",
                "size": 1,
                "processing_fn": None
            }
        },
        "drop_variables": []
    } # Mock catalog

    # 2. Setup Data
    # Mock X: (Batch, Features), Mock y: (Batch, Cond_Dim)

    mock_X = torch.randn(BATCH_SIZE, CODE_DIM)
    mock_y = torch.randn(BATCH_SIZE, 1 )# get_conditional_len(CATALOG))
    #mock_X = torch.randn(BATCH_SIZE, 10)
    
    
    batch = (mock_X, mock_y)

    print(get_conditional_len(CATALOG))
    print(mock_X.shape, mock_y.shape)

    # 3. Initialize Model
    model =LightningFlowMatching( #LightningParamInference( #LightningFlowMatching( # LightningMeanFlowMatching(
        lr=1e-4,
        num_params=3,
        conditional_dim=10,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        catalog=CATALOG
    )
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

    print("🚀 Initializing test run...")

    # 4. Run a Training Step
    try:
        loss = model.training_step(batch, 0)
        print(f"✅ Success! Training step loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
    '''
    # --- Mocks for Missing Components ---
    class PyTorchModelHubMixin: pass

    class BaseModel(nn.Module):
        def __init__(self, code_dim):
            super().__init__()
            self.code_dim = code_dim
        def encode(self, x):
            # Mock encoder: projects input to code_dim
            return torch.randn(x.size(0), self.code_dim)

    # 1. Hyperparameters
    CODE_DIM = 10
    HIDDEN_DIM = 64
    BATCH_SIZE = 512
    INPUT_FEATURES = 128
    CATALOG = {
        "catalog_name": "hsc_embeddings",
        "fp": None,
        "join_method": "concat",
        "variables": {
            "z": {
                "name": "z",
                "size": 1,
                "processing_fn": None
            },
        },
        "drop_variables": []
    } # Mock catalog

    # 2. Setup Data
    # Mock X: (Batch, Features), Mock y: (Batch, Cond_Dim)
    mock_X = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    mock_y = torch.randn(BATCH_SIZE, get_conditional_len(CATALOG))
    batch = (mock_X, mock_y)

    # 3. Initialize Model
    base_model = BaseModel(code_dim=CODE_DIM)

    model = LightningFlowMatching( # LightningMeanFlowMatching(
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
    vf_out = model.vf(z, t, mock_y)
        
    print(f"📊 VelocityField Output Shape: {vf_out.shape} (Expected: [{BATCH_SIZE}, {CODE_DIM}])")

