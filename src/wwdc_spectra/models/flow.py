# Need to make a torch model wrapper for spender
import torch
import torch.nn as nn
import lightning as L

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
        y = model_extras["y"]
        cfg_scale = model_extras["cfg_scale"]

        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        v = self.velocity_model(x_t=x, t=t, y=y)
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
