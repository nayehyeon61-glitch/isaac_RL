# models/wm_integration.py
from typing import Optional
import os
import torch
import torch.nn as nn

try:
    from ..model.model import WorldModel
except Exception:
    from model.model import WorldModel

def _safe(t: torch.Tensor, clip: float = 100.0) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
    return torch.clamp(t, -clip, clip)

class WMFeatureAdapter(nn.Module):
    """
    obs -> adapter(MLP)-> pseudo-DOF -> WorldModel.encoder(z) -> ODE(z_next)
    return [obs, z, z_next]
    """
    def __init__(self, obs_dim: int, dof_dim: int = 35, z_dim: int = 64,
                 wm_ckpt_path: str = "world_model_diffode.pt",
                 device: torch.device = torch.device("cpu"),
                 allow_grad_through_adapter: bool = True,
                 detach_z: bool = False):
        super().__init__()
        self.z_dim = int(z_dim)
        self.dof_dim = int(dof_dim)

        self.adapter = nn.Sequential(
            nn.LazyLinear(256), nn.SiLU(),
            nn.Linear(256, dof_dim)
        )

        if not os.path.exists(wm_ckpt_path):
            raise FileNotFoundError(
                f"[WMFeatureAdapter] checkpoint not found: {wm_ckpt_path}\n"
                "Run train_world_lafan.py first or fix configs.wm.ckpt_path."
            )
        self.wm = WorldModel(dof_dim=dof_dim, z_dim=z_dim)
        sd = torch.load(wm_ckpt_path, map_location=device)
        self.wm.load_state_dict(sd, strict=False)
        self.wm.to(device)
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

        for p in self.adapter.parameters():
            p.requires_grad = allow_grad_through_adapter

        self.detach_z = bool(detach_z)
        self._adapter_input_dim: Optional[int] = None

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        dev = obs.device
        self.adapter.to(dev)
        self.wm.to(dev)
        obs = _safe(obs).float()

        if self._adapter_input_dim is None:
            self._adapter_input_dim = obs.size(-1)
        elif obs.size(-1) != self._adapter_input_dim:
            old_first = self.adapter[0]
            if isinstance(old_first, nn.LazyLinear):
                self._adapter_input_dim = obs.size(-1)
            else:
                new_first = nn.Linear(obs.size(-1), 256).to(dev)
                with torch.no_grad():
                    if isinstance(old_first, nn.Linear) and old_first.in_features == new_first.in_features:
                        new_first.weight.copy_(old_first.weight)
                        if old_first.bias is not None and new_first.bias is not None:
                            new_first.bias.copy_(old_first.bias)
                self.adapter[0] = new_first
                self._adapter_input_dim = obs.size(-1)

        dof = _safe(self.adapter(obs))
        z = self.wm.encoder(dof)
        if self.detach_z:
            z = z.detach()
        z_next = self.wm.ode(z)
        feats = torch.cat([obs, z, z_next], dim=-1)
        return feats
