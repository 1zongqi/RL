"""Observation encoders.

Provide pluggable encoders for flat 18-D observations and PaperObs.
"""

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    features: torch.Tensor
    obs_mode: str
    obs_dim: int


class BaseEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.out_dim = output_dim

    def forward(self, obs: Dict[str, torch.Tensor], info: Dict[str, Any]) -> EncoderOutput:
        raise NotImplementedError


class FlatEncoder18D(BaseEncoder):
    def __init__(self):
        super().__init__(output_dim=18)

    def forward(self, obs: Dict[str, torch.Tensor], info: Dict[str, Any]) -> EncoderOutput:
        tensor = obs['flat']  # shape [B, 18]
        return EncoderOutput(features=tensor.float(), obs_mode='flat18', obs_dim=self.out_dim)


class PaperObsEncoder(BaseEncoder):
    def __init__(self, fov_size: int = 3):
        self.fov_size = fov_size
        spatial_dim = 6
        scalar_dim = 2
        fov_channels = 3
        fov_dim = fov_size * fov_size * fov_channels
        out_dim = spatial_dim + scalar_dim + fov_dim
        super().__init__(output_dim=out_dim)

    def forward(self, obs: Dict[str, torch.Tensor], info: Dict[str, Any]) -> EncoderOutput:
        norm_info = info.get('norm_consts', {})
        battery_full = norm_info.get('battery_full', 1.0)
        deadline_max = norm_info.get('deadline_max', 1.0)
        grid_width = norm_info.get('grid_width', 1.0)
        grid_height = norm_info.get('grid_height', 1.0)

        px = obs['px'].float() / max(grid_width, 1e-6)
        py = obs['py'].float() / max(grid_height, 1e-6)
        gx = obs['gx'].float() / max(grid_width, 1e-6)
        gy = obs['gy'].float() / max(grid_height, 1e-6)
        ex = obs['ex'].float() / max(grid_width, 1e-6)
        ey = obs['ey'].float() / max(grid_height, 1e-6)
        deadline = obs['deadline'].float() / max(deadline_max, 1e-6)
        battery = obs['battery'].float() / max(battery_full, 1e-6)

        spatial = torch.stack([px, py, gx, gy, ex, ey], dim=-1)
        scalar = torch.stack([deadline, battery], dim=-1)

        fov = obs['fov']
        if fov.dim() == 4:
            B = fov.shape[0]
            fov_flat = fov.view(B, -1)
        elif fov.dim() == 3:
            fov_flat = fov.view(1, -1)
        else:
            if fov.dim() == 1:
                fov_flat = fov.unsqueeze(0)
            else:
                fov_flat = fov

        features = torch.cat([spatial, scalar, fov_flat.float()], dim=-1)
        return EncoderOutput(features=features, obs_mode='paper_obs', obs_dim=self.out_dim)


