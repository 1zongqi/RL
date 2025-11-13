"""
Lightweight encoders for IRDC intrinsic reward module.

Currently implements simple wrappers around existing per-agent feature tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class IRDCEncoderOutput:
    features: torch.Tensor
    obs_dim: int
    obs_mode: str
    metadata: Dict[str, torch.Tensor]


class IdentityIRDCEncoder:
    """
    Pass-through encoder that assumes upstream code already produced per-agent features.

    This keeps IRDC decoupled from policy encoders while still supporting future swaps.
    """

    def __init__(self, obs_dim: int, obs_mode: str = "flat") -> None:
        self.out_dim = obs_dim
        self.obs_mode = obs_mode

    def __call__(self, obs: torch.Tensor, info: Optional[Dict] = None) -> IRDCEncoderOutput:
        if obs.dim() not in (2, 3):
            raise ValueError(f"IRDC encoder expects rank-2/3 tensors, got {obs.shape}")
        return IRDCEncoderOutput(
            features=obs,
            obs_dim=self.out_dim,
            obs_mode=self.obs_mode,
            metadata=info or {},
        )

