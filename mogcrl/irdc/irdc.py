"""
Intrinsic Reward with Dynamic Context (IRDC).

Implements Encoder → RNN → (optional) GATv2 → head pipeline with running-normalised
intrinsic rewards and mask-aware sequence handling.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nets.modules import RNNBlock, init_hidden


class SimpleGATv2(nn.Module):
    """
    Lightweight stand-in for GATv2-like neighbour aggregation.

    Falls back to adjacency-weighted mean followed by an MLP projection. This keeps the module
    dependency-free while leaving room for a future swap to torch_geometric.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden_dim = heads * out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape [T, B, D] or [B, D].
            adj: Adjacency tensor with shape [T, B, B] or [B, B]. If None, identity is used.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_time = True
        else:
            squeeze_time = False

        T, B, _ = x.shape

        if adj is None:
            adj = torch.eye(B, device=x.device).unsqueeze(0).expand(T, B, B)
        elif adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(T, B, B)

        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        msg = torch.bmm(adj, x) / deg  # [T, B, D]
        out = self.net(msg)

        if squeeze_time:
            out = out.squeeze(0)
        return out


class RunningNorm(nn.Module):
    """Momentum-based running normalisation for intrinsic rewards."""

    def __init__(self, eps: float = 1e-5, momentum: float = 0.001) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("var", torch.tensor(1.0))
        self.eps = eps
        self.momentum = momentum

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        mean = x.mean()
        var = x.var(unbiased=False).clamp_min(self.eps)
        self.mean = (1 - self.momentum) * self.mean + self.momentum * mean
        self.var = (1 - self.momentum) * self.var + self.momentum * var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x.detach())
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class IRDC(nn.Module):
    """
    Intrinsic reward module.

    Pipeline: (obs, action) -> encoder MLP -> RNN -> optional GAT -> head -> r_int.
    Supports [B, D] and [T, B, D] layouts, mask-aware recurrence, and running normalisation.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        mlp_hidden = cfg.get("mlp_hidden", [128, 128])
        rnn_cfg = cfg.get("rnn", {}) or {}

        self.use_gat = bool(cfg.get("use_gat", False))
        self.normalize = bool(cfg.get("normalize", True))

        encoder_layers: list[nn.Module] = []
        last_dim = obs_dim + act_dim
        for hidden_dim in mlp_hidden:
            encoder_layers.append(nn.Linear(last_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()

        self.rnn = RNNBlock(
            input_size=last_dim,
            hidden_size=int(rnn_cfg.get("hidden_size", mlp_hidden[-1] if mlp_hidden else obs_dim)),
            num_layers=int(rnn_cfg.get("num_layers", 1)),
            dropout=float(rnn_cfg.get("dropout", 0.0)),
        )

        if self.use_gat:
            gat_cfg = cfg.get("gat", {}) or {}
            self.gat = SimpleGATv2(
                in_dim=self.rnn.hidden_size,
                out_dim=self.rnn.hidden_size,
                heads=int(gat_cfg.get("heads", 2)),
                dropout=float(gat_cfg.get("dropout", 0.0)),
            )
        else:
            self.gat = None

        head_layers: list[nn.Module] = [
            nn.Linear(self.rnn.hidden_size, self.rnn.hidden_size),
            nn.ReLU(),
            nn.Linear(self.rnn.hidden_size, 1),
        ]
        self.head = nn.Sequential(*head_layers)
        self.norm = RunningNorm() if self.normalize else nn.Identity()

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return init_hidden(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device)

    @staticmethod
    def _to_time_major(
        tensor: torch.Tensor, batch_first: Optional[bool]
    ) -> Tuple[torch.Tensor, bool]:
        if tensor.dim() == 2:
            return tensor.unsqueeze(0), True
        if tensor.dim() == 3:
            if batch_first is None:
                batch_first = False
            return (tensor.transpose(0, 1), True) if batch_first else (tensor, False)
        raise ValueError(f"Unsupported tensor shape {tuple(tensor.shape)}")

    @staticmethod
    def _broadcast_mask(mask: Optional[torch.Tensor], T: int, B: int, device: torch.device) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 1:
            mask = mask.view(1, B).expand(T, -1)
        elif mask.dim() == 2:
            if mask.shape[0] != T:
                raise ValueError("Mask time dimension mismatch for IRDC.")
        else:
            raise ValueError("Mask for IRDC must be 1-D or 2-D.")
        return mask.to(device)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        h_in: Optional[torch.Tensor] = None,
        batch_first: Optional[bool] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            obs: Observation features [B, Do] or [T, B, Do].
            act: Discrete action indices [B] or [T, B].
            mask: Episode mask (1 - continue, 0 - reset) [B] or [T, B].
            h_in: Initial hidden state [L, B, H].
            batch_first: Layout hint for 3-D inputs.
            adj: Optional adjacency for GAT aggregation.

        Returns:
            r_int: Intrinsic reward tensor (layout matches input).
            h_out: Final hidden state.
            extras: Dict containing intermediate features.
        """
        obs_tm, obs_was_bf = self._to_time_major(obs, batch_first)
        if act.dim() == 1:
            act_tm = act.unsqueeze(0)
            act_was_bf = True
        else:
            act_input = act.unsqueeze(-1) if act.dim() == 2 else act
            act_tm, act_was_bf = self._to_time_major(act_input, batch_first)
            if act_tm.dim() == 3:
                act_tm = act_tm.squeeze(-1)

        T, B, Do = obs_tm.shape
        if act_tm.shape[0] != T or act_tm.shape[1] != B:
            raise ValueError("Observation and action time/batch dimensions must match.")

        mask_tm = self._broadcast_mask(mask, T, B, obs_tm.device)

        act_oh = F.one_hot(act_tm.long(), num_classes=self.act_dim).float()
        x = torch.cat([obs_tm, act_oh], dim=-1)

        encoded = self.encoder(x)
        rnn_out, h_out = self.rnn(encoded, h_in=h_in, mask=mask_tm, batch_first=False)

        features = rnn_out
        if self.gat is not None:
            features = self.gat(features, adj)

        logits = self.head(features).squeeze(-1)
        if mask_tm is not None:
            logits = logits * mask_tm.float()

        r_int = self.norm(logits)

        if obs_was_bf:
            r_int = r_int.squeeze(0)
        if act_was_bf and not obs_was_bf:
            # when obs was time-major but act wasn't, align output: already handled.
            pass

        extras = {
            "encoded": encoded,
            "features": features,
        }
        return r_int, h_out, extras

