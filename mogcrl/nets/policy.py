"""Categorical 策略网络（支持可选 RNN）。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .modules import RNNBlock, init_hidden
from .gat import GATv2Block


class IdentityEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class CategoricalPolicy(nn.Module):
    """Encoder → (可选) GRU → MLP → Categorical logits."""

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        act_dim: int = 0,
        hidden: Optional[list] = None,
        use_rnn: bool = False,
        rnn_cfg: Optional[Dict[str, Any]] = None,
        use_gat: bool = False,
        gat_cfg: Optional[Dict[str, Any]] = None,
        gat_fuse: str = "concat",
        obs_dim: Optional[int] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        hidden = hidden or [128, 128]

        if encoder is None:
            if obs_dim is None:
                raise ValueError("必须提供 encoder 或 obs_dim 之一")
            encoder = IdentityEncoder(obs_dim)
        self.encoder = encoder

        if act_dim <= 0:
            raise ValueError("act_dim 必须为正")
        self.action_dim = act_dim
        self.use_rnn = use_rnn
        self.rnn_cfg = rnn_cfg or {}
        self.use_gat = use_gat
        self.gat_cfg = gat_cfg or {}
        self.gat_fuse = gat_fuse

        encoder_out_dim = getattr(self.encoder, "out_dim", None)
        if encoder_out_dim is None:
            raise ValueError("Encoder 必须提供 out_dim 属性")

        if use_rnn:
            hidden_size = self.rnn_cfg.get('hidden_size', hidden[0] if hidden else encoder_out_dim)
            num_layers = self.rnn_cfg.get('num_layers', 1)
            dropout = self.rnn_cfg.get('dropout', 0.0)
            self.rnn = RNNBlock(
                input_size=encoder_out_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.rnn_hidden_size = hidden_size
            mlp_input = hidden_size
        else:
            self.rnn = None
            self.rnn_hidden_size = encoder_out_dim
            mlp_input = encoder_out_dim

        self.gat: Optional[GATv2Block]
        self.gat_hidden_size: Optional[int]
        if self.use_gat:
            gat_hidden = int(self.gat_cfg.get('hidden', mlp_input))
            gat_heads = int(self.gat_cfg.get('heads', 2))
            gat_dropout = float(self.gat_cfg.get('dropout', 0.0))
            gat_top_k = self.gat_cfg.get('top_k')
            self.gat = GATv2Block(
                in_dim=mlp_input,
                out_dim=gat_hidden,
                heads=gat_heads,
                dropout=gat_dropout,
                top_k=gat_top_k,
            )
            self.gat_hidden_size = gat_hidden
            fuse_mode = gat_fuse.lower()
            if fuse_mode not in {"concat", "residual", "replace", "none"}:
                raise ValueError(f"Unsupported GAT fuse mode: {gat_fuse}")
            self.gat_fuse_mode = "replace" if fuse_mode == "none" else fuse_mode
            if self.gat_fuse_mode == "concat":
                fused_dim = mlp_input + gat_hidden
            elif self.gat_fuse_mode == "residual":
                if gat_hidden != mlp_input:
                    raise ValueError("Residual GAT 需要 hidden 与输入维度一致")
                fused_dim = mlp_input
            else:  # replace
                fused_dim = gat_hidden
            mlp_input = fused_dim
        else:
            self.gat = None
            self.gat_hidden_size = None
            self.gat_fuse_mode = "replace"

        layers: list[nn.Module] = []
        last_dim = mlp_input
        for dim in hidden:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.ReLU())
            last_dim = dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(last_dim, act_dim)

    # ------------------------------------------------------------------ utils
    def initial_state(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_rnn or self.rnn is None:
            return None
        return init_hidden(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device)

    @property
    def act_dim(self) -> int:
        return self.action_dim

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        obs: torch.Tensor,
        *args,
        h_in: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        batch_first: Optional[bool] = None,
        agent_id: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """支持 [B,D] 与 [T,B,D]/[B,T,D] 输入。"""

        if len(args) == 1 and agent_id is None:
            agent_id = args[0]
        _ = agent_id  # 当前实现未使用 agent_id，但保留兼容

        enc = self.encoder(obs)

        if self.use_rnn and self.rnn is not None:
            rnn_out, h_out = self.rnn(enc, h_in=h_in, mask=mask, batch_first=batch_first)
        else:
            rnn_out = enc
            h_out = None

        extras: Dict[str, torch.Tensor] = {}
        if self.use_gat and self.gat is not None:
            gat_out = self.gat(rnn_out, adj)
            extras["gat"] = gat_out
            if self.gat_fuse_mode == "concat":
                rnn_out = torch.cat([rnn_out, gat_out], dim=-1)
            elif self.gat_fuse_mode == "residual":
                rnn_out = rnn_out + gat_out
            else:  # replace
                rnn_out = gat_out

        if rnn_out.dim() == 3:
            T, B, D = rnn_out.shape
            flat = rnn_out.reshape(T * B, D)
            logits = self.head(self.mlp(flat)).view(T, B, self.action_dim)
        else:
            logits = self.head(self.mlp(rnn_out))

        dist = Categorical(logits=logits)
        extras["logits"] = logits
        return dist, h_out, extras

