"""Multi-head critic with optional GRU."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .modules import RNNBlock, init_hidden
from .gat import GATv2Block


class MultiHeadCritic(nn.Module):
    """Critic that outputs three value heads (R/time/batt) with optional GRU."""

    def __init__(
        self,
        obs_dim: int,
        hidden: Optional[list] = None,
        agent_embed_dim: int = 0,
        use_rnn: bool = False,
        rnn_cfg: Optional[Dict[str, Any]] = None,
        use_gat: bool = False,
        gat_cfg: Optional[Dict[str, Any]] = None,
        gat_fuse: str = "concat",
    ) -> None:
        super().__init__()
        hidden = hidden or []
        self.obs_dim = obs_dim
        self.agent_embed_dim = agent_embed_dim
        self.use_rnn = use_rnn
        self.rnn_cfg = rnn_cfg or {}
        self.use_gat = use_gat
        self.gat_cfg = gat_cfg or {}
        self.gat_fuse = gat_fuse

        if agent_embed_dim > 0:
            self.agent_embedding = nn.Embedding(1024, agent_embed_dim)
        else:
            self.agent_embedding = None

        base_input_dim = obs_dim
        rnn_input_dim = base_input_dim + (agent_embed_dim if agent_embed_dim > 0 else 0)

        if use_rnn:
            hidden_size = self.rnn_cfg.get('hidden_size', hidden[0] if hidden else rnn_input_dim)
            num_layers = self.rnn_cfg.get('num_layers', 1)
            dropout = self.rnn_cfg.get('dropout', 0.0)
            self.rnn = RNNBlock(
                input_size=rnn_input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
            self.rnn_hidden_size = hidden_size
            shared_input_dim = hidden_size
        else:
            self.rnn = None
            self.rnn_hidden_size = rnn_input_dim
            shared_input_dim = rnn_input_dim

        self.gat: Optional[GATv2Block]
        self.gat_hidden_size: Optional[int]
        if self.use_gat:
            gat_hidden = int(self.gat_cfg.get('hidden', shared_input_dim))
            gat_heads = int(self.gat_cfg.get('heads', 2))
            gat_dropout = float(self.gat_cfg.get('dropout', 0.0))
            gat_top_k = self.gat_cfg.get('top_k')
            self.gat = GATv2Block(
                in_dim=shared_input_dim,
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
                fused_dim = shared_input_dim + gat_hidden
            elif self.gat_fuse_mode == "residual":
                if gat_hidden != shared_input_dim:
                    raise ValueError("Residual GAT 需要 hidden 与输入维度一致")
                fused_dim = shared_input_dim
            else:
                fused_dim = gat_hidden
            shared_input_dim = fused_dim
        else:
            self.gat = None
            self.gat_hidden_size = None
            self.gat_fuse_mode = "replace"

        layers = []
        last_dim = shared_input_dim
        for h_dim in hidden:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim
        self.shared_net = nn.Sequential(*layers) if layers else nn.Identity()

        self.head_R = nn.Linear(last_dim, 1)
        self.head_time = nn.Linear(last_dim, 1)
        self.head_batt = nn.Linear(last_dim, 1)

    # ------------------------------------------------------------------ utils
    def initial_state(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.use_rnn:
            return None
        return init_hidden(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device)

    def _canonicalise(
        self,
        x: torch.Tensor,
        batch_first: Optional[bool],
        default_batch_first: bool = False,
    ) -> Tuple[torch.Tensor, bool, bool]:
        if x.dim() == 2:
            return x.unsqueeze(0), False, True
        if x.dim() == 3:
            bf = default_batch_first if batch_first is None else batch_first
            return (x.transpose(0, 1), True, False) if bf else (x, False, False)
        raise ValueError(f"Unsupported obs shape {x.shape}")

    def _restore_layout(
        self,
        tensor: torch.Tensor,
        was_batch_first: bool,
        was_single_step: bool,
    ) -> torch.Tensor:
        if was_single_step:
            return tensor.squeeze(0)
        if was_batch_first:
            tensor = tensor.transpose(0, 1)
        return tensor

    def _append_agent_emb(self, x: torch.Tensor, agent_id: Optional[torch.Tensor]) -> torch.Tensor:
        if self.agent_embedding is None or agent_id is None:
            return x
        emb = self.agent_embedding(agent_id.long())
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        if x.dim() == 2:
            return torch.cat([x, emb], dim=-1)
        if x.dim() == 3:
            if emb.dim() == 2:
                emb = emb.unsqueeze(0).expand(x.shape[0], -1, -1)
            return torch.cat([x, emb], dim=-1)
        raise ValueError(f"Unsupported rank {x.dim()} for agent embedding")

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        obs: torch.Tensor,
        agent_id: Optional[torch.Tensor] = None,
        h_in: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        batch_first: Optional[bool] = None,
        adj: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """支持 [B,D] 以及 [T,B,D]/[B,T,D] 两种输入布局。"""

        default_bf = bool(batch_first) if batch_first is not None else False
        obs_seq, was_batch_first, was_single_step = self._canonicalise(obs, batch_first, default_bf)
        seq_with_emb = self._append_agent_emb(obs_seq, agent_id)

        if self.use_rnn:
            features_seq, h_out = self.rnn(seq_with_emb, h_in=h_in, mask=mask, batch_first=False)
        else:
            h_out = h_in
            features_seq = seq_with_emb

        if self.use_gat and self.gat is not None:
            gat_seq = self.gat(features_seq, adj)
            if self.gat_fuse_mode == "concat":
                features_seq = torch.cat([features_seq, gat_seq], dim=-1)
            elif self.gat_fuse_mode == "residual":
                features_seq = features_seq + gat_seq
            else:
                features_seq = gat_seq

        T, B = features_seq.shape[0], features_seq.shape[1]
        flat = features_seq.reshape(T * B, -1)
        shared = self.shared_net(flat)

        v_R = self.head_R(shared)
        v_time = self.head_time(shared)
        v_batt = self.head_batt(shared)

        v_R_seq = v_R.view(T, B, -1)
        v_time_seq = v_time.view(T, B, -1)
        v_batt_seq = v_batt.view(T, B, -1)

        V_R = self._restore_layout(v_R_seq, was_batch_first, was_single_step)
        V_time = self._restore_layout(v_time_seq, was_batch_first, was_single_step)
        V_batt = self._restore_layout(v_batt_seq, was_batch_first, was_single_step)

        return {
            'R': V_R,
            'time': V_time,
            'batt': V_batt,
        }, h_out

