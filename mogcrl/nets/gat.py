"""
Lightweight GATv2 implementation with sequence support.

Features:
- Works with [B, D] or [T, B, D] layouts (time-major internal).
- Accepts adjacency tensors shaped [B, B] or [T, B, B] (boolean/uint8).
- Provides fp16-safe masking and optional neighbour top-k pruning.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Block(nn.Module):
    """Multi-head GATv2 block supporting sequence inputs."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 2,
        dropout: float = 0.0,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.top_k = top_k

        hidden = out_dim * heads
        self.q = nn.Linear(in_dim, hidden, bias=False)
        self.k = nn.Linear(in_dim, hidden, bias=False)
        self.v = nn.Linear(in_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _to_time_major(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x.dim() == 2:
            return x.unsqueeze(0), True
        if x.dim() == 3:
            return x, False
        raise ValueError(f"GATv2Block expects rank-2/3 inputs, got {x.shape}")

    @staticmethod
    def _ensure_adj(adj: Optional[torch.Tensor], T: int, B: int, device: torch.device) -> torch.Tensor:
        if adj is None:
            return torch.eye(B, device=device, dtype=torch.bool).unsqueeze(0).expand(T, B, B)
        if adj.dim() == 2:
            return adj.to(device=device, dtype=torch.bool).unsqueeze(0).expand(T, B, B)
        if adj.dim() == 3:
            if adj.shape[0] != T or adj.shape[1] != B or adj.shape[2] != B:
                raise ValueError("Adjacency shape mismatch.")
            return adj.to(device=device, dtype=torch.bool)
        raise ValueError(f"Unsupported adjacency shape {adj.shape}")

    def _apply_top_k(self, adj: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.top_k is None:
            return adj
        k = min(self.top_k, adj.size(-1))
        if k <= 0:
            return adj
        # logits: [T, H, B, B]; select top-k neighbours per (T,H,B,*)
        values, indices = torch.topk(logits, k=k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        # combine with structural adjacency
        struct = adj.unsqueeze(1).expand_as(mask)
        return (mask & struct)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        x_tm, was_bf = self._to_time_major(x)  # [T, B, D]
        T, B, _ = x_tm.shape
        device = x_tm.device

        adj_bool = self._ensure_adj(adj, T, B, device).to(device=device, dtype=torch.bool)
        adj_bool = adj_bool | torch.eye(B, device=device, dtype=torch.bool).unsqueeze(0)  # ensure self-loop

        q = self.q(x_tm).view(T, B, self.heads, -1)
        k = self.k(x_tm).view(T, B, self.heads, -1)
        v = self.v(x_tm).view(T, B, self.heads, -1)

        qh = q.permute(0, 2, 1, 3)  # [T, H, B, dh]
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)

        dh = qh.size(-1)
        logits = torch.einsum("thbd,thjd->thbj", qh, kh) / (dh ** 0.5)

        if self.top_k is not None:
            adj_head = self._apply_top_k(adj_bool, logits)  # [T,H,B,B] bool
        else:
            adj_head = adj_bool.unsqueeze(1).expand_as(logits)
        adj_head = adj_head.to(device=x_tm.device)

        neg_value = -1e9 if logits.dtype == torch.float16 else float("-inf")
        logits = logits.masked_fill(~adj_head, neg_value)
        att = torch.softmax(logits, dim=-1)
        att = self.drop(att)

        out = torch.einsum("thbj,thjd->thbd", att, vh)
        out = out.permute(0, 2, 1, 3).contiguous().view(T, B, -1)
        out = self.proj(out)

        if was_bf:
            out = out.squeeze(0)
        return out

