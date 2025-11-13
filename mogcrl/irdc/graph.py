"""
Graph utilities for IRDC and GAT front-ends.

Provides helper routines to construct adjacency matrices using PaperObs FOV
or radius-based fallbacks, as well as symmetry/self-loop handling.

When `topk_neighbors` is enabled we inject a tiny, deterministic jitter that
scales with the observed distance range (≈1e-8 × max distance). A boolean
`graphs.deterministic_topk_jitter` flag controls this behaviour; disabling it
removes the perturbation entirely for users who need strict geometric ordering.
"""

from __future__ import annotations

from typing import Optional

import torch


def _ensure_device(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.to(device=device)


def _radius_adjacency(positions: torch.Tensor, radius: float) -> torch.Tensor:
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    return dist_sq <= radius ** 2


def _fov_adjacency(
    positions: torch.Tensor,
    fov: torch.Tensor,
    inactive: Optional[torch.Tensor],
) -> torch.Tensor:
    B = positions.shape[0]
    if fov.ndim != 4:
        raise ValueError(f"FOV tensor must be [B, C, H, W], got {fov.shape}")
    C, H, W = fov.shape[1], fov.shape[2], fov.shape[3]
    if H != W:
        raise ValueError("FOV grid must be square.")
    half = H // 2
    occupied_idx = min(2, C - 1)
    occupied = fov[:, occupied_idx]
    adj = torch.zeros((B, B), dtype=torch.bool, device=positions.device)
    for i in range(B):
        if inactive is not None and not bool(inactive[i]):
            continue
        for j in range(B):
            if i == j:
                continue
            if inactive is not None and not bool(inactive[j]):
                continue
            dx = float((positions[j, 0] - positions[i, 0]).item())
            dy = float((positions[j, 1] - positions[i, 1]).item())
            rx = int(round(dx))
            ry = int(round(dy))
            if abs(rx) > half or abs(ry) > half:
                continue
            row = half - ry
            col = half + rx
            if 0 <= row < H and 0 <= col < W:
                if occupied[i, row, col] > 0.5:
                    adj[i, j] = True
    return adj


def build_adjacency(
    positions: Optional[torch.Tensor],
    fov: Optional[torch.Tensor] = None,
    *,
    mode: str = "fov",
    radius: float = 1.0,
    symmetry: str = "union",
    self_loop: bool = True,
    inactive: Optional[torch.Tensor] = None,
    topk_neighbors: Optional[int] = None,
    deterministic_topk_jitter: bool = True,
) -> torch.Tensor:
    """
    Construct adjacency matrix based on FOV or radius.

    Args:
        positions: [B, 2] tensor of agent positions.
        fov: [B, C, H, W] one-hot FOV tensor (channels: empty/obstacle/occupied).
        mode: 'fov', 'radius', or 'identity'.
        radius: Used for radius fallback.
        symmetry: 'union', 'intersection', or 'directed'.
        self_loop: Whether to enforce diagonal 1s.
        inactive: Optional bool mask (False for inactive agents).
    """

    if positions is None or positions.numel() == 0:
        return torch.zeros((0, 0), dtype=torch.bool)

    device = positions.device
    positions = positions.to(device=device, dtype=torch.float32)
    inactive = _ensure_device(inactive, device)
    B = positions.shape[0]

    if mode == "identity":
        adj = torch.eye(B, dtype=torch.bool, device=device)
    elif mode == "fov" and fov is not None:
        fov = fov.to(device=device, dtype=torch.float32)
        adj = _fov_adjacency(positions, fov, inactive)
        if not adj.any():
            adj = _radius_adjacency(positions, radius)
    elif mode in {"fov", "radius"}:
        adj = _radius_adjacency(positions, radius)
    else:
        raise ValueError(f"Unsupported adjacency mode: {mode}")

    if inactive is not None:
        active_mask = inactive.bool()
        adj = adj & active_mask.view(1, B)
        adj = adj & active_mask.view(B, 1)

    eye = torch.eye(B, dtype=torch.bool, device=device)
    if self_loop:
        adj = adj | eye
        diag_mask = eye
    else:
        adj = adj & ~eye
        diag_mask = torch.zeros_like(eye)

    adj_wo_diag = adj & ~eye

    if topk_neighbors is not None and B > 1:
        k = int(topk_neighbors)
        if k > 0 and k < (B - 1):
            dist = torch.cdist(positions, positions, p=2)
            idx = torch.arange(B, device=device)
            dist[idx, idx] = float("inf")
            dist_masked = dist.masked_fill(~adj_wo_diag, float("inf"))
            if deterministic_topk_jitter:
                finite_mask = torch.isfinite(dist_masked)
                if finite_mask.any():
                    scale = dist_masked[finite_mask].max().clamp_min(1e-12)
                else:
                    scale = dist_masked.new_tensor(1.0)
                rel_eps = scale * 1e-8
                jitter = torch.arange(
                    B,
                    device=device,
                    dtype=dist_masked.dtype,
                ).unsqueeze(0).mul_(rel_eps)
                dist_masked = dist_masked + jitter
            topk_idx = torch.topk(
                dist_masked,
                k=k,
                largest=False,
                dim=1,
            ).indices
            topk_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
            topk_mask.scatter_(1, topk_idx, True)
            topk_mask = topk_mask & ~eye
            adj_wo_diag = adj_wo_diag & topk_mask
    adj = adj_wo_diag | diag_mask

    if symmetry == "union":
        adj = adj | adj.transpose(0, 1)
    elif symmetry == "intersection":
        adj = adj & adj.transpose(0, 1)
    elif symmetry != "directed":
        raise ValueError(f"Unsupported symmetry mode: {symmetry}")

    return adj

