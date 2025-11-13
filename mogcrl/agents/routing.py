from __future__ import annotations

from typing import Dict, List

import torch


class PolicyRouter:
    """Route agents to policy keys based on agent_id mapping."""

    def __init__(self, policy_of: Dict[int, str]) -> None:
        self.policy_of = dict(policy_of)

    def split_indices(self, agent_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        agent_ids: tensor(B,) of agent ids

        Returns
        -------
        Dict mapping policy_key -> tensor of agent indices in the batch.
        """
        if agent_ids.dim() != 1:
            raise ValueError(f"agent_ids 必须是一维张量，当前形状 {agent_ids.shape}")

        buckets: Dict[str, List[int]] = {}
        for idx, aid in enumerate(agent_ids.tolist()):
            key = self.policy_of.get(int(aid), "policy:shared")
            buckets.setdefault(key, []).append(idx)

        return {
            key: torch.tensor(indices, dtype=torch.long, device=agent_ids.device)
            for key, indices in buckets.items()
            if indices
        }


class BatchBuilder:
    """Utility for slicing per-step tensors by policy indices."""

    @staticmethod
    def slice_step(tensors: Dict[str, torch.Tensor | None], indices: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        """
        Slice a dictionary of tensors with shape [B,...] into [Bk,...] using the provided indices.
        Special handling for adjacency matrices ([B,B]).
        """
        sliced: Dict[str, torch.Tensor | None] = {}
        for key, tensor in tensors.items():
            if tensor is None:
                sliced[key] = None
                continue
            if tensor.dim() == 0:
                sliced[key] = tensor
                continue
            if key in {"adj", "adjacency"} and tensor.dim() >= 2:
                sliced[key] = tensor.index_select(0, indices).index_select(1, indices)
            else:
                sliced[key] = tensor.index_select(0, indices)
        return sliced

