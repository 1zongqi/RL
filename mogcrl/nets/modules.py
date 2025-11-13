"""Common neural network modules (RNN blocks, helpers)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def init_hidden(num_layers: int, batch_size: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    """Initialise GRU hidden state with zeros."""
    return torch.zeros(num_layers, batch_size, hidden_size, device=device)


class RNNBlock(nn.Module):
    """GRU block with masking and flexible input layout support."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first_default = batch_first
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

    def _to_time_major(
        self,
        x: torch.Tensor,
        batch_first: Optional[bool],
    ) -> Tuple[torch.Tensor, bool, bool]:
        """Return time-major tensor, flag if original was batch_first, and if input was 2D."""
        if x.dim() == 2:
            return x.unsqueeze(0), True, True
        if x.dim() == 3:
            bf = self.batch_first_default if batch_first is None else batch_first
            return (x.transpose(0, 1), True, False) if bf else (x, False, False)
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

    def _format_mask(
        self,
        mask: Optional[torch.Tensor],
        steps: int,
        batch: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dim() == 1:
            mask = mask.view(1, -1).expand(steps, -1)
        elif mask.dim() == 2:
            if mask.shape[0] != steps:
                raise ValueError("Mask time dimension mismatch")
        else:
            raise ValueError("Mask must have dim 1 or 2")
        return mask.to(device)

    def forward(
        self,
        x: torch.Tensor,
        h_in: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        batch_first: Optional[bool] = None,
        return_batch_first: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward GRU with mask support.

        Args:
            x: [B, D] or [T, B, D] / [B, T, D] tensor.
            h_in: optional hidden state [L, B, H].
            mask: optional mask [B] or [T, B] to reset hidden state.
            batch_first: layout hint for 3-D input.
            return_batch_first: override output layout (True => batch-first).
        """

        x_tm, was_bf, orig_2d = self._to_time_major(x, batch_first)
        T, B, _ = x_tm.shape
        device = x_tm.device

        h = h_in if h_in is not None else init_hidden(self.num_layers, B, self.hidden_size, device)
        mask_tm = self._format_mask(mask, T, B, device)

        outputs = []
        for t in range(T):
            if mask_tm is not None:
                m = mask_tm[t].view(1, B, 1).to(h.dtype)
                h = h * m
            y_t, h = self.gru(x_tm[t].unsqueeze(0), h)
            outputs.append(y_t)

        y_tm = torch.cat(outputs, dim=0)

        if orig_2d:
            y_out = y_tm.squeeze(0)
        else:
            ret_bf = return_batch_first if return_batch_first is not None else was_bf
            y_out = y_tm.transpose(0, 1) if ret_bf else y_tm

        return y_out, h

