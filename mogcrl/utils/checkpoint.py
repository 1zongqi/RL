from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.amp import GradScaler


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    root_dir: Path | str,
    tag: str,
    step: int,
    policy_map: Dict[str, torch.nn.Module],
    critic_map: Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    scaler: Optional["GradScaler"] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    root = Path(root_dir)
    _ensure_dir(root)
    ckpt_path = root / f"{tag}_step{step:08d}.pt"
    payload = {
        "step": step,
        "policy": {k: v.state_dict() for k, v in policy_map.items()},
        "critic": {k: v.state_dict() for k, v in critic_map.items()},
        "optim": {k: opt.state_dict() for k, opt in optimizers.items()},
        "meta": meta or {},
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    ckpt_path: Path | str,
    policy_map: Dict[str, torch.nn.Module],
    critic_map: Dict[str, torch.nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    scaler: Optional["GradScaler"] = None,
) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu")
    for key, state in payload.get("policy", {}).items():
        if key in policy_map:
            policy_map[key].load_state_dict(state)
    for key, state in payload.get("critic", {}).items():
        if key in critic_map:
            critic_map[key].load_state_dict(state)
    if optimizers is not None:
        for key, state in payload.get("optim", {}).items():
            if key in optimizers:
                optimizers[key].load_state_dict(state)
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload


def cleanup_checkpoints(root_dir: Path | str, tag: str, keep: int) -> None:
    root = Path(root_dir)
    if keep <= 0 or not root.exists():
        return
    ckpts = sorted(root.glob(f"{tag}_step*.pt"))
    for ckpt in ckpts[:-keep]:
        ckpt.unlink(missing_ok=True)


def save_best_meta(root_dir: Path | str, meta: Dict[str, Any]) -> None:
    root = Path(root_dir)
    _ensure_dir(root)
    (root / "best.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

