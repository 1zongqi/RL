from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AgentRouting:
    policy_of: Dict[int, str]
    critic_of: Dict[int, str]


def _normalize_space_map(
    spaces: Optional[Iterable],
    num_agents: int,
) -> Optional[Dict[int, object]]:
    if spaces is None:
        return None
    if isinstance(spaces, dict):
        return {int(k): v for k, v in spaces.items()}
    if isinstance(spaces, Sequence):
        return {idx: spaces[idx] for idx in range(len(spaces))}
    raise TypeError("spaces must be dict, sequence, or None")


def _space_signature(space: object) -> Tuple:
    if hasattr(space, "shape"):
        return tuple(getattr(space, "shape"))
    if hasattr(space, "n"):
        return ("discrete", getattr(space, "n"))
    if isinstance(space, (tuple, list)):
        return tuple(space)
    return (space,)


def _validate_spaces(
    assignments: Dict[int, str],
    spaces: Optional[Dict[int, object]],
    label: str,
) -> None:
    if spaces is None:
        return
    per_key: Dict[str, set] = {}
    for agent_id, key in assignments.items():
        sig = _space_signature(spaces[agent_id])
        per_key.setdefault(key, set()).add(sig)
    for key, sigs in per_key.items():
        if len(sigs) > 1:
            raise ValueError(
                f"{label} space mismatch detected for {key}: {sorted(sigs)}"
            )


def build_registry(
    num_agents: int,
    cfg: Dict,
    make_policy: Callable[[], "torch.nn.Module"],
    make_critic: Callable[[], "torch.nn.Module"],
    obs_spaces: Optional[Iterable] = None,
    act_spaces: Optional[Iterable] = None,
) -> Tuple[Dict[str, "torch.nn.Module"], Dict[str, "torch.nn.Module"], AgentRouting]:
    ma_cfg = cfg.get("multiagent", {})
    policy_sharing = ma_cfg.get("policy_sharing", "shared")
    critic_sharing = ma_cfg.get("critic_sharing", "shared")
    groups_cfg = ma_cfg.get("groups", []) or []

    groups: Dict[str, set] = {}
    seen_agents = set()
    for group in groups_cfg:
        name = group["name"]
        agent_ids = set(group.get("agent_ids", []))
        if not agent_ids:
            raise ValueError(f"group '{name}' must list at least one agent id")
        overlap = seen_agents.intersection(agent_ids)
        if overlap:
            raise ValueError(
                f"agent ids {sorted(overlap)} appear in multiple groups"
            )
        for aid in agent_ids:
            if aid < 0 or aid >= num_agents:
                raise ValueError(f"agent id {aid} out of range 0..{num_agents-1}")
        groups[name] = agent_ids
        seen_agents.update(agent_ids)

    obs_map = _normalize_space_map(obs_spaces, num_agents)
    act_map = _normalize_space_map(act_spaces, num_agents)

    policy_map, policy_of = _allocate_models(
        sharing=policy_sharing,
        prefix="policy:",
        num_agents=num_agents,
        groups=groups,
        make_fn=make_policy,
    )
    critic_map, critic_of = _allocate_models(
        sharing=critic_sharing,
        prefix="critic:",
        num_agents=num_agents,
        groups=groups,
        make_fn=make_critic,
    )

    _validate_spaces(policy_of, obs_map, "observation")
    _validate_spaces(policy_of, act_map, "action")

    _assert_expected_counts(
        ma_cfg.get("num_policies"),
        len(policy_map),
        "num_policies",
    )
    _assert_expected_counts(
        ma_cfg.get("num_critics"),
        len(critic_map),
        "num_critics",
    )

    routing = AgentRouting(policy_of=policy_of, critic_of=critic_of)
    return policy_map, critic_map, routing


def _allocate_models(
    sharing: str,
    prefix: str,
    num_agents: int,
    groups: Dict[str, set],
    make_fn: Callable[[], "torch.nn.Module"],
) -> Tuple[Dict[str, "torch.nn.Module"], Dict[int, str]]:
    import torch.nn as nn  # local import to avoid hard dependency at module load

    models: Dict[str, nn.Module] = {}
    assignment: Dict[int, str] = {}

    if sharing == "shared":
        key = f"{prefix}shared"
        models[key] = make_fn()
        for aid in range(num_agents):
            assignment[aid] = key
    elif sharing == "grouped":
        if not groups:
            raise ValueError("grouped sharing requires non-empty multiagent.groups")
        assigned = set()
        for gname, agent_ids in groups.items():
            key = f"{prefix}{gname}"
            models[key] = make_fn()
            for aid in agent_ids:
                assignment[aid] = key
            assigned.update(agent_ids)
        missing = set(range(num_agents)) - assigned
        if missing:
            raise ValueError(
                f"grouped sharing missing agent ids: {sorted(missing)}"
            )
    elif sharing == "independent":
        for aid in range(num_agents):
            key = f"{prefix}{aid}"
            models[key] = make_fn()
            assignment[aid] = key
    else:
        raise ValueError(f"unsupported sharing mode '{sharing}'")

    return models, assignment


def _assert_expected_counts(expected: Optional[int], actual: int, label: str) -> None:
    if expected is None:
        return
    if expected != actual:
        raise ValueError(f"{label} expected {expected} but built {actual}")

