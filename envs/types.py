"""Type definitions for environment adapters."""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union

import numpy as np

# Type aliases for environment dictionaries
ObsDict = Dict[int, Dict[str, Any]]  # agent_id -> observation dict
ActionDict = Dict[int, int]  # agent_id -> action
RewardDict = Dict[int, float]  # agent_id -> reward
InfoDict = Dict[str, Any]  # info dictionary with cost_time, cost_batt, etc.


@dataclass
class PaperObsEntry:
    """Structured observation for PaperObs encoding."""

    px: float
    py: float
    gx: float
    gy: float
    ex: float
    ey: float
    deadline: float
    battery: float
    fov: np.ndarray  # Shape [C, H, W] (one-hot channels)


PaperObsDict = Dict[int, PaperObsEntry]

# Observation fields for each agent (legacy flat obs)
ObsFields = {
    'pos_xy': Tuple[float, float],
    'obstacles_enc': np.ndarray,
    'goal_xy': Tuple[float, float],
    'nearest_charger_xy': Tuple[float, float],
    'remain_time': float,
    'remain_battery': float,
}

# Info fields that must be present
InfoFields = {
    'cost_time': float,
    'cost_batt': float,
}
