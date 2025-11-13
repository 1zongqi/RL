"""
Environment adapter interface and base implementation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import numpy as np

from .types import ObsDict, ActionDict, RewardDict, InfoDict, PaperObsDict, PaperObsEntry


class EnvAdapter(ABC):
    """Abstract base class for environment adapters."""
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[ObsDict, InfoDict]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed for reproducibility.
            
        Returns:
            obs_dict: Dictionary mapping agent_id to observation dict.
                     Each observation must contain: pos_xy, obstacles_enc, goal_xy,
                     nearest_charger_xy, remain_time, remain_battery.
            info_dict: Dictionary with initial info. Must contain cost_time=0, cost_batt=0.
        """
        pass
    
    @abstractmethod
    def step(self, action_dict: ActionDict) -> Tuple[ObsDict, RewardDict, bool, InfoDict]:
        """
        Execute one step in the environment.
        
        Args:
            action_dict: Dictionary mapping agent_id to action.
            
        Returns:
            obs_dict: Dictionary mapping agent_id to observation dict.
            reward_dict: Dictionary mapping agent_id to reward.
            done: Global episode-level boolean (True if episode is done).
            info_dict: Dictionary with step info. Must contain:
                      - cost_time: instantaneous time cost for this step
                      - cost_batt: 1 if remain_battery <= 0 else 0
        """
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        pass
    
    def get_obs_metadata(self) -> Dict[str, Any]:
        """Return observation metadata such as grid size or normalization constants."""
        return {}


class BaseAdapter(EnvAdapter):
    """
    Non-abstract base class providing common functionality.
    Holds a backend environment and handles done aggregation, action validation, etc.
    """
    
    def __init__(self, backend_env: Any, num_agents: int, action_dim: int):
        """
        Initialize the base adapter.
        
        Args:
            backend_env: The underlying environment instance.
            num_agents: Number of agents in the environment.
            action_dim: Action space dimension (for validation).
        """
        self.backend_env = backend_env
        self.num_agents = num_agents
        self.action_dim = action_dim
        self._seed = None
        self.obs_encoder_version = "v1"
    
    def _validate_actions(self, action_dict: ActionDict) -> ActionDict:
        """
        Validate and clip actions to valid range.
        
        Args:
            action_dict: Dictionary mapping agent_id to action.
            
        Returns:
            Validated action dictionary.
        """
        validated = {}
        for agent_id, action in action_dict.items():
            if not (0 <= action < self.action_dim):
                # Clip to valid range
                action = max(0, min(action, self.action_dim - 1))
            validated[agent_id] = action
        return validated
    
    def _aggregate_done(self, done: Any) -> bool:
        """
        Aggregate per-agent done flags into global episode-level boolean.
        
        Args:
            done: Done flag(s) from backend environment.
                  Can be bool, list of bools, or dict of bools.
                  
        Returns:
            Global boolean indicating if episode is done.
        """
        if isinstance(done, bool):
            return done
        elif isinstance(done, (list, tuple)):
            return any(done)
        elif isinstance(done, dict):
            return any(done.values())
        else:
            # Fallback: assume it's a single boolean
            return bool(done)
    
    def _create_info_dict(self, cost_time: float = 0.0, cost_batt: float = 0.0, **kwargs) -> InfoDict:
        """
        Create info dictionary with required fields.
        
        Args:
            cost_time: Instantaneous time cost.
            cost_batt: Instantaneous battery cost (1 if remain_battery <= 0 else 0).
            **kwargs: Additional info fields.
            
        Returns:
            Info dictionary with required fields.
        """
        info = {
            'cost_time': float(cost_time),
            'cost_batt': float(cost_batt),
        }
        info.update(kwargs)
        return info
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        self._seed = seed
        if hasattr(self.backend_env, 'seed'):
            self.backend_env.seed(seed)
        if hasattr(self.backend_env, 'reset'):
            # Some environments accept seed in reset
            pass
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.backend_env, 'close'):
            self.backend_env.close()

    # ------------------------------------------------------------------
    # PaperObs helpers
    # ------------------------------------------------------------------

    def build_paper_obs(self, obs_dict: ObsDict, fov_size: int = 3) -> Tuple[PaperObsDict, bool]:
        """Convert legacy observation dict into PaperObs entries.

        Returns a mapping of agent_id -> PaperObsEntry and whether fallback
        values were used during construction.
        """

        paper_obs: PaperObsDict = {}
        fallback = False

        for agent_id, obs in obs_dict.items():
            pos_xy = obs.get('pos_xy', (0.0, 0.0))
            goal_xy = obs.get('goal_xy', (0.0, 0.0))
            charger_xy = obs.get('nearest_charger_xy', (0.0, 0.0))
            remain_time = float(obs.get('remain_time', 0.0))
            remain_battery = float(obs.get('remain_battery', 0.0))

            px, py = float(pos_xy[0]), float(pos_xy[1])
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            ex, ey = float(charger_xy[0]), float(charger_xy[1])

            obstacles_enc = obs.get('obstacles_enc')
            fov_onehot, used_fallback = self._extract_fov_onehot(obstacles_enc, fov_size)
            fallback = fallback or used_fallback

            paper_obs[agent_id] = PaperObsEntry(
                px=px,
                py=py,
                gx=gx,
                gy=gy,
                ex=ex,
                ey=ey,
                deadline=remain_time,
                battery=remain_battery,
                fov=fov_onehot,
            )

        return paper_obs, fallback

    def _extract_fov_onehot(self, obstacles_enc: Any, fov_size: int) -> Tuple[np.ndarray, bool]:
        """Extract a one-hot FOV window from obstacle encoding."""

        fallback = False
        if not isinstance(obstacles_enc, np.ndarray):
            fallback = True
            return np.zeros((3, fov_size, fov_size), dtype=np.float32), fallback

        arr = np.asarray(obstacles_enc, dtype=np.float32)
        if arr.ndim != 3:
            fallback = True
            return np.zeros((3, fov_size, fov_size), dtype=np.float32), fallback

        _, height, width = arr.shape
        half = fov_size // 2
        pad_y = max(0, half + 1)
        pad_x = max(0, half + 1)
        padded = np.pad(arr, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant')

        center_y = height // 2 + pad_y
        center_x = width // 2 + pad_x
        y_start = center_y - half
        y_end = y_start + fov_size
        x_start = center_x - half
        x_end = x_start + fov_size

        fov_raw = padded[:, y_start:y_end, x_start:x_end]

        if fov_raw.shape[1:] != (fov_size, fov_size):
            fallback = True
            fov_raw = np.zeros((arr.shape[0], fov_size, fov_size), dtype=np.float32)

        obstacle_layer = np.clip(fov_raw[0], 0.0, 1.0) if fov_raw.shape[0] > 0 else np.zeros((fov_size, fov_size), dtype=np.float32)
        occupied_layers = fov_raw[1:] if fov_raw.shape[0] > 1 else np.zeros((0, fov_size, fov_size), dtype=np.float32)
        occupied_layer = np.clip(np.sum(occupied_layers, axis=0), 0.0, 1.0)
        empty_layer = np.clip(1.0 - np.clip(obstacle_layer + occupied_layer, 0.0, 1.0), 0.0, 1.0)

        onehot = np.stack([empty_layer, obstacle_layer, occupied_layer], axis=0).astype(np.float32)
        return onehot, fallback
