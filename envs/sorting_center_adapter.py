"""
Adapter for Sorting Center environment.
"""
import numpy as np
from typing import Dict, Tuple, Optional, List, Any

from .adapter import BaseAdapter
from .types import ObsDict, ActionDict, RewardDict, InfoDict
from . import utils


class SortingCenterAdapter(BaseAdapter):
    """
    Adapter for Sorting Center environment.
    Converts Gymnasium-style interface to EnvAdapter interface.
    """
    
    def __init__(self, backend_env: Any, cfg: Dict):
        """
        Initialize the adapter.
        
        Args:
            backend_env: The Sorting Center environment instance.
            cfg: Configuration dictionary from adapters.sorting section.
        """
        agent_ids = cfg.get('agent_ids', [0])
        num_agents = len(agent_ids)
        action_dim = cfg.get('action_dim', 5)
        
        super().__init__(backend_env, num_agents, action_dim)
        
        self.cfg = cfg
        self.agent_ids = agent_ids
        self.grid_wh = cfg.get('grid_wh', [1.0, 1.0])
        
        # Get charger coordinates from config or extract from environment
        chargers_from_cfg = cfg.get('chargers', [])
        if len(chargers_from_cfg) > 0:
            self.chargers = np.array(chargers_from_cfg, dtype=float)
        else:
            # Try to extract from environment
            self.chargers = np.array([], dtype=float)
        
        self.time_unit = cfg.get('time_unit', 'step')
        self.step_time_sec = cfg.get('step_time_sec', 1.0)
        self.battery_unit = cfg.get('battery_unit', 'step')
        self.max_batt_steps = cfg.get('max_batt_steps', 200.0)
        self.deadline_max = cfg.get('deadline_max', 70.0)
        
        # Store agent battery states (extracted from env state)
        self._agent_batteries = {}
    
    def _to_obs_dict(self, raw_obs: Dict) -> ObsDict:
        """
        Convert raw observation to ObsDict format.
        
        Args:
            raw_obs: Raw observation from environment {'obs': (M,5), 'matrix': (M,3,5,5)}
            
        Returns:
            ObsDict with 6 fields per agent
        """
        # Ensure raw_obs is a dict
        if not isinstance(raw_obs, dict):
            # Fallback: create empty observations
            return {agent_id: {
                'pos_xy': (0.0, 0.0),
                'obstacles_enc': np.zeros((3, 5, 5), dtype=np.float32),
                'goal_xy': (0.0, 0.0),
                'nearest_charger_xy': (0.0, 0.0),
                'remain_time': 0.0,
                'remain_battery': 0.0,
            } for agent_id in self.agent_ids}
        
        # Extract with error handling
        obs_arr = raw_obs.get('obs', np.zeros((self.num_agents, 5), dtype=np.float32))  # (M, 5): [row, col, tgt_row, tgt_col, deadline_remaining]
        matrix = raw_obs.get('matrix', np.zeros((self.num_agents, 3, 5, 5), dtype=np.float32))  # (M, 3, 5, 5)
        
        obs_dict: ObsDict = {}
        
        for idx, agent_id in enumerate(self.agent_ids):
            if idx >= len(obs_arr):
                break
            
            # Extract observation: [row, col, tgt_row, tgt_col, deadline_remaining] (with error handling)
            try:
                if isinstance(obs_arr[idx], np.ndarray):
                    obs_row = obs_arr[idx].tolist()
                elif isinstance(obs_arr[idx], (list, tuple)):
                    obs_row = list(obs_arr[idx])
                else:
                    obs_row = [0.0] * 5
                
                if len(obs_row) < 5:
                    obs_row = obs_row + [0.0] * (5 - len(obs_row))
                row, col, tgt_row, tgt_col, deadline_remaining = obs_row[:5]
            except (TypeError, ValueError, IndexError):
                # Fallback to default values
                row, col, tgt_row, tgt_col, deadline_remaining = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Convert (row, col) to (x=col, y=row) for consistency
            pos_xy = (float(col), float(row))
            goal_xy = (float(tgt_col), float(tgt_row))
            
            # Find nearest charger
            if len(self.chargers) > 0:
                nearest_charger_xy = utils.find_nearest_charger(pos_xy, self.chargers)
            else:
                # Fallback: use (0, 0) if no chargers available
                # Note: chargers should be extracted in reset() if available
                nearest_charger_xy = (0.0, 0.0)
            
            # Convert time to steps
            remain_time = utils.convert_time_to_steps(
                float(deadline_remaining),
                self.time_unit,
                self.step_time_sec
            )
            
            # Get battery (TODO: get from env state if available)
            if agent_id in self._agent_batteries:
                battery_value = self._agent_batteries[agent_id]
            else:
                # Fallback: use a default value if not available
                battery_value = self.max_batt_steps
            
            remain_battery = utils.convert_battery_to_steps(
                battery_value,
                self.battery_unit,
                self.max_batt_steps
            )
            
            # Obstacles encoding: matrix channel 0 (obstacles) with error handling
            try:
                if idx < len(matrix):
                    obstacles_enc = matrix[idx]  # (3, 5, 5) - keep full matrix
                    # Ensure correct shape
                    if not isinstance(obstacles_enc, np.ndarray) or obstacles_enc.shape != (3, 5, 5):
                        obstacles_enc = np.zeros((3, 5, 5), dtype=np.float32)
                    else:
                        obstacles_enc = obstacles_enc.astype(np.float32)
                        # Ensure no NaN/Inf
                        obstacles_enc = np.nan_to_num(obstacles_enc, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    obstacles_enc = np.zeros((3, 5, 5), dtype=np.float32)
            except (TypeError, ValueError, IndexError):
                obstacles_enc = np.zeros((3, 5, 5), dtype=np.float32)
            
            obs_dict[agent_id] = {
                'pos_xy': pos_xy,
                'obstacles_enc': obstacles_enc,  # Already converted to float32
                'goal_xy': goal_xy,
                'nearest_charger_xy': nearest_charger_xy,
                'remain_time': remain_time,
                'remain_battery': remain_battery,
            }
        
        return obs_dict
    
    def _to_reward_dict(self, rewards: List[float]) -> RewardDict:
        """Convert list of rewards to RewardDict."""
        reward_dict: RewardDict = {}
        for idx, agent_id in enumerate(self.agent_ids):
            if idx < len(rewards):
                reward_dict[agent_id] = float(rewards[idx])
        return reward_dict
    
    def _to_info_dict(self, raw_info: Dict, obs_dict: ObsDict) -> InfoDict:
        """
        Convert raw info to InfoDict with cost_time and cost_batt.
        
        Args:
            raw_info: Raw info from environment
            obs_dict: Current observation dictionary
            
        Returns:
            InfoDict with cost_time and cost_batt
        """
        # Ensure raw_info is a dict
        if not isinstance(raw_info, dict):
            raw_info = {}
        
        # Calculate instantaneous costs
        cost_time = 0.0
        cost_batt = 0.0
        
        # Check if any agent has violated constraints
        for agent_id in obs_dict:
            obs = obs_dict.get(agent_id, {})
            
            # Time cost: 1.0 if remain_time < 0 else 0.0
            remain_time = obs.get('remain_time', 0.0)
            if remain_time < 0:
                cost_time = 1.0
            
            # Battery cost: 1.0 if remain_battery <= 0 else 0.0
            remain_battery = obs.get('remain_battery', 0.0)
            if remain_battery <= 0:
                cost_batt = 1.0
        
        # Create info dict
        info_dict = self._create_info_dict(cost_time=cost_time, cost_batt=cost_batt)
        # Update with raw_info if it's a dict
        if isinstance(raw_info, dict):
            info_dict.update(raw_info)
        
        return info_dict
    
    def _to_action_list(self, action_dict: ActionDict) -> List[int]:
        """Convert ActionDict to List[int] for environment."""
        actions = []
        for agent_id in self.agent_ids:
            if agent_id in action_dict:
                actions.append(int(action_dict[agent_id]))
            else:
                actions.append(0)  # Default: stay
        return actions
    
    def reset(self, seed: Optional[int] = None) -> Tuple[ObsDict, InfoDict]:
        """
        Reset the environment.
        
        Returns:
            obs_dict: Observations for all agents
            info_dict: Info with cost_time=0, cost_batt=0
        """
        if seed is not None:
            self.seed(seed)
        
        # Reset backend environment
        # Gymnasium returns (obs, info) tuple
        reset_result = self.backend_env.reset(seed=self._seed)
        if isinstance(reset_result, tuple) and len(reset_result) >= 2:
            raw_obs, raw_info = reset_result[0], reset_result[1]
        else:
            # Fallback for old gym interface
            raw_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            raw_info = reset_result[1] if isinstance(reset_result, tuple) and len(reset_result) > 1 else {}
        
        # Extract charger coordinates from environment if not in config
        if len(self.chargers) == 0 and hasattr(self.backend_env, 'charging_cells'):
            # Convert (row, col) to (x=col, y=row) format
            charger_list = [(float(c), float(r)) for r, c in self.backend_env.charging_cells]
            self.chargers = np.array(charger_list, dtype=float) if charger_list else np.array([], dtype=float)
        
        # Update agent batteries from environment if available
        if hasattr(self.backend_env, 'all_agent'):
            for idx, agent_id in enumerate(self.agent_ids):
                agent_key = f'agent_{idx}'
                if agent_key in self.backend_env.all_agent:
                    agent = self.backend_env.all_agent[agent_key]
                    if 'battery' in agent:
                        self._agent_batteries[agent_id] = float(agent['battery'])
        
        # Convert observations
        try:
            obs_dict = self._to_obs_dict(raw_obs)
        except Exception as e:
            # Fallback: create empty observations
            obs_dict = {agent_id: {
                'pos_xy': (0.0, 0.0),
                'obstacles_enc': np.zeros((3, 5, 5), dtype=np.float32),
                'goal_xy': (0.0, 0.0),
                'nearest_charger_xy': (0.0, 0.0),
                'remain_time': 0.0,
                'remain_battery': 0.0,
            } for agent_id in self.agent_ids}
        
        # Create info dict with initial costs (both 0)
        info_dict = self._create_info_dict(cost_time=0.0, cost_batt=0.0)
        # Update with raw_info if it's a dict
        if isinstance(raw_info, dict):
            info_dict.update(raw_info)
        
        return obs_dict, info_dict

    def get_obs_metadata(self) -> Dict[str, Any]:
        meta = super().get_obs_metadata()
        grid_w, grid_h = self.grid_wh if len(self.grid_wh) >= 2 else (1.0, 1.0)
        meta.update({
            'grid_width': float(grid_w),
            'grid_height': float(grid_h),
            'battery_full': float(self.max_batt_steps),
            'deadline_max': float(self.deadline_max),
            'env_type': 'sorting',
        })
        return meta
    
    def step(self, action_dict: ActionDict) -> Tuple[ObsDict, RewardDict, bool, InfoDict]:
        """
        Execute one step.
        
        Returns:
            obs_dict: Observations for all agents
            reward_dict: Rewards for all agents
            done: Global episode-level boolean
            info_dict: Info with cost_time and cost_batt
        """
        # Validate actions
        validated_actions = self._validate_actions(action_dict)
        
        # Convert to list format
        actions_list = self._to_action_list(validated_actions)
        
        # Step backend environment
        raw_obs, rewards_list, terminated, truncated, raw_info = self.backend_env.step(actions_list)
        
        # Aggregate done
        done = self._aggregate_done(terminated) or self._aggregate_done(truncated)
        
        # Update agent batteries from environment if available
        if hasattr(self.backend_env, 'all_agent'):
            for idx, agent_id in enumerate(self.agent_ids):
                agent_key = f'agent_{idx}'
                if agent_key in self.backend_env.all_agent:
                    agent = self.backend_env.all_agent[agent_key]
                    if 'battery' in agent:
                        self._agent_batteries[agent_id] = float(agent['battery'])
        
        # Convert observations
        try:
            obs_dict = self._to_obs_dict(raw_obs)
        except Exception as e:
            # Fallback: create empty observations
            obs_dict = {agent_id: {
                'pos_xy': (0.0, 0.0),
                'obstacles_enc': np.zeros((3, 5, 5), dtype=np.float32),
                'goal_xy': (0.0, 0.0),
                'nearest_charger_xy': (0.0, 0.0),
                'remain_time': 0.0,
                'remain_battery': 0.0,
            } for agent_id in self.agent_ids}
        
        # Convert rewards
        reward_dict = self._to_reward_dict(rewards_list)
        
        # Create info dict with costs
        info_dict = self._to_info_dict(raw_info, obs_dict)
        
        return obs_dict, reward_dict, done, info_dict

