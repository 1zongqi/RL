"""
Mock warehouse environment for testing and development.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any
import random

from .adapter import BaseAdapter
from .types import ObsDict, ActionDict, RewardDict, InfoDict


class MockWarehouseEnv:
    """
    Simple mock environment for testing.
    This is the backend environment that will be wrapped by MockWarehouseAdapter.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        grid_size: int = 10,
        max_steps: int = 100,
        reward_mode: str = "engineering_shaping",
    ):
        """
        Initialize mock environment.
        
        Args:
            num_agents: Number of agents.
            grid_size: Size of the grid world.
            max_steps: Maximum steps per episode.
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_positions = {}
        self.agent_goals = {}
        self.agent_batteries = {}
        self.charger_positions = [(grid_size // 4, grid_size // 4),
                                  (3 * grid_size // 4, 3 * grid_size // 4)]
        self.reward_mode = str(reward_mode or "engineering_shaping").lower()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_step = 0
        # Initialize agent positions
        for i in range(self.num_agents):
            self.agent_positions[i] = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            self.agent_goals[i] = (
                np.random.randint(0, self.grid_size),
                np.random.randint(0, self.grid_size)
            )
            self.agent_batteries[i] = 100.0  # Full battery
        
        obs = {}
        for i in range(self.num_agents):
            obs[i] = self._get_observation(i)
        
        info = {'initialized': True}
        return obs, info
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute one step.
        
        Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
        """
        rewards = {}
        dones = {}
        
        # Move agents
        for agent_id, action in actions.items():
            x, y = self.agent_positions[agent_id]
            
            # Action: 0=up, 1=down, 2=left, 3=right, 4=stay
            if action == 0:  # up
                y = min(y + 1, self.grid_size - 1)
            elif action == 1:  # down
                y = max(y - 1, 0)
            elif action == 2:  # left
                x = max(x - 1, 0)
            elif action == 3:  # right
                x = min(x + 1, self.grid_size - 1)
            # else: stay (action == 4)
            
            self.agent_positions[agent_id] = (x, y)
            
            # Consume battery
            self.agent_batteries[agent_id] -= 1.0
            
            goal_reached = (self.agent_positions[agent_id] == self.agent_goals[agent_id])
            if self.reward_mode == 'strict_paper':
                if goal_reached:
                    rewards[agent_id] = 1.0
                    # Assign a new random goal to keep tasks ongoing
                    self.agent_goals[agent_id] = (
                        np.random.randint(0, self.grid_size),
                        np.random.randint(0, self.grid_size)
                    )
                else:
                    rewards[agent_id] = 0.0
            else:
                if goal_reached:
                    rewards[agent_id] = 10.0
                else:
                    rewards[agent_id] = -0.1
            
            # Check if battery depleted
            if self.agent_batteries[agent_id] <= 0:
                dones[agent_id] = True
            else:
                dones[agent_id] = False
        
        self.current_step += 1
        done = self.current_step >= self.max_steps or any(dones.values())
        
        obs = {}
        for i in range(self.num_agents):
            obs[i] = self._get_observation(i)
        
        info = {'step': self.current_step}
        return obs, rewards, done, info
    
    def _get_observation(self, agent_id: int) -> Dict:
        """Get observation for an agent."""
        x, y = self.agent_positions[agent_id]
        goal_x, goal_y = self.agent_goals[agent_id]
        
        # Find nearest charger
        nearest_charger = min(
            self.charger_positions,
            key=lambda c: abs(c[0] - x) + abs(c[1] - y)
        )
        
        # Simple obstacles encoding (empty for now)
        obstacles_enc = np.zeros(4, dtype=np.float32)
        
        return {
            'pos_xy': (float(x), float(y)),
            'obstacles_enc': obstacles_enc,
            'goal_xy': (float(goal_x), float(goal_y)),
            'nearest_charger_xy': (float(nearest_charger[0]), float(nearest_charger[1])),
            'remain_time': float(self.max_steps - self.current_step),
            'remain_battery': float(self.agent_batteries[agent_id]),
        }
    
    def close(self):
        """Clean up."""
        pass


class MockWarehouseAdapter(BaseAdapter):
    """
    Adapter for MockWarehouseEnv that implements the EnvAdapter interface.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        grid_size: int = 10,
        max_steps: int = 100,
        action_dim: int = 5,
        reward_mode: str = "engineering_shaping",
    ):
        """
        Initialize the adapter.
        
        Args:
            num_agents: Number of agents.
            grid_size: Size of the grid world.
            max_steps: Maximum steps per episode.
            action_dim: Action space dimension (default 5: up, down, left, right, stay).
        """
        backend_env = MockWarehouseEnv(num_agents, grid_size, max_steps, reward_mode=reward_mode)
        super().__init__(backend_env, num_agents, action_dim)
        self.max_steps = max_steps
        self.reward_mode = backend_env.reward_mode
    
    def reset(self, seed: Optional[int] = None) -> Tuple[ObsDict, InfoDict]:
        """
        Reset the environment.
        
        Returns:
            obs_dict: Observations for all agents.
            info_dict: Info with cost_time=0, cost_batt=0.
        """
        if seed is not None:
            self.seed(seed)
        
        obs_dict, backend_info = self.backend_env.reset(seed=self._seed)
        
        # Create info dict with required fields
        info_dict = self._create_info_dict(cost_time=0.0, cost_batt=0.0)
        info_dict.update(backend_info)
        
        return obs_dict, info_dict

    def get_obs_metadata(self) -> Dict[str, Any]:
        grid_size = float(self.backend_env.grid_size)
        meta = super().get_obs_metadata()
        meta.update({
            'grid_width': grid_size,
            'grid_height': grid_size,
            'battery_full': 100.0,
            'deadline_max': float(self.max_steps),
            'env_type': 'mock',
        })
        return meta
    
    def step(self, action_dict: ActionDict) -> Tuple[ObsDict, RewardDict, bool, InfoDict]:
        """
        Execute one step.
        
        Returns:
            obs_dict: Observations for all agents.
            reward_dict: Rewards for all agents.
            done: Global episode-level boolean.
            info_dict: Info with cost_time and cost_batt.
        """
        # Validate actions
        validated_actions = self._validate_actions(action_dict)
        
        # Step the backend environment
        obs_dict, reward_dict, backend_done, backend_info = self.backend_env.step(validated_actions)
        
        # Aggregate done (should already be global, but ensure it)
        done = self._aggregate_done(backend_done)
        
        strict_mode = (self.reward_mode == 'strict_paper')

        cost_time = 1.0  # absolute cost per step
        cost_batt = 0.0
        rew_R: Dict[int, float] = {}
        rew_time: Dict[int, float] = {}
        rew_batt: Dict[int, float] = {}

        if strict_mode:
            low_batt_detected = False
            for agent_id in range(self.num_agents):
                remain_battery = obs_dict.get(agent_id, {}).get('remain_battery', 100.0)
                rew_R[agent_id] = float(reward_dict.get(agent_id, 0.0))
                rew_time[agent_id] = -1.0
                batt_cost = -0.5 if remain_battery < 15.0 else 0.0
                rew_batt[agent_id] = batt_cost
                if batt_cost < 0.0:
                    low_batt_detected = True

            if low_batt_detected:
                cost_batt = 0.5
        else:
            for agent_id in range(self.num_agents):
                remain_battery = obs_dict.get(agent_id, {}).get('remain_battery', 100.0)
                rew_R[agent_id] = float(reward_dict.get(agent_id, 0.0))
                rew_time[agent_id] = -float(cost_time)
                batt_cost = -1.0 if remain_battery <= 0 else 0.0
                rew_batt[agent_id] = batt_cost
                if batt_cost < 0.0:
                    cost_batt = 1.0

        # Create info dict with required fields
        info_dict = self._create_info_dict(cost_time=cost_time, cost_batt=cost_batt)
        info_dict.update(backend_info)

        info_dict['raw_R'] = [rew_R[aid] for aid in range(self.num_agents)]
        info_dict['raw_time'] = [rew_time[aid] for aid in range(self.num_agents)]
        info_dict['raw_batt'] = [rew_batt[aid] for aid in range(self.num_agents)]

        reward_heads = {'R': rew_R, 'time': rew_time, 'batt': rew_batt}

        return obs_dict, reward_heads, done, info_dict
