"""
RWARE 环境适配器

实现统一的环境接口，适配 RWARE 仿真环境
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
import random


class RWAREAdapter:
    """
    RWARE 环境适配器
    
    提供统一接口：reset, step
    返回三头回报（R/time/batt）并做同向化+归一化
    """
    
    def __init__(
        self,
        seed: int = 42,
        max_agents: int = 32,
        obs_mode: str = "dict",
        reward_mode: str = "engineering_shaping",
        low_batt_threshold: float = 0.15,
        low_batt_penalty: float = 0.5,
    ):
        """
        初始化适配器
        
        参数:
            seed: 随机种子
            max_agents: 最大智能体数量
            obs_mode: 观测模式（当前仅支持 "dict"）
        """
        self.seed = seed
        self.max_agents = max_agents
        self.obs_mode = obs_mode
        self.reward_mode = str(reward_mode or "engineering_shaping").lower()
        self.strict_paper_mode = (self.reward_mode == "strict_paper")
        self.low_batt_threshold = float(low_batt_threshold)
        self.low_batt_penalty = float(low_batt_penalty)
        
        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        
        # 环境状态（占位实现）
        self.num_agents = max_agents
        self.grid_size = 20  # 网格大小
        self.episode_idx = 0
        self.step_idx = 0
        self.max_steps = 256
        
        # 确保 num_agents 与 max_agents 一致
        if hasattr(self, 'num_agents'):
            self.num_agents = max_agents
        
        # 智能体状态（占位）
        self.agent_states = {}  # {agent_id: {pos, goal, battery, time, ...}}
        self.chargers = []  # 充电桩位置列表
        self.workstations = []  # 工作站位置列表
        self.completed_tasks = 0
        self.total_tasks = 0
        
        # 初始化充电桩和工作站位置
        self._init_chargers()
        self._init_workstations()
    
    def _init_chargers(self):
        """初始化充电桩位置（占位）"""
        num_chargers = max(2, self.num_agents // 4)
        self.chargers = []
        for _ in range(num_chargers):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.chargers.append((x, y))
    
    def _init_workstations(self):
        """初始化工作站位置（占位）"""
        # TODO: 替换为真实 RWARE 工作站位置
        num_stations = max(4, self.num_agents // 2)
        self.workstations = []
        for _ in range(num_stations):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.workstations.append((x, y))
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, Dict], Dict[str, Any]]:
        """
        重置环境
        
        参数:
            seed: 可选的随机种子
        
        返回:
            obs_dict: 观测字典，{agent_id: obs_fields}
            info_dict: 信息字典
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.episode_idx += 1
        self.step_idx = 0
        self.completed_tasks = 0
        self.total_tasks = 0
        
        # 初始化智能体状态（占位）
        # 确保使用 max_agents 作为智能体数量
        num_agents = self.max_agents
        obs_dict = {}
        for agent_id in range(num_agents):
            # 随机初始位置
            pos_x = np.random.uniform(0, self.grid_size)
            pos_y = np.random.uniform(0, self.grid_size)
            
            # 最近工作站作为目标（RWARE 特性）
            nearest_workstation = self._find_nearest_workstation(pos_x, pos_y)
            goal_x, goal_y = nearest_workstation
            
            # 最近充电桩
            nearest_charger = self._find_nearest_charger(pos_x, pos_y)
            
            # 初始电量（0.5~1.0）
            remain_battery = np.random.uniform(0.5, 1.0)
            
            # 剩余时间（任务期限）
            remain_time = np.random.uniform(50, 100)
            
            # 计算到目标的距离
            task_dist = np.sqrt((goal_x - pos_x)**2 + (goal_y - pos_y)**2)
            
            # 存储状态
            self.agent_states[agent_id] = {
                'pos': (pos_x, pos_y),
                'goal': (goal_x, goal_y),
                'battery': remain_battery,
                'time': remain_time,
                'task_dist': task_dist,
                'at_goal': False,
                'at_charger': False
            }
            
            # 构建观测
            obs_dict[agent_id] = {
                'pos_xy': (pos_x, pos_y),
                'goal_xy': (goal_x, goal_y),
                'nearest_charger_xy': nearest_charger,
                'remain_time': remain_time,
                'remain_battery': remain_battery,
                'obstacles_enc': np.zeros(8, dtype=np.float32),  # 占位：障碍物编码
                'task_dist': task_dist,
                'queue_len_charger': 0.0  # TODO: 实现真实队列长度
            }
        
        # 信息字典
        info_dict = {
            'step_idx': 0,
            'episode_idx': self.episode_idx,
            'completed_tasks': 0,
            'throughput': 0.0,
            'avg_queue_charger': 0.0,
            'battery_min': min(s['battery'] for s in self.agent_states.values()) if self.agent_states else 1.0,
            'deadline_violation_rate': 0.0
        }
        
        # 确保 num_agents 与 max_agents 一致
        self.num_agents = num_agents
        
        return obs_dict, info_dict
    
    def _find_nearest_charger(self, x: float, y: float) -> Tuple[float, float]:
        """找到最近的充电桩位置"""
        if not self.chargers:
            return (0.0, 0.0)
        
        min_dist = float('inf')
        nearest = self.chargers[0]
        for cx, cy in self.chargers:
            dist = np.sqrt((cx - x)**2 + (cy - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (float(cx), float(cy))
        return nearest
    
    def _find_nearest_workstation(self, x: float, y: float) -> Tuple[float, float]:
        """找到最近的工作站位置（RWARE 特性）"""
        if not self.workstations:
            return (x, y)
        
        min_dist = float('inf')
        nearest = self.workstations[0]
        for wx, wy in self.workstations:
            dist = np.sqrt((wx - x)**2 + (wy - y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (float(wx), float(wy))
        return nearest
    
    def step(
        self,
        action_dict: Dict[int, int]
    ) -> Tuple[Dict[int, Dict], Dict[str, Dict[int, float]], bool, Dict[str, Any]]:
        """
        执行一步
        
        参数:
            action_dict: 动作字典，{agent_id: action}
        
        返回:
            obs_dict: 观测字典
            rew_head_dict: 三头回报字典，{'R': {agent_id: r}, 'time': {...}, 'batt': {...}}
            done: 是否结束
            info_dict: 信息字典
        """
        self.step_idx += 1
        
        # 动作映射（占位）：0=上, 1=下, 2=左, 3=右, 4=停留, 5=充电
        action_deltas = {
            0: (0, 1),
            1: (0, -1),
            2: (-1, 0),
            3: (1, 0),
            4: (0, 0),
            5: (0, 0)  # 充电动作
        }
        
        # 更新智能体状态
        new_obs_dict = {}
        rew_raw_dict = {
            'R': {},
            'time': {},
            'batt': {}
        }
        
        completed_this_step = 0
        battery_min = 1.0
        deadline_violations = 0
        
        # 确保使用 max_agents 作为智能体数量
        num_agents = self.max_agents
        for agent_id in range(num_agents):
            state = self.agent_states[agent_id]
            action = action_dict.get(agent_id, 4)  # 默认停留
            
            pos_x, pos_y = state['pos']
            goal_x, goal_y = state['goal']
            remain_battery = state['battery']
            remain_time = state['time']
            
            # 充电逻辑：低电量时优先去充电桩
            if remain_battery < 0.15 and action != 5:
                nearest_charger = self._find_nearest_charger(pos_x, pos_y)
                cx, cy = nearest_charger
                if abs(cx - pos_x) > 0.1:
                    action = 3 if cx > pos_x else 2
                elif abs(cy - pos_y) > 0.1:
                    action = 0 if cy > pos_y else 1
                else:
                    action = 5  # 到达充电桩，开始充电
            
            # 执行动作
            if action == 5:
                # 充电：恢复电量
                remain_battery = min(1.0, remain_battery + 0.1)
                state['at_charger'] = True
            else:
                # 移动
                dx, dy = action_deltas.get(action, (0, 0))
                new_x = np.clip(pos_x + dx * 0.5, 0, self.grid_size)
                new_y = np.clip(pos_y + dy * 0.5, 0, self.grid_size)
                pos_x, pos_y = new_x, new_y
                
                # 消耗电量（与移动距离相关）
                move_dist = np.sqrt(dx**2 + dy**2) * 0.5
                remain_battery = max(0.0, remain_battery - 0.01 * move_dist)
                state['at_charger'] = False
            
            # 更新位置
            state['pos'] = (pos_x, pos_y)
            state['battery'] = remain_battery
            
            # 检查是否到达目标（工作站）
            dist_to_goal = np.sqrt((goal_x - pos_x)**2 + (goal_y - pos_y)**2)
            state['task_dist'] = dist_to_goal
            
            task_completed = False
            if dist_to_goal < 0.5 and not state['at_goal']:
                state['at_goal'] = True
                task_completed = True
                completed_this_step += 1
                self.completed_tasks += 1
                self.total_tasks += 1
                
                # 生成新目标（最近工作站）
                nearest_workstation = self._find_nearest_workstation(pos_x, pos_y)
                goal_x, goal_y = nearest_workstation
                state['goal'] = (goal_x, goal_y)
                state['at_goal'] = False
                state['time'] = np.random.uniform(50, 100)  # 新任务期限
            
            # 更新剩余时间
            remain_time = max(0.0, remain_time - 1.0)
            state['time'] = remain_time
            
            # 检查超时
            if remain_time <= 0 and not state['at_goal']:
                deadline_violations += 1
            
            # 计算原始回报（区分模式）
            if self.strict_paper_mode:
                r_main = 1.0 if task_completed else 0.0
                r_time = -1.0
                if remain_battery is None:
                    r_batt = 0.0
                else:
                    r_batt = -self.low_batt_penalty if remain_battery < self.low_batt_threshold else 0.0
            else:
                r_main = 1.0 if task_completed else 0.0
                r_main += 0.1 * (1.0 - dist_to_goal / self.grid_size)

                r_time = 0.0
                if task_completed and remain_time > 0:
                    r_time += 1.0
                elif remain_time <= 0 and not task_completed:
                    r_time -= 1.0
                r_time += 0.2 * (remain_time / 100.0)

                r_batt = remain_battery if remain_battery is not None else 0.0
                if state['at_charger']:
                    r_batt += 0.5
                if remain_battery is not None and remain_battery < self.low_batt_threshold:
                    r_batt -= self.low_batt_penalty
            
            rew_raw_dict['R'][agent_id] = r_main
            rew_raw_dict['time'][agent_id] = r_time
            rew_raw_dict['batt'][agent_id] = r_batt
            
            # 更新观测
            nearest_charger = self._find_nearest_charger(pos_x, pos_y)
            new_obs_dict[agent_id] = {
                'pos_xy': (pos_x, pos_y),
                'goal_xy': (goal_x, goal_y),
                'nearest_charger_xy': nearest_charger,
                'remain_time': remain_time,
                'remain_battery': remain_battery,
                'obstacles_enc': np.zeros(8, dtype=np.float32),  # 占位
                'task_dist': dist_to_goal,
                'queue_len_charger': 0.0  # TODO: 实现真实队列长度
            }
            
            battery_min = min(battery_min, remain_battery)
        
        # 归一化回报（同向化：越大越好）
        rew_head_dict = self._normalize_rewards(rew_raw_dict)
        
        # 检查是否结束
        done = (self.step_idx >= self.max_steps) or (completed_this_step >= num_agents)
        
        # 计算信息
        throughput = self.completed_tasks / max(1, self.step_idx)
        violation_rate = deadline_violations / max(1, num_agents)
        
        info_dict = {
            'step_idx': self.step_idx,
            'episode_idx': self.episode_idx,
            'completed_tasks': self.completed_tasks,
            'throughput': throughput,
            'avg_queue_charger': 0.0,  # TODO: 实现真实队列统计
            'battery_min': battery_min,
            'deadline_violation_rate': violation_rate,
            'raw_R': [rew_raw_dict['R'].get(agent_id, 0.0) for agent_id in range(self.max_agents)],
            'raw_time': [rew_raw_dict['time'].get(agent_id, 0.0) for agent_id in range(self.max_agents)],
            'raw_batt': [rew_raw_dict['batt'].get(agent_id, 0.0) for agent_id in range(self.max_agents)],
        }
        
        return new_obs_dict, rew_head_dict, done, info_dict
    
    def _normalize_rewards(
        self,
        rew_raw_dict: Dict[str, Dict[int, float]]
    ) -> Dict[str, Dict[int, float]]:
        """
        归一化回报（同向化：越大越好）
        
        对每个头进行标准化，并做 tanh 压缩
        """
        rew_head_dict = {}
        
        for head in ['R', 'time', 'batt']:
            rewards = list(rew_raw_dict[head].values())
            if len(rewards) == 0:
                rew_head_dict[head] = {i: 0.0 for i in range(self.max_agents)}
                continue
            
            # 转换为 numpy 数组
            arr = np.array(rewards, dtype=np.float32)
            
            # 防止 NaN/Inf
            arr = np.clip(arr, -10.0, 10.0)
            
            # 标准化
            mean = arr.mean()
            std = arr.std()
            if std < 1e-6:
                std = 1.0
            
            normalized = (arr - mean) / std
            
            # tanh 压缩
            normalized = np.tanh(normalized)
            
            # 再次防止极端值
            normalized = np.clip(normalized, -2.0, 2.0)
            
            # 转换回字典
            rew_head_dict[head] = {
                agent_id: float(normalized[i])
                for i, agent_id in enumerate(rew_raw_dict[head].keys())
            }
        
        return rew_head_dict

