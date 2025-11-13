"""
向量化环境采样器

从环境中收集 rollout 数据
"""

import torch
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING, List

from ..utils.typing_alias import RolloutDict
from ..nets.encoders import BaseEncoder, FlatEncoder18D, EncoderOutput
from ..irdc.graph import build_adjacency
from envs.types import PaperObsEntry
from ..agents import AgentRouting, PolicyRouter, BatchBuilder
from ..memory import PolicyBufferStore

if TYPE_CHECKING:
    from ..irdc.irdc import IRDC


class VectorRunner:
    """
    向量化环境采样器
    
    从环境中收集 rollout 数据（obs, act, logp, rew_dict, done, agent_id）
    支持真实环境适配器（FulfillmentAdapter/RWAREAdapter）或占位模式（stub）
    """
    
    def __init__(
        self,
        backend_adapter: Optional[Any] = None,  # 环境适配器对象（FulfillmentAdapter/RWAREAdapter）
        policy: torch.nn.Module = None,
        critic: torch.nn.Module = None,
        policy_map: Optional[Dict[str, torch.nn.Module]] = None,
        critic_map: Optional[Dict[str, torch.nn.Module]] = None,
        routing: Optional[AgentRouting] = None,
        buffer_store: Optional[PolicyBufferStore] = None,
        intrinsic_module: Optional["IRDC"] = None,
        steps_per_agent: int = 64,
        num_agents: int = 8,
        obs_dim: Optional[int] = None,
        act_dim: int = 6,
        episodes: int = 8,
        max_steps_per_ep: int = 256,
        device: str = 'cpu',
        obs_encoder: Optional[BaseEncoder] = None,
        obs_config: Optional[Dict[str, Any]] = None,
        intrinsic_cfg: Optional[Dict[str, Any]] = None,
        frontends_cfg: Optional[Dict[str, Any]] = None,
        gat_cfg: Optional[Dict[str, Any]] = None,
        graphs_cfg: Optional[Dict[str, Any]] = None,
        constraints_cfg: Optional[Dict[str, Any]] = None,
        reward_mode: str = "engineering_shaping",
        policy_key: str = "policy:shared",
        reward_source: str = "raw",
    ):
        """
        初始化采样器
        
        参数:
            backend_adapter: 环境适配器对象（None 表示使用占位模式）
            policy: 策略网络
            critic: Critic 网络
            steps_per_agent: 每个智能体的步数
            num_agents: 智能体数量
            obs_dim: 观测维度（占位模式使用）
            act_dim: 动作维度
            episodes: episode 数量（真实环境模式使用）
            max_steps_per_ep: 每个 episode 的最大步数（真实环境模式使用）
            device: 设备
        """
        self.backend_adapter = backend_adapter
        self.policy = policy
        self.critic = critic
        self.policy_map = policy_map or {}
        self.critic_map = critic_map or {}
        self.routing = routing
        self.buffer_store = buffer_store
        self.multi_policy_mode = bool(self.policy_map)
        if self.multi_policy_mode:
            if self.routing is None:
                raise ValueError("多策略模式需要提供 AgentRouting")
        self.steps_per_agent = steps_per_agent
        self.num_agents = num_agents
        self.obs_encoder = obs_encoder if obs_encoder is not None else FlatEncoder18D()
        self.obs_config = obs_config or {}
        self.use_paper_obs = bool(self.obs_config.get('use_paper_obs', False))
        self.obs_fov_size = int(self.obs_config.get('fov_size', 3))
        self.encoder_version = self.obs_config.get('encoder_version', 'v1')
        self.obs_norm_defaults = self.obs_config.get('norm', {})
        self.obs_dim = obs_dim if obs_dim is not None else self.obs_encoder.out_dim
        self.act_dim = act_dim
        self.episodes = episodes
        self.max_steps_per_ep = max_steps_per_ep
        self.device = device
        self.obs_mode = 'paper_obs' if self.use_paper_obs else 'flat18'
        self._fallback_obs_mode = False
        self.policy_key = policy_key
        self.reward_source = (reward_source or "raw").lower()
        if self.reward_source not in {'raw', 'normalized'}:
            self.reward_source = 'raw'

        # 前端与图配置
        self.frontends_cfg = frontends_cfg or {}
        self.policy_frontend_cfg = self.frontends_cfg.get('policy', {})
        self.critic_frontend_cfg = self.frontends_cfg.get('critic', {})
        self.policy_frontend_use_gat = bool(self.policy_frontend_cfg.get('use_gat', False))
        self.critic_frontend_use_gat = bool(self.critic_frontend_cfg.get('use_gat', False))

        self.gat_cfg = gat_cfg or {}
        self.graphs_cfg = graphs_cfg or {}
        self.graph_neighbor_mode = self.gat_cfg.get('neighbor_mode', 'fov')
        self.graph_radius = float(self.gat_cfg.get('radius', 2.5))
        self.graph_self_loop = bool(self.gat_cfg.get('self_loop', True))
        self.graph_symmetry = self.graphs_cfg.get('symmetry', 'union')
        self.graph_cache_mode = str(self.graphs_cfg.get('cache_adj_in_buffer', 'dense')).lower()
        self.constraints_cfg = constraints_cfg or {}
        self.use_raw_for_training = bool(self.constraints_cfg.get('use_raw_for_training', True))
        self.reward_mode = str(reward_mode or "engineering_shaping").lower()
        topk_cfg = self.graphs_cfg.get('topk_neighbors')
        if topk_cfg is None:
            self.graph_topk = None
        else:
            topk_val = int(topk_cfg)
            self.graph_topk = topk_val if topk_val > 0 else None
        self.graph_max_degree = self.graphs_cfg.get('max_degree')
        self.graph_topk_jitter = bool(self.graphs_cfg.get('deterministic_topk_jitter', True))

        if self.multi_policy_mode:
            self.policy_router = PolicyRouter(self.routing.policy_of)
            self.policy_keys = list(self.policy_map.keys())
            self.policy_to_critic: Dict[str, str] = {}
            for agent_id, policy_key in self.routing.policy_of.items():
                critic_key = self.routing.critic_of.get(agent_id, "critic:shared")
                self.policy_to_critic.setdefault(policy_key, critic_key)
            self.policy_agent_ids: Dict[str, List[int]] = {}
            for agent_id, policy_key in self.routing.policy_of.items():
                self.policy_agent_ids.setdefault(policy_key, []).append(int(agent_id))
            for key in self.policy_agent_ids:
                self.policy_agent_ids[key].sort()
            self.per_policy_h_policy: Dict[str, Optional[torch.Tensor]] = {k: None for k in self.policy_map}
            self.per_policy_h_critic: Dict[str, Optional[torch.Tensor]] = {k: None for k in self.policy_map}
        else:
            self.policy_router = None
            self.policy_keys = []
            self.policy_to_critic = {}
            self.policy_agent_ids = {}

        # Intrinsic reward configuration
        self.irdc = intrinsic_module
        self.intrinsic_cfg = intrinsic_cfg or {}
        self.use_intrinsic = bool(self.intrinsic_cfg.get('use_intrinsic', False) and self.irdc is not None)
        self.intrinsic_epsilon = float(self.intrinsic_cfg.get('epsilon', 0.0)) if self.use_intrinsic else 0.0
        gat_cfg = self.intrinsic_cfg.get('gat', {}) or {}
        self.intrinsic_use_gat = bool(self.intrinsic_cfg.get('use_gat', False))
        self.intrinsic_neighbor_mode = gat_cfg.get('neighbor_mode', 'identity')
        self.intrinsic_neighbor_radius = float(gat_cfg.get('radius', 1.0))
        self.h_irdc: Optional[torch.Tensor] = None
        if self.use_intrinsic:
            self.h_irdc = self.irdc.initial_state(
                self.num_agents,
                torch.device(device) if not isinstance(device, torch.device) else device,
            )
        
        self.require_adj_step = (
            self.policy_frontend_use_gat
            or self.critic_frontend_use_gat
            or (self.use_intrinsic and self.intrinsic_use_gat)
        )
        self.store_graph_meta = self.require_adj_step or self.graph_cache_mode != 'recompute'
        
        # 判断是否使用真实环境
        self.use_real_env = (backend_adapter is not None)

        self._policy_state_device = torch.device(device)
        self._critic_state_device = torch.device(device)

        if self.multi_policy_mode:
            self.use_policy_rnn = any(getattr(module, 'use_rnn', False) for module in self.policy_map.values())
            self.use_rnn_critic = any(getattr(module, 'use_rnn', False) for module in self.critic_map.values())
            self.h_policy: Optional[torch.Tensor] = None
            self.h_critic: Optional[torch.Tensor] = None
            for policy_key in self.policy_keys:
                group_size = len(self.policy_agent_ids.get(policy_key, [])) or self.num_agents
                policy_module = self.policy_map[policy_key]
                if getattr(policy_module, 'use_rnn', False):
                    self.per_policy_h_policy[policy_key] = policy_module.initial_state(group_size, self._policy_state_device)
                else:
                    self.per_policy_h_policy[policy_key] = None

                critic_key = self.policy_to_critic.get(policy_key)
                critic_module = self.critic_map[critic_key] if critic_key in self.critic_map else next(iter(self.critic_map.values()), None)
                if critic_module is not None and getattr(critic_module, 'use_rnn', False):
                    self.per_policy_h_critic[policy_key] = critic_module.initial_state(group_size, self._critic_state_device)
                else:
                    self.per_policy_h_critic[policy_key] = None
        else:
            self.use_policy_rnn = getattr(self.policy, 'use_rnn', False)
            self.use_rnn_critic = getattr(self.critic, 'use_rnn', False)
            self.h_policy: Optional[torch.Tensor] = None
            if self.use_policy_rnn:
                self.h_policy = self.policy.initial_state(self.num_agents, self._policy_state_device)
            self.h_critic: Optional[torch.Tensor] = None
            if self.use_rnn_critic:
                self.h_critic = self.critic.initial_state(self.num_agents, self._critic_state_device)
            self.per_policy_h_policy = {}
            self.per_policy_h_critic = {}
    
    def _obs_dict_to_vector(self, obs_dict: Dict[int, Dict]) -> torch.Tensor:
        """
        将字典观测转换为向量
        
        固定字段顺序：pos_xy(2) + goal_xy(2) + nearest_charger_xy(2) + 
                     remain_time(1) + remain_battery(1) + obstacles_enc(8) + 
                     task_dist(1) + queue_len_charger(1) = 18维
        
        返回:
            obs_vecs: 形状 [num_agents, 18] 的张量
        """
        obs_dim = 18  # 固定维度
        obs_vecs = []
        
        # 确保按 agent_id 排序
        agent_ids = sorted(obs_dict.keys())
        
        for agent_id in agent_ids:
            obs = obs_dict[agent_id]
            vec = []
            
            # pos_xy (2)
            pos_xy = obs.get('pos_xy', (0.0, 0.0))
            vec.extend([float(pos_xy[0]), float(pos_xy[1])])
            
            # goal_xy (2)
            goal_xy = obs.get('goal_xy', (0.0, 0.0))
            vec.extend([float(goal_xy[0]), float(goal_xy[1])])
            
            # nearest_charger_xy (2)
            charger_xy = obs.get('nearest_charger_xy', (0.0, 0.0))
            vec.extend([float(charger_xy[0]), float(charger_xy[1])])
            
            # remain_time (1)
            vec.append(float(obs.get('remain_time', 0.0)))
            
            # remain_battery (1)
            vec.append(float(obs.get('remain_battery', 0.0)))
            
            # obstacles_enc (8)
            obstacles = obs.get('obstacles_enc', np.zeros(8, dtype=np.float32))
            if isinstance(obstacles, np.ndarray):
                obstacles = obstacles.flatten()
                if obstacles.shape[0] < 8:
                    pad = np.zeros(8 - obstacles.shape[0], dtype=np.float32)
                    obstacles = np.concatenate([obstacles, pad], axis=0)
                vec.extend([float(x) for x in obstacles[:8]])
            else:
                vec.extend([0.0] * 8)
            
            # task_dist (1)
            vec.append(float(obs.get('task_dist', 0.0)))
            
            # queue_len_charger (1)
            vec.append(float(obs.get('queue_len_charger', 0.0)))
            
            obs_vecs.append(vec)
        
        # 转换为张量
        if len(obs_vecs) == 0:
            # 如果没有数据，返回零张量
            return torch.zeros((self.num_agents, obs_dim), dtype=torch.float32, device=self.device)
        
        return torch.tensor(obs_vecs, dtype=torch.float32, device=self.device)

    def _align_entities(
        self,
        features: torch.Tensor,
        positions: Optional[torch.Tensor],
        fov: Optional[torch.Tensor],
        active_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B = features.shape[0]
        target = self.num_agents

        if positions is None:
            positions = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(self.device, dtype=torch.float32)
        if active_mask is None:
            active_mask = torch.ones(B, dtype=torch.bool, device=self.device)
        else:
            active_mask = active_mask.to(self.device, dtype=torch.bool)
        if fov is not None:
            fov = fov.to(self.device, dtype=torch.float32)

        if B < target:
            pad = target - B
            feat_pad = torch.zeros((pad,) + features.shape[1:], dtype=features.dtype, device=features.device)
            features = torch.cat([features, feat_pad], dim=0)

            pos_pad = torch.zeros((pad, 2), dtype=torch.float32, device=self.device)
            positions = torch.cat([positions, pos_pad], dim=0)

            mask_pad = torch.zeros(pad, dtype=torch.bool, device=self.device)
            active_mask = torch.cat([active_mask, mask_pad], dim=0)

            if fov is not None:
                fov_pad = torch.zeros((pad,) + fov.shape[1:], dtype=torch.float32, device=self.device)
                fov = torch.cat([fov, fov_pad], dim=0)
        elif B > target:
            features = features[:target]
            positions = positions[:target]
            active_mask = active_mask[:target]
            if fov is not None:
                fov = fov[:target]

        if fov is None:
            fov_aligned = None
        else:
            fov_aligned = fov
        return features, positions, fov_aligned, active_mask

    def _resolve_reward_source(
        self,
        reward_norm: torch.Tensor,
        time_norm: torch.Tensor,
        batt_norm: torch.Tensor,
        reward_raw: torch.Tensor,
        time_raw: torch.Tensor,
        batt_raw: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        if reward_norm.numel() == 0 or mask_tensor.numel() == 0:
            used = 'auto_fallback' if self.reward_source == 'raw' else 'normalized'
            return {
                'reward_source_tensor': reward_norm,
                'time_source_tensor': time_norm,
                'batt_source_tensor': batt_norm,
                'reward_source_used': used,
                'raw_present_steps': 0,
                'raw_finite_ratio': 0.0,
                'raw_reward_sum': 0.0,
                'raw_available': False,
            }

        valid_raw_mask = (
            torch.isfinite(reward_raw)
            & torch.isfinite(time_raw)
            & torch.isfinite(batt_raw)
        )
        raw_step_complete = valid_raw_mask.all(dim=1)
        raw_present_steps = int(raw_step_complete.sum().item())
        raw_total_steps = reward_raw.shape[0]
        raw_finite_ratio = float(raw_present_steps / raw_total_steps) if raw_total_steps > 0 else 0.0

        prefer_raw = self.use_raw_for_training or self.reward_source == 'raw'

        reward_source_tensor = reward_norm
        time_source_tensor = time_norm
        batt_source_tensor = batt_norm
        reward_source_used = 'normalized'

        if prefer_raw:
            if raw_finite_ratio > 0.99:
                reward_source_used = 'raw'
                reward_source_tensor = torch.where(valid_raw_mask, reward_raw, reward_norm)
                time_source_tensor = torch.where(valid_raw_mask, time_raw, time_norm)
                batt_source_tensor = torch.where(valid_raw_mask, batt_raw, batt_norm)
            else:
                reward_source_used = 'auto_fallback'

        raw_available = prefer_raw and (reward_source_used == 'raw')
        raw_reward_sum = float(
            (
                torch.where(
                    valid_raw_mask,
                    reward_raw,
                    torch.zeros_like(reward_raw),
                )
                * mask_tensor
            ).sum().item()
        )

        return {
            'reward_source_tensor': reward_source_tensor,
            'time_source_tensor': time_source_tensor,
            'batt_source_tensor': batt_source_tensor,
            'reward_source_used': reward_source_used,
            'raw_present_steps': raw_present_steps,
            'raw_finite_ratio': raw_finite_ratio,
            'raw_reward_sum': raw_reward_sum,
            'raw_available': raw_available,
        }

    def _compute_adjacency_matrix(
        self,
        positions: torch.Tensor,
        fov: Optional[torch.Tensor],
        active_mask: torch.Tensor,
        fallback: bool,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.zeros((self.num_agents, self.num_agents), dtype=torch.bool, device=self.device)

        effective_mode = self.graph_neighbor_mode
        if effective_mode == 'fov' and (fov is None or fallback):
            effective_mode = 'radius'

        adj = build_adjacency(
            positions,
            fov=fov,
            mode=effective_mode,
            radius=self.graph_radius,
            symmetry=self.graph_symmetry,
            self_loop=self.graph_self_loop,
            inactive=active_mask,
            topk_neighbors=self.graph_topk,
            deterministic_topk_jitter=self.graph_topk_jitter,
        )
        # move to desired device/bool
        return adj.to(self.device, dtype=torch.bool)

    def _encode_observations_flat(self, obs_dict: Dict[int, Dict]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        flat = self._obs_dict_to_vector(obs_dict)
        encoder_output = self.obs_encoder({'flat': flat}, {'norm_consts': {}})
        agent_ids = sorted(obs_dict.keys())
        positions = []
        for agent_id in agent_ids:
            pos_xy = obs_dict[agent_id].get('pos_xy', (0.0, 0.0))
            positions.append([float(pos_xy[0]), float(pos_xy[1])])
        if len(positions) == 0:
            positions_tensor = torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            active_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
        else:
            positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
            active_mask = torch.ones(len(positions), dtype=torch.bool, device=self.device)
        meta = {
            'obs_mode': 'flat18',
            'obs_dim': self.obs_encoder.out_dim,
            'norm_consts': {},
            'obs_fallback': False,
            'positions': positions_tensor,
            'fov': None,
            'active_mask': active_mask,
            'paper_obs_fallback': True,
        }
        return encoder_output.features, meta

    def _get_adapter_metadata(self) -> Dict[str, Any]:
        if self.backend_adapter is None:
            return {}
        if hasattr(self.backend_adapter, 'get_obs_metadata'):
            try:
                meta = self.backend_adapter.get_obs_metadata() or {}
                if not isinstance(meta, dict):
                    return {}
                return meta
            except Exception:
                return {}
        return {}

    def _resolve_norm_consts(self, adapter_meta: Dict[str, Any]) -> Dict[str, float]:
        norm_cfg = self.obs_norm_defaults or {}
        battery_full = adapter_meta.get('battery_full', norm_cfg.get('battery_full', 1.0))
        env_type = adapter_meta.get('env_type')
        if env_type == 'sorting':
            deadline_default = norm_cfg.get('deadline_sorting', norm_cfg.get('deadline_default', 1.0))
        elif env_type == 'fulfillment':
            deadline_default = norm_cfg.get('deadline_fulfillment', norm_cfg.get('deadline_default', 1.0))
        else:
            deadline_default = norm_cfg.get('deadline_default', norm_cfg.get('deadline_sorting', 1.0))
        deadline_max = adapter_meta.get('deadline_max', deadline_default)
        grid_width = adapter_meta.get('grid_width', norm_cfg.get('grid_width', 1.0))
        grid_height = adapter_meta.get('grid_height', norm_cfg.get('grid_height', 1.0))
        return {
            'battery_full': float(max(battery_full, 1e-6)),
            'deadline_max': float(max(deadline_max, 1e-6)),
            'grid_width': float(max(grid_width, 1e-6)),
            'grid_height': float(max(grid_height, 1e-6)),
        }

    def _build_paper_obs_fallback(self, obs_dict: Dict[int, Dict], fov_size: int) -> Tuple[Dict[int, PaperObsEntry], bool]:
        paper_obs = {}
        fallback = True
        for agent_id, obs in obs_dict.items():
            pos_xy = obs.get('pos_xy', (0.0, 0.0))
            goal_xy = obs.get('goal_xy', (0.0, 0.0))
            charger_xy = obs.get('nearest_charger_xy', (0.0, 0.0))
            obstacles_enc = obs.get('obstacles_enc')
            if isinstance(obstacles_enc, np.ndarray) and obstacles_enc.ndim == 3:
                arr = np.asarray(obstacles_enc, dtype=np.float32)
            else:
                arr = np.zeros((3, fov_size, fov_size), dtype=np.float32)
            paper_obs[agent_id] = PaperObsEntry(
                px=float(pos_xy[0]),
                py=float(pos_xy[1]),
                gx=float(goal_xy[0]),
                gy=float(goal_xy[1]),
                ex=float(charger_xy[0]),
                ey=float(charger_xy[1]),
                deadline=float(obs.get('remain_time', 0.0)),
                battery=float(obs.get('remain_battery', 0.0)),
                fov=np.zeros((3, fov_size, fov_size), dtype=np.float32),
            )
        return paper_obs, fallback

    def _encode_observations(self, obs_dict: Dict[int, Dict]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not self.use_paper_obs or self._fallback_obs_mode:
            return self._encode_observations_flat(obs_dict)

        adapter_meta = self._get_adapter_metadata()
        norm_consts = self._resolve_norm_consts(adapter_meta)

        build_fn = getattr(self.backend_adapter, 'build_paper_obs', None)
        try:
            if callable(build_fn):
                paper_obs_dict, fallback = build_fn(obs_dict, self.obs_fov_size)
            else:
                paper_obs_dict, fallback = self._build_paper_obs_fallback(obs_dict, self.obs_fov_size)
        except Exception:
            self._fallback_obs_mode = True
            return self._encode_observations_flat(obs_dict)

        agent_ids = sorted(obs_dict.keys())
        if len(agent_ids) == 0:
            features = torch.zeros((0, self.obs_encoder.out_dim), device=self.device)
            meta = {'obs_mode': 'paper_obs', 'obs_dim': self.obs_encoder.out_dim, 'norm_consts': norm_consts, 'obs_fallback': True}
            return features, meta

        px_list = []
        py_list = []
        gx_list = []
        gy_list = []
        ex_list = []
        ey_list = []
        deadline_list = []
        battery_list = []
        fov_list = []

        for agent_id in agent_ids:
            entry = paper_obs_dict.get(agent_id)
            if entry is None:
                entry = PaperObsEntry(
                    px=0.0,
                    py=0.0,
                    gx=0.0,
                    gy=0.0,
                    ex=0.0,
                    ey=0.0,
                    deadline=0.0,
                    battery=0.0,
                    fov=np.zeros((3, self.obs_fov_size, self.obs_fov_size), dtype=np.float32),
                )
                fallback = True

            if isinstance(entry, PaperObsEntry):
                px_list.append(entry.px)
                py_list.append(entry.py)
                gx_list.append(entry.gx)
                gy_list.append(entry.gy)
                ex_list.append(entry.ex)
                ey_list.append(entry.ey)
                deadline_list.append(entry.deadline)
                battery_list.append(entry.battery)
                fov_list.append(entry.fov)
            else:
                px_list.append(entry.get('px', 0.0))
                py_list.append(entry.get('py', 0.0))
                gx_list.append(entry.get('gx', 0.0))
                gy_list.append(entry.get('gy', 0.0))
                ex_list.append(entry.get('ex', 0.0))
                ey_list.append(entry.get('ey', 0.0))
                deadline_list.append(entry.get('deadline', 0.0))
                battery_list.append(entry.get('battery', 0.0))
                fov_list.append(entry.get('fov', np.zeros((3, self.obs_fov_size, self.obs_fov_size), dtype=np.float32)))

        obs_fields = {
            'px': torch.tensor(px_list, dtype=torch.float32, device=self.device),
            'py': torch.tensor(py_list, dtype=torch.float32, device=self.device),
            'gx': torch.tensor(gx_list, dtype=torch.float32, device=self.device),
            'gy': torch.tensor(gy_list, dtype=torch.float32, device=self.device),
            'ex': torch.tensor(ex_list, dtype=torch.float32, device=self.device),
            'ey': torch.tensor(ey_list, dtype=torch.float32, device=self.device),
            'deadline': torch.tensor(deadline_list, dtype=torch.float32, device=self.device),
            'battery': torch.tensor(battery_list, dtype=torch.float32, device=self.device),
            'fov': torch.tensor(np.stack(fov_list), dtype=torch.float32, device=self.device),
        }

        encoder_info = {'norm_consts': norm_consts}
        encoder_output: EncoderOutput = self.obs_encoder(obs_fields, encoder_info)

        positions_tensor = torch.tensor(
            list(zip(px_list, py_list)),
            dtype=torch.float32,
            device=self.device,
        ) if px_list else torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        fov_tensor = torch.tensor(
            np.stack(fov_list),
            dtype=torch.float32,
            device=self.device,
        ) if fov_list else None
        active_mask = torch.ones(len(px_list), dtype=torch.bool, device=self.device)

        meta = {
            'obs_mode': encoder_output.obs_mode,
            'obs_dim': encoder_output.obs_dim,
            'norm_consts': encoder_info.get('norm_consts', {}),
            'obs_fallback': bool(fallback),
            'positions': positions_tensor,
            'fov': fov_tensor,
            'active_mask': active_mask,
            'paper_obs_fallback': bool(fallback),
        }
        return encoder_output.features, meta

    def _extract_agent_positions(self, obs_dict: Dict[int, Dict[str, Any]]) -> Optional[torch.Tensor]:
        if not isinstance(obs_dict, dict):
            return None
        positions = []
        for agent_id in sorted(obs_dict.keys()):
            entry = obs_dict[agent_id]
            pos = entry.get('pos_xy')
            if pos is None:
                return None
            positions.append([float(pos[0]), float(pos[1])])
        if not positions:
            return None
        return torch.tensor(positions, dtype=torch.float32, device=self.device)

    def _compute_intrinsic_reward(
        self,
        obs_vecs: torch.Tensor,
        actions: torch.Tensor,
        mask_vec: torch.Tensor,
        adjacency: Optional[torch.Tensor],
        positions: torch.Tensor,
        fov: Optional[torch.Tensor],
        active_mask: torch.Tensor,
        fallback: bool,
    ) -> torch.Tensor:
        if not self.use_intrinsic or self.irdc is None:
            return torch.zeros(obs_vecs.shape[0], device=obs_vecs.device)
        intrinsic_adj = adjacency
        if self.intrinsic_use_gat:
            mode = self.intrinsic_neighbor_mode
            effective_mode = mode
            if mode == 'fov' and (fov is None or fallback):
                effective_mode = 'radius'
            if intrinsic_adj is None or mode != self.graph_neighbor_mode:
                intrinsic_adj = build_adjacency(
                    positions,
                    fov=fov if effective_mode == 'fov' else None,
                    mode=effective_mode,
                    radius=self.intrinsic_neighbor_radius,
                    symmetry=self.graph_symmetry,
                    self_loop=self.graph_self_loop,
                    inactive=active_mask,
                    topk_neighbors=self.graph_topk,
                    deterministic_topk_jitter=self.graph_topk_jitter,
                )
        r_int, h_out, _ = self.irdc(
            obs_vecs,
            actions,
            mask=mask_vec,
            h_in=self.h_irdc,
            batch_first=True,
            adj=intrinsic_adj,
        )
        if h_out is not None:
            self.h_irdc = h_out.detach()
        return r_int.squeeze(0) if r_int.dim() == 2 else r_int

    def _critic_key_for_policy(self, policy_key: str) -> str:
        if policy_key in self.policy_to_critic:
            return self.policy_to_critic[policy_key]
        if self.critic_map:
            return next(iter(self.critic_map.keys()))
        raise ValueError("未找到可用的 critic")

    def _ensure_policy_hidden(self, policy_key: str, batch_size: int) -> None:
        if not self.multi_policy_mode:
            return
        module = self.policy_map[policy_key]
        if getattr(module, 'use_rnn', False):
            hidden = self.per_policy_h_policy.get(policy_key)
            if hidden is None or hidden.shape[1] != batch_size:
                self.per_policy_h_policy[policy_key] = module.initial_state(batch_size, self._policy_state_device)
        else:
            self.per_policy_h_policy[policy_key] = None

    def _ensure_critic_hidden(self, policy_key: str, batch_size: int) -> None:
        if not self.multi_policy_mode:
            return
        critic_key = self._critic_key_for_policy(policy_key)
        critic = self.critic_map[critic_key]
        if getattr(critic, 'use_rnn', False):
            hidden = self.per_policy_h_critic.get(policy_key)
            if hidden is None or hidden.shape[1] != batch_size:
                self.per_policy_h_critic[policy_key] = critic.initial_state(batch_size, self._critic_state_device)
        else:
            self.per_policy_h_critic[policy_key] = None

    @staticmethod
    def _stack_policy_steps(
        steps: List[Dict[str, Any]],
        h0_policy: Optional[torch.Tensor],
        h0_critic: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if not steps:
            return {}

        batch: Dict[str, Any] = {}
        keys = {k for step in steps for k in step.keys() if k not in {'values_dict', 'rew_dict'}}
        for key in keys:
            tensors = [step[key] for step in steps if step.get(key) is not None]
            if not tensors:
                batch[key] = None
                continue
            batch[key] = torch.stack(tensors, dim=0)

        # 嵌套字典处理
        if steps[0].get('values_dict') is not None:
            batch['values_dict'] = {}
            for head in steps[0]['values_dict'].keys():
                tensors = [step['values_dict'][head] for step in steps]
                batch['values_dict'][head] = torch.stack(tensors, dim=0)

        if steps[0].get('rew_dict') is not None:
            batch['rew_dict'] = {}
            for head in steps[0]['rew_dict'].keys():
                tensors = [step['rew_dict'][head] for step in steps]
                batch['rew_dict'][head] = torch.stack(tensors, dim=0)

        # agent_id 保持 [Bk]
        agent_ids = steps[0].get('agent_id')
        if agent_ids is not None:
            batch['agent_id'] = torch.stack([agent_ids] * len(steps), dim=0)

        if h0_policy is not None:
            batch['h0_policy'] = h0_policy
        if h0_critic is not None:
            batch['h0_critic'] = h0_critic

        return batch

    def rollout_multi_policy_sync(
        self,
        rollout_length: int,
        store_batches: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        if not self.multi_policy_mode:
            raise RuntimeError("rollout_multi_policy_sync 只能在多策略模式下调用")
        if rollout_length <= 0:
            rollout_length = self.steps_per_agent
        if self.use_real_env:
            return self._rollout_multi_policy_real(rollout_length, store_batches)
        return self._rollout_multi_policy_stub(rollout_length, store_batches)

    def _rollout_multi_policy_real(
        self,
        rollout_length: int,
        store_batches: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        info_agg = {
            'completed_tasks': 0,
            'throughput': 0.0,
            'battery_min': 1.0,
            'deadline_violation_rate': 0.0,
        }
        info_agg.update({
            'avg_reward_per_agent': 0.0,
            'avg_cost_time_per_agent': 0.0,
            'avg_cost_batt_per_agent': 0.0,
            'cost_sum_per_agent': 0.0,
            'viol_time_rate': 0.0,
            'viol_batt_rate': 0.0,
        })
        completed_episodes = 0
        info_agg['r_int_sum'] = 0.0
        info_agg['r_mix_sum'] = 0.0
        info_agg['intrinsic_steps'] = 0
        info_agg['r_int_sum'] = 0.0
        info_agg['r_mix_sum'] = 0.0
        info_agg['intrinsic_steps'] = 0
        degree_stats: List[float] = []
        self_only_stats: List[float] = []
        samples_collected = 0
        reward_series: List[torch.Tensor] = []
        time_series: List[torch.Tensor] = []
        batt_series: List[torch.Tensor] = []
        mask_series: List[torch.Tensor] = []
        raw_reward_series: List[torch.Tensor] = []
        raw_time_series: List[torch.Tensor] = []
        raw_batt_series: List[torch.Tensor] = []

        # 重置隐藏状态
        for policy_key in self.policy_keys:
            group_size = len(self.policy_agent_ids.get(policy_key, [])) or self.num_agents
            policy_module = self.policy_map[policy_key]
            if getattr(policy_module, 'use_rnn', False):
                self.per_policy_h_policy[policy_key] = policy_module.initial_state(group_size, self._policy_state_device)
            else:
                self.per_policy_h_policy[policy_key] = None

            critic_key = self._critic_key_for_policy(policy_key)
            critic_module = self.critic_map[critic_key]
            if getattr(critic_module, 'use_rnn', False):
                self.per_policy_h_critic[policy_key] = critic_module.initial_state(group_size, self._critic_state_device)
            else:
                self.per_policy_h_critic[policy_key] = None

        step_cache: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.policy_keys}
        start_h_policy: Dict[str, Optional[torch.Tensor]] = {}
        start_h_critic: Dict[str, Optional[torch.Tensor]] = {}

        obs_dict, info_dict = self.backend_adapter.reset()
        episode_done = False
        step_count = 0
        total_steps = 0
        total_episode_steps = 0

        mask_vec = torch.ones(self.num_agents, dtype=torch.bool, device=self._policy_state_device)
        agent_ids_template = torch.arange(self.num_agents, device=self.device)

        for step_idx in range(rollout_length):
            if episode_done:
                obs_dict, info_dict = self.backend_adapter.reset()
                episode_done = False
                step_count = 0
                mask_vec.fill_(True)
                for policy_key in self.policy_keys:
                    group_size = len(self.policy_agent_ids.get(policy_key, [])) or self.num_agents
                    policy_module = self.policy_map[policy_key]
                    if getattr(policy_module, 'use_rnn', False):
                        self.per_policy_h_policy[policy_key] = policy_module.initial_state(group_size, self._policy_state_device)
                    else:
                        self.per_policy_h_policy[policy_key] = None
                    critic_key = self._critic_key_for_policy(policy_key)
                    critic_module = self.critic_map[critic_key]
                    if getattr(critic_module, 'use_rnn', False):
                        self.per_policy_h_critic[policy_key] = critic_module.initial_state(group_size, self._critic_state_device)
                    else:
                        self.per_policy_h_critic[policy_key] = None
            encoded_obs, encoder_meta = self._encode_observations(obs_dict)
            obs_vecs, positions_aligned, fov_aligned, active_mask = self._align_entities(
                encoded_obs,
                encoder_meta.get('positions'),
                encoder_meta.get('fov'),
                encoder_meta.get('active_mask'),
            )
            fallback_flag = bool(encoder_meta.get('paper_obs_fallback', False))

            adjacency = None
            if self.require_adj_step or self.graph_cache_mode in {'dense', 'indices'}:
                adjacency = self._compute_adjacency_matrix(
                    positions_aligned,
                    fov_aligned,
                    active_mask,
                    fallback_flag,
                )
                if adjacency is not None:
                    degree = adjacency.float().sum(dim=-1)
                    degree_stats.append(degree.mean().item())
                    self_only_stats.append((degree == 1).float().mean().item())

            buckets = self.policy_router.split_indices(agent_ids_template)

            actions_tensor = torch.zeros(self.num_agents, dtype=torch.long, device=self.device)
            log_prob_tensor = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            logits_tensor = torch.zeros(self.num_agents, self.act_dim, dtype=torch.float32, device=self.device)
            value_R_tensor = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            value_time_tensor = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            value_batt_tensor = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)

            for policy_key, indices in buckets.items():
                batch_size = indices.numel()
                self._ensure_policy_hidden(policy_key, batch_size)
                self._ensure_critic_hidden(policy_key, batch_size)
                if policy_key not in start_h_policy:
                    start_state = self.per_policy_h_policy[policy_key]
                    start_h_policy[policy_key] = start_state.detach().cpu() if start_state is not None else None
                if policy_key not in start_h_critic:
                    start_state = self.per_policy_h_critic[policy_key]
                    start_h_critic[policy_key] = start_state.detach().cpu() if start_state is not None else None

                sliced = BatchBuilder.slice_step(
                    {
                        "obs": obs_vecs,
                        "mask": mask_vec.float(),
                        "adj": adjacency,
                        "agent_id": agent_ids_template,
                    },
                    indices,
                )
                obs_sub = sliced["obs"]
                mask_sub = sliced["mask"]
                agent_ids_sub = sliced["agent_id"].long()
                adj_sub = sliced["adj"]
                if adj_sub is not None:
                    adj_input = adj_sub.unsqueeze(0)
                else:
                    adj_input = None

                policy_module = self.policy_map[policy_key]
                obs_input = obs_sub.unsqueeze(0)
                mask_input = mask_sub.unsqueeze(0) if mask_sub is not None else None
                dist, h_pol_out, policy_extra = policy_module(
                    obs_input,
                    agent_ids_sub,
                    h_in=self.per_policy_h_policy[policy_key],
                    mask=mask_input,
                    batch_first=False,
                    adj=adj_input,
                )
                actions_sub = dist.sample().squeeze(0)
                log_prob_sub = dist.log_prob(actions_sub).squeeze(0)
                logits_sub = policy_extra.get("logits", torch.zeros_like(dist.logits)).squeeze(0)
                if h_pol_out is not None:
                    self.per_policy_h_policy[policy_key] = h_pol_out.detach()

                critic_key = self._critic_key_for_policy(policy_key)
                critic_module = self.critic_map[critic_key]
                values_dict_sub, h_critic_out = critic_module(
                    obs_input,
                    agent_ids_sub,
                    h_in=self.per_policy_h_critic[policy_key],
                    mask=mask_input,
                    batch_first=False,
                    adj=adj_input,
                )
                if h_critic_out is not None:
                    self.per_policy_h_critic[policy_key] = h_critic_out.detach()
                value_R_sub = values_dict_sub['R'].squeeze(-1).squeeze(0)
                value_time_sub = values_dict_sub['time'].squeeze(-1).squeeze(0)
                value_batt_sub = values_dict_sub['batt'].squeeze(-1).squeeze(0)

                actions_tensor.index_copy_(0, indices, actions_sub)
                log_prob_tensor.index_copy_(0, indices, log_prob_sub)
                logits_tensor.index_copy_(0, indices, logits_sub)
                value_R_tensor.index_copy_(0, indices, value_R_sub)
                value_time_tensor.index_copy_(0, indices, value_time_sub)
                value_batt_tensor.index_copy_(0, indices, value_batt_sub)

                step_entry = {
                    'obs': obs_sub.detach().cpu(),
                    'act': actions_sub.detach().cpu(),
                    'logp': log_prob_sub.detach().cpu(),
                    'logits': logits_sub.detach().cpu(),
                    'mask': mask_sub.detach().cpu() if mask_sub is not None else torch.ones(batch_size, dtype=torch.float32),
                    'agent_id': agent_ids_sub.detach().cpu(),
                    'adj': adj_sub.detach().cpu() if adj_sub is not None else None,
                    'values_dict': {
                        'R': value_R_sub.detach().cpu(),
                        'time': value_time_sub.detach().cpu(),
                        'batt': value_batt_sub.detach().cpu(),
                    },
                }
                step_cache[policy_key].append(step_entry)

            action_dict = {int(i): int(actions_tensor[i].item()) for i in range(self.num_agents)}
            next_obs_dict, rew_head_dict, done, info_dict = self.backend_adapter.step(action_dict)
            raw_R_step = info_dict.get('raw_R')
            raw_time_step = info_dict.get('raw_time')
            raw_batt_step = info_dict.get('raw_batt')

            r_ext_tensor = torch.zeros(self.num_agents, dtype=torch.float32, device=self.device)
            r_time_tensor = torch.zeros_like(r_ext_tensor)
            r_batt_tensor = torch.zeros_like(r_ext_tensor)
            for agent_id in range(self.num_agents):
                r_ext_tensor[agent_id] = float(rew_head_dict['R'].get(agent_id, 0.0))
                r_time_tensor[agent_id] = float(rew_head_dict['time'].get(agent_id, 0.0))
                r_batt_tensor[agent_id] = float(rew_head_dict['batt'].get(agent_id, 0.0))

            mask_step = mask_vec.float()
            mask_series.append(mask_step.clone())
            reward_series.append(r_ext_tensor.clone())
            time_series.append(r_time_tensor.clone())
            batt_series.append(r_batt_tensor.clone())

            raw_reward_step_tensor = torch.full(
                (self.num_agents,),
                float('nan'),
                dtype=torch.float32,
                device=self.device,
            )
            raw_time_step_tensor = torch.full_like(raw_reward_step_tensor, float('nan'))
            raw_batt_step_tensor = torch.full_like(raw_reward_step_tensor, float('nan'))

            if isinstance(raw_R_step, (list, tuple)):
                raw_r = torch.as_tensor(raw_R_step, dtype=torch.float32, device=self.device)
                count_r = min(self.num_agents, raw_r.numel())
                if count_r > 0:
                    raw_reward_step_tensor[:count_r] = raw_r[:count_r]
            if isinstance(raw_time_step, (list, tuple)):
                raw_t = torch.as_tensor(raw_time_step, dtype=torch.float32, device=self.device)
                count_t = min(self.num_agents, raw_t.numel())
                if count_t > 0:
                    raw_time_step_tensor[:count_t] = raw_t[:count_t]
            if isinstance(raw_batt_step, (list, tuple)):
                raw_b = torch.as_tensor(raw_batt_step, dtype=torch.float32, device=self.device)
                count_b = min(self.num_agents, raw_b.numel())
                if count_b > 0:
                    raw_batt_step_tensor[:count_b] = raw_b[:count_b]

            r_int_tensor = self._compute_intrinsic_reward(
                obs_vecs,
                actions_tensor,
                mask_vec.float(),
                adjacency,
                positions_aligned,
                fov_aligned,
                active_mask,
                fallback_flag,
            )
            raw_reward_store = torch.where(
                torch.isfinite(raw_reward_step_tensor),
                raw_reward_step_tensor,
                r_ext_tensor,
            )
            raw_time_store = torch.where(
                torch.isfinite(raw_time_step_tensor),
                raw_time_step_tensor,
                r_time_tensor,
            )
            raw_batt_store = torch.where(
                torch.isfinite(raw_batt_step_tensor),
                raw_batt_step_tensor,
                r_batt_tensor,
            )
            r_mix_tensor = raw_reward_store + self.intrinsic_epsilon * r_int_tensor

            raw_reward_series.append(raw_reward_step_tensor)
            raw_time_series.append(raw_time_step_tensor)
            raw_batt_series.append(raw_batt_step_tensor)

            if self.use_intrinsic:
                info_agg['r_int_sum'] += float(r_int_tensor.sum().item())
                info_agg['r_mix_sum'] += float(r_mix_tensor.sum().item())
                info_agg['intrinsic_steps'] += int(r_int_tensor.numel())

            done_tensor = torch.full((self.num_agents,), bool(done), dtype=torch.bool, device=self.device)
            for policy_key, indices in buckets.items():
                step_entry = step_cache[policy_key][-1]
                step_entry['rew_dict'] = {
                    'R': raw_reward_store.index_select(0, indices).detach().cpu(),
                    'time': raw_time_store.index_select(0, indices).detach().cpu(),
                    'batt': raw_batt_store.index_select(0, indices).detach().cpu(),
                    'int': r_int_tensor.index_select(0, indices).detach().cpu(),
                    'mix': r_mix_tensor.index_select(0, indices).detach().cpu(),
                }
                step_entry['done'] = done_tensor.index_select(0, indices).detach().cpu()

            samples_collected += self.num_agents
            obs_dict = next_obs_dict
            episode_done = done
            step_count += 1
            total_steps += 1

            if done:
                mask_vec.zero_()
            else:
                mask_vec.fill_(True)

            info_agg['completed_tasks'] += info_dict.get('completed_tasks', 0)
            info_agg['throughput'] = info_dict.get('throughput', 0.0)
            info_agg['battery_min'] = min(info_agg['battery_min'], info_dict.get('battery_min', 1.0))
            info_agg['deadline_violation_rate'] = info_dict.get('deadline_violation_rate', 0.0)
            info_agg['obs_mode'] = encoder_meta.get('obs_mode', self.obs_mode)
            info_agg['obs_dim'] = encoder_meta.get('obs_dim', self.obs_encoder.out_dim)
            info_agg['uses_paper_obs'] = int(self.use_paper_obs and not self._fallback_obs_mode)
            info_agg['obs_fallback'] = int(encoder_meta.get('obs_fallback', False))
            info_agg['obs_encoder_version'] = self.encoder_version

            if done or step_count >= self.max_steps_per_ep:
                total_episode_steps += step_count
                episode_done = True
                completed_episodes += 1

        batch_outputs: Dict[str, Dict[str, Any]] = {}

        for policy_key in self.policy_keys:
            steps = step_cache.get(policy_key, [])
            batch = self._stack_policy_steps(
                steps,
                start_h_policy.get(policy_key),
                start_h_critic.get(policy_key),
            )
            if not batch:
                continue

            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.detach().cpu()
                elif isinstance(value, dict):
                    batch[key] = {
                        sub_k: sub_v.detach().cpu() if isinstance(sub_v, torch.Tensor) else sub_v
                        for sub_k, sub_v in value.items()
                    }
            batch_outputs[policy_key] = batch
            if store_batches and self.buffer_store is not None:
                self.buffer_store.add_msg(policy_key, batch)
                buffer = self.buffer_store.get(policy_key)
                buffer.set_obs_metadata({
                    'obs_mode': info_agg.get('obs_mode', self.obs_mode),
                    'encoder_version': self.encoder_version,
                    'obs_dim': info_agg.get('obs_dim', self.obs_encoder.out_dim),
                })

        if reward_series:
            reward_tensor = torch.stack(reward_series)
            time_tensor = torch.stack(time_series)
            batt_tensor = torch.stack(batt_series)
            mask_tensor = torch.stack(mask_series)
            raw_reward_tensor = torch.stack(raw_reward_series)
            raw_time_tensor = torch.stack(raw_time_series)
            raw_batt_tensor = torch.stack(raw_batt_series)
        else:
            empty_shape = (0, self.num_agents)
            reward_tensor = torch.empty(empty_shape, dtype=torch.float32, device=self.device)
            time_tensor = torch.empty_like(reward_tensor)
            batt_tensor = torch.empty_like(reward_tensor)
            mask_tensor = torch.empty_like(reward_tensor)
            raw_reward_tensor = torch.empty_like(reward_tensor)
            raw_time_tensor = torch.empty_like(reward_tensor)
            raw_batt_tensor = torch.empty_like(reward_tensor)

        reward_stats = self._resolve_reward_source(
            reward_tensor,
            time_tensor,
            batt_tensor,
            raw_reward_tensor,
            raw_time_tensor,
            raw_batt_tensor,
            mask_tensor,
        )
        reward_source_tensor = reward_stats['reward_source_tensor']
        time_source_tensor = reward_stats['time_source_tensor']
        batt_source_tensor = reward_stats['batt_source_tensor']
        reward_source_used = reward_stats['reward_source_used']
        raw_present_steps = reward_stats['raw_present_steps']
        raw_finite_ratio = reward_stats['raw_finite_ratio']
        raw_available = reward_stats['raw_available']

        mask_sum_total = mask_tensor.sum()
        reward_sum = (reward_source_tensor * mask_tensor).sum()
        time_sum = (time_source_tensor * mask_tensor).sum()
        batt_sum = (batt_source_tensor * mask_tensor).sum()
        viol_time_sum = ((time_source_tensor < 0.0).float() * mask_tensor).sum()
        viol_batt_sum = ((batt_source_tensor < 0.0).float() * mask_tensor).sum()
        raw_reward_sum = reward_stats['raw_reward_sum']

        info_agg['debug_num_agents'] = self.num_agents
        info_agg['debug_rollout_length'] = rollout_length
        info_agg['debug_samples_collected'] = samples_collected
        info_agg['debug_rew_shape_T'] = reward_tensor.shape[0]
        info_agg['debug_rew_shape_B'] = reward_tensor.shape[1] if reward_tensor.dim() > 1 else self.num_agents

        denom = mask_sum_total.clamp_min(1e-6)
        if mask_sum_total.item() > 0:
            info_agg['avg_reward_per_agent'] = float((reward_sum / denom).item())
            info_agg['avg_cost_time_per_agent'] = float((time_sum / denom).item())
            info_agg['avg_cost_batt_per_agent'] = float((batt_sum / denom).item())
            info_agg['cost_sum_per_agent'] = float(((time_sum + batt_sum) / denom).item())
            info_agg['viol_time_rate'] = float((viol_time_sum / denom).item())
            info_agg['viol_batt_rate'] = float((viol_batt_sum / denom).item())

        info_agg['total_reward_sum'] = float(reward_sum.item())
        info_agg['total_cost_time_sum'] = float(time_sum.item())
        info_agg['total_cost_batt_sum'] = float(batt_sum.item())
        info_agg['total_constraint_cost_sum'] = float((time_sum + batt_sum).item())
        info_agg['total_agent_steps'] = float(mask_sum_total.item())
        if completed_episodes == 0 and step_count > 0:
            total_episode_steps += step_count
        info_agg['episode_steps_total'] = float(total_episode_steps)
        avg_episode_len = total_episode_steps / max(1, completed_episodes if completed_episodes > 0 else 1)
        info_agg['episode_length_avg'] = float(avg_episode_len)
        info_agg['episode_length_last'] = float(step_count)

        info_agg['debug_raw_present_steps'] = raw_present_steps
        info_agg['debug_raw_finite_ratio'] = raw_finite_ratio
        info_agg['debug_raw_available'] = float(raw_available)
        info_agg['debug_raw_reward_sum'] = raw_reward_sum
        info_agg['reward_source_used'] = reward_source_used

        if degree_stats:
            info_agg['gat/avg_degree'] = float(np.mean(degree_stats))
            info_agg['gat/pct_self_only'] = float(np.mean(self_only_stats))
        info_agg['gat/neighbor_mode'] = self.graph_neighbor_mode
        info_agg['samples_collected'] = samples_collected
        return info_agg, batch_outputs

    def _rollout_multi_policy_stub(
        self,
        rollout_length: int,
        store_batches: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        info_agg = {
            'completed_tasks': 0,
            'throughput': 0.0,
            'battery_min': 1.0,
            'deadline_violation_rate': 0.0,
        }
        samples_collected = 0
        step_cache: Dict[str, List[Dict[str, Any]]] = {k: [] for k in self.policy_keys}
        start_h_policy: Dict[str, Optional[torch.Tensor]] = {}
        start_h_critic: Dict[str, Optional[torch.Tensor]] = {}

        mask_vec = torch.ones(self.num_agents, dtype=torch.bool, device=self._policy_state_device)
        agent_ids_tensor = torch.arange(self.num_agents, device=self.device)
        total_episode_steps = 0
        total_reward_sum = 0.0
        total_cost_time_sum = 0.0
        total_cost_batt_sum = 0.0
        total_constraint_cost_sum = 0.0
        total_agent_steps = 0.0

        for policy_key in self.policy_keys:
            group_size = len(self.policy_agent_ids.get(policy_key, [])) or self.num_agents
            policy_module = self.policy_map[policy_key]
            if getattr(policy_module, 'use_rnn', False):
                self.per_policy_h_policy[policy_key] = policy_module.initial_state(group_size, self._policy_state_device)
            else:
                self.per_policy_h_policy[policy_key] = None
            critic_key = self._critic_key_for_policy(policy_key)
            critic_module = self.critic_map[critic_key]
            if getattr(critic_module, 'use_rnn', False):
                self.per_policy_h_critic[policy_key] = critic_module.initial_state(group_size, self._critic_state_device)
            else:
                self.per_policy_h_critic[policy_key] = None

        for step_idx in range(rollout_length):
            obs_vecs = torch.randn(self.num_agents, self.obs_dim, device=self.device)
            adjacency = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
            buckets = self.policy_router.split_indices(agent_ids_tensor)

            for policy_key, indices in buckets.items():
                batch_size = indices.numel()
                self._ensure_policy_hidden(policy_key, batch_size)
                self._ensure_critic_hidden(policy_key, batch_size)
                if policy_key not in start_h_policy:
                    start_state = self.per_policy_h_policy[policy_key]
                    start_h_policy[policy_key] = start_state.detach().cpu() if start_state is not None else None
                if policy_key not in start_h_critic:
                    start_state = self.per_policy_h_critic[policy_key]
                    start_h_critic[policy_key] = start_state.detach().cpu() if start_state is not None else None

                sliced = BatchBuilder.slice_step(
                    {
                        "obs": obs_vecs,
                        "mask": mask_vec.float(),
                        "adj": adjacency,
                        "agent_id": agent_ids_tensor,
                    },
                    indices,
                )
                obs_sub = sliced["obs"]
                mask_sub = sliced["mask"]
                agent_ids_sub = sliced["agent_id"].long()
                adj_sub = sliced["adj"]
                if adj_sub is not None:
                    adj_input = adj_sub.unsqueeze(0)
                else:
                    adj_input = None

                policy_module = self.policy_map[policy_key]
                obs_input = obs_sub.unsqueeze(0)
                mask_input = mask_sub.unsqueeze(0)
                dist, h_pol_out, policy_extra = policy_module(
                    obs_input,
                    agent_ids_sub,
                    h_in=self.per_policy_h_policy[policy_key],
                    mask=mask_input,
                    batch_first=False,
                    adj=adj_input,
                )
                actions_sub = dist.sample().squeeze(0)
                log_prob_sub = dist.log_prob(actions_sub).squeeze(0)
                logits_sub = policy_extra.get("logits", torch.zeros_like(dist.logits)).squeeze(0)
                if h_pol_out is not None:
                    self.per_policy_h_policy[policy_key] = h_pol_out.detach()

                critic_key = self._critic_key_for_policy(policy_key)
                critic_module = self.critic_map[critic_key]
                values_dict_sub, h_critic_out = critic_module(
                    obs_input,
                    agent_ids_sub,
                    h_in=self.per_policy_h_critic[policy_key],
                    mask=mask_input,
                    batch_first=False,
                    adj=adj_input,
                )
                if h_critic_out is not None:
                    self.per_policy_h_critic[policy_key] = h_critic_out.detach()
                value_R_sub = values_dict_sub['R'].squeeze(-1).squeeze(0)
                value_time_sub = values_dict_sub['time'].squeeze(-1).squeeze(0)
                value_batt_sub = values_dict_sub['batt'].squeeze(-1).squeeze(0)

                rewards = torch.randn(batch_size, device=self.device) * 0.1
                step_entry = {
                    'obs': obs_sub.detach().cpu(),
                    'act': actions_sub.detach().cpu(),
                    'logp': log_prob_sub.detach().cpu(),
                    'logits': logits_sub.detach().cpu(),
                    'mask': mask_sub.detach().cpu(),
                    'adj': adj_sub.detach().cpu() if adj_sub is not None else None,
                    'agent_id': agent_ids_sub.detach().cpu(),
                    'values_dict': {
                        'R': value_R_sub.detach().cpu(),
                        'time': value_time_sub.detach().cpu(),
                        'batt': value_batt_sub.detach().cpu(),
                    },
                    'rew_dict': {
                        'R': rewards.detach().cpu(),
                        'time': rewards.detach().cpu(),
                        'batt': rewards.detach().cpu(),
                        'int': torch.zeros_like(rewards).cpu(),
                        'mix': rewards.detach().cpu(),
                    },
                    'done': torch.zeros(batch_size, dtype=torch.bool).cpu(),
                }
                step_cache[policy_key].append(step_entry)

            samples_collected += self.num_agents

        batch_outputs: Dict[str, Dict[str, Any]] = {}

        for policy_key in self.policy_keys:
            steps = step_cache.get(policy_key, [])
            batch = self._stack_policy_steps(
                steps,
                start_h_policy.get(policy_key),
                start_h_critic.get(policy_key),
            )
            if not batch:
                continue
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.detach().cpu()
                elif isinstance(value, dict):
                    batch[key] = {
                        sub_k: sub_v.detach().cpu() if isinstance(sub_v, torch.Tensor) else sub_v
                        for sub_k, sub_v in value.items()
                    }
            batch_outputs[policy_key] = batch
            if store_batches and self.buffer_store is not None:
                self.buffer_store.add_msg(policy_key, batch)
                buffer = self.buffer_store.get(policy_key)
                buffer.set_obs_metadata({
                    'obs_mode': self.obs_mode,
                    'encoder_version': self.encoder_version,
                    'obs_dim': self.obs_encoder.out_dim,
                })

        info_agg['samples_collected'] = samples_collected
        info_agg['gat/neighbor_mode'] = self.graph_neighbor_mode
        return info_agg, batch_outputs

    def load_state(
        self,
        policy_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        critic_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> None:
        """
        加载策略/评论器 state_dict（用于 Worker 广播同步）
        """
        with torch.no_grad():
            if policy_state:
                if self.multi_policy_mode:
                    for key, state in policy_state.items():
                        module = self.policy_map.get(key)
                        if module is None:
                            continue
                        prev_mode = module.training
                        module.eval()
                        module.load_state_dict(state)
                        module.train(prev_mode)
                elif self.policy is not None:
                    state = next(iter(policy_state.values()))
                    prev_mode = self.policy.training
                    self.policy.eval()
                    self.policy.load_state_dict(state)
                    self.policy.train(prev_mode)

            if critic_state:
                if self.multi_policy_mode:
                    for key, state in critic_state.items():
                        module = self.critic_map.get(key)
                        if module is None:
                            continue
                        prev_mode = module.training
                        module.eval()
                        module.load_state_dict(state)
                        module.train(prev_mode)
                elif self.critic is not None:
                    state = next(iter(critic_state.values()))
                    prev_mode = self.critic.training
                    self.critic.eval()
                    self.critic.load_state_dict(state)
                    self.critic.train(prev_mode)
    
    def collect(self) -> Tuple[RolloutDict, Dict[str, Any]]:
        """
        收集一轮 rollout 数据
        
        支持真实环境适配器或占位模式（stub）
        
        返回:
            rollout_dict: 包含 obs, act, logp, logits, rew_dict, done, agent_id, values_dict 的字典
            info_agg: 聚合的环境信息字典（用于日志）
        """
        if self.use_real_env:
            return self._collect_real_env()
        else:
            return self._collect_stub()
    
    def _collect_real_env(self) -> Tuple[RolloutDict, Dict[str, Any]]:
        """使用真实环境适配器收集数据"""
        if self.use_policy_rnn and self.h_policy is not None:
            self.h_policy.zero_()
            h0_policy = self.h_policy.detach().clone()
        else:
            h0_policy = None
        if self.use_rnn_critic and self.h_critic is not None:
            self.h_critic.zero_()
            h0_critic = self.h_critic.detach().clone()
        else:
            h0_critic = None
        if self.use_intrinsic and self.h_irdc is not None:
            self.h_irdc.zero_()

        obs_list: list[torch.Tensor] = []
        act_list: list[int] = []
        logp_list: list[float] = []
        logits_list: list[torch.Tensor] = []
        rew_R_list: list[float] = []
        rew_time_list: list[float] = []
        rew_batt_list: list[float] = []
        rew_int_list: list[float] = []
        rew_mix_list: list[float] = []
        done_list: list[bool] = []
        agent_id_list: list[int] = []
        values_R_list: list[float] = []
        values_time_list: list[float] = []
        values_batt_list: list[float] = []
        mask_list: list[float] = []
        norm_reward_list: list[float] = []
        norm_time_list: list[float] = []
        norm_batt_list: list[float] = []
        raw_reward_nan_list: list[float] = []
        raw_time_nan_list: list[float] = []
        raw_batt_nan_list: list[float] = []

        adj_list: list[torch.Tensor] = []
        positions_records: list[torch.Tensor] = []
        fov_records: list[Optional[torch.Tensor]] = []
        active_records: list[torch.Tensor] = []
        degree_stats: list[float] = []
        self_only_stats: list[float] = []

        info_agg = {
            'completed_tasks': 0,
            'throughput': 0.0,
            'battery_min': 1.0,
            'deadline_violation_rate': 0.0,
        }
        if self.use_intrinsic:
            info_agg['r_int_sum'] = 0.0
            info_agg['r_mix_sum'] = 0.0
            info_agg['intrinsic_steps'] = 0
        if self.use_intrinsic:
            info_agg['r_int_sum'] = 0.0
            info_agg['r_mix_sum'] = 0.0
            info_agg['intrinsic_steps'] = 0

        obs_dict, info_dict = self.backend_adapter.reset()
        completed_episodes = 0
        episode_done = False
        step_count = 0
        total_steps = 0
        total_episode_steps = 0
        mask_vec = torch.ones(self.num_agents, dtype=torch.bool, device=self._policy_state_device)
        agent_ids_tensor = torch.arange(self.num_agents, device=self.device)

        while total_steps < self.steps_per_agent * self.num_agents:
            if episode_done:
                obs_dict, info_dict = self.backend_adapter.reset()
                episode_done = False
                step_count = 0
                mask_vec.fill_(True)
                if self.use_policy_rnn and self.h_policy is not None:
                    self.h_policy.zero_()
                if self.use_rnn_critic and self.h_critic is not None:
                    self.h_critic.zero_()
                if self.use_intrinsic and self.h_irdc is not None:
                    self.h_irdc.zero_()

            encoded_obs, encoder_meta = self._encode_observations(obs_dict)
            obs_vecs, positions_aligned, fov_aligned, active_mask = self._align_entities(
                encoded_obs,
                encoder_meta.get('positions'),
                encoder_meta.get('fov'),
                encoder_meta.get('active_mask'),
            )
            fallback_flag = bool(encoder_meta.get('paper_obs_fallback', False))

            cache_requires_adj = self.graph_cache_mode in {'dense', 'indices'}
            need_adj_now = self.require_adj_step or cache_requires_adj
            adjacency = None
            if need_adj_now:
                adjacency = self._compute_adjacency_matrix(
                    positions_aligned,
                    fov_aligned,
                    active_mask,
                    fallback_flag,
                )

            if self.store_graph_meta:
                positions_records.append(positions_aligned.detach())
                active_records.append(active_mask.detach())
                fov_records.append(fov_aligned.detach() if fov_aligned is not None else None)

            if adjacency is not None:
                adj_list.append(adjacency.detach())
                degree = adjacency.float().sum(dim=-1)
                degree_stats.append(degree.mean().item())
                self_only_stats.append((degree == 1).float().mean().item())

            policy_adj = adjacency if self.policy_frontend_use_gat else None
            dist, h_pol_out, policy_extra = self.policy(
                obs_vecs,
                agent_ids_tensor,
                h_in=self.h_policy,
                mask=mask_vec,
                batch_first=True,
                adj=policy_adj,
            )
            logits_all = policy_extra.get('logits')
            actions_tensor = dist.sample()
            log_prob_tensor = dist.log_prob(actions_tensor)
            if self.use_policy_rnn and h_pol_out is not None:
                self.h_policy = h_pol_out.detach()

            critic_adj = adjacency if self.critic_frontend_use_gat else None
            with torch.no_grad():
                values_dict_all, h_crit_out = self.critic(
                    obs_vecs,
                    agent_ids_tensor,
                    h_in=self.h_critic,
                    mask=mask_vec,
                    batch_first=True,
                    adj=critic_adj,
                )
            if self.use_rnn_critic and h_crit_out is not None:
                self.h_critic = h_crit_out.detach()

            action_dict = {int(i): int(actions_tensor[i].item()) for i in range(self.num_agents)}
            next_obs_dict, rew_head_dict, done, info_dict = self.backend_adapter.step(action_dict)
            raw_R_step = info_dict.get('raw_R')
            raw_time_step = info_dict.get('raw_time')
            raw_batt_step = info_dict.get('raw_batt')

            r_int_tensor = self._compute_intrinsic_reward(
                obs_vecs,
                actions_tensor,
                mask_vec,
                adjacency,
                positions_aligned,
                fov_aligned,
                active_mask,
                fallback_flag,
            )

            for agent_id in range(self.num_agents):
                obs_list.append(obs_vecs[agent_id].clone())
                act_list.append(int(actions_tensor[agent_id].item()))
                logp_list.append(float(log_prob_tensor[agent_id].item()))
                logits_entry = logits_all[agent_id].detach() if logits_all is not None else torch.zeros(self.action_dim, device=self.device)
                logits_list.append(logits_entry)
                r_R_norm = rew_head_dict['R'].get(agent_id, 0.0)
                r_time_norm = rew_head_dict['time'].get(agent_id, 0.0)
                r_batt_norm = rew_head_dict['batt'].get(agent_id, 0.0)
                norm_reward_list.append(float(r_R_norm))
                norm_time_list.append(float(r_time_norm))
                norm_batt_list.append(float(r_batt_norm))

                if isinstance(raw_R_step, (list, tuple)) and len(raw_R_step) > agent_id:
                    raw_r_val = float(raw_R_step[agent_id])
                else:
                    raw_r_val = float('nan')
                if isinstance(raw_time_step, (list, tuple)) and len(raw_time_step) > agent_id:
                    raw_t_val = float(raw_time_step[agent_id])
                else:
                    raw_t_val = float('nan')
                if isinstance(raw_batt_step, (list, tuple)) and len(raw_batt_step) > agent_id:
                    raw_b_val = float(raw_batt_step[agent_id])
                else:
                    raw_b_val = float('nan')

                raw_reward_nan_list.append(raw_r_val)
                raw_time_nan_list.append(raw_t_val)
                raw_batt_nan_list.append(raw_b_val)

                reward_store = raw_r_val if math.isfinite(raw_r_val) else float(r_R_norm)
                time_store = raw_t_val if math.isfinite(raw_t_val) else float(r_time_norm)
                batt_store = raw_b_val if math.isfinite(raw_b_val) else float(r_batt_norm)

                rew_R_list.append(reward_store)
                rew_time_list.append(time_store)
                rew_batt_list.append(batt_store)
                r_int_value = float(r_int_tensor[agent_id].item()) if self.use_intrinsic else 0.0
                rew_int_list.append(r_int_value)
                rew_mix_list.append(reward_store + self.intrinsic_epsilon * r_int_value)
                done_list.append(done)
                agent_id_list.append(agent_id)
                mask_list.append(float(mask_vec[agent_id].item()))
                values_R_list.append(values_dict_all['R'][agent_id].squeeze().item())
                values_time_list.append(values_dict_all['time'][agent_id].squeeze().item())
                values_batt_list.append(values_dict_all['batt'][agent_id].squeeze().item())
                if self.use_intrinsic:
                    info_agg['r_int_sum'] += r_int_value
                    info_agg['r_mix_sum'] += rew_mix_list[-1]
                    info_agg['intrinsic_steps'] += 1

            obs_dict = next_obs_dict
            episode_done = done
            step_count += 1
            total_steps += self.num_agents

            if done:
                mask_vec.zero_()
                if self.use_intrinsic and self.h_irdc is not None:
                    self.h_irdc.zero_()
            else:
                mask_vec.fill_(True)

            info_agg['completed_tasks'] += info_dict.get('completed_tasks', 0)
            info_agg['throughput'] = info_dict.get('throughput', 0.0)
            info_agg['battery_min'] = min(info_agg['battery_min'], info_dict.get('battery_min', 1.0))
            info_agg['deadline_violation_rate'] = info_dict.get('deadline_violation_rate', 0.0)
            info_agg['obs_mode'] = encoder_meta.get('obs_mode', self.obs_mode)
            info_agg['obs_dim'] = encoder_meta.get('obs_dim', self.obs_encoder.out_dim)
            info_agg['uses_paper_obs'] = int(self.use_paper_obs and not self._fallback_obs_mode)
            info_agg['obs_fallback'] = int(encoder_meta.get('obs_fallback', False))
            info_agg['obs_encoder_version'] = self.encoder_version

            if step_count >= self.max_steps_per_ep:
                total_episode_steps += step_count
                episode_done = True
                completed_episodes += 1

        obs = torch.stack(obs_list)
        act = torch.tensor(act_list, dtype=torch.long, device=self.device)
        logp = torch.tensor(logp_list, device=self.device)
        logits = torch.stack(logits_list)
        done = torch.tensor(done_list, dtype=torch.bool, device=self.device)
        agent_id = torch.tensor(agent_id_list, dtype=torch.long, device=self.device)
        mask = torch.tensor(mask_list, dtype=torch.float32, device=self.device)

        if norm_reward_list:
            reward_norm_tensor = torch.tensor(norm_reward_list, dtype=torch.float32, device=self.device).view(-1, self.num_agents)
        else:
            reward_norm_tensor = torch.empty((0, self.num_agents), dtype=torch.float32, device=self.device)
        if norm_time_list:
            time_norm_tensor = torch.tensor(norm_time_list, dtype=torch.float32, device=self.device).view(-1, self.num_agents)
        else:
            time_norm_tensor = torch.empty_like(reward_norm_tensor)
        if norm_batt_list:
            batt_norm_tensor = torch.tensor(norm_batt_list, dtype=torch.float32, device=self.device).view(-1, self.num_agents)
        else:
            batt_norm_tensor = torch.empty_like(reward_norm_tensor)

        if raw_reward_nan_list:
            raw_reward_tensor = torch.tensor(raw_reward_nan_list, dtype=torch.float32, device=self.device).view(-1, self.num_agents)
        else:
            raw_reward_tensor = torch.empty_like(reward_norm_tensor)
        if raw_time_nan_list:
            raw_time_tensor = torch.tensor(raw_time_nan_list, dtype=torch.float32, device=self.device).view(-1, self.num_agents)
        else:
            raw_time_tensor = torch.empty_like(reward_norm_tensor)
        if raw_batt_nan_list:
            raw_batt_tensor = torch.tensor(raw_batt_nan_list, dtype=torch.float32, device=self.device).view(-1, self.num_agents)
        else:
            raw_batt_tensor = torch.empty_like(reward_norm_tensor)

        if mask.numel() > 0:
            mask_tensor = mask.view(-1, self.num_agents)
        else:
            mask_tensor = torch.empty_like(reward_norm_tensor)

        reward_stats = self._resolve_reward_source(
            reward_norm_tensor,
            time_norm_tensor,
            batt_norm_tensor,
            raw_reward_tensor,
            raw_time_tensor,
            raw_batt_tensor,
            mask_tensor,
        )
        reward_source_tensor = reward_stats['reward_source_tensor']
        time_source_tensor = reward_stats['time_source_tensor']
        batt_source_tensor = reward_stats['batt_source_tensor']
        reward_source_used = reward_stats['reward_source_used']
        raw_present_steps = reward_stats['raw_present_steps']
        raw_finite_ratio = reward_stats['raw_finite_ratio']
        raw_available = reward_stats['raw_available']
        raw_reward_sum = reward_stats['raw_reward_sum']

        mask_sum_total = mask_tensor.sum()
        reward_sum = (reward_source_tensor * mask_tensor).sum()
        time_sum = (time_source_tensor * mask_tensor).sum()
        batt_sum = (batt_source_tensor * mask_tensor).sum()
        viol_time_sum = ((time_source_tensor < 0.0).float() * mask_tensor).sum()
        viol_batt_sum = ((batt_source_tensor < 0.0).float() * mask_tensor).sum()

        info_agg['debug_num_agents'] = self.num_agents
        info_agg['debug_rollout_length'] = self.steps_per_agent
        info_agg['debug_samples_collected'] = len(mask_list)
        info_agg['debug_rew_shape_T'] = reward_norm_tensor.shape[0]
        info_agg['debug_rew_shape_B'] = reward_norm_tensor.shape[1] if reward_norm_tensor.dim() > 1 else self.num_agents
        info_agg['samples_collected'] = len(mask_list)

        denom = mask_sum_total.clamp_min(1e-6)
        if mask_sum_total.item() > 0:
            info_agg['avg_reward_per_agent'] = float((reward_sum / denom).item())
            info_agg['avg_cost_time_per_agent'] = float((time_sum / denom).item())
            info_agg['avg_cost_batt_per_agent'] = float((batt_sum / denom).item())
            info_agg['cost_sum_per_agent'] = float(((time_sum + batt_sum) / denom).item())
            info_agg['viol_time_rate'] = float((viol_time_sum / denom).item())
            info_agg['viol_batt_rate'] = float((viol_batt_sum / denom).item())

        total_reward_sum = float(reward_sum.item())
        total_cost_time_sum = float(time_sum.item())
        total_cost_batt_sum = float(batt_sum.item())
        total_agent_steps = float(mask_sum_total.item())


        info_agg['debug_raw_present_steps'] = raw_present_steps
        info_agg['debug_raw_finite_ratio'] = raw_finite_ratio
        info_agg['debug_raw_available'] = float(raw_available)
        info_agg['debug_raw_reward_sum'] = raw_reward_sum
        info_agg['reward_source_used'] = reward_source_used

        rew_dict = {
            'R': torch.tensor(rew_R_list, device=self.device),
            'time': torch.tensor(rew_time_list, device=self.device),
            'batt': torch.tensor(rew_batt_list, device=self.device),
            'int': torch.tensor(rew_int_list, device=self.device) if rew_int_list else torch.zeros_like(obs[:, 0]),
            'mix': torch.tensor(rew_mix_list, device=self.device) if rew_mix_list else torch.tensor(rew_R_list, device=self.device),
        }
        values_dict = {
            'R': torch.tensor(values_R_list, device=self.device),
            'time': torch.tensor(values_time_list, device=self.device),
            'batt': torch.tensor(values_batt_list, device=self.device),
        }

        if completed_episodes == 0 and step_count > 0:
            total_episode_steps += step_count
        info_agg['episodes'] = max(1, completed_episodes)
        info_agg['episode_steps_total'] = float(total_episode_steps)
        avg_episode_len = total_episode_steps / max(1, completed_episodes if completed_episodes > 0 else 1)
        info_agg['episode_length_avg'] = float(avg_episode_len)
        info_agg['episode_length_last'] = float(step_count)
        info_agg['total_reward_sum'] = total_reward_sum
        info_agg['total_cost_time_sum'] = total_cost_time_sum
        info_agg['total_cost_batt_sum'] = total_cost_batt_sum
        info_agg['total_constraint_cost_sum'] = float((time_sum + batt_sum).item())
        info_agg['total_agent_steps'] = total_agent_steps

        rollout_dict: RolloutDict = {
            'obs': obs,
            'act': act,
            'logp': logp,
            'logits': logits,
            'rew_dict': rew_dict,
            'done': done,
            'agent_id': agent_id,
            'values_dict': values_dict,
            'mask': mask,
            'adj_cache_mode': self.graph_cache_mode,
            'policy_key': self.policy_key,
            'num_agents': self.num_agents,
            'time_steps': self.steps_per_agent,
            'h0_policy': h0_policy,
            'h0_critic': h0_critic,
        }

        if adj_list:
            rollout_dict['adj'] = torch.stack(adj_list)

        if self.store_graph_meta and positions_records:
            rollout_dict['positions'] = torch.stack(positions_records)
            rollout_dict['active_mask'] = torch.stack(active_records)
            if any(f is not None for f in fov_records):
                template = next(f for f in fov_records if f is not None)
                stacked_fov = torch.stack([f if f is not None else torch.zeros_like(template) for f in fov_records])
                rollout_dict['fov'] = stacked_fov
                rollout_dict['fov_missing_mask'] = torch.tensor(
                    [f is None for f in fov_records], dtype=torch.bool, device=self.device
                )
            else:
                rollout_dict['fov'] = None
                rollout_dict['fov_missing_mask'] = None

        if degree_stats:
            info_agg['gat/avg_degree'] = float(sum(degree_stats) / len(degree_stats))
            info_agg['gat/pct_self_only'] = float(sum(self_only_stats) / len(self_only_stats))
        info_agg['gat/neighbor_mode'] = self.graph_neighbor_mode

        if self.use_policy_rnn:
            self.h_policy = self.policy.initial_state(self.num_agents, self._policy_state_device)
        if self.use_rnn_critic:
            self.h_critic = self.critic.initial_state(self.num_agents, self._critic_state_device)

        if self.use_intrinsic and self.h_irdc is not None:
            self.h_irdc.zero_()

        rollout_dict['num_agents'] = self.num_agents

        return rollout_dict, info_agg

    def _collect_stub(self) -> Tuple[RolloutDict, Dict[str, Any]]:
        """使用占位模式收集数据（保留接口兼容性）"""
        if self.use_policy_rnn and self.h_policy is not None:
            self.h_policy.zero_()
            h0_policy = self.h_policy.detach().clone()
        else:
            h0_policy = None
        if self.use_rnn_critic and self.h_critic is not None:
            self.h_critic.zero_()
            h0_critic = self.h_critic.detach().clone()
        else:
            h0_critic = None

        obs_list: list[torch.Tensor] = []
        act_list: list[int] = []
        logp_list: list[float] = []
        logits_list: list[torch.Tensor] = []
        rew_R_list: list[float] = []
        rew_time_list: list[float] = []
        rew_batt_list: list[float] = []
        rew_int_list: list[float] = []
        rew_mix_list: list[float] = []
        done_list: list[bool] = []
        agent_id_list: list[int] = []
        values_R_list: list[float] = []
        values_time_list: list[float] = []
        values_batt_list: list[float] = []
        mask_list: list[float] = []
        adj_list: list[torch.Tensor] = []
        positions_records: list[torch.Tensor] = []
        active_records: list[torch.Tensor] = []

        info_agg = {
            'completed_tasks': 0,
            'throughput': 0.0,
            'battery_min': 1.0,
            'deadline_violation_rate': 0.0,
        }
        info_agg.update({
            'avg_reward_per_agent': 0.0,
            'avg_cost_time_per_agent': 0.0,
            'avg_cost_batt_per_agent': 0.0,
            'cost_sum_per_agent': 0.0,
            'viol_time_rate': 0.0,
            'viol_batt_rate': 0.0,
        })
        if self.use_intrinsic:
            info_agg['r_int_sum'] = 0.0
            info_agg['r_mix_sum'] = 0.0
            info_agg['intrinsic_steps'] = 0

        obs_vecs = torch.randn(self.num_agents, self.obs_dim, device=self.device)
        positions = torch.zeros(self.num_agents, 2, device=self.device)
        mask_vec = torch.ones(self.num_agents, dtype=torch.bool, device=self._policy_state_device)
        agent_ids_tensor = torch.arange(self.num_agents, device=self.device)
        completed_episodes = 1
        total_episode_steps = 0

        for step in range(self.steps_per_agent):
            adjacency = None
            if self.require_adj_step or self.graph_cache_mode in {'dense', 'indices'}:
                adjacency = self._compute_adjacency_matrix(
                    positions,
                    fov=None,
                    active_mask=torch.ones(self.num_agents, dtype=torch.bool, device=self.device),
                    fallback=True,
                )
                adj_list.append(adjacency.detach())

            policy_adj = adjacency if self.policy_frontend_use_gat else None
            dist, h_pol_out, policy_extra = self.policy(
                obs_vecs,
                agent_ids_tensor,
                h_in=self.h_policy,
                mask=mask_vec,
                batch_first=True,
                adj=policy_adj,
            )
            logits_all = policy_extra.get('logits')
            actions_tensor = dist.sample()
            log_prob_tensor = dist.log_prob(actions_tensor)
            if self.use_policy_rnn and h_pol_out is not None:
                self.h_policy = h_pol_out.detach()

            critic_adj = adjacency if self.critic_frontend_use_gat else None
            with torch.no_grad():
                values_dict_all, h_crit_out = self.critic(
                    obs_vecs,
                    agent_ids_tensor,
                    h_in=self.h_critic,
                    mask=mask_vec,
                    batch_first=True,
                    adj=critic_adj,
                )
            if self.use_rnn_critic and h_crit_out is not None:
                self.h_critic = h_crit_out.detach()

            r_int_tensor = self._compute_intrinsic_reward(
                obs_vecs,
                actions_tensor,
                mask_vec,
                adjacency,
                positions,
                None,
                torch.ones(self.num_agents, dtype=torch.bool, device=self.device),
                True,
            )

            action_dict = {int(i): int(actions_tensor[i].item()) for i in range(self.num_agents)}

            done_flag = (step == self.steps_per_agent - 1)
            for agent_id in range(self.num_agents):
                obs = obs_vecs[agent_id]
                act_val = float(action_dict[agent_id])
                eps_R = torch.randn(1, device=self.device).item() * 0.1
                eps_T = torch.randn(1, device=self.device).item() * 0.1
                eps_B = torch.randn(1, device=self.device).item() * 0.1

                r_ext = 0.5 * obs.mean().item() + 0.10 * (act_val - (self.act_dim - 1) / 2.0) + eps_R
                r_time = -0.3 * obs[:4].sum().item() + 0.20 * ((self.act_dim - 1) - act_val) + eps_T
                r_batt = 0.3 * obs[4:8].sum().item() - 0.20 * act_val + eps_B

                obs_list.append(obs.clone())
                act_list.append(action_dict[agent_id])
                logp_list.append(float(log_prob_tensor[agent_id].item()))
                logits_list.append(logits_all[agent_id].detach() if logits_all is not None else torch.zeros(self.action_dim, device=self.device))
                rew_R_list.append(r_ext)
                rew_time_list.append(r_time)
                rew_batt_list.append(r_batt)
                r_int_value = float(r_int_tensor[agent_id].item()) if self.use_intrinsic else 0.0
                rew_int_list.append(r_int_value)
                rew_mix_list.append(r_ext + self.intrinsic_epsilon * r_int_value)
                if self.use_intrinsic:
                    info_agg['r_int_sum'] += r_int_value
                    info_agg['r_mix_sum'] += rew_mix_list[-1]
                    info_agg['intrinsic_steps'] += 1
                done_list.append(done_flag)
                agent_id_list.append(agent_id)
                mask_list.append(float(mask_vec[agent_id].item()))
                values_R_list.append(values_dict_all['R'][agent_id].squeeze().item())
                values_time_list.append(values_dict_all['time'][agent_id].squeeze().item())
                values_batt_list.append(values_dict_all['batt'][agent_id].squeeze().item())

            positions_records.append(positions.detach().clone())
            active_records.append(torch.ones(self.num_agents, dtype=torch.bool, device=self.device))

            if self.use_intrinsic:
                mask_vec.fill_(True)
                if done_flag and self.h_irdc is not None:
                    self.h_irdc.zero_()
            else:
                if done_flag:
                    mask_vec.zero_()
                else:
                    mask_vec.fill_(True)

            obs_vecs = obs_vecs + 0.1 * torch.randn_like(obs_vecs)
            positions = positions + 0.2 * torch.randn_like(positions)

        obs = torch.stack(obs_list)
        act = torch.tensor(act_list, dtype=torch.long, device=self.device)
        logp = torch.tensor(logp_list, device=self.device)
        logits = torch.stack(logits_list)
        done = torch.tensor(done_list, dtype=torch.bool, device=self.device)
        agent_id = torch.tensor(agent_id_list, dtype=torch.long, device=self.device)
        mask = torch.tensor(mask_list, dtype=torch.float32, device=self.device)

        rew_dict = {
            'R': torch.tensor(rew_R_list, device=self.device),
            'time': torch.tensor(rew_time_list, device=self.device),
            'batt': torch.tensor(rew_batt_list, device=self.device),
            'int': torch.tensor(rew_int_list, device=self.device) if rew_int_list else torch.zeros_like(obs[:, 0]),
            'mix': torch.tensor(rew_mix_list, device=self.device) if rew_mix_list else torch.tensor(rew_R_list, device=self.device),
        }
        values_dict = {
            'R': torch.tensor(values_R_list, device=self.device),
            'time': torch.tensor(values_time_list, device=self.device),
            'batt': torch.tensor(values_batt_list, device=self.device),
        }

        reward_norm_tensor = rew_dict['R'].view(-1, self.num_agents).float() if rew_dict['R'].numel() > 0 else torch.empty((0, self.num_agents), dtype=torch.float32, device=self.device)
        time_norm_tensor = rew_dict['time'].view(-1, self.num_agents).float() if rew_dict['time'].numel() > 0 else torch.empty_like(reward_norm_tensor)
        batt_norm_tensor = rew_dict['batt'].view(-1, self.num_agents).float() if rew_dict['batt'].numel() > 0 else torch.empty_like(reward_norm_tensor)
        if mask.numel() > 0:
            mask_tensor = mask.view(-1, self.num_agents).float()
        else:
            mask_tensor = torch.empty_like(reward_norm_tensor)

        raw_reward_tensor = torch.full_like(reward_norm_tensor, float('nan'))
        raw_time_tensor = torch.full_like(time_norm_tensor, float('nan'))
        raw_batt_tensor = torch.full_like(batt_norm_tensor, float('nan'))

        reward_stats = self._resolve_reward_source(
            reward_norm_tensor,
            time_norm_tensor,
            batt_norm_tensor,
            raw_reward_tensor,
            raw_time_tensor,
            raw_batt_tensor,
            mask_tensor,
        )
        reward_source_tensor = reward_stats['reward_source_tensor']
        time_source_tensor = reward_stats['time_source_tensor']
        batt_source_tensor = reward_stats['batt_source_tensor']
        reward_source_used = reward_stats['reward_source_used']
        raw_present_steps = reward_stats['raw_present_steps']
        raw_finite_ratio = reward_stats['raw_finite_ratio']
        raw_available = reward_stats['raw_available']
        raw_reward_sum = reward_stats['raw_reward_sum']

        mask_sum_total = mask_tensor.sum()
        reward_sum = (reward_source_tensor * mask_tensor).sum()
        time_sum = (time_source_tensor * mask_tensor).sum()
        batt_sum = (batt_source_tensor * mask_tensor).sum()
        viol_time_sum = ((time_source_tensor < 0.0).float() * mask_tensor).sum()
        viol_batt_sum = ((batt_source_tensor < 0.0).float() * mask_tensor).sum()

        info_agg['debug_num_agents'] = self.num_agents
        info_agg['debug_rollout_length'] = self.steps_per_agent
        info_agg['debug_samples_collected'] = len(mask_list)
        info_agg['debug_rew_shape_T'] = reward_norm_tensor.shape[0]
        info_agg['debug_rew_shape_B'] = reward_norm_tensor.shape[1] if reward_norm_tensor.dim() > 1 else self.num_agents
        info_agg['samples_collected'] = len(mask_list)

        denom = mask_sum_total.clamp_min(1e-6)
        if mask_sum_total.item() > 0:
            info_agg['avg_reward_per_agent'] = float((reward_sum / denom).item())
            info_agg['avg_cost_time_per_agent'] = float((time_sum / denom).item())
            info_agg['avg_cost_batt_per_agent'] = float((batt_sum / denom).item())
            info_agg['cost_sum_per_agent'] = float(((time_sum + batt_sum) / denom).item())
            info_agg['viol_time_rate'] = float((viol_time_sum / denom).item())
            info_agg['viol_batt_rate'] = float((viol_batt_sum / denom).item())

        total_reward_sum = float(reward_sum.item())
        total_cost_time_sum = float(time_sum.item())
        total_cost_batt_sum = float(batt_sum.item())
        total_agent_steps = float(mask_sum_total.item())

        info_agg['total_reward_sum'] = total_reward_sum
        info_agg['total_cost_time_sum'] = total_cost_time_sum
        info_agg['total_cost_batt_sum'] = total_cost_batt_sum
        info_agg['total_constraint_cost_sum'] = float((time_sum + batt_sum).item())
        info_agg['total_agent_steps'] = total_agent_steps

        info_agg['debug_raw_present_steps'] = raw_present_steps
        info_agg['debug_raw_finite_ratio'] = raw_finite_ratio
        info_agg['debug_raw_available'] = float(raw_available)
        info_agg['debug_raw_reward_sum'] = raw_reward_sum
        info_agg['reward_source_used'] = reward_source_used

        rollout_dict: RolloutDict = {
            'obs': obs,
            'act': act,
            'logp': logp,
            'logits': logits,
            'rew_dict': rew_dict,
            'done': done,
            'agent_id': agent_id,
            'values_dict': values_dict,
            'mask': mask,
            'adj_cache_mode': self.graph_cache_mode,
            'policy_key': self.policy_key,
            'num_agents': self.num_agents,
            'time_steps': self.steps_per_agent,
            'h0_policy': h0_policy,
            'h0_critic': h0_critic,
        }
        if adj_list:
            rollout_dict['adj'] = torch.stack(adj_list)
        if positions_records:
            rollout_dict['positions'] = torch.stack(positions_records)
            rollout_dict['active_mask'] = torch.stack(active_records)
            rollout_dict['fov'] = None
            rollout_dict['fov_missing_mask'] = None

        info_agg['gat/neighbor_mode'] = self.graph_neighbor_mode
        if adj_list:
            degrees = torch.stack([a.float().sum(dim=-1) for a in adj_list])
            info_agg['gat/avg_degree'] = float(degrees.mean().item())
            info_agg['gat/pct_self_only'] = float((degrees == 1).float().mean().item())

        if self.use_policy_rnn:
            self.h_policy = self.policy.initial_state(self.num_agents, self._policy_state_device)
        if self.use_rnn_critic:
            self.h_critic = self.critic.initial_state(self.num_agents, self._critic_state_device)
        if self.use_intrinsic and self.h_irdc is not None:
            self.h_irdc.zero_()

        rollout_dict['num_agents'] = self.num_agents
        info_agg['episodes'] = max(1, completed_episodes)

        return rollout_dict, info_agg
