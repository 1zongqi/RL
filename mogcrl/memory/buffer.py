"""
On-policy 回放缓冲区

存储 obs, act, logp, rew_dict, done, agent_id 等数据
实现完整的 GAE(λ) 计算
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Iterable, Tuple
from ..utils.typing_alias import RolloutDict
from ..irdc.graph import build_adjacency


class OnPolicyBuffer:
    """
    On-policy 回放缓冲区
    
    存储单次 rollout 的数据，支持 GAE 计算和批次采样
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        num_agents: int,
        device: str = 'cpu',
        obs_mode: str = 'flat18',
        encoder_version: str = 'v1',
        graph_cache_mode: str = 'dense',
        graph_settings: Optional[Dict[str, Any]] = None,
        owner_key: str = "policy:shared",
        seq_len: int = 32,
        burn_in: int = 8,
    ):
        """
        初始化缓冲区
        
        参数:
            capacity: 容量（步数）
            obs_dim: 观测维度
            act_dim: 动作维度
            num_agents: 智能体数量
            device: 设备（'cpu' 或 'cuda'）
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.device = device
        self.obs_mode = obs_mode
        self.encoder_version = encoder_version
        self.owner_key = owner_key
        self.seq_len = int(seq_len)
        self.burn_in = int(burn_in)
        self.obs_metadata: Dict[str, Any] = {
            'obs_mode': obs_mode,
            'encoder_version': encoder_version,
        }
        self.graph_cache_mode = graph_cache_mode.lower()
        self.graph_settings = graph_settings or {}
        self.adj_records: list[Optional[torch.Tensor]] = [None] * capacity
        self.positions_records: list[Optional[torch.Tensor]] = [None] * capacity
        self.fov_records: list[Optional[torch.Tensor]] = [None] * capacity
        self.active_records: list[Optional[torch.Tensor]] = [None] * capacity
        
        # 初始化存储
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.act = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.logp = torch.zeros((capacity,), device=device)
        self.logits = torch.zeros((capacity, act_dim), device=device)  # 保存 old_logits
        self.rew_dict = {
            'R': torch.zeros((capacity,), device=device),
            'time': torch.zeros((capacity,), device=device),
            'batt': torch.zeros((capacity,), device=device),
            'int': torch.zeros((capacity,), device=device),
            'mix': torch.zeros((capacity,), device=device),
        }
        self.done = torch.zeros((capacity,), dtype=torch.bool, device=device)
        self.mask = torch.ones((capacity,), device=device)
        self.agent_id = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.values_dict = {
            'R': torch.zeros((capacity,), device=device),
            'time': torch.zeros((capacity,), device=device),
            'batt': torch.zeros((capacity,), device=device)
        }
        
        # GAE 计算结果
        self.adv_dict = None
        self.ret_dict = None
        
        self.ptr = 0
        self.size = 0
        self.has_mix_reward = False
        self._batch_size: Optional[int] = None
        self._h0_policy: Optional[torch.Tensor] = None
        self._h0_critic: Optional[torch.Tensor] = None

    def add_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Add a time-major batch: expects shapes [T,B,...]."""
        if 'obs' not in batch:
            raise KeyError("add_batch expects 'obs' in batch")

        obs = batch['obs']
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if obs.dim() != 3:
            raise ValueError("obs must be [T,B,D]")

        T, B = obs.shape[0], obs.shape[1]
        if self._batch_size is None:
            self._batch_size = B
        elif self._batch_size != B:
            raise ValueError(
                f"Batch size mismatch for {self.owner_key}: expected {self._batch_size}, got {B}"
            )

        if 'h0_policy' in batch:
            self._h0_policy = batch['h0_policy']
        if 'h0_critic' in batch:
            self._h0_critic = batch['h0_critic']

        def _slice_tensor(tensor: torch.Tensor, t_idx: int, b_idx: int) -> torch.Tensor:
            if tensor.dim() == 0:
                return tensor
            if tensor.dim() == 1:
                if tensor.shape[0] == T:
                    return tensor[t_idx]
                if tensor.shape[0] == B:
                    return tensor[b_idx]
            if tensor.dim() == 2:
                if tensor.shape[0] == T and tensor.shape[1] == B:
                    return tensor[t_idx, b_idx]
                if tensor.shape[0] == T:
                    return tensor[t_idx]
                if tensor.shape[0] == B:
                    return tensor[b_idx]
            if tensor.dim() == 3:
                if tensor.shape[0] == T and tensor.shape[1] == B:
                    return tensor[t_idx, b_idx]
                if tensor.shape[0] == T:
                    return tensor[t_idx]
            if tensor.dim() == 4 and tensor.shape[0] == T:
                return tensor[t_idx]
            return tensor

        for t in range(T):
            for b in range(B):
                sample: Dict[str, Any] = {}
                for key, value in batch.items():
                    if key == 'policy_key':
                        continue
                    if isinstance(value, dict):
                        sample[key] = {sub_k: _slice_tensor(sub_v, t, b) for sub_k, sub_v in value.items()}
                    elif isinstance(value, torch.Tensor):
                        sample[key] = _slice_tensor(value, t, b)
                    else:
                        sample[key] = value
                self.add(sample)

    def set_obs_metadata(self, meta: Dict[str, Any]) -> None:
        self.obs_metadata.update(meta)
    
    def add(self, sample_dict: Dict[str, Any]):
        """
        添加一个样本
        
        参数:
            sample_dict: 包含 obs, act, logp, rew_dict, done, agent_id 的字典
        """
        idx = self.ptr
        
        # 检查并添加数据
        if 'obs' in sample_dict:
            obs = sample_dict['obs']
            if obs.dim() == 1:
                assert obs.shape[0] == self.obs_dim, \
                    f"obs shape mismatch: {obs.shape} != ({self.obs_dim},)"
                self.obs[idx] = obs
            else:
                # 批量添加
                assert obs.shape[1] == self.obs_dim, \
                    f"obs shape mismatch: {obs.shape[1]} != {self.obs_dim}"
                for i, o in enumerate(obs):
                    if idx + i < self.capacity:
                        self.obs[idx + i] = o
        
        if 'act' in sample_dict:
            act = sample_dict['act']
            if act.dim() == 0:
                self.act[idx] = act
            else:
                for i, a in enumerate(act):
                    if idx + i < self.capacity:
                        self.act[idx + i] = a
        
        if 'logp' in sample_dict:
            logp = sample_dict['logp']
            if logp.dim() == 0:
                self.logp[idx] = logp
            else:
                for i, lp in enumerate(logp):
                    if idx + i < self.capacity:
                        self.logp[idx + i] = lp
        
        if 'logits' in sample_dict:
            logits = sample_dict['logits']
            if logits.dim() == 1:
                assert logits.shape[0] == self.act_dim, \
                    f"logits shape mismatch: {logits.shape} != ({self.act_dim},)"
                self.logits[idx] = logits
            else:
                for i, lg in enumerate(logits):
                    if idx + i < self.capacity:
                        self.logits[idx + i] = lg
        
        if 'rew_dict' in sample_dict:
            rew_dict = sample_dict['rew_dict']
            for key in self.rew_dict.keys():
                if key in rew_dict:
                    rew = rew_dict[key]
                    if key == 'mix':
                        self.has_mix_reward = True
                    if isinstance(rew, (float, int)):
                        self.rew_dict[key][idx] = torch.as_tensor(rew, device=self.device, dtype=self.rew_dict[key].dtype)
                    elif isinstance(rew, torch.Tensor):
                        if rew.dim() == 0:
                            self.rew_dict[key][idx] = rew.to(self.device)
                        else:
                            for i, r in enumerate(rew):
                                if idx + i < self.capacity:
                                    self.rew_dict[key][idx + i] = r.to(self.device)
                    else:
                        raise TypeError(f"Unsupported reward type {type(rew)} for key {key}")
        
        if 'done' in sample_dict:
            done = sample_dict['done']
            if done.dim() == 0:
                self.done[idx] = done
            else:
                for i, d in enumerate(done):
                    if idx + i < self.capacity:
                        self.done[idx + i] = d

        if 'mask' in sample_dict:
            mask = sample_dict['mask']
            if mask.dim() == 0:
                self.mask[idx] = mask
            else:
                for i, m in enumerate(mask):
                    if idx + i < self.capacity:
                        self.mask[idx + i] = m
        
        if 'agent_id' in sample_dict:
            agent_id = sample_dict['agent_id']
            if agent_id.dim() == 0:
                self.agent_id[idx] = agent_id
            else:
                for i, aid in enumerate(agent_id):
                    if idx + i < self.capacity:
                        self.agent_id[idx + i] = aid
        
        if 'values_dict' in sample_dict:
            values_dict = sample_dict['values_dict']
            for key in ['R', 'time', 'batt']:
                if key in values_dict:
                    val = values_dict[key]
                    if val.dim() == 0:
                        self.values_dict[key][idx] = val
                    else:
                        for i, v in enumerate(val):
                            if idx + i < self.capacity:
                                self.values_dict[key][idx + i] = v
        
        if 'adj' in sample_dict:
            adj = sample_dict['adj']
            if isinstance(adj, torch.Tensor):
                if adj.dim() == 2:
                    self.adj_records[idx] = adj.detach().to(self.device, dtype=torch.bool)
                else:
                    for i, a in enumerate(adj):
                        if idx + i < self.capacity:
                            self.adj_records[idx + i] = a.detach().to(self.device, dtype=torch.bool)
            else:
                self.adj_records[idx] = None

        if 'positions' in sample_dict:
            pos = sample_dict['positions']
            if isinstance(pos, torch.Tensor):
                if pos.dim() == 2:
                    self.positions_records[idx] = pos.detach().to('cpu')
                else:
                    for i, p in enumerate(pos):
                        if idx + i < self.capacity:
                            self.positions_records[idx + i] = p.detach().to('cpu')
            else:
                self.positions_records[idx] = None

        if 'fov' in sample_dict:
            fov = sample_dict['fov']
            if isinstance(fov, torch.Tensor):
                if fov.dim() == 4:
                    self.fov_records[idx] = fov.detach().to('cpu')
                else:
                    for i, fv in enumerate(fov):
                        if idx + i < self.capacity:
                            self.fov_records[idx + i] = fv.detach().to('cpu')
            else:
                self.fov_records[idx] = None

        if 'active_mask' in sample_dict:
            active = sample_dict['active_mask']
            if isinstance(active, torch.Tensor):
                if active.dim() == 1:
                    self.active_records[idx] = active.detach().to('cpu')
                else:
                    for i, am in enumerate(active):
                        if idx + i < self.capacity:
                            self.active_records[idx + i] = am.detach().to('cpu')
            else:
                self.active_records[idx] = None
        
        # 更新指针和大小
        if 'obs' in sample_dict and sample_dict['obs'].dim() > 1:
            batch_size = sample_dict['obs'].shape[0]
            self.ptr = (self.ptr + batch_size) % self.capacity
            self.size = min(self.size + batch_size, self.capacity)
        else:
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def finalize(self, last_values_dict: Optional[Dict[str, torch.Tensor]] = None):
        """
        完成数据收集，准备计算 GAE
        
        参数:
            last_values_dict: 最后一步的 V 值（用于 bootstrap），可选
        """
        # 如果提供了最后一步的 V 值，用于 bootstrap
        if last_values_dict is None:
            last_values_dict = {
                'R': torch.zeros(1, device=self.device),
                'time': torch.zeros(1, device=self.device),
                'batt': torch.zeros(1, device=self.device)
            }
    
    def compute_gae(
        self,
        gamma: float = 0.95,
        lam: float = 0.95,
        last_values_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        实现完整的 GAE(λ) 计算
        
        参数:
            gamma: 折扣因子
            lam: GAE lambda 参数
            last_values_dict: 最后一步的 V 值（用于 bootstrap），可选
        
        返回:
            advantages_dict: 优势函数字典，shape [size]
            returns_dict: 回报字典，shape [size]
        """
        if self.size == 0:
            return {}, {}
        
        # 如果没有提供最后一步的 V 值，使用零
        if last_values_dict is None:
            last_values_dict = {
                'R': torch.zeros(1, device=self.device),
                'time': torch.zeros(1, device=self.device),
                'batt': torch.zeros(1, device=self.device)
            }
        
        advantages_dict = {}
        returns_dict = {}
        
        # 对每个头部计算 GAE
        for key in ['R', 'time', 'batt']:
            if key == 'R' and self.has_mix_reward:
                rewards = self.rew_dict['mix'][:self.size]
            else:
                rewards = self.rew_dict[key][:self.size]  # [T]
            values = self.values_dict[key][:self.size]  # [T]
            mask_steps = self.mask[:self.size]  # [T]
            
            # 获取最后一步的 V 值
            last_value = last_values_dict.get(key, torch.zeros(1, device=self.device))
            if last_value.dim() > 0:
                last_value = last_value[0]
            
            # 计算 GAE
            advantages = torch.zeros_like(rewards)
            last_gae = 0.0
            
            # 从后往前计算
            for t in reversed(range(self.size)):
                if t == self.size - 1:
                    next_value = last_value
                else:
                    next_value = values[t + 1]

                mask_t = mask_steps[t]

                delta = rewards[t] + gamma * mask_t * next_value - values[t]
                advantages[t] = last_gae = delta + gamma * lam * mask_t * last_gae
            
            # 计算 returns = advantages + values
            returns = advantages + values
            
            # 对 returns 做 tanh 压缩，避免极端值拖垮 Critic
            returns = torch.tanh(returns)
            
            advantages_dict[key] = advantages
            returns_dict[key] = returns
        
        # 对每个头的 advantages 做标准化（零均值单位方差）
        for key in advantages_dict:
            adv = advantages_dict[key]
            adv_std = adv.std()
            if adv_std > 1e-8:
                advantages_dict[key] = (adv - adv.mean()) / adv_std
        
        self.adv_dict = advantages_dict
        self.ret_dict = returns_dict
        
        return advantages_dict, returns_dict
    
    def total_timesteps(self) -> int:
        return self.size

    def batch_B(self) -> Optional[int]:
        return self._batch_size

    def ready(self, min_timesteps: Optional[int] = None) -> bool:
        threshold = int(min_timesteps) if min_timesteps is not None else self.seq_len
        return self.size >= threshold

    def iter_minibatches(
        self,
        seq_len: Optional[int] = None,
        burn_in: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        if self.size == 0:
            return

        B = self._batch_size or self.num_agents or 1
        if B <= 0 or self.size % B != 0:
            return
        T = self.size // B

        def reshape(tensor: torch.Tensor, *extra_shape: int) -> torch.Tensor:
            return tensor[:T * B].view(T, B, *extra_shape) if tensor.numel() else tensor

        obs_tm = reshape(self.obs, self.obs_dim)
        act_tm = reshape(self.act)
        logp_tm = reshape(self.logp)
        mask_tm = reshape(self.mask)
        done_tm = reshape(self.done)
        logits_tm = reshape(self.logits, self.act_dim)

        rew_tm = {k: reshape(v) for k, v in self.rew_dict.items()}
        values_tm = {k: reshape(v) for k, v in self.values_dict.items()}

        agent_tm = reshape(self.agent_id)

        adj_entries: List[torch.Tensor] = []
        for idx in range(T * B):
            adj_tensor = self.adj_records[idx]
            if adj_tensor is None:
                adj_tensor = torch.zeros(B, dtype=torch.bool)
            adj_entries.append(adj_tensor.clone())
        if adj_entries:
            adj_tm = torch.stack(adj_entries).view(T, B, -1)
        else:
            adj_tm = None

        batch = {
            'obs': obs_tm,
            'act': act_tm,
            'logp': logp_tm,
            'old_logp': logp_tm,
            'mask': mask_tm,
            'done': done_tm,
            'logits': logits_tm,
            'rew_dict': rew_tm,
            'values_dict': values_tm,
            'agent_id': agent_tm,
            'adj': adj_tm,
            'h0_policy': self._h0_policy,
            'h0_critic': self._h0_critic,
            'seq_len': seq_len or self.seq_len,
            'burn_in': burn_in or self.burn_in,
        }
        if self.adv_dict is not None:
            batch['adv_dict'] = {
                key: reshape(self.adv_dict[key]).detach()
                for key in self.adv_dict.keys()
            }
        if self.ret_dict is not None:
            batch['ret_dict'] = {
                key: reshape(self.ret_dict[key]).detach()
                for key in self.ret_dict.keys()
            }

        if device is not None:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=device, non_blocking=True)
                elif isinstance(value, dict):
                    batch[key] = {
                        sub_k: sub_v.to(device=device, non_blocking=True)
                        if isinstance(sub_v, torch.Tensor) else sub_v
                        for sub_k, sub_v in value.items()
                    }

        yield batch

    def get_minibatches(self, *args, **kwargs):
        return list(self.iter_minibatches(*args, **kwargs))
    
    def get_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        获取批次数据
        
        参数:
            batch_size: 批次大小
        
        返回:
            batches: 批次列表，每个批次包含 obs, act, logp, old_logits, adv_dict, ret_dict 等
                    每个批次还包含 'batch_indices' 键，用于映射 adv_fused
        """
        if self.size == 0:
            return []
        
        # 确保 adv_dict 和 ret_dict 已计算
        if self.adv_dict is None or self.ret_dict is None:
            self.compute_gae()
        
        # 生成随机索引
        indices = torch.randperm(self.size, device=self.device)
        
        batches = []
        for i in range(0, self.size, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            adj_batch = self._gather_adj(batch_indices)

            batch = {
                'obs': self.obs[batch_indices],
                'act': self.act[batch_indices],
                'old_logp': self.logp[batch_indices],
                'old_logits': self.logits[batch_indices],
                'agent_id': self.agent_id[batch_indices],
                'batch_indices': batch_indices,  # 添加索引，用于映射 adv_fused
                'adv_dict': {
                    'R': self.adv_dict['R'][batch_indices],
                    'time': self.adv_dict['time'][batch_indices],
                    'batt': self.adv_dict['batt'][batch_indices]
                },
                'ret_dict': {
                    'R': self.ret_dict['R'][batch_indices],
                    'time': self.ret_dict['time'][batch_indices],
                    'batt': self.ret_dict['batt'][batch_indices]
                },
                'values_dict': {
                    'R': self.values_dict['R'][batch_indices],
                    'time': self.values_dict['time'][batch_indices],
                    'batt': self.values_dict['batt'][batch_indices]
                }
            }
            if 'int' in self.rew_dict:
                batch['rew_int'] = self.rew_dict['int'][batch_indices]
            if 'mix' in self.rew_dict:
                batch['rew_mix'] = self.rew_dict['mix'][batch_indices]
            if adj_batch is not None:
                batch['adj'] = adj_batch
            batches.append(batch)
        
        return batches

    def _gather_adj(self, batch_indices: torch.Tensor) -> Optional[torch.Tensor]:
        if self.graph_cache_mode == 'dense' or self.graph_cache_mode == 'indices':
            mats = []
            for idx in batch_indices.tolist():
                adj = self.adj_records[idx]
                if adj is None:
                    adj = torch.eye(self.num_agents, dtype=torch.bool, device=self.device)
                mats.append(adj.to(self.device, dtype=torch.bool))
            return torch.stack(mats) if mats else None
        if self.graph_cache_mode == 'recompute':
            mats = []
            for idx in batch_indices.tolist():
                pos = self.positions_records[idx]
                active = self.active_records[idx]
                fov = self.fov_records[idx]
                if pos is None:
                    pos_tensor = torch.zeros((self.num_agents, 2), dtype=torch.float32, device=self.device)
                else:
                    pos_tensor = pos.to(self.device, dtype=torch.float32)
                inactive = active.to(self.device, dtype=torch.bool) if active is not None else None
                fov_tensor = fov.to(self.device, dtype=torch.float32) if fov is not None else None
                mode = self.graph_settings.get('neighbor_mode', 'fov')
                radius = float(self.graph_settings.get('radius', 1.0))
                symmetry = self.graph_settings.get('symmetry', 'union')
                self_loop = bool(self.graph_settings.get('self_loop', True))
                topk_cfg = self.graph_settings.get('topk_neighbors')
                topk_val = int(topk_cfg) if topk_cfg is not None else None
                jitter_enabled = bool(self.graph_settings.get('deterministic_topk_jitter', True))
                adj = build_adjacency(
                    pos_tensor,
                    fov=fov_tensor,
                    mode=mode,
                    radius=radius,
                    symmetry=symmetry,
                    self_loop=self_loop,
                    inactive=inactive,
                    topk_neighbors=topk_val,
                    deterministic_topk_jitter=jitter_enabled,
                )
                mats.append(adj.to(self.device, dtype=torch.bool))
            return torch.stack(mats) if mats else None
        return None
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0
        self.adv_dict = None
        self.ret_dict = None
        self.has_mix_reward = False
        self.adj_records = [None] * self.capacity
        self.positions_records = [None] * self.capacity
        self.fov_records = [None] * self.capacity
        self.active_records = [None] * self.capacity
        self._batch_size = None
        self._h0_policy = None
        self._h0_critic = None


class PolicyBufferStore:
    """
    管理多个 OnPolicyBuffer，并根据 policy_key 聚合数据。
    """

    def __init__(self, cfg: Dict[str, Any], make_buffer_fn: Callable[..., OnPolicyBuffer]) -> None:
        self.cfg = cfg or {}
        self.make_buffer_fn = make_buffer_fn
        self.buffers: Dict[str, OnPolicyBuffer] = {}
        self.min_timesteps = int(self.cfg.get("training", {}).get("min_timesteps_per_update", 2048))

    def add_msg(self, policy_key: str, batch_cpu: Dict[str, torch.Tensor]) -> None:
        if policy_key not in self.buffers:
            self.buffers[policy_key] = self.make_buffer_fn(owner_key=policy_key)
        self.buffers[policy_key].add_batch(batch_cpu)

    def ready_keys(self) -> List[str]:
        return [
            key for key, buffer in self.buffers.items()
            if buffer.ready(self.min_timesteps)
        ]

    def iter_minibatches(self, policy_key: str, **kwargs):
        return self.buffers[policy_key].iter_minibatches(**kwargs)

    def get(self, policy_key: str) -> OnPolicyBuffer:
        return self.buffers[policy_key]

    def clear(self) -> None:
        for buffer in self.buffers.values():
            buffer.clear()


def _merge_time_major_batches(
    dst: Optional[Dict[str, torch.Tensor]],
    src: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    if dst:
        merged = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in dst.items()}
    for key, value in src.items():
        if isinstance(value, torch.Tensor):
            if key.startswith("h0_") or key == "agent_ids":
                merged[key] = value.clone()
            elif key in merged and isinstance(merged[key], torch.Tensor):
                merged[key] = torch.cat([merged[key], value], dim=0)
            else:
                merged[key] = value.clone()
        else:
            merged[key] = value
    return merged
