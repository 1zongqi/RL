"""
论文同构版本核心封装

实现无对偶更新的约束检测+融合+PPO 流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Iterator
from ..memory.buffer import OnPolicyBuffer
from ..core.fusion import select_or_fuse
from ..core.constraints import check_violations_per_sample, estimate_Vk
from ..core.update import update_actor, update_critic


def mix_rewards(
    reward_ext: torch.Tensor,
    reward_int: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Combine extrinsic and intrinsic rewards."""
    if reward_int is None or epsilon == 0.0:
        return reward_ext
    return reward_ext + epsilon * reward_int


def build_adv_and_ret(
    buffer: OnPolicyBuffer,
    critic: nn.Module,
    gamma: float = 0.95,
    gae_lambda: float = 0.95
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    构建优势函数和回报（论文同构版本）

    引入 mask: delta = r + γ * mask * V_{t+1} - V_t，确保跨 episode 时清零。
    """
    size = buffer.size
    device = buffer.device

    if size == 0:
        empty = torch.zeros(0, device=device)
        adv_dict = {'R': empty, 'time': empty, 'batt': empty}
        ret_dict = {'R': empty, 'time': empty, 'batt': empty}
        values_dict = {'R': empty, 'time': empty, 'batt': empty}
        buffer.adv_dict = adv_dict
        buffer.ret_dict = ret_dict
        return adv_dict, ret_dict, values_dict

    last_obs = buffer.obs[size - 1:size]
    last_agent_id = buffer.agent_id[size - 1:size]

    with torch.no_grad():
        last_values_raw, _ = critic(last_obs, last_agent_id)

    last_values = {}
    for key in ['R', 'time', 'batt']:
        value_tensor = last_values_raw[key]
        if value_tensor.dim() > 1:
            value_tensor = value_tensor.squeeze(-1)
        last_values[key] = value_tensor.reshape(-1)[0]

    mask = (~buffer.done[:size]).float()

    adv_dict: Dict[str, torch.Tensor] = {}
    ret_dict: Dict[str, torch.Tensor] = {}
    values_dict: Dict[str, torch.Tensor] = {}

    for key in ['R', 'time', 'batt']:
        rewards = buffer.rew_dict[key][:size]
        values = buffer.values_dict[key][:size]

        next_values = torch.zeros_like(values)
        if size > 1:
            next_values[:-1] = values[1:]
        next_values[-1] = last_values[key]

        deltas = rewards + gamma * mask * next_values - values

        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=device)
        for t in reversed(range(size)):
            gae = deltas[t] + gamma * gae_lambda * mask[t] * gae
            advantages[t] = gae

        returns = torch.tanh(advantages + values)

        adv_dict[key] = advantages
        ret_dict[key] = returns
        values_dict[key] = values

    for key, adv in adv_dict.items():
        std = adv.std()
        if std > 1e-8:
            adv_dict[key] = (adv - adv.mean()) / std

    buffer.adv_dict = adv_dict
    buffer.ret_dict = ret_dict

    return adv_dict, ret_dict, values_dict


def build_fused_adv(
    adv_dict: Dict[str, torch.Tensor],
    ret_dict: Dict[str, torch.Tensor],
    xi_time: float,
    xi_batt: float,
    tau: float = 0.6,
    ablate_no_fuse: bool = False,
    ablate_no_batt_priority: bool = False
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    构建融合后的优势函数（论文同构版本）
    
    先逐样本判违，再调用 select_or_fuse 进行逐样本融合
    
    参数:
        adv_dict: 优势函数字典，包含 'R', 'time', 'batt'，每个值为 shape [N] 的张量
        ret_dict: 回报字典，包含 'R', 'time', 'batt'，每个值为 shape [N] 的张量
        xi_time: 时效约束阈值
        xi_batt: 电量约束阈值
        tau: 余弦相似度阈值
        ablate_no_fuse: 消融开关1 - 去融合：双违 → adv_B（branch='B'）
        ablate_no_batt_priority: 消融开关2 - 去电量优先：双违 → 融合（branch='FUSE'），忽略 cos/τ 判断
    
    返回:
        adv_fused: 融合后的优势，shape [N]
        branch_stats: 分支统计字典（包含分支比例和 cos 统计）
        viol_T_mask: 时效违反掩码，shape [N] bool
        viol_B_mask: 电量违反掩码，shape [N] bool
    """
    # 获取回报
    ret_time = ret_dict['time']  # [N]
    ret_batt = ret_dict['batt']  # [N]
    
    # 逐样本判违
    viol_T_mask, viol_B_mask = check_violations_per_sample(
        ret_time, ret_batt, xi_time, xi_batt
    )
    
    # 获取优势
    adv_R = adv_dict['R']  # [N]
    adv_T = adv_dict['time']  # [N]
    adv_B = adv_dict['batt']  # [N]
    
    # 逐样本融合（透传消融参数）
    adv_fused, branch_stats = select_or_fuse(
        adv_R, adv_T, adv_B,
        viol_T_mask, viol_B_mask,
        tau=tau,
        ablate_no_fuse=ablate_no_fuse,
        ablate_no_batt_priority=ablate_no_batt_priority
    )
    
    return adv_fused, branch_stats, viol_T_mask, viol_B_mask


def ppo_update_one_epoch(
    policy: nn.Module,
    critic: nn.Module,
    optimizer_pi: torch.optim.Optimizer,
    optimizer_v: torch.optim.Optimizer,
    batch_iter: Iterator[Dict[str, Any]],
    adv_fused_key: str = 'adv_fused',
    clip_eps: float = 0.20,
    kl_target: float = 0.04,
    ent_coef: float = 0.01,
    v_weights: Tuple[float, float, float] = (1.0, 0.7, 0.7)
) -> Dict[str, float]:
    """
    执行一轮 PPO 更新（论文同构版本）
    
    从 batch 中取 adv_fused，调用 update_actor 和 update_critic
    
    参数:
        policy: 策略网络
        critic: Critic 网络
        optimizer_pi: Actor 优化器
        optimizer_v: Critic 优化器
        batch_iter: 批次迭代器
        adv_fused_key: batch 中融合优势的键名
        clip_eps: PPO clip 阈值
        kl_target: KL 散度目标
        ent_coef: 熵正则化系数
        v_weights: Critic 分头权重 (R, time, batt)
    
    返回:
        stats: 包含更新统计的字典
    """
    total_kl = 0.0
    total_clip_frac = 0.0
    total_entropy = 0.0
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    num_updates = 0
    
    kl_penalty = 0.0  # 初始 KL penalty
    
    for batch in batch_iter:
        # 确保 adv_fused 不需要梯度
        if adv_fused_key in batch:
            batch[adv_fused_key] = batch[adv_fused_key].detach()
        
        # 更新 Actor
        actor_stats = update_actor(
            policy, optimizer_pi, batch,
            kl_target=kl_target,
            clip_eps=clip_eps,
            kl_penalty=kl_penalty,
            ent_coef=ent_coef
        )
        
        # 更新 KL penalty（自适应）
        kl_penalty = actor_stats.get('beta', 0.0)
        
        # 更新 Critic（使用分头权重）
        critic_stats = update_critic_with_weights(
            critic, optimizer_v, batch,
            v_weights=v_weights
        )
        
        # 累计统计
        total_kl += actor_stats['kl']
        total_clip_frac += actor_stats['clip_frac']
        total_entropy += actor_stats['entropy']
        total_actor_loss += actor_stats['loss']
        total_critic_loss += critic_stats['total_loss']
        num_updates += 1
    
    # 平均统计
    if num_updates > 0:
        avg_kl = total_kl / num_updates
        avg_clip_frac = total_clip_frac / num_updates
        avg_entropy = total_entropy / num_updates
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
    else:
        avg_kl = 0.0
        avg_clip_frac = 0.0
        avg_entropy = 0.0
        avg_actor_loss = 0.0
        avg_critic_loss = 0.0
    
    stats = {
        'kl': avg_kl,
        'clip_frac': avg_clip_frac,
        'entropy': avg_entropy,
        'actor_loss': avg_actor_loss,
        'critic_loss': avg_critic_loss,
        'beta': kl_penalty
    }
    
    return stats


def update_critic_with_weights(
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_batch: Dict[str, Any],
    v_weights: Tuple[float, float, float] = (1.0, 0.7, 0.7)
) -> Dict[str, float]:
    """
    更新 Critic 网络（支持分头权重和 Huber 损失）
    
    参数:
        critic: Critic 网络
        optimizer: 优化器
        data_batch: 数据批次，包含 obs, ret_dict, agent_id 等
        v_weights: 分头权重 (R, time, batt)
    
    返回:
        stats: 包含各头部损失信息的字典
    """
    obs = data_batch['obs']
    ret_dict = data_batch['ret_dict']
    agent_id = data_batch.get('agent_id', None)
    
    # 前向传播得到 V 值
    values_dict, _ = critic(obs, agent_id)
    
    # 计算各头部的 Huber 损失（或 MSE）
    weight_R, weight_T, weight_B = v_weights
    
    # 使用 Huber 损失（更鲁棒）
    loss_R = F.smooth_l1_loss(values_dict['R'].squeeze(-1), ret_dict['R']) * weight_R
    loss_time = F.smooth_l1_loss(values_dict['time'].squeeze(-1), ret_dict['time']) * weight_T
    loss_batt = F.smooth_l1_loss(values_dict['batt'].squeeze(-1), ret_dict['batt']) * weight_B
    
    # 总损失
    total_loss = loss_R + loss_time + loss_batt
    
    # 反向传播和更新
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    optimizer.step()
    
    stats = {
        'loss_R': float(loss_R.item()),
        'loss_time': float(loss_time.item()),
        'loss_batt': float(loss_batt.item()),
        'total_loss': float(total_loss.item())
    }
    
    return stats

