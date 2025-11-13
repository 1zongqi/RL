"""
Actor-Critic 更新逻辑

实现 PPO-Clip 风格的信任域更新
"""

import math
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.amp import GradScaler, autocast

from .constraints import constraint_violation_mask, ConstraintStats
from .trpo_qp import (
    compute_policy_gradient,
    compute_trpo_step_size,
    line_search,
    fisher_vector_product,
    solve_qp_2d,
    compute_b_k,
)


@dataclass
class PolicyGrads:
    """封装主任务与约束的策略梯度及Fisher向量积闭包。"""

    g_R: torch.Tensor
    g_time: torch.Tensor
    g_batt: torch.Tensor
    avp_fn: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class BranchLog:
    """记录分支选择及关键指标，便于后验分析。"""

    branch: str
    cos: float
    viol_time: bool
    viol_batt: bool
    gap_time: float
    gap_batt: float
    lambda_time: float
    lambda_batt: float
    kl: float = 0.0
    backtracks: int = 0


def _get_lambda_state(policy: nn.Module) -> Dict[str, float]:
    """为策略网络挂载/读取拉格朗日乘子状态。"""

    if not hasattr(policy, '_lagrange_multipliers'):
        policy._lagrange_multipliers = {'time': 0.0, 'batt': 0.0}
    return policy._lagrange_multipliers


def _update_lambda(lambda_value: float, gap: float, lr: float, lambda_max: float) -> float:
    """按 gap 更新乘子并投影到 [0, lambda_max]。"""

    candidate = lambda_value + lr * gap
    if not math.isfinite(candidate):
        candidate = lambda_value
    candidate = max(0.0, min(lambda_max, candidate))
    return candidate


def _compute_cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """安全地计算两向量夹角余弦。"""

    denom = torch.norm(vec_a) * torch.norm(vec_b)
    if denom.item() < 1e-12:
        return float('nan')
    cos_val = torch.dot(vec_a, vec_b) / (denom + 1e-12)
    return float(cos_val.item())


def select_update_branch(
    stats: ConstraintStats,
    grads: PolicyGrads,
    cos_sigma: float,
    use_energy_priority: bool
) -> Tuple[str, float]:
    """依据违约状态与梯度夹角选择更新分支。"""

    cos_val = _compute_cosine(grads.g_time, grads.g_batt)

    if stats.viol_time and stats.viol_batt:
        if use_energy_priority and (not math.isnan(cos_val)) and cos_val < cos_sigma:
            return 'ENERGY_PRIORITY', cos_val
        return 'QP_FUSION_2D', cos_val

    if stats.viol_time:
        return 'PRIMAL_DUAL_TIME', cos_val

    if stats.viol_batt:
        return 'PRIMAL_DUAL_BATT', cos_val

    return 'PRIMAL_DUAL_REWARD', cos_val


def apply_primal_dual(
    grads: PolicyGrads,
    stats: ConstraintStats,
    branch: str,
    lambda_state: Dict[str, float],
    lambda_lr: float,
    lambda_max: float
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """根据单约束/无约束情况执行 primal-dual 更新。"""

    lambda_time = _update_lambda(lambda_state['time'], stats.gap_time, lambda_lr, lambda_max)
    lambda_batt = _update_lambda(lambda_state['batt'], stats.gap_batt, lambda_lr, lambda_max)

    lambda_state['time'] = lambda_time
    lambda_state['batt'] = lambda_batt

    direction = grads.g_R.clone()

    if branch in ('PRIMAL_DUAL_TIME', 'PRIMAL_DUAL_REWARD') and lambda_time > 0.0:
        direction = direction - lambda_time * grads.g_time

    if branch in ('PRIMAL_DUAL_BATT', 'PRIMAL_DUAL_REWARD') and lambda_batt > 0.0:
        direction = direction - lambda_batt * grads.g_batt

    return direction, {'lambda_time': lambda_time, 'lambda_batt': lambda_batt}


def apply_energy_priority(
    grads: PolicyGrads,
    stats: ConstraintStats,
    lambda_state: Dict[str, float],
    lambda_lr: float,
    lambda_max: float
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """当双约束冲突时优先沿电量约束方向更新。"""

    lambda_state['time'] = _update_lambda(lambda_state['time'], stats.gap_time, lambda_lr, lambda_max)
    lambda_state['batt'] = _update_lambda(lambda_state['batt'], stats.gap_batt, lambda_lr, lambda_max)

    direction = grads.g_batt.clone()
    if torch.norm(direction) < 1e-12:
        direction = grads.g_R.clone()

    return direction, {'lambda_time': lambda_state['time'], 'lambda_batt': lambda_state['batt']}


def _ensure_direction(direction: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """若方向近零则退回主任务梯度，以避免停更。"""

    if torch.norm(direction) < 1e-12:
        return fallback.clone()
    return direction


def actor_loss(
    dist_old: Categorical,
    dist_new: Categorical,
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    adv_fused: torch.Tensor,
    clip_eps: float = 0.15,
    kl_penalty: float = 0.0,
    ent_coef: float = 0.005
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 Actor 损失（PPO-Clip 风格）
    
    参数:
        dist_old: 旧策略分布
        dist_new: 新策略分布
        old_logp: 旧策略的 log_prob，shape [B]
        new_logp: 新策略的 log_prob，shape [B]
        adv_fused: 融合后的优势函数，shape [B]
        clip_eps: PPO clip 阈值
        kl_penalty: KL 散度惩罚系数
    
    返回:
        loss: 损失标量
        stats: 包含 'kl', 'clip_frac' 的字典
    """
    # 计算重要性采样比率
    ratio = torch.exp(new_logp - old_logp)  # [B]
    
    # PPO-Clip 损失
    surr1 = ratio * adv_fused  # [B]
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_fused  # [B]
    L_clip = torch.min(surr1, surr2).mean()  # 标量
    
    # KL 散度惩罚
    kl = torch.distributions.kl.kl_divergence(dist_old, dist_new).mean()  # 标量
    kl_penalty_term = kl_penalty * kl
    
    # 熵正则化（鼓励探索）
    entropy = dist_new.entropy().mean()
    
    # 总损失（取负，因为要最大化）
    loss = -(L_clip) + kl_penalty_term + (-ent_coef) * entropy
    
    # 计算 clip 比例
    clip_frac = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean().item()
    
    stats = {
        'kl': float(kl.item()),
        'clip_frac': clip_frac,
        'entropy': float(entropy.item())
    }
    
    return loss, stats


def update_actor(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_batch: Dict[str, Any],
    kl_target: float = 0.02,
    clip_eps: float = 0.15,
    kl_penalty: float = 0.0,
    ent_coef: float = 0.005
) -> Dict[str, float]:
    """
    更新 Actor 策略
    
    自适应 KL penalty：如果 KL > 1.5*target，增加 penalty；如果 KL < target/1.5，减少 penalty
    
    参数:
        policy: 策略网络
        optimizer: 优化器
        data_batch: 数据批次，包含 obs, act, old_logp, old_logits, adv_fused 等
        kl_target: KL 散度目标阈值
        clip_eps: PPO clip 阈值
        kl_penalty: 当前 KL 惩罚系数（会被自适应调整）
    
    返回:
        stats: 包含 'loss', 'kl', 'clip_frac', 'beta' 的字典
    """
    obs = data_batch['obs']
    act = data_batch['act']
    old_logp = data_batch['old_logp'].detach()  # 确保不需要梯度
    old_logits = data_batch['old_logits'].detach()  # 确保不需要梯度
    adv_fused = data_batch['adv_fused'].detach()  # 确保不需要梯度
    agent_id = data_batch.get('agent_id', None)
    
    # 前向传播得到新策略
    dist_new, _, info_new = policy(obs, agent_id)
    new_logits = None
    if isinstance(info_new, dict):
        new_logits = info_new.get("logits")
    new_logp = dist_new.log_prob(act)
    
    # 构建旧策略分布（用于 KL 计算）
    dist_old = Categorical(logits=old_logits)
    
    # 计算损失
    loss, loss_stats = actor_loss(
        dist_old, dist_new, old_logp, new_logp,
        adv_fused, clip_eps, kl_penalty, ent_coef
    )
    
    # 自适应调整 KL penalty（仅用于记录，不重新计算损失）
    # 注意：重新计算损失会导致重复反向传播，所以这里只调整 beta 用于下次迭代
    kl = loss_stats['kl']
    beta = kl_penalty
    
    if kl > 1.5 * kl_target:
        beta = kl_penalty * 1.5
    elif kl < kl_target / 1.5:
        beta = kl_penalty / 1.5
    
    # 反向传播和更新
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    
    stats = {
        'loss': float(loss.item()),
        'kl': loss_stats['kl'],
        'clip_frac': loss_stats['clip_frac'],
        'entropy': loss_stats['entropy'],
        'beta': float(beta)
    }
    
    return stats


def update_critic(
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_batch: Dict[str, Any],
    critic_key: str = "critic:shared",
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    更新 Critic 网络
    
    三头 MSE 损失：loss = MSE(V_R, ret_R) + MSE(V_time, ret_time) + MSE(V_batt, ret_batt)
    
    参数:
        critic: Critic 网络
        optimizer: 优化器
        data_batch: 数据批次，包含 obs, ret_dict, agent_id 等
    
    返回:
        stats: 包含各头部损失信息的字典
    """
    obs = data_batch['obs']
    ret_dict = data_batch['ret_dict']
    agent_id = data_batch.get('agent_id', None)
    mask = data_batch.get('mask')
    adj = data_batch.get('adj')
    active_mask = data_batch.get('active_mask')

    def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if weight is None:
            return F.mse_loss(pred, target)
        diff = (pred - target) ** 2
        return (diff * weight).sum() / weight.sum().clamp_min(1.0)

    loss_R = loss_time = loss_batt = total_loss = None  # type: ignore
    device_type = (
        obs.device.type
        if isinstance(obs, torch.Tensor)
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cast_ctx = (
        autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled)
        if amp_enabled
        else nullcontext()
    )
    with cast_ctx:
        if adj is not None or obs.dim() == 3:
            if obs.dim() == 2:
                raise ValueError("Sequence adjacency provided but observations are flat.")
            values_dict, _ = critic(
                obs,
                agent_id,
                h_in=None,
                mask=mask,
                batch_first=False,
                adj=adj,
            )
            pred_R = values_dict['R'].squeeze(-1)
            pred_time = values_dict['time'].squeeze(-1)
            pred_batt = values_dict['batt'].squeeze(-1)
            target_R = ret_dict['R']
            target_time = ret_dict['time']
            target_batt = ret_dict['batt']
            if target_R.dim() == 1:
                target_R = target_R.view_as(pred_R)
                target_time = target_time.view_as(pred_time)
                target_batt = target_batt.view_as(pred_batt)
            weight = active_mask.to(pred_R.dtype) if active_mask is not None else None
            loss_R = _weighted_mse(pred_R, target_R, weight)
            loss_time = _weighted_mse(pred_time, target_time, weight)
            loss_batt = _weighted_mse(pred_batt, target_batt, weight)
        else:
            values_dict, _ = critic(obs, agent_id)
            loss_R = F.mse_loss(values_dict['R'].squeeze(-1), ret_dict['R'])
            loss_time = F.mse_loss(values_dict['time'].squeeze(-1), ret_dict['time'])
            loss_batt = F.mse_loss(values_dict['batt'].squeeze(-1), ret_dict['batt'])
        total_loss = loss_R + loss_time + loss_batt
    
    optimizer.zero_grad()
    if scaler is not None and amp_enabled:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        optimizer.step()
    
    stats = {
        'loss_R': float(loss_R.item()),
        'loss_time': float(loss_time.item()),
        'loss_batt': float(loss_batt.item()),
        'total_loss': float(total_loss.item()),
        'critic_key': critic_key,
    }
    
    return stats


# ==================== TRPO-QP 相关函数 ====================

def flatten_params(model: nn.Module) -> torch.Tensor:
    """展平模型参数为一维向量"""
    params = []
    for param in model.parameters():
        params.append(param.data.reshape(-1))
    return torch.cat(params)


def assign_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """将扁平参数赋值回模型"""
    idx = 0
    for param in model.parameters():
        param_length = param.numel()
        param.data.copy_(flat_params[idx:idx + param_length].view(param.shape))
        idx += param_length


def trpo_qp_update_actor(
    policy: nn.Module,
    critic: nn.Module,
    rollouts: Dict[str, Any],
    xi_time: float,
    xi_batt: float,
    config: Dict[str, Any],
    device: str,
    policy_key: str = "policy:shared",
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    基于TRPO-QP的actor更新
    
    参数:
        policy: 策略网络
        critic: Critic网络
        rollouts: rollout数据字典
        xi_time: 时效约束阈值
        xi_batt: 电量约束阈值
        config: TRPO-QP配置字典
        device: 设备
    
    返回:
        stats: 统计信息字典
    """
    from torch.distributions import Categorical

    print("[dbg] >>> ENTER trpo_qp_update_actor()")
    warned_defaults = config.setdefault('_warned_defaults', set())

    def cfg_value(key: str, default):
        if key not in config and key not in warned_defaults:
            print(f"[TRPO-QP] Missing config '{key}', fallback to default {default}")
            warned_defaults.add(key)
        return config.get(key, default)

    delta = float(cfg_value('delta', 0.01))
    zeta = float(cfg_value('zeta', 0.01))
    cos_sigma = float(cfg_value('cos_sigma', 0.7))
    use_energy_priority = bool(cfg_value('use_energy_priority', True))
    lambda_lr = float(cfg_value('lambda_lr', 5e-7))
    lambda_max = float(cfg_value('lambda_max', 100.0))
    cg_iters = int(cfg_value('cg_iters', 10))
    cg_damping = float(cfg_value('cg_damping', 1e-3))
    max_backtracks = int(cfg_value('max_backtracks', 10))
    accept_ratio = float(cfg_value('accept_ratio', 0.1))
    qp_solver = config.get('qp_solver', 'proxqp')

    if amp_enabled:
        logging.getLogger(__name__).warning(
            "[%s] AMP 已为 TRPO 策略路径禁用，强制使用 FP32。", policy_key
        )
        amp_enabled = False
        scaler = None

    old_params = flatten_params(policy)

    obs = rollouts['obs']
    act = rollouts['act']
    agent_id = rollouts.get('agent_id')
    old_logits = rollouts['logits'].float()
    mask_flat = rollouts.get('mask')
    adj_seq = rollouts.get('adj')
    active_mask_seq = rollouts.get('active_mask')

    adv_dict = rollouts.get('adv_dict')
    if adv_dict is None:
        ret_dict = rollouts['ret_dict']
        values_dict = rollouts['values_dict']
        adv_dict = {
            'R': ret_dict['R'] - values_dict['R'],
            'time': ret_dict['time'] - values_dict['time'],
            'batt': ret_dict['batt'] - values_dict['batt'],
        }

    if adj_seq is not None:
        T, B = adj_seq.shape[0], adj_seq.shape[1]
        obs_input = obs.view(T, B, -1).float()
        act_input = act.view(T, B)
        agent_input = agent_id.view(T, B) if agent_id is not None else None
        mask_input = mask_flat.view(T, B) if mask_flat is not None else None
        if active_mask_seq is not None:
            active_input = active_mask_seq.to(obs.device)
        else:
            active_input = torch.ones((T, B), device=obs.device)
        with torch.no_grad():
            old_dist = Categorical(logits=old_logits.view(T * B, -1))
        adv_R_input = adv_dict['R'].view(T, B).float()
        adv_time_input = adv_dict['time'].view(T, B).float()
        adv_batt_input = adv_dict['batt'].view(T, B).float()
    else:
        obs_input = obs.float()
        act_input = act
        agent_input = agent_id
        mask_input = mask_flat
        active_input = active_mask_seq
        with torch.no_grad():
            old_dist = Categorical(logits=old_logits)
        adv_R_input = adv_dict['R'].float()
        adv_time_input = adv_dict['time'].float()
        adv_batt_input = adv_dict['batt'].float()

    advantages = adv_R_input if 'adv_R_input' in locals() else None
    if 'advantages' in locals() and advantages is not None:
        adv_t = advantages.detach().flatten()
        print(f"[dbg] adv_mean={adv_t.mean().item():.3e} adv_std={adv_t.std(unbiased=False).item():.3e}")
    else:
        print("[dbg] WARN: advantages not found")

    g_R = compute_policy_gradient(
        policy,
        obs_input,
        act_input,
        adv_R_input,
        agent_input,
        mask=mask_input,
        adj=adj_seq,
        active_mask=active_input,
    ).float()
    g_time = compute_policy_gradient(
        policy,
        obs_input,
        act_input,
        adv_time_input,
        agent_input,
        mask=mask_input,
        adj=adj_seq,
        active_mask=active_input,
    ).float()
    g_batt = compute_policy_gradient(
        policy,
        obs_input,
        act_input,
        adv_batt_input,
        agent_input,
        mask=mask_input,
        adj=adj_seq,
        active_mask=active_input,
    ).float()

    flat_grad = g_R
    if 'flat_grad' in locals() and flat_grad is not None:
        pg_norm = float(flat_grad.norm().item())
        print(f"[dbg] pg_norm={pg_norm:.3e}")
    else:
        print("[dbg] WARN: flat_grad not found")

    def avp_fn(v: torch.Tensor) -> torch.Tensor:
        v_float = v.float()
        hv = fisher_vector_product(
            policy,
            obs_input,
            act_input,
            old_dist,
            v_float,
            agent_id=agent_input,
            mask=mask_input,
            adj=adj_seq,
            active_mask=active_input,
        ).float()
        if torch.isnan(hv).any() or torch.isinf(hv).any():
            raise ValueError('Avp produced NaN/Inf')
        return hv

    grads = PolicyGrads(g_R=g_R, g_time=g_time, g_batt=g_batt, avp_fn=avp_fn)
    constraint_stats = constraint_violation_mask(rollouts, xi_time, xi_batt)

    branch, cos_val = select_update_branch(constraint_stats, grads, cos_sigma, use_energy_priority)

    lambda_state = _get_lambda_state(policy)
    lambda_info: Dict[str, float]
    qp_info: Dict[str, Any] = {}
    direction: torch.Tensor
    cg_damping_used = cg_damping

    if branch.startswith('PRIMAL_DUAL'):
        direction, lambda_info = apply_primal_dual(
            grads,
            constraint_stats,
            branch,
            lambda_state,
            lambda_lr,
            lambda_max,
        )
    elif branch == 'ENERGY_PRIORITY':
        direction, lambda_info = apply_energy_priority(
            grads,
            constraint_stats,
            lambda_state,
            lambda_lr,
            lambda_max,
        )
    else:  # QP_FUSION_2D
        lambda_state['time'] = _update_lambda(lambda_state['time'], constraint_stats.gap_time, lambda_lr, lambda_max)
        lambda_state['batt'] = _update_lambda(lambda_state['batt'], constraint_stats.gap_batt, lambda_lr, lambda_max)
        lambda_info = {'lambda_time': lambda_state['time'], 'lambda_batt': lambda_state['batt']}

        gaps = (
            max(constraint_stats.gap_time, 0.0),
            max(constraint_stats.gap_batt, 0.0),
        )

        g_candidate = grads.g_R.clone()
        qp_info = {}
        direction = None

        for attempt in range(2):
            local_damping = cg_damping * (10 ** attempt)
            try:
                b_time = compute_b_k(
                    grads.g_time,
                    grads.avp_fn,
                    delta,
                    gaps[0],
                    zeta,
                    cg_iters,
                    local_damping,
                )
                b_batt = compute_b_k(
                    grads.g_batt,
                    grads.avp_fn,
                    delta,
                    gaps[1],
                    zeta,
                    cg_iters,
                    local_damping,
                )
            except Exception:
                continue

            g_star, qp_info = solve_qp_2d(
                grads.g_time,
                grads.g_batt,
                grads.avp_fn,
                b_time,
                b_batt,
                solver=qp_solver,
            )

            if torch.isnan(g_star).any() or torch.isinf(g_star).any():
                continue

            direction = g_star
            cg_damping_used = local_damping
            qp_info['b_time'] = b_time
            qp_info['b_batt'] = b_batt
            qp_info['cg_damping_used'] = local_damping
            break

        if direction is None:
            direction = g_candidate
            qp_info.setdefault('status', 'fallback')
            qp_info['reason'] = 'ill-conditioned'
            qp_info['cg_damping_used'] = cg_damping * 10

    direction = _ensure_direction(direction, grads.g_R)

    try:
        alpha_init = compute_trpo_step_size(direction, grads.avp_fn, delta)
    except ValueError:
        direction = grads.g_R.clone()
        alpha_init = compute_trpo_step_size(direction, grads.avp_fn, delta)

    expected_improve = float(torch.dot(direction.detach().float(), grads.g_R.detach().float()).item() * alpha_init)
    if 'expected_improve' in locals():
        print(f"[dbg] expected_improve={float(expected_improve):.3e}")
    else:
        print("[dbg] WARN: expected_improve not found")

    adv_main = adv_R_input
    alpha_final, ls_info = line_search(
        policy,
        old_params,
        direction,
        alpha_init,
        obs_input,
        act_input,
        old_dist,
        adv_main,
        delta,
        max_backtracks,
        accept_ratio,
        agent_id=agent_input,
        mask=mask_input,
        adj=adj_seq,
        active_mask=active_input,
    )

    if ls_info['accepted']:
        new_params = old_params + alpha_final * direction
        assign_flat_params(policy, new_params)
    else:
        assign_flat_params(policy, old_params)

    alpha = float(alpha_final)
    measured_kl = float(ls_info.get('kl', 0.0))
    bt_count = int(ls_info.get('backtracks', 0))
    if 'alpha' in locals() and 'measured_kl' in locals():
        print(f"[dbg] alpha_final={alpha:.3e} kl_measured={measured_kl:.3e} backtracks={bt_count}")
    # === /DEBUG ===

    branch_log = BranchLog(
        branch=branch,
        cos=float(cos_val) if cos_val is not None else float('nan'),
        viol_time=constraint_stats.viol_time,
        viol_batt=constraint_stats.viol_batt,
        gap_time=float(constraint_stats.gap_time),
        gap_batt=float(constraint_stats.gap_batt),
        lambda_time=lambda_info['lambda_time'],
        lambda_batt=lambda_info['lambda_batt'],
        kl=ls_info.get('kl', 0.0),
        backtracks=ls_info.get('backtracks', 0),
    )

    stats: Dict[str, Any] = {
        'alpha_init': float(alpha_init),
        'alpha_final': float(alpha_final),
        'branch': branch_log.branch,
        'cos': branch_log.cos,
        'viol_time': int(branch_log.viol_time),
        'viol_batt': int(branch_log.viol_batt),
        'gap_time': branch_log.gap_time,
        'gap_batt': branch_log.gap_batt,
        'lambda_time': branch_log.lambda_time,
        'lambda_batt': branch_log.lambda_batt,
        'kl': branch_log.kl,
        'surrogate_improve': ls_info.get('surrogate_improve', 0.0),
        'backtracks': branch_log.backtracks,
        'ls_accepted': bool(ls_info.get('accepted', False)),
        'cg_damping_used': cg_damping_used,
        'policy_key': policy_key,
    }

    if qp_info:
        stats.update({f"qp_{k}": v for k, v in qp_info.items()})

    if not ls_info.get('accepted', False):
        logging.getLogger(__name__).warning(
            "[%s] TRPO line search rejected; parameters reverted", policy_key
        )

    return stats