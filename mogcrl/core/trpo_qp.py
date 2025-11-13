"""
基于TRPO的二次规划约束优化算法

实现Fisher信息矩阵Hessian-向量积、共轭梯度求解、2×2降维QP、
步长规范化、线性回溯等核心组件
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Callable, Optional
from torch.distributions import Categorical


def fisher_vector_product(
    policy: nn.Module,
    obs: torch.Tensor,
    act: torch.Tensor,
    old_dist: Categorical,
    v: torch.Tensor,
    agent_id: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    adj: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算Fisher信息矩阵的Hessian-向量积 H*v
    
    其中H是KL散度相对于策略参数的Hessian矩阵
    
    参数:
        policy: 策略网络
        obs: 观测 [N, obs_dim]
        act: 动作 [N] (用于采样数据)
        old_dist: 旧策略分布
        v: 向量 [D] (D是参数总维度)
    
    返回:
        Hv: Hessian-vector product [D]
    """
    # 1. 计算新策略分布
    if adj is not None:
        if obs.dim() == 2:
            T, B = adj.shape[0], adj.shape[1]
            obs_seq = obs.view(T, B, -1)
            agent_id_seq = agent_id.view(T, B) if agent_id is not None else None
            mask_seq = mask.view(T, B) if mask is not None else None
            active_seq = active_mask.view(T, B) if active_mask is not None else None
        else:
            obs_seq = obs
            agent_id_seq = agent_id
            mask_seq = mask
            active_seq = active_mask
        new_dist_seq, _, _ = policy(
            obs_seq,
            agent_id_seq,
            mask=mask_seq,
            batch_first=False,
            adj=adj,
        )
        logits = new_dist_seq.logits.view(-1, new_dist_seq.logits.shape[-1])
        new_dist = Categorical(logits=logits)
        kl_all = torch.distributions.kl_divergence(old_dist, new_dist)
        if active_seq is not None:
            weight = active_seq.reshape(-1).to(kl_all.dtype)
            kl = (kl_all * weight).sum() / weight.sum().clamp_min(1.0)
        else:
            kl = kl_all.mean()
    else:
        if agent_id is None:
            agent_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        new_dist, _, _ = policy(obs, agent_id)
        kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
    
    # 3. 计算一阶梯度 g = ∇θ KL
    # 注意：需要retain_graph=True，因为在batch_gradient_fusion中会为多个目标（reward/time/battery）
    # 多次调用fisher_vector_product，它们共享同一个策略网络的计算图
    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True, retain_graph=True)
    flat_g = torch.cat([g.reshape(-1) for g in grads])
    
    # 4. 计算方向导数 gᵀv
    gv = (flat_g * v).sum()
    
    # 5. 计算二阶导数 Hv = ∇θ(gᵀv)
    # 注意：这里也需要retain_graph=True，因为在CG循环中会多次调用此函数
    # 计算图会在CG完成后自动释放
    hvs = torch.autograd.grad(gv, policy.parameters(), retain_graph=True)
    flat_hv = torch.cat([hv.reshape(-1) for hv in hvs]).detach()
    
    return flat_hv


def conjugate_gradients(
    Avp_fn: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    iters: int = 10,
    tol: float = 1e-10,
    use_float64: bool = True
) -> torch.Tensor:
    """
    标准共轭梯度算法求解 Ax = b
    
    参数:
        Avp_fn: 函数 A*v，计算矩阵向量积
        b: 右端向量 [D]
        iters: 最大迭代次数
        tol: 收敛容差
        use_float64: 是否使用float64提高数值稳定性
    
    返回:
        x: 近似解 [D]
    """
    # 转换为float64提高数值稳定性
    original_dtype = b.dtype
    if use_float64 and b.dtype == torch.float32:
        b = b.double()
    
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)
    
    for i in range(iters):
        # Avp_fn可能需要特定精度的输入
        if use_float64 and original_dtype == torch.float32:
            # 对于float64模式，尝试转换
            try:
                p_input = p.float()
                Ap_result = Avp_fn(p_input)
                # 确保输出也是double
                if Ap_result.dtype != torch.float64:
                    Ap = Ap_result.double()
                else:
                    Ap = Ap_result
            except:
                # 如果转换失败，直接使用
                Ap = Avp_fn(p)
        else:
            Ap = Avp_fn(p)
        
        alpha = rs_old / (torch.dot(p, Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        
        if torch.sqrt(rs_new) < tol:
            break
        
        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new
    
    # 转换回原始精度
    if use_float64 and original_dtype == torch.float32:
        x = x.float()
    
    return x


def solve_Hinv_v(
    Avp_fn: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    cg_iters: int = 10,
    cg_damping: float = 1e-3,
    tol: float = 1e-10,
    use_float64: bool = True
) -> torch.Tensor:
    """
    用共轭梯度求解 H^{-1}v
    
    实际求解 (H + damping*I)x = v，得到 x ≈ H^{-1}v
    
    参数:
        Avp_fn: lambda x: H*x，计算Hessian-vector product
        v: 右端向量 [D]
        cg_iters: CG最大迭代次数
        cg_damping: 阻尼系数（数值稳定性）
        tol: 收敛容差
        use_float64: 是否使用float64提高数值稳定性
    
    返回:
        x: 近似解 H^{-1}v [D]
    """
    def damped_Avp(x):
        return Avp_fn(x) + cg_damping * x
    
    return conjugate_gradients(damped_Avp, v, iters=cg_iters, tol=tol, use_float64=use_float64)


def compute_b_k(
    g_k: torch.Tensor,
    Avp_fn: Callable[[torch.Tensor], torch.Tensor],
    delta: float,
    gap_k: float,
    zeta: float,
    cg_iters: int = 10,
    cg_damping: float = 1e-3,
    eps: float = 1e-12
) -> float:
    """
    计算截断阈值 b_k (公式3-5)
    
    b_k = min(√(2δ·g_kᵀH^{-1}g_k), gap_k + ζ)
    
    参数:
        g_k: 约束梯度 [D]
        Avp_fn: Hessian-vector product函数
        delta: 信赖域大小
        gap_k: 约束 gap（>0 表示viol，≤0 表示满足）
        zeta: 松弛系数
        cg_iters: CG迭代次数
        cg_damping: CG阻尼系数
    
    返回:
        b_k: 截断阈值（标量）
    """
    # 1. 求解 H^{-1}·g_k
    Hinv_gk = solve_Hinv_v(Avp_fn, g_k, cg_iters=cg_iters, cg_damping=cg_damping)
    
    # 2. 计算 term1 = √(2δ·g_kᵀH^{-1}g_k)
    gHg_inv = float(torch.dot(g_k, Hinv_gk).item())
    quad = max(gHg_inv, 0.0)
    term1 = np.sqrt(max(2.0 * delta * quad, eps))  # 确保非负且>0
    
    # 3. 计算 term2 = F_k - d_k + ζ
    term2 = float(gap_k + zeta)
    
    # 4. 返回最小值
    b_k = min(term1, term2)
    
    return b_k


def solve_qp_2d_subspace(
    g_time: torch.Tensor,
    g_batt: torch.Tensor,
    Avp_fn: Callable[[torch.Tensor], torch.Tensor],
    b_time: float,
    b_batt: float,
    solver: str = 'proxqp'
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    在span{g_time, g_batt}子空间求解2×2 QP
    
    g = x_t·g_time + x_b·g_batt
    
    min  0.5·[x_t, x_b]ᵀ·G·[x_t, x_b]
    s.t. (g_timeᵀg) + b_time ≤ 0
         (g_battᵀg) + b_batt ≤ 0
    
    参数:
        g_time: 时效约束梯度 [D]
        g_batt: 电量约束梯度 [D]
        Avp_fn: Hessian-vector product函数
        b_time, b_batt: 截断阈值
        solver: QP求解器名称
    
    返回:
        g_star: 融合梯度 [D]
        qp_status: 求解状态信息
    """
    # 1. 计算 H·g_time, H·g_batt（使用Avp）
    Hg_t = Avp_fn(g_time)
    Hg_b = Avp_fn(g_batt)
    
    # 2. 构造 2×2 矩阵 G_ij = g_iᵀ·(H·g_j)
    G = np.array([
        [float(torch.dot(g_time, Hg_t).item()), float(torch.dot(g_time, Hg_b).item())],
        [float(torch.dot(g_batt, Hg_t).item()), float(torch.dot(g_batt, Hg_b).item())]
    ], dtype=np.float64)
    
    # 确保G对称且正定（数值稳定性）
    G = 0.5 * (G + G.T)
    min_eig = np.linalg.eigvalsh(G)[0]
    if min_eig < 1e-8:
        G = G + (1e-8 - min_eig) * np.eye(2)
    
    # 3. 构造约束矩阵 S_ij = g_iᵀ·g_j
    S = np.array([
        [float(torch.dot(g_time, g_time).item()), float(torch.dot(g_time, g_batt).item())],
        [float(torch.dot(g_batt, g_time).item()), float(torch.dot(g_batt, g_batt).item())]
    ], dtype=np.float64)
    
    # 4. 约束：A·x + b_vec ≤ 0，转换为 G·x ≤ h
    A = S  # 2×2
    b_vec = np.array([b_time, b_batt], dtype=np.float64)
    
    # 5. 调用qpsolvers求解 - 多层回退机制
    # min 0.5·xᵀGx, s.t. A·x + b_vec ≤ 0 → G·x ≤ -b_vec
    
    from qpsolvers import solve_qp
    
    # 尝试多个求解器，按优先级顺序
    solvers_to_try = ['proxqp', 'osqp', 'quadprog'] if solver == 'proxqp' else [solver, 'proxqp']
    
    for solver_name in solvers_to_try:
        try:
            x = solve_qp(P=G, q=np.zeros(2), G=A, h=-b_vec, solver=solver_name)
            
            if x is not None:
                # 6. 还原到原空间：g_star = x_t·g_time + x_b·g_batt
                g_star = x[0] * g_time + x[1] * g_batt
                
                return g_star, {
                    'status': 'success',
                    'solver': solver_name,
                    'x_t': float(x[0]),
                    'x_b': float(x[1])
                }
        except Exception as e:
            # 这个求解器失败，继续尝试下一个
            continue
    
    # 所有求解器都失败，回退到电量优先
    return g_batt, {'status': 'all_qp_failed', 'fallback': 'battery', 'solvers_tried': solvers_to_try}


def solve_qp_2d(
    g_time: torch.Tensor,
    g_batt: torch.Tensor,
    Avp_fn: Callable[[torch.Tensor], torch.Tensor],
    b_time: float,
    b_batt: float,
    solver: str = 'proxqp',
    eps: float = 1e-10
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """在考虑退化情形的前提下求解2D QP。

    若任一约束梯度近似为零向量，则退化为另一约束或主任务方向，
    以避免求解器数值不稳定。
    """

    norm_time = torch.norm(g_time).item()
    norm_batt = torch.norm(g_batt).item()

    if norm_time < eps and norm_batt < eps:
        return g_time.new_zeros(g_time.shape), {
            'status': 'degenerate',
            'reason': 'both_zero'
        }

    if norm_time < eps:
        return g_batt, {
            'status': 'degenerate',
            'reason': 'time_zero'
        }

    if norm_batt < eps:
        return g_time, {
            'status': 'degenerate',
            'reason': 'batt_zero'
        }

    return solve_qp_2d_subspace(g_time, g_batt, Avp_fn, b_time, b_batt, solver=solver)


def compute_trpo_step_size(
    g_star: torch.Tensor,
    Avp_fn: Callable[[torch.Tensor], torch.Tensor],
    delta: float
) -> float:
    """
    计算TRPO步长 (公式3-6)
    
    α = min(1, √(2δ/(g*ᵀHg*)))
    
    注意：使用Avp计算g*ᵀHg*，不是H^{-1}
    
    参数:
        g_star: 融合梯度 [D]
        Avp_fn: Hessian-vector product函数
        delta: 信赖域大小
    
    返回:
        alpha: 步长系数（标量）
    """
    # 1. 计算 H·g_star
    Hg = Avp_fn(g_star)
    
    # 2. 计算 g*ᵀHg*
    quad = float(torch.dot(g_star, Hg).item())
    
    # 3. 计算步长
    alpha = min(1.0, np.sqrt(2.0 * delta / (max(quad, 1e-12))))
    
    return alpha


def line_search(
    policy: nn.Module,
    old_params: torch.Tensor,
    g_star: torch.Tensor,
    alpha_init: float,
    obs: torch.Tensor,
    act: torch.Tensor,
    old_dist: Categorical,
    adv: torch.Tensor,
    delta: float,
    max_backtracks: int = 10,
    accept_ratio: float = 0.1,
    agent_id: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    adj: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    线性回溯搜索最优步长
    
    尝试步长 alpha, alpha*0.5, alpha*0.25, ...
    检查 KL ≤ delta 且 surrogate 提升 > 0
    
    参数:
        policy: 策略网络
        old_params: 旧策略参数 [D]
        g_star: 搜索方向 [D]
        alpha_init: 初始步长
        obs: 观测 [N, obs_dim]
        act: 动作 [N]
        old_dist: 旧策略分布
        adv: 优势函数 [N]
        delta: 信赖域大小
        max_backtracks: 最大回溯次数
        accept_ratio: surrogate最小提升比例
    
    返回:
        alpha_final: 最终步长
        info: 搜索信息
    """
    from .update import flatten_params, assign_flat_params
    
    # 计算旧策略的surrogate和old_logp
    # 注意：在old策略点，重要性采样比率π_old/π_old = 1
    # 所以old_surrogate就是优势函数的均值
    with torch.no_grad():
        if adj is not None:
            if obs.dim() == 2:
                T, B = adj.shape[0], adj.shape[1]
                act_flat = act.view(-1)
                adv_flat = adv.view(-1)
                weight = active_mask.reshape(-1).to(adv_flat.dtype) if active_mask is not None else None
            else:
                T, B = obs.shape[0], obs.shape[1]
                act_flat = act.reshape(-1)
                adv_flat = adv.reshape(-1)
                weight = active_mask.reshape(-1).to(adv_flat.dtype) if active_mask is not None else None
        else:
            act_flat = act
            adv_flat = adv
            weight = None
        old_logp = old_dist.log_prob(act_flat)
        if weight is not None:
            old_surrogate = (adv_flat * weight).sum() / weight.sum().clamp_min(1.0)
        else:
            old_surrogate = adv_flat.mean()
    
    alpha = alpha_init
    improve_value = 0.0
    improve_ratio = -float('inf')
    kl = torch.tensor(float('nan'))
    for i in range(max_backtracks):
        # 尝试新参数
        new_params = old_params + alpha * g_star
        assign_flat_params(policy, new_params)
        
        # 计算新策略的KL和surrogate
        with torch.no_grad():
            if adj is not None:
                if obs.dim() == 2:
                    obs_seq = obs.view(T, B, -1)
                    agent_id_seq = agent_id.view(T, B) if agent_id is not None else None
                    mask_seq = mask.view(T, B) if mask is not None else None
                else:
                    obs_seq = obs
                    agent_id_seq = agent_id
                    mask_seq = mask
                new_dist_seq, _, _ = policy(
                    obs_seq,
                    agent_id_seq,
                    mask=mask_seq,
                    batch_first=False,
                    adj=adj,
                )
                new_logits = new_dist_seq.logits.view(-1, new_dist_seq.logits.shape[-1])
                new_dist = Categorical(logits=new_logits)
                kl_all = torch.distributions.kl_divergence(old_dist, new_dist)
                if weight is not None:
                    kl = (kl_all * weight).sum() / weight.sum().clamp_min(1.0)
                else:
                    kl = kl_all.mean()
                new_logp = new_dist.log_prob(act_flat)
            else:
                agent_id_eval = agent_id if agent_id is not None else torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
                new_dist, _, _ = policy(obs, agent_id_eval)
                new_logp = new_dist.log_prob(act)
                kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            
            # Surrogate提升
            ratio = torch.exp(new_logp - old_logp)
            if weight is not None:
                new_surrogate = (ratio * adv_flat * weight).sum() / weight.sum().clamp_min(1.0)
            else:
                new_surrogate = (ratio * adv_flat).mean()
            improve = new_surrogate - old_surrogate
        
        # 检查接受条件
        old_surrogate_value = float(old_surrogate.item() if hasattr(old_surrogate, "item") else old_surrogate)
        improve_value = float(improve.item() if hasattr(improve, "item") else improve)
        improve_ratio = improve_value / (abs(old_surrogate_value) + 1e-8)

        if kl.item() <= delta and improve_ratio >= accept_ratio:
            return alpha, {
                'accepted': True,
                'backtracks': i,
                'kl': float(kl.item()),
                'surrogate_improve': improve_value,
                'improve_ratio': improve_ratio,
            }
        
        # 减小步长
        alpha *= 0.5
    
    # 回溯失败，恢复旧参数
    assign_flat_params(policy, old_params)
    return 0.0, {
        'accepted': False,
        'backtracks': max_backtracks,
        'kl': float(kl.item()) if isinstance(kl, torch.Tensor) else float(kl),
        'surrogate_improve': improve_value,
        'improve_ratio': improve_ratio,
    }


def compute_cos_statistics(cos_values: list) -> Dict[str, float]:
    """
    计算余弦相似度统计
    
    参数:
        cos_values: 余弦相似度值列表
    
    返回:
        stats: 统计字典（mean/p25/p50/p75）
    """
    if len(cos_values) == 0:
        return {
            'cos_mean': np.nan,
            'cos_p25': np.nan,
            'cos_p50': np.nan,
            'cos_p75': np.nan
        }
    
    cos_arr = np.array(cos_values)
    return {
        'cos_mean': float(np.mean(cos_arr)),
        'cos_p25': float(np.percentile(cos_arr, 25)),
        'cos_p50': float(np.percentile(cos_arr, 50)),
        'cos_p75': float(np.percentile(cos_arr, 75))
    }


def compute_policy_gradient(
    policy: nn.Module,
    obs: torch.Tensor,
    act: torch.Tensor,
    adv: torch.Tensor,
    agent_id: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    adj: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算策略梯度 ∇θ E[π(a|s) * A]
    
    参数:
        policy: 策略网络
        obs: 观测 [N, obs_dim]
        act: 动作 [N]
        adv: 优势函数 [N]
        agent_id: 智能体ID [N]
    
    返回:
        flat_grad: 扁平化的梯度 [D]
    """
    if adj is not None:
        if obs.dim() == 2:
            T, B = adj.shape[0], adj.shape[1]
            obs_seq = obs.view(T, B, -1)
            act_seq = act.view(T, B)
            adv_seq = adv.view(T, B)
            agent_id_seq = agent_id.view(T, B) if agent_id is not None else None
            mask_seq = mask.view(T, B) if mask is not None else None
            active_seq = active_mask.view(T, B) if active_mask is not None else None
        else:
            obs_seq = obs
            act_seq = act
            adv_seq = adv
            agent_id_seq = agent_id
            mask_seq = mask
            active_seq = active_mask
        dist, _, _ = policy(
            obs_seq,
            agent_id_seq,
            mask=mask_seq,
            batch_first=False,
            adj=adj,
        )
        log_prob = dist.log_prob(act_seq)
        if active_seq is not None:
            weight = active_seq.to(log_prob.dtype)
            loss = -((log_prob * adv_seq) * weight).sum() / weight.sum().clamp_min(1.0)
        else:
            loss = -(log_prob * adv_seq).mean()
    else:
        if agent_id is None:
            agent_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        dist, _, _ = policy(obs, agent_id)
        log_prob = dist.log_prob(act)
        loss = -(log_prob * adv).mean()
    
    # 计算梯度
    policy.zero_grad()
    loss.backward()
    
    # 收集并扁平化梯度
    flat_grad = torch.cat([p.grad.reshape(-1) for p in policy.parameters()]).detach()
    
    return flat_grad


def batch_gradient_fusion(
    policy: nn.Module,
    critic: nn.Module,
    rollouts: Dict[str, torch.Tensor],
    xi_time: float,
    xi_batt: float,
    delta: float,
    zeta: float,
    cos_sigma: float,
    cg_iters: int,
    cg_damping: float,
    device: str
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    批级别的约束判断与梯度融合
    
    流程：
    1. 计算当前策略的 V_time, V_batt（F_k）
    2. 判断约束违反情况
    3. 根据违反情况选择/融合梯度
    
    参数:
        policy: 策略网络
        critic: Critic网络
        rollouts: rollout数据字典
        xi_time: 时效约束阈值
        xi_batt: 电量约束阈值
        delta: 信赖域大小
        zeta: 松弛系数
        cos_sigma: 余弦相似度阈值
        cg_iters: CG迭代次数
        cg_damping: CG阻尼系数
        device: 设备
    
    返回:
        g_fused: 融合后的梯度 [D]
        stats: 统计信息字典
    """
    # 1. 估计约束值 F_k（使用returns的均值）
    # 注意：F_k的物理含义
    # - time/batt的reward是负值成本（如-0.5表示成本0.5）
    # - returns = Σγ^t·reward也是负值
    # - F_k = returns.mean()是负值效用（越大越好，越接近0表示成本越小）
    # - 例如：F_time=-0.1优于F_time=-0.3
    ret_dict = rollouts.get('ret_dict')
    if ret_dict is None:
        # 如果没有ret_dict，使用rew_dict估计
        ret_time = rollouts['rew_dict']['time']
        ret_batt = rollouts['rew_dict']['batt']
    else:
        ret_time = ret_dict['time']  # [N]
        ret_batt = ret_dict['batt']  # [N]
    
    F_time = float(ret_time.mean().item())
    F_batt = float(ret_batt.mean().item())
    
    # 2. 判断约束违反
    # 因为F_k是负值效用（越大越好），xi_k是阈值（也是负值）
    # F_k < xi_k 表示效用低于阈值，即违反约束
    # 例如：F_time=-0.3 < xi_time=-0.2 → 违反（成本过高）
    viol_time = F_time < xi_time
    viol_batt = F_batt < xi_batt
    
    # 3. 提取数据
    obs = rollouts['obs']
    act = rollouts['act']
    old_logits = rollouts['logits']
    old_dist = Categorical(logits=old_logits)
    
    # 获取优势函数
    if 'adv_dict' in rollouts:
        adv_R = rollouts['adv_dict']['R']
        adv_T = rollouts['adv_dict']['time']
        adv_B = rollouts['adv_dict']['batt']
    else:
        # 简单估计：returns - values
        adv_R = rollouts['ret_dict']['R'] - rollouts['values_dict']['R']
        adv_T = rollouts['ret_dict']['time'] - rollouts['values_dict']['time']
        adv_B = rollouts['ret_dict']['batt'] - rollouts['values_dict']['batt']
    
    agent_id = rollouts.get('agent_id')
    
    # 4. 准备Avp函数闭包
    def Avp_fn(v):
        return fisher_vector_product(policy, obs, act, old_dist, v)
    
    # 5. 根据违反情况选择梯度
    if not viol_time and not viol_batt:
        # 无违反：最大化主奖励
        g_fused = compute_policy_gradient(policy, obs, act, adv_R, agent_id)
        method = 'reward'
        cos_sim = np.nan
        qp_info = {}
    
    elif viol_time and not viol_batt:
        # 仅违时效
        g_fused = compute_policy_gradient(policy, obs, act, adv_T, agent_id)
        method = 'time_only'
        cos_sim = np.nan
        qp_info = {}
    
    elif not viol_time and viol_batt:
        # 仅违电量
        g_fused = compute_policy_gradient(policy, obs, act, adv_B, agent_id)
        method = 'batt_only'
        cos_sim = np.nan
        qp_info = {}
    
    else:
        # 双违：计算两个约束梯度
        g_time = compute_policy_gradient(policy, obs, act, adv_T, agent_id)
        g_batt = compute_policy_gradient(policy, obs, act, adv_B, agent_id)
        
        # 计算余弦相似度
        cos_sim = float(torch.dot(g_time, g_batt).item() / 
                       (torch.norm(g_time) * torch.norm(g_batt) + 1e-12))
        
        if cos_sim < -cos_sigma:
            # 相似度差距大 → 电量优先
            g_fused = g_batt
            method = 'batt_priority'
            qp_info = {'cos_sim': cos_sim}
        else:
            # 相似度差距小 → 2D-QP融合
            # 计算 b_k
            gap_time = xi_time - F_time
            gap_batt = xi_batt - F_batt

            b_time = compute_b_k(
                g_time, Avp_fn, delta, gap_time, zeta, cg_iters, cg_damping
            )
            b_batt = compute_b_k(
                g_batt, Avp_fn, delta, gap_batt, zeta, cg_iters, cg_damping
            )
            
            # 求解 2D-QP
            g_fused, qp_info = solve_qp_2d_subspace(
                g_time, g_batt, Avp_fn, b_time, b_batt
            )
            qp_info['cos_sim'] = cos_sim
            qp_info['b_time'] = b_time
            qp_info['b_batt'] = b_batt
            method = f"qp_fusion_{qp_info['status']}"
    
    # 统计分支使用情况
    branch_counts = {
        'branch_R': 1 if method == 'reward' else 0,
        'branch_T': 1 if method == 'time_only' else 0,
        'branch_B': 1 if method == 'batt_only' else 0,
        'branch_FUSE': 1 if 'qp_fusion' in method or method == 'batt_priority' else 0
    }
    
    # 为批级别模式计算cos统计（虽然只有一个值）
    cos_stats = compute_cos_statistics([cos_sim] if not np.isnan(cos_sim) else [])
    
    stats = {
        'method': method,
        'viol_time': viol_time,
        'viol_batt': viol_batt,
        'F_time': F_time,
        'F_batt': F_batt,
        'cos_sim': cos_sim,
        **branch_counts,
        **cos_stats,
        **qp_info
    }
    
    return g_fused, stats


def per_agent_gradient_fusion(
    policy: nn.Module,
    critic: nn.Module,
    rollouts: Dict[str, torch.Tensor],
    xi_time: float,
    xi_batt: float,
    config: Dict[str, Any],
    device: str
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    逐智能体的梯度融合
    
    将rollouts按agent_id分组，对每个智能体独立判断和融合
    
    参数:
        policy: 策略网络
        critic: Critic网络
        rollouts: rollout数据字典
        xi_time: 时效约束阈值
        xi_batt: 电量约束阈值
        config: TRPO-QP配置
        device: 设备
    
    返回:
        g_avg: 平均梯度 [D]
        stats: 统计信息
    """
    from ..utils.param_utils import group_rollouts_by_agent
    
    # 按agent_id分组
    rollouts_by_agent = group_rollouts_by_agent(rollouts)
    
    agent_gradients = []
    agent_stats = []
    cos_values = []  # 收集所有cos值
    
    for agent_id, agent_data in rollouts_by_agent.items():
        g_fused, stats = batch_gradient_fusion(
            policy, critic, agent_data, xi_time, xi_batt,
            config['delta'], config['zeta'], config['cos_sigma'],
            config['cg_iters'], config['cg_damping'], device
        )
        agent_gradients.append(g_fused)
        agent_stats.append(stats)
        
        # 收集cos值（仅在双违情况下）
        if 'cos_sim' in stats and not np.isnan(stats['cos_sim']):
            cos_values.append(stats['cos_sim'])
    
    # 平均所有智能体的梯度
    g_avg = torch.stack(agent_gradients).mean(dim=0)
    
    # 计算cos统计
    cos_stats = compute_cos_statistics(cos_values)
    
    # 聚合branch统计
    branch_stats = {
        'branch_R': np.mean([s.get('branch_R', 0) for s in agent_stats]),
        'branch_T': np.mean([s.get('branch_T', 0) for s in agent_stats]),
        'branch_B': np.mean([s.get('branch_B', 0) for s in agent_stats]),
        'branch_FUSE': np.mean([s.get('branch_FUSE', 0) for s in agent_stats])
    }
    
    # 聚合统计信息
    avg_stats = {
        'method': agent_stats[0]['method'] if agent_stats else 'none',
        'viol_time': np.mean([s['viol_time'] for s in agent_stats]),
        'viol_batt': np.mean([s['viol_batt'] for s in agent_stats]),
        'F_time': np.mean([s['F_time'] for s in agent_stats]),
        'F_batt': np.mean([s['F_batt'] for s in agent_stats]),
        'num_agents': len(agent_stats),
        **branch_stats,  # 添加branch统计
        **cos_stats  # 添加cos统计
    }
    
    return g_avg, avg_stats


