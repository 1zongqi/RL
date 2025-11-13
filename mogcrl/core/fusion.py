"""
多约束梯度融合策略

实现电量优先策略和余弦相似度阈值融合
增加分支统计计数
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


def cosine_sim(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算两个梯度向量的余弦相似度
    
    参数:
        g1: 第一个梯度，shape [B, ...] 或 [B, D]
        g2: 第二个梯度，shape [B, ...] 或 [B, D]
        eps: 数值稳定性常数
    
    返回:
        cosine_similarity: 余弦相似度标量或向量，shape [B] 或标量
    """
    # 展平为 [B, D] 以便计算
    g1_flat = g1.view(g1.shape[0], -1)
    g2_flat = g2.view(g2.shape[0], -1)
    
    # 计算范数
    n1 = torch.norm(g1_flat, dim=1, keepdim=True) + eps
    n2 = torch.norm(g2_flat, dim=1, keepdim=True) + eps
    
    # 余弦相似度
    cos_sim = (g1_flat * g2_flat).sum(dim=1, keepdim=True) / (n1 * n2)
    
    # 如果 batch_size=1，返回标量
    if cos_sim.shape[0] == 1:
        return cos_sim.squeeze()
    return cos_sim.squeeze(-1)


def fuse_time_batt(grad_t: torch.Tensor, grad_b: torch.Tensor, tau: float = 0.4) -> torch.Tensor:
    """
    融合时效和电量梯度
    
    规则：
    - 如果 |cos_sim| < tau（高度不相似），优先使用电量梯度
    - 否则，使用相似度加权融合
    
    参数:
        grad_t: 时效约束梯度，shape [B, ...]
        grad_b: 电量约束梯度，shape [B, ...]
        tau: 余弦相似度阈值
    
    返回:
        fused_grad: 融合后的梯度，shape 与输入相同
    """
    cs = cosine_sim(grad_t, grad_b)
    cs_abs = torch.abs(cs)
    
    # 处理批量情况：逐元素判断
    # 如果高度不相似，优先电量；否则融合
    # 使用 torch.where 进行逐元素选择
    
    # 相似度加权融合
    w_t = 0.5 * (1 + cs)
    w_b = 0.5 * (1 - cs)
    
    # 确保权重可广播
    if isinstance(w_t, torch.Tensor):
        if w_t.dim() == 0:
            # 标量情况
            if cs_abs < tau:
                return grad_b
            w_t = w_t.item()
            w_b = w_b.item()
        else:
            # 批量情况：逐元素处理
            w_t = w_t.view(-1, *([1] * (grad_t.dim() - 1)))
            w_b = w_b.view(-1, *([1] * (grad_b.dim() - 1)))
            cs_abs = cs_abs.view(-1, *([1] * (grad_t.dim() - 1)))
            
            # 逐元素选择：不相似用电量，相似用融合
            mask = cs_abs < tau
            fused_grad = torch.where(mask, grad_b, w_t * grad_t + w_b * grad_b)
            return fused_grad
    else:
        # 标量情况
        if cs_abs < tau:
            return grad_b
    
    # 标量融合
    fused_grad = w_t * grad_t + w_b * grad_b
    return fused_grad


def select_or_fuse(
    adv_R: torch.Tensor,
    adv_T: torch.Tensor,
    adv_B: torch.Tensor,
    viol_T_mask: torch.Tensor,
    viol_B_mask: torch.Tensor,
    tau: float = 0.6,
    ablate_no_fuse: bool = False,
    ablate_no_batt_priority: bool = False
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    逐样本分支选择或融合优势函数（论文同构版本）
    
    规则（逐样本）：
    - 无违：adv_R
    - 仅违T：adv_T
    - 仅违B：adv_B
    - 双违：
      * 消融模式1 (ablate_no_fuse=True): 直接使用 adv_B，跳过 cos 判断
      * 消融模式2 (ablate_no_batt_priority=True): 直接融合（wt=0.5, wb=0.5），跳过 cos 判断
      * 默认: if |cos(adv_T_i, adv_B_i)| < tau -> adv_B；else -> 融合 wt*adv_T + wb*adv_B
        其中 wt=0.5*(1+cos), wb=0.5*(1-cos)
    
    注意：cos 仅在"双违子集"上按样本计算，统计也仅基于双违样本
    
    参数:
        adv_R: 主奖励优势，shape [N]
        adv_T: 时效约束优势，shape [N]
        adv_B: 电量约束优势，shape [N]
        viol_T_mask: 时效违反掩码，shape [N] bool
        viol_B_mask: 电量违反掩码，shape [N] bool
        tau: 余弦相似度阈值
        ablate_no_fuse: 消融开关1 - 去融合：双违 → adv_B（branch='B'）
        ablate_no_batt_priority: 消融开关2 - 去电量优先：双违 → 融合（branch='FUSE'），忽略 cos/τ 判断
    
    返回:
        adv_fused: 融合后的优势，shape [N]
        stats: 包含分支比例和 cos 统计的字典（含 ablation 标志）
    """
    N = adv_R.shape[0]
    device = adv_R.device
    
    # 初始化融合优势
    adv_fused = torch.zeros_like(adv_R)
    
    # 分支掩码
    no_viol_mask = ~viol_T_mask & ~viol_B_mask  # 无违
    only_T_mask = viol_T_mask & ~viol_B_mask    # 仅违T
    only_B_mask = ~viol_T_mask & viol_B_mask    # 仅违B
    both_viol_mask = viol_T_mask & viol_B_mask  # 双违
    
    # 分支1：无违 -> adv_R
    adv_fused[no_viol_mask] = adv_R[no_viol_mask]
    
    # 分支2：仅违T -> adv_T
    adv_fused[only_T_mask] = adv_T[only_T_mask]
    
    # 分支3：仅违B -> adv_B
    adv_fused[only_B_mask] = adv_B[only_B_mask]
    
    # 分支4：双违 -> 根据消融模式决定融合或优先
    cos_values = []  # 收集双违样本的 cos 值
    
    if both_viol_mask.any():
        # 获取双违样本的索引
        both_indices = torch.where(both_viol_mask)[0]
        
        # 提取双违样本的优势
        adv_T_both = adv_T[both_indices]  # [M]
        adv_B_both = adv_B[both_indices]  # [M]
        
        # 消融模式1：去融合 - 双违直接使用 adv_B
        if ablate_no_fuse:
            adv_fused[both_indices] = adv_B[both_indices]
            # 不计算 cos，cos_values 保持为空
        # 消融模式2：去电量优先 - 双违直接融合（wt=0.5, wb=0.5）
        elif ablate_no_batt_priority:
            # 直接融合：wt=0.5, wb=0.5
            adv_fused[both_indices] = 0.5 * adv_T_both + 0.5 * adv_B_both
            # 不计算 cos，cos_values 保持为空
        # 默认模式：计算 cos 并决定融合或优先
        else:
            # 计算余弦相似度（逐样本）
            # 对于标量优势，cos = (adv_T * adv_B) / (|adv_T| * |adv_B| + eps)
            eps = 1e-8
            norm_T = torch.abs(adv_T_both) + eps
            norm_B = torch.abs(adv_B_both) + eps
            cos_both = (adv_T_both * adv_B_both) / (norm_T * norm_B)  # [M]
            
            # 收集 cos 值
            cos_values = cos_both.cpu().tolist()
            
            # 判断每个双违样本：|cos| < tau -> B优先，否则融合
            cos_abs = torch.abs(cos_both)
            use_B_priority = cos_abs < tau  # [M] bool
            
            # B 优先的样本
            B_priority_indices = both_indices[use_B_priority]
            adv_fused[B_priority_indices] = adv_B[B_priority_indices]
            
            # 融合的样本
            fuse_indices = both_indices[~use_B_priority]
            if len(fuse_indices) > 0:
                adv_T_fuse = adv_T[fuse_indices]
                adv_B_fuse = adv_B[fuse_indices]
                cos_fuse = cos_both[~use_B_priority]
                
                # 融合权重：wt=0.5*(1+cos), wb=0.5*(1-cos)
                wt = 0.5 * (1.0 + cos_fuse)
                wb = 0.5 * (1.0 - cos_fuse)
                
                # 融合：wt*adv_T + wb*adv_B
                adv_fused[fuse_indices] = wt * adv_T_fuse + wb * adv_B_fuse
    
    # 统计分支比例
    branch_R_count = no_viol_mask.sum().item()
    branch_T_count = only_T_mask.sum().item()
    branch_B_count = only_B_mask.sum().item()
    branch_FUSE_count = both_viol_mask.sum().item()
    
    branch_R_ratio = branch_R_count / N if N > 0 else 0.0
    branch_T_ratio = branch_T_count / N if N > 0 else 0.0
    branch_B_ratio = branch_B_count / N if N > 0 else 0.0
    branch_FUSE_ratio = branch_FUSE_count / N if N > 0 else 0.0
    
    # 计算 cos 统计（仅基于双违样本）
    if len(cos_values) > 0:
        import numpy as np
        cos_arr = np.array(cos_values)
        cos_mean = float(np.mean(cos_arr))
        cos_p25 = float(np.percentile(cos_arr, 25))
        cos_p50 = float(np.percentile(cos_arr, 50))
        cos_p75 = float(np.percentile(cos_arr, 75))
    else:
        # 双违样本为空或消融模式，cos 统计置 NaN
        cos_mean = float('nan')
        cos_p25 = float('nan')
        cos_p50 = float('nan')
        cos_p75 = float('nan')
    
    # 确定消融标志
    if ablate_no_fuse:
        ablation_flag = 'no_fuse'
    elif ablate_no_batt_priority:
        ablation_flag = 'no_batt_priority'
    else:
        ablation_flag = '/'
    
    stats = {
        'branch_R': branch_R_ratio,
        'branch_T': branch_T_ratio,
        'branch_B': branch_B_ratio,
        'branch_FUSE': branch_FUSE_ratio,
        'cos_mean': cos_mean,
        'cos_p25': cos_p25,
        'cos_p50': cos_p50,
        'cos_p75': cos_p75,
        'ablation': ablation_flag
    }
    
    return adv_fused, stats
