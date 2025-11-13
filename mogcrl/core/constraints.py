"""
约束估计与判定模块

提供批次级约束估计、违约检测与统计结构体，供TRPO-QP更新使用。
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import torch

from ..utils.typing_alias import RolloutDict


@dataclass
class ConstraintStats:
    """约束统计量与违约信息。

    Attributes:
        viol_time: 是否违反时效约束。
        viol_batt: 是否违反电量约束。
        gap_time: 违约幅度（>0 表示违反；单位：与 returns 同量纲）。
        gap_batt: 电量违约幅度（>0 表示违反）。
        F_time: 批次平均时效 returns。
        F_batt: 批次平均电量 returns。
    """

    viol_time: bool
    viol_batt: bool
    gap_time: float
    gap_batt: float
    F_time: float
    F_batt: float

def estimate_constraint_returns(rollouts: RolloutDict) -> Dict[str, float]:
    """估计批次级的约束 returns 均值。

    优先使用 `ret_dict`（GAE 之后的 returns），若缺失则回退到
    `values_dict` 或 `rew_dict`，确保 time/batt 均有数值。
    """

    def _mean_or_zero(tensor):
        return float(tensor.mean().item()) if tensor is not None else 0.0

    if 'ret_dict' in rollouts and rollouts['ret_dict'] is not None:
        ret_dict = rollouts['ret_dict']
        return {
            'R': _mean_or_zero(ret_dict.get('R')),
            'time': _mean_or_zero(ret_dict.get('time')),
            'batt': _mean_or_zero(ret_dict.get('batt')),
        }

    if 'values_dict' in rollouts and rollouts['values_dict'] is not None:
        values_dict = rollouts['values_dict']
        return {
            'R': _mean_or_zero(values_dict.get('R')),
            'time': _mean_or_zero(values_dict.get('time')),
            'batt': _mean_or_zero(values_dict.get('batt')),
        }

    if 'rew_dict' in rollouts and rollouts['rew_dict'] is not None:
        rew_dict = rollouts['rew_dict']
        return {
            'R': _mean_or_zero(rew_dict.get('R')),
            'time': _mean_or_zero(rew_dict.get('time')),
            'batt': _mean_or_zero(rew_dict.get('batt')),
        }

    return {'R': 0.0, 'time': 0.0, 'batt': 0.0}


def estimate_Vk(critic: torch.nn.Module, rollouts: RolloutDict) -> Dict[str, float]:
    """
    估计各约束的 V 值
    
    使用 returns_dict 的均值估计长期 V_k（或 values 的均值近似）
    注意：这一步是近似估计，后续可改为稳态加权
    
    参数:
        critic: 多头部 critic 网络
        rollouts: 回放数据字典，包含 obs, rew_dict, values_dict 等
    
    返回:
        Vk_dict: 包含 {'R': V0, 'time': V1, 'batt': V2} 的字典
    """
    # 优先使用 returns_dict（如果已计算 GAE）
    if 'ret_dict' in rollouts and rollouts['ret_dict'] is not None:
        ret_dict = rollouts['ret_dict']
        Vk_dict = {}
        for key in ['R', 'time', 'batt']:
            if key in ret_dict:
                # 使用 returns 的均值估计 V_k
                Vk_dict[key] = float(ret_dict[key].mean().item())
            else:
                Vk_dict[key] = 0.0
    elif 'values_dict' in rollouts and rollouts['values_dict'] is not None:
        # 如果没有 returns，使用 values 的均值
        values_dict = rollouts['values_dict']
        Vk_dict = {}
        for key in ['R', 'time', 'batt']:
            if key in values_dict:
                Vk_dict[key] = float(values_dict[key].mean().item())
            else:
                Vk_dict[key] = 0.0
    elif 'rew_dict' in rollouts and rollouts['rew_dict'] is not None:
        # 最后回退到 reward 的均值估计
        rew_dict = rollouts['rew_dict']
        Vk_dict = {}
        for key in ['R', 'time', 'batt']:
            if key in rew_dict:
                Vk_dict[key] = float(rew_dict[key].mean().item())
            else:
                Vk_dict[key] = 0.0
    else:
        # 完全占位值
        Vk_dict = {'R': 0.0, 'time': 0.0, 'batt': 0.0}
    
    return Vk_dict


def constraint_violation_mask(
    rollouts: RolloutDict,
    xi_time: float,
    xi_batt: float
) -> ConstraintStats:
    """根据批次 returns 评估约束违反及违约幅度。

    Args:
        rollouts: 包含 `ret_dict`/`values_dict`/`rew_dict` 的 rollout 数据。
        xi_time: 时效约束阈值。
        xi_batt: 电量约束阈值。

    Returns:
        ConstraintStats: 包含违约布尔、gap 与均值估计。
    """

    returns = estimate_constraint_returns(rollouts)
    F_time = returns.get('time', 0.0)
    F_batt = returns.get('batt', 0.0)

    gap_time = xi_time - F_time
    gap_batt = xi_batt - F_batt

    viol_time = gap_time > 0.0
    viol_batt = gap_batt > 0.0

    return ConstraintStats(
        viol_time=bool(viol_time),
        viol_batt=bool(viol_batt),
        gap_time=float(gap_time),
        gap_batt=float(gap_batt),
        F_time=float(F_time),
        F_batt=float(F_batt),
    )


def check_violations(
    Vk: Dict[str, float],
    xi_time: float,
    xi_batt: float
) -> Tuple[bool, bool]:
    """
    检查约束是否违反（批次级别，保留用于兼容）
    
    规则：V_k < xi_k 表示违反（因为方向是"越大越好"）
    
    参数:
        Vk: V 值字典，包含 'R', 'time', 'batt'
        xi_time: 时效约束阈值
        xi_batt: 电量约束阈值
    
    返回:
        (viol_time, viol_batt): 两个布尔值，表示是否违反
    """
    V_time = Vk.get('time', 0.0)
    V_batt = Vk.get('batt', 0.0)
    
    viol_time = V_time < xi_time
    viol_batt = V_batt < xi_batt
    
    return viol_time, viol_batt


def check_violations_per_sample(
    ret_time: torch.Tensor,
    ret_batt: torch.Tensor,
    xi_time: float,
    xi_batt: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    逐样本检查约束是否违反（论文同构版本）
    
    规则：ret_k < xi_k 表示违反（因为方向是"越大越好"）
    
    参数:
        ret_time: 时效回报，shape [N]
        ret_batt: 电量回报，shape [N]
        xi_time: 时效约束阈值
        xi_batt: 电量约束阈值
    
    返回:
        (viol_T_mask, viol_B_mask): 两个 bool 张量，shape [N]，表示每个样本是否违反
    """
    viol_T_mask = ret_time < xi_time  # [N] bool
    viol_B_mask = ret_batt < xi_batt  # [N] bool
    
    return viol_T_mask, viol_B_mask
