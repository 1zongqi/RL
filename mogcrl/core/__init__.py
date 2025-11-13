"""
MOGCRL 核心模块

包含：
- fusion: 多约束梯度融合策略
- constraints: 约束估计与判定
- update: Actor-Critic 更新逻辑
"""

from .fusion import cosine_sim, fuse_time_batt, select_or_fuse
from .constraints import estimate_Vk, check_violations, check_violations_per_sample
from .update import actor_loss, update_actor, update_critic
from .paper_parity import build_adv_and_ret, build_fused_adv, ppo_update_one_epoch

__all__ = [
    'cosine_sim', 'fuse_time_batt', 'select_or_fuse',
    'estimate_Vk', 'check_violations', 'check_violations_per_sample',
    'actor_loss', 'update_actor', 'update_critic',
    'build_adv_and_ret', 'build_fused_adv', 'ppo_update_one_epoch'
]

