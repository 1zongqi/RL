"""
指标聚合工具

用于聚合环境指标、分支比例、违反率等统计信息
"""

import torch
import numpy as np
from typing import Dict, List, Any


def aggregate_env_metrics(info_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    从 runner.collect() 返回的 info_agg 中聚合环境指标
    
    参数:
        info_list: info_agg 字典列表（通常只有一个元素，但支持列表以保持接口一致性）
    
    返回:
        metrics: 包含 throughput, deadline_violation_rate, min_SOC, completed_tasks 的字典
    """
    if len(info_list) == 0:
        return {
            'throughput': 0.0,
            'deadline_violation_rate': 0.0,
            'min_SOC': 1.0,
            'completed_tasks': 0.0,
            'episodes': 0.0,
            'avg_reward_per_agent': 0.0,
            'avg_cost_time_per_agent': 0.0,
            'avg_cost_batt_per_agent': 0.0,
            'cost_sum_per_agent': 0.0,
            'viol_time_rate': 0.0,
            'viol_batt_rate': 0.0,
            'episode_steps_total': 0.0,
            'episode_length_avg': 0.0,
            'episode_length_last': 0.0,
            'total_reward_sum': 0.0,
            'total_cost_time_sum': 0.0,
            'total_cost_batt_sum': 0.0,
            'total_constraint_cost_sum': 0.0,
            'total_agent_steps': 0.0,
        }
    
    # 如果传入的是单个字典，转换为列表
    if isinstance(info_list, dict):
        info_list = [info_list]
    
    # 聚合所有 info 字典
    total_throughput = 0.0
    total_deadline_violation_rate = 0.0
    min_SOC = 1.0
    total_completed_tasks = 0.0
    count = 0
    episodes_total = 0.0
    sum_avg_reward = 0.0
    sum_cost_time = 0.0
    sum_cost_batt = 0.0
    sum_cost_sum = 0.0
    sum_episode_steps_total = 0.0
    sum_episode_length_avg = 0.0
    last_episode_length = 0.0
    sum_total_reward = 0.0
    sum_total_cost_time = 0.0
    sum_total_cost_batt = 0.0
    sum_total_constraint_cost = 0.0
    sum_total_agent_steps = 0.0
    sum_viol_time = 0.0
    sum_viol_batt = 0.0
    reward_count = 0
    cost_time_count = 0
    cost_batt_count = 0
    cost_sum_count = 0
    viol_time_count = 0
    viol_batt_count = 0
    
    for info in info_list:
        if info is None:
            continue
        
        # throughput
        if 'throughput' in info:
            total_throughput += float(info['throughput'])
        
        # deadline_violation_rate
        if 'deadline_violation_rate' in info:
            total_deadline_violation_rate += float(info['deadline_violation_rate'])
        
        # min_SOC (对应 battery_min)
        if 'battery_min' in info:
            min_SOC = min(min_SOC, float(info['battery_min']))
        elif 'min_SOC' in info:
            min_SOC = min(min_SOC, float(info['min_SOC']))
        
        # completed_tasks
        if 'completed_tasks' in info:
            total_completed_tasks += float(info['completed_tasks'])
        
        if 'episodes' in info:
            episodes_total += float(info['episodes'])
        if 'avg_reward_per_agent' in info:
            sum_avg_reward += float(info['avg_reward_per_agent'])
            reward_count += 1
        if 'avg_cost_time_per_agent' in info:
            sum_cost_time += float(info['avg_cost_time_per_agent'])
            cost_time_count += 1
        if 'avg_cost_batt_per_agent' in info:
            sum_cost_batt += float(info['avg_cost_batt_per_agent'])
            cost_batt_count += 1
        if 'cost_sum_per_agent' in info:
            sum_cost_sum += float(info['cost_sum_per_agent'])
            cost_sum_count += 1
        if 'viol_time_rate' in info:
            sum_viol_time += float(info['viol_time_rate'])
            viol_time_count += 1
        if 'viol_batt_rate' in info:
            sum_viol_batt += float(info['viol_batt_rate'])
            viol_batt_count += 1
        if 'episode_steps_total' in info:
            sum_episode_steps_total += float(info['episode_steps_total'])
        if 'episode_length_avg' in info:
            sum_episode_length_avg += float(info['episode_length_avg'])
        if 'episode_length_last' in info:
            last_episode_length = float(info['episode_length_last'])
        if 'total_reward_sum' in info:
            sum_total_reward += float(info['total_reward_sum'])
        if 'total_cost_time_sum' in info:
            sum_total_cost_time += float(info['total_cost_time_sum'])
        if 'total_cost_batt_sum' in info:
            sum_total_cost_batt += float(info['total_cost_batt_sum'])
        if 'total_constraint_cost_sum' in info:
            sum_total_constraint_cost += float(info['total_constraint_cost_sum'])
        if 'total_agent_steps' in info:
            sum_total_agent_steps += float(info['total_agent_steps'])

        count += 1
    
    if count == 0:
        return {
            'throughput': 0.0,
            'deadline_violation_rate': 0.0,
            'min_SOC': 1.0,
            'completed_tasks': 0.0,
            'episodes': 0.0,
            'avg_reward_per_agent': 0.0,
            'avg_cost_time_per_agent': 0.0,
            'avg_cost_batt_per_agent': 0.0,
            'cost_sum_per_agent': 0.0,
            'viol_time_rate': 0.0,
            'viol_batt_rate': 0.0,
            'episode_steps_total': 0.0,
            'episode_length_avg': 0.0,
            'episode_length_last': 0.0,
            'total_reward_sum': 0.0,
            'total_cost_time_sum': 0.0,
            'total_cost_batt_sum': 0.0,
            'total_constraint_cost_sum': 0.0,
            'total_agent_steps': 0.0,
        }
    
    metrics = {
        'throughput': total_throughput / count if count > 0 else 0.0,
        'deadline_violation_rate': total_deadline_violation_rate / count if count > 0 else 0.0,
        'min_SOC': min_SOC,
        'completed_tasks': total_completed_tasks
    }
    metrics['episodes'] = episodes_total if episodes_total > 0 else float(count)
    metrics['avg_reward_per_agent'] = (sum_avg_reward / reward_count) if reward_count > 0 else 0.0
    metrics['avg_cost_time_per_agent'] = (sum_cost_time / cost_time_count) if cost_time_count > 0 else 0.0
    metrics['avg_cost_batt_per_agent'] = (sum_cost_batt / cost_batt_count) if cost_batt_count > 0 else 0.0
    metrics['cost_sum_per_agent'] = (sum_cost_sum / cost_sum_count) if cost_sum_count > 0 else 0.0
    metrics['viol_time_rate'] = (sum_viol_time / viol_time_count) if viol_time_count > 0 else 0.0
    metrics['viol_batt_rate'] = (sum_viol_batt / viol_batt_count) if viol_batt_count > 0 else 0.0
    metrics['episode_steps_total'] = sum_episode_steps_total
    metrics['episode_length_avg'] = (sum_episode_length_avg / count) if count > 0 else 0.0
    metrics['episode_length_last'] = last_episode_length
    metrics['total_reward_sum'] = sum_total_reward
    metrics['total_cost_time_sum'] = sum_total_cost_time
    metrics['total_cost_batt_sum'] = sum_total_cost_batt
    metrics['total_constraint_cost_sum'] = sum_total_constraint_cost
    metrics['total_agent_steps'] = sum_total_agent_steps

    debug_keys = [
        'debug_raw_present_steps',
        'debug_raw_finite_ratio',
        'debug_raw_available',
        'debug_raw_reward_sum',
        'debug_num_agents',
        'debug_rollout_length',
        'debug_samples_collected',
        'debug_rew_shape_T',
        'debug_rew_shape_B',
    ]
    for key in debug_keys:
        for info in info_list:
            if key in info:
                metrics[key] = float(info[key])
                break

    for info in info_list:
        value = info.get('reward_source_used')
        if value is not None:
            metrics['reward_source_used'] = value
            break

    return metrics


def aggregate_branch_and_viol(
    viol_T_mask: torch.Tensor,
    viol_B_mask: torch.Tensor,
    branch_stats: Dict[str, float]
) -> Dict[str, float]:
    """
    聚合分支比例和违反率
    
    参数:
        viol_T_mask: 时效违反掩码，shape [N] bool
        viol_B_mask: 电量违反掩码，shape [N] bool
        branch_stats: 分支统计字典（来自 select_or_fuse）
    
    返回:
        metrics: 包含违反率和分支比例的字典
    """
    N = viol_T_mask.shape[0]
    
    # 计算违反率
    viol_time_rate = viol_T_mask.float().mean().item() if N > 0 else 0.0
    viol_batt_rate = viol_B_mask.float().mean().item() if N > 0 else 0.0
    
    # 合并分支统计
    metrics = {
        'viol_time_rate': viol_time_rate,
        'viol_batt_rate': viol_batt_rate,
        **branch_stats
    }
    
    return metrics

