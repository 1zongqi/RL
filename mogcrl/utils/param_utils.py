"""
参数处理工具函数

提供模型参数的扁平化、赋值、梯度提取等功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
import numpy as np


def flatten_params(model: nn.Module) -> torch.Tensor:
    """
    展平模型参数为一维向量
    
    参数:
        model: PyTorch模型
    
    返回:
        flat_params: 扁平化的参数 [D]
    """
    params = []
    for param in model.parameters():
        params.append(param.data.reshape(-1))
    
    flat_params = torch.cat(params)
    return flat_params


def assign_flat_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    将扁平参数赋值回模型
    
    参数:
        model: PyTorch模型
        flat_params: 扁平化的参数 [D]
    """
    idx = 0
    for param in model.parameters():
        param_length = param.numel()
        param.data.copy_(flat_params[idx:idx + param_length].view(param.shape))
        idx += param_length


def get_flat_grad(model: nn.Module) -> torch.Tensor:
    """
    获取模型梯度的扁平版本
    
    参数:
        model: PyTorch模型
    
    返回:
        flat_grad: 扁平化的梯度 [D]
    """
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.reshape(-1))
        else:
            grads.append(torch.zeros_like(param.data).reshape(-1))
    
    flat_grad = torch.cat(grads)
    return flat_grad


def group_rollouts_by_agent(rollouts: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    按agent_id分组rollouts数据
    
    参数:
        rollouts: rollout数据字典，包含所有智能体的数据
    
    返回:
        rollouts_by_agent: {agent_id: agent_rollouts} 字典
    """
    if 'agent_id' not in rollouts:
        # 如果没有agent_id，返回整个rollouts作为单个agent
        return {0: rollouts}
    
    agent_ids = rollouts['agent_id']
    unique_agents = torch.unique(agent_ids).cpu().numpy()
    
    rollouts_by_agent = {}
    
    for agent_id in unique_agents:
        agent_id = int(agent_id)
        mask = (agent_ids == agent_id)
        
        agent_rollouts = {}
        for key, value in rollouts.items():
            if isinstance(value, torch.Tensor):
                if value.shape[0] == mask.shape[0]:
                    agent_rollouts[key] = value[mask]
                else:
                    agent_rollouts[key] = value
            elif isinstance(value, dict):
                # 处理嵌套字典（如rew_dict, adv_dict）
                agent_rollouts[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor) and sub_value.shape[0] == mask.shape[0]:
                        agent_rollouts[key][sub_key] = sub_value[mask]
                    else:
                        agent_rollouts[key][sub_key] = sub_value
            else:
                agent_rollouts[key] = value
        
        rollouts_by_agent[agent_id] = agent_rollouts
    
    return rollouts_by_agent


def set_flat_grad(model: nn.Module, flat_grad: torch.Tensor) -> None:
    """
    将扁平梯度赋值到模型参数的grad属性
    
    参数:
        model: PyTorch模型
        flat_grad: 扁平化的梯度 [D]
    """
    idx = 0
    for param in model.parameters():
        param_length = param.numel()
        if param.grad is None:
            param.grad = torch.zeros_like(param.data)
        param.grad.copy_(flat_grad[idx:idx + param_length].view(param.shape))
        idx += param_length


def count_parameters(model: nn.Module) -> int:
    """
    计算模型的参数总数
    
    参数:
        model: PyTorch模型
    
    返回:
        total_params: 参数总数
    """
    return sum(p.numel() for p in model.parameters())



