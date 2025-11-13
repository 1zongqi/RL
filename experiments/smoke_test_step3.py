#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3 连通性测试脚本

验证真实环境适配器与 VectorRunner 的集成
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs import FulfillmentAdapter
from mogcrl.nets import CategoricalPolicy, MultiHeadCritic
from mogcrl.runners import VectorRunner
from mogcrl.utils.seeding import set_seed


def main():
    """主测试函数"""
    print("=" * 60)
    print("开始 Step 3 连通性测试...")
    print("=" * 60)
    
    set_seed(123)
    device = 'cpu'
    
    # 1) 实例化 FulfillmentAdapter
    print("\n1. 测试 FulfillmentAdapter...")
    adapter = FulfillmentAdapter(seed=123, max_agents=8, obs_mode='dict')
    
    # 2) 测试 reset
    print("2. 测试 reset()...")
    obs_dict, info_dict = adapter.reset()
    print(f"  obs_dict keys (agent 0): {list(obs_dict[0].keys())}")
    print(f"  info_dict keys: {list(info_dict.keys())}")
    
    # 3) 测试 step
    print("3. 测试 step()...")
    action_dict = {i: torch.randint(0, 6, (1,)).item() for i in range(8)}
    next_obs_dict, rew_head_dict, done, info_dict = adapter.step(action_dict)
    print(f"  rew_head_dict keys: {list(rew_head_dict.keys())}")
    print(f"  rew_head_dict['R'] keys: {list(rew_head_dict['R'].keys())[:3]}...")
    print(f"  rew_head_dict['R'] 单步均值: {sum(rew_head_dict['R'].values()) / len(rew_head_dict['R']):.4f}")
    
    # 4) 测试 VectorRunner.collect()
    print("4. 测试 VectorRunner.collect()...")
    obs_dim = 18  # 真实环境的固定维度
    act_dim = 6
    
    policy = CategoricalPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden=[128, 128],
        share_params=True,
        agent_embed_dim=16
    ).to(device)
    
    critic = MultiHeadCritic(
        obs_dim=obs_dim,
        hidden=[128, 128],
        agent_embed_dim=16
    ).to(device)
    
    runner = VectorRunner(
        backend_adapter=adapter,
        policy=policy,
        critic=critic,
        steps_per_agent=8,
        num_agents=8,
        obs_dim=obs_dim,
        act_dim=act_dim,
        episodes=1,
        max_steps_per_ep=64,
        device=device
    )
    
    rollouts, info_agg = runner.collect()
    
    # 5) 检查输出键和形状
    print("5. 检查输出键和形状...")
    required_keys = ['obs', 'act', 'logp', 'logits', 'rew_dict', 'done', 'agent_id', 'values_dict']
    for key in required_keys:
        assert key in rollouts, f"Missing key: {key}"
        print(f"  {key}: {type(rollouts[key])}")
        if isinstance(rollouts[key], torch.Tensor):
            print(f"    shape: {rollouts[key].shape}")
        elif isinstance(rollouts[key], dict):
            print(f"    keys: {list(rollouts[key].keys())}")
            if 'R' in rollouts[key] and isinstance(rollouts[key]['R'], torch.Tensor):
                print(f"    rew_dict['R'] shape: {rollouts[key]['R'].shape}")
    
    # 验证形状与 Step 2 一致
    total_steps = rollouts['obs'].shape[0]
    assert rollouts['obs'].shape == (total_steps, obs_dim), \
        f"obs shape mismatch: {rollouts['obs'].shape} != ({total_steps}, {obs_dim})"
    assert rollouts['act'].shape == (total_steps,), \
        f"act shape mismatch: {rollouts['act'].shape} != ({total_steps},)"
    assert rollouts['logp'].shape == (total_steps,), \
        f"logp shape mismatch: {rollouts['logp'].shape} != ({total_steps},)"
    assert rollouts['logits'].shape == (total_steps, act_dim), \
        f"logits shape mismatch: {rollouts['logits'].shape} != ({total_steps}, {act_dim})"
    assert rollouts['rew_dict']['R'].shape == (total_steps,), \
        f"rew_dict['R'] shape mismatch: {rollouts['rew_dict']['R'].shape} != ({total_steps},)"
    
    print("\n" + "=" * 60)
    print("Step3 smoke test passed.")
    print("=" * 60)


if __name__ == '__main__':
    main()

