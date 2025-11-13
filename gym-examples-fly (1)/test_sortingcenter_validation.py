"""
验证脚本：测试 SortingCenterEnv 环境是否符合 MOGCRL 论文要求

验证项：
1. 观测结构正确性（obs长度=5，matrix=3×5×5）
2. 真实占用网格渲染（障碍物、充电站、智能体可见）
3. Deadline倒计时逻辑
4. 到达充电站时电池立即充满
5. 默认智能体数量=32
6. 指标输出正确性
"""

import sys
import os

# 添加当前目录到路径，以便导入gym_examples
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gym
import numpy as np

# 导入环境
from gym_examples.envs.sorting_center_env import SortingCenterEnv


def run_validation():
    """运行验证测试"""
    print("=" * 60)
    print("SortingCenterEnv 验证测试")
    print("=" * 60)
    
    # 创建环境（不传agent_num参数，测试默认值）
    print("\n[1] 初始化环境（测试默认智能体数量）...")
    env = SortingCenterEnv(render_mode=None)
    
    # 验证默认智能体数量
    assert env.agent_num == 32, f"默认智能体数量应为32，实际为{env.agent_num}"
    assert env.config['num_agents'] == 32, f"配置中智能体数量应为32，实际为{env.config['num_agents']}"
    print(f"✓ 默认智能体数量: {env.agent_num}")
    
    # 重置环境
    print("\n[2] 重置环境...")
    obs, info = env.reset(seed=42)
    
    # 验证观测结构
    print("\n[3] 验证观测结构...")
    assert 'obs' in obs, "观测中应包含'obs'键"
    assert 'matrix' in obs, "观测中应包含'matrix'键"
    
    obs_shape = obs['obs'].shape
    matrix_shape = obs['matrix'].shape
    
    print(f"  obs shape: {obs_shape}")
    print(f"  matrix shape: {matrix_shape}")
    
    assert obs_shape == (env.agent_num, 5), f"obs形状应为({env.agent_num}, 5)，实际为{obs_shape}"
    assert matrix_shape == (env.agent_num, 3, 5, 5), f"matrix形状应为({env.agent_num}, 3, 5, 5)，实际为{matrix_shape}"
    print("✓ 观测结构正确")
    
    # 验证第一个智能体的观测
    agent0_obs = obs['obs'][0]
    agent0_matrix = obs['matrix'][0]
    
    print(f"\n[4] 第一个智能体观测示例:")
    print(f"  obs (5维): {agent0_obs}")
    print(f"  - x (行): {agent0_obs[0]}")
    print(f"  - y (列): {agent0_obs[1]}")
    print(f"  - tgt_x: {agent0_obs[2]}")
    print(f"  - tgt_y: {agent0_obs[3]}")
    print(f"  - deadline_remaining: {agent0_obs[4]}")
    print(f"  矩阵非零元素数: {np.count_nonzero(agent0_matrix)}")
    
    # 检查矩阵各通道
    channel0_nonzero = np.count_nonzero(agent0_matrix[0])  # 障碍物/货架
    channel1_nonzero = np.count_nonzero(agent0_matrix[1])  # 充电站
    channel2_nonzero = np.count_nonzero(agent0_matrix[2])  # 其他智能体
    
    print(f"  通道0 (障碍物): {channel0_nonzero} 个非零")
    print(f"  通道1 (充电站): {channel1_nonzero} 个非零")
    print(f"  通道2 (其他智能体): {channel2_nonzero} 个非零")
    
    # 获取初始deadline值
    initial_deadline = agent0_obs[4]
    print(f"\n  初始deadline: {initial_deadline}")
    assert initial_deadline == env.config['task_deadline'], \
        f"初始deadline应为{env.config['task_deadline']}，实际为{initial_deadline}"
    
    # 运行episode
    print("\n[5] 运行episode（100步）...")
    total_reward = 0
    previous_deadline = initial_deadline
    recharge_triggered = False
    step_count = 0
    
    for step in range(100):
        # 随机动作
        actions = np.random.randint(0, env.action_space.n, size=env.agent_num)
        
        # 执行一步
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        total_reward += np.mean(rewards)
        step_count += 1
        
        # 检查deadline倒计时
        agent0_obs = obs['obs'][0]
        current_deadline = agent0_obs[4]
        
        # 每20步打印调试信息
        if step % 20 == 0:
            print(f"\n  --- Step {step} ---")
            print(f"  Agent 0 obs: {agent0_obs}")
            print(f"  Deadline remaining: {current_deadline}")
            
            # 检查deadline是否递减
            if step > 0:
                assert current_deadline <= previous_deadline, \
                    f"Deadline应递减，但{previous_deadline} -> {current_deadline}"
                if current_deadline < previous_deadline:
                    print(f"  ✓ Deadline从 {previous_deadline} 递减到 {current_deadline}")
            
            # 检查矩阵
            agent0_matrix = obs['matrix'][0]
            matrix_nonzero = np.count_nonzero(agent0_matrix)
            print(f"  矩阵非零元素数: {matrix_nonzero}")
            
            # 显示矩阵各通道的统计
            ch0 = np.count_nonzero(agent0_matrix[0])
            ch1 = np.count_nonzero(agent0_matrix[1])
            ch2 = np.count_nonzero(agent0_matrix[2])
            print(f"  通道0 (障碍物): {ch0}, 通道1 (充电站): {ch1}, 通道2 (其他智能体): {ch2}")
            
            # 检查智能体位置和充电站
            agent0_pos = env.all_agent['agent_0']['agent_location']
            agent0_battery = env.all_agent['agent_0']['battery']
            print(f"  Agent 0 位置: {agent0_pos}, 电量: {agent0_battery}")
            
            # 检查是否在充电站
            if tuple(agent0_pos) in env.charging_cells:
                print(f"  ⚡ Agent 0 在充电站！")
                # 验证电池是否充满
                assert agent0_battery == env.config['battery_init'], \
                    f"在充电站时电池应充满为{env.config['battery_init']}，实际为{agent0_battery}"
                recharge_triggered = True
                print(f"  ✓ 电池已充满: {agent0_battery}")
        
        # 检查充电逻辑（实时检查，不只在每20步）
        agent0_pos_tuple = tuple(env.all_agent['agent_0']['agent_location'])
        if agent0_pos_tuple in env.charging_cells:
            battery = env.all_agent['agent_0']['battery']
            assert battery == env.config['battery_init'], \
                f"在充电站时电池应充满为{env.config['battery_init']}，实际为{battery}"
        
        previous_deadline = current_deadline
        
        if terminated or truncated:
            print(f"\n  Episode结束于步数 {step}")
            break
    
    print(f"\n[6] Episode总结:")
    print(f"  总步数: {step_count}")
    print(f"  平均奖励: {total_reward / step_count:.3f}")
    print(f"  总奖励: {total_reward:.3f}")
    
    # 验证指标
    print(f"\n[7] 验证指标...")
    print(f"  Throughput: {info.get('throughput', 0)}")
    print(f"  TC-throughput: {info.get('tc_throughput', 0)}")
    print(f"  Battery cost: {info.get('battery_cost', 0)}")
    print(f"  Global step: {info.get('global_step', 0)}")
    
    assert 'throughput' in info, "info中应包含'throughput'"
    assert 'tc_throughput' in info, "info中应包含'tc_throughput'"
    assert 'battery_cost' in info, "info中应包含'battery_cost'"
    print("✓ 所有指标都存在")
    
    # 验证充电是否触发过（至少检查所有智能体）
    print(f"\n[8] 验证充电功能...")
    if recharge_triggered:
        print("✓ 充电功能已验证（在测试过程中检测到充电）")
    else:
        print("⚠ 测试过程中未检测到充电（可能因为随机动作未到达充电站）")
        # 手动移动一个智能体到充电站测试
        print("  手动测试充电功能...")
        test_agent_id = 0
        test_agent = env.all_agent['agent_0']
        original_pos = test_agent['agent_location'][:]
        original_battery = test_agent['battery']
        
        # 找到一个充电站并移动智能体
        if env.charging_cells:
            charging_pos = list(env.charging_cells[0])
            test_agent['agent_location'] = charging_pos
            test_agent['battery'] = 100  # 设置低电量
            
            # 执行一步（应该触发充电）
            actions = np.zeros(env.agent_num, dtype=int)  # 全部停止
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # 验证电池是否充满
            assert test_agent['battery'] == env.config['battery_init'], \
                f"手动测试：在充电站时电池应充满为{env.config['battery_init']}，实际为{test_agent['battery']}"
            print(f"  ✓ 手动测试通过：电池从 {original_battery} 充满到 {test_agent['battery']}")
    
    # 验证占用矩阵的真实性
    print(f"\n[9] 验证占用矩阵的真实性...")
    # 检查是否有障碍物在矩阵中可见
    agent0_pos = env.all_agent['agent_0']['agent_location']
    agent0_matrix = obs['matrix'][0]
    
    # 检查通道0（障碍物）是否有非零值
    if np.count_nonzero(agent0_matrix[0]) > 0:
        print("✓ 占用矩阵通道0（障碍物）包含非零值")
        # 找到非零位置
        nonzero_positions = np.where(agent0_matrix[0] > 0)
        print(f"  障碍物位置（局部坐标）: {list(zip(nonzero_positions[0], nonzero_positions[1]))[:5]}...")
    else:
        print("⚠ 当前智能体周围5×5窗口内无障碍物")
    
    # 检查通道1（充电站）
    if np.count_nonzero(agent0_matrix[1]) > 0:
        print("✓ 占用矩阵通道1（充电站）包含非零值")
    else:
        print("⚠ 当前智能体周围5×5窗口内无充电站")
    
    # 检查通道2（其他智能体）
    if np.count_nonzero(agent0_matrix[2]) > 0:
        print("✓ 占用矩阵通道2（其他智能体）包含非零值")
        nonzero_count = np.count_nonzero(agent0_matrix[2])
        print(f"  检测到 {nonzero_count} 个其他智能体在局部窗口内")
    else:
        print("⚠ 当前智能体周围5×5窗口内无其他智能体")
    
    print("\n" + "=" * 60)
    print("所有验证测试完成！")
    print("=" * 60)
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    try:
        run_validation()
        print("\n✅ 所有测试通过！")
    except AssertionError as e:
        print(f"\n❌ 断言失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise

