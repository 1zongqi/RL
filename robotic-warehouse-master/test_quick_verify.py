"""
快速验证 FulfillmentWarehouseEnv 环境
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("FulfillmentWarehouseEnv 快速验证")
print("=" * 60)

try:
    print("\n[1] 测试模块导入...")
    from rware.fulfillment_warehouse import FulfillmentWarehouseEnv
    print("    [OK] 模块导入成功")
    
    print("\n[2] 测试环境创建...")
    env = FulfillmentWarehouseEnv(num_agents=4, use_energy=False)
    print(f"    [OK] 环境创建成功")
    print(f"      - 网格大小: {env.grid_size} (期望: (33, 46))")
    print(f"      - 交付点数: {len(env.delivery_cells)} (期望: 240)")
    print(f"      - 拾取点数: {len(env.pickup_cells)} (期望: 192)")
    print(f"      - 充电站数: {len(env.charging_cells)} (期望: 8)")
    
    # 验证精确计数
    assert env.grid_size == (33, 46), f"网格大小应为(33, 46)"
    assert len(env.delivery_cells) == 240, f"交付点数量应为240"
    assert len(env.pickup_cells) == 192, f"拾取点数量应为192"
    assert len(env.charging_cells) == 8, f"充电站数量应为8"
    
    print("\n[3] 测试环境重置...")
    obs, info = env.reset(seed=0)
    print(f"    [OK] 环境重置成功")
    print(f"      - 观测数量: {len(obs)}")
    print(f"      - 观测形状: {obs[0].shape}")
    print(f"      - 第一个智能体观测: {obs[0]}")
    
    print("\n[4] 测试动作空间...")
    print(f"      - 动作空间: {env.action_space}")
    actions = env.action_space.sample()
    print(f"      - 采样动作: {actions}")
    
    print("\n[5] 测试环境步进...")
    for step in range(5):
        actions = env.action_space.sample()
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"      Step {step+1}: rewards={rewards[:2]}, terminated={terminated}, truncated={truncated}")
        if step == 0:
            print(f"      - 观测形状验证: {obs[0].shape == (10,)}")
            print(f"      - 奖励形状验证: {len(rewards) == env.num_agents}")
    
    print("\n[6] 测试指标追踪...")
    print(f"      - throughput: {info['throughput']}")
    print(f"      - tc_throughput: {info['tc_throughput']}")
    print(f"      - collisions: {info['collisions']}")
    
    print("\n[7] 测试地图验证...")
    delivery_set = set(env.delivery_cells)
    pickup_set = set(env.pickup_cells)
    charger_set = set(env.charging_cells)
    
    assert len(delivery_set & pickup_set) == 0, f"交付点和拾取点重叠: {delivery_set & pickup_set}"
    assert len(delivery_set & charger_set) == 0, f"交付点和充电站重叠: {delivery_set & charger_set}"
    assert len(pickup_set & charger_set) == 0, f"拾取点和充电站重叠: {pickup_set & charger_set}"
    
    # 验证坐标范围
    delivery_x = [c[0] for c in env.delivery_cells]
    delivery_y = [c[1] for c in env.delivery_cells]
    assert min(delivery_x) == 8 and max(delivery_x) == 37, "交付点x坐标应在[8,37]"
    assert min(delivery_y) == 12 and max(delivery_y) == 19, "交付点y坐标应在[12,19]"
    
    print("    [OK] 地图验证通过（集合互不相交，坐标范围正确）")
    
    print("\n[8] 测试任务系统...")
    for agent_id in range(env.num_agents):
        task = env.agent_tasks[agent_id]
        assert task is not None, f"智能体 {agent_id} 应该有任务"
        assert "pickup_pos" in task, "任务应该有拾取点位置"
        assert "delivery_pos" in task, "任务应该有交付点位置"
    print("    [OK] 任务系统验证通过")
    
    print("\n[9] 测试能量模型（启用）...")
    env_energy = FulfillmentWarehouseEnv(num_agents=2, use_energy=True)
    obs, info = env_energy.reset(seed=0)
    initial_battery = env_energy.agent_batteries[0]
    print(f"      - 初始电池: {initial_battery}")
    assert initial_battery == env_energy.START_BATTERY, "初始电池应该是满的"
    
    actions = [env_energy.ACTION_STAY] * env_energy.num_agents
    actions[0] = env_energy.ACTION_UP
    obs, r, term, trunc, info = env_energy.step(actions)
    assert env_energy.agent_batteries[0] < initial_battery, "移动应该消耗电池"
    print(f"      - 移动后电池: {env_energy.agent_batteries[0]}")
    print("    [OK] 能量模型验证通过")
    env_energy.close()
    
    print("\n[10] 测试Gymnasium注册...")
    import rware  # 触发注册
    import gymnasium as gym
    env_gym = gym.make("FulfillmentWarehouse-v0")
    # gym.make 可能返回包装的环境，检查底层环境
    unwrapped = env_gym.unwrapped if hasattr(env_gym, 'unwrapped') else env_gym
    assert isinstance(unwrapped, FulfillmentWarehouseEnv), f"应该能通过gym.make创建环境，实际类型: {type(unwrapped)}"
    obs, info = env_gym.reset(seed=0)
    print(f"      - 默认智能体数量: {unwrapped.num_agents}")
    print("    [OK] Gymnasium注册验证通过")
    env_gym.close()
    
    env.close()
    
    print("\n" + "=" * 60)
    print("[PASS] 所有验证通过！环境运行正常。")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[FAIL] 验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
