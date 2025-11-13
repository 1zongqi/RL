"""
FulfillmentWarehouseEnv 渲染测试脚本

使用方法：
    python test_fulfillment_render.py

功能：
    - 创建 FulfillmentWarehouseEnv 环境
    - 显示渲染窗口
    - 智能体自动执行随机动作
    - 实时显示任务完成情况

按键：
    Ctrl+C - 停止运行并关闭窗口
"""
from rware.fulfillment_warehouse import FulfillmentWarehouseEnv
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='FulfillmentWarehouseEnv 渲染测试')
    parser.add_argument('--num_agents', type=int, default=8,
                        help='智能体数量（默认：8）')
    parser.add_argument('--use_energy', action='store_true',
                        help='启用能量模型')
    parser.add_argument('--speed', type=float, default=0.1,
                        help='渲染速度（秒，默认：0.1，越小越快）')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='最大运行步数（默认：1000）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（默认：None）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FulfillmentWarehouseEnv 渲染测试")
    print("=" * 60)
    
    print("\n正在创建环境...")
    print(f"- 智能体数量: {args.num_agents}")
    print(f"- 能量模型: {'启用' if args.use_energy else '禁用'}")
    print(f"- 渲染速度: {args.speed}秒/步")
    print(f"- 最大步数: {args.max_steps}")
    
    # 创建环境，必须设置 render_mode="human" 才会显示窗口
    env = FulfillmentWarehouseEnv(
        num_agents=args.num_agents,
        use_energy=args.use_energy,
        render_mode="human"  # 重要：设置为 "human" 才会显示窗口
    )
    
    print("\n环境创建成功！")
    print(f"- 网格大小: {env.grid_size}")
    print(f"- 交付点数（货架）: {len(env.delivery_cells)}")
    print(f"- 拾取点数: {len(env.pickup_cells)}")
    print(f"- 充电站数: {len(env.charging_cells)}")
    print("\n" + "-" * 60)
    print("开始运行...（按 Ctrl+C 停止）")
    print("-" * 60)
    print("\n说明：")
    print("  - 黑色方块：货架（交付目标）")
    print("  - 绿色方块：拾取点（智能体生成位置）")
    print("  - 黄色方块：充电站")
    print("  - 橙色圆圈：正常智能体")
    if args.use_energy:
        print("  - 红色圆圈：低电量智能体（需要充电）")
    print("  - 蓝色标记：任务目标的4邻域位置")
    print("\n")
    
    # 重置环境
    obs, info = env.reset(seed=args.seed)
    
    episode = 1
    step_count = 0
    total_throughput = 0
    total_tc_throughput = 0
    total_collisions = 0
    
    # 检查渲染是否可用
    render_available = True
    try:
        env.render()
    except ImportError as e:
        print("\n警告：pyglet 未安装，渲染功能不可用")
        print("提示：运行 'pip install pyglet==1.5.28' 安装渲染依赖")
        render_available = False
    
    try:
        # 循环运行并渲染
        for step in range(args.max_steps):
            # 渲染环境（如果可用）
            if render_available:
                try:
                    env.render()
                except ImportError:
                    render_available = False
                    print("\n渲染功能不可用，继续运行但不显示窗口")
            
            # 延时，方便观察动画
            if render_available:
                time.sleep(args.speed)
            
            # 从动作空间采样动作（随机动作）
            actions = env.action_space.sample()
            
            # 执行动作
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            step_count += 1
            
            # 累计统计信息
            total_throughput += info['throughput'] - total_throughput  # 只计算新增的
            total_tc_throughput += info['tc_throughput'] - total_tc_throughput
            total_collisions += info['collisions'] - total_collisions
            
            # 打印当前状态信息
            if step % 50 == 0:
                print(f"Episode {episode}, Step {step:4d} | "
                      f"Throughput: {info['throughput']:3d} | "
                      f"TC-Throughput: {info['tc_throughput']:3d} | "
                      f"Collisions: {info['collisions']:3d}", end='')
                if args.use_energy:
                    print(f" | Battery Cost: {info.get('battery_cost', 0):3d}")
                else:
                    print()
            
            # 检查任务完成情况
            if info['throughput'] > 0 and step > 0:
                prev_info = env._get_info()
                if info['throughput'] > prev_info.get('throughput', 0):
                    print(f"\n[完成] Step {step}: 任务完成！")
            
            # Episode 结束
            if truncated:
                print(f"\n" + "=" * 60)
                print(f"Episode {episode} 结束")
                print(f"- 总步数: {step_count}")
                print(f"- 完成任务数: {info['throughput']}")
                print(f"- 按时完成任务数: {info['tc_throughput']}")
                print(f"- 碰撞次数: {info['collisions']}")
                if args.use_energy:
                    print(f"- 电池成本: {info.get('battery_cost', 0)}")
                print("=" * 60)
                
                # 重置环境
                obs, info = env.reset(seed=args.seed)
                episode += 1
                step_count = 0
                
                # 重置后稍等
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("用户中断")
        print("=" * 60)
    
    finally:
        # 打印最终统计
        print("\n最终统计：")
        print(f"- 总 Episode 数: {episode}")
        print(f"- 总步数: {step_count}")
        print(f"- 总完成任务数: {info['throughput']}")
        print(f"- 总按时完成任务数: {info['tc_throughput']}")
        print(f"- 总碰撞次数: {info['collisions']}")
        if args.use_energy:
            print(f"- 总电池成本: {info.get('battery_cost', 0)}")
        
        # 关闭环境
        env.close()
        print("\n环境已关闭。")


if __name__ == "__main__":
    main()

