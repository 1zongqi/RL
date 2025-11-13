"""
Training Reward and Cost Curve Plotting Tool

Read per-episode reward and cost data from training-generated CSV files and plot dual-axis curves
Supports fallback data column strategies and English labels
"""

import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# Font configuration (English labels, no Chinese fonts needed)
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False


def moving_avg(x, k):
    """
    移动平均平滑
    
    参数:
        x: 输入数组
        k: 窗口大小
    
    返回:
        平滑后的数组
    """
    if len(x) < k:
        k = max(1, len(x) // 2)
    
    if k <= 1:
        return x
    
    # 使用卷积实现移动平均
    weights = np.ones(k) / k
    smoothed = np.convolve(x, weights, mode='valid')
    
    # 填充开头部分（使用原始值）
    pad_len = len(x) - len(smoothed)
    pad_values = x[:pad_len]
    
    return np.concatenate([pad_values, smoothed])


def find_latest_metrics_csv():
    """
    自动搜索最新的metrics.csv或episode_metrics.csv文件
    
    返回:
        最新CSV文件的路径，如果未找到则返回None
    """
    # 搜索patterns
    patterns = [
        'results/**/episode_metrics.csv',
        'results/**/metrics.csv',
        '**/episode_metrics.csv',
        '**/metrics.csv'
    ]
    
    all_csvs = []
    for pattern in patterns:
        all_csvs.extend(glob.glob(pattern, recursive=True))
    
    if not all_csvs:
        return None
    
    # 按修改时间排序，返回最新的
    latest_csv = max(all_csvs, key=lambda p: os.path.getmtime(p))
    return latest_csv


class MissingColumnError(ValueError):
    """Raised when reward or cost columns are unavailable."""


def extract_reward_cost(
    df,
    reward_key=None,
    cost_key=None,
    scale_mode='total',
    max_steps_per_episode=None,
    default_episode_steps=400,
):
    """
    从DataFrame提取奖励和成本列，支持兜底策略
    
    奖励优先级：avg_reward_per_agent → reward_per_agent → avg_reward → V_R → throughput
    成本优先级：avg_cost_per_agent → cost_per_agent → avg_cost → (viol_time_rate+viol_batt_rate)*20 → deadline_violation_rate*40
    
    参数:
        df: pandas DataFrame
    
    返回:
        x: 横轴数据（episode或iter或索引）
        reward: 奖励数据
        cost: 成本数据
        x_label: 横轴标签
        reward_label: 奖励标签
        cost_label: 成本标签
    """
    # 提取横轴（优先episode，否则iter，否则索引）
    if 'episode' in df.columns:
        x = df['episode'].values
        x_label = 'Episode'
    elif 'iter' in df.columns:
        x = df['iter'].values
        x_label = 'Iteration'
    else:
        x = np.arange(len(df))
        x_label = 'Index'
    
    # 提取奖励列（按优先级或命令行指定）
    reward = None
    reward_label = 'Average reward per agent per episode'

    steps_per_episode = None
    if 'episode_length' in df.columns:
        steps_per_episode = df['episode_length'].astype(float).values
    elif 'episode_length_avg' in df.columns:
        steps_per_episode = df['episode_length_avg'].astype(float).values
    elif 'episode_steps_total' in df.columns:
        episodes_col = df['episodes'].astype(float).values if 'episodes' in df.columns else np.ones(len(df))
        episodes_col = np.clip(episodes_col, 1e-6, None)
        steps_per_episode = df['episode_steps_total'].astype(float).values / episodes_col
    elif max_steps_per_episode is not None:
        steps_per_episode = np.full(len(df), float(max_steps_per_episode))
    elif 'total_agent_steps' in df.columns and 'debug_num_agents' in df.columns:
        agents = np.clip(df['debug_num_agents'].astype(float).values, 1e-6, None)
        steps_per_episode = df['total_agent_steps'].astype(float).values / agents
    else:
        steps_per_episode = np.full(len(df), float(default_episode_steps))

    has_rollout_length = 'debug_rollout_length' in df.columns

    total_reward_possible = (
        'debug_raw_reward_sum' in df.columns
        or (
            'avg_reward_per_agent' in df.columns
            and steps_per_episode is not None
        )
    )

    total_cost_possible = (
        'cost_sum_per_agent' in df.columns and steps_per_episode is not None
    ) or (
        {'avg_cost_batt_per_agent', 'avg_cost_time_per_agent'}.issubset(df.columns)
        and steps_per_episode is not None
    )

    total_mode_enabled = (scale_mode == 'total') or (scale_mode == 'auto' and total_reward_possible and total_cost_possible)

    if reward_key:
        if reward_key not in df.columns:
            raise MissingColumnError(f"指定的奖励列 '{reward_key}' 不存在，当前可用列: {df.columns.tolist()}")
        reward = df[reward_key].values
        reward_label = reward_key
        print(f"Using reward column: {reward_key}")
    elif total_mode_enabled and total_reward_possible:
        if 'debug_raw_reward_sum' in df.columns:
            reward = df['debug_raw_reward_sum'].values
            reward_label = 'Total reward per episode'
            print("Using reward column: debug_raw_reward_sum (total reward)")
        else:
            reward = df['avg_reward_per_agent'].values * steps_per_episode
            reward_label = 'Total reward per episode (avg_reward_per_agent × episode_length)'
            print("Using reward column: avg_reward_per_agent * episode_length (total reward)")
    else:
        reward_priorities = [
            ('avg_reward_per_agent', 'Average reward per agent per episode'),
            ('reward_per_agent', 'Reward per agent'),
            ('avg_reward', 'Average reward'),
            ('V_R', 'Value function (V_R)'),
            ('throughput', 'Throughput')
        ]

        for col_name, label in reward_priorities:
            if col_name in df.columns:
                reward = df[col_name].values
                reward_label = label
                print(f"Using reward column: {col_name}")
                break

    if reward is None:
        raise MissingColumnError(f"缺少奖励相关列，当前可用列: {df.columns.tolist()}")
    
    # 提取成本列（按优先级）
    cost = None
    cost_label = 'Average agent-level constraint cost'

    has_cost_time = 'cost_time' in df.columns
    has_cost_batt = 'cost_batt' in df.columns

    if has_cost_time or has_cost_batt:
        time_vals = df['cost_time'].values if has_cost_time else np.zeros(len(df))
        batt_vals = df['cost_batt'].values if has_cost_batt else np.zeros(len(df))

        time_scale = 20.0
        batt_scale = 40.0

        cost = time_vals * time_scale + batt_vals * batt_scale

        if has_cost_time and has_cost_batt:
            print(f"Using cost columns: cost_time*{time_scale} + cost_batt*{batt_scale}")
        elif has_cost_time:
            print(f"Using cost column: cost_time*{time_scale}")
        else:
            print(f"Using cost column: cost_batt*{batt_scale}")

    
    if cost_key:
        if cost_key not in df.columns:
            raise MissingColumnError(f"指定的成本列 '{cost_key}' 不存在，当前可用列: {df.columns.tolist()}")
        cost = df[cost_key].values
        cost_label = cost_key
        print(f"Using cost column: {cost_key}")
    elif total_mode_enabled and total_cost_possible:
        if 'cost_sum_per_agent' in df.columns:
            cost = df['cost_sum_per_agent'].values * steps_per_episode
            cost_label = 'Total constraint cost per episode'
            print("Using constraint cost: cost_sum_per_agent * episode_length (total cost)")
        else:
            per_agent_cost = df['avg_cost_batt_per_agent'].values + df['avg_cost_time_per_agent'].values
            cost = per_agent_cost * steps_per_episode
            cost_label = 'Total constraint cost per episode'
            print("Using constraint cost: (avg_cost_time_per_agent + avg_cost_batt_per_agent) * episode_length (total cost)")
    else:
        cost_priorities = [
            ('avg_cost_per_agent', 'Average agent-level constraint cost'),
            ('cost_per_agent', 'Average agent-level constraint cost'),
            ('avg_cost', 'Average agent-level constraint cost')
        ]

        for col_name, label in cost_priorities:
            if col_name in df.columns:
                cost = df[col_name].values
                cost_label = label
                print(f"Using cost column: {col_name}")
                break
    
    # 如果没有找到成本列，使用组合策略
    if cost is None:
        if 'viol_time_rate' in df.columns and 'viol_batt_rate' in df.columns:
            cost = (df['viol_time_rate'] + df['viol_batt_rate']).values * 20.0
            cost_label = 'Combined Violation Cost'
            print("Using combined cost: (viol_time_rate + viol_batt_rate) * 20")
        elif 'deadline_violation_rate' in df.columns:
            cost = df['deadline_violation_rate'].values * 40.0
            cost_label = 'Deadline Violation Cost'
            print("Using cost: deadline_violation_rate * 40")
        else:
            raise MissingColumnError(f"缺少成本相关列，当前可用列: {df.columns.tolist()}")
    
    return x, reward, cost, x_label, reward_label, cost_label


def plot_reward_cost(x, reward, cost, save_path, smooth=25, x_label='Episode',
                      reward_label='Average reward per agent per episode',
                      cost_label='Average agent-level constraint cost'):
    """
    绘制双轴曲线图（左轴奖励，右轴成本）
    
    参数:
        x: 横轴数据
        reward: 奖励数据
        cost: 成本数据
        save_path: 输出PNG路径
        smooth: 移动平均窗口大小
        x_label: 横轴标签
        reward_label: 奖励标签
        cost_label: 成本标签
    """
    # 平滑数据
    if smooth > 1:
        reward_smooth = moving_avg(reward, smooth)
        cost_smooth = moving_avg(cost, smooth)
    else:
        reward_smooth = reward
        cost_smooth = cost
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左轴：奖励（蓝线）
    color_reward = 'tab:blue'
    ax1.set_xlabel(x_label, fontsize=13)
    ax1.set_ylabel(reward_label, color=color_reward, fontsize=12)
    line1 = ax1.plot(x, reward_smooth, color=color_reward, linewidth=2.2, 
                     label=reward_label, alpha=0.9)
    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.grid(True, alpha=0.25, linestyle='--')
    
    # 右轴：成本（红线）
    ax2 = ax1.twinx()
    color_cost = 'red'
    ax2.set_ylabel(cost_label, color=color_cost, fontsize=12)
    line2 = ax2.plot(x, cost_smooth, color=color_cost, linewidth=2.0,
                     label=cost_label, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color_cost)
    
    # 标题
    plt.title('Training Convergence: Reward vs Cost', fontsize=15, fontweight='bold', pad=15)
    
    # 图例（合并两条线）
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)
    
    # 紧凑布局
    fig.tight_layout()
    
    # 保存
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"\nChart saved to: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot training reward and cost curves',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--csv', type=str, default=None,
                       help='CSV file path (optional, auto-search results/**/episode_metrics.csv or metrics.csv if not specified)')
    parser.add_argument('--out', type=str, default='results/reward_cost_curve.png',
                       help='Output PNG file path')
    parser.add_argument('--smooth', type=int, default=25,
                       help='Moving average window size for smoothing curves')
    parser.add_argument('--reward-key', type=str, default=None,
                       help='Column name to use as reward (default: auto-select total metrics when available)')
    parser.add_argument('--cost-key', type=str, default=None,
                       help='Column name to use as cost (default: auto-select total metrics when available)')
    parser.add_argument('--scale-mode', type=str, choices=['auto', 'per_agent', 'total'], default='total',
                       help="Scaling mode for reward/cost. 'total' (default) prefers unnormalized totals when columns exist.")
    parser.add_argument('--episode-steps', type=float, default=None,
                       help='Override episode length when logs do not contain episode_length columns.')
    
    args = parser.parse_args()
    
    # 确定CSV路径
    if args.csv is None:
        print("CSV file not specified, searching automatically...")
        csv_path = find_latest_metrics_csv()
        if csv_path is None:
            print("Error: Cannot find metrics.csv or episode_metrics.csv")
            print("Please use --csv to specify the CSV file path")
            sys.exit(1)
        print(f"Found CSV file: {csv_path}")
    else:
        csv_path = args.csv
        if not os.path.exists(csv_path):
            print(f"Error: CSV file does not exist: {csv_path}")
            sys.exit(1)
    
    # 读取CSV
    print(f"\nReading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read {len(df)} rows")
        print(f"Available columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error: Failed to read CSV: {e}")
        sys.exit(1)
    
    # 提取奖励和成本数据
    reward_key = args.reward_key if args.reward_key not in (None, '') else None
    cost_key = args.cost_key if args.cost_key not in (None, '') else None
    scale_mode = args.scale_mode
    episode_steps_override = args.episode_steps
    default_episode_steps = episode_steps_override if episode_steps_override is not None else 400

    fallback_attempted = False
    while True:
        try:
            x, reward, cost, x_label, reward_label, cost_label = extract_reward_cost(
                df, reward_key=reward_key, cost_key=cost_key,
                scale_mode=scale_mode,
                max_steps_per_episode=episode_steps_override,
                default_episode_steps=default_episode_steps
            )
            print(f"\nData extraction successful:")
            print(f"  X-axis: {x_label}, range [{x.min()}, {x.max()}]")
            print(f"  Reward: {reward_label}, range [{reward.min():.4f}, {reward.max():.4f}]")
            print(f"  Cost: {cost_label}, range [{cost.min():.4f}, {cost.max():.4f}]")
            break
        except MissingColumnError as e:
            if not fallback_attempted:
                candidate = Path(csv_path).with_name('episode_metrics.csv')
                if candidate.exists() and candidate.resolve() != Path(csv_path).resolve():
                    print(f"Warning: {e}")
                    print(f"Attempting to use episode metrics at: {candidate}")
                    csv_path = str(candidate)
                    print(f"\nReading CSV: {csv_path}")
                    try:
                        df = pd.read_csv(csv_path)
                        print(f"Successfully read {len(df)} rows")
                        print(f"Available columns: {df.columns.tolist()}")
                    except Exception as read_err:
                        print(f"Error: Failed to read fallback CSV: {read_err}")
                        sys.exit(1)
                    fallback_attempted = True
                    # Reset keys for automatic selection on fallback
                    reward_key = None
                    cost_key = None
                    if scale_mode == 'total':
                        scale_mode = 'auto'
                    continue
            if not fallback_attempted and scale_mode != 'per_agent':
                print(f"Warning: {e}")
                print("Falling back to per-agent scaled metrics...")
                reward_key = None
                cost_key = None
                scale_mode = 'per_agent'
                fallback_attempted = True
                continue
            print(f"Error: Data extraction failed: {e}")
            print("Hint: 指定 --reward-key / --cost-key 或确保训练生成 episode_metrics.csv。")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Data extraction failed: {e}")
            sys.exit(1)
    
    # 绘制曲线
    print(f"\nPlotting curves (smoothing window={args.smooth})...")
    try:
        plot_reward_cost(x, reward, cost, args.out, args.smooth,
                        x_label, reward_label, cost_label)
        print("\nPlotting complete!")
    except Exception as e:
        print(f"Error: Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

