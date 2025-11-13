"""
TRPO-QP 多种子训练汇总脚本

运行多个seed的TRPO-QP训练，生成指标汇总与图表。
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_single_experiment(seed, config_path, iters, xi_time, xi_batt,
                          num_agents, backend, save_dir):
    """运行单次实验"""
    exp_dir = os.path.join(save_dir, f'trpo_qp_seed{seed}')
    os.makedirs(exp_dir, exist_ok=True)
    
    cmd = [
        'python', 'experiments/train_mogcrl_trpo_qp.py',
        '--config', config_path,
        '--seed', str(seed),
        '--iters', str(iters),
        '--xi_time', str(xi_time),
        '--xi_batt', str(xi_batt),
        '--num_agents', str(num_agents),
        '--backend', backend,
        '--save_dir', exp_dir
    ]
    
    print(f"\n{'='*70}")
    print(f"运行实验: TRPO-QP, Seed={seed}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run(cmd, check=True, cwd=str(project_root))
        print(f"\n实验完成: TRPO-QP, seed={seed}")
        return exp_dir
    except subprocess.CalledProcessError as e:
        print(f"\n实验失败: TRPO-QP, seed={seed}, 错误: {e}")
        return None


def load_metrics(exp_dir):
    """加载实验指标"""
    metrics_path = os.path.join(exp_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    else:
        print(f"警告: 未找到指标文件 {metrics_path}")
        return None


def plot_comparison(results, save_dir):
    """生成图表"""

    plt.figure(figsize=(12, 5))

    # 1. V_R曲线
    plt.subplot(1, 2, 1)
    all_curves = []
    for seed_data in results.get('trpo_qp', []):
        if seed_data is not None and 'V_R' in seed_data:
            all_curves.append(seed_data['V_R'].values)

    if all_curves:
        min_len = min(len(c) for c in all_curves)
        curves_aligned = np.array([c[:min_len] for c in all_curves])
        mean_curve = curves_aligned.mean(axis=0)
        std_curve = curves_aligned.std(axis=0)
        iters = np.arange(min_len)

        plt.plot(iters, mean_curve, label='TRPO-QP', linewidth=2)
        plt.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)

    plt.xlabel('Iteration')
    plt.ylabel('V_R (Main Reward)')
    plt.title('Main Reward (TRPO-QP)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 约束违反率曲线
    plt.subplot(1, 2, 2)
    all_viol = []
    for seed_data in results.get('trpo_qp', []):
        if seed_data is not None and 'V_time' in seed_data and 'V_batt' in seed_data:
            viol = ((seed_data['V_time'] < -0.2).astype(float) +
                    (seed_data['V_batt'] < -0.2).astype(float)) / 2.0
            all_viol.append(viol.values)

    if all_viol:
        min_len = min(len(v) for v in all_viol)
        viol_aligned = np.array([v[:min_len] for v in all_viol])
        mean_viol = viol_aligned.mean(axis=0)
        std_viol = viol_aligned.std(axis=0)
        iters = np.arange(min_len)

        plt.plot(iters, mean_viol, label='TRPO-QP', linewidth=2)
        plt.fill_between(iters, mean_viol - std_viol, mean_viol + std_viol, alpha=0.3)

    plt.xlabel('Iteration')
    plt.ylabel('Violation Rate')
    plt.title('Constraint Violation Rate (TRPO-QP)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'trpo_qp_training_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n训练总结图已保存: {plot_path}")
    plt.close()

    # 3. 生成汇总表
    summary_data = []
    for idx, seed_data in enumerate(results.get('trpo_qp', [])):
        if seed_data is not None and len(seed_data) > 0:
            last_10 = seed_data.tail(10)
            violation_rate = (
                ((last_10['V_time'] < -0.2).astype(float) +
                 (last_10['V_batt'] < -0.2).astype(float)) / 2.0
            ).mean()
            summary_data.append({
                'Seed': idx,
                'V_R_mean': last_10['V_R'].mean(),
                'V_time_mean': last_10['V_time'].mean(),
                'V_batt_mean': last_10['V_batt'].mean(),
                'Violation_rate': violation_rate
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(save_dir, 'trpo_qp_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"汇总表已保存: {summary_path}")
        print("\n汇总统计:")
        print(summary_df.mean(numeric_only=True))


def main():
    parser = argparse.ArgumentParser(description='TRPO-QP多种子汇总实验')
    
    parser.add_argument('--config', type=str, default='configs/mogcrl.yaml',
                       help='配置文件路径')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                       help='随机种子列表')
    parser.add_argument('--iters', type=int, default=200,
                       help='训练迭代次数')
    parser.add_argument('--xi_time', type=float, default=-0.2,
                       help='时效约束阈值')
    parser.add_argument('--xi_batt', type=float, default=-0.2,
                       help='电量约束阈值')
    parser.add_argument('--num_agents', type=int, default=2,
                       help='智能体数量')
    parser.add_argument('--backend', type=str, default='mock',
                       choices=['mock', 'fulfillment', 'rware'],
                       help='环境后端')
    parser.add_argument('--save_dir', type=str, default='results/trpo_qp_summary',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TRPO-QP多种子汇总实验")
    print(f"随机种子: {args.seeds}")
    print(f"迭代次数: {args.iters}")
    print(f"约束阈值: xi_time={args.xi_time}, xi_batt={args.xi_batt}")
    print(f"环境: {args.backend}, 智能体数: {args.num_agents}")
    print("="*70)
    
    # 运行所有实验
    results = {'trpo_qp': []}

    for seed in args.seeds:
        exp_dir = run_single_experiment(
            seed, args.config, args.iters,
            args.xi_time, args.xi_batt, args.num_agents,
            args.backend, args.save_dir
        )

        if exp_dir is not None:
            metrics = load_metrics(exp_dir)
            results['trpo_qp'].append(metrics)
        else:
            results['trpo_qp'].append(None)
    
    # 生成对比图表
    print("\n" + "="*70)
    print("生成训练总结图表...")
    print("="*70)
    
    plot_comparison(results, args.save_dir)
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"结果保存在: {args.save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()



