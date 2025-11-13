"""
TRPO-QP消融实验脚本

测试不同超参数对算法性能的影响：
1. cos_sigma扫描（禁用QP vs 启用QP）
2. per_agent_update对比（批级别 vs 逐智能体）
3. delta网格扫描（信赖域大小）
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


def run_ablation_experiment(config_path, param_name, param_value, seed, 
                            iters, xi_time, xi_batt, num_agents, backend, save_dir):
    """运行单次消融实验"""
    exp_name = f'{param_name}_{param_value}_seed{seed}'
    exp_dir = os.path.join(save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 构建命令（这里简化，实际需要修改配置文件）
    # 注：完整实现需要动态修改config或通过命令行传递参数
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
    print(f"消融实验: {param_name}={param_value}, Seed={seed}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=str(project_root))
        print(f"\n实验完成: {exp_name}")
        return exp_dir
    except subprocess.CalledProcessError as e:
        print(f"\n实验失败: {exp_name}, 错误: {e}")
        return None


def plot_ablation_results(results, param_name, save_dir):
    """绘制消融实验结果"""
    plt.figure(figsize=(15, 5))
    
    # 1. V_R曲线
    plt.subplot(1, 3, 1)
    for param_value, data_list in results.items():
        all_curves = []
        for data in data_list:
            if data is not None and 'V_R' in data:
                all_curves.append(data['V_R'].values)
        
        if len(all_curves) > 0:
            min_len = min(len(c) for c in all_curves)
            curves_aligned = np.array([c[:min_len] for c in all_curves])
            mean_curve = curves_aligned.mean(axis=0)
            std_curve = curves_aligned.std(axis=0)
            iters = np.arange(min_len)
            
            plt.plot(iters, mean_curve, label=f'{param_name}={param_value}', linewidth=2)
            plt.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('V_R')
    plt.title(f'Main Reward vs {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. V_time曲线
    plt.subplot(1, 3, 2)
    for param_value, data_list in results.items():
        all_curves = []
        for data in data_list:
            if data is not None and 'V_time' in data:
                all_curves.append(data['V_time'].values)
        
        if len(all_curves) > 0:
            min_len = min(len(c) for c in all_curves)
            curves_aligned = np.array([c[:min_len] for c in all_curves])
            mean_curve = curves_aligned.mean(axis=0)
            std_curve = curves_aligned.std(axis=0)
            iters = np.arange(min_len)
            
            plt.plot(iters, mean_curve, label=f'{param_name}={param_value}', linewidth=2)
            plt.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('V_time')
    plt.title(f'Time Constraint vs {param_name}')
    plt.axhline(y=-0.2, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. V_batt曲线
    plt.subplot(1, 3, 3)
    for param_value, data_list in results.items():
        all_curves = []
        for data in data_list:
            if data is not None and 'V_batt' in data:
                all_curves.append(data['V_batt'].values)
        
        if len(all_curves) > 0:
            min_len = min(len(c) for c in all_curves)
            curves_aligned = np.array([c[:min_len] for c in all_curves])
            mean_curve = curves_aligned.mean(axis=0)
            std_curve = curves_aligned.std(axis=0)
            iters = np.arange(min_len)
            
            plt.plot(iters, mean_curve, label=f'{param_name}={param_value}', linewidth=2)
            plt.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    
    plt.xlabel('Iteration')
    plt.ylabel('V_batt')
    plt.title(f'Battery Constraint vs {param_name}')
    plt.axhline(y=-0.2, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'ablation_{param_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n消融图已保存: {plot_path}")
    plt.close()


def load_metrics(exp_dir):
    """加载实验指标"""
    metrics_path = os.path.join(exp_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None


def main():
    parser = argparse.ArgumentParser(description='TRPO-QP消融实验')
    
    parser.add_argument('--config', type=str, default='configs/mogcrl.yaml',
                       help='配置文件路径')
    parser.add_argument('--ablation_type', type=str, required=True,
                       choices=['cos_sigma', 'per_agent', 'delta'],
                       help='消融实验类型')
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
    parser.add_argument('--save_dir', type=str, default=None,
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 设置保存目录
    if args.save_dir is None:
        args.save_dir = f'results/ablation_{args.ablation_type}'
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 定义不同消融实验的参数值
    if args.ablation_type == 'cos_sigma':
        param_name = 'cos_sigma'
        param_values = [-1.1, 0.2, 0.4, 0.6, 0.8, 1.1]
        print("\n消融实验: 余弦相似度阈值扫描")
        print("  -1.1: 总是电量优先")
        print("  0.2-0.8: 不同融合阈值")
        print("  1.1: 禁用电量优先（总是融合）")
    
    elif args.ablation_type == 'per_agent':
        param_name = 'per_agent_update'
        param_values = [False, True]
        print("\n消融实验: 批级别 vs 逐智能体更新")
    
    elif args.ablation_type == 'delta':
        param_name = 'delta'
        param_values = [0.005, 0.01, 0.015, 0.02]
        print("\n消融实验: 信赖域大小扫描")
    
    print(f"参数值: {param_values}")
    print(f"随机种子: {args.seeds}")
    print(f"迭代次数: {args.iters}")
    print("="*70)
    
    # 运行所有实验
    results = {val: [] for val in param_values}
    
    for param_value in param_values:
        for seed in args.seeds:
            exp_dir = run_ablation_experiment(
                args.config, param_name, param_value, seed,
                args.iters, args.xi_time, args.xi_batt,
                args.num_agents, args.backend, args.save_dir
            )
            
            if exp_dir is not None:
                metrics = load_metrics(exp_dir)
                results[param_value].append(metrics)
            else:
                results[param_value].append(None)
    
    # 生成对比图表
    print("\n" + "="*70)
    print("生成消融实验图表...")
    print("="*70)
    
    plot_ablation_results(results, param_name, args.save_dir)
    
    # 生成汇总表
    summary_data = []
    for param_value in param_values:
        for i, data in enumerate(results[param_value]):
            if data is not None and len(data) > 0:
                last_10 = data.tail(10)
                summary_data.append({
                    param_name: param_value,
                    'seed': args.seeds[i],
                    'final_V_R': last_10['V_R'].mean(),
                    'final_V_time': last_10['V_time'].mean(),
                    'final_V_batt': last_10['V_batt'].mean()
                })
    
    if len(summary_data) > 0:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(args.save_dir, f'ablation_{param_name}_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n汇总表已保存: {summary_path}")
        print("\n汇总统计:")
        print(summary_df.groupby(param_name).mean())
    
    print("\n" + "="*70)
    print("消融实验完成！")
    print(f"结果保存在: {args.save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()



