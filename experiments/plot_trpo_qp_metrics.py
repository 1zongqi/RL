"""
TRPO-QP专用指标绘图工具

绘制真正的约束值（F_time, F_batt）和价值函数（V_R）
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def moving_avg(x, k):
    """移动平均平滑"""
    if len(x) < k:
        k = max(1, len(x) // 2)
    if k <= 1:
        return x
    weights = np.ones(k) / k
    smoothed = np.convolve(x, weights, mode='valid')
    pad_len = len(x) - len(smoothed)
    pad_values = x[:pad_len]
    return np.concatenate([pad_values, smoothed])


def plot_trpo_qp_metrics(csv_path, out_path, smooth=25):
    """
    绘制TRPO-QP训练指标
    
    包含三个子图：
    1. 价值函数 V_R
    2. 约束值 F_time 和 F_batt（带阈值线）
    3. 方法使用分布（分支统计）
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    print(f"读取CSV: {csv_path}")
    print(f"数据行数: {len(df)}")
    
    # 提取数据
    x = df['iter'].values if 'iter' in df.columns else np.arange(len(df))
    V_R = df['V_R'].values
    F_time = df['F_time'].values
    F_batt = df['F_batt'].values
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('TRPO-QP Training Metrics', fontsize=16, fontweight='bold')
    
    # ========== 子图1: 价值函数 V_R ==========
    ax1 = axes[0]
    V_R_smooth = moving_avg(V_R, smooth)
    
    ax1.plot(x, V_R, alpha=0.3, color='blue', linewidth=0.5, label='V_R (raw)')
    ax1.plot(x, V_R_smooth, color='blue', linewidth=2, label=f'V_R (smooth={smooth})')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Value Function (V_R)', fontsize=12)
    ax1.set_title('Reward Value Function', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ========== 子图2: 约束值 F_time 和 F_batt ==========
    ax2 = axes[1]
    F_time_smooth = moving_avg(F_time, smooth)
    F_batt_smooth = moving_avg(F_batt, smooth)
    
    # 获取阈值
    xi_time = df['xi_time'].values[0] if 'xi_time' in df.columns else -0.2
    xi_batt = df['xi_batt'].values[0] if 'xi_batt' in df.columns else -0.2
    
    # 绘制F_time
    ax2.plot(x, F_time, alpha=0.2, color='orange', linewidth=0.5)
    ax2.plot(x, F_time_smooth, color='orange', linewidth=2, label=f'F_time (smooth)')
    ax2.axhline(y=xi_time, color='orange', linestyle='--', linewidth=2, 
                label=f'ξ_time={xi_time}', alpha=0.7)
    
    # 绘制F_batt
    ax2.plot(x, F_batt, alpha=0.2, color='green', linewidth=0.5)
    ax2.plot(x, F_batt_smooth, color='green', linewidth=2, label=f'F_batt (smooth)')
    ax2.axhline(y=xi_batt, color='green', linestyle='--', linewidth=2, 
                label=f'ξ_batt={xi_batt}', alpha=0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Constraint Value', fontsize=12)
    ax2.set_title('Constraint Values (F_time, F_batt) vs Thresholds', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # ========== 子图3: 方法使用分布 ==========
    ax3 = axes[2]
    
    # 统计每种方法的使用次数
    method_counts = df['method'].value_counts()
    methods = method_counts.index.tolist()
    counts = method_counts.values.tolist()
    
    # 颜色映射
    color_map = {
        'reward': '#4CAF50',
        'time_only': '#FF9800',
        'batt_only': '#2196F3',
        'batt_priority': '#9C27B0',
        'qp_fusion_success': '#F44336'
    }
    
    colors = [color_map.get(m, '#757575') for m in methods]
    
    bars = ax3.bar(methods, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Fusion Method', fontsize=12)
    ax3.set_ylabel('Usage Count', fontsize=12)
    ax3.set_title('Gradient Fusion Method Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {out_path}")
    
    # 打印统计信息
    print(f"\n方法使用统计:")
    for method, count in zip(methods, counts):
        print(f"  {method}: {count} 次 ({count/len(df)*100:.1f}%)")
    
    print(f"\n约束值范围:")
    print(f"  F_time: [{F_time.min():.4f}, {F_time.max():.4f}]")
    print(f"  F_batt: [{F_batt.min():.4f}, {F_batt.max():.4f}]")
    print(f"\n约束阈值:")
    print(f"  ξ_time: {xi_time}")
    print(f"  ξ_batt: {xi_batt}")


def main():
    parser = argparse.ArgumentParser(description='Plot TRPO-QP training metrics')
    parser.add_argument('--csv', type=str, required=True, help='CSV file path')
    parser.add_argument('--out', type=str, required=True, help='Output PNG path')
    parser.add_argument('--smooth', type=int, default=25, help='Smoothing window')
    
    args = parser.parse_args()
    
    plot_trpo_qp_metrics(args.csv, args.out, args.smooth)
    print("\n绘图完成！")


if __name__ == '__main__':
    main()


