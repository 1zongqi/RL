"""
阈值扫描与消融实验工具

实现网格遍历 (xi_time, xi_batt)，调用训练接口执行短训练/评估
支持两种消融模式：A) 去融合（双违恒用电量优先）；B) 去电量优先（双违恒用融合）
输出 CSV（含分支/违反率/三指标/KL等）+ 若干热力图/线图
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.train_mogcrl_paper import run_training_once
from mogcrl.utils.io_utils import ensure_dir, append_csv
from mogcrl.utils.plotting import lineplot_from_csv, heatmap_from_csv


def parse_float_list(s: str) -> List[float]:
    """
    解析逗号分隔的浮点数列表
    
    参数:
        s: 逗号分隔的字符串，如 "-0.6,-0.4,-0.2,0.0,0.2"
    
    返回:
        浮点数列表
    """
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def parse_ablation_list(s: str) -> List[str]:
    """
    解析逗号分隔的消融模式列表
    
    参数:
        s: 逗号分隔的字符串，如 "none,no_fuse,no_batt_priority"
    
    返回:
        消融模式列表
    """
    return [x.strip() for x in s.split(',') if x.strip()]


def filter_csv_and_plot(
    csv_path: str,
    x: str,
    y: str,
    hue: str,
    filter_dict: Dict[str, Any],
    out_png: str,
    title: str = ""
) -> None:
    """
    筛选 CSV 数据并绘制线图
    
    参数:
        csv_path: CSV 文件路径
        x: X 轴字段名
        y: Y 轴字段名
        hue: 分组字段名（可选）
        filter_dict: 筛选条件字典
        out_png: 输出 PNG 文件路径
        title: 图表标题
    """
    import csv
    
    # 读取并筛选数据
    filtered_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            match = True
            for key, value in filter_dict.items():
                if key not in row:
                    match = False
                    break
                # 尝试数值比较
                try:
                    if isinstance(value, (int, float)):
                        if abs(float(row[key]) - value) > 1e-6:
                            match = False
                            break
                    else:
                        if str(row[key]) != str(value):
                            match = False
                            break
                except (ValueError, TypeError):
                    if str(row[key]) != str(value):
                        match = False
                        break
            if match:
                filtered_data.append(row)
    
    if len(filtered_data) == 0:
        print(f"警告: 筛选后没有数据，跳过绘图: {out_png}")
        return
    
    # 创建临时 CSV 文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8') as tmp_file:
        writer = csv.DictWriter(tmp_file, fieldnames=filtered_data[0].keys())
        writer.writeheader()
        writer.writerows(filtered_data)
        tmp_csv_path = tmp_file.name
    
    try:
        # 调用 lineplot_from_csv
        lineplot_from_csv(
            csv_path=tmp_csv_path,
            x=x,
            y=y,
            hue=hue if hue else None,
            out_png=out_png,
            title=title
        )
    finally:
        # 删除临时文件
        os.unlink(tmp_csv_path)


def main():
    parser = argparse.ArgumentParser(description='阈值扫描与消融实验工具')
    
    # 环境参数
    parser.add_argument('--backend', type=str, default='fulfillment',
                       choices=['fulfillment', 'rware', 'stub'],
                       help='环境后端')
    parser.add_argument('--episodes', type=int, default=8,
                       help='每轮采样的 episode 数')
    parser.add_argument('--max_steps_per_ep', type=int, default=256,
                       help='每个 episode 的最大步数')
    parser.add_argument('--steps_per_agent', type=int, default=128,
                       help='每个智能体的采样步数')
    parser.add_argument('--num_agents', type=int, default=8,
                       help='智能体数量')
    
    # 训练参数
    parser.add_argument('--iters', type=int, default=30,
                       help='每个配置的训练轮数（短训练）')
    parser.add_argument('--warmup', type=int, default=10,
                       help='前几轮不计指标（可选）')
    parser.add_argument('--repeat', type=int, default=2,
                       help='每组重复次数，取均值')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='折扣因子')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                       help='GAE lambda 参数')
    parser.add_argument('--lr_actor', type=float, default=5e-4,
                       help='Actor 学习率')
    parser.add_argument('--lr_critic', type=float, default=6e-4,
                       help='Critic 学习率')
    parser.add_argument('--clip_eps', type=float, default=0.20,
                       help='PPO clip 阈值')
    parser.add_argument('--kl_target', type=float, default=0.04,
                       help='KL 散度目标')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                       help='熵正则化系数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--update_epochs', type=int, default=6,
                       help='每轮更新的 epoch 数')
    
    # MOGCRL 参数
    parser.add_argument('--tau', type=float, default=0.6,
                       help='余弦相似度阈值')
    parser.add_argument('--scan_time', type=str, default="-0.6,-0.4,-0.2,0.0,0.2",
                       help='xi_time 扫描范围（逗号分隔）')
    parser.add_argument('--scan_batt', type=str, default="-0.6,-0.4,-0.2,0.0,0.2",
                       help='xi_batt 扫描范围（逗号分隔）')
    parser.add_argument('--v_weight_R', type=float, default=1.0,
                       help='Critic R 头权重')
    parser.add_argument('--v_weight_T', type=float, default=0.7,
                       help='Critic time 头权重')
    parser.add_argument('--v_weight_B', type=float, default=0.7,
                       help='Critic batt 头权重')
    
    # 消融参数
    parser.add_argument('--ablation', type=str, default="none,no_fuse,no_batt_priority",
                       help='消融模式列表（逗号分隔）：none,no_fuse,no_batt_priority')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='results/ablation_scan',
                       help='保存目录')
    parser.add_argument('--use_paper_trainer', action='store_true',
                       help='使用论文同构训练器（已默认使用）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 解析扫描范围
    xi_time_list = parse_float_list(args.scan_time)
    xi_batt_list = parse_float_list(args.scan_batt)
    ablation_list = parse_ablation_list(args.ablation)
    
    # 创建保存目录
    ensure_dir(args.save_dir)
    csv_path = os.path.join(args.save_dir, 'scan_results.csv')
    
    # CSV 字段顺序（固定）
    field_order = [
        "ablation", "xi_time", "xi_batt", "tau",
        "KL", "clip_frac", "entropy", "critic_loss",
        "branch_R", "branch_T", "branch_B", "branch_FUSE",
        "cos_mean", "cos_p50", "cos_p75",
        "viol_time_rate", "viol_batt_rate",
        "throughput", "deadline_violation_rate", "min_SOC", "completed_tasks"
    ]
    
    # 计算总任务数
    total_tasks = len(ablation_list) * len(xi_time_list) * len(xi_batt_list) * args.repeat
    current_task = 0
    
    print("=" * 60)
    print("开始阈值扫描与消融实验")
    print(f"消融模式: {ablation_list}")
    print(f"xi_time 范围: {xi_time_list}")
    print(f"xi_batt 范围: {xi_batt_list}")
    print(f"tau: {args.tau}")
    print(f"每个配置训练轮数: {args.iters}")
    print(f"重复次数: {args.repeat}")
    print(f"总任务数: {total_tasks}")
    print("=" * 60)
    
    # 存储 Pareto 候选（throughput 高 & deadline_violation_rate 低）
    pareto_candidates = []
    
    # 遍历所有组合
    for ablation in ablation_list:
        for xi_time in xi_time_list:
            for xi_batt in xi_batt_list:
                # 重复实验
                all_metrics = []
                
                for repeat_idx in range(args.repeat):
                    current_task += 1
                    print(f"\n[{current_task}/{total_tasks}] ablation={ablation}, "
                          f"xi_time={xi_time:.2f}, xi_batt={xi_batt:.2f}, "
                          f"repeat={repeat_idx+1}/{args.repeat}")
                    
                    # 组装配置
                    config = {
                        'backend': args.backend,
                        'episodes': args.episodes,
                        'max_steps_per_ep': args.max_steps_per_ep,
                        'steps_per_agent': args.steps_per_agent,
                        'num_agents': args.num_agents,
                        'iters': args.iters,
                        'gamma': args.gamma,
                        'gae_lambda': args.gae_lambda,
                        'lr_actor': args.lr_actor,
                        'lr_critic': args.lr_critic,
                        'clip_eps': args.clip_eps,
                        'kl_target': args.kl_target,
                        'ent_coef': args.ent_coef,
                        'batch_size': args.batch_size,
                        'update_epochs': args.update_epochs,
                        'tau': args.tau,
                        'xi_time': xi_time,
                        'xi_batt': xi_batt,
                        'v_weight_R': args.v_weight_R,
                        'v_weight_T': args.v_weight_T,
                        'v_weight_B': args.v_weight_B,
                        'ablation': ablation,
                        'seed': args.seed + repeat_idx,  # 不同重复使用不同种子
                        'device': args.device
                    }
                    
                    # 调用训练接口
                    try:
                        metrics = run_training_once(config)
                        if metrics:
                            all_metrics.append(metrics)
                    except Exception as e:
                        print(f"  错误: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # 计算均值
                if len(all_metrics) > 0:
                    # 聚合所有指标
                    avg_metrics = {}
                    for key in all_metrics[0].keys():
                        values = [m.get(key, 0.0) for m in all_metrics if key in m]
                        if len(values) > 0:
                            # 跳过字符串类型字段（如 'ablation'）
                            if isinstance(values[0], str):
                                avg_metrics[key] = values[0]  # 字符串字段取第一个值
                            else:
                                # 处理 NaN（仅对数值类型）
                                values = [v if not np.isnan(v) else 0.0 for v in values]
                                avg_metrics[key] = np.mean(values)
                    
                    # 构建 CSV 行
                    row = {
                        'ablation': ablation,
                        'xi_time': xi_time,
                        'xi_batt': xi_batt,
                        'tau': args.tau,
                        'KL': avg_metrics.get('kl', 0.0),
                        'clip_frac': avg_metrics.get('clip_frac', 0.0),
                        'entropy': avg_metrics.get('entropy', 0.0),
                        'critic_loss': avg_metrics.get('critic_loss', 0.0),
                        'branch_R': avg_metrics.get('branch_R', 0.0),
                        'branch_T': avg_metrics.get('branch_T', 0.0),
                        'branch_B': avg_metrics.get('branch_B', 0.0),
                        'branch_FUSE': avg_metrics.get('branch_FUSE', 0.0),
                        'cos_mean': avg_metrics.get('cos_mean', 0.0) if not np.isnan(avg_metrics.get('cos_mean', 0.0)) else 0.0,
                        'cos_p50': avg_metrics.get('cos_p50', 0.0) if not np.isnan(avg_metrics.get('cos_p50', 0.0)) else 0.0,
                        'cos_p75': avg_metrics.get('cos_p75', 0.0) if not np.isnan(avg_metrics.get('cos_p75', 0.0)) else 0.0,
                        'viol_time_rate': avg_metrics.get('viol_time_rate', 0.0),
                        'viol_batt_rate': avg_metrics.get('viol_batt_rate', 0.0),
                        'throughput': avg_metrics.get('throughput', 0.0),
                        'deadline_violation_rate': avg_metrics.get('deadline_violation_rate', 0.0),
                        'min_SOC': avg_metrics.get('min_SOC', 1.0),
                        'completed_tasks': avg_metrics.get('completed_tasks', 0.0)
                    }
                    
                    # 写入 CSV
                    append_csv(csv_path, row, field_order)
                    
                    # 检查 Pareto 候选（throughput 高 & deadline_violation_rate 低）
                    throughput = row['throughput']
                    deadline_violation = row['deadline_violation_rate']
                    if throughput > 0 and deadline_violation < 1.0:
                        pareto_candidates.append({
                            'ablation': ablation,
                            'xi_time': xi_time,
                            'xi_batt': xi_batt,
                            'throughput': throughput,
                            'deadline_violation_rate': deadline_violation
                        })
                    
                    print(f"  完成: throughput={throughput:.4f}, "
                          f"deadline_violation_rate={deadline_violation:.4f}, "
                          f"branch_FUSE={row['branch_FUSE']:.4f}")
                else:
                    print(f"  警告: 所有重复实验均失败")
    
    # 打印 Pareto 候选
    if pareto_candidates:
        print("\n" + "=" * 60)
        print("Pareto 候选（throughput 高 & deadline_violation_rate 低）:")
        # 按 throughput 降序排序
        pareto_candidates.sort(key=lambda x: x['throughput'], reverse=True)
        for i, cand in enumerate(pareto_candidates[:10]):  # 显示前10个
            print(f"  {i+1}. ablation={cand['ablation']}, "
                  f"xi_time={cand['xi_time']:.2f}, xi_batt={cand['xi_batt']:.2f}, "
                  f"throughput={cand['throughput']:.4f}, "
                  f"deadline_violation_rate={cand['deadline_violation_rate']:.4f}")
        print("=" * 60)
    
    print(f"\n扫描完成！结果保存在: {csv_path}")
    
    # 生成图表
    print("\n开始生成图表...")
    plots_dir = os.path.join(args.save_dir, 'plots')
    ensure_dir(plots_dir)
    
    # 为每个消融模式生成图表
    for ablation in ablation_list:
        ablation_filter = {'ablation': ablation, 'tau': args.tau}
        
        # 线图1：viol_time_rate vs xi_time（固定 ablation、xi_batt、tau）
        # 选择一个固定的 xi_batt 值（取中间值）
        if len(xi_batt_list) > 0:
            fixed_xi_batt = xi_batt_list[len(xi_batt_list) // 2]
            filter_dict = {**ablation_filter, 'xi_batt': fixed_xi_batt}
            out_png = os.path.join(plots_dir, f'viol_time_rate_vs_xi_time_{ablation}.png')
            try:
                filter_csv_and_plot(
                    csv_path=csv_path,
                    x='xi_time',
                    y='viol_time_rate',
                    hue=None,
                    filter_dict=filter_dict,
                    out_png=out_png,
                    title=f'Viol Time Rate vs xi_time (ablation={ablation}, xi_batt={fixed_xi_batt:.2f})'
                )
                print(f"  生成线图1: {out_png}")
            except Exception as e:
                print(f"  生成线图1失败: {e}")
        
        # 线图2：branch_FUSE vs xi_batt（固定 ablation、xi_time、tau）
        if len(xi_time_list) > 0:
            fixed_xi_time = xi_time_list[len(xi_time_list) // 2]
            filter_dict = {**ablation_filter, 'xi_time': fixed_xi_time}
            out_png = os.path.join(plots_dir, f'branch_FUSE_vs_xi_batt_{ablation}.png')
            try:
                filter_csv_and_plot(
                    csv_path=csv_path,
                    x='xi_batt',
                    y='branch_FUSE',
                    hue=None,
                    filter_dict=filter_dict,
                    out_png=out_png,
                    title=f'Branch FUSE vs xi_batt (ablation={ablation}, xi_time={fixed_xi_time:.2f})'
                )
                print(f"  生成线图2: {out_png}")
            except Exception as e:
                print(f"  生成线图2失败: {e}")
        
        # 线图3：throughput vs xi_time（固定 ablation、tau，hue=xi_batt）
        out_png = os.path.join(plots_dir, f'throughput_vs_xi_time_{ablation}.png')
        try:
            filter_csv_and_plot(
                csv_path=csv_path,
                x='xi_time',
                y='throughput',
                hue='xi_batt',
                filter_dict=ablation_filter,
                out_png=out_png,
                title=f'Throughput vs xi_time (ablation={ablation})'
            )
            print(f"  生成线图3: {out_png}")
        except Exception as e:
            print(f"  生成线图3失败: {e}")
        
        # 热力图1：throughput 的 (xi_time, xi_batt) 分布
        out_png = os.path.join(plots_dir, f'heatmap_throughput_{ablation}.png')
        try:
            heatmap_from_csv(
                csv_path=csv_path,
                index='xi_time',
                columns='xi_batt',
                values='throughput',
                filter_dict=ablation_filter,
                out_png=out_png,
                title=f'Throughput Heatmap (ablation={ablation})'
            )
            print(f"  生成热力图1: {out_png}")
        except Exception as e:
            print(f"  生成热力图1失败: {e}")
        
        # 热力图2：deadline_violation_rate 的 (xi_time, xi_batt) 分布
        out_png = os.path.join(plots_dir, f'heatmap_deadline_violation_{ablation}.png')
        try:
            heatmap_from_csv(
                csv_path=csv_path,
                index='xi_time',
                columns='xi_batt',
                values='deadline_violation_rate',
                filter_dict=ablation_filter,
                out_png=out_png,
                title=f'Deadline Violation Rate Heatmap (ablation={ablation})'
            )
            print(f"  生成热力图2: {out_png}")
        except Exception as e:
            print(f"  生成热力图2失败: {e}")
        
        # 热力图3：branch_FUSE 的 (xi_time, xi_batt) 分布（可选）
        out_png = os.path.join(plots_dir, f'heatmap_branch_FUSE_{ablation}.png')
        try:
            heatmap_from_csv(
                csv_path=csv_path,
                index='xi_time',
                columns='xi_batt',
                values='branch_FUSE',
                filter_dict=ablation_filter,
                out_png=out_png,
                title=f'Branch FUSE Ratio Heatmap (ablation={ablation})'
            )
            print(f"  生成热力图3: {out_png}")
        except Exception as e:
            print(f"  生成热力图3失败: {e}")
    
    # 生成跨消融模式的对比线图
    # 线图1：viol_time_rate vs xi_time（按 ablation 分组）
    # 选择一个固定的 xi_batt 值
    if len(xi_batt_list) > 0:
        fixed_xi_batt = xi_batt_list[len(xi_batt_list) // 2]
        filter_dict = {'xi_batt': fixed_xi_batt, 'tau': args.tau}
        out_png = os.path.join(plots_dir, f'viol_time_rate_vs_xi_time_comparison.png')
        try:
            filter_csv_and_plot(
                csv_path=csv_path,
                x='xi_time',
                y='viol_time_rate',
                hue='ablation',
                filter_dict=filter_dict,
                out_png=out_png,
                title=f'Viol Time Rate vs xi_time (xi_batt={fixed_xi_batt:.2f}, comparison)'
            )
            print(f"  生成对比线图1: {out_png}")
        except Exception as e:
            print(f"  生成对比线图1失败: {e}")
    
    print(f"\n图表生成完成！保存在: {plots_dir}")


if __name__ == '__main__':
    main()

