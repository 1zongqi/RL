"""
绘图工具函数

使用 matplotlib 绘制线图和热力图
"""

import csv
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path


def lineplot_from_csv(
    csv_path: str,
    x: str,
    y: str,
    hue: Optional[str] = None,
    out_png: str = "",
    title: str = ""
) -> None:
    """
    从 CSV 文件读取数据并绘制线图
    
    参数:
        csv_path: CSV 文件路径
        x: X 轴字段名
        y: Y 轴字段名
        hue: 分组字段名（可选），用于绘制多条线
        out_png: 输出 PNG 文件路径
        title: 图表标题
    """
    # 读取 CSV 数据
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    if len(data) == 0:
        print(f"警告: {csv_path} 中没有数据")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 转换数据类型
    for row in data:
        try:
            row[x] = float(row[x])
            row[y] = float(row[y])
            if hue and hue in row:
                row[hue] = str(row[hue])  # hue 作为字符串分组
        except (ValueError, KeyError):
            continue
    
    # 如果没有 hue，直接绘制一条线
    if hue is None or hue not in data[0]:
        x_vals = [float(row[x]) for row in data if x in row and y in row]
        y_vals = [float(row[y]) for row in data if x in row and y in row]
        if len(x_vals) > 0:
            # 排序
            sorted_pairs = sorted(zip(x_vals, y_vals))
            x_vals, y_vals = zip(*sorted_pairs) if sorted_pairs else ([], [])
            ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6)
    else:
        # 按 hue 分组绘制多条线
        groups = {}
        for row in data:
            if x in row and y in row and hue in row:
                h_val = str(row[hue])
                if h_val not in groups:
                    groups[h_val] = {'x': [], 'y': []}
                try:
                    groups[h_val]['x'].append(float(row[x]))
                    groups[h_val]['y'].append(float(row[y]))
                except ValueError:
                    continue
        
        # 为每个组绘制一条线
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for i, (h_val, group_data) in enumerate(sorted(groups.items())):
            if len(group_data['x']) > 0:
                # 排序
                sorted_pairs = sorted(zip(group_data['x'], group_data['y']))
                x_vals, y_vals = zip(*sorted_pairs) if sorted_pairs else ([], [])
                ax.plot(x_vals, y_vals, marker='o', linewidth=2, markersize=6,
                       label=str(h_val), color=colors[i])
        
        # 添加图例
        ax.legend(loc='best', fontsize=10)
    
    # 设置标签和标题
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 网格
    ax.grid(True, alpha=0.3)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存
    if out_png:
        ensure_dir = Path(out_png).parent
        ensure_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {out_png}")
    else:
        plt.show()
    
    plt.close()


def lineplot_multi_series(
    csv_path: str,
    x: str,
    y_list: List[str],
    out_png: str,
    title: str = ""
) -> None:
    """
    从 CSV 文件读取数据并绘制多条 Y 轴的线图
    
    参数:
        csv_path: CSV 文件路径
        x: X 轴字段名
        y_list: Y 轴字段名列表（多条线）
        out_png: 输出 PNG 文件路径
        title: 图表标题
    """
    # 读取 CSV 数据
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    if len(data) == 0:
        print(f"警告: {csv_path} 中没有数据")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 转换数据类型并提取数据
    x_vals = []
    y_data = {y: [] for y in y_list}
    
    for row in data:
        if x in row:
            try:
                x_val = float(row[x])
                x_vals.append(x_val)
                for y in y_list:
                    if y in row:
                        y_data[y].append(float(row[y]))
                    else:
                        y_data[y].append(np.nan)
            except (ValueError, KeyError):
                continue
    
    if len(x_vals) == 0:
        print(f"警告: 没有有效的数据点")
        return
    
    # 排序
    sorted_indices = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
    x_vals = [x_vals[i] for i in sorted_indices]
    for y in y_list:
        y_data[y] = [y_data[y][i] for i in sorted_indices]
    
    # 为每个 Y 字段绘制一条线
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_list)))
    for i, y in enumerate(y_list):
        if len(y_data[y]) > 0:
            ax.plot(x_vals, y_data[y], marker='o', linewidth=2, markersize=6,
                   label=y, color=colors[i])
    
    # 添加图例
    ax.legend(loc='best', fontsize=10)
    
    # 设置标签和标题
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 网格
    ax.grid(True, alpha=0.3)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存
    ensure_dir = Path(out_png).parent
    ensure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {out_png}")
    
    plt.close()


def heatmap_from_csv(
    csv_path: str,
    index: str,
    columns: str,
    values: str,
    filter_dict: Optional[Dict[str, Any]] = None,
    out_png: str = "",
    title: str = ""
) -> None:
    """
    从 CSV 文件读取数据并绘制热力图
    
    参数:
        csv_path: CSV 文件路径
        index: 行索引字段名（如 'xi_time'）
        columns: 列索引字段名（如 'xi_batt'）
        values: 数值字段名（如 'throughput'）
        filter_dict: 筛选条件字典（如 {'ablation': 'none', 'tau': 0.6}）
        out_png: 输出 PNG 文件路径
        title: 图表标题
    """
    # 读取 CSV 数据
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    if len(data) == 0:
        print(f"警告: {csv_path} 中没有数据")
        return
    
    # 按 filter_dict 筛选数据
    if filter_dict:
        filtered_data = []
        for row in data:
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
        data = filtered_data
    
    if len(data) == 0:
        print(f"警告: 筛选后没有数据")
        return
    
    # 提取唯一值
    index_vals = sorted(set(float(row[index]) for row in data if index in row))
    column_vals = sorted(set(float(row[columns]) for row in data if columns in row))
    
    if len(index_vals) == 0 or len(column_vals) == 0:
        print(f"警告: 无法提取索引或列值")
        return
    
    # 构建 pivot 表
    pivot_data = {}
    for row in data:
        if index in row and columns in row and values in row:
            try:
                idx_val = float(row[index])
                col_val = float(row[columns])
                val = float(row[values])
                
                # 处理 NaN
                if np.isnan(val):
                    val = 0.0
                
                key = (idx_val, col_val)
                # 如果已有值，取平均（或覆盖，这里选择覆盖）
                pivot_data[key] = val
            except (ValueError, TypeError):
                continue
    
    # 构建矩阵
    matrix = np.zeros((len(index_vals), len(column_vals)))
    for i, idx_val in enumerate(index_vals):
        for j, col_val in enumerate(column_vals):
            key = (idx_val, col_val)
            if key in pivot_data:
                matrix[i, j] = pivot_data[key]
            else:
                matrix[i, j] = np.nan
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(column_vals)))
    ax.set_yticks(np.arange(len(index_vals)))
    ax.set_xticklabels([f'{v:.2f}' for v in column_vals], rotation=45, ha='right')
    ax.set_yticklabels([f'{v:.2f}' for v in index_vals])
    
    # 设置标签
    ax.set_xlabel(columns, fontsize=12)
    ax.set_ylabel(index, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加数值标注（可选，如果矩阵较小）
    if len(index_vals) <= 10 and len(column_vals) <= 10:
        for i in range(len(index_vals)):
            for j in range(len(column_vals)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.2f}',
                                ha="center", va="center", color="white" if val < matrix.mean() else "black",
                                fontsize=8)
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(values, fontsize=12)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存
    if out_png:
        ensure_dir = Path(out_png).parent
        ensure_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {out_png}")
    else:
        plt.show()
    
    plt.close()
