"""
IO 工具函数

提供 CSV 读写和路径创建功能
"""

import os
import csv
from typing import List, Dict, Any
from pathlib import Path


def ensure_dir(path: str) -> None:
    """
    确保目录存在，不存在则创建
    
    参数:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, Any]], field_order: List[str]) -> None:
    """
    写入 CSV 文件（不存在则创建并写 header）
    
    参数:
        path: CSV 文件路径
        rows: 数据行列表，每个元素是一个字典
        field_order: 字段顺序列表
    """
    ensure_dir(os.path.dirname(path))
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: str, row: Dict[str, Any], field_order: List[str]) -> None:
    """
    追加一行到 CSV 文件（不存在则创建并写 header）
    
    参数:
        path: CSV 文件路径
        row: 数据行字典
        field_order: 字段顺序列表
    """
    ensure_dir(os.path.dirname(path))
    
    # 检查文件是否存在
    file_exists = os.path.exists(path)
    
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

