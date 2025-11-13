"""
实验日志工具

统一日志记录、配置保存、曲线绘制接口
支持 CSV 导出
"""

import os
import json
import csv
from typing import Dict, Any, Optional
from pathlib import Path


class ExpLogger:
    """
    实验日志记录器
    
    功能：
    - 记录训练指标到控制台和文件
    - 保存配置
    - 导出 CSV 格式的指标文件
    - TODO: 绘制曲线
    """
    
    def __init__(self, save_dir: str, exp_name: str = 'exp', is_print: bool = True):
        """
        初始化日志记录器
        
        参数:
            save_dir: 保存目录
            exp_name: 实验名称
            is_print: 是否打印到控制台
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.exp_name = exp_name
        self.is_print = is_print
        
        # 日志文件
        self.log_file = self.save_dir / 'run.log'
        
        # CSV 文件
        self.csv_file = self.save_dir / 'metrics.csv'
        
        # 打开日志文件（追加模式）
        self.log_fp = open(self.log_file, 'a', encoding='utf-8')
        
        # 初始化 CSV 文件（如果不存在）
        self.csv_initialized = False
        self.csv_writer = None
        self.csv_fp = None
        
        # 指标缓存（用于后续绘制曲线）
        self.metrics_cache = []
    
    def _init_csv(self, metrics_keys: list):
        """初始化 CSV 文件"""
        if not self.csv_initialized:
            self.csv_fp = open(self.csv_file, 'w', newline='', encoding='utf-8')
            fieldnames = ['step'] + sorted(metrics_keys)
            self.csv_writer = csv.DictWriter(self.csv_fp, fieldnames=fieldnames, extrasaction='ignore')
            self.csv_writer.writeheader()
            self.csv_initialized = True
            self.fieldnames = set(fieldnames)
        else:
            # 检查是否有新字段
            new_keys = set(['step'] + list(metrics_keys))
            if new_keys != self.fieldnames:
                # 有新字段，需要重新初始化CSV（关闭旧文件，重写）
                if self.csv_fp:
                    self.csv_fp.close()
                
                # 合并所有字段
                all_fieldnames = sorted(self.fieldnames.union(new_keys))
                
                # 重新打开文件并写入表头
                self.csv_fp = open(self.csv_file, 'w', newline='', encoding='utf-8')
                self.csv_writer = csv.DictWriter(self.csv_fp, fieldnames=all_fieldnames, extrasaction='ignore')
                self.csv_writer.writeheader()
                
                # 重写之前的数据
                for cached_metrics in self.metrics_cache:
                    self.csv_writer.writerow(cached_metrics)
                
                self.csv_fp.flush()
                self.fieldnames = set(all_fieldnames)
    
    def log_dict(self, metrics: Dict[str, float]):
        """
        记录指标字典
        
        同时写到 stdout、日志文件和 CSV 文件
        支持动态字段扩展
        
        参数:
            metrics: 指标字典（应包含 'iter' 键作为步数）
        """
        # 获取步数（从 metrics 中提取 'iter' 或使用默认值）
        step = metrics.get('iter', len(self.metrics_cache))
        
        # 构建日志字符串
        metric_strs = []
        for k, v in sorted(metrics.items()):
            if k == 'iter':
                continue
            if isinstance(v, float):
                if abs(v) < 1e-3:
                    metric_strs.append(f"{k}={v:.6f}")
                else:
                    metric_strs.append(f"{k}={v:.4f}")
            else:
                metric_strs.append(f"{k}={v}")
        log_str = f"Iter {step}: " + ", ".join(metric_strs)
        
        # 打印到控制台（如果启用）
        if self.is_print:
            print(log_str)
        
        # 写入日志文件
        self.log_fp.write(log_str + '\n')
        self.log_fp.flush()
        
        # 写入 CSV 文件（动态扩展字段）
        # 注意：需要在缓存之前检查，因为_init_csv可能会使用metrics_cache重写
        self._init_csv(list(metrics.keys()))
        
        # 缓存指标（用于后续绘制和CSV重写）
        self.metrics_cache.append(metrics.copy())
        
        # 写入当前行
        row = metrics.copy()
        self.csv_writer.writerow(row)
        self.csv_fp.flush()
    
    def save_config(self, config: Dict[str, Any]):
        """
        保存配置到 JSON 文件
        
        参数:
            config: 配置字典
        """
        config_file = self.save_dir / 'config.json'
        
        # 转换不可序列化的值
        config_serializable = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                config_serializable[k] = v
            else:
                config_serializable[k] = str(v)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存至: {config_file}")
    
    def flush(self):
        """刷新日志缓冲区"""
        self.log_fp.flush()
        if self.csv_fp:
            self.csv_fp.flush()
    
    def close(self):
        """关闭日志文件"""
        self.log_fp.close()
        if self.csv_fp:
            self.csv_fp.close()
    
    def log_episode_stats(
        self,
        episode: int,
        stats: Dict[str, float],
        csv_path: Optional[str] = None
    ):
        """
        记录每个episode的奖励和成本统计
        
        参数:
            episode: 回合数/迭代数
            avg_reward_per_agent: 本回合每个智能体的平均奖励
            avg_cost_per_agent: 本回合每个智能体的平均违约成本
            csv_path: CSV文件路径（可选，默认使用save_dir/episode_metrics.csv）
        """
        import datetime
        
        # 确定CSV路径
        if csv_path is None:
            csv_path = self.save_dir / 'episode_metrics.csv'
        else:
            csv_path = Path(csv_path)
        
        # 确保目录存在
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查文件是否存在
        file_exists = csv_path.exists()
        
        # 写入CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = ['episode'] + sorted(stats.keys()) + ['timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()
            
            # 写入数据行
            row = {'episode': episode, 'timestamp': datetime.datetime.now().isoformat()}
            for key, value in stats.items():
                row[key] = value
            writer.writerow(row)

        if self.is_print:
            summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in sorted(stats.items()))
            print(f"[Episode {episode}] {summary}")
    
    def __del__(self):
        """析构函数，确保文件关闭"""
        if hasattr(self, 'log_fp') and not self.log_fp.closed:
            self.log_fp.close()
        if hasattr(self, 'csv_fp') and self.csv_fp and not self.csv_fp.closed:
            self.csv_fp.close()
