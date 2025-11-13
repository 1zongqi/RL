"""
随机种子工具

统一设置 numpy, torch, random 的随机种子
"""

import numpy as np
import torch
import random
import os


def set_seed(seed: int):
    """
    设置随机种子
    
    参数:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # 如果使用 CUDA，也设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

