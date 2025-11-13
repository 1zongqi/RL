"""
类型别名定义

统一常用类型别名，便于代码维护
"""

import torch
from typing import Dict, Any

# PyTorch 张量别名
Tensor = torch.Tensor

# Rollout 数据字典类型
RolloutDict = Dict[str, Any]

