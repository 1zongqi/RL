"""
环境适配器模块

提供统一的环境接口，支持 Mock、Sorting、Fulfillment 和 RWARE 后端
"""

from .mock_env import MockWarehouseAdapter
from .sorting_center_adapter import SortingCenterAdapter
from .fulfillment_adapter import FulfillmentAdapter
from .rware_adapter import RWAREAdapter

# 为向后兼容提供别名
FulfillmentWarehouseAdapter = FulfillmentAdapter

__all__ = [
    'MockWarehouseAdapter',
    'SortingCenterAdapter', 
    'FulfillmentAdapter',
    'FulfillmentWarehouseAdapter',  # 别名
    'RWAREAdapter'
]
