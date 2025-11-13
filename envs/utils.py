"""
Utility functions for environment adapters.
Common functions used across multiple adapters.
"""
import numpy as np
from typing import Tuple, List, Union


def nearest_point(point: Tuple[float, float], points_array: np.ndarray) -> Tuple[float, float]:
    """
    Find the nearest point in an array to a given point.
    
    Args:
        point: Target point (x, y)
        points_array: Array of points, shape (N, 2) or (N,)
        
    Returns:
        Nearest point (x, y) as tuple
    """
    if len(points_array) == 0:
        return (0.0, 0.0)
    
    points_array = np.asarray(points_array)
    if points_array.ndim == 1:
        points_array = points_array.reshape(1, -1)
    
    # Calculate distances
    distances = np.sqrt(np.sum((points_array - np.array(point)) ** 2, axis=1))
    nearest_idx = np.argmin(distances)
    
    nearest = points_array[nearest_idx]
    return (float(nearest[0]), float(nearest[1]))


def find_nearest_charger(pos_xy: Tuple[float, float], charger_coords: Union[List, np.ndarray]) -> Tuple[float, float]:
    """
    Find the nearest charger position to the given agent position.
    
    Args:
        pos_xy: Agent position (x, y)
        charger_coords: List or array of charger positions [[x1, y1], [x2, y2], ...]
        
    Returns:
        Nearest charger position (x, y) as tuple
    """
    if len(charger_coords) == 0:
        return (0.0, 0.0)
    
    return nearest_point(pos_xy, charger_coords)


def convert_time_to_steps(time_value: float, time_unit: str, step_time_sec: float = 1.0) -> float:
    """
    Convert time value to steps.
    
    Args:
        time_value: Time value to convert
        time_unit: Unit of time_value ('step' or 'second')
        step_time_sec: Seconds per step (used when time_unit='second')
        
    Returns:
        Time value in steps
    """
    if time_unit == 'step':
        return float(time_value)
    elif time_unit == 'second':
        return float(time_value / step_time_sec)
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}. Must be 'step' or 'second'.")


def convert_battery_to_steps(battery_value: float, battery_unit: str, max_batt_steps: float = 200.0) -> float:
    """
    Convert battery value to steps.
    
    Args:
        battery_value: Battery value to convert
        battery_unit: Unit of battery_value ('step' or 'ratio')
        max_batt_steps: Maximum battery in steps (used when battery_unit='ratio')
        
    Returns:
        Battery value in steps
    """
    if battery_unit == 'step':
        return float(battery_value)
    elif battery_unit == 'ratio':
        # Convert ratio [0, 1] to steps [0, max_batt_steps]
        return float(battery_value * max_batt_steps)
    else:
        raise ValueError(f"Unknown battery_unit: {battery_unit}. Must be 'step' or 'ratio'.")


def normalize_obs(obs_array: np.ndarray, min_val: float = None, max_val: float = None) -> np.ndarray:
    """
    Normalize observation array to [0, 1] range.
    
    Args:
        obs_array: Observation array
        min_val: Minimum value for normalization (if None, use array min)
        max_val: Maximum value for normalization (if None, use array max)
        
    Returns:
        Normalized array
    """
    obs_array = np.asarray(obs_array, dtype=np.float32)
    
    if min_val is None:
        min_val = np.min(obs_array)
    if max_val is None:
        max_val = np.max(obs_array)
    
    if max_val == min_val:
        return np.zeros_like(obs_array)
    
    normalized = (obs_array - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)

