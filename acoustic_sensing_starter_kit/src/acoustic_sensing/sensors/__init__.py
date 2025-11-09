"""
Real-time acoustic sensors

This module contains real-time sensing capabilities:
- Optimized real-time sensor implementation
- Sensor configuration and management
"""

from .real_time_sensor import OptimizedRealTimeSensor
from .sensor_config import SensorConfig

__all__ = [
    'OptimizedRealTimeSensor',
    'SensorConfig'
]