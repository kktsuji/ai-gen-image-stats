"""Utility functions and helpers.

This module provides cross-cutting utilities for CLI parsing,
configuration management, device handling, and metrics computation.
"""

# Import only modules that have been implemented
# Other imports will be added as the refactoring progresses
from src.utils.device import (
    DeviceManager,
    get_cuda_device_count,
    get_device,
    get_device_info,
    get_device_manager,
    is_cuda_available,
    to_device,
)

__all__ = [
    "DeviceManager",
    "get_device_manager",
    "get_device",
    "is_cuda_available",
    "get_cuda_device_count",
    "to_device",
    "get_device_info",
]
