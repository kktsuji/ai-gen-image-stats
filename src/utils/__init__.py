"""Utility functions and helpers.

This module provides cross-cutting utilities for CLI parsing,
configuration management, device handling, metrics computation, and logging.
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
from src.utils.logging import get_log_file_path, get_logger, setup_logging

__all__ = [
    "DeviceManager",
    "get_device_manager",
    "get_device",
    "is_cuda_available",
    "get_cuda_device_count",
    "to_device",
    "get_device_info",
    "setup_logging",
    "get_log_file_path",
    "get_logger",
]
