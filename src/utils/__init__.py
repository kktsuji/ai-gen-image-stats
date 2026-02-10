"""Utility functions and helpers.

This module provides cross-cutting utilities for CLI parsing,
configuration management, device handling, and metrics computation.
"""

from src.utils.cli import parse_args
from src.utils.config import load_config, merge_configs
from src.utils.device import get_device, set_device
from src.utils.metrics import compute_metrics

__all__ = [
    "get_device",
    "set_device",
    "load_config",
    "merge_configs",
    "parse_args",
    "compute_metrics",
]
