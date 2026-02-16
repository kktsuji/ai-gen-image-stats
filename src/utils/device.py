"""
Device management utilities for PyTorch.

Provides utilities for CPU/GPU device detection and management,
with support for forcing CPU-only mode for testing.
"""

import logging
from typing import Optional, Union

import torch

# Module-level logger
logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and provides device utilities."""

    def __init__(self, force_cpu: bool = False, device_id: Optional[int] = None):
        """
        Initialize device manager.

        Args:
            force_cpu: If True, force CPU usage even if GPU is available
            device_id: Specific GPU device ID to use (0, 1, 2, etc.)
                      If None, uses default GPU (cuda:0)
        """
        self._force_cpu = force_cpu
        self._device_id = device_id
        self._device = self._select_device()

    def _select_device(self) -> torch.device:
        """
        Select the appropriate device based on availability and settings.

        Returns:
            torch.device: The selected device
        """
        if self._force_cpu:
            logger.debug("Device selection: CPU (forced)")
            return torch.device("cpu")

        if not torch.cuda.is_available():
            logger.debug("Device selection: CPU (CUDA not available)")
            return torch.device("cpu")

        if self._device_id is not None:
            if self._device_id >= torch.cuda.device_count():
                logger.error(
                    f"GPU device {self._device_id} not available. "
                    f"Only {torch.cuda.device_count()} device(s) found."
                )
                raise ValueError(
                    f"GPU device {self._device_id} not available. "
                    f"Only {torch.cuda.device_count()} device(s) found."
                )
            logger.debug(f"Device selection: cuda:{self._device_id}")
            return torch.device(f"cuda:{self._device_id}")

        logger.debug("Device selection: cuda (default GPU)")
        return torch.device("cuda")

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self._device.type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self._device.type == "cpu"

    def to_device(
        self, obj: Union[torch.Tensor, torch.nn.Module]
    ) -> Union[torch.Tensor, torch.nn.Module]:
        """
        Move tensor or model to the managed device.

        Args:
            obj: Tensor or Module to move

        Returns:
            The object moved to device
        """
        return obj.to(self._device)

    def get_device_name(self) -> str:
        """
        Get a human-readable device name.

        Returns:
            str: Device name (e.g., "CPU", "CUDA:0", "CUDA:0 (Tesla V100)")
        """
        if self.is_cpu:
            return "CPU"

        device_name = str(self._device).upper()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(self._device)
            return f"{device_name} ({gpu_name})"

        return device_name

    def get_device_properties(self) -> dict:
        """
        Get device properties and capabilities.

        Returns:
            dict: Device properties including memory, compute capability, etc.
        """
        props = {
            "device": str(self._device),
            "type": self._device.type,
            "cuda_available": torch.cuda.is_available(),
        }

        if self.is_cuda and torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(self._device)
            props.update(
                {
                    "name": device_props.name,
                    "total_memory_gb": device_props.total_memory / (1024**3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "multi_processor_count": device_props.multi_processor_count,
                }
            )

        return props

    def empty_cache(self) -> None:
        """Empty CUDA cache if using GPU."""
        if self.is_cuda:
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        """String representation of device manager."""
        return f"DeviceManager(device={self._device}, force_cpu={self._force_cpu})"


# Global device manager instance (initialized on first use)
_global_device_manager: Optional[DeviceManager] = None


def get_device_manager(
    force_cpu: bool = False, device_id: Optional[int] = None, reset: bool = False
) -> DeviceManager:
    """
    Get or create the global device manager instance.

    Args:
        force_cpu: If True, force CPU usage
        device_id: Specific GPU device ID to use
        reset: If True, create a new device manager even if one exists

    Returns:
        DeviceManager: The global device manager instance
    """
    global _global_device_manager

    if _global_device_manager is None or reset:
        _global_device_manager = DeviceManager(force_cpu=force_cpu, device_id=device_id)

    return _global_device_manager


def get_device(
    force_cpu: bool = False, device_id: Optional[int] = None
) -> torch.device:
    """
    Get the current device.

    Convenience function that creates/uses global device manager.

    Args:
        force_cpu: If True, force CPU usage
        device_id: Specific GPU device ID to use

    Returns:
        torch.device: The selected device
    """
    manager = get_device_manager(force_cpu=force_cpu, device_id=device_id)
    return manager.device


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        bool: True if CUDA is available
    """
    return torch.cuda.is_available()


def get_cuda_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        int: Number of CUDA devices (0 if CUDA not available)
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def to_device(
    obj: Union[torch.Tensor, torch.nn.Module],
    device: Optional[torch.device] = None,
    force_cpu: bool = False,
) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or model to specified device.

    Args:
        obj: Tensor or Module to move
        device: Target device (if None, uses global device manager)
        force_cpu: If True and device is None, force CPU

    Returns:
        The object moved to device
    """
    if device is None:
        manager = get_device_manager(force_cpu=force_cpu)
        device = manager.device

    return obj.to(device)


def get_device_info() -> dict:
    """
    Get comprehensive device information.

    Returns:
        dict: System device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "device_count": get_cuda_device_count(),
    }

    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "id": i,
                    "name": device_props.name,
                    "total_memory_gb": device_props.total_memory / (1024**3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                }
            )
        info["devices"] = devices

    return info
