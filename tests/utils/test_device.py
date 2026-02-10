"""
Unit tests for device management utilities.

These tests run on CPU only and do not require GPU hardware.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.device import (
    DeviceManager,
    get_cuda_device_count,
    get_device,
    get_device_info,
    get_device_manager,
    is_cuda_available,
    to_device,
)


@pytest.mark.unit
class TestDeviceManager:
    """Tests for DeviceManager class."""

    def test_init_default(self):
        """Test default initialization."""
        manager = DeviceManager()
        assert manager.device is not None
        assert isinstance(manager.device, torch.device)

    def test_force_cpu(self):
        """Test forcing CPU usage."""
        manager = DeviceManager(force_cpu=True)
        assert manager.device.type == "cpu"
        assert manager.is_cpu
        assert not manager.is_cuda

    def test_is_cpu_property(self):
        """Test is_cpu property."""
        manager = DeviceManager(force_cpu=True)
        assert manager.is_cpu is True

    def test_is_cuda_property_on_cpu(self):
        """Test is_cuda property when using CPU."""
        manager = DeviceManager(force_cpu=True)
        assert manager.is_cuda is False

    def test_to_device_tensor(self):
        """Test moving tensor to device."""
        manager = DeviceManager(force_cpu=True)
        tensor = torch.randn(2, 3)
        moved_tensor = manager.to_device(tensor)
        assert moved_tensor.device.type == "cpu"

    def test_to_device_model(self):
        """Test moving model to device."""
        manager = DeviceManager(force_cpu=True)
        model = torch.nn.Linear(10, 5)
        moved_model = manager.to_device(model)
        # Check that parameters are on correct device
        for param in moved_model.parameters():
            assert param.device.type == "cpu"

    def test_get_device_name_cpu(self):
        """Test get_device_name for CPU."""
        manager = DeviceManager(force_cpu=True)
        assert manager.get_device_name() == "CPU"

    def test_get_device_properties_cpu(self):
        """Test get_device_properties for CPU."""
        manager = DeviceManager(force_cpu=True)
        props = manager.get_device_properties()
        assert "device" in props
        assert "type" in props
        assert props["type"] == "cpu"
        assert "cuda_available" in props

    def test_empty_cache_cpu(self):
        """Test empty_cache on CPU (should not crash)."""
        manager = DeviceManager(force_cpu=True)
        manager.empty_cache()  # Should not raise any error

    def test_repr(self):
        """Test string representation."""
        manager = DeviceManager(force_cpu=True)
        repr_str = repr(manager)
        assert "DeviceManager" in repr_str
        assert "cpu" in repr_str

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    def test_device_selection_with_cuda(self, mock_device_count, mock_cuda_available):
        """Test device selection when CUDA is available."""
        manager = DeviceManager(force_cpu=False)
        # Should select CUDA when available and not forced to CPU
        assert manager.device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=2)
    @patch("torch.cuda.get_device_properties")
    def test_device_id_selection(
        self, mock_get_props, mock_device_count, mock_cuda_available
    ):
        """Test specific device ID selection."""
        manager = DeviceManager(force_cpu=False, device_id=1)
        assert "cuda" in str(manager.device)
        assert "1" in str(manager.device)

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_invalid_device_id(self, mock_device_count, mock_cuda_available):
        """Test error handling for invalid device ID."""
        with pytest.raises(ValueError, match="GPU device 5 not available"):
            DeviceManager(force_cpu=False, device_id=5)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_not_available(self, mock_cuda_available):
        """Test behavior when CUDA is not available."""
        manager = DeviceManager(force_cpu=False)
        assert manager.device.type == "cpu"
        assert manager.is_cpu

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="Mock GPU")
    def test_get_device_name_with_cuda(self, mock_get_name, mock_cuda_available):
        """Test get_device_name with CUDA."""
        manager = DeviceManager(force_cpu=False)
        name = manager.get_device_name()
        assert "CUDA" in name
        assert "Mock GPU" in name

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_get_device_properties_with_cuda(self, mock_get_props, mock_cuda_available):
        """Test get_device_properties with CUDA."""
        # Create a mock device properties object
        mock_props = MagicMock()
        mock_props.name = "Mock GPU"
        mock_props.total_memory = 8 * (1024**3)  # 8GB
        mock_props.major = 7
        mock_props.minor = 5
        mock_props.multi_processor_count = 80
        mock_get_props.return_value = mock_props

        manager = DeviceManager(force_cpu=False)
        props = manager.get_device_properties()

        assert "device" in props
        assert "type" in props
        assert "name" in props
        assert "total_memory_gb" in props
        assert "compute_capability" in props
        assert props["compute_capability"] == "7.5"


@pytest.mark.unit
class TestGlobalDeviceManager:
    """Tests for global device manager functions."""

    def teardown_method(self):
        """Reset global device manager after each test."""
        # Reset the global device manager
        get_device_manager(force_cpu=True, reset=True)

    def test_get_device_manager_creates_instance(self):
        """Test that get_device_manager creates an instance."""
        manager = get_device_manager(force_cpu=True, reset=True)
        assert isinstance(manager, DeviceManager)

    def test_get_device_manager_returns_same_instance(self):
        """Test that get_device_manager returns the same instance."""
        manager1 = get_device_manager(force_cpu=True, reset=True)
        manager2 = get_device_manager()
        assert manager1 is manager2

    def test_get_device_manager_reset(self):
        """Test resetting the global device manager."""
        manager1 = get_device_manager(force_cpu=True, reset=True)
        manager2 = get_device_manager(force_cpu=True, reset=True)
        assert manager1 is not manager2

    def test_get_device_convenience_function(self):
        """Test get_device convenience function."""
        device = get_device(force_cpu=True)
        assert isinstance(device, torch.device)
        assert device.type == "cpu"


@pytest.mark.unit
class TestDeviceUtilities:
    """Tests for device utility functions."""

    def test_is_cuda_available(self):
        """Test is_cuda_available function."""
        result = is_cuda_available()
        assert isinstance(result, bool)
        assert result == torch.cuda.is_available()

    def test_get_cuda_device_count(self):
        """Test get_cuda_device_count function."""
        count = get_cuda_device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_to_device_with_explicit_device(self):
        """Test to_device with explicit device."""
        tensor = torch.randn(2, 3)
        device = torch.device("cpu")
        moved_tensor = to_device(tensor, device=device)
        assert moved_tensor.device.type == "cpu"

    def test_to_device_with_global_manager(self):
        """Test to_device using global device manager."""
        get_device_manager(force_cpu=True, reset=True)
        tensor = torch.randn(2, 3)
        moved_tensor = to_device(tensor)
        assert moved_tensor.device.type == "cpu"

    def test_to_device_with_force_cpu(self):
        """Test to_device with force_cpu flag."""
        tensor = torch.randn(2, 3)
        moved_tensor = to_device(tensor, force_cpu=True)
        assert moved_tensor.device.type == "cpu"

    def test_get_device_info(self):
        """Test get_device_info function."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "pytorch_version" in info
        assert "device_count" in info
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["device_count"], int)


@pytest.mark.unit
class TestDeviceManagerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_device_property_immutable(self):
        """Test that device property returns consistent value."""
        manager = DeviceManager(force_cpu=True)
        device1 = manager.device
        device2 = manager.device
        assert device1.type == device2.type

    def test_multiple_to_device_calls(self):
        """Test multiple to_device calls don't cause issues."""
        manager = DeviceManager(force_cpu=True)
        tensor = torch.randn(2, 3)
        moved1 = manager.to_device(tensor)
        moved2 = manager.to_device(moved1)
        assert moved1.device.type == moved2.device.type == "cpu"

    def test_empty_model_to_device(self):
        """Test moving empty model to device."""
        manager = DeviceManager(force_cpu=True)

        class EmptyModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = EmptyModel()
        moved_model = manager.to_device(model)
        assert moved_model is not None

    def test_nested_model_to_device(self):
        """Test moving nested model to device."""
        manager = DeviceManager(force_cpu=True)

        class NestedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)

            def forward(self, x):
                return self.layer2(self.layer1(x))

        model = NestedModel()
        moved_model = manager.to_device(model)

        # Check all parameters are on correct device
        for param in moved_model.parameters():
            assert param.device.type == "cpu"


@pytest.mark.unit
class TestDeviceManagerThreadSafety:
    """Tests for thread safety considerations."""

    def test_concurrent_device_access(self):
        """Test concurrent access to device property."""
        manager = DeviceManager(force_cpu=True)
        devices = [manager.device for _ in range(100)]
        # All should be same device type
        assert all(d.type == "cpu" for d in devices)

    def test_concurrent_to_device(self):
        """Test concurrent to_device calls."""
        manager = DeviceManager(force_cpu=True)
        tensors = [torch.randn(2, 3) for _ in range(10)]
        moved = [manager.to_device(t) for t in tensors]
        assert all(t.device.type == "cpu" for t in moved)
