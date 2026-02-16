"""
Pytest configuration and shared fixtures for all tests.

This module provides common test fixtures used across all test tiers:
- Mock datasets and data loaders
- Temporary directories for outputs
- Device fixtures (CPU/GPU)
- Reproducibility helpers (fixed seeds)
- Mock configurations
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from PIL import Image

# ==============================================================================
# Pytest Configuration
# ==============================================================================


def pytest_configure(config):
    """Register custom markers for test tiers."""
    config.addinivalue_line(
        "markers", "unit: Fast unit tests (CPU only, < 100ms per test)"
    )
    config.addinivalue_line(
        "markers", "component: Component tests with small data (1-5s per test)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with mini datasets (10-60s per test)"
    )
    config.addinivalue_line(
        "markers", "smoke: Full workflow smoke tests (5-15min, GPU preferred)"
    )
    config.addinivalue_line("markers", "slow: Tests that take > 10 seconds")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU hardware")


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests if CUDA is not available."""
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


# ==============================================================================
# Reproducibility Fixtures
# ==============================================================================


@pytest.fixture(scope="function", autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    # Make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Device Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def device_cpu():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device_gpu():
    """Return GPU device if available, otherwise skip test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def device_auto():
    """Return GPU if available, otherwise CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Temporary Directory Fixtures
# ==============================================================================


@pytest.fixture(scope="function")
def tmp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "generated").mkdir(exist_ok=True)

    return output_dir


@pytest.fixture(scope="function")
def tmp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# ==============================================================================
# Logging Fixtures
# ==============================================================================


@pytest.fixture
def capture_logs(caplog):
    """
    Fixture to capture log messages in tests.

    This fixture sets the log level to DEBUG and provides access to captured
    log records. Tests can assert on log messages using caplog.text or
    caplog.records.

    Example:
        def test_something(capture_logs):
            logger.info("Test message")
            assert "Test message" in capture_logs.text
    """
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def temp_log_file(tmp_path):
    """
    Temporary log file for testing.

    Creates a temporary log file in a test-specific directory.
    The file is automatically cleaned up after the test.

    Returns:
        Path: Path to the temporary log file

    Example:
        def test_file_logging(temp_log_file):
            setup_logging(log_file=temp_log_file)
            logger.info("Test")
            assert temp_log_file.exists()
    """
    log_file = tmp_path / "test.log"
    return log_file


@pytest.fixture(scope="function", autouse=False)
def clean_logging_handlers():
    """
    Clean up logging handlers after test to prevent pollution.

    This fixture ensures that logging handlers created during tests
    don't affect other tests. It removes all handlers from the root
    logger after the test completes.

    Usage:
        @pytest.mark.usefixtures("clean_logging_handlers")
        def test_logging_setup():
            setup_logging(...)
            # Test code
            # Handlers are automatically cleaned up
    """
    yield
    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)


@pytest.fixture(scope="function")
def suppress_console_logging():
    """
    Suppress console logging during tests.

    This fixture temporarily sets the console logging level to CRITICAL
    to prevent test output pollution. Useful for tests that trigger
    many log messages but don't need to verify console output.

    Example:
        def test_verbose_operation(suppress_console_logging):
            # Runs without console noise
            do_something_verbose()
    """
    # Store original level
    root_logger = logging.getLogger()
    original_level = root_logger.level

    # Suppress console output
    root_logger.setLevel(logging.CRITICAL)

    yield

    # Restore original level
    root_logger.setLevel(original_level)


# ==============================================================================
# Mock Data Fixtures
# ==============================================================================


@pytest.fixture(scope="function")
def mock_image_tensor():
    """Return a tiny mock image tensor (1, 3, 32, 32)."""
    return torch.randn(1, 3, 32, 32)


@pytest.fixture(scope="function")
def mock_batch_tensor():
    """Return a tiny batch of image tensors (2, 3, 32, 32)."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture(scope="function")
def mock_labels():
    """Return mock labels for a batch of 2 samples."""
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture(scope="function")
def mock_pil_image():
    """Return a mock PIL image (32x32 RGB)."""
    return Image.new("RGB", (32, 32), color="red")


@pytest.fixture(scope="function")
def mock_dataset_small(tmp_data_dir):
    """
    Create a small mock image dataset in temporary directory.

    Structure:
        tmp_data_dir/
            0.Normal/
                img_001.jpg
                img_002.jpg
            1.Abnormal/
                img_003.jpg
                img_004.jpg

    Returns:
        Path: Root directory containing the dataset
    """
    # Create class directories
    class_0_dir = tmp_data_dir / "0.Normal"
    class_1_dir = tmp_data_dir / "1.Abnormal"
    class_0_dir.mkdir(exist_ok=True)
    class_1_dir.mkdir(exist_ok=True)

    # Create mock images for class 0
    for i in range(2):
        img = Image.new("RGB", (64, 64), color="green")
        img.save(class_0_dir / f"img_{i:03d}.jpg")

    # Create mock images for class 1
    for i in range(2, 4):
        img = Image.new("RGB", (64, 64), color="red")
        img.save(class_1_dir / f"img_{i:03d}.jpg")

    return tmp_data_dir


@pytest.fixture(scope="function")
def mock_dataset_medium(tmp_data_dir):
    """
    Create a medium-sized mock image dataset (10 images per class).

    For integration tests that need more data.
    """
    class_0_dir = tmp_data_dir / "0.Normal"
    class_1_dir = tmp_data_dir / "1.Abnormal"
    class_0_dir.mkdir(exist_ok=True)
    class_1_dir.mkdir(exist_ok=True)

    # Create mock images for class 0
    for i in range(10):
        img = Image.new("RGB", (64, 64), color="green")
        img.save(class_0_dir / f"img_{i:03d}.jpg")

    # Create mock images for class 1
    for i in range(10, 20):
        img = Image.new("RGB", (64, 64), color="red")
        img.save(class_1_dir / f"img_{i:03d}.jpg")

    return tmp_data_dir


# ==============================================================================
# Mock Configuration Fixtures
# ==============================================================================


@pytest.fixture(scope="function")
def mock_config_classifier() -> Dict[str, Any]:
    """Return a minimal classifier configuration for testing."""
    return {
        "experiment": "classifier",
        "model": {"name": "resnet50", "pretrained": False, "num_classes": 2},
        "data": {
            "train_path": "data/train",
            "val_path": "data/val",
            "batch_size": 2,
            "num_workers": 0,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": "none",
        },
        "output": {"checkpoint_dir": "outputs/checkpoints", "log_dir": "outputs/logs"},
    }


@pytest.fixture(scope="function")
def mock_config_diffusion() -> Dict[str, Any]:
    """Return a minimal diffusion configuration for testing."""
    return {
        "experiment": "diffusion",
        "device": "cpu",
        "seed": None,
        "model": {
            "image_size": 32,
            "in_channels": 3,
            "model_channels": 32,
            "channel_multipliers": [1, 2],
            "num_classes": None,
            "num_timesteps": 100,
            "beta_schedule": "cosine",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "class_dropout_prob": 0.1,
            "use_attention": [False, True],
        },
        "data": {
            "train_path": "data/train",
            "val_path": None,
            "batch_size": 2,
            "num_workers": 0,
            "image_size": 32,
            "horizontal_flip": True,
            "rotation_degrees": 0,
            "color_jitter": False,
            "color_jitter_strength": 0.1,
            "pin_memory": True,
            "drop_last": False,
            "shuffle_train": True,
            "return_labels": False,
        },
        "output": {
            "log_dir": "outputs/logs",
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "optimizer_kwargs": {
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": None,
            "scheduler_kwargs": {},
            "use_ema": True,
            "ema_decay": 0.9999,
            "use_amp": False,
            "gradient_clip_norm": None,
            "checkpoint_dir": "outputs/checkpoints",
            "save_best_only": False,
            "save_frequency": 10,
            "validation": {
                "frequency": 1,
                "metric": "loss",
            },
            "visualization": {
                "sample_images": True,
                "sample_interval": 10,
                "samples_per_class": 2,
                "guidance_scale": 3.0,
            },
        },
        "generation": {
            "checkpoint": None,
            "num_samples": 100,
            "guidance_scale": 3.0,
            "use_ema": True,
            "output_dir": None,
            "save_grid": True,
            "grid_nrow": 10,
        },
    }


@pytest.fixture(scope="function")
def mock_config_gan() -> Dict[str, Any]:
    """Return a minimal GAN configuration for testing."""
    return {
        "experiment": "gan",
        "model": {"name": "dcgan", "latent_dim": 100, "image_size": 32, "channels": 3},
        "data": {"train_path": "data/train", "batch_size": 2, "num_workers": 0},
        "training": {
            "epochs": 2,
            "learning_rate": 0.0002,
            "optimizer": "adam",
            "beta1": 0.5,
        },
        "output": {"checkpoint_dir": "outputs/checkpoints", "log_dir": "outputs/logs"},
    }


# ==============================================================================
# Fixtures Directory Helpers
# ==============================================================================


@pytest.fixture(scope="session")
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_configs_dir(fixtures_dir):
    """Return path to fixtures/configs directory."""
    configs_dir = fixtures_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


@pytest.fixture(scope="session")
def fixtures_mock_data_dir(fixtures_dir):
    """Return path to fixtures/mock_data directory."""
    mock_data_dir = fixtures_dir / "mock_data"
    mock_data_dir.mkdir(parents=True, exist_ok=True)
    return mock_data_dir


# ==============================================================================
# Assertion Helpers
# ==============================================================================


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple):
    """Assert that a tensor has the expected shape."""
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float):
    """Assert that all tensor values are within the specified range."""
    assert tensor.min() >= min_val, f"Tensor minimum {tensor.min()} is below {min_val}"
    assert tensor.max() <= max_val, f"Tensor maximum {tensor.max()} is above {max_val}"


# Make helpers available as pytest namespace
pytest.assert_tensor_shape = assert_tensor_shape
pytest.assert_tensor_range = assert_tensor_range
