"""
Example unit tests demonstrating the testing infrastructure.

This file shows how to use pytest markers, fixtures, and organize tests
according to the four-tier testing strategy.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

# ==============================================================================
# Unit Tests (Tier 1: < 100ms per test)
# ==============================================================================


@pytest.mark.unit
def test_random_seed_fixture_works(reset_random_seeds):
    """Verify that random seeds are properly set for reproducibility."""
    # reset_random_seeds is autouse fixture, so it's applied automatically
    rand1 = torch.rand(5)

    # Reset seeds again
    torch.manual_seed(42)
    rand2 = torch.rand(5)

    # Should get same values with same seed
    assert torch.allclose(rand1, rand2), (
        "Random seed fixture should ensure reproducibility"
    )


@pytest.mark.unit
def test_device_cpu_fixture(device_cpu):
    """Verify CPU device fixture works."""
    assert device_cpu.type == "cpu"

    # Create tensor on CPU
    tensor = torch.ones(3, 3, device=device_cpu)
    assert tensor.device.type == "cpu"


@pytest.mark.unit
def test_tmp_directories_created(tmp_output_dir, tmp_data_dir):
    """Verify temporary directory fixtures create proper structure."""
    # Check output directory structure
    assert tmp_output_dir.exists()
    assert (tmp_output_dir / "checkpoints").exists()
    assert (tmp_output_dir / "logs").exists()
    assert (tmp_output_dir / "generated").exists()

    # Check data directory
    assert tmp_data_dir.exists()


@pytest.mark.unit
def test_mock_config_fixtures(
    mock_config_classifier, mock_config_diffusion, mock_config_gan
):
    """Verify mock configuration fixtures provide valid configurations."""
    # Classifier config
    assert mock_config_classifier["experiment"] == "classifier"
    assert "model" in mock_config_classifier
    assert "data" in mock_config_classifier
    assert "training" in mock_config_classifier

    # Diffusion config
    assert mock_config_diffusion["experiment"] == "diffusion"
    assert mock_config_diffusion["training"]["epochs"] == 2

    # GAN config
    assert mock_config_gan["experiment"] == "gan"
    assert mock_config_gan["data"]["batch_size"] == 2


# ==============================================================================
# Component Tests (Tier 2: 1-5 seconds per test)
# ==============================================================================


@pytest.mark.component
def test_mock_tensor_fixtures(mock_image_tensor, mock_batch_tensor, mock_labels):
    """Verify tensor fixtures provide correct shapes and types."""
    # Single image tensor
    assert mock_image_tensor.shape == (1, 3, 32, 32)
    assert mock_image_tensor.dtype == torch.float32

    # Batch tensor
    assert mock_batch_tensor.shape == (2, 3, 32, 32)
    assert mock_batch_tensor.dtype == torch.float32

    # Labels
    assert mock_labels.shape == (2,)
    assert mock_labels.dtype == torch.long
    assert torch.all((mock_labels >= 0) & (mock_labels <= 1))


@pytest.mark.component
def test_mock_pil_image_fixture(mock_pil_image):
    """Verify PIL image fixture provides valid image."""
    from PIL import Image

    assert isinstance(mock_pil_image, Image.Image)
    assert mock_pil_image.size == (32, 32)
    assert mock_pil_image.mode == "RGB"


@pytest.mark.component
def test_mock_dataset_small_fixture(mock_dataset_small):
    """Verify small mock dataset fixture creates proper structure."""
    # Check class directories exist
    class_0_dir = mock_dataset_small / "0.Normal"
    class_1_dir = mock_dataset_small / "1.Abnormal"

    assert class_0_dir.exists()
    assert class_1_dir.exists()

    # Check images were created
    class_0_images = list(class_0_dir.glob("*.jpg"))
    class_1_images = list(class_1_dir.glob("*.jpg"))

    assert len(class_0_images) == 2, "Should have 2 images in class 0"
    assert len(class_1_images) == 2, "Should have 2 images in class 1"


@pytest.mark.component
def test_mock_dataset_medium_fixture(mock_dataset_medium):
    """Verify medium mock dataset fixture creates proper structure."""
    class_0_dir = mock_dataset_medium / "0.Normal"
    class_1_dir = mock_dataset_medium / "1.Abnormal"

    # Check larger dataset
    class_0_images = list(class_0_dir.glob("*.jpg"))
    class_1_images = list(class_1_dir.glob("*.jpg"))

    assert len(class_0_images) == 10, "Should have 10 images in class 0"
    assert len(class_1_images) == 10, "Should have 10 images in class 1"


# ==============================================================================
# Integration Tests (Tier 3: 10-60 seconds per test)
# ==============================================================================


@pytest.mark.integration
def test_fixtures_directory_structure(
    fixtures_dir, fixtures_configs_dir, fixtures_mock_data_dir
):
    """Verify fixtures directory structure is properly created."""
    assert fixtures_dir.exists()
    assert fixtures_configs_dir.exists()
    assert fixtures_mock_data_dir.exists()

    # Check for config files
    classifier_config = fixtures_configs_dir / "classifier_minimal.json"
    diffusion_config = fixtures_configs_dir / "diffusion_minimal.json"
    gan_config = fixtures_configs_dir / "gan_minimal.json"

    assert classifier_config.exists(), "Classifier config should exist"
    assert diffusion_config.exists(), "Diffusion config should exist"
    assert gan_config.exists(), "GAN config should exist"


# ==============================================================================
# GPU Tests (Only run if GPU available)
# ==============================================================================


@pytest.mark.gpu
@pytest.mark.component
def test_device_gpu_fixture(device_gpu):
    """Verify GPU device fixture works (skipped if no GPU)."""
    assert device_gpu.type == "cuda"

    # Create tensor on GPU
    tensor = torch.ones(3, 3, device=device_gpu)
    assert tensor.device.type == "cuda"


@pytest.mark.component
def test_device_auto_fixture(device_auto):
    """Verify auto device fixture selects appropriate device."""
    # Should be either CPU or CUDA
    assert device_auto.type in ["cpu", "cuda"]

    # Tensor should work on selected device
    tensor = torch.ones(3, 3, device=device_auto)
    assert tensor.device.type == device_auto.type


# ==============================================================================
# Assertion Helper Tests
# ==============================================================================


@pytest.mark.unit
def test_assert_tensor_shape_helper():
    """Verify tensor shape assertion helper works."""
    tensor = torch.zeros(2, 3, 32, 32)

    # Should pass
    pytest.assert_tensor_shape(tensor, (2, 3, 32, 32))

    # Should fail
    with pytest.raises(AssertionError):
        pytest.assert_tensor_shape(tensor, (2, 3, 64, 64))


@pytest.mark.unit
def test_assert_tensor_range_helper():
    """Verify tensor range assertion helper works."""
    tensor = torch.rand(10, 10) * 2.0 - 1.0  # Range: -1 to 1

    # Should pass
    pytest.assert_tensor_range(tensor, -1.0, 1.0)

    # Should fail on min
    with pytest.raises(AssertionError):
        pytest.assert_tensor_range(tensor, 0.0, 1.0)
