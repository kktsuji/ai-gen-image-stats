"""
Unit tests for data transforms module.

Tests cover normalization, augmentation, and transform composition.
All tests run on CPU without requiring GPU.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data.transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    clamp_image,
    denormalize_image,
    get_base_transforms,
    get_denormalization_transform,
    get_diffusion_transforms,
    get_gan_transforms,
    get_normalization_transform,
    get_train_transforms,
    get_val_transforms,
)


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    # Create a simple 256x256 RGB image
    array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(array)


@pytest.fixture
def sample_tensor():
    """Create a sample tensor image for testing."""
    # Create a normalized tensor image (C, H, W)
    return torch.randn(3, 224, 224)


@pytest.mark.unit
class TestNormalizationConstants:
    """Test that normalization constants are correct."""

    def test_imagenet_mean_length(self):
        assert len(IMAGENET_MEAN) == 3

    def test_imagenet_std_length(self):
        assert len(IMAGENET_STD) == 3

    def test_cifar10_mean_length(self):
        assert len(CIFAR10_MEAN) == 3

    def test_cifar10_std_length(self):
        assert len(CIFAR10_STD) == 3

    def test_imagenet_mean_range(self):
        assert all(0 <= m <= 1 for m in IMAGENET_MEAN)

    def test_imagenet_std_positive(self):
        assert all(s > 0 for s in IMAGENET_STD)


@pytest.mark.unit
class TestGetNormalizationTransform:
    """Test normalization transform creation."""

    def test_imagenet_normalization(self):
        norm = get_normalization_transform("imagenet")
        assert isinstance(norm, transforms.Normalize)

    def test_cifar10_normalization(self):
        norm = get_normalization_transform("cifar10")
        assert isinstance(norm, transforms.Normalize)

    def test_custom_normalization(self):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        norm = get_normalization_transform("custom", mean=mean, std=std)
        assert isinstance(norm, transforms.Normalize)

    def test_custom_without_mean_raises_error(self):
        with pytest.raises(ValueError, match="Custom normalization requires"):
            get_normalization_transform("custom", std=[0.5, 0.5, 0.5])

    def test_custom_without_std_raises_error(self):
        with pytest.raises(ValueError, match="Custom normalization requires"):
            get_normalization_transform("custom", mean=[0.5, 0.5, 0.5])

    def test_unknown_dataset_raises_error(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_normalization_transform("unknown_dataset")

    def test_case_insensitive(self):
        norm1 = get_normalization_transform("ImageNet")
        norm2 = get_normalization_transform("IMAGENET")
        assert isinstance(norm1, transforms.Normalize)
        assert isinstance(norm2, transforms.Normalize)


@pytest.mark.unit
class TestGetDenormalizationTransform:
    """Test denormalization transform creation."""

    def test_imagenet_denormalization(self):
        denorm = get_denormalization_transform("imagenet")
        assert isinstance(denorm, transforms.Normalize)

    def test_cifar10_denormalization(self):
        denorm = get_denormalization_transform("cifar10")
        assert isinstance(denorm, transforms.Normalize)

    def test_custom_denormalization(self):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        denorm = get_denormalization_transform("custom", mean=mean, std=std)
        assert isinstance(denorm, transforms.Normalize)

    def test_unknown_dataset_raises_error(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_denormalization_transform("unknown_dataset")


@pytest.mark.unit
class TestGetBaseTransforms:
    """Test base transform composition."""

    def test_default_base_transforms(self, sample_image):
        transform = get_base_transforms()
        output = transform(sample_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_base_transforms_with_resize(self, sample_image):
        transform = get_base_transforms(resize_mode="resize", crop_size=128)
        output = transform(sample_image)
        assert output.shape == (3, 128, 128)

    def test_base_transforms_with_resize_crop(self, sample_image):
        transform = get_base_transforms(
            image_size=256, crop_size=200, resize_mode="resize_crop"
        )
        output = transform(sample_image)
        assert output.shape == (3, 200, 200)

    def test_invalid_resize_mode_raises_error(self):
        with pytest.raises(ValueError, match="Unknown resize_mode"):
            get_base_transforms(resize_mode="invalid")

    def test_output_in_valid_range(self, sample_image):
        transform = get_base_transforms()
        output = transform(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0


@pytest.mark.unit
class TestGetTrainTransforms:
    """Test training transform composition."""

    def test_default_train_transforms(self, sample_image):
        transform = get_train_transforms()
        output = transform(sample_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_train_transforms_with_normalization(self, sample_image):
        transform = get_train_transforms(normalize="imagenet")
        output = transform(sample_image)
        assert output.shape == (3, 224, 224)

    def test_train_transforms_with_all_augmentations(self, sample_image):
        transform = get_train_transforms(
            horizontal_flip=True,
            color_jitter=True,
            rotation_degrees=15,
        )
        output = transform(sample_image)
        assert output.shape == (3, 224, 224)

    def test_train_transforms_custom_size(self, sample_image):
        transform = get_train_transforms(image_size=128, crop_size=96)
        output = transform(sample_image)
        assert output.shape == (3, 96, 96)

    def test_train_transforms_deterministic_when_no_augmentation(self, sample_image):
        # With no random augmentations, should be deterministic
        transform = get_train_transforms(
            horizontal_flip=False,
            color_jitter=False,
            rotation_degrees=0,
        )
        # This test mainly checks that the function works without augmentations
        output = transform(sample_image)
        assert isinstance(output, torch.Tensor)


@pytest.mark.unit
class TestGetValTransforms:
    """Test validation transform composition."""

    def test_default_val_transforms(self, sample_image):
        transform = get_val_transforms()
        output = transform(sample_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_val_transforms_with_normalization(self, sample_image):
        transform = get_val_transforms(normalize="cifar10")
        output = transform(sample_image)
        assert output.shape == (3, 224, 224)

    def test_val_transforms_custom_size(self, sample_image):
        transform = get_val_transforms(image_size=200, crop_size=180)
        output = transform(sample_image)
        assert output.shape == (3, 180, 180)

    def test_val_transforms_deterministic(self, sample_image):
        # Validation transforms should be deterministic
        transform = get_val_transforms()
        output1 = transform(sample_image)
        output2 = transform(sample_image)
        assert torch.allclose(output1, output2)


@pytest.mark.unit
class TestGetDiffusionTransforms:
    """Test diffusion model transform composition."""

    def test_default_diffusion_transforms(self, sample_image):
        transform = get_diffusion_transforms()
        output = transform(sample_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 64, 64)

    def test_diffusion_transforms_custom_size(self, sample_image):
        transform = get_diffusion_transforms(image_size=128)
        output = transform(sample_image)
        assert output.shape == (3, 128, 128)

    def test_diffusion_transforms_without_flip(self, sample_image):
        transform = get_diffusion_transforms(horizontal_flip=False)
        output = transform(sample_image)
        assert output.shape == (3, 64, 64)

    def test_diffusion_output_range(self, sample_image):
        # Diffusion transforms don't normalize, so output should be in [0, 1]
        transform = get_diffusion_transforms()
        output = transform(sample_image)
        assert output.min() >= 0.0
        assert output.max() <= 1.0


@pytest.mark.unit
class TestGetGANTransforms:
    """Test GAN transform composition."""

    def test_default_gan_transforms(self, sample_image):
        transform = get_gan_transforms()
        output = transform(sample_image)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 64, 64)

    def test_gan_transforms_custom_size(self, sample_image):
        transform = get_gan_transforms(image_size=128)
        output = transform(sample_image)
        assert output.shape == (3, 128, 128)

    def test_gan_output_range(self, sample_image):
        # GAN transforms normalize to [-1, 1]
        transform = get_gan_transforms()
        output = transform(sample_image)
        # Output should be approximately in [-1, 1] range
        assert output.min() >= -1.1  # Allow small margin
        assert output.max() <= 1.1

    def test_gan_transforms_without_flip(self, sample_image):
        transform = get_gan_transforms(horizontal_flip=False)
        output = transform(sample_image)
        assert output.shape == (3, 64, 64)


@pytest.mark.unit
class TestDenormalizeImage:
    """Test image denormalization utility."""

    def test_denormalize_3d_tensor(self):
        # Create a normalized tensor
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        image = torch.randn(3, 32, 32)

        denorm = denormalize_image(image, mean, std)

        assert denorm.shape == image.shape
        assert isinstance(denorm, torch.Tensor)

    def test_denormalize_4d_tensor(self):
        # Create a batch of normalized tensors
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        image = torch.randn(8, 3, 32, 32)

        denorm = denormalize_image(image, mean, std)

        assert denorm.shape == image.shape
        assert isinstance(denorm, torch.Tensor)

    def test_denormalize_invalid_dimension_raises_error(self):
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        image = torch.randn(32, 32)  # 2D tensor

        with pytest.raises(ValueError, match="Expected 3D or 4D tensor"):
            denormalize_image(image, mean, std)

    def test_denormalize_reverses_normalization(self):
        # Test that denormalization reverses normalization
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]

        # Create original tensor in [0, 1]
        original = torch.rand(3, 32, 32)

        # Normalize
        norm_transform = transforms.Normalize(mean=mean, std=std)
        normalized = norm_transform(original)

        # Denormalize
        denormalized = denormalize_image(normalized, mean, std)

        # Should be close to original
        assert torch.allclose(denormalized, original, atol=1e-6)


@pytest.mark.unit
class TestClampImage:
    """Test image clamping utility."""

    def test_clamp_within_range(self):
        image = torch.tensor([[[0.5, 0.8], [0.2, 0.9]]])
        clamped = clamp_image(image)
        assert torch.all(clamped >= 0.0)
        assert torch.all(clamped <= 1.0)
        assert torch.allclose(clamped, image)

    def test_clamp_above_range(self):
        image = torch.tensor([[[1.5, 2.0], [0.5, 1.2]]])
        clamped = clamp_image(image)
        assert torch.all(clamped >= 0.0)
        assert torch.all(clamped <= 1.0)
        assert torch.max(clamped) == 1.0

    def test_clamp_below_range(self):
        image = torch.tensor([[[-0.5, -1.0], [0.5, 0.8]]])
        clamped = clamp_image(image)
        assert torch.all(clamped >= 0.0)
        assert torch.all(clamped <= 1.0)
        assert torch.min(clamped) == 0.0

    def test_clamp_mixed_range(self):
        image = torch.tensor([[[-0.5, 1.5], [0.3, 0.7]]])
        clamped = clamp_image(image)
        expected = torch.tensor([[[0.0, 1.0], [0.3, 0.7]]])
        assert torch.allclose(clamped, expected)

    def test_clamp_preserves_shape(self):
        shapes = [(3, 32, 32), (1, 64, 64), (4, 3, 128, 128)]
        for shape in shapes:
            image = torch.randn(*shape)
            clamped = clamp_image(image)
            assert clamped.shape == image.shape


@pytest.mark.component
class TestTransformIntegration:
    """Test transform pipelines with integration scenarios."""

    def test_train_val_consistency(self, sample_image):
        """Ensure train and val transforms produce compatible outputs."""
        train_transform = get_train_transforms(
            image_size=256,
            crop_size=224,
            horizontal_flip=False,
            color_jitter=False,
            rotation_degrees=0,
            normalize="imagenet",
        )
        val_transform = get_val_transforms(
            image_size=256,
            crop_size=224,
            normalize="imagenet",
        )

        train_output = train_transform(sample_image)
        val_output = val_transform(sample_image)

        assert train_output.shape == val_output.shape

    def test_normalization_denormalization_roundtrip(self, sample_tensor):
        """Test that normalize -> denormalize is close to identity."""
        # Apply normalization
        norm = get_normalization_transform("imagenet")
        normalized = norm(sample_tensor)

        # Apply denormalization
        denorm = get_denormalization_transform("imagenet")
        denormalized = denorm(normalized)

        # Should be close to original
        assert torch.allclose(denormalized, sample_tensor, atol=1e-5)

    def test_pipeline_with_real_workflow(self, sample_image):
        """Test a realistic preprocessing pipeline."""
        # Simulating classifier training workflow
        transform = get_train_transforms(
            image_size=256,
            crop_size=224,
            horizontal_flip=True,
            color_jitter=True,
            rotation_degrees=10,
            normalize="imagenet",
        )

        # Should handle multiple applications
        for _ in range(5):
            output = transform(sample_image)
            assert output.shape == (3, 224, 224)
            # Normalized outputs may be outside [0, 1]
            assert torch.isfinite(output).all()
