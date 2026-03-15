"""Tests for Diffusion Sampler Functions

This module contains unit and integration tests for the sampler functions.
Tests verify:
- Unconditional and conditional sampling
- EMA weight management
- Guidance scale application
- Device handling
- Input validation and error handling
"""

from unittest.mock import patch

import pytest
import torch

from src.experiments.diffusion.model import EMA, create_ddpm
from src.experiments.diffusion.sampler import (
    sample,
    sample_by_class,
    sample_with_intermediates,
)

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def unconditional_model(device):
    """Create a small unconditional DDPM model for testing."""
    model = create_ddpm(
        image_size=32,
        num_classes=0,  # Unconditional
        device=device,
        num_timesteps=10,  # Small for fast testing
        in_channels=3,
        model_channels=32,  # Small model
        channel_multipliers=(1, 2),
    )
    return model


@pytest.fixture
def conditional_model(device):
    """Create a small conditional DDPM model for testing."""
    model = create_ddpm(
        image_size=32,
        num_classes=2,  # Binary classification
        device=device,
        num_timesteps=10,  # Small for fast testing
        in_channels=3,
        model_channels=32,  # Small model
        channel_multipliers=(1, 2),
    )
    return model


@pytest.fixture
def ema_model(conditional_model, device):
    """Create EMA wrapper for testing."""
    ema = EMA(conditional_model, decay=0.999, device=device)
    return ema


# ========================================
# Unit Tests
# ========================================


@pytest.mark.unit
class TestUnconditionalSampling:
    """Test unconditional sample generation."""

    def test_sample_shape(self, unconditional_model, device):
        """Test that generated samples have correct shape."""
        num_samples = 4
        samples = sample(
            unconditional_model, device, num_samples=num_samples, use_ema=False
        )

        assert samples.shape == (num_samples, 3, 32, 32)
        assert samples.device.type == device.split(":")[0]

    def test_sample_value_range(self, unconditional_model, device):
        """Test that samples are in expected range [-1, 1]."""
        samples = sample(unconditional_model, device, num_samples=2, use_ema=False)

        # Diffusion models typically output in [-1, 1] range
        # Allow some tolerance for numerical precision
        assert samples.min() >= -1.1
        assert samples.max() <= 1.1

    def test_sample_deterministic_with_seed(self, unconditional_model, device):
        """Test that sampling is deterministic with fixed seed."""
        # Generate samples with same seed
        torch.manual_seed(42)
        samples1 = sample(unconditional_model, device, num_samples=2, use_ema=False)

        torch.manual_seed(42)
        samples2 = sample(unconditional_model, device, num_samples=2, use_ema=False)

        assert torch.allclose(samples1, samples2, atol=1e-5)

    def test_sample_different_batch_sizes(self, unconditional_model, device):
        """Test sampling with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            samples = sample(
                unconditional_model, device, num_samples=batch_size, use_ema=False
            )
            assert samples.shape[0] == batch_size


@pytest.mark.unit
class TestConditionalSampling:
    """Test conditional sample generation."""

    def test_conditional_sample_shape(self, conditional_model, device):
        """Test conditional sampling with correct shape."""
        num_samples = 4
        class_labels = torch.randint(0, 2, (num_samples,), device=device)
        samples = sample(
            conditional_model,
            device,
            num_samples=num_samples,
            class_labels=class_labels,
            use_ema=False,
        )

        assert samples.shape == (num_samples, 3, 32, 32)

    def test_conditional_sample_with_labels(self, conditional_model, device):
        """Test that conditional sampling accepts class labels."""
        class_labels = torch.tensor([0, 1, 0, 1], device=device)
        samples = sample(
            conditional_model,
            device,
            num_samples=4,
            class_labels=class_labels,
            use_ema=False,
        )

        assert samples.shape[0] == 4

    def test_conditional_sample_guidance_scale(self, conditional_model, device):
        """Test that guidance scale parameter is accepted."""
        class_labels = torch.tensor([0, 1], device=device)

        # Test with different guidance scales
        for guidance_scale in [0.0, 1.0, 3.0]:
            samples = sample(
                conditional_model,
                device,
                num_samples=2,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                use_ema=False,
            )
            assert samples.shape == (2, 3, 32, 32)

    def test_conditional_sample_invalid_labels_shape(self, conditional_model, device):
        """Test that invalid label shape raises error."""
        # Wrong number of labels
        class_labels = torch.tensor([0, 1], device=device)

        with pytest.raises(ValueError, match="class_labels length"):
            sample(
                conditional_model,
                device,
                num_samples=4,  # Mismatch with labels
                class_labels=class_labels,
                use_ema=False,
            )

    def test_conditional_sample_labels_moved_to_device(self, conditional_model, device):
        """Test that labels are moved to correct device."""
        # Create labels on CPU
        class_labels = torch.tensor([0, 1, 0, 1])

        # Should not raise error - function should move labels to device
        samples = sample(
            conditional_model,
            device,
            num_samples=4,
            class_labels=class_labels,
            use_ema=False,
        )

        assert samples.shape[0] == 4


@pytest.mark.unit
class TestEMAWeightManagement:
    """Test EMA weight application and restoration."""

    def test_sample_with_ema_enabled(self, conditional_model, ema_model, device):
        """Test sampling with EMA weights."""
        # Store original param values
        original_params = {
            name: param.clone() for name, param in conditional_model.named_parameters()
        }

        # Generate samples with EMA.
        # Mock model.sample() to skip the expensive diffusion loop — this test
        # only needs to verify that EMA weights are applied and then restored.
        fake_samples = torch.zeros(2, 3, 32, 32, device=device)
        with patch.object(conditional_model, "sample", return_value=fake_samples):
            sample(
                conditional_model,
                device,
                num_samples=2,
                use_ema=True,
                ema=ema_model,
            )

        # After sampling, original weights should be restored
        for name, param in conditional_model.named_parameters():
            assert torch.allclose(param, original_params[name])

    def test_sample_without_ema(self, conditional_model, device):
        """Test sampling without EMA (no EMA instance)."""
        # Should work fine without EMA
        samples = sample(
            conditional_model, device, num_samples=2, use_ema=True, ema=None
        )
        assert samples.shape == (2, 3, 32, 32)

    def test_sample_ema_disabled(self, conditional_model, ema_model, device):
        """Test sampling with EMA available but disabled."""
        # Sample with use_ema=False
        samples = sample(
            conditional_model, device, num_samples=2, use_ema=False, ema=ema_model
        )
        assert samples.shape == (2, 3, 32, 32)


@pytest.mark.unit
class TestSampleByClass:
    """Test sample_by_class function."""

    def test_sample_by_class_shape(self, conditional_model, device):
        """Test that sample_by_class generates correct number of samples."""
        samples_per_class = 2
        num_classes = 2

        samples, labels = sample_by_class(
            conditional_model,
            device,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            use_ema=False,
        )

        expected_total = samples_per_class * num_classes
        assert samples.shape == (expected_total, 3, 32, 32)
        assert len(labels) == expected_total

    def test_sample_by_class_labels_order(self, conditional_model, device):
        """Test that labels are in correct order."""
        samples_per_class = 3
        num_classes = 2

        samples, labels = sample_by_class(
            conditional_model,
            device,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            use_ema=False,
        )

        # Labels should be: [0, 0, 0, 1, 1, 1]
        expected_labels = [0, 0, 0, 1, 1, 1]
        assert labels == expected_labels

    def test_sample_by_class_with_guidance(self, conditional_model, device):
        """Test sample_by_class with guidance scale."""
        samples, labels = sample_by_class(
            conditional_model,
            device,
            samples_per_class=2,
            num_classes=2,
            guidance_scale=3.0,
            use_ema=False,
        )

        assert samples.shape == (4, 3, 32, 32)
        assert len(labels) == 4

    def test_sample_by_class_many_classes(self, device):
        """Test sample_by_class with more classes."""
        # Create model with more classes
        model = create_ddpm(
            image_size=32,
            num_classes=5,
            device=device,
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        samples_per_class = 2
        num_classes = 5

        samples, labels = sample_by_class(
            model,
            device,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            use_ema=False,
        )

        assert samples.shape == (10, 3, 32, 32)
        assert len(labels) == 10
        assert labels == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


@pytest.mark.unit
class TestDeviceHandling:
    """Test device management in sampler functions."""

    def test_sample_on_correct_device(self, unconditional_model, device):
        """Test that samples are generated on correct device."""
        samples = sample(unconditional_model, device, num_samples=2, use_ema=False)

        assert str(samples.device).startswith(device.split(":")[0])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sample_cuda_device(self):
        """Test sampling on CUDA device."""
        model = create_ddpm(
            image_size=32,
            num_classes=0,
            device="cuda",
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        samples = sample(model, "cuda", num_samples=2, use_ema=False)

        assert samples.device.type == "cuda"

    def test_sample_cpu_device(self):
        """Test sampling on CPU device."""
        model = create_ddpm(
            image_size=32,
            num_classes=0,
            device="cpu",
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        samples = sample(model, "cpu", num_samples=2, use_ema=False)

        assert samples.device.type == "cpu"


@pytest.mark.unit
class TestInputValidation:
    """Test input validation and error handling."""

    def test_invalid_num_samples_type(self, unconditional_model, device):
        """Test that invalid num_samples type is handled."""
        # This should work - pytest will catch if it raises TypeError
        with pytest.raises(TypeError):
            sample(unconditional_model, device, num_samples="invalid", use_ema=False)  # type: ignore[arg-type]

    def test_negative_guidance_scale(self, conditional_model, device):
        """Test that negative guidance scale is handled (may be valid)."""
        class_labels = torch.tensor([0, 1], device=device)

        # Negative guidance scale might be valid in some contexts
        # Just verify it doesn't crash
        samples = sample(
            conditional_model,
            device,
            num_samples=2,
            class_labels=class_labels,
            guidance_scale=-1.0,
            use_ema=False,
        )

        assert samples.shape == (2, 3, 32, 32)


# ========================================
# Integration Tests
# ========================================


@pytest.mark.integration
class TestSamplerIntegration:
    """Integration tests for sampler with real workflows."""

    def test_end_to_end_unconditional(self, unconditional_model, device):
        """Test complete unconditional sampling workflow."""
        samples = sample(unconditional_model, device, num_samples=8, use_ema=False)

        # Verify output
        assert samples.shape == (8, 3, 32, 32)
        assert samples.min() >= -1.1
        assert samples.max() <= 1.1

    def test_end_to_end_conditional(self, conditional_model, ema_model, device):
        """Test complete conditional sampling workflow with EMA."""
        # Generate class-balanced samples
        samples, labels = sample_by_class(
            conditional_model,
            device,
            samples_per_class=4,
            num_classes=2,
            guidance_scale=3.0,
            use_ema=True,
            ema=ema_model,
        )

        # Verify output
        assert samples.shape == (8, 3, 32, 32)
        assert len(labels) == 8
        assert labels[:4] == [0, 0, 0, 0]
        assert labels[4:] == [1, 1, 1, 1]

    def test_batch_generation_workflow(self, conditional_model, device):
        """Test generating multiple batches."""
        all_samples = []
        for class_idx in [0, 1]:
            labels = torch.full((4,), class_idx, dtype=torch.long, device=device)
            samples = sample(
                conditional_model,
                device,
                num_samples=4,
                class_labels=labels,
                use_ema=False,
            )
            all_samples.append(samples)

        combined = torch.cat(all_samples, dim=0)
        assert combined.shape == (8, 3, 32, 32)


# ========================================
# Performance Tests (Optional)
# ========================================


@pytest.mark.slow
class TestSamplerPerformance:
    """Performance tests for sampler (marked as slow)."""

    def test_large_batch_sampling(self, unconditional_model, device):
        """Test sampling with larger batch size."""
        samples = sample(unconditional_model, device, num_samples=32, use_ema=False)

        assert samples.shape == (32, 3, 32, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sampling_speed(self):
        """Test that CUDA sampling is reasonably fast."""
        import time

        model = create_ddpm(
            image_size=32,
            num_classes=0,
            device="cuda",
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        # Warm-up
        _ = sample(model, "cuda", num_samples=2, use_ema=False)

        # Time the sampling
        start = time.time()
        _ = sample(model, "cuda", num_samples=16, use_ema=False)
        elapsed = time.time() - start

        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 30.0  # 30 seconds should be enough for small model


@pytest.mark.unit
class TestTrainingModeRestoration:
    """Test that sampling functions restore model.training state."""

    def test_sample_restores_training_mode(self, unconditional_model, device):
        """sample() restores model to training mode when it was training before."""
        unconditional_model.train()
        assert unconditional_model.training is True

        fake_samples = torch.zeros(2, 3, 32, 32, device=device)
        with patch.object(unconditional_model, "sample", return_value=fake_samples):
            sample(unconditional_model, device, num_samples=2, use_ema=False)

        assert unconditional_model.training is True

    def test_sample_preserves_eval_mode(self, unconditional_model, device):
        """sample() keeps model in eval mode when it was eval before."""
        unconditional_model.eval()
        assert unconditional_model.training is False

        fake_samples = torch.zeros(2, 3, 32, 32, device=device)
        with patch.object(unconditional_model, "sample", return_value=fake_samples):
            sample(unconditional_model, device, num_samples=2, use_ema=False)

        assert unconditional_model.training is False

    def test_sample_with_intermediates_restores_training_mode(
        self, unconditional_model, device
    ):
        """sample_with_intermediates() restores training mode."""
        unconditional_model.train()
        assert unconditional_model.training is True

        # return_intermediates=True returns (T+1, N, C, H, W)
        fake_steps = torch.zeros(11, 2, 3, 32, 32, device=device)
        with patch.object(unconditional_model, "sample", return_value=fake_steps):
            sample_with_intermediates(
                unconditional_model, device, num_samples=2, use_ema=False
            )

        assert unconditional_model.training is True

    def test_sample_restores_training_mode_on_error(self, unconditional_model, device):
        """sample() restores training mode even when model.sample() raises."""
        unconditional_model.train()

        with patch.object(
            unconditional_model, "sample", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                sample(unconditional_model, device, num_samples=2, use_ema=False)

        assert unconditional_model.training is True


@pytest.mark.unit
class TestSampleWithIntermediatesValidation:
    """Test input validation for sample_with_intermediates."""

    def test_invalid_class_labels_length(self, conditional_model, device):
        """sample_with_intermediates() raises ValueError for mismatched labels."""
        class_labels = torch.tensor([0, 1], device=device)

        with pytest.raises(ValueError, match="class_labels length"):
            sample_with_intermediates(
                conditional_model,
                device,
                num_samples=4,  # Mismatch with labels
                class_labels=class_labels,
                use_ema=False,
            )


@pytest.mark.unit
class TestSampleRejectsShowProgress:
    """Test that show_progress is no longer accepted by sample functions."""

    def test_sample_rejects_show_progress_kwarg(self, unconditional_model, device):
        """sample() raises TypeError when show_progress is passed."""
        with pytest.raises(TypeError):
            sample(
                unconditional_model,
                device,
                num_samples=1,
                use_ema=False,
                show_progress=True,  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
            )
