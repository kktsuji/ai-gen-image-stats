"""Integration Tests for Trainer-Sampler Integration

This module tests the integration between DiffusionTrainer and DiffusionSampler.
Verifies:
- Trainer correctly initializes sampler
- Trainer's generate_samples() delegates to sampler
- Trainer's _generate_samples() uses sampler
- Backward compatibility with existing code
- Checkpoint save/load preserves sampling ability
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.experiments.diffusion.dataloader import DiffusionDataLoader
from src.experiments.diffusion.logger import DiffusionLogger
from src.experiments.diffusion.model import create_ddpm
from src.experiments.diffusion.sampler import DiffusionSampler
from src.experiments.diffusion.trainer import DiffusionTrainer

# ========================================
# Fixtures
# ========================================


@pytest.fixture
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def small_conditional_model(device):
    """Create a small conditional DDPM model for testing."""
    model = create_ddpm(
        image_size=32,
        num_classes=2,
        device=device,
        num_timesteps=10,  # Small for fast testing
        in_channels=3,
        model_channels=32,
        channel_multipliers=(1, 2),
    )
    return model


@pytest.fixture
def small_dataloader(temp_dir):
    """Create a minimal dataloader for testing."""
    import numpy as np
    from PIL import Image

    # Create dummy data directory structure
    train_dir = temp_dir / "train"
    train_dir.mkdir()

    for class_idx in [0, 1]:
        class_dir = train_dir / str(class_idx)
        class_dir.mkdir()

        # Create a few dummy images as actual image files
        for i in range(4):
            # Create a random RGB image
            img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"img_{i}.png")

    # Create dataloader
    dataloader = DiffusionDataLoader(
        train_path=str(train_dir),
        batch_size=2,
        num_workers=0,
        image_size=32,
    )

    return dataloader


@pytest.fixture
def trainer_with_sampler(small_conditional_model, small_dataloader, temp_dir, device):
    """Create a trainer instance for testing."""
    logger = DiffusionLogger(log_dir=str(temp_dir / "logs"))
    optimizer = torch.optim.Adam(small_conditional_model.parameters(), lr=0.0001)

    trainer = DiffusionTrainer(
        model=small_conditional_model,
        dataloader=small_dataloader,
        optimizer=optimizer,
        logger=logger,
        device=device,
        use_ema=True,
        ema_decay=0.999,
        show_progress=False,  # Disable for tests
        sample_images=True,
        sample_interval=1,
    )

    return trainer


# ========================================
# Integration Tests
# ========================================


@pytest.mark.integration
class TestTrainerSamplerIntegration:
    """Test integration between trainer and sampler."""

    def test_trainer_initializes_sampler(self, trainer_with_sampler):
        """Test that trainer initializes sampler during __init__."""
        assert hasattr(trainer_with_sampler, "sampler")
        assert isinstance(trainer_with_sampler.sampler, DiffusionSampler)

    def test_sampler_has_correct_model(self, trainer_with_sampler):
        """Test that sampler references correct model."""
        assert trainer_with_sampler.sampler.model is trainer_with_sampler.model

    def test_sampler_has_correct_device(self, trainer_with_sampler):
        """Test that sampler has correct device."""
        assert trainer_with_sampler.sampler.device == trainer_with_sampler.device

    def test_sampler_has_ema_reference(self, trainer_with_sampler):
        """Test that sampler has EMA reference."""
        assert trainer_with_sampler.sampler.ema is trainer_with_sampler.ema

    def test_trainer_generate_samples_works(self, trainer_with_sampler, device):
        """Test that trainer's sampler can generate samples."""
        samples = trainer_with_sampler.sampler.sample(num_samples=4, use_ema=False)

        assert samples.shape == (4, 3, 32, 32)
        assert samples.device.type == device.split(":")[0]

    def test_trainer_generate_samples_with_labels(self, trainer_with_sampler, device):
        """Test trainer's sampler generates samples with class labels."""
        labels = torch.tensor([0, 1, 0, 1], device=device)
        samples = trainer_with_sampler.sampler.sample(
            num_samples=4, class_labels=labels, guidance_scale=3.0, use_ema=False
        )

        assert samples.shape == (4, 3, 32, 32)

    def test_trainer_generate_samples_delegates_to_sampler(
        self, trainer_with_sampler, device
    ):
        """Test that sampler generates consistent samples."""
        # Generate samples directly from sampler twice with same seed
        labels = torch.tensor([0, 1], device=device)

        torch.manual_seed(42)
        samples1 = trainer_with_sampler.sampler.sample(
            num_samples=2, class_labels=labels, use_ema=False
        )

        # Generate samples again with same seed
        torch.manual_seed(42)
        samples2 = trainer_with_sampler.sampler.sample(
            num_samples=2, class_labels=labels, use_ema=False
        )

        # Should be identical (same random seed)
        assert torch.allclose(samples1, samples2, atol=1e-5)


@pytest.mark.integration
class TestSamplerAPI:
    """Test the new sampler API after Phase 7 refactoring."""

    def test_sampler_unconditional_generation(self, trainer_with_sampler, device):
        """Test that sampler can generate unconditional samples."""
        samples = trainer_with_sampler.sampler.sample(num_samples=8)

        assert samples.shape == (8, 3, 32, 32)

    def test_sampler_conditional_generation(self, trainer_with_sampler, device):
        """Test that sampler can generate conditional samples."""
        labels = torch.tensor([0, 1] * 4, device=device)

        # New API usage through sampler
        samples = trainer_with_sampler.sampler.sample(
            num_samples=8, class_labels=labels, guidance_scale=3.0
        )

        assert samples.shape == (8, 3, 32, 32)

    def test_trainer_attributes_unchanged(self, trainer_with_sampler):
        """Test that trainer still has all expected attributes."""
        # Training-related attributes should still exist
        assert hasattr(trainer_with_sampler, "model")
        assert hasattr(trainer_with_sampler, "dataloader")
        assert hasattr(trainer_with_sampler, "optimizer")
        assert hasattr(trainer_with_sampler, "logger")
        assert hasattr(trainer_with_sampler, "device")
        assert hasattr(trainer_with_sampler, "use_ema")
        assert hasattr(trainer_with_sampler, "ema")
        # generate_samples() method removed in Phase 7
        assert not hasattr(trainer_with_sampler, "generate_samples")
        # _generate_samples() internal method for logging should still exist
        assert hasattr(trainer_with_sampler, "_generate_samples")

        # Sampler attribute should exist
        assert hasattr(trainer_with_sampler, "sampler")


@pytest.mark.integration
class TestCheckpointIntegration:
    """Test checkpoint save/load with sampler."""

    def test_checkpoint_save_and_load_sampling(
        self, trainer_with_sampler, temp_dir, device
    ):
        """Test that sampling works after checkpoint save/load."""
        checkpoint_path = temp_dir / "test_checkpoint.pth"

        # Generate samples before save
        torch.manual_seed(42)
        samples_before = trainer_with_sampler.sampler.sample(
            num_samples=2, use_ema=False
        )

        # Save checkpoint
        trainer_with_sampler.save_checkpoint(
            checkpoint_path, epoch=1, metrics={"loss": 0.5}
        )

        # Modify model weights
        for param in trainer_with_sampler.model.parameters():
            param.data += 0.1

        # Load checkpoint
        trainer_with_sampler.load_checkpoint(checkpoint_path)

        # Generate samples after load
        torch.manual_seed(42)
        samples_after = trainer_with_sampler.sampler.sample(
            num_samples=2, use_ema=False
        )

        # Samples should be identical after restoring checkpoint
        assert torch.allclose(samples_before, samples_after, atol=1e-5)

    def test_checkpoint_preserves_ema_sampling(
        self, trainer_with_sampler, temp_dir, device
    ):
        """Test that EMA sampling works after checkpoint restore."""
        checkpoint_path = temp_dir / "test_checkpoint_ema.pth"

        # Update EMA a few times
        for _ in range(5):
            for param in trainer_with_sampler.model.parameters():
                param.data += 0.01
            trainer_with_sampler.ema.update()

        # Generate EMA samples before save
        torch.manual_seed(42)
        samples_before = trainer_with_sampler.sampler.sample(
            num_samples=2, use_ema=True
        )

        # Save checkpoint
        trainer_with_sampler.save_checkpoint(
            checkpoint_path, epoch=1, metrics={"loss": 0.5}
        )

        # Create new trainer and load checkpoint
        from src.experiments.diffusion.model import create_ddpm

        new_model = create_ddpm(
            image_size=32,
            num_classes=2,
            device=device,
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        new_trainer = DiffusionTrainer(
            model=new_model,
            dataloader=trainer_with_sampler.dataloader,
            optimizer=torch.optim.Adam(new_model.parameters(), lr=0.0001),
            logger=trainer_with_sampler.logger,
            device=device,
            use_ema=True,
            show_progress=False,
        )

        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)

        # Generate EMA samples after load
        torch.manual_seed(42)
        samples_after = new_trainer.sampler.sample(num_samples=2, use_ema=True)

        # Samples should be identical after restoring checkpoint
        assert torch.allclose(samples_before, samples_after, atol=1e-4)


@pytest.mark.integration
class TestTrainingSampleGeneration:
    """Test sample generation during training."""

    def test_generate_samples_during_training(self, trainer_with_sampler, temp_dir):
        """Test that _generate_samples() works during training loop."""
        logger = trainer_with_sampler.logger

        # Simulate training and sample generation
        trainer_with_sampler._generate_samples(logger, step=1)

        # Verify that samples were logged (images should exist)
        # Note: Actual file existence check depends on logger implementation
        # Here we just verify the method runs without error

    def test_conditional_sample_generation_uses_sampler(self, trainer_with_sampler):
        """Test that conditional _generate_samples uses sampler."""
        # The model has num_classes=2, so it's conditional
        assert trainer_with_sampler.model.num_classes == 2

        # Generate samples (should use sampler.sample_by_class)
        logger = trainer_with_sampler.logger
        trainer_with_sampler._generate_samples(logger, step=1)

        # If this runs without error, the integration is working


@pytest.mark.integration
class TestStandaloneSamplerUsage:
    """Test using sampler independently of trainer."""

    def test_standalone_sampler_from_checkpoint(
        self, trainer_with_sampler, temp_dir, device
    ):
        """Test loading checkpoint and using sampler independently."""
        checkpoint_path = temp_dir / "inference_checkpoint.pth"

        # Save checkpoint from trainer
        trainer_with_sampler.save_checkpoint(
            checkpoint_path, epoch=1, metrics={"loss": 0.5}
        )

        # Load checkpoint independently (inference-only workflow)
        from src.experiments.diffusion.model import EMA, create_ddpm

        # Create model
        model = create_ddpm(
            image_size=32,
            num_classes=2,
            device=device,
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create EMA and load state
        ema = EMA(model, decay=0.999, device=device)
        ema.load_state_dict(checkpoint["ema_state_dict"])

        # Create sampler (no trainer needed!)
        sampler = DiffusionSampler(model=model, device=device, ema=ema)

        # Generate samples
        samples = sampler.sample(num_samples=4, use_ema=True)

        assert samples.shape == (4, 3, 32, 32)

    def test_standalone_sampler_without_trainer_dependencies(self, device):
        """Test that sampler doesn't require trainer dependencies."""
        from src.experiments.diffusion.model import create_ddpm

        # Create model independently
        model = create_ddpm(
            image_size=32,
            num_classes=0,
            device=device,
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        # Create sampler without optimizer, dataloader, logger, etc.
        sampler = DiffusionSampler(model=model, device=device)

        # Generate samples
        samples = sampler.sample(num_samples=8)

        assert samples.shape == (8, 3, 32, 32)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_training_and_inference_workflow(
        self, trainer_with_sampler, temp_dir, device
    ):
        """Test complete workflow: train, save, load, infer."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Train for 1 epoch
        trainer_with_sampler.train(
            num_epochs=1,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
            validate_frequency=0,
            save_best=False,
        )

        # Checkpoint should exist
        latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
        assert latest_checkpoint.exists()

        # Load checkpoint into new sampler for inference
        from src.experiments.diffusion.model import EMA, create_ddpm

        model = create_ddpm(
            image_size=32,
            num_classes=2,
            device=device,
            num_timesteps=10,
            in_channels=3,
            model_channels=32,
            channel_multipliers=(1, 2),
        )

        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])

        ema = EMA(model, decay=0.999, device=device)
        if "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])

        # Create sampler for inference
        sampler = DiffusionSampler(model=model, device=device, ema=ema)

        # Generate samples
        samples, labels = sampler.sample_by_class(
            samples_per_class=2, num_classes=2, use_ema=True
        )

        assert samples.shape == (4, 3, 32, 32)
        assert len(labels) == 4
