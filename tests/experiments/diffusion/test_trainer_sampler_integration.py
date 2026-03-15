"""Integration Tests for Trainer-Sampler Integration

This module tests the integration between DiffusionTrainer and sampler functions.
Verifies:
- Trainer's _generate_samples() uses sampler functions
- Backward compatibility with existing code
- Checkpoint save/load preserves sampling ability
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.experiments.diffusion.model import create_ddpm
from src.experiments.diffusion.sampler import sample, sample_by_class
from src.experiments.diffusion.trainer import DiffusionTrainer
from src.utils.data.loaders import create_train_loader
from src.utils.experiment_logger import ExperimentLogger

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
    import json

    import numpy as np
    from PIL import Image

    # Create dummy data directory structure
    train_dir = temp_dir / "train"
    train_dir.mkdir()

    train_entries = []
    for class_idx in [0, 1]:
        class_dir = train_dir / str(class_idx)
        class_dir.mkdir()

        # Create a few dummy images as actual image files
        for i in range(4):
            # Create a random RGB image
            img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = class_dir / f"img_{i}.png"
            img.save(img_path)
            train_entries.append({"path": str(img_path), "label": class_idx})

    # Create split JSON file
    split_file = temp_dir / "split.json"
    split_data = {
        "metadata": {"classes": {"0": 0, "1": 1}},
        "train": train_entries,
        "val": [],
    }
    split_file.write_text(json.dumps(split_data))

    # Create train loader
    from src.utils.data.transforms import get_diffusion_val_transforms

    transform = get_diffusion_val_transforms(image_size=32)
    train_loader = create_train_loader(
        split_file=str(split_file),
        batch_size=2,
        transform=transform,
        num_workers=0,
    )

    return train_loader


@pytest.fixture
def trainer_instance(small_conditional_model, small_dataloader, temp_dir, device):
    """Create a trainer instance for testing."""
    logger = ExperimentLogger(
        log_dir=str(temp_dir / "logs"),
        subdirs={"images": "samples", "denoising": "denoising"},
    )
    optimizer = torch.optim.Adam(small_conditional_model.parameters(), lr=0.0001)

    trainer = DiffusionTrainer(
        model=small_conditional_model,
        train_loader=small_dataloader,
        optimizer=optimizer,
        logger=logger,
        device=device,
        use_ema=True,
        ema_decay=0.999,
        show_progress=False,  # Disable for tests
        log_images_interval=1,
        log_denoising_interval=1,
    )

    return trainer


# ========================================
# Integration Tests
# ========================================


@pytest.mark.integration
class TestTrainerSamplerIntegration:
    """Test integration between trainer and sampler functions."""

    def test_trainer_no_sampler_attribute(self, trainer_instance):
        """Test that trainer no longer has a sampler attribute."""
        assert not hasattr(trainer_instance, "sampler")

    def test_trainer_can_generate_samples_via_function(self, trainer_instance, device):
        """Test that sampler functions work with trainer's model."""
        samples = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=4,
            use_ema=False,
        )

        assert samples.shape == (4, 3, 32, 32)
        assert samples.device.type == device.split(":")[0]

    def test_trainer_can_generate_samples_with_labels(self, trainer_instance, device):
        """Test sampler functions with trainer's model and class labels."""
        labels = torch.tensor([0, 1, 0, 1], device=device)
        samples = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=4,
            class_labels=labels,
            guidance_scale=3.0,
            use_ema=False,
        )

        assert samples.shape == (4, 3, 32, 32)

    def test_sampler_function_deterministic(self, trainer_instance, device):
        """Test that sampler functions produce consistent samples."""
        labels = torch.tensor([0, 1], device=device)

        torch.manual_seed(42)
        samples1 = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=2,
            class_labels=labels,
            use_ema=False,
        )

        torch.manual_seed(42)
        samples2 = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=2,
            class_labels=labels,
            use_ema=False,
        )

        assert torch.allclose(samples1, samples2, atol=1e-5)


@pytest.mark.integration
class TestSamplerAPI:
    """Test the sampler function API."""

    def test_unconditional_generation(self, trainer_instance, device):
        """Test unconditional sample generation."""
        samples = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=8,
        )

        assert samples.shape == (8, 3, 32, 32)

    def test_conditional_generation(self, trainer_instance, device):
        """Test conditional sample generation."""
        labels = torch.tensor([0, 1] * 4, device=device)

        samples = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=8,
            class_labels=labels,
            guidance_scale=3.0,
        )

        assert samples.shape == (8, 3, 32, 32)

    def test_trainer_attributes_unchanged(self, trainer_instance):
        """Test that trainer still has all expected attributes."""
        # Training-related attributes should still exist
        assert hasattr(trainer_instance, "model")
        assert hasattr(trainer_instance, "train_loader")
        assert hasattr(trainer_instance, "optimizer")
        assert hasattr(trainer_instance, "logger")
        assert hasattr(trainer_instance, "device")
        assert hasattr(trainer_instance, "use_ema")
        assert hasattr(trainer_instance, "ema")
        # generate_samples() method removed in Phase 7
        assert not hasattr(trainer_instance, "generate_samples")
        # _generate_samples() internal method for logging should still exist
        assert hasattr(trainer_instance, "_generate_samples")

        # Sampler attribute should NOT exist (converted to functions)
        assert not hasattr(trainer_instance, "sampler")


@pytest.mark.integration
class TestCheckpointIntegration:
    """Test checkpoint save/load with sampler functions."""

    def test_checkpoint_save_and_load_sampling(
        self, trainer_instance, temp_dir, device
    ):
        """Test that sampling works after checkpoint save/load."""
        checkpoint_path = temp_dir / "test_checkpoint.pth"

        # Generate samples before save
        torch.manual_seed(42)
        samples_before = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=2,
            use_ema=False,
        )

        # Save checkpoint
        trainer_instance.save_checkpoint(
            checkpoint_path, epoch=1, metrics={"loss": 0.5}
        )

        # Modify model weights
        for param in trainer_instance.model.parameters():
            param.data += 0.1

        # Load checkpoint
        trainer_instance.load_checkpoint(checkpoint_path)

        # Generate samples after load
        torch.manual_seed(42)
        samples_after = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=2,
            use_ema=False,
        )

        # Samples should be identical after restoring checkpoint
        assert torch.allclose(samples_before, samples_after, atol=1e-5)

    def test_checkpoint_preserves_ema_sampling(
        self, trainer_instance, temp_dir, device
    ):
        """Test that EMA sampling works after checkpoint restore."""
        checkpoint_path = temp_dir / "test_checkpoint_ema.pth"

        # Update EMA a few times
        for _ in range(5):
            for param in trainer_instance.model.parameters():
                param.data += 0.01
            trainer_instance.ema.update()

        # Generate EMA samples before save
        torch.manual_seed(42)
        samples_before = sample(
            trainer_instance.model,
            trainer_instance.device,
            num_samples=2,
            use_ema=True,
            ema=trainer_instance.ema,
        )

        # Save checkpoint
        trainer_instance.save_checkpoint(
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
            train_loader=trainer_instance.train_loader,
            optimizer=torch.optim.Adam(new_model.parameters(), lr=0.0001),
            logger=trainer_instance.logger,
            device=device,
            use_ema=True,
            show_progress=False,
        )

        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)

        # Generate EMA samples after load
        torch.manual_seed(42)
        samples_after = sample(
            new_trainer.model,
            new_trainer.device,
            num_samples=2,
            use_ema=True,
            ema=new_trainer.ema,
        )

        # Samples should be identical after restoring checkpoint
        assert torch.allclose(samples_before, samples_after, atol=1e-4)


@pytest.mark.integration
class TestTrainingSampleGeneration:
    """Test sample generation during training."""

    def test_generate_samples_during_training(self, trainer_instance, temp_dir):
        """Test that _generate_samples() works during training loop."""
        logger = trainer_instance.logger

        # Simulate training and sample generation
        trainer_instance._generate_samples(logger, step=1, epoch=1)

        # Verify the method runs without error

    def test_conditional_sample_generation_uses_functions(self, trainer_instance):
        """Test that conditional _generate_samples uses sampler functions."""
        # The model has num_classes=2, so it's conditional
        assert trainer_instance.model.num_classes == 2

        # Generate samples (should use sample_with_intermediates function)
        logger = trainer_instance.logger
        trainer_instance._generate_samples(logger, step=1, epoch=1)

        # If this runs without error, the integration is working


@pytest.mark.integration
class TestStandaloneSamplerUsage:
    """Test using sampler functions independently of trainer."""

    def test_standalone_sampler_from_checkpoint(
        self, trainer_instance, temp_dir, device
    ):
        """Test loading checkpoint and using sampler functions independently."""
        checkpoint_path = temp_dir / "inference_checkpoint.pth"

        # Save checkpoint from trainer
        trainer_instance.save_checkpoint(
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

        # Use sampler functions directly (no trainer needed!)
        samples = sample(model, device, num_samples=4, use_ema=True, ema=ema)

        assert samples.shape == (4, 3, 32, 32)

    def test_standalone_sampler_without_trainer_dependencies(self, device):
        """Test that sampler functions don't require trainer dependencies."""
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

        # Use sampler functions without optimizer, dataloader, logger, etc.
        samples = sample(model, device, num_samples=8)

        assert samples.shape == (8, 3, 32, 32)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_training_and_inference_workflow(
        self, trainer_instance, temp_dir, device
    ):
        """Test complete workflow: train, save, load, infer."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()

        # Train for 1 epoch
        trainer_instance.train(
            num_epochs=1,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
            validate_frequency=0,
            save_best=False,
        )

        # Checkpoint should exist
        latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
        assert latest_checkpoint.exists()

        # Load checkpoint into new model for inference
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

        # Use sampler functions for inference
        samples, labels = sample_by_class(
            model,
            device,
            samples_per_class=2,
            num_classes=2,
            use_ema=True,
            ema=ema,
        )

        assert samples.shape == (4, 3, 32, 32)
        assert len(labels) == 4
