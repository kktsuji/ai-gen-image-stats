"""Tests for Diffusion Trainer

This module contains tests for the DiffusionTrainer class.
Tests are organized into:
- Unit tests: Interface implementation and basic functionality
- Component tests: Training/validation loops with small data
- Integration tests: End-to-end training workflows with sample generation
"""

import tempfile
from pathlib import Path
from typing import Dict, Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.base.dataloader import BaseDataLoader
from src.base.logger import BaseLogger
from src.base.model import BaseModel
from src.experiments.diffusion.trainer import DiffusionTrainer

# Test fixtures and helper classes


class SimpleDiffusionModel(BaseModel):
    """Simple diffusion model for testing."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 8,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes

        # Simple conv network for noise prediction
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, in_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simple forward pass that returns noise prediction."""
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

    def compute_loss(
        self,
        x: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute simple MSE loss between prediction and target noise."""
        if criterion is None:
            criterion = nn.MSELoss()

        # Simulate noise prediction
        noise = torch.randn_like(x)
        predicted_noise = self.forward(x)

        return criterion(predicted_noise, noise)

    def sample(
        self,
        batch_size: int = 1,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        return_intermediates: bool = False,
        use_dynamic_threshold: bool = True,
        dynamic_threshold_percentile: float = 0.995,
    ) -> torch.Tensor:
        """Generate samples (simplified for testing)."""
        device = next(self.parameters()).device
        shape = (batch_size, self.in_channels, self.image_size, self.image_size)

        samples = torch.randn(shape, device=device)
        if return_intermediates:
            # Return (T+1, N, C, H, W) — simulate 11 denoising steps
            num_steps = 11
            intermediates = torch.randn(
                num_steps,
                batch_size,
                self.in_channels,
                self.image_size,
                self.image_size,
                device=device,
            )
            intermediates[-1] = samples
            return intermediates
        return samples


class SimpleDiffusionDataLoader(BaseDataLoader):
    """Simple dataloader for testing diffusion trainer."""

    def __init__(
        self,
        num_train_samples: int = 20,
        num_val_samples: int = 10,
        batch_size: int = 4,
        in_channels: int = 3,
        image_size: int = 8,
        return_labels: bool = True,
        num_classes: int = 2,
    ):
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.return_labels = return_labels
        self.num_classes = num_classes

    def get_train_loader(self) -> DataLoader:
        images = torch.randn(
            self.num_train_samples, self.in_channels, self.image_size, self.image_size
        )

        if self.return_labels:
            labels = torch.randint(0, self.num_classes, (self.num_train_samples,))
            dataset = TensorDataset(images, labels)
        else:
            dataset = TensorDataset(images)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self) -> Optional[DataLoader]:
        if self.num_val_samples == 0:
            return None

        images = torch.randn(
            self.num_val_samples, self.in_channels, self.image_size, self.image_size
        )

        if self.return_labels:
            labels = torch.randint(0, self.num_classes, (self.num_val_samples,))
            dataset = TensorDataset(images, labels)
        else:
            dataset = TensorDataset(images)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class SimpleDiffusionLogger(BaseLogger):
    """Simple logger for testing diffusion trainer."""

    def __init__(self):
        self.logged_metrics = []
        self.logged_images = []
        self.logged_denoising_sequences = []

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        self.logged_metrics.append({"metrics": metrics, "step": step, "epoch": epoch})

    def log_images(
        self,
        images: torch.Tensor,
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.logged_images.append(
            {"images": images, "tag": tag, "step": step, "epoch": epoch, **kwargs}
        )

    def log_denoising_process(
        self,
        denoising_sequence: torch.Tensor,
        step: int,
        epoch: Optional[int] = None,
        num_steps_to_show: int = 8,
    ) -> None:
        self.logged_denoising_sequences.append(
            {"sequence": denoising_sequence, "step": step, "epoch": epoch}
        )


# Fixtures


@pytest.fixture
def simple_diffusion_model():
    """Create a simple diffusion model for testing."""
    return SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)


@pytest.fixture
def simple_diffusion_model_unconditional():
    """Create a simple unconditional diffusion model for testing."""
    return SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=None)


@pytest.fixture
def simple_diffusion_dataloader():
    """Create a simple diffusion dataloader for testing."""
    return SimpleDiffusionDataLoader(
        num_train_samples=20,
        num_val_samples=10,
        batch_size=4,
        return_labels=True,
    )


@pytest.fixture
def simple_diffusion_dataloader_unconditional():
    """Create a simple unconditional diffusion dataloader for testing."""
    return SimpleDiffusionDataLoader(
        num_train_samples=20,
        num_val_samples=10,
        batch_size=4,
        return_labels=False,
    )


@pytest.fixture
def simple_diffusion_dataloader_no_val():
    """Create a simple diffusion dataloader without validation data."""
    return SimpleDiffusionDataLoader(
        num_train_samples=20, num_val_samples=0, batch_size=4, return_labels=True
    )


@pytest.fixture
def simple_diffusion_optimizer(simple_diffusion_model):
    """Create a simple optimizer for testing."""
    return torch.optim.Adam(simple_diffusion_model.parameters(), lr=0.001)


@pytest.fixture
def simple_diffusion_logger():
    """Create a simple logger for testing."""
    return SimpleDiffusionLogger()


@pytest.fixture
def diffusion_trainer(
    simple_diffusion_model,
    simple_diffusion_dataloader,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
):
    """Create a diffusion trainer for testing."""
    return DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
        use_ema=False,  # Disable EMA for simplicity in tests
        use_amp=False,  # Disable AMP for CPU tests
        log_images_interval=None,  # Disable sampling during training for speed
        log_denoising_interval=None,
    )


def _make_diffusion_trainer(
    num_train_samples: int = 8,
    num_val_samples: int = 0,
    batch_size: int = 4,
    return_labels: bool = True,
    num_classes: Optional[int] = 2,
    in_channels: int = 3,
    image_size: int = 8,
    gradient_clip_norm: Optional[float] = None,
    config: Optional[Dict] = None,
) -> tuple:
    """Factory for creating a DiffusionTrainer with sensible defaults for tests."""
    model = SimpleDiffusionModel(
        in_channels=in_channels, image_size=image_size, num_classes=num_classes
    )
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        batch_size=batch_size,
        return_labels=return_labels,
        num_classes=num_classes if num_classes is not None else 2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        log_images_interval=None,
        log_denoising_interval=None,
        gradient_clip_norm=gradient_clip_norm,
        config=config,
    )
    return trainer, model, dataloader, optimizer, logger


# Unit Tests


@pytest.mark.unit
def test_diffusion_trainer_initialization(diffusion_trainer):
    """Test that DiffusionTrainer initializes correctly."""
    assert diffusion_trainer is not None
    assert diffusion_trainer.model is not None
    assert diffusion_trainer.dataloader is not None
    assert diffusion_trainer.optimizer is not None
    assert diffusion_trainer.logger is not None
    assert diffusion_trainer.device == "cpu"
    assert diffusion_trainer.show_progress is False
    assert diffusion_trainer.use_ema is False
    assert diffusion_trainer.use_amp is False


@pytest.mark.unit
def test_diffusion_trainer_with_ema(
    simple_diffusion_model,
    simple_diffusion_dataloader,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
):
    """Test that DiffusionTrainer initializes correctly with EMA."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        use_ema=True,
        ema_decay=0.999,
    )

    assert trainer.use_ema is True
    assert trainer.ema is not None
    assert trainer.ema.decay == 0.999


@pytest.mark.unit
def test_diffusion_trainer_implements_base_interface(diffusion_trainer):
    """Test that DiffusionTrainer implements all required BaseTrainer methods."""
    assert hasattr(diffusion_trainer, "train_epoch")
    assert hasattr(diffusion_trainer, "validate_epoch")
    assert hasattr(diffusion_trainer, "get_model")
    assert hasattr(diffusion_trainer, "get_dataloader")
    assert hasattr(diffusion_trainer, "get_optimizer")
    assert hasattr(diffusion_trainer, "get_logger")
    assert callable(diffusion_trainer.train_epoch)
    assert callable(diffusion_trainer.validate_epoch)
    assert callable(diffusion_trainer.get_model)
    assert callable(diffusion_trainer.get_dataloader)
    assert callable(diffusion_trainer.get_optimizer)
    assert callable(diffusion_trainer.get_logger)


@pytest.mark.unit
def test_diffusion_trainer_getters(diffusion_trainer):
    """Test that getter methods return correct objects."""
    model = diffusion_trainer.get_model()
    dataloader = diffusion_trainer.get_dataloader()
    optimizer = diffusion_trainer.get_optimizer()
    logger = diffusion_trainer.get_logger()

    assert isinstance(model, BaseModel)
    assert isinstance(dataloader, BaseDataLoader)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(logger, BaseLogger)


# Component Tests


@pytest.mark.component
def test_diffusion_trainer_train_epoch(diffusion_trainer):
    """Test that train_epoch runs without errors and returns metrics."""
    metrics = diffusion_trainer.train_epoch()

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
    assert metrics["loss"] >= 0.0


@pytest.mark.component
def test_diffusion_trainer_validate_epoch(diffusion_trainer):
    """Test that validate_epoch runs without errors and returns metrics."""
    metrics = diffusion_trainer.validate_epoch()

    assert metrics is not None
    assert isinstance(metrics, dict)
    assert "val_loss" in metrics
    assert isinstance(metrics["val_loss"], float)
    assert metrics["val_loss"] >= 0.0


@pytest.mark.component
def test_diffusion_trainer_validate_epoch_no_val_data(
    simple_diffusion_model,
    simple_diffusion_dataloader_no_val,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
):
    """Test that validate_epoch returns None when no validation data."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader_no_val,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
    )

    metrics = trainer.validate_epoch()
    assert metrics is None


@pytest.mark.component
def test_diffusion_trainer_unconditional(
    simple_diffusion_model_unconditional,
    simple_diffusion_dataloader_unconditional,
    simple_diffusion_logger,
):
    """Test trainer with unconditional model (no labels)."""
    # Create optimizer from the unconditional model's parameters
    optimizer = torch.optim.Adam(
        simple_diffusion_model_unconditional.parameters(), lr=0.001
    )
    trainer = DiffusionTrainer(
        model=simple_diffusion_model_unconditional,
        dataloader=simple_diffusion_dataloader_unconditional,
        optimizer=optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
    )

    # Test training epoch
    train_metrics = trainer.train_epoch()
    assert "loss" in train_metrics

    # Test validation epoch
    val_metrics = trainer.validate_epoch()
    assert "val_loss" in val_metrics  # type: ignore[operator]


@pytest.mark.component
def test_diffusion_trainer_with_gradient_clipping(
    simple_diffusion_model,
    simple_diffusion_dataloader,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
):
    """Test trainer with gradient clipping."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        gradient_clip_norm=1.0,
    )

    # Should run without errors
    metrics = trainer.train_epoch()
    assert "loss" in metrics


# Integration Tests


@pytest.mark.integration
def test_diffusion_trainer_full_workflow():
    """Test complete training workflow."""
    # Create components
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    # Run training for 2 epochs
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(
            num_epochs=2,
            checkpoint_dir=tmpdir,
            validate_frequency=1,
        )

        # Check that metrics were logged
        assert len(logger.logged_metrics) > 0

        # Check that checkpoints were saved
        checkpoint_dir = Path(tmpdir)
        assert (checkpoint_dir / "latest_checkpoint.pth").exists()
        assert (checkpoint_dir / "final_model.pth").exists()


@pytest.mark.integration
def test_diffusion_trainer_latest_checkpoint_written_when_enabled():
    """latest_checkpoint.pth is created when save_latest_checkpoint=True."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(
            num_epochs=1,
            checkpoint_dir=tmpdir,
            validate_frequency=0,
            save_latest_checkpoint=True,
        )

        assert (Path(tmpdir) / "latest_checkpoint.pth").exists()


@pytest.mark.integration
def test_diffusion_trainer_latest_checkpoint_not_written_when_disabled():
    """latest_checkpoint.pth is NOT created when save_latest_checkpoint=False."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(
            num_epochs=1,
            checkpoint_dir=tmpdir,
            validate_frequency=0,
            save_latest_checkpoint=False,
        )

        assert not (Path(tmpdir) / "latest_checkpoint.pth").exists()


@pytest.mark.integration
def test_diffusion_trainer_checkpoint_save_and_load():
    """Test checkpoint saving and loading."""
    # Create components
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
    )

    # Train for 1 epoch
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"

        # Save checkpoint
        trainer.save_checkpoint(
            checkpoint_path,
            epoch=1,
            metrics={"loss": 0.5},
        )

        assert checkpoint_path.exists()

        # Create new trainer and load checkpoint
        model2 = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        trainer2 = DiffusionTrainer(
            model=model2,
            dataloader=dataloader,
            optimizer=optimizer2,
            logger=logger,
            device="cpu",
            use_ema=False,
        )

        # Load checkpoint
        checkpoint_info = trainer2.load_checkpoint(checkpoint_path)

        assert checkpoint_info["epoch"] == 1
        assert "metrics" in checkpoint_info
        assert checkpoint_info["metrics"]["loss"] == 0.5


@pytest.mark.integration
def test_diffusion_trainer_with_ema_checkpoint():
    """Test checkpoint saving and loading with EMA."""
    # Create components
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    # Create trainer with EMA
    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=True,
        ema_decay=0.999,
    )

    # Train for 1 epoch
    trainer.train_epoch()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_ema_checkpoint.pth"

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path, epoch=1)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        assert "ema_state_dict" in checkpoint


@pytest.mark.integration
def test_load_checkpoint_without_ema_reinitializes_shadow():
    """Loading a checkpoint without ema_state_dict re-initializes shadow from model weights."""
    # Create and train a model, then save checkpoint
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=True,
        ema_decay=0.999,
    )

    # Train to update model weights away from init
    trainer.train_epoch()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path, epoch=1)

        # Remove ema_state_dict from checkpoint to simulate old format
        checkpoint = torch.load(checkpoint_path)
        assert "ema_state_dict" in checkpoint
        del checkpoint["ema_state_dict"]
        torch.save(checkpoint, checkpoint_path)

        # Create a new trainer with EMA and load the modified checkpoint
        model2 = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        trainer2 = DiffusionTrainer(
            model=model2,
            dataloader=dataloader,
            optimizer=optimizer2,
            logger=logger,
            device="cpu",
            show_progress=False,
            use_ema=True,
            ema_decay=0.999,
        )

        # Load checkpoint without ema_state_dict
        trainer2.load_checkpoint(checkpoint_path)

        # EMA shadow should match the loaded model weights, not the initial random weights
        loaded_model = trainer2.get_model()
        for name, param in loaded_model.named_parameters():
            if param.requires_grad:
                assert name in trainer2.ema.shadow  # type: ignore[union-attr]
                assert torch.allclose(trainer2.ema.shadow[name], param.data), (  # type: ignore[union-attr]
                    f"EMA shadow for {name} does not match model weights"
                )


@pytest.mark.integration
def test_load_checkpoint_with_ema_restores_saved_state():
    """Loading a checkpoint with ema_state_dict restores the saved EMA state."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=True,
        ema_decay=0.999,
    )

    # Train to update EMA shadow
    trainer.train_epoch()

    # Capture EMA shadow state before saving
    saved_shadow = {name: tensor.clone() for name, tensor in trainer.ema.shadow.items()}  # type: ignore[union-attr]

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_ema_full.pth"
        trainer.save_checkpoint(checkpoint_path, epoch=1)

        # Create a new trainer and load checkpoint
        model2 = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        trainer2 = DiffusionTrainer(
            model=model2,
            dataloader=dataloader,
            optimizer=optimizer2,
            logger=logger,
            device="cpu",
            show_progress=False,
            use_ema=True,
            ema_decay=0.999,
        )

        trainer2.load_checkpoint(checkpoint_path)

        # EMA shadow should match the saved EMA state
        for name, tensor in saved_shadow.items():
            assert name in trainer2.ema.shadow  # type: ignore[union-attr]
            assert torch.allclose(trainer2.ema.shadow[name], tensor), (  # type: ignore[union-attr]
                f"EMA shadow for {name} does not match saved state"
            )


@pytest.mark.integration
def test_diffusion_trainer_with_scheduler():
    """Test trainer with learning rate scheduler."""
    # Create components
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    logger = SimpleDiffusionLogger()

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        scheduler=scheduler,
    )

    # Get initial learning rate
    initial_lr = scheduler.get_last_lr()[0]

    # Train for 2 epochs
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(num_epochs=2, checkpoint_dir=tmpdir)

    # Check that learning rate was updated
    final_lr = scheduler.get_last_lr()[0]
    assert final_lr < initial_lr


@pytest.mark.integration
def test_diffusion_trainer_sample_generation_during_training():
    """Test that samples are generated during training when enabled."""
    # Create components
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    # Create trainer with sample generation enabled
    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=1,  # Generate every epoch
        log_denoising_interval=1,
        num_samples=4,
    )

    # Train for 2 epochs
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(num_epochs=2, checkpoint_dir=tmpdir)

    # Check that samples were logged
    assert len(logger.logged_images) > 0

    # Check sample properties
    for log_entry in logger.logged_images:
        assert "images" in log_entry
        assert "tag" in log_entry
        assert log_entry["images"].shape[1] == 3  # RGB

    # Check that sample comparisons and denoising were also logged
    assert len(logger.logged_denoising_sequences) > 0


@pytest.mark.integration
def test_diffusion_trainer_unconditional_sample_generation():
    """Test unconditional sample generation during training."""
    # Create unconditional components
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=None)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16,
        num_val_samples=8,
        batch_size=4,
        return_labels=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    # Create trainer with sample generation
    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=1,
        log_denoising_interval=None,
        num_samples=8,
    )

    # Train for 1 epoch
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(num_epochs=1, checkpoint_dir=tmpdir)

    # Check that samples were generated
    assert len(logger.logged_images) > 0


@pytest.mark.integration
def test_diffusion_trainer_logs_epoch_summary(
    simple_diffusion_model,
    simple_diffusion_dataloader,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
    capture_logs,
):
    """Test that diffusion trainer logs epoch summary."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
    )

    # Train for one epoch
    trainer.train_epoch()

    # Check that training activity was logged with epoch summary
    assert len(capture_logs.records) > 0
    assert any("Train" in r.message for r in capture_logs.records)


@pytest.mark.integration
def test_diffusion_trainer_logs_sample_generation(
    simple_diffusion_model,
    simple_diffusion_dataloader,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
    capture_logs,
):
    """Test that diffusion trainer logs sample generation triggers."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
        log_images_interval=1,
        log_denoising_interval=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for 1 epoch - should trigger sample generation
        trainer.train(num_epochs=1, checkpoint_dir=tmpdir)

        # Check that training and sample generation were logged
        assert len(capture_logs.records) > 0
        assert any("Train" in r.message for r in capture_logs.records)


@pytest.mark.integration
def test_diffusion_trainer_logs_ema_updates(
    simple_diffusion_model,
    simple_diffusion_dataloader,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
    capture_logs,
):
    """Test that diffusion trainer logs EMA updates (if verbose)."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
        use_ema=True,
        ema_decay=0.9999,
    )

    # Train for one epoch
    trainer.train_epoch()

    # Check that EMA training was logged
    assert len(capture_logs.records) > 0
    assert any("Train" in r.message for r in capture_logs.records)


@pytest.mark.unit
class TestDiffusionTrainerClassWeights:
    """Test DiffusionTrainer with class_weights tensor."""

    def test_train_epoch_with_class_weights(self):
        """DiffusionTrainer with class_weights tensor runs training correctly."""

        class DiffusionModelWithNoisePair(SimpleDiffusionModel):
            """Model that returns (predicted_noise, noise) tuple for weighted loss."""

            def forward(
                self,
                x: torch.Tensor,
                t: Optional[torch.Tensor] = None,
                class_labels: Optional[torch.Tensor] = None,
            ) -> tuple:
                noise = torch.randn_like(x)
                predicted = super().forward(x, t, class_labels)
                return predicted, noise

        model = DiffusionModelWithNoisePair(in_channels=3, image_size=8, num_classes=2)
        dataloader = SimpleDiffusionDataLoader(
            num_train_samples=8,
            num_val_samples=0,
            batch_size=4,
            return_labels=True,
            num_classes=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logger = SimpleDiffusionLogger()

        class_weights = torch.tensor([1.0, 2.0])
        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device="cpu",
            show_progress=False,
            use_ema=False,
            use_amp=False,
            log_images_interval=None,
            log_denoising_interval=None,
            class_weights=class_weights,
        )

        # weighted_criterion should be set
        assert trainer.weighted_criterion is not None

        # Training should complete without error
        metrics = trainer.train_epoch()
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)


@pytest.mark.unit
class TestDiffusionTrainerSchedulerCheckpoint:
    """Test DiffusionTrainer checkpoint save/load with scheduler."""

    def test_save_checkpoint_with_scheduler(self):
        """Trainer with scheduler, verify scheduler_state_dict in checkpoint."""
        trainer, model, _, optimizer, _ = _make_diffusion_trainer()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        trainer.scheduler = scheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "scheduler_checkpoint.pth"
            trainer.save_checkpoint(checkpoint_path, epoch=1)

            checkpoint = torch.load(checkpoint_path, weights_only=True)
            assert "scheduler_state_dict" in checkpoint

    def test_load_checkpoint_restores_scheduler(self):
        """Load checkpoint with scheduler state restores scheduler."""
        trainer, _, _, optimizer, _ = _make_diffusion_trainer()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        trainer.scheduler = scheduler

        # Step optimizer then scheduler to change its state
        optimizer.step()
        scheduler.step()
        lr_after_step = scheduler.get_last_lr()[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "scheduler_checkpoint.pth"
            trainer.save_checkpoint(checkpoint_path, epoch=1)

            # Create new trainer with fresh scheduler
            trainer2, _, _, optimizer2, _ = _make_diffusion_trainer()
            scheduler2 = torch.optim.lr_scheduler.StepLR(
                optimizer2, step_size=1, gamma=0.5
            )
            trainer2.scheduler = scheduler2

            trainer2.load_checkpoint(checkpoint_path)
            # Scheduler state should be restored
            assert scheduler2.get_last_lr()[0] == lr_after_step


@pytest.mark.unit
class TestDiffusionTrainerTensorBoardGraph:
    """Test TensorBoard model graph logging."""

    def test_tensorboard_graph_logging(self):
        """Mock tb_writer with log_graph=True.

        Note: This test manually constructs the trainer because tb_writer must
        be set on the logger *before* DiffusionTrainer.__init__ runs (which is
        where graph logging happens).
        """
        from unittest.mock import MagicMock

        model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
        dataloader = SimpleDiffusionDataLoader(
            num_train_samples=8, num_val_samples=0, batch_size=4
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logger = SimpleDiffusionLogger()

        mock_tb_writer = MagicMock()
        logger.tb_writer = mock_tb_writer  # type: ignore[attr-defined]

        config = {
            "logging": {
                "metrics": {
                    "tensorboard": {
                        "log_graph": True,
                    }
                }
            },
            "model": {
                "architecture": {
                    "image_size": 8,
                    "in_channels": 3,
                }
            },
        }

        DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device="cpu",
            show_progress=False,
            use_ema=False,
            log_images_interval=None,
            log_denoising_interval=None,
            config=config,
        )

        # add_graph should have been called during init
        mock_tb_writer.add_graph.assert_called_once()

    def test_tensorboard_graph_logging_failure(self):
        """add_graph raises, warning logged, trainer still initializes."""
        from unittest.mock import MagicMock

        model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
        dataloader = SimpleDiffusionDataLoader(
            num_train_samples=8,
            num_val_samples=0,
            batch_size=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logger = SimpleDiffusionLogger()

        # Mock tb_writer that raises on add_graph
        mock_tb_writer = MagicMock()
        mock_tb_writer.add_graph.side_effect = RuntimeError("graph logging failed")
        logger.tb_writer = mock_tb_writer  # type: ignore[attr-defined]

        config = {
            "logging": {
                "metrics": {
                    "tensorboard": {
                        "log_graph": True,
                    }
                }
            },
            "model": {
                "architecture": {
                    "image_size": 8,
                    "in_channels": 3,
                }
            },
        }

        # Should not raise - warning is logged instead
        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device="cpu",
            show_progress=False,
            use_ema=False,
            log_images_interval=None,
            log_denoising_interval=None,
            config=config,
        )
        assert trainer is not None


@pytest.mark.integration
def test_diffusion_trainer_logs_checkpoint_operations(
    simple_diffusion_model,
    simple_diffusion_dataloader,
    simple_diffusion_optimizer,
    simple_diffusion_logger,
    capture_logs,
):
    """Test that diffusion trainer logs checkpoint operations."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model,
        dataloader=simple_diffusion_dataloader,
        optimizer=simple_diffusion_optimizer,
        logger=simple_diffusion_logger,
        device="cpu",
        show_progress=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Train and save checkpoint
        trainer.train(
            num_epochs=1, checkpoint_dir=checkpoint_dir, checkpoint_frequency=1
        )

        # Check that checkpoint-related logging occurred
        assert len(capture_logs.records) > 0
        assert any(
            "checkpoint" in r.message.lower() or "Train" in r.message
            for r in capture_logs.records
        )


# ---- New tests for per-interval visualization ----


@pytest.mark.integration
def test_log_images_called_at_interval():
    """Verify log_images is called at the correct interval."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=2,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(num_epochs=4, checkpoint_dir=tmpdir, validate_frequency=0)

    # log_images called at epoch 2 and 4 (every 2 epochs)
    # Each trigger logs both samples and quality_comparison = 4 total
    assert len(logger.logged_images) == 4
    # denoising should NOT be called
    assert len(logger.logged_denoising_sequences) == 0


@pytest.mark.integration
def test_quality_comparison_logged_via_log_images():
    """Verify quality comparison images are logged via log_images."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=3,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(num_epochs=6, checkpoint_dir=tmpdir, validate_frequency=0)

    # log_images at epochs 3, 6: each triggers samples + quality_comparison = 4 total
    quality_entries = [
        e for e in logger.logged_images if e["tag"] == "quality_comparison"
    ]
    assert len(quality_entries) == 2
    assert len(logger.logged_denoising_sequences) == 0


@pytest.mark.integration
def test_log_denoising_called_at_interval():
    """Verify log_denoising_process is called at the correct interval."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=None,
        log_denoising_interval=2,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(num_epochs=4, checkpoint_dir=tmpdir, validate_frequency=0)

    # log_denoising_process at epochs 2, 4
    assert len(logger.logged_denoising_sequences) == 2
    assert len(logger.logged_images) == 0


@pytest.mark.integration
def test_samples_generated_once_when_all_intervals_trigger():
    """Verify sampler is called once when all three intervals trigger simultaneously."""
    from unittest.mock import patch

    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=1,
        log_denoising_interval=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(
            trainer.sampler,
            "sample_with_intermediates",
            wraps=trainer.sampler.sample_with_intermediates,
        ) as mock_sample:
            trainer.train(num_epochs=1, checkpoint_dir=tmpdir, validate_frequency=0)

            # Sampler called exactly once per epoch - one epoch means one call
            assert mock_sample.call_count == 1

    # Both logger methods should have been called
    # log_images is called twice per epoch (samples + quality_comparison)
    assert len(logger.logged_images) == 2
    assert len(logger.logged_denoising_sequences) == 1


@pytest.mark.integration
def test_visualization_disabled_when_all_intervals_null():
    """Verify no sampling occurs when all intervals are None."""
    from unittest.mock import patch

    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(trainer.sampler, "sample_with_intermediates") as mock_sample:
            trainer.train(num_epochs=3, checkpoint_dir=tmpdir, validate_frequency=0)

            # Sampler should never be called
            assert mock_sample.call_count == 0

    assert len(logger.logged_images) == 0
    assert len(logger.logged_denoising_sequences) == 0


# Bug fix tests


@pytest.mark.unit
def test_load_checkpoint_missing_file(diffusion_trainer):
    """Test load_checkpoint with missing file raises FileNotFoundError, not NameError.

    Bug 1: logger was undefined in load_checkpoint error handling.
    """
    nonexistent_path = Path("/tmp/nonexistent_checkpoint_12345.pth")
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        diffusion_trainer.load_checkpoint(nonexistent_path)


@pytest.mark.unit
def test_load_checkpoint_corrupt_file(diffusion_trainer):
    """Test load_checkpoint with corrupt file propagates original error, not NameError.

    Bug 1: logger was undefined in load_checkpoint error handling.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        f.write(b"this is not a valid checkpoint file")
        corrupt_path = Path(f.name)

    try:
        with pytest.raises((RuntimeError, IndexError)):
            diffusion_trainer.load_checkpoint(corrupt_path)
    finally:
        corrupt_path.unlink(missing_ok=True)


@pytest.mark.unit
def test_train_epoch_unexpected_batch_length():
    """Test train_epoch emits critical log when batch data has unexpected length.

    Bug 2: premature raise made the _logger.critical() call unreachable.
    """
    from unittest.mock import patch

    # Create a dataset that yields 3-element tuples
    images = torch.randn(4, 3, 8, 8)
    extra1 = torch.randint(0, 2, (4,))
    extra2 = torch.randint(0, 2, (4,))
    dataset = TensorDataset(images, extra1, extra2)
    loader = DataLoader(dataset, batch_size=4)

    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with patch.object(trainer.dataloader, "get_train_loader", return_value=loader):
        with pytest.raises(ValueError, match="Unexpected batch data length"):
            with patch("src.experiments.diffusion.trainer._logger") as mock_logger:
                trainer.train_epoch()

        # Re-run to verify the critical log is emitted
        with patch("src.experiments.diffusion.trainer._logger") as mock_logger:
            with pytest.raises(ValueError, match="Unexpected batch data length"):
                trainer.train_epoch()
            mock_logger.critical.assert_called()


@pytest.mark.component
def test_diffusion_trainer_with_plateau_scheduler():
    """Test that training with ReduceLROnPlateau runs without error (Bug 1)."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=0, factor=0.5
    )
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        scheduler=scheduler,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise TypeError or AttributeError
        trainer.train(num_epochs=2, checkpoint_dir=tmpdir, validate_frequency=1)


@pytest.mark.component
def test_diffusion_trainer_plateau_scheduler_receives_metric():
    """Test that ReduceLROnPlateau receives metric and adjusts LR (Bug 1)."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=0,
        factor=0.5,
        threshold=1e10,
        threshold_mode="abs",
    )
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        scheduler=scheduler,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    initial_lr = optimizer.param_groups[0]["lr"]

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(num_epochs=4, checkpoint_dir=tmpdir, validate_frequency=0)

    # With patience=0 and factor=0.5, LR should decrease after seeing metrics
    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr < initial_lr, (
        f"LR should have decreased from {initial_lr} but is {final_lr}"
    )


@pytest.mark.component
def test_best_model_saved_with_validation_data():
    """Test best_model.pth is saved when best_metric='loss' with validation data (Bug 2)."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(
            num_epochs=2,
            checkpoint_dir=tmpdir,
            validate_frequency=1,
            save_best=True,
            best_metric="loss",
        )
        assert (Path(tmpdir) / "best_model.pth").exists(), (
            "best_model.pth should be created with best_metric='loss' and validation data"
        )


@pytest.mark.component
def test_best_model_saved_with_matching_val_metric_key():
    """Test best_model.pth is saved when best_metric='val_loss' matches key directly (Bug 2)."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=8, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.train(
            num_epochs=2,
            checkpoint_dir=tmpdir,
            validate_frequency=1,
            save_best=True,
            best_metric="val_loss",
        )
        assert (Path(tmpdir) / "best_model.pth").exists(), (
            "best_model.pth should be created with best_metric='val_loss'"
        )


@pytest.mark.component
def test_resume_training_scheduler_advances():
    """Test that LR scheduler advances during resumed training (Bug 4)."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        scheduler=scheduler,
        log_images_interval=None,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for 2 epochs, save checkpoint
        trainer.train(
            num_epochs=2,
            checkpoint_dir=tmpdir,
            validate_frequency=0,
        )
        lr_after_initial = optimizer.param_groups[0]["lr"]
        checkpoint_path = Path(tmpdir) / "latest_checkpoint.pth"
        assert checkpoint_path.exists()

        # Resume for 2 more epochs
        trainer.resume_training(
            checkpoint_path=checkpoint_path,
            num_epochs=2,
            checkpoint_dir=tmpdir,
            validate_frequency=0,
        )
        lr_after_resume = optimizer.param_groups[0]["lr"]

    # LR should continue decreasing from the checkpoint value
    assert lr_after_resume < lr_after_initial, (
        f"LR should decrease during resumed training: {lr_after_initial} -> {lr_after_resume}"
    )


@pytest.mark.integration
def test_resume_training_generates_samples():
    """Test that visualization is generated during resumed training (Bug 4)."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
        use_amp=False,
        log_images_interval=1,
        log_denoising_interval=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for 1 epoch, save checkpoint
        trainer.train(
            num_epochs=1,
            checkpoint_dir=tmpdir,
            validate_frequency=0,
        )
        images_after_initial = len(logger.logged_images)
        checkpoint_path = Path(tmpdir) / "latest_checkpoint.pth"

        # Resume for 2 more epochs
        trainer.resume_training(
            checkpoint_path=checkpoint_path,
            num_epochs=2,
            checkpoint_dir=tmpdir,
            validate_frequency=0,
        )

    # Should have logged images during resumed training
    assert len(logger.logged_images) > images_after_initial, (
        "Visualization should be generated during resumed training"
    )


@pytest.mark.unit
def test_train_zero_epochs_no_crash():
    """train() with num_epochs=0 does not crash with UnboundLocalError."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Should not raise UnboundLocalError
        trainer.train(num_epochs=0, checkpoint_dir=tmpdir)


@pytest.mark.unit
def test_resume_training_zero_epochs_no_crash():
    """resume_training() with num_epochs=0 does not crash with UnboundLocalError."""
    model = SimpleDiffusionModel(in_channels=3, image_size=8)
    dataloader = SimpleDiffusionDataLoader(
        num_train_samples=16, num_val_samples=0, batch_size=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger = SimpleDiffusionLogger()

    trainer = DiffusionTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device="cpu",
        show_progress=False,
        use_ema=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for 1 epoch to create a checkpoint
        trainer.train(num_epochs=1, checkpoint_dir=tmpdir, validate_frequency=0)
        checkpoint_path = Path(tmpdir) / "final_model.pth"

        # Resume with 0 epochs — should not raise
        trainer.resume_training(
            checkpoint_path=checkpoint_path,
            num_epochs=0,
            checkpoint_dir=tmpdir,
            validate_frequency=0,
        )


# =============================================================================
# Unit Tests: ClassWeightedMSELoss
# =============================================================================


@pytest.mark.unit
class TestClassWeightedMSELoss:
    """Test ClassWeightedMSELoss class."""

    def test_equal_weights_matches_standard_mse(self):
        """Test that equal weights produce same result as nn.MSELoss."""
        from src.experiments.diffusion.trainer import ClassWeightedMSELoss

        batch_size = 4
        class_weights = torch.tensor([1.0, 1.0])
        criterion = ClassWeightedMSELoss(class_weights)
        standard_mse = nn.MSELoss()

        predicted = torch.randn(batch_size, 3, 8, 8)
        target = torch.randn(batch_size, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])

        weighted_loss = criterion(predicted, target, labels)
        standard_loss = standard_mse(predicted, target)

        assert weighted_loss.item() == pytest.approx(standard_loss.item(), rel=1e-5)

    def test_higher_weight_increases_loss(self):
        """Test that higher weight for a class increases its loss contribution."""
        from src.experiments.diffusion.trainer import ClassWeightedMSELoss

        batch_size = 4
        predicted = torch.randn(batch_size, 3, 8, 8)
        target = torch.randn(batch_size, 3, 8, 8)
        labels = torch.tensor([0, 0, 1, 1])

        # Equal weights
        equal_criterion = ClassWeightedMSELoss(torch.tensor([1.0, 1.0]))
        equal_loss = equal_criterion(predicted, target, labels)

        # Higher weight for class 1
        weighted_criterion = ClassWeightedMSELoss(torch.tensor([1.0, 5.0]))
        weighted_loss = weighted_criterion(predicted, target, labels)

        # Weighted loss should differ from equal loss
        assert weighted_loss.item() != pytest.approx(equal_loss.item(), rel=1e-3)

    def test_output_is_scalar(self):
        """Test that loss output is a scalar tensor."""
        from src.experiments.diffusion.trainer import ClassWeightedMSELoss

        class_weights = torch.tensor([1.0, 2.0])
        criterion = ClassWeightedMSELoss(class_weights)

        predicted = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])

        loss = criterion(predicted, target, labels)
        assert loss.ndim == 0  # Scalar

    def test_gradient_flows(self):
        """Test that gradients flow through the weighted loss."""
        from src.experiments.diffusion.trainer import ClassWeightedMSELoss

        class_weights = torch.tensor([1.0, 3.0])
        criterion = ClassWeightedMSELoss(class_weights)

        predicted = torch.randn(4, 3, 8, 8, requires_grad=True)
        target = torch.randn(4, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])

        loss = criterion(predicted, target, labels)
        loss.backward()

        assert predicted.grad is not None
        assert predicted.grad.shape == predicted.shape

    def test_single_class_batch(self):
        """Test with all samples from same class."""
        from src.experiments.diffusion.trainer import ClassWeightedMSELoss

        class_weights = torch.tensor([2.0, 1.0])
        criterion = ClassWeightedMSELoss(class_weights)

        predicted = torch.randn(4, 3, 8, 8)
        target = torch.randn(4, 3, 8, 8)
        labels = torch.tensor([0, 0, 0, 0])

        loss = criterion(predicted, target, labels)
        assert loss.item() > 0

    def test_trainer_with_class_weights(self):
        """Test DiffusionTrainer accepts class_weights parameter."""
        model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
        dataloader = SimpleDiffusionDataLoader(batch_size=4, return_labels=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logger = SimpleDiffusionLogger()

        class_weights = torch.tensor([1.0, 3.0])

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device="cpu",
            show_progress=False,
            use_ema=False,
            class_weights=class_weights,
        )

        assert trainer.weighted_criterion is not None
        assert trainer.class_weights is not None


@pytest.mark.unit
class TestDiffusionTrainerGradientClipping:
    """Test gradient clipping paths in DiffusionTrainer."""

    def test_gradient_clipping_non_amp(self):
        """Gradient clipping works in non-AMP training path."""
        trainer, *_ = _make_diffusion_trainer(
            gradient_clip_norm=0.001,  # Very small to trigger clipping warning
        )

        metrics = trainer.train_epoch()
        assert "loss" in metrics


@pytest.mark.unit
class TestDiffusionTrainerTrainMethod:
    """Test DiffusionTrainer.train() method coverage."""

    def test_train_with_config_logs_hyperparams(self):
        """train() logs hyperparams when config is set."""
        trainer, *_ = _make_diffusion_trainer(
            num_val_samples=4,
            config={"training": {"epochs": 1}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train(
                num_epochs=1,
                checkpoint_dir=tmpdir,
                checkpoint_frequency=1,
                validate_frequency=1,
                best_metric="val_loss",
            )

        assert trainer.current_epoch == 1

    def test_train_with_plateau_scheduler(self):
        """train() with ReduceLROnPlateau scheduler."""
        trainer, _, _, optimizer, _ = _make_diffusion_trainer(num_val_samples=4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=0
        )
        trainer.scheduler = scheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train(
                num_epochs=2,
                checkpoint_dir=tmpdir,
                validate_frequency=1,
            )

        assert trainer.current_epoch == 2


@pytest.mark.unit
class TestDiffusionTrainerResumeTraining:
    """Test DiffusionTrainer.resume_training() method."""

    def test_resume_training_basic(self):
        """resume_training loads checkpoint and continues."""
        trainer, *_ = _make_diffusion_trainer(
            num_val_samples=4,
            config={"training": {"epochs": 2}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train 1 epoch and save
            trainer.train(num_epochs=1, checkpoint_dir=tmpdir, validate_frequency=0)
            ckpt_path = Path(tmpdir) / "final_model.pth"
            assert ckpt_path.exists()

            # Create new trainer and resume
            trainer2, *_ = _make_diffusion_trainer(
                num_val_samples=4,
                config={"training": {"epochs": 2}},
            )

            trainer2.resume_training(
                checkpoint_path=ckpt_path,
                num_epochs=2,
                checkpoint_dir=tmpdir,
                validate_frequency=1,
                save_best=True,
                best_metric="val_loss",
            )

            assert trainer2.current_epoch == 3  # 1 + 2

    def test_resume_training_with_plateau_scheduler(self):
        """resume_training with ReduceLROnPlateau scheduler."""
        model = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
        dataloader = SimpleDiffusionDataLoader(
            num_train_samples=8,
            num_val_samples=4,
            batch_size=4,
            return_labels=True,
            num_classes=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=0
        )
        logger = SimpleDiffusionLogger()

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device="cpu",
            show_progress=False,
            use_ema=False,
            scheduler=scheduler,
            log_images_interval=None,
            log_denoising_interval=None,
            config={"training": {"epochs": 1}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train(num_epochs=1, checkpoint_dir=tmpdir, validate_frequency=0)
            ckpt_path = Path(tmpdir) / "final_model.pth"

            # Resume with new plateau scheduler
            model2 = SimpleDiffusionModel(in_channels=3, image_size=8, num_classes=2)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
            scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer2, mode="min", factor=0.1, patience=0
            )
            trainer2 = DiffusionTrainer(
                model=model2,
                dataloader=dataloader,
                optimizer=optimizer2,
                logger=logger,
                device="cpu",
                show_progress=False,
                use_ema=False,
                scheduler=scheduler2,
                log_images_interval=None,
                log_denoising_interval=None,
                config={"training": {"epochs": 2}},
            )

            trainer2.resume_training(
                checkpoint_path=ckpt_path,
                num_epochs=2,
                checkpoint_dir=tmpdir,
                validate_frequency=1,
            )

            assert trainer2.current_epoch == 3


@pytest.mark.unit
class TestDiffusionTrainerLoadCheckpointErrors:
    """Test DiffusionTrainer load_checkpoint error paths."""

    def test_load_checkpoint_model_mismatch_strict(self):
        """load_checkpoint with mismatched model in strict mode raises."""
        trainer, *_ = _make_diffusion_trainer()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pth"
            trainer.save_checkpoint(ckpt_path, epoch=1)

            # Create trainer with different model architecture
            trainer2, *_ = _make_diffusion_trainer(in_channels=1)

            with pytest.raises(RuntimeError):
                trainer2.load_checkpoint(ckpt_path, strict=True)

    def test_load_checkpoint_model_mismatch_non_strict(self):
        """load_checkpoint with mismatched model in non-strict mode warns."""
        trainer, *_ = _make_diffusion_trainer()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pth"
            trainer.save_checkpoint(ckpt_path, epoch=1)

            # Create trainer with different model architecture
            trainer2, *_ = _make_diffusion_trainer(in_channels=1)

            # Non-strict should not raise
            trainer2.load_checkpoint(ckpt_path, strict=False)

    def test_load_checkpoint_optimizer_failure(self):
        """load_checkpoint with optimizer load failure warns and continues."""
        from unittest.mock import patch as mock_patch

        trainer, _, _, optimizer, _ = _make_diffusion_trainer()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pth"
            trainer.save_checkpoint(ckpt_path, epoch=1)

            # Mock optimizer.load_state_dict to raise
            with mock_patch.object(
                optimizer,
                "load_state_dict",
                side_effect=RuntimeError("incompatible optimizer"),
            ):
                # Should not raise - warning is logged
                trainer.load_checkpoint(ckpt_path)
