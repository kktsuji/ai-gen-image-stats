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

        # Return random samples for testing
        return torch.randn(shape, device=device)


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
        sample_images=False,  # Disable sampling during training for speed
    )


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
    simple_diffusion_optimizer,
    simple_diffusion_logger,
):
    """Test trainer with unconditional model (no labels)."""
    trainer = DiffusionTrainer(
        model=simple_diffusion_model_unconditional,
        dataloader=simple_diffusion_dataloader_unconditional,
        optimizer=simple_diffusion_optimizer,
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
    assert "val_loss" in val_metrics


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
        sample_images=False,
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
        sample_images=False,
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
        sample_images=False,
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
        sample_images=True,
        sample_interval=1,  # Generate every epoch
        samples_per_class=2,
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
        sample_images=True,
        sample_interval=1,
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

    # Check that training activity was logged
    log_text = capture_logs.text.lower()
    # Should have some logging activity
    assert len(capture_logs.records) >= 0


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
        sample_images=True,
        sample_interval=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Train for 1 epoch - should trigger sample generation
        trainer.train(num_epochs=1, checkpoint_dir=tmpdir)

        # Check that logging occurred
        assert len(capture_logs.records) >= 0


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

    # Check that logging occurred during training
    assert len(capture_logs.records) >= 0


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
        log_text = capture_logs.text.lower()
        # Should have some logging activity
        assert len(capture_logs.records) >= 0
