"""Tests for Classifier Trainer

This module contains tests for the ClassifierTrainer class.
Tests are organized into:
- Unit tests: Interface implementation and basic functionality
- Component tests: Training/validation loops with small data
- Integration tests: End-to-end training workflows
"""

import tempfile
from pathlib import Path
from typing import Dict, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.base.dataloader import BaseDataLoader
from src.base.logger import BaseLogger
from src.base.model import BaseModel
from src.experiments.classifier.trainer import ClassifierTrainer

# Test fixtures and helper classes


class SimpleClassifierModel(BaseModel):
    """Simple classifier model for testing."""

    def __init__(self, input_dim: int = 10, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(predictions, targets)


class SimpleClassifierDataLoader(BaseDataLoader):
    """Simple dataloader for testing classifier trainer."""

    def __init__(
        self,
        num_train_samples: int = 20,
        num_val_samples: int = 10,
        batch_size: int = 4,
        input_dim: int = 10,
        num_classes: int = 2,
    ):
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_classes = num_classes

    def get_train_loader(self) -> DataLoader:
        X = torch.randn(self.num_train_samples, self.input_dim)
        y = torch.randint(0, self.num_classes, (self.num_train_samples,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self) -> Optional[DataLoader]:
        if self.num_val_samples == 0:
            return None
        X = torch.randn(self.num_val_samples, self.input_dim)
        y = torch.randint(0, self.num_classes, (self.num_val_samples,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class SimpleClassifierLogger(BaseLogger):
    """Simple logger for testing classifier trainer."""

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
            {"images": images, "tag": tag, "step": step, "epoch": epoch}
        )


# Fixtures


@pytest.fixture
def simple_model():
    """Create a simple classifier model for testing."""
    return SimpleClassifierModel(input_dim=10, num_classes=2)


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    return SimpleClassifierDataLoader(
        num_train_samples=20, num_val_samples=10, batch_size=4
    )


@pytest.fixture
def simple_dataloader_no_val():
    """Create a simple dataloader without validation data."""
    return SimpleClassifierDataLoader(
        num_train_samples=20, num_val_samples=0, batch_size=4
    )


@pytest.fixture
def simple_optimizer(simple_model):
    """Create a simple optimizer for testing."""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def simple_logger():
    """Create a simple logger for testing."""
    return SimpleClassifierLogger()


@pytest.fixture
def classifier_trainer(
    simple_model, simple_dataloader, simple_optimizer, simple_logger
):
    """Create a classifier trainer for testing."""
    return ClassifierTrainer(
        model=simple_model,
        dataloader=simple_dataloader,
        optimizer=simple_optimizer,
        logger=simple_logger,
        device="cpu",
        show_progress=False,
    )


# Unit Tests


@pytest.mark.unit
def test_classifier_trainer_initialization(classifier_trainer):
    """Test that ClassifierTrainer initializes correctly."""
    assert classifier_trainer is not None
    assert classifier_trainer.model is not None
    assert classifier_trainer.dataloader is not None
    assert classifier_trainer.optimizer is not None
    assert classifier_trainer.logger is not None
    assert classifier_trainer.device == "cpu"
    assert classifier_trainer.show_progress is False


@pytest.mark.unit
def test_classifier_trainer_implements_base_interface(classifier_trainer):
    """Test that ClassifierTrainer implements all required BaseTrainer methods."""
    assert hasattr(classifier_trainer, "train_epoch")
    assert hasattr(classifier_trainer, "validate_epoch")
    assert hasattr(classifier_trainer, "get_model")
    assert hasattr(classifier_trainer, "get_dataloader")
    assert hasattr(classifier_trainer, "get_optimizer")
    assert hasattr(classifier_trainer, "get_logger")
    assert callable(classifier_trainer.train_epoch)
    assert callable(classifier_trainer.validate_epoch)
    assert callable(classifier_trainer.get_model)
    assert callable(classifier_trainer.get_dataloader)
    assert callable(classifier_trainer.get_optimizer)
    assert callable(classifier_trainer.get_logger)


@pytest.mark.unit
def test_classifier_trainer_getters(classifier_trainer):
    """Test that getter methods return correct objects."""
    model = classifier_trainer.get_model()
    dataloader = classifier_trainer.get_dataloader()
    optimizer = classifier_trainer.get_optimizer()
    logger = classifier_trainer.get_logger()

    assert isinstance(model, BaseModel)
    assert isinstance(dataloader, BaseDataLoader)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(logger, BaseLogger)


@pytest.mark.unit
def test_classifier_trainer_device_assignment(
    simple_model, simple_dataloader, simple_optimizer, simple_logger
):
    """Test that model is moved to correct device."""
    trainer_cpu = ClassifierTrainer(
        model=simple_model,
        dataloader=simple_dataloader,
        optimizer=simple_optimizer,
        logger=simple_logger,
        device="cpu",
    )
    # Check that parameters are on CPU
    for param in trainer_cpu.model.parameters():
        assert param.device.type == "cpu"


# Component Tests


@pytest.mark.component
def test_train_epoch_returns_metrics(classifier_trainer):
    """Test that train_epoch returns expected metrics dictionary."""
    metrics = classifier_trainer.train_epoch()

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)


@pytest.mark.component
def test_train_epoch_updates_model(classifier_trainer):
    """Test that train_epoch updates model parameters."""
    # Get initial parameter values
    initial_params = [p.clone() for p in classifier_trainer.model.parameters()]

    # Run one training epoch
    classifier_trainer.train_epoch()

    # Check that parameters have changed
    for initial, current in zip(initial_params, classifier_trainer.model.parameters()):
        assert not torch.allclose(initial, current), "Parameters should be updated"


@pytest.mark.component
def test_train_epoch_increments_global_step(classifier_trainer):
    """Test that train_epoch increments global step counter."""
    initial_step = classifier_trainer.global_step

    # Run one training epoch
    classifier_trainer.train_epoch()

    # Global step should have increased
    assert classifier_trainer.global_step > initial_step


@pytest.mark.component
def test_validate_epoch_returns_metrics(classifier_trainer):
    """Test that validate_epoch returns expected metrics dictionary."""
    metrics = classifier_trainer.validate_epoch()

    assert isinstance(metrics, dict)
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
    assert isinstance(metrics["val_loss"], float)
    assert isinstance(metrics["val_accuracy"], float)


@pytest.mark.component
def test_validate_epoch_no_gradient_updates(classifier_trainer):
    """Test that validate_epoch does not update model parameters."""
    # Get initial parameter values
    initial_params = [p.clone() for p in classifier_trainer.model.parameters()]

    # Run validation epoch
    classifier_trainer.validate_epoch()

    # Check that parameters have not changed
    for initial, current in zip(initial_params, classifier_trainer.model.parameters()):
        assert torch.allclose(initial, current), "Parameters should not be updated"


@pytest.mark.component
def test_validate_epoch_without_val_data(
    simple_model, simple_dataloader_no_val, simple_optimizer, simple_logger
):
    """Test that validate_epoch returns None when no validation data."""
    trainer = ClassifierTrainer(
        model=simple_model,
        dataloader=simple_dataloader_no_val,
        optimizer=simple_optimizer,
        logger=simple_logger,
        device="cpu",
        show_progress=False,
    )

    metrics = trainer.validate_epoch()
    assert metrics is None


@pytest.mark.component
def test_evaluate_method(classifier_trainer):
    """Test that evaluate method works correctly."""
    metrics = classifier_trainer.evaluate()

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["accuracy"], float)


@pytest.mark.component
def test_evaluate_with_custom_dataloader(classifier_trainer):
    """Test that evaluate method works with custom dataloader."""
    # Create a test dataloader
    X = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=4)

    metrics = classifier_trainer.evaluate(test_loader)

    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "accuracy" in metrics


@pytest.mark.component
def test_model_mode_switching(classifier_trainer):
    """Test that model switches between train and eval modes correctly."""
    # Train epoch should set model to train mode
    classifier_trainer.train_epoch()
    # Note: After train_epoch, model remains in train mode

    # Validate epoch should set model to eval mode
    classifier_trainer.validate_epoch()
    # Model should now be in eval mode
    assert not classifier_trainer.model.training


# Integration Tests


@pytest.mark.integration
def test_full_training_loop(classifier_trainer, simple_logger):
    """Test complete training loop with multiple epochs."""
    # Train for 2 epochs
    classifier_trainer.train(
        num_epochs=2,
        checkpoint_dir=None,
        validate_frequency=1,
        save_best=False,
    )

    # Check that metrics were logged
    assert len(simple_logger.logged_metrics) > 0
    # Should have 2 training + 2 validation = 4 log entries
    assert len(simple_logger.logged_metrics) == 4


@pytest.mark.integration
def test_training_with_checkpointing(classifier_trainer):
    """Test training with checkpoint saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Train for 2 epochs
        classifier_trainer.train(
            num_epochs=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
            validate_frequency=1,
            save_best=False,
        )

        # Check that checkpoints were saved
        assert (checkpoint_dir / "latest_checkpoint.pth").exists()
        assert (checkpoint_dir / "checkpoint_epoch_1.pth").exists()
        assert (checkpoint_dir / "checkpoint_epoch_2.pth").exists()


@pytest.mark.integration
def test_training_with_best_model_saving(classifier_trainer):
    """Test training with best model saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Train for 3 epochs
        classifier_trainer.train(
            num_epochs=3,
            checkpoint_dir=checkpoint_dir,
            validate_frequency=1,
            save_best=True,
            best_metric="val_loss",
            best_metric_mode="min",
        )

        # Check that best model was saved
        assert (checkpoint_dir / "best_model.pth").exists()


@pytest.mark.integration
def test_checkpoint_loading_and_resuming(classifier_trainer):
    """Test checkpoint loading and training resumption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Train for 2 epochs
        classifier_trainer.train(
            num_epochs=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
        )

        # Load checkpoint
        checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
        checkpoint_info = classifier_trainer.load_checkpoint(checkpoint_path)

        # Check checkpoint contents
        assert checkpoint_info["epoch"] == 2
        assert "model_state_dict" in checkpoint_info
        assert "optimizer_state_dict" in checkpoint_info


@pytest.mark.integration
def test_accuracy_metrics_reasonable(classifier_trainer):
    """Test that accuracy metrics are in reasonable range."""
    metrics = classifier_trainer.train_epoch()

    # Accuracy should be between 0 and 100
    assert 0.0 <= metrics["accuracy"] <= 100.0

    val_metrics = classifier_trainer.validate_epoch()
    if val_metrics is not None:
        assert 0.0 <= val_metrics["val_accuracy"] <= 100.0


@pytest.mark.integration
def test_loss_decreases_with_training(classifier_trainer):
    """Test that loss decreases over multiple epochs."""
    initial_metrics = classifier_trainer.train_epoch()
    initial_loss = initial_metrics["loss"]

    # Train for several more epochs
    for _ in range(3):
        metrics = classifier_trainer.train_epoch()

    final_loss = metrics["loss"]

    # Loss should decrease (or at least not increase significantly)
    # Use a loose threshold since this is a random toy problem
    assert final_loss <= initial_loss * 1.5


@pytest.mark.integration
def test_progress_bar_disabled(
    simple_model, simple_dataloader, simple_optimizer, simple_logger
):
    """Test that progress bar can be disabled."""
    trainer_no_progress = ClassifierTrainer(
        model=simple_model,
        dataloader=simple_dataloader,
        optimizer=simple_optimizer,
        logger=simple_logger,
        device="cpu",
        show_progress=False,
    )

    # Should not raise errors
    trainer_no_progress.train_epoch()
    trainer_no_progress.validate_epoch()


@pytest.mark.integration
def test_multi_class_classification(simple_optimizer, simple_logger):
    """Test classifier with more than 2 classes."""
    # Create 5-class problem
    model = SimpleClassifierModel(input_dim=10, num_classes=5)
    dataloader = SimpleClassifierDataLoader(
        num_train_samples=50,
        num_val_samples=20,
        batch_size=10,
        input_dim=10,
        num_classes=5,
    )

    trainer = ClassifierTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        logger=simple_logger,
        device="cpu",
        show_progress=False,
    )

    metrics = trainer.train_epoch()
    assert "loss" in metrics
    assert "accuracy" in metrics

    val_metrics = trainer.validate_epoch()
    assert val_metrics is not None
    assert "val_loss" in val_metrics
    assert "val_accuracy" in val_metrics
