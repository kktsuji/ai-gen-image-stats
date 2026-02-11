"""Tests for Base Trainer Interface

This module contains tests for the BaseTrainer abstract class and its interface.
Tests are organized into unit tests (fast, CPU only) and component tests (training loops).
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.base.dataloader import BaseDataLoader
from src.base.logger import BaseLogger
from src.base.model import BaseModel
from src.base.trainer import BaseTrainer

# Test fixtures and helper classes


class SimpleTestModel(BaseModel):
    """Simple model for testing trainer."""

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(predictions, targets)


class SimpleTestDataLoader(BaseDataLoader):
    """Simple dataloader for testing trainer."""

    def __init__(
        self,
        num_train_samples: int = 20,
        num_val_samples: int = 10,
        batch_size: int = 4,
    ):
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.batch_size = batch_size

    def get_train_loader(self) -> DataLoader:
        X = torch.randn(self.num_train_samples, 10)
        y = torch.randint(0, 2, (self.num_train_samples,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_loader(self) -> Optional[DataLoader]:
        if self.num_val_samples == 0:
            return None
        X = torch.randn(self.num_val_samples, 10)
        y = torch.randint(0, 2, (self.num_val_samples,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class SimpleTestLogger(BaseLogger):
    """Simple logger for testing trainer."""

    def __init__(self):
        self.logged_metrics = []
        self.logged_images = []

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        self.logged_metrics.append({"metrics": metrics, "step": step, "epoch": epoch})

    def log_images(
        self,
        images: Any,
        tag: str,
        step: int,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.logged_images.append(
            {"images": images, "tag": tag, "step": step, "epoch": epoch}
        )


class MinimalValidTrainer(BaseTrainer):
    """Minimal valid trainer implementation for testing."""

    def __init__(
        self,
        model: BaseModel,
        dataloader: BaseDataLoader,
        optimizer: torch.optim.Optimizer,
        logger: BaseLogger,
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.logger = logger

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        train_loader = self.dataloader.get_train_loader()
        total_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(data)
            loss = self.model.compute_loss(predictions, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            self._global_step += 1

        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def validate_epoch(self) -> Optional[Dict[str, float]]:
        val_loader = self.dataloader.get_val_loader()
        if val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                predictions = self.model(data)
                loss = self.model.compute_loss(predictions, target)
                total_loss += loss.item()
                num_batches += 1

        return {"val_loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def get_model(self) -> BaseModel:
        return self.model

    def get_dataloader(self) -> BaseDataLoader:
        return self.dataloader

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

    def get_logger(self) -> BaseLogger:
        return self.logger


class TrainerWithMetrics(MinimalValidTrainer):
    """Trainer that tracks additional metrics for testing."""

    def train_epoch(self) -> Dict[str, float]:
        base_metrics = super().train_epoch()
        # Add accuracy metric
        base_metrics["accuracy"] = 0.85  # Mock accuracy
        return base_metrics

    def validate_epoch(self) -> Optional[Dict[str, float]]:
        base_metrics = super().validate_epoch()
        if base_metrics is not None:
            base_metrics["val_accuracy"] = 0.90  # Mock validation accuracy
        return base_metrics


class IncompleteTrainer(BaseTrainer):
    """Trainer that doesn't implement all required methods."""

    def __init__(self):
        super().__init__()

    # Missing train_epoch, validate_epoch, get_model, etc.


# Unit Tests


@pytest.mark.unit
class TestBaseTrainerInterface:
    """Test that BaseTrainer enforces its interface requirements."""

    def test_cannot_instantiate_base_trainer_directly(self):
        """BaseTrainer is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseTrainer()

    def test_cannot_instantiate_incomplete_implementation(self):
        """Trainers that don't implement all abstract methods cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteTrainer()

    def test_can_instantiate_complete_implementation(self):
        """Trainers that implement all abstract methods can be instantiated."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        assert isinstance(trainer, BaseTrainer)

    def test_required_abstract_methods_exist(self):
        """Verify all required abstract methods are defined."""
        required_methods = [
            "train_epoch",
            "validate_epoch",
            "get_model",
            "get_dataloader",
            "get_optimizer",
            "get_logger",
        ]

        for method_name in required_methods:
            assert hasattr(BaseTrainer, method_name)
            method = getattr(BaseTrainer, method_name)
            assert getattr(method, "__isabstractmethod__", False)


@pytest.mark.unit
class TestTrainerProperties:
    """Test trainer properties and state tracking."""

    def test_initial_state(self):
        """Test trainer initializes with correct default state."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)

        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_metric is None

    def test_current_epoch_property(self):
        """Test current_epoch property works correctly."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        trainer._current_epoch = 5

        assert trainer.current_epoch == 5

    def test_global_step_property(self):
        """Test global_step property works correctly."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        trainer._global_step = 100

        assert trainer.global_step == 100

    def test_best_metric_property(self):
        """Test best_metric property works correctly."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        trainer._best_metric = 0.123

        assert trainer.best_metric == 0.123


# Component Tests


@pytest.mark.component
class TestTrainEpoch:
    """Test single training epoch execution."""

    def test_train_epoch_returns_metrics(self):
        """Test that train_epoch returns a metrics dictionary."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        metrics = trainer.train_epoch()

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_train_epoch_updates_model_weights(self):
        """Test that training updates model parameters."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        # Save initial weights
        initial_weights = [p.clone() for p in model.parameters()]

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        trainer.train_epoch()

        # Check that weights changed
        final_weights = list(model.parameters())
        weights_changed = any(
            not torch.allclose(initial, final)
            for initial, final in zip(initial_weights, final_weights)
        )
        assert weights_changed, "Model weights should change after training"

    def test_train_epoch_increments_global_step(self):
        """Test that train_epoch increments global step counter."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, batch_size=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        initial_step = trainer.global_step

        trainer.train_epoch()

        # Should increment by number of batches (8 samples / 4 batch_size = 2 batches)
        assert trainer.global_step == initial_step + 2


@pytest.mark.component
class TestValidateEpoch:
    """Test single validation epoch execution."""

    def test_validate_epoch_returns_metrics(self):
        """Test that validate_epoch returns a metrics dictionary."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, num_val_samples=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        val_metrics = trainer.validate_epoch()

        assert isinstance(val_metrics, dict)
        assert "val_loss" in val_metrics
        assert isinstance(val_metrics["val_loss"], float)

    def test_validate_epoch_no_validation_data(self):
        """Test validate_epoch with no validation data."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, num_val_samples=0)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        val_metrics = trainer.validate_epoch()

        assert val_metrics is None

    def test_validate_epoch_does_not_update_weights(self):
        """Test that validation does not modify model parameters."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, num_val_samples=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        # Save initial weights
        initial_weights = [p.clone() for p in model.parameters()]

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        trainer.validate_epoch()

        # Check that weights did NOT change
        final_weights = list(model.parameters())
        for initial, final in zip(initial_weights, final_weights):
            torch.testing.assert_close(initial, final)


@pytest.mark.component
class TestCheckpointSaveLoad:
    """Test checkpoint save and load functionality."""

    def test_save_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint saving."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_path = tmp_path / "checkpoint.pth"

        trainer.save_checkpoint(checkpoint_path, epoch=5)

        assert checkpoint_path.exists()

    def test_save_checkpoint_with_metrics(self, tmp_path):
        """Test saving checkpoint with metrics."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_path = tmp_path / "checkpoint.pth"

        metrics = {"loss": 0.5, "accuracy": 0.95}
        trainer.save_checkpoint(checkpoint_path, epoch=10, metrics=metrics)

        # Load and verify
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint["epoch"] == 10
        assert checkpoint["metrics"] == metrics

    def test_save_checkpoint_creates_directory(self, tmp_path):
        """Test that save_checkpoint creates parent directories."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_path = tmp_path / "new_dir" / "subdir" / "checkpoint.pth"

        trainer.save_checkpoint(checkpoint_path, epoch=1)

        assert checkpoint_path.exists()

    def test_load_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint loading."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_path = tmp_path / "checkpoint.pth"

        # Save checkpoint
        trainer._current_epoch = 5
        trainer._global_step = 100
        trainer.save_checkpoint(checkpoint_path, epoch=5)

        # Create new trainer and load
        model2 = SimpleTestModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        trainer2 = MinimalValidTrainer(model2, dataloader, optimizer2, logger)

        checkpoint_info = trainer2.load_checkpoint(checkpoint_path)

        assert checkpoint_info["epoch"] == 5
        assert trainer2.current_epoch == 5
        assert trainer2.global_step == 100

    def test_load_checkpoint_restores_model_weights(self, tmp_path):
        """Test that loading checkpoint restores model weights."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_path = tmp_path / "checkpoint.pth"

        # Train for one epoch to change weights
        trainer.train_epoch()
        trained_weights = [p.clone() for p in model.parameters()]

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path, epoch=1)

        # Create new model with different weights
        model2 = SimpleTestModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        trainer2 = MinimalValidTrainer(model2, dataloader, optimizer2, logger)

        # Load checkpoint
        trainer2.load_checkpoint(checkpoint_path)

        # Verify weights match the saved model
        for saved, loaded in zip(trained_weights, model2.parameters()):
            torch.testing.assert_close(saved, loaded)

    def test_load_checkpoint_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent checkpoint file."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        nonexistent_path = tmp_path / "does_not_exist.pth"

        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(nonexistent_path)

    def test_load_checkpoint_without_optimizer(self, tmp_path):
        """Test loading checkpoint without loading optimizer state."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_path = tmp_path / "checkpoint.pth"

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path, epoch=1)

        # Create new trainer with different optimizer
        model2 = SimpleTestModel()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        trainer2 = MinimalValidTrainer(model2, dataloader, optimizer2, logger)

        # Load without optimizer state
        trainer2.load_checkpoint(checkpoint_path, load_optimizer=False)

        # Optimizer learning rate should remain unchanged
        assert optimizer2.param_groups[0]["lr"] == 0.01


@pytest.mark.component
class TestTrainLoop:
    """Test the main training loop."""

    def test_train_runs_specified_epochs(self, tmp_path):
        """Test that train() runs for the specified number of epochs."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, num_val_samples=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)

        num_epochs = 3
        trainer.train(num_epochs=num_epochs, checkpoint_dir=None, validate_frequency=0)

        assert trainer.current_epoch == num_epochs
        assert len(logger.logged_metrics) == num_epochs  # One per epoch

    def test_train_logs_metrics(self):
        """Test that train() logs metrics to the logger."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)

        trainer.train(num_epochs=2, checkpoint_dir=None, validate_frequency=0)

        # Should have logged metrics for each epoch
        assert len(logger.logged_metrics) == 2
        for logged in logger.logged_metrics:
            assert "loss" in logged["metrics"]

    def test_train_validation_frequency(self):
        """Test that validation runs at specified frequency."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, num_val_samples=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)

        # Train for 4 epochs, validate every 2 epochs
        trainer.train(num_epochs=4, checkpoint_dir=None, validate_frequency=2)

        # Should have 4 training logs + 2 validation logs
        assert len(logger.logged_metrics) == 6

        # Check validation metrics appear at correct epochs
        val_logs = [
            log for log in logger.logged_metrics if "val_loss" in log["metrics"]
        ]
        assert len(val_logs) == 2
        assert val_logs[0]["epoch"] == 2
        assert val_logs[1]["epoch"] == 4

    def test_train_saves_checkpoints(self, tmp_path):
        """Test that train() saves checkpoints at specified frequency."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.train(
            num_epochs=4,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=2,
            validate_frequency=0,
        )

        # Should have epoch checkpoints at 2 and 4, plus latest
        assert (checkpoint_dir / "checkpoint_epoch_2.pth").exists()
        assert (checkpoint_dir / "checkpoint_epoch_4.pth").exists()
        assert (checkpoint_dir / "latest_checkpoint.pth").exists()

    def test_train_saves_best_model(self, tmp_path):
        """Test that train() saves best model based on metrics."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8, num_val_samples=4)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = TrainerWithMetrics(model, dataloader, optimizer, logger)
        checkpoint_dir = tmp_path / "checkpoints"

        trainer.train(
            num_epochs=3,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
            validate_frequency=1,
            save_best=True,
            best_metric="val_loss",
            best_metric_mode="min",
        )

        # Should have saved best model
        assert (checkpoint_dir / "best_model.pth").exists()


@pytest.mark.component
class TestResumeTraining:
    """Test resuming training from checkpoints."""

    def test_resume_training_continues_from_checkpoint(self, tmp_path):
        """Test that resume_training loads checkpoint and continues."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader(num_train_samples=8)
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_path = tmp_path / "checkpoint.pth"

        # Train for 2 epochs and save
        trainer.train(num_epochs=2, checkpoint_dir=None, validate_frequency=0)
        trainer.save_checkpoint(checkpoint_path, epoch=2)

        # Create new trainer and resume training for 3 more epochs
        model2 = SimpleTestModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        trainer2 = MinimalValidTrainer(model2, dataloader, optimizer2, logger)

        trainer2.resume_training(
            checkpoint_path=checkpoint_path,
            num_epochs=3,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=1,
            validate_frequency=0,
        )

        # Should have trained for 3 more epochs
        assert trainer2.current_epoch == 5  # 2 (from checkpoint) + 3 (resumed)


@pytest.mark.component
class TestBestMetricTracking:
    """Test best metric tracking functionality."""

    def test_is_best_metric_min_mode(self):
        """Test best metric comparison in min mode."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)

        # First value is always best
        assert trainer._is_best_metric(0.5, "min")
        trainer._best_metric = 0.5

        # Lower value is better
        assert trainer._is_best_metric(0.3, "min")

        # Higher value is not better
        assert not trainer._is_best_metric(0.7, "min")

    def test_is_best_metric_max_mode(self):
        """Test best metric comparison in max mode."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)

        # First value is always best
        assert trainer._is_best_metric(0.5, "max")
        trainer._best_metric = 0.5

        # Higher value is better
        assert trainer._is_best_metric(0.7, "max")

        # Lower value is not better
        assert not trainer._is_best_metric(0.3, "max")

    def test_is_best_metric_invalid_mode(self):
        """Test that invalid metric mode raises error."""
        model = SimpleTestModel()
        dataloader = SimpleTestDataLoader()
        optimizer = torch.optim.Adam(model.parameters())
        logger = SimpleTestLogger()

        trainer = MinimalValidTrainer(model, dataloader, optimizer, logger)
        trainer._best_metric = 0.3  # Set initial best metric

        with pytest.raises(ValueError, match="Invalid metric mode"):
            trainer._is_best_metric(0.5, "invalid")
