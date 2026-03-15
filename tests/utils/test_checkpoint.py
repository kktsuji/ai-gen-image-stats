"""Tests for CheckpointManager utility."""

import pytest
import torch
import torch.nn as nn

from src.utils.checkpoint import CheckpointManager


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def checkpoint_manager(model, optimizer):
    return CheckpointManager(
        model_fn=lambda: model,
        optimizer_fn=lambda: optimizer,
    )


@pytest.mark.unit
class TestCheckpointSaveLoad:
    """Test save/load round-trip."""

    def test_save_creates_file(self, checkpoint_manager, tmp_path):
        """save() creates a checkpoint file."""
        path = tmp_path / "test.pth"
        checkpoint_manager.save(path, epoch=1, global_step=10)
        assert path.exists()

    def test_save_load_round_trip(self, checkpoint_manager, tmp_path):
        """Checkpoint can be saved and loaded back."""
        path = tmp_path / "test.pth"
        checkpoint_manager.save(
            path,
            epoch=5,
            global_step=100,
            metrics={"loss": 0.5},
            best_metric=0.3,
            best_metric_name="loss",
        )

        checkpoint = checkpoint_manager.load(path)
        assert checkpoint["epoch"] == 5
        assert checkpoint["global_step"] == 100
        assert checkpoint["metrics"]["loss"] == 0.5
        assert checkpoint["best_metric"] == 0.3
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_load_without_optimizer(self, checkpoint_manager, tmp_path):
        """Checkpoint can be loaded without optimizer state."""
        path = tmp_path / "test.pth"
        checkpoint_manager.save(path, epoch=1, global_step=10)
        checkpoint = checkpoint_manager.load(path, load_optimizer=False)
        assert checkpoint["epoch"] == 1

    def test_save_with_extra_kwargs(self, checkpoint_manager, tmp_path):
        """Extra kwargs are saved in checkpoint."""
        path = tmp_path / "test.pth"
        checkpoint_manager.save(
            path, epoch=1, global_step=10, custom_key="custom_value"
        )
        checkpoint = checkpoint_manager.load(path)
        assert checkpoint["custom_key"] == "custom_value"


@pytest.mark.unit
class TestCheckpointErrors:
    """Test error paths."""

    def test_load_missing_file(self, checkpoint_manager):
        """load() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load("/nonexistent/path.pth")

    def test_save_creates_parent_dirs(self, checkpoint_manager, tmp_path):
        """save() creates parent directories if needed."""
        path = tmp_path / "deep" / "nested" / "test.pth"
        checkpoint_manager.save(path, epoch=1, global_step=10)
        assert path.exists()


@pytest.mark.unit
class TestIsBestMetric:
    """Test is_best_metric static method."""

    def test_first_value_is_always_best(self):
        """First metric value (None best) is always best."""
        assert CheckpointManager.is_best_metric(0.5, None, "min") is True
        assert CheckpointManager.is_best_metric(0.5, None, "max") is True

    def test_min_mode(self):
        """Min mode: lower is better."""
        assert CheckpointManager.is_best_metric(0.3, 0.5, "min") is True
        assert CheckpointManager.is_best_metric(0.7, 0.5, "min") is False

    def test_max_mode(self):
        """Max mode: higher is better."""
        assert CheckpointManager.is_best_metric(0.7, 0.5, "max") is True
        assert CheckpointManager.is_best_metric(0.3, 0.5, "max") is False

    def test_invalid_mode(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric mode"):
            CheckpointManager.is_best_metric(0.5, 0.3, "invalid")
