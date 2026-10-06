"""Tests for checkpoint utility functions."""

import pytest
import torch
import torch.nn as nn

from src.utils.checkpoint import (
    file_sha256,
    is_best_metric,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
)


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


@pytest.mark.unit
class TestCheckpointSaveLoad:
    """Test save/load round-trip."""

    def test_save_creates_file(self, model, optimizer, tmp_path):
        """save_checkpoint() creates a checkpoint file."""
        path = tmp_path / "test.pth"
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, global_step=10)
        assert path.exists()

    def test_save_load_round_trip(self, model, optimizer, tmp_path):
        """Checkpoint can be saved and loaded back."""
        path = tmp_path / "test.pth"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=5,
            global_step=100,
            metrics={"loss": 0.5},
            best_metric=0.3,
            best_metric_name="loss",
        )

        checkpoint = load_checkpoint(path, model=model, optimizer=optimizer)
        assert checkpoint["epoch"] == 5
        assert checkpoint["global_step"] == 100
        assert checkpoint["metrics"]["loss"] == 0.5
        assert checkpoint["best_metric"] == 0.3
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_without_optimizer_state(self, model, optimizer, tmp_path):
        """save_optimizer=False omits optimizer state but keeps model weights."""
        path = tmp_path / "test.pth"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=3,
            global_step=30,
            save_optimizer=False,
        )

        checkpoint = load_checkpoint(path, model=model)
        assert checkpoint["epoch"] == 3
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" not in checkpoint

    def test_load_warns_when_optimizer_provided_but_absent(
        self, model, optimizer, tmp_path, caplog
    ):
        """Loading a save_optimizer=False checkpoint with an optimizer warns."""
        path = tmp_path / "test.pth"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=3,
            global_step=30,
            save_optimizer=False,
        )

        import logging

        with caplog.at_level(logging.WARNING, logger="src.utils.checkpoint"):
            load_checkpoint(path, model=model, optimizer=optimizer)

        assert "fresh optimizer state" in caplog.text
        assert optimizer.state_dict()["state"] == {}

    def test_load_without_optimizer(self, model, optimizer, tmp_path):
        """Checkpoint can be loaded without optimizer state."""
        path = tmp_path / "test.pth"
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, global_step=10)
        checkpoint = load_checkpoint(path, model=model)
        assert checkpoint["epoch"] == 1

    def test_save_with_extra_kwargs(self, model, optimizer, tmp_path):
        """Extra kwargs are saved in checkpoint."""
        path = tmp_path / "test.pth"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            global_step=10,
            custom_key="custom_value",
        )
        checkpoint = load_checkpoint(path, model=model, optimizer=optimizer)
        assert checkpoint["custom_key"] == "custom_value"


@pytest.mark.unit
class TestCheckpointErrors:
    """Test error paths."""

    def test_load_missing_file(self, model):
        """load_checkpoint() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.pth", model=model)

    def test_save_creates_parent_dirs(self, model, optimizer, tmp_path):
        """save_checkpoint() creates parent directories if needed."""
        path = tmp_path / "deep" / "nested" / "test.pth"
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, global_step=10)
        assert path.exists()


@pytest.mark.unit
class TestIsBestMetric:
    """Test is_best_metric function."""

    def test_first_value_is_always_best(self):
        """First metric value (None best) is always best."""
        assert is_best_metric(0.5, None, "min") is True
        assert is_best_metric(0.5, None, "max") is True

    def test_min_mode(self):
        """Min mode: lower is better."""
        assert is_best_metric(0.3, 0.5, "min") is True
        assert is_best_metric(0.7, 0.5, "min") is False

    def test_max_mode(self):
        """Max mode: higher is better."""
        assert is_best_metric(0.7, 0.5, "max") is True
        assert is_best_metric(0.3, 0.5, "max") is False

    def test_invalid_mode(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid metric mode"):
            is_best_metric(0.5, 0.3, "invalid")


class TwoPartModel(nn.Module):
    """A backbone layer plus a head named ``fc`` (as in the classifiers)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.body = nn.Linear(4, 3)
        self.bn = nn.BatchNorm1d(3)
        self.fc = nn.Linear(3, num_classes)


def _save_model(model, path, prefix=""):
    state = {f"{prefix}{k}": v for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": state}, path)


@pytest.mark.unit
class TestLoadModelWeights:
    """load_model_weights: initialization from another run's checkpoint."""

    def test_loads_every_key(self, tmp_path):
        torch.manual_seed(0)
        source = TwoPartModel()
        _save_model(source, tmp_path / "a.pth")
        target = TwoPartModel()
        load_model_weights(target, tmp_path / "a.pth")
        for k, v in source.state_dict().items():
            assert torch.equal(target.state_dict()[k], v)

    def test_skip_prefix_keeps_head(self, tmp_path):
        """A skipped head may differ in shape (e.g. 6 classes -> 2)."""
        _save_model(TwoPartModel(num_classes=6), tmp_path / "a.pth")
        source_body = torch.load(tmp_path / "a.pth")["model_state_dict"]["body.weight"]
        target = TwoPartModel(num_classes=2)
        head_before = target.fc.weight.clone()
        load_model_weights(target, tmp_path / "a.pth", skip_prefixes=("fc.",))
        assert torch.equal(target.body.weight, source_body)
        assert torch.equal(target.fc.weight, head_before)

    def test_head_shape_mismatch_raises(self, tmp_path):
        _save_model(TwoPartModel(num_classes=6), tmp_path / "a.pth")
        with pytest.raises(ValueError, match="shape mismatch"):
            load_model_weights(TwoPartModel(num_classes=2), tmp_path / "a.pth")

    def test_missing_key_raises(self, tmp_path):
        state = TwoPartModel().state_dict()
        del state["bn.running_mean"]
        torch.save({"model_state_dict": state}, tmp_path / "a.pth")
        with pytest.raises(ValueError, match="missing"):
            load_model_weights(TwoPartModel(), tmp_path / "a.pth")

    def test_unexpected_key_raises(self, tmp_path):
        state = TwoPartModel().state_dict()
        state["extra.weight"] = torch.zeros(1)
        torch.save({"model_state_dict": state}, tmp_path / "a.pth")
        with pytest.raises(ValueError, match="unexpected"):
            load_model_weights(TwoPartModel(), tmp_path / "a.pth")

    def test_strips_compile_prefix(self, tmp_path):
        source = TwoPartModel()
        _save_model(source, tmp_path / "a.pth", prefix="_orig_mod.")
        target = TwoPartModel()
        load_model_weights(target, tmp_path / "a.pth")
        assert torch.equal(target.fc.weight, source.fc.weight)

    def test_file_sha256(self, tmp_path):
        path = tmp_path / "x.bin"
        path.write_bytes(b"abc")
        assert file_sha256(path) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
