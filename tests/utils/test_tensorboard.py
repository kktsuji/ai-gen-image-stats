"""
Unit tests for src/utils/tensorboard.py

Tests cover:
- is_tensorboard_available()
- create_tensorboard_writer() with enabled/disabled/missing package
- All safe_log_* functions with valid writer, None writer, and error conditions
- close_tensorboard_writer() with valid writer and None
- _flatten_dict() with nested dictionaries
"""

import logging
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.utils.tensorboard import (
    TENSORBOARD_AVAILABLE,
    _flatten_dict,
    close_tensorboard_writer,
    create_tensorboard_writer,
    is_tensorboard_available,
    safe_log_figure,
    safe_log_histogram,
    safe_log_hparams,
    safe_log_images,
    safe_log_scalar,
    safe_log_scalars,
    safe_log_text,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_tb_dir(tmp_path):
    """Temporary directory for TensorBoard logs."""
    return tmp_path / "tensorboard"


@pytest.fixture
def mock_writer():
    """A mock SummaryWriter for testing safe_log_* functions."""
    return MagicMock()


# ============================================================================
# Tests: is_tensorboard_available
# ============================================================================


@pytest.mark.unit
def test_is_tensorboard_available_returns_bool():
    """is_tensorboard_available should always return a boolean."""
    result = is_tensorboard_available()
    assert isinstance(result, bool)


@pytest.mark.unit
def test_is_tensorboard_available_matches_constant():
    """Return value should match the module-level constant."""
    assert is_tensorboard_available() == TENSORBOARD_AVAILABLE


@pytest.mark.unit
def test_is_tensorboard_available_false_when_import_fails():
    """Should return False when SummaryWriter cannot be imported."""
    with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
        assert is_tensorboard_available() is False


@pytest.mark.unit
def test_is_tensorboard_available_true_when_installed():
    """Should return True when tensorboard is installed."""
    with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", True):
        assert is_tensorboard_available() is True


# ============================================================================
# Tests: create_tensorboard_writer
# ============================================================================


@pytest.mark.unit
def test_create_tensorboard_writer_disabled_returns_none(tmp_tb_dir):
    """Should return None when enabled=False."""
    writer = create_tensorboard_writer(log_dir=tmp_tb_dir, enabled=False)
    assert writer is None


@pytest.mark.unit
def test_create_tensorboard_writer_not_available_returns_none(tmp_tb_dir):
    """Should return None when tensorboard package is not installed."""
    with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
        writer = create_tensorboard_writer(log_dir=tmp_tb_dir, enabled=True)
    assert writer is None


@pytest.mark.unit
@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
def test_create_tensorboard_writer_enabled_creates_writer(tmp_tb_dir):
    """Should return a SummaryWriter when enabled and package is available."""
    writer = create_tensorboard_writer(log_dir=tmp_tb_dir, enabled=True)
    assert writer is not None
    writer.close()


@pytest.mark.unit
@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
def test_create_tensorboard_writer_creates_log_directory(tmp_tb_dir):
    """Should create the log directory if it does not exist."""
    nested_dir = tmp_tb_dir / "nested" / "subdir"
    writer = create_tensorboard_writer(log_dir=nested_dir, enabled=True)
    assert nested_dir.exists()
    writer.close()


@pytest.mark.unit
@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
def test_create_tensorboard_writer_accepts_string_path(tmp_tb_dir):
    """Should accept a str log_dir in addition to Path."""
    writer = create_tensorboard_writer(log_dir=str(tmp_tb_dir), enabled=True)
    assert writer is not None
    writer.close()


@pytest.mark.unit
def test_create_tensorboard_writer_exception_returns_none(tmp_tb_dir):
    """Should return None and log error when SummaryWriter raises."""
    with (
        patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", True),
        patch("src.utils.tensorboard.SummaryWriter", side_effect=RuntimeError("boom")),
    ):
        writer = create_tensorboard_writer(log_dir=tmp_tb_dir, enabled=True)
    assert writer is None


@pytest.mark.unit
def test_create_tensorboard_writer_disabled_logs_info(tmp_tb_dir, caplog):
    """Should log an info message when disabled."""
    with caplog.at_level(logging.INFO, logger="src.utils.tensorboard"):
        create_tensorboard_writer(log_dir=tmp_tb_dir, enabled=False)
    assert any("disabled" in r.message.lower() for r in caplog.records)


@pytest.mark.unit
def test_create_tensorboard_writer_not_available_logs_warning(tmp_tb_dir, caplog):
    """Should log a warning when package is missing but enabled=True."""
    with patch("src.utils.tensorboard.TENSORBOARD_AVAILABLE", False):
        with caplog.at_level(logging.WARNING, logger="src.utils.tensorboard"):
            create_tensorboard_writer(log_dir=tmp_tb_dir, enabled=True)
    assert any("not installed" in r.message.lower() for r in caplog.records)


# ============================================================================
# Tests: safe_log_scalar
# ============================================================================


@pytest.mark.unit
def test_safe_log_scalar_none_writer_is_noop():
    """Should do nothing when writer is None."""
    safe_log_scalar(None, "loss", 0.5, step=1)  # Must not raise


@pytest.mark.unit
def test_safe_log_scalar_calls_add_scalar(mock_writer):
    """Should call writer.add_scalar with correct arguments."""
    safe_log_scalar(mock_writer, "train/loss", 0.42, step=10)
    mock_writer.add_scalar.assert_called_once_with("train/loss", 0.42, 10)


@pytest.mark.unit
def test_safe_log_scalar_converts_tensor(mock_writer):
    """Should convert a scalar Tensor to a Python float before logging."""
    value = torch.tensor(0.75)
    safe_log_scalar(mock_writer, "val/acc", value, step=5)
    mock_writer.add_scalar.assert_called_once_with("val/acc", 0.75, 5)


@pytest.mark.unit
def test_safe_log_scalar_handles_exception(mock_writer, caplog):
    """Should not propagate exceptions from add_scalar."""
    mock_writer.add_scalar.side_effect = RuntimeError("write error")
    with caplog.at_level(logging.DEBUG, logger="src.utils.tensorboard"):
        safe_log_scalar(mock_writer, "loss", 1.0, step=0)  # Must not raise


# ============================================================================
# Tests: safe_log_scalars
# ============================================================================


@pytest.mark.unit
def test_safe_log_scalars_none_writer_is_noop():
    """Should do nothing when writer is None."""
    safe_log_scalars(None, "metrics", {"loss": 0.5, "acc": 0.9}, step=1)


@pytest.mark.unit
def test_safe_log_scalars_calls_add_scalars(mock_writer):
    """Should call writer.add_scalars with processed dict."""
    safe_log_scalars(mock_writer, "metrics", {"loss": 0.5, "acc": 0.9}, step=3)
    mock_writer.add_scalars.assert_called_once_with(
        "metrics", {"loss": 0.5, "acc": 0.9}, 3
    )


@pytest.mark.unit
def test_safe_log_scalars_converts_tensors(mock_writer):
    """Should convert Tensor values to Python floats."""
    tag_dict = {"loss": torch.tensor(0.3), "acc": torch.tensor(0.95)}
    safe_log_scalars(mock_writer, "metrics", tag_dict, step=7)
    call_args = mock_writer.add_scalars.call_args[0]
    assert call_args[1] == {"loss": pytest.approx(0.3), "acc": pytest.approx(0.95)}


@pytest.mark.unit
def test_safe_log_scalars_handles_exception(mock_writer):
    """Should not propagate exceptions from add_scalars."""
    mock_writer.add_scalars.side_effect = RuntimeError("boom")
    safe_log_scalars(mock_writer, "metrics", {"loss": 1.0}, step=0)  # Must not raise


# ============================================================================
# Tests: safe_log_images
# ============================================================================


@pytest.mark.unit
def test_safe_log_images_none_writer_is_noop():
    """Should do nothing when writer is None."""
    img = torch.rand(4, 3, 32, 32)
    safe_log_images(None, "samples", img, step=1)


@pytest.mark.unit
def test_safe_log_images_calls_add_images_4d(mock_writer):
    """Should call add_images with 4D (N,C,H,W) tensor unchanged."""
    img = torch.rand(4, 3, 32, 32)
    safe_log_images(mock_writer, "samples", img, step=2)
    mock_writer.add_images.assert_called_once()
    call_args = mock_writer.add_images.call_args[0]
    assert call_args[0] == "samples"
    assert call_args[1].shape == (4, 3, 32, 32)


@pytest.mark.unit
def test_safe_log_images_unsqueezes_3d(mock_writer):
    """Should unsqueeze a 3D (C,H,W) tensor to (1,C,H,W) before logging."""
    img = torch.rand(3, 32, 32)
    safe_log_images(mock_writer, "sample", img, step=0)
    call_args = mock_writer.add_images.call_args[0]
    assert call_args[1].shape == (1, 3, 32, 32)


@pytest.mark.unit
def test_safe_log_images_handles_exception(mock_writer):
    """Should not propagate exceptions from add_images."""
    mock_writer.add_images.side_effect = RuntimeError("boom")
    img = torch.rand(2, 3, 16, 16)
    safe_log_images(mock_writer, "samples", img, step=0)  # Must not raise


# ============================================================================
# Tests: safe_log_histogram
# ============================================================================


@pytest.mark.unit
def test_safe_log_histogram_none_writer_is_noop():
    """Should do nothing when writer is None."""
    safe_log_histogram(None, "weights", torch.randn(100), step=1)


@pytest.mark.unit
def test_safe_log_histogram_calls_add_histogram(mock_writer):
    """Should call writer.add_histogram with correct tag and step."""
    values = torch.randn(100)
    safe_log_histogram(mock_writer, "layer/weights", values, step=5)
    mock_writer.add_histogram.assert_called_once()
    call_args = mock_writer.add_histogram.call_args[0]
    assert call_args[0] == "layer/weights"


@pytest.mark.unit
def test_safe_log_histogram_handles_exception(mock_writer):
    """Should not propagate exceptions from add_histogram."""
    mock_writer.add_histogram.side_effect = RuntimeError("boom")
    safe_log_histogram(mock_writer, "w", torch.randn(10), step=0)  # Must not raise


# ============================================================================
# Tests: safe_log_figure
# ============================================================================


@pytest.mark.unit
def test_safe_log_figure_none_writer_is_noop():
    """Should do nothing when writer is None."""
    fig = MagicMock()
    safe_log_figure(None, "confusion_matrix", fig, step=1)
    fig.assert_not_called()


@pytest.mark.unit
def test_safe_log_figure_calls_add_figure(mock_writer):
    """Should call writer.add_figure with correct arguments."""
    fig = MagicMock()
    safe_log_figure(mock_writer, "confusion_matrix/matrix", fig, step=3, close=True)
    mock_writer.add_figure.assert_called_once_with(
        "confusion_matrix/matrix", fig, 3, close=True
    )


@pytest.mark.unit
def test_safe_log_figure_handles_exception(mock_writer):
    """Should not propagate exceptions from add_figure."""
    mock_writer.add_figure.side_effect = RuntimeError("boom")
    safe_log_figure(mock_writer, "fig", MagicMock(), step=0)  # Must not raise


# ============================================================================
# Tests: safe_log_text
# ============================================================================


@pytest.mark.unit
def test_safe_log_text_none_writer_is_noop():
    """Should do nothing when writer is None."""
    safe_log_text(None, "notes", "hello world", step=1)


@pytest.mark.unit
def test_safe_log_text_calls_add_text(mock_writer):
    """Should call writer.add_text with correct arguments."""
    safe_log_text(mock_writer, "notes", "run started", step=0)
    mock_writer.add_text.assert_called_once_with("notes", "run started", 0)


@pytest.mark.unit
def test_safe_log_text_handles_exception(mock_writer):
    """Should not propagate exceptions from add_text."""
    mock_writer.add_text.side_effect = RuntimeError("boom")
    safe_log_text(mock_writer, "tag", "text", step=0)  # Must not raise


# ============================================================================
# Tests: safe_log_hparams
# ============================================================================


@pytest.mark.unit
def test_safe_log_hparams_none_writer_is_noop():
    """Should do nothing when writer is None."""
    safe_log_hparams(None, {"lr": 0.001})


@pytest.mark.unit
def test_safe_log_hparams_calls_add_hparams(mock_writer):
    """Should call writer.add_hparams with flattened dict."""
    hparams = {"training": {"lr": 0.001, "batch_size": 32}}
    safe_log_hparams(mock_writer, hparams)
    mock_writer.add_hparams.assert_called_once()
    call_kwargs = mock_writer.add_hparams.call_args[0]
    flat = call_kwargs[0]
    assert "training/lr" in flat
    assert "training/batch_size" in flat


@pytest.mark.unit
def test_safe_log_hparams_with_metric_dict(mock_writer):
    """Should pass metric_dict to add_hparams when provided."""
    safe_log_hparams(mock_writer, {"lr": 0.01}, metric_dict={"best_acc": 0.95})
    call_args = mock_writer.add_hparams.call_args[0]
    assert call_args[1] == {"best_acc": 0.95}


@pytest.mark.unit
def test_safe_log_hparams_empty_metric_dict_by_default(mock_writer):
    """Should pass empty dict as metric_dict when not provided."""
    safe_log_hparams(mock_writer, {"lr": 0.01})
    call_args = mock_writer.add_hparams.call_args[0]
    assert call_args[1] == {}


@pytest.mark.unit
def test_safe_log_hparams_handles_exception(mock_writer):
    """Should not propagate exceptions from add_hparams."""
    mock_writer.add_hparams.side_effect = RuntimeError("boom")
    safe_log_hparams(mock_writer, {"lr": 0.001})  # Must not raise


# ============================================================================
# Tests: close_tensorboard_writer
# ============================================================================


@pytest.mark.unit
def test_close_tensorboard_writer_none_is_noop():
    """Should do nothing when writer is None."""
    close_tensorboard_writer(None)  # Must not raise


@pytest.mark.unit
def test_close_tensorboard_writer_calls_close(mock_writer):
    """Should call writer.close()."""
    close_tensorboard_writer(mock_writer)
    mock_writer.close.assert_called_once()


@pytest.mark.unit
def test_close_tensorboard_writer_handles_exception(mock_writer, caplog):
    """Should log error but not raise when close() fails."""
    mock_writer.close.side_effect = RuntimeError("close error")
    with caplog.at_level(logging.ERROR, logger="src.utils.tensorboard"):
        close_tensorboard_writer(mock_writer)  # Must not raise
    assert any(
        "close" in r.message.lower() or "failed" in r.message.lower()
        for r in caplog.records
    )


# ============================================================================
# Tests: _flatten_dict
# ============================================================================


@pytest.mark.unit
def test_flatten_dict_flat_dict():
    """Should return unchanged flat dict."""
    d = {"lr": 0.001, "batch_size": 32}
    result = _flatten_dict(d)
    assert result == {"lr": 0.001, "batch_size": 32}


@pytest.mark.unit
def test_flatten_dict_nested_dict():
    """Should flatten one level of nesting with '/' separator."""
    d = {"training": {"lr": 0.001, "epochs": 100}}
    result = _flatten_dict(d)
    assert result == {"training/lr": 0.001, "training/epochs": 100}


@pytest.mark.unit
def test_flatten_dict_deeply_nested():
    """Should flatten arbitrarily deep nesting."""
    d = {"a": {"b": {"c": 42}}}
    result = _flatten_dict(d)
    assert result == {"a/b/c": 42}


@pytest.mark.unit
def test_flatten_dict_custom_separator():
    """Should use the provided separator."""
    d = {"a": {"b": 1}}
    result = _flatten_dict(d, sep=".")
    assert result == {"a.b": 1}


@pytest.mark.unit
def test_flatten_dict_non_primitive_converted_to_str():
    """Non-primitive values should be converted to strings."""
    d = {"model": {"architecture": [512, 256, 128]}}
    result = _flatten_dict(d)
    assert result["model/architecture"] == str([512, 256, 128])


@pytest.mark.unit
def test_flatten_dict_mixed_types():
    """Should handle int, float, str, bool values without conversion."""
    d = {"a": 1, "b": 0.5, "c": "hello", "d": True}
    result = _flatten_dict(d)
    assert result == {"a": 1, "b": 0.5, "c": "hello", "d": True}


@pytest.mark.unit
def test_flatten_dict_empty():
    """Should return empty dict for empty input."""
    assert _flatten_dict({}) == {}


@pytest.mark.unit
def test_flatten_dict_with_parent_key():
    """Should prepend parent_key to all keys."""
    d = {"lr": 0.001}
    result = _flatten_dict(d, parent_key="training")
    assert result == {"training/lr": 0.001}


# ============================================================================
# Integration-level: writer round-trip
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
def test_full_writer_lifecycle(tmp_tb_dir):
    """Create a writer, log a scalar, and close it without errors."""
    writer = create_tensorboard_writer(log_dir=tmp_tb_dir, enabled=True)
    assert writer is not None

    safe_log_scalar(writer, "loss", 0.5, step=0)
    safe_log_scalars(writer, "metrics", {"loss": 0.5, "acc": 0.9}, step=0)

    img = torch.rand(2, 3, 16, 16)
    safe_log_images(writer, "samples", img, step=0)

    safe_log_text(writer, "config", "batch_size=32", step=0)

    close_tensorboard_writer(writer)

    # Event files should be present in log_dir
    event_files = list(tmp_tb_dir.glob("events.out.tfevents.*"))
    assert len(event_files) > 0
