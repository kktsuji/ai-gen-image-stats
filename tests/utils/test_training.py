"""Unit tests for training utility factories.

Tests cover:
- create_optimizer: adam, adamw, sgd, invalid type, valid_types restriction,
  gradient_clip excluded, extra kwargs forwarded
- create_scheduler: cosine, step, plateau, None returns None, "none" returns None,
  T_max="auto" resolves to num_epochs, invalid type raises
"""

import pytest
import torch
import torch.nn as nn

from src.utils.training import create_optimizer, create_scheduler

# ---------------------------------------------------------------------------
# Minimal model parameters for testing
# ---------------------------------------------------------------------------


def _make_params():
    """Return a small set of model parameters for optimizer tests."""
    model = nn.Linear(4, 2)
    return model.parameters()


# ==============================================================================
# TestCreateOptimizer
# ==============================================================================


@pytest.mark.unit
class TestCreateOptimizer:
    """Test create_optimizer factory function."""

    def test_adam(self):
        """Test creating an Adam optimizer."""
        optimizer_config = {"type": "adam", "learning_rate": 1e-3}
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_adamw(self):
        """Test creating an AdamW optimizer."""
        optimizer_config = {"type": "adamw", "learning_rate": 1e-4}
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4

    def test_sgd(self):
        """Test creating an SGD optimizer."""
        optimizer_config = {"type": "sgd", "learning_rate": 0.01}
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_sgd_default_momentum(self):
        """Test that SGD gets default momentum of 0.9 when not specified."""
        optimizer_config = {"type": "sgd", "learning_rate": 0.01}
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_sgd_custom_momentum(self):
        """Test that SGD respects custom momentum."""
        optimizer_config = {"type": "sgd", "learning_rate": 0.01, "momentum": 0.5}
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert optimizer.param_groups[0]["momentum"] == 0.5

    def test_invalid_type_raises(self):
        """Test that an unknown optimizer type raises ValueError."""
        optimizer_config = {"type": "rmsprop", "learning_rate": 1e-3}

        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(_make_params(), optimizer_config)

    def test_valid_types_restriction(self):
        """Test that valid_types restricts allowed optimizer types."""
        optimizer_config = {"type": "sgd", "learning_rate": 0.01}

        # Should fail when sgd is not in valid_types
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(
                _make_params(), optimizer_config, valid_types=["adam", "adamw"]
            )

    def test_gradient_clip_excluded(self):
        """Test that gradient_clip_norm is excluded from optimizer kwargs."""
        optimizer_config = {
            "type": "adam",
            "learning_rate": 1e-3,
            "gradient_clip_norm": 1.0,
        }
        # Should not raise — gradient_clip_norm is excluded from kwargs
        optimizer = create_optimizer(_make_params(), optimizer_config)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_extra_kwargs_forwarded(self):
        """Test that extra kwargs (e.g., weight_decay) are forwarded to optimizer."""
        optimizer_config = {
            "type": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        }
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert optimizer.param_groups[0]["weight_decay"] == 1e-4

    def test_betas_forwarded(self):
        """Test that betas are forwarded to Adam optimizer."""
        optimizer_config = {
            "type": "adam",
            "learning_rate": 1e-3,
            "betas": (0.9, 0.98),
        }
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert optimizer.param_groups[0]["betas"] == (0.9, 0.98)

    def test_case_insensitive_type(self):
        """Test that optimizer type matching is case-insensitive."""
        optimizer_config = {"type": "Adam", "learning_rate": 1e-3}
        optimizer = create_optimizer(_make_params(), optimizer_config)

        assert isinstance(optimizer, torch.optim.Adam)


# ==============================================================================
# TestCreateScheduler
# ==============================================================================


@pytest.mark.unit
class TestCreateScheduler:
    """Test create_scheduler factory function."""

    def _make_optimizer(self):
        """Create a minimal optimizer for scheduler tests."""
        model = nn.Linear(4, 2)
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def test_cosine_scheduler(self):
        """Test creating a CosineAnnealingLR scheduler."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "cosine", "T_max": 10}
        scheduler = create_scheduler(optimizer, scheduler_config, num_epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_cosine_t_max_auto_resolves_to_num_epochs(self):
        """Test that T_max='auto' is resolved to num_epochs."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "cosine", "T_max": "auto"}
        scheduler = create_scheduler(optimizer, scheduler_config, num_epochs=50)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 50

    def test_step_scheduler(self):
        """Test creating a StepLR scheduler."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "step", "step_size": 10, "gamma": 0.5}
        scheduler = create_scheduler(optimizer, scheduler_config, num_epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_plateau_scheduler(self):
        """Test creating a ReduceLROnPlateau scheduler."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "plateau", "patience": 5}
        scheduler = create_scheduler(optimizer, scheduler_config, num_epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_none_type_returns_none(self):
        """Test that type=None returns None."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": None}
        scheduler = create_scheduler(optimizer, scheduler_config, num_epochs=100)

        assert scheduler is None

    def test_none_string_returns_none(self):
        """Test that type='none' (string) returns None."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "none"}
        scheduler = create_scheduler(optimizer, scheduler_config, num_epochs=100)

        assert scheduler is None

    def test_none_string_case_insensitive(self):
        """Test that type='NONE' returns None (case-insensitive)."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "NONE"}
        scheduler = create_scheduler(optimizer, scheduler_config, num_epochs=100)

        assert scheduler is None

    def test_invalid_type_raises(self):
        """Test that an unknown scheduler type raises ValueError."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "exponential"}

        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(optimizer, scheduler_config, num_epochs=100)

    def test_valid_types_restriction(self):
        """Test that valid_types restricts allowed scheduler types."""
        optimizer = self._make_optimizer()
        scheduler_config = {"type": "plateau"}

        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(
                optimizer,
                scheduler_config,
                num_epochs=100,
                valid_types=[None, "none", "cosine", "step"],
            )
