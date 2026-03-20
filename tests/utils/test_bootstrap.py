"""Tests for Bootstrap Confidence Interval Utilities

Tests for the bootstrap CI computation in src/utils/bootstrap.py.
"""

import numpy as np
import pytest

from src.utils.bootstrap import bootstrap_classification_metrics

# ============================================================================
# Unit Tests - bootstrap_classification_metrics
# ============================================================================


@pytest.mark.unit
class TestBootstrapClassificationMetrics:
    """Tests for the high-level bootstrap_classification_metrics function."""

    @pytest.fixture()
    def binary_data(self):
        """Create binary classification data for testing."""
        rng = np.random.RandomState(42)
        n = 100
        targets = rng.randint(0, 2, size=n)
        predictions = targets.copy()
        # Flip ~20% of predictions
        flip_mask = rng.random(n) < 0.2
        predictions[flip_mask] = 1 - predictions[flip_mask]
        probs = np.zeros((n, 2))
        for i in range(n):
            if predictions[i] == targets[i]:
                probs[i, targets[i]] = 0.8
                probs[i, 1 - targets[i]] = 0.2
            else:
                probs[i, targets[i]] = 0.3
                probs[i, 1 - targets[i]] = 0.7
        return targets, predictions, probs

    def test_returns_dict_with_requested_metrics(self, binary_data):
        """Returns CIs for all requested metrics."""
        targets, predictions, probs = binary_data
        metrics = ["recall_1", "balanced_accuracy", "accuracy"]

        result = bootstrap_classification_metrics(
            targets,
            predictions,
            probs,
            num_classes=2,
            metric_names=metrics,
            n_bootstrap=100,
            seed=42,
        )

        assert set(result.keys()) == set(metrics)
        for name in metrics:
            lo, hi = result[name]
            assert isinstance(lo, float)
            assert isinstance(hi, float)
            assert lo <= hi

    def test_all_supported_binary_metrics(self, binary_data):
        """All supported metric types work for binary classification."""
        targets, predictions, probs = binary_data
        metrics = [
            "recall_0",
            "recall_1",
            "precision_0",
            "precision_1",
            "f1_0",
            "f1_1",
            "balanced_accuracy",
            "accuracy",
            "loss",
            "roc_auc",
            "pr_auc",
        ]

        result = bootstrap_classification_metrics(
            targets,
            predictions,
            probs,
            num_classes=2,
            metric_names=metrics,
            n_bootstrap=200,
            seed=42,
        )

        for name in metrics:
            lo, hi = result[name]
            assert not np.isnan(lo), f"{name} lower is NaN"
            assert not np.isnan(hi), f"{name} upper is NaN"
            assert lo <= hi, f"{name}: lower > upper"

    def test_correlated_bootstrap_with_seed(self, binary_data):
        """Same seed produces identical CIs."""
        targets, predictions, probs = binary_data
        metrics = ["recall_1", "balanced_accuracy"]

        r1 = bootstrap_classification_metrics(
            targets,
            predictions,
            probs,
            num_classes=2,
            metric_names=metrics,
            n_bootstrap=500,
            seed=99,
        )
        r2 = bootstrap_classification_metrics(
            targets,
            predictions,
            probs,
            num_classes=2,
            metric_names=metrics,
            n_bootstrap=500,
            seed=99,
        )

        for name in metrics:
            assert r1[name] == r2[name]

    def test_loss_metric_positive(self, binary_data):
        """Bootstrap loss CI bounds should be positive."""
        targets, predictions, probs = binary_data

        result = bootstrap_classification_metrics(
            targets,
            predictions,
            probs,
            num_classes=2,
            metric_names=["loss"],
            n_bootstrap=200,
            seed=42,
        )
        lo, hi = result["loss"]
        assert lo > 0
        assert hi > 0

    def test_unknown_metric_skipped(self, binary_data):
        """Unknown metric names are skipped with a warning."""
        targets, predictions, probs = binary_data

        result = bootstrap_classification_metrics(
            targets,
            predictions,
            probs,
            num_classes=2,
            metric_names=["recall_1", "nonexistent_metric"],
            n_bootstrap=50,
            seed=42,
        )

        assert "recall_1" in result
        assert "nonexistent_metric" not in result

    def test_handles_single_class_bootstrap_sample(self):
        """Handles degenerate bootstrap samples that contain only one class."""
        # Small dataset where some bootstrap samples will be single-class
        targets = np.array([0, 0, 0, 0, 1])
        predictions = np.array([0, 0, 0, 0, 1])
        probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.1, 0.9]])

        result = bootstrap_classification_metrics(
            targets,
            predictions,
            probs,
            num_classes=2,
            metric_names=["roc_auc", "recall_1"],
            n_bootstrap=200,
            seed=42,
        )

        # roc_auc may have NaN CIs due to single-class samples, or valid CIs
        # Either way it should not crash
        assert "roc_auc" in result
        assert "recall_1" in result
