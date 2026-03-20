"""Tests for Bootstrap Confidence Interval Utilities

Tests for the bootstrap CI computation in src/utils/bootstrap.py.
"""

import numpy as np
import pytest

from src.utils.bootstrap import bootstrap_ci, bootstrap_classification_metrics

# ============================================================================
# Unit Tests - bootstrap_ci
# ============================================================================


@pytest.mark.unit
class TestBootstrapCI:
    """Tests for the low-level bootstrap_ci function."""

    def test_returns_tuple_of_two_floats(self):
        """bootstrap_ci returns a (lower, upper) tuple."""

        def mean_fn(idx: np.ndarray) -> float:
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            return float(np.mean(data[idx]))

        lo, hi = bootstrap_ci(mean_fn, n_samples=5, n_bootstrap=100, seed=42)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi

    def test_ci_contains_point_estimate(self):
        """95% CI should contain the point estimate for well-behaved data."""
        data = np.random.RandomState(42).normal(10.0, 1.0, size=100)

        def mean_fn(idx: np.ndarray) -> float:
            return float(np.mean(data[idx]))

        lo, hi = bootstrap_ci(
            mean_fn, n_samples=100, n_bootstrap=5000, confidence_level=0.95, seed=42
        )
        point_estimate = float(np.mean(data))
        assert lo <= point_estimate <= hi

    def test_wider_ci_with_higher_confidence(self):
        """99% CI should be wider than 90% CI."""
        data = np.random.RandomState(42).normal(0.0, 1.0, size=50)

        def mean_fn(idx: np.ndarray) -> float:
            return float(np.mean(data[idx]))

        lo_90, hi_90 = bootstrap_ci(
            mean_fn, n_samples=50, n_bootstrap=5000, confidence_level=0.90, seed=42
        )
        lo_99, hi_99 = bootstrap_ci(
            mean_fn, n_samples=50, n_bootstrap=5000, confidence_level=0.99, seed=42
        )
        assert (hi_99 - lo_99) >= (hi_90 - lo_90)

    def test_reproducibility_with_seed(self):
        """Same seed produces same CI."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def mean_fn(idx: np.ndarray) -> float:
            return float(np.mean(data[idx]))

        result1 = bootstrap_ci(mean_fn, n_samples=5, n_bootstrap=1000, seed=123)
        result2 = bootstrap_ci(mean_fn, n_samples=5, n_bootstrap=1000, seed=123)
        assert result1 == result2

    def test_handles_metric_fn_raising_valueerror(self):
        """Iterations where metric_fn raises ValueError are excluded."""
        call_count = 0

        def flaky_fn(idx: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ValueError("degenerate sample")
            return float(np.mean(idx))

        lo, hi = bootstrap_ci(flaky_fn, n_samples=10, n_bootstrap=100, seed=42)
        assert not np.isnan(lo)
        assert not np.isnan(hi)

    def test_handles_all_nan_returns(self):
        """Returns (nan, nan) when all iterations produce NaN."""

        def nan_fn(_idx: np.ndarray) -> float:
            return float("nan")

        lo, hi = bootstrap_ci(nan_fn, n_samples=5, n_bootstrap=50, seed=42)
        assert np.isnan(lo)
        assert np.isnan(hi)

    def test_handles_all_errors(self):
        """Returns (nan, nan) when all iterations raise."""

        def error_fn(_idx: np.ndarray) -> float:
            raise ValueError("always fails")

        lo, hi = bootstrap_ci(error_fn, n_samples=5, n_bootstrap=50, seed=42)
        assert np.isnan(lo)
        assert np.isnan(hi)


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
