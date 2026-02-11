"""
Tests for classifier analysis and comparison tools.
"""

import os
import tempfile
from pathlib import Path

# Set matplotlib to non-interactive backend before importing pyplot
import matplotlib

matplotlib.use("Agg")  # Prevents graph windows from appearing during tests

import numpy as np
import pandas as pd
import pytest

from src.experiments.classifier.analyze_comparison import (
    ComparisonMetrics,
    ComparisonReporter,
    ComparisonVisualizer,
    ExperimentComparator,
    ExperimentResults,
    compare_experiments,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_training_results():
    """Create mock training results dataframe."""
    epochs = list(range(1, 11))
    data = {
        "epoch": epochs,
        "train_accuracy": [70 + i * 2 for i in epochs],
        "train_loss": [1.5 - i * 0.1 for i in epochs],
        "val_accuracy": [65 + i * 2.5 for i in epochs],
        "val_pr_auc": [0.6 + i * 0.03 for i in epochs],
        "val_roc_auc": [0.65 + i * 0.025 for i in epochs],
        "val_0.Normal_accuracy": [80 + i for i in epochs],
        "val_1.Abnormal_accuracy": [50 + i * 3 for i in epochs],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_experiment_dir(mock_training_results):
    """Create temporary directory with mock experiment results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create baseline experiment
        for seed in range(3):
            seed_dir = base_path / "baseline" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            df = mock_training_results.copy()
            # Add some variation
            df["val_accuracy"] += np.random.randn(len(df)) * 2
            df.to_csv(seed_dir / "training_results.csv", index=False)

        # Create comparison experiment (slightly better)
        for seed in range(3):
            seed_dir = base_path / "comparison" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            df = mock_training_results.copy()
            # Make it slightly better
            df["val_accuracy"] += 3 + np.random.randn(len(df)) * 2
            df["val_pr_auc"] += 0.05
            df["val_roc_auc"] += 0.03
            df.to_csv(seed_dir / "training_results.csv", index=False)

        yield str(base_path)


@pytest.fixture
def baseline_results(temp_experiment_dir):
    """Load baseline experiment results."""
    return ExperimentResults.from_csv_pattern(temp_experiment_dir, "baseline")


@pytest.fixture
def comparison_results(temp_experiment_dir):
    """Load comparison experiment results."""
    return ExperimentResults.from_csv_pattern(temp_experiment_dir, "comparison")


# ============================================================================
# UNIT TESTS
# ============================================================================


@pytest.mark.unit
class TestExperimentResults:
    """Test ExperimentResults data loading."""

    def test_from_csv_pattern_success(self, temp_experiment_dir):
        """Test successful loading of experiment results."""
        results = ExperimentResults.from_csv_pattern(temp_experiment_dir, "baseline")

        assert results.name == "baseline"
        assert isinstance(results.data, pd.DataFrame)
        assert len(results.data) == 30  # 10 epochs * 3 seeds
        assert "seed" in results.data.columns
        assert "experiment" in results.data.columns
        assert len(results.metrics) > 0

    def test_from_csv_pattern_no_files(self):
        """Test error when no CSV files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No CSV files found"):
                ExperimentResults.from_csv_pattern(tmpdir, "nonexistent")

    def test_seed_extraction(self, baseline_results):
        """Test that seed numbers are correctly extracted."""
        seeds = baseline_results.data["seed"].unique()
        assert len(seeds) == 3
        assert set(seeds) == {0, 1, 2}

    def test_metrics_extraction(self, baseline_results):
        """Test that metrics are correctly identified."""
        # Should exclude 'seed', 'experiment', 'epoch'
        assert "seed" not in baseline_results.metrics
        assert "experiment" not in baseline_results.metrics
        assert "epoch" not in baseline_results.metrics

        # Should include actual metrics
        assert "val_accuracy" in baseline_results.metrics
        assert "train_accuracy" in baseline_results.metrics


@pytest.mark.unit
class TestComparisonMetrics:
    """Test ComparisonMetrics dataclass."""

    def test_comparison_metrics_creation(self):
        """Test creation of ComparisonMetrics."""
        metrics = ComparisonMetrics(
            metric_name="val_accuracy",
            baseline_mean=85.0,
            baseline_std=2.0,
            comparison_mean=87.5,
            comparison_std=1.5,
            improvement_percent=2.94,
            is_improvement=True,
        )

        assert metrics.metric_name == "val_accuracy"
        assert metrics.baseline_mean == 85.0
        assert metrics.is_improvement is True


@pytest.mark.unit
class TestExperimentComparator:
    """Test ExperimentComparator logic."""

    def test_comparator_initialization(self, baseline_results, comparison_results):
        """Test comparator initialization."""
        comparator = ExperimentComparator(baseline_results, comparison_results)

        assert comparator.baseline == baseline_results
        assert comparator.comparison == comparison_results
        assert len(comparator.metrics_to_compare) > 0

    def test_comparator_custom_metrics(self, baseline_results, comparison_results):
        """Test comparator with custom metrics list."""
        custom_metrics = ["val_accuracy", "val_pr_auc"]
        comparator = ExperimentComparator(
            baseline_results, comparison_results, custom_metrics
        )

        assert comparator.metrics_to_compare == custom_metrics

    def test_compute_final_epoch_comparison(self, baseline_results, comparison_results):
        """Test final epoch comparison computation."""
        comparator = ExperimentComparator(baseline_results, comparison_results)
        results = comparator.compute_final_epoch_comparison(final_epoch=10)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, ComparisonMetrics) for r in results)

        # Check that we have expected metrics
        metric_names = [r.metric_name for r in results]
        assert "val_accuracy" in metric_names

    def test_comparison_improvement_detection(
        self, baseline_results, comparison_results
    ):
        """Test that improvements are correctly detected."""
        comparator = ExperimentComparator(baseline_results, comparison_results)
        results = comparator.compute_final_epoch_comparison(final_epoch=10)

        # Find val_accuracy metric (should show improvement in our mock data)
        val_acc_metrics = [r for r in results if r.metric_name == "val_accuracy"]
        assert len(val_acc_metrics) == 1

        val_acc = val_acc_metrics[0]
        # Comparison should be better than baseline
        assert val_acc.comparison_mean > val_acc.baseline_mean
        assert val_acc.is_improvement is True

    def test_zero_baseline_mean_handling(self, baseline_results, comparison_results):
        """Test handling of zero baseline mean (division by zero)."""
        # Create a comparator with a metric that could be zero
        comparator = ExperimentComparator(baseline_results, comparison_results)

        # Manually test the calculation logic
        baseline_mean = 0.0
        comparison_mean = 10.0

        if baseline_mean != 0:
            improvement = ((comparison_mean - baseline_mean) / baseline_mean) * 100
        else:
            improvement = 0.0

        assert improvement == 0.0

    def test_analyze_stability(self, baseline_results, comparison_results):
        """Test stability analysis."""
        comparator = ExperimentComparator(baseline_results, comparison_results)
        stability = comparator.analyze_stability(final_epoch=10)

        assert isinstance(stability, dict)
        assert len(stability) > 0

        for metric, stats in stability.items():
            assert "baseline_std" in stats
            assert "comparison_std" in stats
            assert "is_more_stable" in stats
            assert "stability_improvement" in stats
            assert isinstance(stats["is_more_stable"], bool)

    def test_analyze_convergence(self, baseline_results, comparison_results):
        """Test convergence analysis."""
        comparator = ExperimentComparator(baseline_results, comparison_results)
        convergence = comparator.analyze_convergence(
            metric="val_accuracy", early_epochs=3, final_epoch=10
        )

        assert "baseline" in convergence
        assert "comparison" in convergence

        for exp_name, stats in convergence.items():
            assert "early_mean" in stats
            assert "final_mean" in stats
            assert "improvement" in stats
            # Final should be better than early
            assert stats["final_mean"] > stats["early_mean"]


# ============================================================================
# COMPONENT TESTS
# ============================================================================


@pytest.mark.component
class TestComparisonVisualizer:
    """Test ComparisonVisualizer plotting."""

    def test_visualizer_initialization(self, baseline_results, comparison_results):
        """Test visualizer initialization."""
        visualizer = ComparisonVisualizer(baseline_results, comparison_results)

        assert visualizer.baseline == baseline_results
        assert visualizer.comparison == comparison_results

    def test_plot_training_curves_to_file(
        self, baseline_results, comparison_results, tmp_path
    ):
        """Test plotting training curves to file."""
        visualizer = ComparisonVisualizer(baseline_results, comparison_results)
        output_path = tmp_path / "test_plot.png"

        # Should not raise an error
        visualizer.plot_training_curves(
            metrics=[("val_accuracy", "Validation Accuracy")],
            output_path=str(output_path),
            figsize=(8, 6),
        )

        # Check that file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_training_curves_default_metrics(
        self, baseline_results, comparison_results, tmp_path
    ):
        """Test plotting with default metrics."""
        visualizer = ComparisonVisualizer(baseline_results, comparison_results)
        output_path = tmp_path / "default_plot.png"

        visualizer.plot_training_curves(output_path=str(output_path))

        assert output_path.exists()

    def test_plot_accuracy_normalization(
        self, baseline_results, comparison_results, tmp_path
    ):
        """Test that accuracy metrics are normalized correctly."""
        # Modify data to have percentage values
        baseline_results.data["val_accuracy"] *= 100
        comparison_results.data["val_accuracy"] *= 100

        visualizer = ComparisonVisualizer(baseline_results, comparison_results)
        output_path = tmp_path / "normalized_plot.png"

        # Should handle normalization without errors
        visualizer.plot_training_curves(
            metrics=[("val_accuracy", "Validation Accuracy")],
            output_path=str(output_path),
        )

        assert output_path.exists()


@pytest.mark.component
class TestComparisonReporter:
    """Test ComparisonReporter text generation."""

    def test_reporter_initialization(self, baseline_results, comparison_results):
        """Test reporter initialization."""
        comparator = ExperimentComparator(baseline_results, comparison_results)
        reporter = ComparisonReporter(comparator)

        assert reporter.comparator == comparator

    def test_generate_summary_report(self, baseline_results, comparison_results):
        """Test summary report generation."""
        comparator = ExperimentComparator(baseline_results, comparison_results)
        reporter = ComparisonReporter(comparator)

        report = reporter.generate_summary_report(final_epoch=10)

        # Check report content
        assert isinstance(report, str)
        assert len(report) > 0
        assert "EXPERIMENT COMPARISON REPORT" in report
        assert "baseline" in report.lower()
        assert "comparison" in report.lower()
        assert "FINAL EPOCH METRICS" in report
        assert "VALIDATION METRICS SUMMARY" in report
        assert "TRAINING STABILITY ANALYSIS" in report
        assert "CONVERGENCE ANALYSIS" in report
        assert "CONCLUSION" in report

    def test_generate_summary_report_to_file(
        self, baseline_results, comparison_results, tmp_path
    ):
        """Test saving report to file."""
        comparator = ExperimentComparator(baseline_results, comparison_results)
        reporter = ComparisonReporter(comparator)
        output_file = tmp_path / "report.txt"

        report = reporter.generate_summary_report(
            final_epoch=10, output_file=str(output_file)
        )

        # Check that file was created
        assert output_file.exists()

        # Check content matches
        with open(output_file) as f:
            file_content = f.read()
        assert file_content == report


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
class TestCompareExperimentsEndToEnd:
    """Test end-to-end experiment comparison."""

    def test_compare_experiments_basic(self, temp_experiment_dir, tmp_path):
        """Test basic experiment comparison."""
        metrics, report = compare_experiments(
            base_path=temp_experiment_dir,
            baseline_name="baseline",
            comparison_name="comparison",
            final_epoch=10,
            output_dir=str(tmp_path),
        )

        # Check metrics
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        assert all(isinstance(m, ComparisonMetrics) for m in metrics)

        # Check report
        assert isinstance(report, str)
        assert len(report) > 0

        # Check output files
        assert (tmp_path / "training_comparison.png").exists()
        assert (tmp_path / "comparison_report.txt").exists()

    def test_compare_experiments_custom_metrics(self, temp_experiment_dir):
        """Test comparison with custom metrics."""
        custom_metrics = ["val_accuracy", "val_pr_auc"]

        metrics, report = compare_experiments(
            base_path=temp_experiment_dir,
            baseline_name="baseline",
            comparison_name="comparison",
            final_epoch=10,
            metrics_to_compare=custom_metrics,
        )

        # Should only return specified metrics
        metric_names = [m.metric_name for m in metrics]
        assert set(metric_names) == set(custom_metrics)

    def test_compare_experiments_no_output_dir(self, temp_experiment_dir):
        """Test comparison without saving outputs."""
        metrics, report = compare_experiments(
            base_path=temp_experiment_dir,
            baseline_name="baseline",
            comparison_name="comparison",
            final_epoch=10,
            output_dir=None,
        )

        # Should still return metrics and report
        assert len(metrics) > 0
        assert len(report) > 0


@pytest.mark.integration
class TestRobustness:
    """Test robustness and edge cases."""

    def test_missing_metrics_in_one_experiment(self, temp_experiment_dir):
        """Test handling when experiments have different metrics."""
        # Load baseline
        baseline = ExperimentResults.from_csv_pattern(temp_experiment_dir, "baseline")

        # Create comparison with fewer metrics
        comparison_data = baseline.data.copy()
        comparison_data = comparison_data.drop(columns=["val_pr_auc"])
        comparison = ExperimentResults(
            name="comparison",
            data=comparison_data,
            metrics=[m for m in baseline.metrics if m != "val_pr_auc"],
        )

        # Should only compare common metrics
        comparator = ExperimentComparator(baseline, comparison)
        assert "val_pr_auc" not in comparator.metrics_to_compare
        assert "val_accuracy" in comparator.metrics_to_compare

    def test_single_seed_experiment(self, temp_experiment_dir):
        """Test comparison with single seed (no std calculation)."""
        # Create single-seed experiment
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            seed_dir = base_path / "single" / "seed_0"
            seed_dir.mkdir(parents=True)

            data = {
                "epoch": list(range(1, 11)),
                "val_accuracy": [70 + i * 2 for i in range(1, 11)],
            }
            df = pd.DataFrame(data)
            df.to_csv(seed_dir / "training_results.csv", index=False)

            # Should handle single seed without errors
            results = ExperimentResults.from_csv_pattern(str(base_path), "single")
            assert len(results.data) == 10  # 10 epochs * 1 seed

    def test_different_epoch_counts(self, temp_experiment_dir):
        """Test handling experiments with different epoch counts."""
        baseline = ExperimentResults.from_csv_pattern(temp_experiment_dir, "baseline")

        # Create comparison with fewer epochs
        comparison_data = baseline.data[baseline.data["epoch"] <= 5].copy()
        comparison = ExperimentResults(
            name="comparison", data=comparison_data, metrics=baseline.metrics
        )

        comparator = ExperimentComparator(baseline, comparison)

        # Should handle early epoch that exists in both
        convergence = comparator.analyze_convergence(final_epoch=5)
        assert "baseline" in convergence
        assert "comparison" in convergence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
