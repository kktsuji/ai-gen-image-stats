"""
Classifier Analysis and Comparison Tools

This module provides utilities for analyzing and comparing classifier training results
across different experiments (e.g., baseline vs. synthetic data augmentation).
"""

import os
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ExperimentResults:
    """Container for experiment results across multiple seeds."""

    name: str
    data: pd.DataFrame
    metrics: List[str]

    @classmethod
    def from_csv_pattern(
        cls, base_path: str, experiment_name: str, pattern: str = "seed_*"
    ) -> "ExperimentResults":
        """
        Load experiment results from CSV files matching a pattern.

        Args:
            base_path: Base directory containing experiment results
            experiment_name: Name of the experiment
            pattern: Glob pattern for finding result files (default: "seed_*")

        Returns:
            ExperimentResults object containing all loaded data

        Raises:
            FileNotFoundError: If no matching CSV files are found
        """
        results = []
        search_pattern = os.path.join(
            base_path, experiment_name, pattern, "training_results.csv"
        )
        csv_paths = sorted(glob(search_pattern))

        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found matching: {search_pattern}")

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            # Extract seed number from path
            seed = int(csv_path.split("seed_")[1].split(os.sep)[0])
            df["seed"] = seed
            df["experiment"] = experiment_name
            results.append(df)

        combined_data = pd.concat(results, ignore_index=True)
        metrics = [
            col
            for col in combined_data.columns
            if col not in ["seed", "experiment", "epoch"]
        ]

        return cls(name=experiment_name, data=combined_data, metrics=metrics)


@dataclass
class ComparisonMetrics:
    """Metrics comparing two experiments."""

    metric_name: str
    baseline_mean: float
    baseline_std: float
    comparison_mean: float
    comparison_std: float
    improvement_percent: float
    is_improvement: bool


class ExperimentComparator:
    """Compare results between two experiments."""

    def __init__(
        self,
        baseline: ExperimentResults,
        comparison: ExperimentResults,
        metrics_to_compare: Optional[List[str]] = None,
    ):
        """
        Initialize comparator with two experiments.

        Args:
            baseline: Baseline experiment results
            comparison: Comparison experiment results
            metrics_to_compare: List of metric names to compare (default: all shared metrics)
        """
        self.baseline = baseline
        self.comparison = comparison

        if metrics_to_compare is None:
            # Use all common metrics
            baseline_metrics = set(baseline.metrics)
            comparison_metrics = set(comparison.metrics)
            self.metrics_to_compare = sorted(baseline_metrics & comparison_metrics)
        else:
            self.metrics_to_compare = metrics_to_compare

    def compute_final_epoch_comparison(
        self, final_epoch: int = 10
    ) -> List[ComparisonMetrics]:
        """
        Compare metrics at the final epoch across all seeds.

        Args:
            final_epoch: Epoch number to use for comparison (default: 10)

        Returns:
            List of ComparisonMetrics for each metric
        """
        baseline_final = self.baseline.data[self.baseline.data["epoch"] == final_epoch]
        comparison_final = self.comparison.data[
            self.comparison.data["epoch"] == final_epoch
        ]

        results = []
        for metric in self.metrics_to_compare:
            baseline_values = baseline_final[metric].values
            comparison_values = comparison_final[metric].values

            baseline_mean = baseline_values.mean()
            baseline_std = baseline_values.std()
            comparison_mean = comparison_values.mean()
            comparison_std = comparison_values.std()

            # Calculate improvement percentage
            if baseline_mean != 0:
                improvement = ((comparison_mean - baseline_mean) / baseline_mean) * 100
            else:
                improvement = 0.0

            results.append(
                ComparisonMetrics(
                    metric_name=metric,
                    baseline_mean=float(baseline_mean),
                    baseline_std=float(baseline_std),
                    comparison_mean=float(comparison_mean),
                    comparison_std=float(comparison_std),
                    improvement_percent=float(improvement),
                    is_improvement=bool(improvement > 0),
                )
            )

        return results

    def analyze_stability(
        self, metrics: Optional[List[str]] = None, final_epoch: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze training stability by comparing standard deviations.

        Args:
            metrics: List of metrics to analyze (default: validation metrics)
            final_epoch: Epoch to analyze (default: 10)

        Returns:
            Dictionary mapping metric name to stability statistics
        """
        if metrics is None:
            metrics = [m for m in self.metrics_to_compare if m.startswith("val_")]

        baseline_final = self.baseline.data[self.baseline.data["epoch"] == final_epoch]
        comparison_final = self.comparison.data[
            self.comparison.data["epoch"] == final_epoch
        ]

        stability = {}
        for metric in metrics:
            baseline_std = baseline_final[metric].std()
            comparison_std = comparison_final[metric].std()

            stability[metric] = {
                "baseline_std": float(baseline_std),
                "comparison_std": float(comparison_std),
                "is_more_stable": bool(comparison_std < baseline_std),
                "stability_improvement": float(baseline_std - comparison_std),
            }

        return stability

    def analyze_convergence(
        self, metric: str = "val_accuracy", early_epochs: int = 3, final_epoch: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze convergence by comparing early vs final epoch performance.

        Args:
            metric: Metric to analyze
            early_epochs: Number of early epochs to average (default: 3)
            final_epoch: Final epoch to compare against (default: 10)

        Returns:
            Dictionary containing convergence analysis for both experiments
        """
        results = {}

        for exp_name, exp_data in [
            ("baseline", self.baseline.data),
            ("comparison", self.comparison.data),
        ]:
            early_values = exp_data[exp_data["epoch"] <= early_epochs][metric].mean()
            final_values = exp_data[exp_data["epoch"] == final_epoch][metric].mean()

            results[exp_name] = {
                "early_mean": early_values,
                "final_mean": final_values,
                "improvement": final_values - early_values,
            }

        return results


class ComparisonVisualizer:
    """Create visualizations comparing experiments."""

    def __init__(self, baseline: ExperimentResults, comparison: ExperimentResults):
        """
        Initialize visualizer with two experiments.

        Args:
            baseline: Baseline experiment results
            comparison: Comparison experiment results
        """
        self.baseline = baseline
        self.comparison = comparison

    def plot_training_curves(
        self,
        metrics: Optional[List[Tuple[str, str]]] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:
        """
        Plot training curves comparing baseline and comparison experiments.

        Args:
            metrics: List of (metric_name, display_title) tuples
            output_path: Path to save the figure (if None, display only)
            figsize: Figure size (width, height)
        """
        if metrics is None:
            # Use default metrics, but filter to only those that exist in both datasets
            default_metrics = [
                ("val_accuracy", "Validation Accuracy"),
                ("val_pr_auc", "Validation PR-AUC"),
                ("val_roc_auc", "Validation ROC-AUC"),
                ("train_accuracy", "Training Accuracy"),
                ("train_pr_auc", "Training PR-AUC"),
                ("train_roc_auc", "Training ROC-AUC"),
            ]
            # Filter to only available metrics
            baseline_cols = set(self.baseline.data.columns)
            comparison_cols = set(self.comparison.data.columns)
            available_cols = baseline_cols & comparison_cols
            metrics = [
                (metric, title)
                for metric, title in default_metrics
                if metric in available_cols
            ]

        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(
            f"Training Comparison: {self.baseline.name} vs. {self.comparison.name}",
            fontsize=14,
            fontweight="bold",
        )

        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (metric, title) in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Plot baseline
            baseline_grouped = self.baseline.data.groupby("epoch")[metric].agg(
                ["mean", "std"]
            )
            # Normalize accuracy metrics to 0-1 range if they're in percentage
            if "accuracy" in metric and baseline_grouped["mean"].max() > 1.5:
                baseline_grouped["mean"] /= 100.0
                baseline_grouped["std"] /= 100.0

            ax.plot(
                baseline_grouped.index,
                baseline_grouped["mean"],
                label=f"Baseline ({self.baseline.name})",
                color="blue",
                linewidth=2,
            )
            ax.fill_between(
                baseline_grouped.index,
                baseline_grouped["mean"] - baseline_grouped["std"],
                baseline_grouped["mean"] + baseline_grouped["std"],
                alpha=0.2,
                color="blue",
            )

            # Plot comparison
            comparison_grouped = self.comparison.data.groupby("epoch")[metric].agg(
                ["mean", "std"]
            )
            if "accuracy" in metric and comparison_grouped["mean"].max() > 1.5:
                comparison_grouped["mean"] /= 100.0
                comparison_grouped["std"] /= 100.0

            ax.plot(
                comparison_grouped.index,
                comparison_grouped["mean"],
                label=f"Comparison ({self.comparison.name})",
                color="red",
                linewidth=2,
            )
            ax.fill_between(
                comparison_grouped.index,
                comparison_grouped["mean"] - comparison_grouped["std"],
                comparison_grouped["mean"] + comparison_grouped["std"],
                alpha=0.2,
                color="red",
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()


class ComparisonReporter:
    """Generate text reports for experiment comparisons."""

    def __init__(self, comparator: ExperimentComparator):
        """
        Initialize reporter with a comparator.

        Args:
            comparator: ExperimentComparator instance
        """
        self.comparator = comparator

    def generate_summary_report(
        self, final_epoch: int = 10, output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive comparison summary report.

        Args:
            final_epoch: Epoch to use for final comparison
            output_file: Optional file path to save the report

        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EXPERIMENT COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"\nBaseline: {self.comparator.baseline.name}")
        lines.append(f"Comparison: {self.comparator.comparison.name}")
        lines.append(f"Final Epoch: {final_epoch}")

        # Final epoch metrics comparison
        comparison_metrics = self.comparator.compute_final_epoch_comparison(final_epoch)

        lines.append("\n" + "=" * 80)
        lines.append("FINAL EPOCH METRICS - MEAN ± STD ACROSS SEEDS")
        lines.append("=" * 80)

        for metric in comparison_metrics:
            lines.append(f"\n{metric.metric_name}:")
            lines.append(
                f"  Baseline:   {metric.baseline_mean:.4f} ± {metric.baseline_std:.4f}"
            )
            lines.append(
                f"  Comparison: {metric.comparison_mean:.4f} ± {metric.comparison_std:.4f}"
            )
            symbol = "✓" if metric.is_improvement else "✗"
            lines.append(f"  Change: {metric.improvement_percent:+.2f}% {symbol}")

        # Validation metrics summary
        lines.append("\n" + "=" * 80)
        lines.append("VALIDATION METRICS SUMMARY")
        lines.append("=" * 80)

        val_metrics = [
            m for m in comparison_metrics if m.metric_name.startswith("val_")
        ]
        improvements = sum(1 for m in val_metrics if m.is_improvement)
        total = len(val_metrics)

        lines.append(f"\nImproved: {improvements}/{total} validation metrics")
        for metric in val_metrics:
            symbol = "✓" if metric.is_improvement else "✗"
            metric_display = (
                metric.metric_name.replace("val_", "").replace("_", " ").title()
            )
            lines.append(
                f"  {symbol} {metric_display}: {metric.improvement_percent:+.2f}%"
            )

        # Stability analysis
        lines.append("\n" + "=" * 80)
        lines.append("TRAINING STABILITY ANALYSIS")
        lines.append("=" * 80)
        lines.append("(Lower standard deviation = more stable)")

        stability = self.comparator.analyze_stability(final_epoch=final_epoch)
        for metric, stats in stability.items():
            lines.append(f"\n{metric}:")
            lines.append(f"  Baseline std:   {stats['baseline_std']:.4f}")
            lines.append(f"  Comparison std: {stats['comparison_std']:.4f}")
            symbol = "✓" if stats["is_more_stable"] else "✗"
            lines.append(
                f"  {symbol} {'More stable' if stats['is_more_stable'] else 'Less stable'}"
            )

        # Convergence analysis
        lines.append("\n" + "=" * 80)
        lines.append("CONVERGENCE ANALYSIS")
        lines.append("=" * 80)

        convergence = self.comparator.analyze_convergence(
            metric="val_accuracy", final_epoch=final_epoch
        )
        for exp_name, stats in convergence.items():
            display_name = "Baseline" if exp_name == "baseline" else "Comparison"
            lines.append(f"\n{display_name}:")
            lines.append(f"  Early epochs (1-3): {stats['early_mean']:.4f}")
            lines.append(
                f"  Final epoch ({final_epoch}):     {stats['final_mean']:.4f}"
            )
            lines.append(f"  Improvement:        {stats['improvement']:+.4f}")

        # Overall conclusion
        lines.append("\n" + "=" * 80)
        lines.append("CONCLUSION")
        lines.append("=" * 80)

        improvement_count = sum(1 for m in val_metrics if m.is_improvement)
        if improvement_count >= len(val_metrics) * 0.5:
            lines.append("\n✓ Comparison experiment shows positive impact overall")
        else:
            lines.append("\n✗ Comparison experiment does not show clear improvement")

        report_text = "\n".join(lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")

        return report_text


def compare_experiments(
    base_path: str,
    baseline_name: str,
    comparison_name: str,
    final_epoch: int = 10,
    output_dir: Optional[str] = None,
    metrics_to_compare: Optional[List[str]] = None,
) -> Tuple[List[ComparisonMetrics], str]:
    """
    High-level function to compare two experiments.

    Args:
        base_path: Base directory containing experiment results
        baseline_name: Name of the baseline experiment
        comparison_name: Name of the comparison experiment
        final_epoch: Final epoch for comparison
        output_dir: Directory to save outputs (plots and reports)
        metrics_to_compare: List of metrics to compare (default: all common metrics)

    Returns:
        Tuple of (comparison_metrics, report_text)
    """
    # Load experiments
    baseline = ExperimentResults.from_csv_pattern(base_path, baseline_name)
    comparison = ExperimentResults.from_csv_pattern(base_path, comparison_name)

    # Create comparator
    comparator = ExperimentComparator(baseline, comparison, metrics_to_compare)

    # Generate report
    reporter = ComparisonReporter(comparator)
    report_text = reporter.generate_summary_report(final_epoch=final_epoch)
    print(report_text)

    # Create visualization
    visualizer = ComparisonVisualizer(baseline, comparison)
    plot_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "training_comparison.png")
        report_path = os.path.join(output_dir, "comparison_report.txt")

        with open(report_path, "w") as f:
            f.write(report_text)

    visualizer.plot_training_curves(output_path=plot_path)

    # Get comparison metrics for return
    comparison_metrics = comparator.compute_final_epoch_comparison(final_epoch)

    return comparison_metrics, report_text


if __name__ == "__main__":
    """
    Example usage for comparing baseline vs synthetic data experiments.
    """
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python analyze_comparison.py <base_path> <baseline_name> <comparison_name> [output_dir]"
        )
        print("\nExample:")
        print("  python analyze_comparison.py ./outputs train train-synth ./analysis")
        sys.exit(1)

    base_path = sys.argv[1]
    baseline_name = sys.argv[2]
    comparison_name = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else None

    compare_experiments(
        base_path=base_path,
        baseline_name=baseline_name,
        comparison_name=comparison_name,
        output_dir=output_dir,
    )
