"""Bootstrap Confidence Interval Utilities

Provides bootstrap resampling for computing confidence intervals
on classifier evaluation metrics.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

_logger = logging.getLogger(__name__)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n_samples: int,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function that takes bootstrap indices and returns a scalar metric.
        n_samples: Total number of samples to resample from.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level in (0, 1), e.g. 0.95 for 95% CI.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    bootstrap_values: List[float] = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, size=n_samples)
        try:
            value = metric_fn(indices)
            if not np.isnan(value):
                bootstrap_values.append(value)
        except (ValueError, ZeroDivisionError):
            continue

    if not bootstrap_values:
        return (float("nan"), float("nan"))

    alpha = 1.0 - confidence_level
    lower = float(np.nanpercentile(bootstrap_values, 100 * alpha / 2))
    upper = float(np.nanpercentile(bootstrap_values, 100 * (1 - alpha / 2)))
    return (lower, upper)


def bootstrap_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    metric_names: List[str],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap CIs for classification metrics.

    All metrics share the same bootstrap samples (correlated bootstrap).

    Args:
        targets: Ground truth labels, shape (N,).
        predictions: Predicted labels, shape (N,).
        probs: Predicted probabilities, shape (N, num_classes).
        num_classes: Number of classes.
        metric_names: List of metric names to compute CIs for.
            Supported: recall_{i}, precision_{i}, f1_{i}, balanced_accuracy,
            roc_auc, pr_auc, accuracy, loss.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level in (0, 1).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping metric_name -> (ci_lower, ci_upper).
    """
    n_samples = len(targets)
    rng = np.random.RandomState(seed)

    # Pre-generate all bootstrap index arrays for correlated bootstrap
    all_indices = rng.randint(0, n_samples, size=(n_bootstrap, n_samples))

    # Build metric lambdas
    metric_fns = _build_metric_fns(
        targets, predictions, probs, num_classes, metric_names
    )

    results: Dict[str, Tuple[float, float]] = {}

    # Compute all bootstrap values in one pass over indices
    metric_values: Dict[str, List[float]] = {name: [] for name in metric_fns}

    for i in range(n_bootstrap):
        indices = all_indices[i]
        for name, fn in metric_fns.items():
            try:
                value = fn(indices)
                if not np.isnan(value):
                    metric_values[name].append(value)
            except (ValueError, ZeroDivisionError):
                continue

    alpha = 1.0 - confidence_level
    for name, values in metric_values.items():
        if not values:
            results[name] = (float("nan"), float("nan"))
        else:
            lower = float(np.nanpercentile(values, 100 * alpha / 2))
            upper = float(np.nanpercentile(values, 100 * (1 - alpha / 2)))
            results[name] = (lower, upper)

    return results


def _build_metric_fns(
    targets: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    metric_names: List[str],
) -> Dict[str, Callable[[np.ndarray], float]]:
    """Build metric lambda functions for bootstrap evaluation.

    Args:
        targets: Ground truth labels.
        predictions: Predicted labels.
        probs: Predicted probabilities.
        num_classes: Number of classes.
        metric_names: List of metric names.

    Returns:
        Dict mapping metric name -> callable(indices) -> float.
    """
    fns: Dict[str, Callable[[np.ndarray], float]] = {}
    labels = list(range(num_classes))

    for name in metric_names:
        if name == "balanced_accuracy":

            def _balanced_acc(idx: np.ndarray) -> float:
                return float(balanced_accuracy_score(targets[idx], predictions[idx]))

            fns[name] = _balanced_acc

        elif name == "accuracy":

            def _accuracy(idx: np.ndarray) -> float:
                return float(100.0 * np.mean(targets[idx] == predictions[idx]))

            fns[name] = _accuracy

        elif name == "loss":

            def _loss(idx: np.ndarray) -> float:
                t = targets[idx]
                p = probs[idx]
                # Cross-entropy from stored probabilities
                n_cls = p.shape[1]
                y_onehot = np.eye(n_cls)[t]
                return float(-(y_onehot * np.log(p + 1e-12)).sum(axis=1).mean())

            fns[name] = _loss

        elif name == "roc_auc":

            def _roc_auc(idx: np.ndarray) -> float:
                t = targets[idx]
                unique = np.unique(t)
                if len(unique) < 2:
                    return float("nan")
                if num_classes == 2:
                    return float(roc_auc_score(t, probs[idx, 1]))
                elif len(unique) == num_classes:
                    return float(
                        roc_auc_score(
                            t, probs[idx], multi_class="ovr", average="weighted"
                        )
                    )
                return float("nan")

            fns[name] = _roc_auc

        elif name == "pr_auc":

            def _pr_auc(idx: np.ndarray) -> float:
                t = targets[idx]
                unique = np.unique(t)
                if len(unique) < 2:
                    return float("nan")
                if num_classes == 2:
                    return float(average_precision_score(t, probs[idx, 1]))
                elif len(unique) == num_classes:
                    return float(
                        average_precision_score(t, probs[idx], average="weighted")
                    )
                return float("nan")

            fns[name] = _pr_auc

        elif name.startswith("precision_"):
            cls_idx = int(name.split("_")[1])

            def _precision(idx: np.ndarray, c: int = cls_idx) -> float:
                prec, _, _, _ = precision_recall_fscore_support(
                    targets[idx],
                    predictions[idx],
                    labels=labels,
                    zero_division=0.0,  # type: ignore[arg-type]
                )
                return float(prec[c])  # type: ignore[index]

            fns[name] = _precision

        elif name.startswith("recall_"):
            cls_idx = int(name.split("_")[1])

            def _recall(idx: np.ndarray, c: int = cls_idx) -> float:
                _, rec, _, _ = precision_recall_fscore_support(
                    targets[idx],
                    predictions[idx],
                    labels=labels,
                    zero_division=0.0,  # type: ignore[arg-type]
                )
                return float(rec[c])  # type: ignore[index]

            fns[name] = _recall

        elif name.startswith("f1_"):
            cls_idx = int(name.split("_")[1])

            def _f1(idx: np.ndarray, c: int = cls_idx) -> float:
                _, _, f1, _ = precision_recall_fscore_support(
                    targets[idx],
                    predictions[idx],
                    labels=labels,
                    zero_division=0.0,  # type: ignore[arg-type]
                )
                return float(f1[c])  # type: ignore[index]

            fns[name] = _f1

        else:
            _logger.warning(f"Unknown metric for bootstrap: {name}")

    return fns
