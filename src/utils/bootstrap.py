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
    rng = np.random.default_rng(seed)
    labels = list(range(num_classes))

    # Build metric functions; precision/recall/f1 are grouped separately
    # so that precision_recall_fscore_support is called at most once per
    # bootstrap iteration.
    scalar_fns, prf_metrics = _build_metric_fns(
        targets, predictions, probs, num_classes, metric_names
    )

    all_metric_names = list(scalar_fns.keys()) + list(prf_metrics.keys())
    metric_values: Dict[str, List[float]] = {name: [] for name in all_metric_names}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)

        # Scalar metrics (independent calls)
        for name, fn in scalar_fns.items():
            try:
                value = fn(indices)
                if not np.isnan(value):
                    metric_values[name].append(value)
            except (ValueError, ZeroDivisionError):
                continue

        # Precision/recall/f1 — one call for all per-class metrics
        if prf_metrics:
            try:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    targets[indices],
                    predictions[indices],
                    labels=labels,
                    zero_division=0.0,  # type: ignore[arg-type]
                )
                for name, (kind, cls_idx) in prf_metrics.items():
                    if kind == "precision":
                        v = float(prec[cls_idx])  # type: ignore[index]
                    elif kind == "recall":
                        v = float(rec[cls_idx])  # type: ignore[index]
                    else:
                        v = float(f1[cls_idx])  # type: ignore[index]
                    if not np.isnan(v):
                        metric_values[name].append(v)
            except (ValueError, ZeroDivisionError):
                pass

    results: Dict[str, Tuple[float, float]] = {}
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
) -> Tuple[
    Dict[str, Callable[[np.ndarray], float]],
    Dict[str, Tuple[str, int]],
]:
    """Build metric functions for bootstrap evaluation.

    Precision, recall, and f1 metrics are grouped so that
    ``precision_recall_fscore_support`` is called only once per bootstrap
    iteration regardless of how many per-class metrics are requested.

    Args:
        targets: Ground truth labels.
        predictions: Predicted labels.
        probs: Predicted probabilities.
        num_classes: Number of classes.
        metric_names: List of metric names.

    Returns:
        Tuple of (scalar_fns, prf_metrics).
        scalar_fns: Dict mapping metric name -> callable(indices) -> float,
            for metrics that are computed independently.
        prf_metrics: Dict mapping metric name -> (kind, class_idx) where
            kind is "precision", "recall", or "f1". These are computed
            together via a single ``precision_recall_fscore_support`` call.
    """
    scalar_fns: Dict[str, Callable[[np.ndarray], float]] = {}
    prf_metrics: Dict[str, Tuple[str, int]] = {}
    labels = list(range(num_classes))

    for name in metric_names:
        if name == "balanced_accuracy":

            def _balanced_acc(idx: np.ndarray) -> float:
                return float(balanced_accuracy_score(targets[idx], predictions[idx]))

            scalar_fns[name] = _balanced_acc

        elif name == "accuracy":

            def _accuracy(idx: np.ndarray) -> float:
                return float(100.0 * np.mean(targets[idx] == predictions[idx]))

            scalar_fns[name] = _accuracy

        elif name == "loss":

            def _loss(idx: np.ndarray) -> float:
                t = targets[idx]
                p = probs[idx]
                # Cross-entropy recomputed from stored probabilities.
                # This may differ slightly from the point-estimate loss
                # reported by the trainer, which is computed batch-by-batch
                # from logits via nn.CrossEntropyLoss.
                n_cls = p.shape[1]
                y_onehot = np.eye(n_cls)[t]
                return float(-(y_onehot * np.log(p + 1e-12)).sum(axis=1).mean())

            scalar_fns[name] = _loss

        elif name == "roc_auc":

            def _roc_auc(idx: np.ndarray) -> float:
                t = targets[idx]
                unique = np.unique(t)
                if len(unique) < 2:
                    return float("nan")
                if num_classes == 2:
                    return float(roc_auc_score(t, probs[idx, 1]))
                return float(
                    roc_auc_score(
                        t,
                        probs[idx],
                        multi_class="ovr",
                        average="weighted",
                        labels=labels,
                    )
                )

            scalar_fns[name] = _roc_auc

        elif name == "pr_auc":

            def _pr_auc(idx: np.ndarray) -> float:
                t = targets[idx]
                unique = np.unique(t)
                if len(unique) < 2:
                    return float("nan")
                if num_classes == 2:
                    return float(average_precision_score(t, probs[idx, 1]))
                # One-hot encode with all classes to handle missing classes
                # in bootstrap samples
                t_onehot = np.eye(num_classes)[t]
                return float(
                    average_precision_score(t_onehot, probs[idx], average="weighted")
                )

            scalar_fns[name] = _pr_auc

        elif name.startswith(("precision_", "recall_", "f1_")):
            kind = name.split("_")[0]
            try:
                cls_idx = int(name.split("_")[1])
            except ValueError:
                _logger.warning(f"Cannot parse class index from metric: {name}")
                continue
            if cls_idx >= num_classes:
                _logger.warning(
                    f"Class index {cls_idx} out of range for "
                    f"{num_classes} classes: {name}"
                )
                continue
            prf_metrics[name] = (kind, cls_idx)

        else:
            _logger.warning(f"Unknown metric for bootstrap: {name}")

    return scalar_fns, prf_metrics
