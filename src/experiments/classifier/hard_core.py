"""Hard-core direct evaluation (abnormal-vs-suspicious restricted PR-AUC).

The multi-class one-vs-rest PR-AUC is inflated because the "rest" includes easy
negatives (e.g. ``red``). The genuine residual ceiling lives at the
**abnormal <-> suspicious** boundary. This module measures that boundary
directly by restricting evaluation to samples whose *true* label is either the
positive (abnormal) class or a single contrast (suspicious) class.

Two scores are reported:

- **renorm** (primary): ``P(abn) / (P(abn) + P(susp))``. This is the functional
  analog of a dedicated 2-class classifier's softmax over the two logits, so it
  is the apples-to-apples comparison against the binary oracle ceiling (0.915).
- **raw** (secondary / sanity): ``P(abn)``. Divergence from ``renorm`` reflects
  how much probability mass leaks to the other classes.

The metrics are pure functions of the saved ``(targets, probs)`` arrays, so they
can be computed both at eval time (``src/main.py``) and post-hoc from
``predictions_{split}.npz`` (``hard_core_analysis.py``).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_logger = logging.getLogger(__name__)

# The metric keys produced by :func:`compute_hard_core_metrics`. ``hardcore_n``
# is an int sample count; the rest are floats (NaN when not computable).
HARD_CORE_METRIC_KEYS: List[str] = [
    "hardcore_pr_auc_renorm",
    "hardcore_pr_auc_raw",
    "hardcore_roc_auc_renorm",
    "hardcore_leak_rate",
]

DEFAULT_CONTRAST_NAME = "suspicious"


def resolve_contrast_class(
    class_names: Optional[List[str]],
    override: Optional[int] = None,
    name: str = DEFAULT_CONTRAST_NAME,
) -> Optional[int]:
    """Resolve the contrast (suspicious) class index.

    Precedence: an explicit ``override`` index always wins; otherwise the index
    of the class literally named ``name`` in ``class_names`` (ordered by index);
    otherwise ``None`` (the caller should skip the hard-core metric).

    Args:
        class_names: Class names ordered by label index, or None/empty.
        override: Explicit contrast class index, or None to auto-detect.
        name: Class name to look up when auto-detecting (default "suspicious").

    Returns:
        The contrast class index, or None if it cannot be resolved.
    """
    if override is not None:
        return override
    if not class_names:
        return None
    for idx, cname in enumerate(class_names):
        if cname == name:
            return idx
    return None


def compute_hard_core_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    positive_class: int,
    contrast_class: int,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute the abnormal-vs-contrast restricted ranking metrics.

    Evaluation is restricted to samples whose *true* label is ``positive_class``
    or ``contrast_class``; the positive class is the ranking target.

    Args:
        targets: Ground-truth labels, shape (N,).
        probs: Predicted class probabilities, shape (N, num_classes).
        positive_class: Index of the positive (abnormal) class.
        contrast_class: Index of the contrast (suspicious) class.
        eps: Stabilizer for the renormalization denominator.

    Returns:
        Dict with ``hardcore_pr_auc_renorm``, ``hardcore_pr_auc_raw``,
        ``hardcore_roc_auc_renorm``, ``hardcore_leak_rate`` (floats, NaN when not
        computable) and ``hardcore_n`` (int restricted sample count). Degenerate
        inputs (bad shapes/indices, empty restricted set, single class present)
        yield NaN rather than raising.
    """
    out: Dict[str, float] = {
        "hardcore_pr_auc_renorm": float("nan"),
        "hardcore_pr_auc_raw": float("nan"),
        "hardcore_roc_auc_renorm": float("nan"),
        "hardcore_leak_rate": float("nan"),
        "hardcore_n": 0,
    }

    try:
        targets = np.asarray(targets).astype(int)
    except (TypeError, ValueError):
        _logger.warning("Hard-core: targets must be coercible to int labels")
        return out
    probs = np.asarray(probs, dtype=float)

    if probs.ndim != 2:
        _logger.warning("Hard-core: probs must be 2-D, got shape %s", probs.shape)
        return out
    if targets.ndim != 1:
        _logger.warning("Hard-core: targets must be 1-D, got shape %s", targets.shape)
        return out
    if targets.shape[0] != probs.shape[0]:
        _logger.warning(
            "Hard-core: targets/probs length mismatch: %d vs %d",
            targets.shape[0],
            probs.shape[0],
        )
        return out
    num_classes = probs.shape[1]
    if not (0 <= positive_class < num_classes) or not (
        0 <= contrast_class < num_classes
    ):
        _logger.warning(
            "Hard-core: positive_class=%d / contrast_class=%d out of range for "
            "%d classes",
            positive_class,
            contrast_class,
            num_classes,
        )
        return out
    if positive_class == contrast_class:
        _logger.warning("Hard-core: positive_class equals contrast_class; skipping")
        return out

    mask = (targets == positive_class) | (targets == contrast_class)
    n = int(mask.sum())
    out["hardcore_n"] = n
    if n == 0:
        return out

    # abn->susp leak: fraction of true-abnormal whose argmax prediction lands on
    # the contrast class. Independent of the restricted ranking below, so it is
    # computed even when the restricted set is degenerate for AUC.
    abn_mask = targets == positive_class
    if int(abn_mask.sum()) > 0:
        argmax_pred = probs[abn_mask].argmax(axis=1)
        out["hardcore_leak_rate"] = float(np.mean(argmax_pred == contrast_class))

    y_true = (targets[mask] == positive_class).astype(int)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == n:
        # Only one class present in the restricted set: AP/ROC undefined.
        return out

    p_abn = probs[mask, positive_class]
    p_con = probs[mask, contrast_class]
    score_renorm = p_abn / (p_abn + p_con + eps)

    out["hardcore_pr_auc_renorm"] = float(average_precision_score(y_true, score_renorm))
    out["hardcore_pr_auc_raw"] = float(average_precision_score(y_true, p_abn))
    out["hardcore_roc_auc_renorm"] = float(roc_auc_score(y_true, score_renorm))
    return out
