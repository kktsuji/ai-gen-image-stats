"""Configurable classification losses for imbalanced learning (Track B / item 4).

The minority (abnormal) class is rare, so plain cross-entropy under-weights it.
This module provides drop-in alternatives that the classifier models dispatch to
via :func:`build_loss`:

- ``focal`` — Focal loss (Lin et al., 2017): down-weights easy examples by
  ``(1 - p_t) ** gamma``. With ``gamma=0`` and ``alpha=None`` it reduces exactly
  to cross-entropy.
- ``class_balanced`` — Class-Balanced loss (Cui et al., 2019): re-weights classes
  by the inverse effective number of samples and wraps either cross-entropy or
  focal. Reuses :func:`src.utils.data.samplers.compute_effective_num_weights` so
  the effective-number formula lives in exactly one place.

All loss callables share the model ``compute_loss`` signature
``(predictions, targets, reduction="mean") -> Tensor`` so the trainer is agnostic
to which loss is in use.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.utils.data.samplers import compute_effective_num_weights

LossFn = Callable[..., torch.Tensor]


def focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-class focal loss.

    Args:
        predictions: Raw logits of shape [batch_size, num_classes].
        targets: Integer class indices of shape [batch_size].
        gamma: Focusing parameter (>= 0). gamma=0 (and alpha=None) is cross-entropy.
        alpha: Optional per-class weight tensor of shape [num_classes]. Indexed by
            target to weight each sample's loss.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (mean/sum) or per-sample losses (none).
    """
    log_probs = F.log_softmax(predictions, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1.0 - pt) ** gamma) * log_pt

    if alpha is not None:
        alpha = alpha.to(predictions.device)
        loss = alpha.gather(0, targets) * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(
        f"Invalid reduction: {reduction!r}. Must be 'mean', 'sum', or 'none'"
    )


def class_balanced_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    base: str = "cross_entropy",
    gamma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Class-Balanced loss: effective-number reweighting over a base loss.

    Args:
        predictions: Raw logits of shape [batch_size, num_classes].
        targets: Integer class indices of shape [batch_size].
        weights: Per-class weight tensor of shape [num_classes] (effective-number
            weights, typically normalized to sum to num_classes).
        base: 'cross_entropy' or 'focal' — the loss that the weights re-balance.
        gamma: Focusing parameter when base='focal'.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (mean/sum) or per-sample losses (none).

    Raises:
        ValueError: If base is not recognized.
    """
    weights = weights.to(predictions.device)
    if base == "cross_entropy":
        return F.cross_entropy(
            predictions, targets, weight=weights, reduction=reduction
        )
    if base == "focal":
        return focal_loss(
            predictions, targets, gamma=gamma, alpha=weights, reduction=reduction
        )
    raise ValueError(
        f"Invalid class_balanced base: {base!r}. Must be 'cross_entropy' or 'focal'"
    )


def _effective_num_weight_tensor(
    class_counts: List[int],
    beta: float,
    num_classes: int,
) -> torch.Tensor:
    """Build a normalized effective-number weight tensor from per-class counts.

    Reuses :func:`compute_effective_num_weights` (which expects a flat list of
    targets) by reconstructing a minimal targets list from the counts, so the
    Cui et al. formula is not duplicated here. Weights are normalized to sum to
    num_classes (a class with the average count gets weight ~1).

    The reconstructed ``targets`` list costs O(N) memory where N is the total
    number of training samples. This runs once at model build time (not per
    batch), so the cost is acceptable for the deduplication it buys.
    """
    targets: List[int] = []
    for cls_idx, count in enumerate(class_counts):
        targets.extend([cls_idx] * int(count))
    weights_dict: Dict[int, float] = compute_effective_num_weights(targets, beta=beta)

    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx, w in weights_dict.items():
        weights[cls_idx] = w
    total = float(weights.sum())
    if total > 0:
        weights = weights * (num_classes / total)
    return weights


def build_loss(
    loss_config: Dict[str, Any],
    class_counts: Optional[List[int]],
    num_classes: int,
) -> LossFn:
    """Build a loss callable from a validated ``model.loss`` config.

    The returned callable has the model ``compute_loss`` signature
    ``(predictions, targets, reduction="mean") -> Tensor``.

    Args:
        loss_config: Validated loss config. ``type`` is one of 'cross_entropy',
            'focal', 'class_balanced'; remaining keys are type-specific
            (gamma/alpha for focal, beta/base/gamma for class_balanced).
        class_counts: Per-class training sample counts, required for
            'class_balanced'. Ignored otherwise.
        num_classes: Number of classes.

    Returns:
        A loss callable.

    Raises:
        ValueError: If the loss type is unknown, or class_counts is missing for
            'class_balanced'.
    """
    loss_type = loss_config.get("type", "cross_entropy")

    if loss_type == "cross_entropy":

        def _ce(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            return F.cross_entropy(predictions, targets, reduction=reduction)

        return _ce

    if loss_type == "focal":
        gamma = float(loss_config["gamma"])
        alpha_list = loss_config.get("alpha")
        alpha = (
            torch.tensor(alpha_list, dtype=torch.float32)
            if alpha_list is not None
            else None
        )

        def _focal(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            return focal_loss(
                predictions, targets, gamma=gamma, alpha=alpha, reduction=reduction
            )

        return _focal

    if loss_type == "class_balanced":
        if class_counts is None:
            raise ValueError(
                "class_balanced loss requires class_counts (per-class training "
                "sample counts)"
            )
        # A class with zero training samples gets effective-number weight 0 (it is
        # absent from compute_effective_num_weights' output), which silently zeroes
        # its loss contribution. Fail loudly instead of training a model that is
        # never penalized for misclassifying that class.
        if any(c <= 0 for c in class_counts):
            raise ValueError(
                "class_balanced loss requires every class to have at least one "
                f"training sample; got class_counts={class_counts}"
            )
        beta = float(loss_config["beta"])
        # validate_loss_section requires "base" for class_balanced, so it is always
        # present here (no default needed).
        base = loss_config["base"]
        gamma = float(loss_config.get("gamma", 0.0))
        weights = _effective_num_weight_tensor(class_counts, beta, num_classes)

        def _cb(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            reduction: str = "mean",
        ) -> torch.Tensor:
            return class_balanced_loss(
                predictions,
                targets,
                weights=weights,
                base=base,
                gamma=gamma,
                reduction=reduction,
            )

        return _cb

    raise ValueError(
        f"Invalid loss type: {loss_type!r}. Must be one of "
        f"'cross_entropy', 'focal', 'class_balanced'"
    )
