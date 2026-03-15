"""Checkpoint Utility Functions

Standalone functions for saving and loading training checkpoints.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    is_best: bool = False,
    metrics: Optional[Dict[str, float]] = None,
    best_metric: Optional[float] = None,
    best_metric_name: Optional[str] = None,
    trainer_class: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Save training checkpoint.

    Args:
        path: Path to save checkpoint file
        model: The model whose state to save
        optimizer: The optimizer whose state to save
        epoch: Current epoch number
        global_step: Current global training step
        is_best: Whether this is the best model so far
        metrics: Dictionary of current metrics
        best_metric: Best metric value so far
        best_metric_name: Name of the best metric
        trainer_class: Name of the trainer class
        **kwargs: Additional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "is_best": is_best,
    }

    if trainer_class is not None:
        checkpoint["trainer_class"] = trainer_class

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
        checkpoint["best_metric_name"] = best_metric_name

    checkpoint.update(kwargs)

    torch.save(checkpoint, path)

    if is_best:
        logger.info(f"✓ Best model checkpoint saved: {path}")
    else:
        logger.info(f"✓ Checkpoint saved: {path}")

    logger.info(f"  Epoch: {epoch}, Global step: {global_step}")

    if metrics:
        metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        logger.info(f"  Metrics: {metrics_str}")

    if best_metric is not None:
        logger.info(f"  Best {best_metric_name}: {best_metric:.6f}")

    logger.debug(f"  Checkpoint keys: {list(checkpoint.keys())}")
    logger.debug(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into (None to skip)
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary containing full checkpoint data

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"Checkpoint not found: {path}")
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as e:
        logger.critical(f"Failed to load checkpoint from {path}")
        logger.exception(f"Error details: {e}")
        raise

    logger.debug(f"  Checkpoint keys: {list(checkpoint.keys())}")
    logger.debug(f"  Trainer class: {checkpoint.get('trainer_class', 'unknown')}")

    # Load model state
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    except Exception as e:
        logger.error("Failed to load model state dict")
        logger.exception(f"Error details: {e}")
        if strict:
            raise
        else:
            logger.warning("Continuing with non-strict loading")

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.error("Failed to load optimizer state dict")
            logger.exception(f"Error details: {e}")
            logger.warning("Continuing without optimizer state")

    logger.info("✓ Checkpoint loaded successfully")
    logger.info(
        f"  Epoch: {checkpoint.get('epoch', 0)}, "
        f"Global step: {checkpoint.get('global_step', 0)}"
    )

    if "metrics" in checkpoint:
        metrics_str = ", ".join(
            [f"{k}: {v:.6f}" for k, v in checkpoint["metrics"].items()]
        )
        logger.info(f"  Loaded metrics: {metrics_str}")

    if checkpoint.get("best_metric") is not None:
        logger.info(
            f"  Best {checkpoint.get('best_metric_name')}: "
            f"{checkpoint['best_metric']:.6f}"
        )

    return checkpoint


def is_best_metric(
    current_value: float, best_value: Optional[float], mode: str
) -> bool:
    """Check if current metric value is the best so far.

    Args:
        current_value: Current metric value
        best_value: Previous best metric value (None if first)
        mode: 'min' or 'max' for comparison

    Returns:
        True if current value is better than best so far
    """
    if best_value is None:
        return True

    if mode == "min":
        return current_value < best_value
    elif mode == "max":
        return current_value > best_value
    else:
        raise ValueError(f"Invalid metric mode: {mode}. Must be 'min' or 'max'")
