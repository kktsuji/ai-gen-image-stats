"""Training utility factories for optimizers and schedulers."""

from typing import Any, Dict, List, Optional

import torch
import torch.optim


def create_optimizer(
    model_parameters: Any,
    optimizer_config: Dict[str, Any],
    valid_types: Optional[List[str]] = None,
) -> torch.optim.Optimizer:
    """Create an optimizer from configuration.

    Args:
        model_parameters: Model parameters to optimize.
        optimizer_config: Optimizer configuration dict with at least a 'type' key.
        valid_types: Optional list of allowed optimizer types. Defaults to
            ["adam", "adamw", "sgd"].

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer type is unknown or not in valid_types.
    """
    if valid_types is None:
        valid_types = ["adam", "adamw", "sgd"]

    optimizer_type = (optimizer_config.get("type") or "").lower()

    if optimizer_type not in valid_types:
        raise ValueError(
            f"Unknown optimizer type: '{optimizer_type}'. Valid types: {valid_types}"
        )

    # Extract known non-kwarg keys
    excluded_keys = {"type", "learning_rate", "gradient_clip_norm"}
    kwargs = {k: v for k, v in optimizer_config.items() if k not in excluded_keys}

    lr = optimizer_config.get("learning_rate", 1e-4)

    if optimizer_type == "adam":
        return torch.optim.Adam(model_parameters, lr=lr, **kwargs)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model_parameters, lr=lr, **kwargs)
    elif optimizer_type == "sgd":
        if "momentum" not in kwargs:
            kwargs["momentum"] = 0.9
        return torch.optim.SGD(model_parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unhandled optimizer type: '{optimizer_type}'")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any],
    num_epochs: int,
    valid_types: Optional[List[Optional[str]]] = None,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Create a learning rate scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_config: Scheduler configuration dict with at least a 'type' key.
        num_epochs: Total number of training epochs (used to resolve 'auto' T_max).
        valid_types: Optional list of allowed scheduler types. Defaults to
            [None, "none", "cosine", "step", "plateau"].

    Returns:
        Configured scheduler instance, or None if type is None or "none".

    Raises:
        ValueError: If scheduler type is unknown or not in valid_types.
    """
    if valid_types is None:
        valid_types = [None, "none", "cosine", "step", "plateau"]

    scheduler_type = scheduler_config.get("type")
    if isinstance(scheduler_type, str):
        scheduler_type_lower = scheduler_type.lower()
    else:
        scheduler_type_lower = scheduler_type

    if scheduler_type_lower not in valid_types:
        raise ValueError(
            f"Unknown scheduler type: '{scheduler_type}'. "
            f"Valid types: {[t for t in valid_types if t is not None]}"
        )

    if scheduler_type is None or (
        isinstance(scheduler_type, str) and scheduler_type.lower() == "none"
    ):
        return None

    # Extract known non-kwarg keys
    excluded_keys = {"type"}
    kwargs = {k: v for k, v in scheduler_config.items() if k not in excluded_keys}

    scheduler_type_str = scheduler_type.lower()

    if scheduler_type_str == "cosine":
        if "T_max" in kwargs:
            t_max = kwargs.pop("T_max")
        elif "t_max" in kwargs:
            t_max = kwargs.pop("t_max")
        else:
            t_max = "auto"
        if t_max == "auto":
            t_max = num_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, **kwargs
        )
    elif scheduler_type_str == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type_str == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unhandled scheduler type: '{scheduler_type_str}'")
