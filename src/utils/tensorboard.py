"""TensorBoard Utility Functions

This module provides utility functions for TensorBoard integration including
optional writer creation, safe logging with error handling, and path resolution.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

# Optional import - don't fail if tensorboard not installed
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_tensorboard_available() -> bool:
    """Check if TensorBoard is available.

    Returns:
        True if tensorboard package is installed, False otherwise
    """
    return TENSORBOARD_AVAILABLE


def create_tensorboard_writer(
    log_dir: Union[str, Path],
    flush_secs: int = 30,
    enabled: bool = True,
) -> Optional["SummaryWriter"]:
    """Create a TensorBoard SummaryWriter if enabled and available.

    Args:
        log_dir: Directory to save TensorBoard logs
        flush_secs: Flush frequency in seconds
        enabled: Whether TensorBoard logging is enabled

    Returns:
        SummaryWriter instance if enabled and available, None otherwise
    """
    if not enabled:
        logger.info("TensorBoard logging disabled by configuration")
        return None

    if not is_tensorboard_available():
        logger.warning(
            "TensorBoard logging enabled but tensorboard package not installed. "
            "Install with: pip install tensorboard"
        )
        return None

    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
        logger.info(f"TensorBoard logging enabled: {log_dir}")
        return writer
    except Exception as e:
        logger.error(f"Failed to create TensorBoard writer: {e}")
        return None


def safe_log_scalar(
    writer: Optional["SummaryWriter"],
    tag: str,
    value: Union[float, int, torch.Tensor],
    step: int,
) -> None:
    """Safely log a scalar value to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        value: Scalar value to log
        step: Global step value
    """
    if writer is None:
        return

    try:
        if isinstance(value, torch.Tensor):
            value = value.item()
        writer.add_scalar(tag, value, step)
    except Exception as e:
        logger.debug(f"Failed to log scalar '{tag}': {e}")


def safe_log_scalars(
    writer: Optional["SummaryWriter"],
    main_tag: str,
    tag_scalar_dict: Dict[str, Union[float, int, torch.Tensor]],
    step: int,
) -> None:
    """Safely log multiple scalar values to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        main_tag: Parent name for the group
        tag_scalar_dict: Dictionary of tag-value pairs
        step: Global step value
    """
    if writer is None:
        return

    try:
        processed_dict = {}
        for key, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                processed_dict[key] = value.item()
            else:
                processed_dict[key] = value
        writer.add_scalars(main_tag, processed_dict, step)
    except Exception as e:
        logger.debug(f"Failed to log scalars '{main_tag}': {e}")


def safe_log_images(
    writer: Optional["SummaryWriter"],
    tag: str,
    img_tensor: torch.Tensor,
    step: int,
    dataformats: str = "NCHW",
) -> None:
    """Safely log images to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        img_tensor: Image tensor (N, C, H, W) or (C, H, W)
        step: Global step value
        dataformats: Format of image tensor (default: NCHW)
    """
    if writer is None:
        return

    try:
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        writer.add_images(tag, img_tensor, step, dataformats=dataformats)
    except Exception as e:
        logger.debug(f"Failed to log images '{tag}': {e}")


def safe_log_histogram(
    writer: Optional["SummaryWriter"],
    tag: str,
    values: Union[torch.Tensor, Any],
    step: int,
    bins: str = "tensorflow",
) -> None:
    """Safely log histogram to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        values: Values to histogram
        step: Global step value
        bins: Binning method ('tensorflow', 'auto', 'fd', etc.)
    """
    if writer is None:
        return

    try:
        writer.add_histogram(tag, values, step, bins=bins)
    except Exception as e:
        logger.debug(f"Failed to log histogram '{tag}': {e}")


def safe_log_figure(
    writer: Optional["SummaryWriter"],
    tag: str,
    figure: Any,
    step: int,
    close: bool = True,
) -> None:
    """Safely log matplotlib figure to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        figure: Matplotlib figure object
        step: Global step value
        close: Whether to close figure after logging
    """
    if writer is None:
        return

    try:
        writer.add_figure(tag, figure, step, close=close)
    except Exception as e:
        logger.debug(f"Failed to log figure '{tag}': {e}")


def safe_log_text(
    writer: Optional["SummaryWriter"],
    tag: str,
    text_string: str,
    step: int,
) -> None:
    """Safely log text to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        text_string: Text content
        step: Global step value
    """
    if writer is None:
        return

    try:
        writer.add_text(tag, text_string, step)
    except Exception as e:
        logger.debug(f"Failed to log text '{tag}': {e}")


def safe_log_hparams(
    writer: Optional["SummaryWriter"],
    hparam_dict: Dict[str, Any],
    metric_dict: Optional[Dict[str, float]] = None,
) -> None:
    """Safely log hyperparameters to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        hparam_dict: Dictionary of hyperparameters
        metric_dict: Dictionary of metric values (optional)
    """
    if writer is None:
        return

    try:
        flat_hparams = _flatten_dict(hparam_dict)
        writer.add_hparams(flat_hparams, metric_dict or {})
    except Exception as e:
        logger.debug(f"Failed to log hyperparameters: {e}")


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            if not isinstance(v, (int, float, str, bool)):
                v = str(v)
            items.append((new_key, v))
    return dict(items)


def close_tensorboard_writer(writer: Optional["SummaryWriter"]) -> None:
    """Safely close a TensorBoard writer.

    Args:
        writer: SummaryWriter instance (can be None)
    """
    if writer is None:
        return

    try:
        writer.close()
        logger.debug("TensorBoard writer closed")
    except Exception as e:
        logger.error(f"Failed to close TensorBoard writer: {e}")
