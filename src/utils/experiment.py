"""Shared experiment setup utilities."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from src.utils.config import resolve_output_path, save_config
from src.utils.device import get_device
from src.utils.git import get_git_info
from src.utils.logging import get_log_file_path, setup_logging

# Module-level logger
logger = logging.getLogger(__name__)


def setup_experiment_common(
    config: Dict[str, Any],
    experiment_label: str,
) -> Tuple[str, Path]:
    """Shared experiment setup: directories, logging, device, seed, config snapshot.

    Args:
        config: The merged experiment configuration dictionary.
        experiment_label: Banner label to log (e.g. "CLASSIFIER EXPERIMENT STARTED").

    Returns:
        Tuple of (device_str, log_dir_path).
    """
    # 1. Create log directory
    log_dir = resolve_output_path(config, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. Get logging configuration (with defaults)
    logging_config = config.get("logging", {})
    console_level = logging_config.get("console_level", "INFO")
    file_level = logging_config.get("file_level", "DEBUG")
    log_format = logging_config.get("format")
    date_format = logging_config.get("date_format")
    timezone = logging_config.get("timezone", "UTC")
    module_levels = logging_config.get("module_levels")

    # 3. Setup logging FIRST (before any other operations)
    log_file = get_log_file_path(
        output_base_dir=config["output"]["base_dir"],
        log_subdir=config["output"]["subdirs"]["logs"],
        timezone=timezone,
    )

    setup_logging(
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        log_format=log_format,
        date_format=date_format,
        timezone=timezone,
        module_levels=module_levels,
    )

    # 4. Log experiment start with banner
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(experiment_label)
    logger.info("=" * 80)

    # 5. Log Git information for reproducibility
    git_info = get_git_info()
    if git_info["repository_url"]:
        logger.info(f"Repository: {git_info['repository_url']}")
    if git_info["commit_hash"]:
        logger.info(f"Commit: {git_info['commit_hash']}")

    logger.info(f"Log file: {log_file}")
    logger.info(f"Console log level: {console_level}")
    logger.info(f"File log level: {file_level}")
    logger.info(f"Log directory: {log_dir}")

    # 6. Resolve device
    compute_config = config.get("compute", {})
    device_config = compute_config.get("device", "auto")
    if device_config == "auto":
        device = get_device()
    else:
        device = device_config
    device = str(device)

    logger.info(f"Using device: {device}")

    # 7. Set seed if configured
    seed = compute_config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")

    # 8. Save configuration to log directory
    config_save_path = log_dir / "config.yaml"
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")

    return (device, log_dir)


def create_experiment_logger(
    config: Dict[str, Any],
    log_dir: Path,
    subdirs: Optional[Dict[str, str]] = None,
) -> Any:
    """Create ExperimentLogger with tensorboard config from experiment config.

    Args:
        config: Full experiment configuration dictionary.
        log_dir: Log directory path (from setup_experiment_common).
        subdirs: Optional subdirectory mapping for the logger.

    Returns:
        Configured ExperimentLogger instance.
    """
    from src.utils.experiment_logger import ExperimentLogger

    logging_config = config.get("logging", {})
    tensorboard_config = logging_config.get("metrics", {}).get("tensorboard", {})
    tb_log_dir = (
        resolve_output_path(config, "tensorboard")
        if "tensorboard" in config.get("output", {}).get("subdirs", {})
        else None
    )

    return ExperimentLogger(
        log_dir=log_dir,
        subdirs=subdirs,
        tensorboard_config=tensorboard_config,
        tb_log_dir=tb_log_dir,
    )


def run_training(
    trainer: Any,
    metrics_logger: Any,
    *,
    num_epochs: int,
    checkpoint_dir: str,
    save_best: bool,
    checkpoint_frequency: int,
    save_latest: bool,
    validate_frequency: int,
    best_metric: str,
) -> None:
    """Run trainer.train() with standard error handling and cleanup.

    Wraps the training call with KeyboardInterrupt and Exception handling,
    and ensures metrics_logger.close() is called in all exit paths.

    Args:
        trainer: Trainer instance with a train() method.
        metrics_logger: ExperimentLogger to close on exit.
        num_epochs: Number of training epochs.
        checkpoint_dir: Directory for saving checkpoints.
        save_best: Whether to save only the best checkpoint.
        checkpoint_frequency: How often to save checkpoints (in epochs).
        save_latest: Whether to save a latest checkpoint symlink.
        validate_frequency: How often to run validation (in epochs).
        best_metric: Metric name for best-model selection.
    """
    logger.info("")
    logger.info(f"Starting training for {num_epochs} epochs...")

    try:
        trainer.train(
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            save_best=save_best,
            checkpoint_frequency=checkpoint_frequency,
            save_latest_checkpoint=save_latest,
            validate_frequency=validate_frequency,
            best_metric=best_metric,
        )
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Training interrupted by user")
        metrics_logger.close()
        sys.exit(0)
    except Exception as e:
        logger.error("")
        logger.exception(f"Training failed with error: {e}")
        metrics_logger.close()
        raise

    metrics_logger.close()
    logger.info("")
    logger.info("Training completed successfully!")
