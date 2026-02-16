"""Base Trainer Interface

This module defines the abstract base class for all trainers in the experiments.
All experiment-specific trainers should inherit from BaseTrainer and implement
the required abstract methods.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

from src.base.dataloader import BaseDataLoader
from src.base.logger import BaseLogger
from src.base.model import BaseModel

# Module-level logger
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    This class provides a common interface and shared functionality for trainers
    across different experiment types (GAN, Diffusion, Classifier). It implements
    the training loop structure, checkpointing, and validation orchestration.

    All trainers should inherit from this class and implement:
    - train_epoch(): Execute one training epoch
    - validate_epoch(): Execute one validation epoch
    - get_model(): Return the model being trained
    - get_dataloader(): Return the dataloader
    - get_optimizer(): Return the optimizer
    - get_logger(): Return the logger

    The class provides concrete implementations for:
    - train(): Main training loop with checkpointing
    - save_checkpoint(): Save training state
    - load_checkpoint(): Load training state
    - resume_training(): Resume from checkpoint

    Example:
        >>> class MyClassifierTrainer(BaseTrainer):
        ...     def __init__(self, model, dataloader, optimizer, logger):
        ...         super().__init__()
        ...         self.model = model
        ...         self.dataloader = dataloader
        ...         self.optimizer = optimizer
        ...         self.logger = logger
        ...         self.epoch = 0
        ...
        ...     def train_epoch(self) -> Dict[str, float]:
        ...         self.model.train()
        ...         total_loss = 0.0
        ...         train_loader = self.dataloader.get_train_loader()
        ...
        ...         for batch_idx, (data, target) in enumerate(train_loader):
        ...             self.optimizer.zero_grad()
        ...             predictions = self.model(data)
        ...             loss = self.model.compute_loss(predictions, target)
        ...             loss.backward()
        ...             self.optimizer.step()
        ...             total_loss += loss.item()
        ...
        ...         return {'loss': total_loss / len(train_loader)}
        ...
        ...     def validate_epoch(self) -> Dict[str, float]:
        ...         self.model.eval()
        ...         val_loader = self.dataloader.get_val_loader()
        ...
        ...         with torch.no_grad():
        ...             # Validation logic
        ...             pass
        ...
        ...         return {'val_loss': 0.0}
        ...
        ...     def get_model(self) -> BaseModel:
        ...         return self.model
        ...
        ...     def get_dataloader(self) -> BaseDataLoader:
        ...         return self.dataloader
        ...
        ...     def get_optimizer(self) -> torch.optim.Optimizer:
        ...         return self.optimizer
        ...
        ...     def get_logger(self) -> BaseLogger:
        ...         return self.logger
    """

    def __init__(self):
        """Initialize base trainer.

        Subclasses should call super().__init__() and set up their own
        model, dataloader, optimizer, and logger.
        """
        self._current_epoch = 0
        self._global_step = 0
        self._best_metric = None
        self._best_metric_name = None

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch.

        This method must be implemented by all subclasses. It should:
        1. Set model to training mode
        2. Iterate through training data
        3. Perform forward pass, backward pass, and optimization
        4. Return metrics for the epoch

        Returns:
            Dictionary of metric names to values for the epoch.
            Must include at least 'loss' key.

        Example:
            >>> def train_epoch(self) -> Dict[str, float]:
            ...     self.model.train()
            ...     total_loss = 0.0
            ...     train_loader = self.dataloader.get_train_loader()
            ...
            ...     for batch_idx, (data, target) in enumerate(train_loader):
            ...         self.optimizer.zero_grad()
            ...         predictions = self.model(data)
            ...         loss = self.model.compute_loss(predictions, target)
            ...         loss.backward()
            ...         self.optimizer.step()
            ...         total_loss += loss.item()
            ...         self._global_step += 1
            ...
            ...     return {'loss': total_loss / len(train_loader)}
        """
        raise NotImplementedError("Subclasses must implement train_epoch()")

    @abstractmethod
    def validate_epoch(self) -> Optional[Dict[str, float]]:
        """Execute one validation epoch.

        This method must be implemented by all subclasses. It should:
        1. Set model to evaluation mode
        2. Iterate through validation data
        3. Compute validation metrics
        4. Return metrics for the epoch

        Returns:
            Dictionary of metric names to values for the epoch, or None if
            validation is not applicable for this experiment.

        Example:
            >>> def validate_epoch(self) -> Optional[Dict[str, float]]:
            ...     val_loader = self.dataloader.get_val_loader()
            ...     if val_loader is None:
            ...         return None
            ...
            ...     self.model.eval()
            ...     total_loss = 0.0
            ...
            ...     with torch.no_grad():
            ...         for data, target in val_loader:
            ...             predictions = self.model(data)
            ...             loss = self.model.compute_loss(predictions, target)
            ...             total_loss += loss.item()
            ...
            ...     return {'val_loss': total_loss / len(val_loader)}
        """
        raise NotImplementedError("Subclasses must implement validate_epoch()")

    @abstractmethod
    def get_model(self) -> BaseModel:
        """Return the model being trained.

        Returns:
            The model instance
        """
        raise NotImplementedError("Subclasses must implement get_model()")

    @abstractmethod
    def get_dataloader(self) -> BaseDataLoader:
        """Return the dataloader.

        Returns:
            The dataloader instance
        """
        raise NotImplementedError("Subclasses must implement get_dataloader()")

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer.

        Returns:
            The optimizer instance
        """
        raise NotImplementedError("Subclasses must implement get_optimizer()")

    @abstractmethod
    def get_logger(self) -> BaseLogger:
        """Return the logger.

        Returns:
            The logger instance
        """
        raise NotImplementedError("Subclasses must implement get_logger()")

    def train(
        self,
        num_epochs: int,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 1,
        validate_frequency: int = 1,
        save_best: bool = True,
        best_metric: str = "loss",
        best_metric_mode: str = "min",
    ) -> None:
        """Main training loop.

        This method orchestrates the training process:
        1. Runs training epochs
        2. Performs validation at specified frequency
        3. Saves checkpoints
        4. Tracks best model
        5. Logs metrics

        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints (None to disable)
            checkpoint_frequency: Save checkpoint every N epochs
            validate_frequency: Validate every N epochs (0 to disable)
            save_best: Whether to save the best model separately
            best_metric: Metric name to use for best model selection
            best_metric_mode: 'min' or 'max' for best metric comparison

        Example:
            >>> trainer.train(
            ...     num_epochs=10,
            ...     checkpoint_dir='outputs/checkpoints',
            ...     checkpoint_frequency=5,
            ...     validate_frequency=1,
            ...     save_best=True,
            ...     best_metric='val_loss',
            ...     best_metric_mode='min'
            ... )
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric_name = best_metric
        metrics_logger = self.get_logger()

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.debug(f"Checkpoint directory: {checkpoint_dir}")
        logger.debug(f"Validate frequency: {validate_frequency}")
        logger.debug(f"Checkpoint frequency: {checkpoint_frequency}")

        for epoch in range(num_epochs):
            self._current_epoch = epoch + 1
            epoch_start_time = time.time()
            logger.info(f"Epoch {self._current_epoch}/{num_epochs} started")

            # Training epoch
            train_metrics = self.train_epoch()
            epoch_time = time.time() - epoch_start_time

            logger.info(
                f"Epoch {self._current_epoch} completed in {epoch_time:.1f}s - "
                f"Training metrics: {train_metrics}"
            )
            logger.debug(f"Epoch time: {epoch_time:.2f}s")

            # Log training metrics
            metrics_logger.log_metrics(
                train_metrics, step=self._global_step, epoch=self._current_epoch
            )

            # Validation
            val_metrics = None
            if validate_frequency > 0 and (epoch + 1) % validate_frequency == 0:
                logger.info(f"Running validation for epoch {self._current_epoch}")
                val_metrics = self.validate_epoch()
                if val_metrics is not None:
                    logger.info(f"Validation metrics: {val_metrics}")
                    metrics_logger.log_metrics(
                        val_metrics, step=self._global_step, epoch=self._current_epoch
                    )

            # Determine current metric value for best model tracking
            current_metric_value = None
            if save_best:
                # Try validation metrics first, then training metrics
                metrics_to_check = val_metrics if val_metrics else train_metrics
                current_metric_value = metrics_to_check.get(best_metric)

                if current_metric_value is not None:
                    is_best = self._is_best_metric(
                        current_metric_value, best_metric_mode
                    )

                    if is_best:
                        prev_best = self._best_metric
                        self._best_metric = current_metric_value
                        prev_best_str = (
                            f"{prev_best:.6f}" if prev_best is not None else "N/A"
                        )
                        logger.info(
                            f"New best {best_metric}: {current_metric_value:.6f} "
                            f"(previous: {prev_best_str})"
                        )
                        if checkpoint_dir is not None:
                            best_path = checkpoint_dir / "best_model.pth"
                            self.save_checkpoint(
                                best_path,
                                epoch=self._current_epoch,
                                is_best=True,
                                metrics={
                                    **train_metrics,
                                    **(val_metrics if val_metrics else {}),
                                },
                            )
                            logger.info(f"Best model checkpoint saved: {best_path}")

            # Regular checkpoint saving
            if checkpoint_dir is not None:
                if (epoch + 1) % checkpoint_frequency == 0:
                    checkpoint_path = (
                        checkpoint_dir / f"checkpoint_epoch_{self._current_epoch}.pth"
                    )
                    self.save_checkpoint(
                        checkpoint_path,
                        epoch=self._current_epoch,
                        is_best=False,
                        metrics={
                            **train_metrics,
                            **(val_metrics if val_metrics else {}),
                        },
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path}")

                # Always save latest checkpoint
                latest_path = checkpoint_dir / "latest_checkpoint.pth"
                self.save_checkpoint(
                    latest_path,
                    epoch=self._current_epoch,
                    is_best=False,
                    metrics={**train_metrics, **(val_metrics if val_metrics else {})},
                )
                logger.debug(f"Latest checkpoint updated: {latest_path}")

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Save training checkpoint.

        Saves the complete training state including model, optimizer,
        epoch counter, and metrics. This allows resuming training later.

        Args:
            path: Path to save checkpoint file
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            metrics: Dictionary of current metrics
            **kwargs: Additional metadata to save

        Example:
            >>> trainer.save_checkpoint(
            ...     'checkpoints/epoch_10.pth',
            ...     epoch=10,
            ...     is_best=False,
            ...     metrics={'loss': 0.5, 'accuracy': 0.95}
            ... )
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model = self.get_model()
        optimizer = self.get_optimizer()

        checkpoint = {
            "epoch": epoch,
            "global_step": self._global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "is_best": is_best,
            "trainer_class": self.__class__.__name__,
        }

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if self._best_metric is not None:
            checkpoint["best_metric"] = self._best_metric
            checkpoint["best_metric_name"] = self._best_metric_name

        # Add any additional metadata
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)

        # Enhanced logging
        if is_best:
            logger.info(f"✓ Best model checkpoint saved: {path}")
        else:
            logger.info(f"✓ Checkpoint saved: {path}")

        logger.info(f"  Epoch: {epoch}, Global step: {self._global_step}")

        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            logger.info(f"  Metrics: {metrics_str}")

        if self._best_metric is not None:
            logger.info(f"  Best {self._best_metric_name}: {self._best_metric:.6f}")

        logger.debug(f"  Checkpoint keys: {list(checkpoint.keys())}")
        logger.debug(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load training checkpoint.

        Loads the training state from a checkpoint file, restoring model,
        optimizer, and training progress.

        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            strict: Whether to strictly enforce state dict keys match

        Returns:
            Dictionary containing checkpoint metadata (epoch, metrics, etc.)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist

        Example:
            >>> checkpoint_info = trainer.load_checkpoint('checkpoints/best_model.pth')
            >>> print(f"Loaded model from epoch {checkpoint_info['epoch']}")
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
        model = self.get_model()
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        except Exception as e:
            logger.error(f"Failed to load model state dict")
            logger.exception(f"Error details: {e}")
            if strict:
                raise
            else:
                logger.warning("Continuing with non-strict loading")

        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer = self.get_optimizer()
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.error(f"Failed to load optimizer state dict")
                logger.exception(f"Error details: {e}")
                logger.warning("Continuing without optimizer state")

        # Restore training progress
        self._current_epoch = checkpoint.get("epoch", 0)
        self._global_step = checkpoint.get("global_step", 0)
        self._best_metric = checkpoint.get("best_metric", None)
        self._best_metric_name = checkpoint.get("best_metric_name", None)

        logger.info(f"✓ Checkpoint loaded successfully")
        logger.info(f"  Epoch: {self._current_epoch}, Global step: {self._global_step}")

        if "metrics" in checkpoint:
            metrics_str = ", ".join(
                [f"{k}: {v:.6f}" for k, v in checkpoint["metrics"].items()]
            )
            logger.info(f"  Loaded metrics: {metrics_str}")

        if self._best_metric is not None:
            logger.info(f"  Best {self._best_metric_name}: {self._best_metric:.6f}")

        return checkpoint

    def resume_training(
        self,
        checkpoint_path: Union[str, Path],
        num_epochs: int,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 1,
        validate_frequency: int = 1,
        save_best: bool = True,
        best_metric: str = "loss",
        best_metric_mode: str = "min",
    ) -> None:
        """Resume training from a checkpoint.

        Loads the checkpoint and continues training for the specified
        number of additional epochs.

        Args:
            checkpoint_path: Path to checkpoint file to resume from
            num_epochs: Number of additional epochs to train
            checkpoint_dir: Directory to save new checkpoints
            checkpoint_frequency: Save checkpoint every N epochs
            validate_frequency: Validate every N epochs
            save_best: Whether to save the best model separately
            best_metric: Metric name to use for best model selection
            best_metric_mode: 'min' or 'max' for best metric comparison

        Example:
            >>> trainer.resume_training(
            ...     checkpoint_path='checkpoints/latest_checkpoint.pth',
            ...     num_epochs=10,
            ...     checkpoint_dir='outputs/checkpoints'
            ... )
        """
        checkpoint_info = self.load_checkpoint(checkpoint_path)
        start_epoch = checkpoint_info["epoch"]
        logger.info(f"Resuming training from epoch {start_epoch}")
        logger.info(f"Will train for {num_epochs} additional epochs")

        # Adjust for resume: we'll train for num_epochs total epochs starting from checkpoint
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric_name = best_metric
        metrics_logger = self.get_logger()

        for epoch in range(num_epochs):
            self._current_epoch = start_epoch + epoch + 1
            epoch_start_time = time.time()
            logger.info(f"Epoch {self._current_epoch} started (resumed)")

            # Training epoch
            train_metrics = self.train_epoch()
            epoch_time = time.time() - epoch_start_time

            logger.info(
                f"Epoch {self._current_epoch} completed in {epoch_time:.1f}s - "
                f"Training metrics: {train_metrics}"
            )
            logger.debug(f"Epoch time: {epoch_time:.2f}s")

            # Log training metrics
            metrics_logger.log_metrics(
                train_metrics, step=self._global_step, epoch=self._current_epoch
            )

            # Validation
            val_metrics = None
            if validate_frequency > 0 and (epoch + 1) % validate_frequency == 0:
                logger.info(f"Running validation for epoch {self._current_epoch}")
                val_metrics = self.validate_epoch()
                if val_metrics is not None:
                    logger.info(f"Validation metrics: {val_metrics}")
                    metrics_logger.log_metrics(
                        val_metrics, step=self._global_step, epoch=self._current_epoch
                    )

            # Determine current metric value for best model tracking
            current_metric_value = None
            if save_best:
                # Try validation metrics first, then training metrics
                metrics_to_check = val_metrics if val_metrics else train_metrics
                current_metric_value = metrics_to_check.get(best_metric)

                if current_metric_value is not None:
                    is_best = self._is_best_metric(
                        current_metric_value, best_metric_mode
                    )

                    if is_best:
                        prev_best = self._best_metric
                        self._best_metric = current_metric_value
                        prev_best_str = (
                            f"{prev_best:.6f}" if prev_best is not None else "N/A"
                        )
                        logger.info(
                            f"New best {best_metric}: {current_metric_value:.6f} "
                            f"(previous: {prev_best_str})"
                        )
                        if checkpoint_dir is not None:
                            best_path = checkpoint_dir / "best_model.pth"
                            self.save_checkpoint(
                                best_path,
                                epoch=self._current_epoch,
                                is_best=True,
                                metrics={
                                    **train_metrics,
                                    **(val_metrics if val_metrics else {}),
                                },
                            )
                            logger.info(f"Best model checkpoint saved: {best_path}")

            # Regular checkpoint saving
            if checkpoint_dir is not None:
                if (epoch + 1) % checkpoint_frequency == 0:
                    checkpoint_path_new = (
                        checkpoint_dir / f"checkpoint_epoch_{self._current_epoch}.pth"
                    )
                    self.save_checkpoint(
                        checkpoint_path_new,
                        epoch=self._current_epoch,
                        is_best=False,
                        metrics={
                            **train_metrics,
                            **(val_metrics if val_metrics else {}),
                        },
                    )
                    logger.info(f"Checkpoint saved: {checkpoint_path_new}")

                # Always save latest checkpoint
                latest_path = checkpoint_dir / "latest_checkpoint.pth"
                self.save_checkpoint(
                    latest_path,
                    epoch=self._current_epoch,
                    is_best=False,
                    metrics={**train_metrics, **(val_metrics if val_metrics else {})},
                )
                logger.debug(f"Latest checkpoint updated: {latest_path}")

    def _is_best_metric(self, current_value: float, mode: str) -> bool:
        """Check if current metric value is the best so far.

        Args:
            current_value: Current metric value
            mode: 'min' or 'max' for comparison

        Returns:
            True if current value is better than best so far
        """
        if self._best_metric is None:
            return True

        if mode == "min":
            return current_value < self._best_metric
        elif mode == "max":
            return current_value > self._best_metric
        else:
            raise ValueError(f"Invalid metric mode: {mode}. Must be 'min' or 'max'")

    @property
    def current_epoch(self) -> int:
        """Get the current training epoch."""
        return self._current_epoch

    @property
    def global_step(self) -> int:
        """Get the global training step counter."""
        return self._global_step

    @property
    def best_metric(self) -> Optional[float]:
        """Get the best metric value achieved so far."""
        return self._best_metric
