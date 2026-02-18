"""Classifier Trainer

This module implements the trainer for classification experiments.
It inherits from BaseTrainer and provides classification-specific
training and validation logic.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.base.dataloader import BaseDataLoader
from src.base.logger import BaseLogger
from src.base.model import BaseModel
from src.base.trainer import BaseTrainer

# Module-level logger for application logging
_logger = logging.getLogger(__name__)


class ClassifierTrainer(BaseTrainer):
    """Trainer for classification experiments.

    This trainer implements the training and validation loops for classification
    tasks. It inherits from BaseTrainer and provides classification-specific
    logic including:
    - Training epoch with loss computation and backpropagation
    - Validation epoch with accuracy metrics
    - Support for multi-class and binary classification
    - Optional progress bars for training visualization
    - Automatic metric computation (accuracy, loss, etc.)

    The trainer handles the complete training lifecycle:
    1. Training loop with gradient updates
    2. Validation loop with metric computation
    3. Checkpointing and logging (handled by base class)

    Args:
        model: The classification model to train
        dataloader: DataLoader providing training and validation data
        optimizer: Optimizer for updating model parameters
        logger: Logger for recording metrics and checkpoints
        device: Device to run training on ('cpu' or 'cuda')
        show_progress: Whether to show progress bars during training

    Example:
        >>> model = InceptionV3Classifier(num_classes=2)
        >>> dataloader = ClassifierDataLoader(train_path="data/train")
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> logger = ClassifierLogger(log_dir="outputs/logs")
        >>> trainer = ClassifierTrainer(model, dataloader, optimizer, logger)
        >>> trainer.train(num_epochs=10, checkpoint_dir="outputs/checkpoints")
    """

    def __init__(
        self,
        model: BaseModel,
        dataloader: BaseDataLoader,
        optimizer: torch.optim.Optimizer,
        logger: BaseLogger,
        device: str = "cpu",
        show_progress: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_interval: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the classifier trainer.

        Args:
            model: The classification model to train
            dataloader: DataLoader providing training and validation data
            optimizer: Optimizer for updating model parameters
            logger: Logger for recording metrics and checkpoints
            device: Device to run training on ('cpu' or 'cuda')
            show_progress: Whether to show progress bars during training
            scheduler: Optional learning rate scheduler
            log_interval: Log batch-level metrics every N batches (0 to disable)
            config: Full experiment configuration dictionary used for hyperparameter
                logging and model graph visualization
        """
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.show_progress = show_progress
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.config = config or {}

        # Move model to device
        self.model.to(self.device)

        # Debug logging: Model structure
        _logger.debug(f"Model: {self.model.__class__.__name__}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        _logger.debug(f"Total parameters: {total_params:,}")
        _logger.debug(f"Trainable parameters: {trainable_params:,}")
        _logger.debug(f"Device: {self.device}")

        # Optional model graph logging to TensorBoard
        tb_config = (
            self.config.get("logging", {}).get("metrics", {}).get("tensorboard", {})
        )
        if (
            hasattr(self.logger, "tb_writer")
            and self.logger.tb_writer is not None
            and tb_config.get("log_graph", False)
        ):
            try:
                crop_size = (
                    self.config.get("data", {})
                    .get("preprocessing", {})
                    .get("crop_size", 224)
                )
                dummy_input = torch.randn(1, 3, crop_size, crop_size).to(self.device)
                self.logger.tb_writer.add_graph(self.model, dummy_input)
                _logger.info("Logged model graph to TensorBoard")
            except Exception as e:
                _logger.warning(f"Failed to log model graph to TensorBoard: {e}")

    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch.

        Performs a complete pass through the training data, computing loss,
        performing backpropagation, and updating model parameters.

        Returns:
            Dictionary containing training metrics:
            - 'loss': Average loss over the epoch
            - 'accuracy': Classification accuracy over the epoch

        Example:
            >>> metrics = trainer.train_epoch()
            >>> print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
        """
        self.model.train()
        train_loader = self.dataloader.get_train_loader()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Debug: Log dataset info
        _logger.debug(f"Training on {len(train_loader)} batches")
        if hasattr(train_loader, "dataset"):
            _logger.debug(f"Dataset size: {len(train_loader.dataset)}")

        # Create progress bar if enabled
        iterator = (
            tqdm(train_loader, desc=f"Epoch {self._current_epoch} [Train]")
            if self.show_progress
            else train_loader
        )

        for batch_idx, (data, target) in enumerate(iterator):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Debug: Log batch shapes on first batch
            if batch_idx == 0:
                _logger.debug(f"Input batch shape: {data.shape}")
                _logger.debug(f"Target batch shape: {target.shape}")
                if torch.cuda.is_available() and self.device == "cuda":
                    _logger.debug(
                        f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                    )

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(data)

            # Compute loss
            loss = self.model.compute_loss(predictions, target)

            # Backward pass
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self._global_step += 1

            # Compute accuracy
            _, predicted = torch.max(predictions.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Batch-level logging
            if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                _logger.debug(
                    f"Epoch [{self._current_epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] - "
                    f"Loss: {loss.item():.4f}, Acc: {100.0 * correct / total:.2f}%, LR: {current_lr:.6f}"
                )

            # Update progress bar
            if self.show_progress:
                iterator.set_postfix(
                    {
                        "loss": total_loss / num_batches,
                        "acc": 100.0 * correct / total,
                    }
                )

        # Compute epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # Warning for unusual conditions
        if avg_loss > 10.0:
            _logger.warning(
                f"High loss detected: {avg_loss:.4f} - training may be unstable"
            )
        if accuracy < 10.0 and self._current_epoch > 5:
            _logger.warning(
                f"Low accuracy detected: {accuracy:.2f}% after {self._current_epoch} epochs"
            )

        _logger.info(
            f"Epoch {self._current_epoch} [Train] - "
            f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        _logger.debug(f"Training batches: {num_batches}, Total samples: {total}")

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def train(
        self,
        num_epochs: int,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_frequency: int = 1,
        validate_frequency: int = 1,
        save_best: bool = True,
        best_metric: str = "loss",
        best_metric_mode: str = "min",
        save_latest_checkpoint: bool = True,
    ) -> None:
        """Main training loop with scheduler support.

        Overrides base trainer's train() to add learning rate scheduler stepping.

        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints (None to disable)
            checkpoint_frequency: Save checkpoint every N epochs
            validate_frequency: Validate every N epochs (0 to disable)
            save_best: Whether to save the best model separately
            best_metric: Metric name to use for best model selection
            best_metric_mode: 'min' or 'max' for best metric comparison
            save_latest_checkpoint: If True, writes latest_checkpoint.pth after every epoch.
                                     If False, only periodic and best checkpoints are written.
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric_name = best_metric
        logger = self.get_logger()

        # Log hyperparameters to TensorBoard at start of training
        if self.config:
            logger.log_hyperparams(self.config)

        for epoch in range(num_epochs):
            self._current_epoch = epoch + 1

            # Training epoch
            train_metrics = self.train_epoch()

            # Step the scheduler if provided
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]["lr"]

                # Warning for very low learning rate
                if new_lr < 1e-7:
                    _logger.warning(
                        f"Very low learning rate: {new_lr:.2e} - training may be ineffective"
                    )

                if old_lr != new_lr:
                    _logger.info(
                        f"Learning rate changed: {old_lr:.6e} -> {new_lr:.6e} "
                        f"(epoch {self._current_epoch})"
                    )
                else:
                    _logger.debug(f"Learning rate: {new_lr:.6e}")

            # Log training metrics
            logger.log_metrics(
                train_metrics, step=self._global_step, epoch=self._current_epoch
            )

            # Validation
            val_metrics = None
            if validate_frequency > 0 and (epoch + 1) % validate_frequency == 0:
                val_metrics = self.validate_epoch()
                if val_metrics is not None:
                    logger.log_metrics(
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
                        self._best_metric = current_metric_value
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

                if save_latest_checkpoint:
                    latest_path = checkpoint_dir / "latest_checkpoint.pth"
                    self.save_checkpoint(
                        latest_path,
                        epoch=self._current_epoch,
                        is_best=False,
                        metrics={
                            **train_metrics,
                            **(val_metrics if val_metrics else {}),
                        },
                    )

    def validate_epoch(self) -> Optional[Dict[str, float]]:
        """Execute one validation epoch.

        Evaluates the model on the validation dataset without updating parameters.
        Computes validation metrics including loss and accuracy.

        Returns:
            Dictionary containing validation metrics, or None if no validation data:
            - 'val_loss': Average validation loss
            - 'val_accuracy': Validation accuracy

        Example:
            >>> metrics = trainer.validate_epoch()
            >>> if metrics:
            ...     print(f"Val Loss: {metrics['val_loss']:.4f}")
        """
        val_loader = self.dataloader.get_val_loader()
        if val_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Create progress bar if enabled
        iterator = (
            tqdm(val_loader, desc=f"Epoch {self._current_epoch} [Val]")
            if self.show_progress
            else val_loader
        )

        with torch.no_grad():
            for data, target in iterator:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                predictions = self.model(data)

                # Compute loss
                loss = self.model.compute_loss(predictions, target)

                # Track metrics
                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy
                _, predicted = torch.max(predictions.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # Update progress bar
                if self.show_progress:
                    iterator.set_postfix(
                        {
                            "loss": total_loss / num_batches,
                            "acc": 100.0 * correct / total,
                        }
                    )

        # Compute epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        _logger.info(
            f"Epoch {self._current_epoch} [Val] - "
            f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        _logger.debug(f"Validation batches: {num_batches}, Total samples: {total}")

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }

    def get_model(self) -> BaseModel:
        """Return the classification model.

        Returns:
            The model being trained
        """
        return self.model

    def get_dataloader(self) -> BaseDataLoader:
        """Return the dataloader.

        Returns:
            The dataloader providing training/validation data
        """
        return self.dataloader

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer.

        Returns:
            The optimizer used for parameter updates
        """
        return self.optimizer

    def get_logger(self) -> BaseLogger:
        """Return the logger.

        Returns:
            The logger for recording metrics
        """
        return self.logger

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on a specific dataloader.

        This method allows evaluating the model on any dataloader,
        not just the validation set. Useful for test set evaluation.

        Args:
            dataloader: DataLoader to evaluate on. If None, uses validation loader.

        Returns:
            Dictionary containing evaluation metrics:
            - 'loss': Average loss
            - 'accuracy': Classification accuracy

        Example:
            >>> test_loader = DataLoader(test_dataset, batch_size=32)
            >>> metrics = trainer.evaluate(test_loader)
            >>> print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        """
        if dataloader is None:
            dataloader = self.dataloader.get_val_loader()

        if dataloader is None:
            return {"loss": 0.0, "accuracy": 0.0}

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for data, target in dataloader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                predictions = self.model(data)

                # Compute loss
                loss = self.model.compute_loss(predictions, target)

                # Track metrics
                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy
                _, predicted = torch.max(predictions.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # Compute metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
