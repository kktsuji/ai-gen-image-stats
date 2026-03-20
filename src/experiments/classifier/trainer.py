"""Classifier Trainer

This module implements the trainer for classification experiments.
It inherits from BaseTrainer and provides classification-specific
training and validation logic.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.checkpoint import is_best_metric, load_checkpoint, save_checkpoint

# Module-level logger for application logging
_logger = logging.getLogger(__name__)


class ClassifierTrainer:
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
        >>> logger = ExperimentLogger(log_dir="outputs/logs")
        >>> trainer = ClassifierTrainer(model, dataloader, optimizer, logger)
        >>> trainer.train(num_epochs=10, checkpoint_dir="outputs/checkpoints")
    """

    def __init__(
        self,
        model: Any,
        train_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Any = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        show_progress: bool = True,
        scheduler: Optional[
            Union[
                torch.optim.lr_scheduler.LRScheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ]
        ] = None,
        log_interval: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the classifier trainer.

        Args:
            model: The classification model to train
            train_loader: DataLoader for training data (required for training)
            optimizer: Optimizer for updating model parameters (required for training)
            logger: Logger for recording metrics and checkpoints
            val_loader: Optional DataLoader for validation data
            device: Device to run training on ('cpu' or 'cuda')
            show_progress: Whether to show progress bars during training
            scheduler: Optional learning rate scheduler
            log_interval: Log batch-level metrics every N batches (0 to disable)
            config: Full experiment configuration dictionary used for hyperparameter
                logging and model graph visualization
        """
        self._current_epoch = 0
        self._global_step = 0
        self._best_metric: Optional[float] = None
        self._best_metric_name: Optional[str] = None

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        _tb_writer = getattr(self.logger, "tb_writer", None)
        if _tb_writer is not None and tb_config.get("log_graph", False):
            try:
                crop_size = (
                    self.config.get("data", {})
                    .get("preprocessing", {})
                    .get("crop_size", 224)
                )
                dummy_input = torch.randn(1, 3, crop_size, crop_size).to(self.device)
                _tb_writer.add_graph(self.model, dummy_input)
                _logger.info("Logged model graph to TensorBoard")
            except Exception as e:
                _logger.warning(f"Failed to log model graph to TensorBoard: {e}")

    @classmethod
    def for_evaluation(
        cls,
        model: Any,
        val_loader: DataLoader,
        device: str = "cpu",
        show_progress: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> "ClassifierTrainer":
        """Create a trainer configured for evaluation only (no training components).

        Args:
            model: The classification model to evaluate.
            val_loader: DataLoader for evaluation data.
            device: Device to run evaluation on ('cpu' or 'cuda').
            show_progress: Whether to show progress bars.
            config: Full experiment configuration dictionary.

        Returns:
            ClassifierTrainer instance without train_loader or optimizer.
        """
        return cls(
            model=model,
            val_loader=val_loader,
            device=device,
            show_progress=show_progress,
            config=config,
        )

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
        if self.train_loader is None:
            raise RuntimeError("train_loader is required for training")
        if self.optimizer is None:
            raise RuntimeError("optimizer is required for training")

        self.model.train()
        train_loader = self.train_loader

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Debug: Log dataset info
        _logger.debug(f"Training on {len(train_loader)} batches")
        if hasattr(train_loader, "dataset"):
            _logger.debug(f"Dataset size: {len(train_loader.dataset)}")  # type: ignore[arg-type]

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
            loss = cast(torch.Tensor, self.model.compute_loss(predictions, target))

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
                iterator.set_postfix(  # type: ignore[attr-defined]
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
        if self.train_loader is None:
            raise RuntimeError("train_loader is required for training")
        if self.optimizer is None:
            raise RuntimeError("optimizer is required for training")
        optimizer = self.optimizer

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric_name = best_metric
        logger = self.get_logger()

        # Log hyperparameters to TensorBoard at start of training
        if self.config:
            logger.log_hyperparams(self.config)

        # Initialize metrics before loop so they're defined even when num_epochs=0
        train_metrics: Dict[str, float] = {}
        val_metrics: Optional[Dict[str, float]] = None

        for epoch in range(num_epochs):
            self._current_epoch = epoch + 1

            # Training epoch
            train_metrics = self.train_epoch()

            # Step the scheduler if provided
            if self.scheduler is not None:
                old_lr = optimizer.param_groups[0]["lr"]
                self.scheduler.step()  # type: ignore[call-arg]
                new_lr = optimizer.param_groups[0]["lr"]

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
                    is_best = is_best_metric(
                        current_metric_value, self._best_metric, best_metric_mode
                    )

                    if is_best:
                        self._best_metric = current_metric_value
                        if checkpoint_dir is not None:
                            best_path = checkpoint_dir / "best_model.pth"
                            save_checkpoint(
                                best_path,
                                model=self.model,
                                optimizer=optimizer,
                                epoch=self._current_epoch,
                                global_step=self._global_step,
                                is_best=True,
                                metrics={
                                    **train_metrics,
                                    **(val_metrics if val_metrics else {}),
                                },
                                best_metric=self._best_metric,
                                best_metric_name=self._best_metric_name,
                                trainer_class=self.__class__.__name__,
                            )

            # Regular checkpoint saving
            if checkpoint_dir is not None:
                if (epoch + 1) % checkpoint_frequency == 0:
                    checkpoint_path = (
                        checkpoint_dir / f"checkpoint_epoch_{self._current_epoch}.pth"
                    )
                    save_checkpoint(
                        checkpoint_path,
                        model=self.model,
                        optimizer=optimizer,
                        epoch=self._current_epoch,
                        global_step=self._global_step,
                        is_best=False,
                        metrics={
                            **train_metrics,
                            **(val_metrics if val_metrics else {}),
                        },
                        best_metric=self._best_metric,
                        best_metric_name=self._best_metric_name,
                        trainer_class=self.__class__.__name__,
                    )

                if save_latest_checkpoint:
                    latest_path = checkpoint_dir / "latest_checkpoint.pth"
                    save_checkpoint(
                        latest_path,
                        model=self.model,
                        optimizer=optimizer,
                        epoch=self._current_epoch,
                        global_step=self._global_step,
                        is_best=False,
                        metrics={
                            **train_metrics,
                            **(val_metrics if val_metrics else {}),
                        },
                        best_metric=self._best_metric,
                        best_metric_name=self._best_metric_name,
                        trainer_class=self.__class__.__name__,
                    )

        # Save final checkpoint after all epochs complete
        if checkpoint_dir is not None:
            final_path = checkpoint_dir / "final_model.pth"
            save_checkpoint(
                final_path,
                model=self.model,
                optimizer=optimizer,
                epoch=self._current_epoch,
                global_step=self._global_step,
                is_best=False,
                metrics={
                    **train_metrics,
                    **(val_metrics if val_metrics else {}),
                },
                best_metric=self._best_metric,
                best_metric_name=self._best_metric_name,
                trainer_class=self.__class__.__name__,
            )
            _logger.info(f"Final model checkpoint saved: {final_path}")

    def _compute_classification_metrics(
        self,
        all_targets: List[int],
        all_predictions: List[int],
        all_probs: np.ndarray,
        num_classes: int,
        prefix: str = "val_",
    ) -> Dict[str, float]:
        """Compute per-class classification metrics.

        Args:
            all_targets: List of ground truth labels.
            all_predictions: List of predicted labels.
            all_probs: Array of predicted probabilities, shape (N, num_classes).
            num_classes: Number of classes.
            prefix: Metric key prefix (e.g., 'val_' or '').

        Returns:
            Dictionary of classification metrics.
        """
        targets_arr = np.array(all_targets)
        preds_arr = np.array(all_predictions)

        metrics: Dict[str, float] = {}

        # Balanced accuracy
        metrics[f"{prefix}balanced_accuracy"] = float(
            balanced_accuracy_score(targets_arr, preds_arr)
        )

        # Per-class precision, recall, F1
        prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
            targets_arr,
            preds_arr,
            labels=list(range(num_classes)),
            zero_division=0.0,  # type: ignore[arg-type]
        )
        for cls_idx in range(num_classes):
            metrics[f"{prefix}precision_{cls_idx}"] = float(prec_arr[cls_idx])  # type: ignore[index]
            metrics[f"{prefix}recall_{cls_idx}"] = float(rec_arr[cls_idx])  # type: ignore[index]
            metrics[f"{prefix}f1_{cls_idx}"] = float(f1_arr[cls_idx])  # type: ignore[index]

        # ROC-AUC and PR-AUC (require at least 2 classes present in targets)
        unique_classes = np.unique(targets_arr)
        if len(unique_classes) >= 2:
            if num_classes == 2:
                metrics[f"{prefix}roc_auc"] = float(
                    roc_auc_score(targets_arr, all_probs[:, 1])
                )
                metrics[f"{prefix}pr_auc"] = float(
                    average_precision_score(targets_arr, all_probs[:, 1])
                )
            elif len(unique_classes) == num_classes:
                metrics[f"{prefix}roc_auc"] = float(
                    roc_auc_score(
                        targets_arr, all_probs, multi_class="ovr", average="weighted"
                    )
                )
                metrics[f"{prefix}pr_auc"] = float(
                    average_precision_score(targets_arr, all_probs, average="weighted")
                )
            else:
                _logger.warning(
                    "Skipping multiclass AUC metrics: "
                    "not all classes are present in targets"
                )
        else:
            _logger.warning("Skipping AUC metrics: only one class in targets")

        # Confusion matrix (flattened as cm_i_j)
        cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(num_classes)))
        for i in range(num_classes):
            for j in range(num_classes):
                metrics[f"{prefix}cm_{i}_{j}"] = float(cm[i, j])

        return metrics

    def _run_inference(
        self,
        dataloader: DataLoader,
        desc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run inference on a dataloader and collect predictions.

        Shared implementation for validate_epoch and evaluate.
        Assumes the model returns raw logits (pre-softmax), as used with
        CrossEntropyLoss. Softmax is applied here to obtain probabilities
        for AUC and per-class metrics.

        Args:
            dataloader: DataLoader to run inference on.
            desc: Description for the progress bar (None to use default).

        Returns:
            Dictionary with keys: avg_loss, accuracy, total, num_batches,
            all_targets, all_predictions, all_probs.
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        all_targets: List[int] = []
        all_predictions: List[int] = []
        all_probs_list: List[np.ndarray] = []

        # Create progress bar if enabled
        iterator = (
            tqdm(dataloader, desc=desc or "Inference")
            if self.show_progress
            else dataloader
        )

        with torch.no_grad():
            for data, target in iterator:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                predictions = self.model(data)

                # Compute loss
                loss = cast(torch.Tensor, self.model.compute_loss(predictions, target))

                # Track metrics
                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy
                _, predicted = torch.max(predictions.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # Accumulate for per-class metrics
                all_targets.extend(target.cpu().tolist())
                all_predictions.extend(predicted.cpu().tolist())
                # Model must return raw logits (pre-softmax).
                # Guard: if outputs look like probabilities already, warn.
                if num_batches == 1:
                    row_sums = predictions.sum(dim=1)
                    all_non_negative = predictions.min().item() >= 0
                    all_sum_to_one = (row_sums - 1.0).abs().max().item() < 1e-3
                    if all_non_negative and all_sum_to_one:
                        _logger.warning(
                            "Model outputs look like probabilities (non-negative, "
                            "sum≈1). Expected raw logits. AUC metrics may be wrong."
                        )
                probs = torch.softmax(predictions, dim=1).cpu().numpy()
                all_probs_list.append(probs)

                # Update progress bar
                if self.show_progress:
                    iterator.set_postfix(  # type: ignore[attr-defined]
                        {
                            "loss": total_loss / num_batches,
                            "acc": 100.0 * correct / total,
                        }
                    )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        all_probs = (
            np.concatenate(all_probs_list, axis=0) if all_probs_list else np.array([])
        )

        return {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "total": total,
            "num_batches": num_batches,
            "all_targets": all_targets,
            "all_predictions": all_predictions,
            "all_probs": all_probs,
        }

    def validate_epoch(self) -> Optional[Dict[str, float]]:
        """Execute one validation epoch.

        Evaluates the model on the validation dataset without updating parameters.
        Computes validation metrics including loss, accuracy, and per-class metrics.

        Returns:
            Dictionary containing validation metrics, or None if no validation data:
            - 'val_loss': Average validation loss
            - 'val_accuracy': Validation accuracy
            - 'val_balanced_accuracy': Balanced accuracy
            - 'val_precision_N', 'val_recall_N', 'val_f1_N': Per-class metrics
            - 'val_roc_auc', 'val_pr_auc': AUC metrics (when computable)
            - 'val_cm_I_J': Confusion matrix entries

        Example:
            >>> metrics = trainer.validate_epoch()
            >>> if metrics:
            ...     print(f"Val Loss: {metrics['val_loss']:.4f}")
        """
        if self.val_loader is None:
            return None

        inference = self._run_inference(
            self.val_loader,
            desc=f"Epoch {self._current_epoch} [Val]",
        )

        avg_loss = inference["avg_loss"]
        accuracy = inference["accuracy"]
        total = inference["total"]

        _logger.info(
            f"Epoch {self._current_epoch} [Val] - "
            f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        _logger.debug(
            f"Validation batches: {inference['num_batches']}, Total samples: {total}"
        )

        result: Dict[str, float] = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }

        # Compute per-class metrics
        if total > 0:
            all_probs = inference["all_probs"]
            num_classes = all_probs.shape[1]
            classification_metrics = self._compute_classification_metrics(
                inference["all_targets"],
                inference["all_predictions"],
                all_probs,
                num_classes,
                prefix="val_",
            )
            result.update(classification_metrics)

            _logger.info(
                f"Epoch {self._current_epoch} [Val] - "
                f"Balanced Acc: {result.get('val_balanced_accuracy', 0.0):.4f}, "
                f"ROC-AUC: {result.get('val_roc_auc', 0.0):.4f}"
            )

        return result

    def get_model(self) -> Any:
        """Return the classification model."""
        return self.model

    def get_train_loader(self) -> Optional[DataLoader]:
        """Return the training DataLoader."""
        return self.train_loader

    def get_val_loader(self) -> Optional[DataLoader]:
        """Return the validation DataLoader."""
        return self.val_loader

    def get_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """Return the optimizer."""
        return self.optimizer

    def get_logger(self) -> Any:
        """Return the logger."""
        return self.logger

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Save training checkpoint."""
        if self.optimizer is None:
            raise RuntimeError("optimizer is required to save a checkpoint")
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self._global_step,
            is_best=is_best,
            metrics=metrics,
            best_metric=self._best_metric,
            best_metric_name=self._best_metric_name,
            trainer_class=self.__class__.__name__,
            **kwargs,
        )

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer if load_optimizer else None,
            strict=strict,
        )
        self._current_epoch = checkpoint.get("epoch", 0)
        self._global_step = checkpoint.get("global_step", 0)
        self._best_metric = checkpoint.get("best_metric", None)
        self._best_metric_name = checkpoint.get("best_metric_name", None)
        return checkpoint

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
            - Per-class precision, recall, F1, AUC metrics, confusion matrix

        Example:
            >>> test_loader = DataLoader(test_dataset, batch_size=32)
            >>> metrics = trainer.evaluate(test_loader)
            >>> print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        """
        if dataloader is None:
            dataloader = self.val_loader

        if dataloader is None:
            return {"loss": 0.0, "accuracy": 0.0}

        inference = self._run_inference(dataloader, desc="Evaluate")

        result: Dict[str, float] = {
            "loss": inference["avg_loss"],
            "accuracy": inference["accuracy"],
        }

        # Compute per-class metrics
        if inference["total"] > 0:
            all_probs = inference["all_probs"]
            num_classes = all_probs.shape[1]
            classification_metrics = self._compute_classification_metrics(
                inference["all_targets"],
                inference["all_predictions"],
                all_probs,
                num_classes,
                prefix="",
            )
            result.update(classification_metrics)

        return result
