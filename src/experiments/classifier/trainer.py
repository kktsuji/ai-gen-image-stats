"""Classifier Trainer

This module implements the trainer for classification experiments.
It inherits from BaseTrainer and provides classification-specific
training and validation logic.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.base.dataloader import BaseDataLoader
from src.base.logger import BaseLogger
from src.base.model import BaseModel
from src.base.trainer import BaseTrainer


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
    ):
        """Initialize the classifier trainer.

        Args:
            model: The classification model to train
            dataloader: DataLoader providing training and validation data
            optimizer: Optimizer for updating model parameters
            logger: Logger for recording metrics and checkpoints
            device: Device to run training on ('cpu' or 'cuda')
            show_progress: Whether to show progress bars during training
        """
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.show_progress = show_progress

        # Move model to device
        self.model.to(self.device)

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

        # Create progress bar if enabled
        iterator = (
            tqdm(train_loader, desc=f"Epoch {self._current_epoch} [Train]")
            if self.show_progress
            else train_loader
        )

        for batch_idx, (data, target) in enumerate(iterator):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

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

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

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
