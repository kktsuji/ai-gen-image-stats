"""Diffusion Trainer

This module implements the trainer for diffusion model experiments.
It inherits from BaseTrainer and provides diffusion-specific
training and generation logic.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from src.base.dataloader import BaseDataLoader
from src.base.logger import BaseLogger
from src.base.model import BaseModel
from src.base.trainer import BaseTrainer
from src.experiments.diffusion.model import EMA
from src.experiments.diffusion.sampler import DiffusionSampler

# Module-level logger
_logger = logging.getLogger(__name__)


class DiffusionTrainer(BaseTrainer):
    """Trainer for diffusion model experiments.

    This trainer implements the training and validation loops for diffusion
    models (DDPM). It inherits from BaseTrainer and provides diffusion-specific
    logic including:
    - Training epoch with diffusion loss computation
    - Validation epoch with loss tracking
    - Sample generation mode for inference
    - EMA (Exponential Moving Average) support for better sample quality
    - Automatic mixed precision (AMP) support for efficient training
    - Optional gradient clipping and explosion detection

    The trainer handles the complete training lifecycle:
    1. Training loop with noise prediction and denoising
    2. Validation loop with metric computation
    3. Sample generation with optional class conditioning
    4. Checkpointing with EMA and AMP state (handled by base class)
    5. Sample visualization during training

    Note:
        For inference-only workflows (generating samples from a trained model),
        use DiffusionSampler directly instead of creating a trainer. This avoids
        the overhead of initializing optimizer, dataloaders, and other training
        components. See DiffusionSampler documentation for examples.

        During training, samples can be accessed via `trainer.sampler.sample()`.

    Args:
        model: The diffusion model to train (DDPM)
        dataloader: DataLoader providing training and validation data
        optimizer: Optimizer for updating model parameters
        logger: Logger for recording metrics and generated samples
        device: Device to run training on ('cpu' or 'cuda')
        show_progress: Whether to show progress bars during training
        use_ema: Whether to use exponential moving average for sampling
        ema_decay: Decay rate for EMA (default: 0.9999)
        use_amp: Whether to use automatic mixed precision training
        gradient_clip_norm: Maximum gradient norm for clipping (None to disable)
        scheduler: Optional learning rate scheduler
        sample_images: Whether to generate sample images during training
        sample_interval: Generate samples every N epochs
        samples_per_class: Number of samples to generate per class
        guidance_scale: Classifier-free guidance scale for conditional generation

    Example:
        >>> from src.experiments.diffusion.model import create_ddpm
        >>> from src.experiments.diffusion.dataloader import DiffusionDataLoader
        >>> from src.experiments.diffusion.logger import DiffusionLogger
        >>>
        >>> model = create_ddpm(image_size=64, num_classes=2, device="cuda")
        >>> dataloader = DiffusionDataLoader(train_path="data/train", batch_size=128)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        >>> logger = DiffusionLogger(log_dir="outputs/logs/diffusion")
        >>>
        >>> trainer = DiffusionTrainer(
        ...     model=model,
        ...     dataloader=dataloader,
        ...     optimizer=optimizer,
        ...     logger=logger,
        ...     device="cuda",
        ...     use_ema=True,
        ...     use_amp=True,
        ...     sample_images=True,
        ...     sample_interval=10
        ... )
        >>>
        >>> trainer.train(num_epochs=200, checkpoint_dir="outputs/checkpoints")
    """

    def __init__(
        self,
        model: BaseModel,
        dataloader: BaseDataLoader,
        optimizer: torch.optim.Optimizer,
        logger: BaseLogger,
        device: str = "cpu",
        show_progress: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        use_amp: bool = False,
        gradient_clip_norm: Optional[float] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        sample_images: bool = True,
        sample_interval: int = 10,
        samples_per_class: int = 2,
        guidance_scale: float = 3.0,
        log_interval: int = 100,
    ):
        """Initialize the diffusion trainer.

        Args:
            model: The diffusion model to train
            dataloader: DataLoader providing training and validation data
            optimizer: Optimizer for updating model parameters
            logger: Logger for recording metrics and checkpoints
            device: Device to run training on ('cpu' or 'cuda')
            show_progress: Whether to show progress bars during training
            use_ema: Whether to use exponential moving average
            ema_decay: Decay rate for EMA
            use_amp: Whether to use automatic mixed precision
            gradient_clip_norm: Maximum gradient norm for clipping
            scheduler: Optional learning rate scheduler
            sample_images: Whether to generate samples during training
            sample_interval: Generate samples every N epochs
            samples_per_class: Number of samples per class to generate
            guidance_scale: Classifier-free guidance scale
            log_interval: Log batch-level metrics every N batches (0 to disable)
        """
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.logger = logger
        self.device = device
        self.show_progress = show_progress
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.log_interval = log_interval

        # Sample generation settings
        self.sample_images = sample_images
        self.sample_interval = sample_interval
        self.samples_per_class = samples_per_class
        self.guidance_scale = guidance_scale

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

        # Setup EMA
        self.use_ema = use_ema
        self.ema = None
        if use_ema:
            self.ema = EMA(model, decay=ema_decay, device=device)
            _logger.info(f"EMA enabled with decay={ema_decay}")

        # Setup automatic mixed precision
        self.use_amp = use_amp and device == "cuda"
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(device)
            _logger.info("Automatic mixed precision (AMP) enabled")

        # Loss criterion
        self.criterion = nn.MSELoss()

        # Initialize sampler for sample generation
        self.sampler = DiffusionSampler(
            model=self.model,
            device=self.device,
            ema=self.ema,
        )

    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch.

        Performs a complete pass through the training data, computing diffusion
        loss (noise prediction MSE), performing backpropagation, and updating
        model parameters. Optionally updates EMA parameters.

        Returns:
            Dictionary containing training metrics:
            - 'loss': Average diffusion loss over the epoch

        Example:
            >>> metrics = trainer.train_epoch()
            >>> print(f"Diffusion Loss: {metrics['loss']:.4f}")
        """
        self.model.train()
        train_loader = self.dataloader.get_train_loader()

        total_loss = 0.0
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

        for batch_idx, batch_data in enumerate(iterator):
            # Handle both conditional and unconditional data loading
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    # Conditional: (images, labels)
                    images, labels = batch_data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                elif len(batch_data) == 1:
                    # Unconditional: (images,)
                    images = batch_data[0].to(self.device)
                    labels = None
                else:
                    raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
                    _logger.critical(
                        f"Unexpected batch data format: length {len(batch_data)}"
                    )
                    raise ValueError(f"Unexpected batch data length: {len(batch_data)}")
            else:
                # Direct tensor (shouldn't happen with DataLoader, but handle it)
                images = batch_data.to(self.device)
                labels = None

            # Debug: Log batch shapes on first batch
            if batch_idx == 0:
                _logger.debug(f"Input batch shape: {images.shape}")
                if labels is not None:
                    _logger.debug(f"Labels batch shape: {labels.shape}")
                if torch.cuda.is_available() and self.device == "cuda":
                    _logger.debug(
                        f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                    )

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with automatic mixed precision
            if self.use_amp:
                with torch.amp.autocast(self.device):
                    loss = self.model.compute_loss(
                        images, class_labels=labels, criterion=self.criterion
                    )

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first for accurate clipping)
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                    if grad_norm > self.gradient_clip_norm:
                        _logger.warning(
                            f"Gradient clipped: norm {grad_norm:.4f} exceeded threshold {self.gradient_clip_norm}"
                        )
                    _logger.debug(f"Gradient norm: {grad_norm:.4f}")

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard float32 training
                loss = self.model.compute_loss(
                    images, class_labels=labels, criterion=self.criterion
                )

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                    if grad_norm > self.gradient_clip_norm:
                        _logger.warning(
                            f"Gradient clipped: norm {grad_norm:.4f} exceeded threshold {self.gradient_clip_norm}"
                        )
                    _logger.debug(f"Gradient norm: {grad_norm:.4f}")

                # Optimizer step
                self.optimizer.step()

            # Update EMA
            if self.use_ema and self.ema is not None:
                self.ema.update()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self._global_step += 1

            # Batch-level logging
            if self.log_interval > 0 and (batch_idx + 1) % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                _logger.debug(
                    f"Epoch [{self._current_epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] - "
                    f"Loss: {loss.item():.6f}, LR: {current_lr:.6e}"
                )

            # Update progress bar
            if self.show_progress:
                iterator.set_postfix({"loss": total_loss / num_batches})

        # Compute epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Warning for unusual conditions
        if avg_loss > 1.0:
            _logger.warning(
                f"High diffusion loss detected: {avg_loss:.6f} - training may be unstable"
            )

        _logger.info(f"Epoch {self._current_epoch} [Train] - Loss: {avg_loss:.6f}")
        _logger.debug(f"Training batches: {num_batches}")
        if self.use_ema:
            _logger.debug("EMA weights updated")

        return {"loss": avg_loss}

    def validate_epoch(self) -> Optional[Dict[str, float]]:
        """Execute one validation epoch.

        Performs a complete pass through the validation data, computing
        diffusion loss without gradient updates.

        Returns:
            Dictionary containing validation metrics:
            - 'val_loss': Average validation loss over the epoch
            Or None if validation data is not available

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
        num_batches = 0

        # Create progress bar if enabled
        iterator = (
            tqdm(val_loader, desc=f"Epoch {self._current_epoch} [Val]")
            if self.show_progress
            else val_loader
        )

        with torch.no_grad():
            for batch_data in iterator:
                # Handle both conditional and unconditional data loading
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 2:
                        # Conditional: (images, labels)
                        images, labels = batch_data
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                    elif len(batch_data) == 1:
                        # Unconditional: (images,)
                        images = batch_data[0].to(self.device)
                        labels = None
                    else:
                        raise ValueError(
                            f"Unexpected batch data length: {len(batch_data)}"
                        )
                else:
                    # Direct tensor (shouldn't happen with DataLoader, but handle it)
                    images = batch_data.to(self.device)
                    labels = None

                # Forward pass
                loss = self.model.compute_loss(
                    images, class_labels=labels, criterion=self.criterion
                )

                # Track metrics
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                if self.show_progress:
                    iterator.set_postfix({"val_loss": total_loss / num_batches})

        # Compute epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        _logger.info(f"Epoch {self._current_epoch} [Val] - Loss: {avg_loss:.6f}")
        _logger.debug(f"Validation batches: {num_batches}")

        return {"val_loss": avg_loss}

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
        """Main training loop with scheduler and sample generation support.

        Overrides base trainer's train() to add learning rate scheduler stepping
        and periodic sample generation for visual quality assessment.

        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints (None to disable)
            checkpoint_frequency: Save checkpoint every N epochs
            validate_frequency: Validate every N epochs (0 to disable)
            save_best: Whether to save the best model separately
            best_metric: Metric name to use for best model selection
            best_metric_mode: 'min' or 'max' for best metric comparison
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_metric_name = best_metric
        logger = self.get_logger()

        for epoch in range(num_epochs):
            self._current_epoch = epoch + 1

            # Training epoch
            train_metrics = self.train_epoch()

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

            # Learning rate scheduler step
            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

                # Warning for very low learning rate
                if current_lr < 1e-7:
                    _logger.warning(
                        f"Very low learning rate: {current_lr:.2e} - training may be ineffective"
                    )

                if old_lr != current_lr:
                    _logger.info(
                        f"Learning rate changed: {old_lr:.6e} -> {current_lr:.6e} "
                        f"(epoch {self._current_epoch})"
                    )
                else:
                    _logger.debug(f"Learning rate: {current_lr:.6e}")
                logger.log_metrics(
                    {"learning_rate": current_lr},
                    step=self._global_step,
                    epoch=self._current_epoch,
                )

            # Generate sample images
            if (
                self.sample_images
                and self.sample_interval > 0
                and (epoch + 1) % self.sample_interval == 0
            ):
                _logger.info(f"Generating sample images at epoch {self._current_epoch}")
                self._generate_samples(logger, self._global_step)

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

                # Always save latest checkpoint
                latest_path = checkpoint_dir / "latest_checkpoint.pth"
                self.save_checkpoint(
                    latest_path,
                    epoch=self._current_epoch,
                    is_best=False,
                    metrics={**train_metrics, **(val_metrics if val_metrics else {})},
                )

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Save training checkpoint including EMA and AMP state.

        Extends base checkpoint saving to include EMA and gradient scaler state.

        Args:
            path: Path to save checkpoint file
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            metrics: Dictionary of current metrics
            **kwargs: Additional metadata to save
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

        # Save EMA state
        if self.use_ema and self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        # Save scheduler state
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save scaler state for AMP
        if self.use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Add any additional metadata
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)

        # Enhanced logging
        if is_best:
            _logger.info(f"✓ Best model checkpoint saved: {path}")
        else:
            _logger.info(f"✓ Checkpoint saved: {path}")

        _logger.info(f"  Epoch: {epoch}, Global step: {self._global_step}")

        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            _logger.info(f"  Metrics: {metrics_str}")

        if self._best_metric is not None:
            _logger.info(f"  Best {self._best_metric_name}: {self._best_metric:.6f}")

        # Log diffusion-specific state
        if self.use_ema and self.ema is not None:
            _logger.debug("  EMA state included")
        if self.use_amp and self.scaler is not None:
            _logger.debug("  AMP scaler state included")
        if self.scheduler is not None:
            _logger.debug("  Scheduler state included")

        _logger.debug(f"  Checkpoint keys: {list(checkpoint.keys())}")
        _logger.debug(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")

    def load_checkpoint(
        self,
        path: Union[str, Path],
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load training checkpoint including EMA and AMP state.

        Extends base checkpoint loading to restore EMA and gradient scaler state.

        Args:
            path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            strict: Whether to strictly enforce state dict keys match

        Returns:
            Dictionary containing checkpoint metadata (epoch, metrics, etc.)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            _logger.error(f"Checkpoint not found: {path}")
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        _logger.info(f"Loading checkpoint from {path}")

        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as e:
            _logger.critical(f"Failed to load checkpoint from {path}")
            logger.exception(f"Error details: {e}")
            raise

        _logger.debug(f"  Checkpoint keys: {list(checkpoint.keys())}")
        _logger.debug(f"  Trainer class: {checkpoint.get('trainer_class', 'unknown')}")

        # Load model state
        model = self.get_model()
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        except Exception as e:
            _logger.error(f"Failed to load model state dict")
            logger.exception(f"Error details: {e}")
            if strict:
                raise
            else:
                _logger.warning("Continuing with non-strict loading")

        # Ensure model is on correct device after loading
        model.to(self.device)

        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer = self.get_optimizer()
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                _logger.error(f"Failed to load optimizer state dict")
                logger.exception(f"Error details: {e}")
                _logger.warning("Continuing without optimizer state")

        # Load EMA state
        if self.use_ema and self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
            _logger.debug("  EMA state restored")

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            _logger.debug("  Scheduler state restored")

        # Load scaler state for AMP
        if (
            self.use_amp
            and self.scaler is not None
            and "scaler_state_dict" in checkpoint
        ):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            _logger.debug("  AMP scaler state restored")

        # Restore training progress
        self._current_epoch = checkpoint.get("epoch", 0)
        self._global_step = checkpoint.get("global_step", 0)
        self._best_metric = checkpoint.get("best_metric", None)
        self._best_metric_name = checkpoint.get("best_metric_name", None)

        _logger.info(f"✓ Checkpoint loaded successfully")
        _logger.info(f"  Epoch: {self._current_epoch}, Global step: {self._global_step}")

        if "metrics" in checkpoint:
            metrics_str = ", ".join(
                [f"{k}: {v:.6f}" for k, v in checkpoint["metrics"].items()]
            )
            _logger.info(f"  Loaded metrics: {metrics_str}")

        if self._best_metric is not None:
            _logger.info(f"  Best {self._best_metric_name}: {self._best_metric:.6f}")

        return checkpoint

    def _generate_samples(self, logger: BaseLogger, step: int) -> None:
        """Generate and log sample images during training.

        Internal method called periodically during training to generate
        samples for visual quality assessment. Uses DiffusionSampler for generation.

        Args:
            logger: Logger to save generated samples
            step: Current training step
        """
        # Check if model supports conditional generation
        num_classes = getattr(self.model, "num_classes", None)

        if num_classes is not None and num_classes > 0:
            # Conditional generation: generate samples for each class using sampler
            all_samples, class_labels_list = self.sampler.sample_by_class(
                samples_per_class=self.samples_per_class,
                num_classes=num_classes,
                guidance_scale=self.guidance_scale,
                use_ema=self.use_ema,
                show_progress=False,
            )

            # Log samples
            logger.log_images(
                all_samples,
                tag=f"samples_epoch_{self._current_epoch}",
                step=step,
                class_labels=class_labels_list,
            )
        else:
            # Unconditional generation using sampler
            samples = self.sampler.sample(
                num_samples=8,  # Fixed number for visualization
                guidance_scale=0.0,
                use_ema=self.use_ema,
            )

            # Log samples
            logger.log_images(
                samples, tag=f"samples_epoch_{self._current_epoch}", step=step
            )

    def get_model(self) -> BaseModel:
        """Return the model being trained.

        Returns:
            The diffusion model instance
        """
        return self.model

    def get_dataloader(self) -> BaseDataLoader:
        """Return the dataloader.

        Returns:
            The dataloader instance
        """
        return self.dataloader

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer.

        Returns:
            The optimizer instance
        """
        return self.optimizer

    def get_logger(self) -> BaseLogger:
        """Return the logger.

        Returns:
            The logger instance
        """
        return self.logger
