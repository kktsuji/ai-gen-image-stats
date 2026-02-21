"""Main entry point for AI Image Generation and Statistics Framework.

This module provides the CLI entry point for running different experiments
(classifier, diffusion, GAN) with a unified interface. It dispatches to the
appropriate experiment implementation based on the configuration.

Usage:
    # Train classifier
    python -m src.main configs/classifier/baseline.yaml
    python -m src.main configs/classifier/inceptionv3.yaml

    # Train diffusion model
    python -m src.main configs/diffusion/default.yaml

    # Generate synthetic data (for diffusion experiments)
    python -m src.main configs/diffusion/generate.yaml

Generation Mode (Diffusion):
    When mode='generate' in the config, the diffusion experiment uses a
    lightweight DiffusionSampler for inference instead of the full trainer.
    This eliminates unnecessary dependencies like optimizer and dataloader,
    providing faster initialization and lower memory usage for generation-only
    workflows.

    Required config sections for generation mode:
    - model: Defines the architecture
    - generation.checkpoint: Path to trained model checkpoint
    - generation.sampling: Sampling parameters (num_samples, use_ema, etc.)
    - generation.output: Output configuration (save_grid, save_individual, etc.)

Note:
    All parameters must be specified in the YAML config file.
    CLI parameter overrides are not supported.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.utils.cli import parse_args
from src.utils.cli import validate_config as validate_cli_config
from src.utils.config import save_config
from src.utils.device import get_device
from src.utils.git import get_git_info
from src.utils.logging import get_log_file_path, setup_logging

# Module-level logger
logger = logging.getLogger(__name__)


def setup_experiment_classifier(config: Dict[str, Any]) -> None:
    """Setup and run classifier experiment.

    Args:
        config: Merged configuration dictionary

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If experiment execution fails
    """
    from src.experiments.classifier.config import (
        validate_config as validate_classifier_config,
    )
    from src.experiments.classifier.dataloader import ClassifierDataLoader
    from src.experiments.classifier.logger import ClassifierLogger
    from src.experiments.classifier.models import (
        InceptionV3Classifier,
        ResNetClassifier,
    )
    from src.experiments.classifier.trainer import ClassifierTrainer
    from src.utils.config import resolve_output_path

    # Validate classifier config (strict mode - no defaults)
    validate_classifier_config(config)

    # Create output directories first (needed for log file)
    checkpoint_dir = resolve_output_path(config, "checkpoints")
    log_dir = resolve_output_path(config, "logs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get logging configuration (with defaults)
    logging_config = config.get("logging", {})
    console_level = logging_config.get("console_level", "INFO")
    file_level = logging_config.get("file_level", "DEBUG")
    log_format = logging_config.get("format")
    date_format = logging_config.get("date_format")
    timezone = logging_config.get("timezone", "UTC")
    module_levels = logging_config.get("module_levels")

    # Setup logging FIRST (before any other operations)
    log_file = get_log_file_path(
        output_base_dir=config["output"]["base_dir"],
        log_subdir=config["output"]["subdirs"]["logs"],
        timezone=timezone,
    )

    # Initialize logging
    setup_logging(
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        log_format=log_format,
        date_format=date_format,
        timezone=timezone,
        module_levels=module_levels,
    )

    # Log experiment start with banner
    logger.info("=" * 80)
    logger.info("CLASSIFIER EXPERIMENT STARTED")
    logger.info("=" * 80)

    # Log Git information for reproducibility
    git_info = get_git_info()
    if git_info["repository_url"]:
        logger.info(f"Repository: {git_info['repository_url']}")
    if git_info["commit_hash"]:
        logger.info(f"Commit: {git_info['commit_hash']}")

    logger.info(f"Log file: {log_file}")
    logger.info(f"Console log level: {console_level}")
    logger.info(f"File log level: {file_level}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Log directory: {log_dir}")

    # Set up device
    device_config = config.get("compute", {}).get("device", "auto")

    if device_config == "auto":
        device = get_device()
    else:
        device = device_config

    logger.info(f"Using device: {device}")

    # Set random seed if specified
    seed = config.get("compute", {}).get("seed")

    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")

    # Save configuration to log directory
    config_save_path = log_dir / "config.yaml"
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")

    # Initialize dataloader
    data_config = config["data"]
    train_path = data_config["paths"]["train"]
    val_path = data_config["paths"].get("val")
    batch_size = data_config["loading"]["batch_size"]
    num_workers = data_config["loading"]["num_workers"]
    pin_memory = data_config["loading"].get("pin_memory", True)
    drop_last = data_config["loading"].get("drop_last", False)
    shuffle_train = data_config["loading"].get("shuffle_train", True)
    image_size = data_config["preprocessing"].get("image_size", 256)
    crop_size = data_config["preprocessing"].get("crop_size", 224)
    normalize = data_config["preprocessing"].get("normalize", "imagenet")
    horizontal_flip = data_config["augmentation"].get("horizontal_flip", True)
    rotation_degrees = data_config["augmentation"].get("rotation_degrees", 0)
    color_jitter_config = data_config["augmentation"].get("color_jitter", {})
    color_jitter = (
        color_jitter_config.get("enabled", False)
        if isinstance(color_jitter_config, dict)
        else color_jitter_config
    )

    dataloader = ClassifierDataLoader(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        crop_size=crop_size,
        horizontal_flip=horizontal_flip,
        color_jitter=color_jitter,
        rotation_degrees=rotation_degrees,
        normalize=normalize,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle_train=shuffle_train,
    )

    # Get class names from dataset
    train_loader = dataloader.get_train_loader()
    if hasattr(train_loader.dataset, "classes"):
        class_names = train_loader.dataset.classes
    else:
        num_classes = config["model"]["architecture"]["num_classes"]
        class_names = [f"Class {i}" for i in range(num_classes)]

    logger.info(f"Number of classes: {len(class_names)}")
    logger.debug(f"Classes: {class_names}")

    # Initialize model
    model_config = config["model"]
    model_name = model_config["architecture"]["name"].lower()
    num_classes = model_config["architecture"]["num_classes"]
    pretrained = model_config["initialization"].get("pretrained", True)
    freeze_backbone = model_config["initialization"].get("freeze_backbone", False)
    trainable_layers = model_config["initialization"].get("trainable_layers")

    if model_name == "inceptionv3":
        dropout = model_config.get("regularization", {}).get("dropout", 0.5)
        model = InceptionV3Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            trainable_layers=trainable_layers,
            dropout=dropout,
        )
    elif model_name in ["resnet50", "resnet101", "resnet152"]:
        model = ResNetClassifier(
            variant=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: inceptionv3, resnet50, resnet101, resnet152"
        )

    model = model.to(device)
    logger.info(f"Model: {model_name}")
    logger.info(f"Pretrained: {pretrained}")
    logger.info(f"Freeze backbone: {freeze_backbone}")

    # Initialize optimizer
    training_config = config["training"]
    optimizer_config = training_config["optimizer"]
    optimizer_name = optimizer_config["type"].lower()
    learning_rate = optimizer_config["learning_rate"]
    weight_decay = optimizer_config.get("weight_decay", 0.0)
    # TODO: Implement gradient clipping in trainer
    # gradient_clip_norm = optimizer_config.get("gradient_clip_norm")

    # Build optimizer kwargs
    optimizer_kwargs = {"weight_decay": weight_decay}
    if "betas" in optimizer_config:
        optimizer_kwargs["betas"] = optimizer_config["betas"]
    if "momentum" in optimizer_config:
        optimizer_kwargs["momentum"] = optimizer_config["momentum"]

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, **optimizer_kwargs
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, **optimizer_kwargs
        )
    elif optimizer_name == "sgd":
        # SGD typically needs momentum
        if "momentum" not in optimizer_kwargs:
            optimizer_kwargs["momentum"] = 0.9
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, **optimizer_kwargs
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw, sgd"
        )

    logger.info(f"Optimizer: {optimizer_name}")
    logger.info(f"Learning rate: {learning_rate}")

    # Initialize scheduler if specified
    scheduler = None
    scheduler_config = training_config.get("scheduler", {})
    scheduler_name = scheduler_config.get("type")

    if scheduler_name and scheduler_name.lower() != "none":
        # Handle auto T_max
        if scheduler_name.lower() == "cosine":
            t_max = scheduler_config.get("T_max", "auto")
            if t_max == "auto":
                t_max = training_config["epochs"]
            eta_min = scheduler_config.get("eta_min", 1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        elif scheduler_name.lower() == "step":
            step_size = scheduler_config.get("step_size", 30)
            gamma = scheduler_config.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_name.lower() == "plateau":
            mode = scheduler_config.get("mode", "min")
            factor = scheduler_config.get("factor", 0.1)
            patience = scheduler_config.get("patience", 10)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )
        else:
            raise ValueError(
                f"Unknown scheduler: {scheduler_name}. "
                f"Supported: cosine, step, plateau, none"
            )
        logger.info(f"Scheduler: {scheduler_name}")

    # Initialize metrics logger (for training metrics)
    tensorboard_config = logging_config.get("metrics", {}).get("tensorboard", {})
    tb_log_dir = (
        resolve_output_path(config, "tensorboard")
        if "tensorboard" in config.get("output", {}).get("subdirs", {})
        else None
    )
    metrics_logger = ClassifierLogger(
        log_dir=log_dir,
        class_names=class_names,
        tensorboard_config=tensorboard_config,
        tb_log_dir=tb_log_dir,
    )

    # Initialize trainer
    trainer = ClassifierTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=metrics_logger,
        device=device,
        show_progress=True,
        config=config,
    )

    # Set scheduler if available
    if scheduler is not None:
        trainer.scheduler = scheduler

    # Get training parameters
    num_epochs = training_config["epochs"]
    save_best = training_config["checkpointing"].get("save_best_only", True)
    checkpoint_frequency = training_config["checkpointing"].get("save_frequency", 10)
    save_latest = training_config["checkpointing"].get("save_latest", True)
    validate_frequency = training_config["validation"].get("frequency", 1)
    best_metric = training_config["validation"].get("metric", "accuracy")

    logger.info(f"Starting training for {num_epochs} epochs...")

    try:
        trainer.train(
            num_epochs=num_epochs,
            checkpoint_dir=str(checkpoint_dir),
            save_best=save_best,
            checkpoint_frequency=checkpoint_frequency,
            validate_frequency=validate_frequency,
            best_metric=best_metric,
            save_latest_checkpoint=save_latest,
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

    # Close metrics logger
    metrics_logger.close()
    logger.info("Training completed successfully!")


def setup_experiment_diffusion(config: Dict[str, Any]) -> None:
    """Setup and run diffusion experiment.

    This function handles both training and generation modes:

    Training Mode (mode='train'):
        - Initializes full training infrastructure with trainer, dataloader, optimizer
        - Trains the diffusion model and saves checkpoints

    Generation Mode (mode='generate'):
        - Uses lightweight DiffusionSampler for inference
        - Loads checkpoint and generates samples without training dependencies
        - No optimizer or dataloader needed (class info from config)
        - Handles EMA weights if available in checkpoint

    Args:
        config: Merged configuration dictionary

    Raises:
        ValueError: If configuration is invalid or checkpoint missing
        FileNotFoundError: If checkpoint file not found in generation mode
        RuntimeError: If experiment execution fails
    """
    from src.experiments.diffusion.config import (
        validate_config as validate_diffusion_config,
    )
    from src.experiments.diffusion.dataloader import DiffusionDataLoader
    from src.experiments.diffusion.logger import DiffusionLogger
    from src.experiments.diffusion.model import create_ddpm
    from src.experiments.diffusion.trainer import DiffusionTrainer
    from src.utils.config import (
        derive_image_size_from_model,
        derive_return_labels_from_model,
        resolve_output_path,
    )

    # Validate diffusion config (strict mode - no defaults)
    validate_diffusion_config(config)

    # Get mode (train or generate)
    mode = config.get("mode", "train")

    # Create output directories first (needed for log file)
    log_dir = resolve_output_path(config, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get logging configuration (with defaults)
    logging_config = config.get("logging", {})
    console_level = logging_config.get("console_level", "INFO")
    file_level = logging_config.get("file_level", "DEBUG")
    log_format = logging_config.get("format")
    date_format = logging_config.get("date_format")
    timezone = logging_config.get("timezone", "UTC")
    module_levels = logging_config.get("module_levels")

    # Setup logging FIRST (before any other operations)
    log_file = get_log_file_path(
        output_base_dir=config["output"]["base_dir"],
        log_subdir=config["output"]["subdirs"]["logs"],
        timezone=timezone,
    )

    # Initialize logging
    setup_logging(
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        log_format=log_format,
        date_format=date_format,
        timezone=timezone,
        module_levels=module_levels,
    )

    # Log experiment start with banner
    logger.info("=" * 80)
    logger.info(f"DIFFUSION EXPERIMENT STARTED - Mode: {mode.upper()}")
    logger.info("=" * 80)

    # Log Git information for reproducibility
    git_info = get_git_info()
    if git_info["repository_url"]:
        logger.info(f"Repository: {git_info['repository_url']}")
    if git_info["commit_hash"]:
        logger.info(f"Commit: {git_info['commit_hash']}")

    logger.info(f"Log file: {log_file}")
    logger.info(f"Console log level: {console_level}")
    logger.info(f"File log level: {file_level}")
    logger.info(f"Log directory: {log_dir}")

    # Set up device (now in compute section)
    compute_config = config.get("compute", {})
    device_config = compute_config.get("device", "auto")
    if device_config == "auto":
        device = get_device()
    else:
        device = device_config

    logger.info(f"Using device: {device}")

    # Set random seed if specified (now in compute section)
    seed = compute_config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to: {seed}")

    # Save configuration to log directory
    config_save_path = log_dir / "config.yaml"
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")

    # Initialize model
    model_config = config["model"]
    arch_config = model_config["architecture"]
    diff_config = model_config["diffusion"]
    cond_config = model_config["conditioning"]

    model = create_ddpm(
        image_size=arch_config["image_size"],
        in_channels=arch_config["in_channels"],
        model_channels=arch_config["model_channels"],
        channel_multipliers=tuple(arch_config["channel_multipliers"]),
        num_classes=cond_config["num_classes"],
        num_timesteps=diff_config["num_timesteps"],
        beta_schedule=diff_config["beta_schedule"],
        beta_start=diff_config["beta_start"],
        beta_end=diff_config["beta_end"],
        class_dropout_prob=cond_config["class_dropout_prob"],
        use_attention=tuple(arch_config["use_attention"]),
        device=device,
    )

    logger.info(f"Model: DDPM")
    logger.info(f"Image size: {arch_config['image_size']}")
    logger.info(f"Num classes: {cond_config['num_classes']}")
    logger.info(f"Num timesteps: {diff_config['num_timesteps']}")

    # Check if in generation mode
    if mode == "generate":
        # Generation mode: load checkpoint and generate samples
        # Import sampler for inference-only workflow
        from src.experiments.diffusion.sampler import DiffusionSampler

        generation_config = config["generation"]
        checkpoint_path = generation_config.get("checkpoint")
        if not checkpoint_path:
            raise ValueError("generation.checkpoint is required for generation mode")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("")
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Validate checkpoint contains model weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and any(
            k.startswith("module.") for k in checkpoint.keys()
        ):
            # Direct state dict (possibly from DataParallel)
            model.load_state_dict(checkpoint)
        else:
            raise ValueError(
                f"Checkpoint does not contain 'model_state_dict'. "
                f"Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}"
            )

        # Get generation configuration
        sampling_config = generation_config["sampling"]
        output_config = generation_config["output"]
        num_samples = sampling_config["num_samples"]
        num_classes = cond_config["num_classes"]

        # Validate num_samples
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        # Validate compatibility with conditional generation
        if num_classes is not None and num_samples < num_classes:
            logger.warning(
                f"num_samples ({num_samples}) < num_classes ({num_classes}). "
                f"Some classes will have no samples."
            )

        # Load EMA if requested and available
        ema = None
        if sampling_config["use_ema"]:
            if "ema_state_dict" in checkpoint:
                from src.experiments.diffusion.model import EMA

                ema = EMA(model, decay=0.9999, device=device)
                ema.load_state_dict(checkpoint["ema_state_dict"])
                logger.info("Loaded EMA weights from checkpoint")
            else:
                logger.warning("use_ema=True but checkpoint has no EMA weights")
                logger.warning("Falling back to standard model weights")
                sampling_config["use_ema"] = False

        # Initialize metrics logger for metadata
        metrics_logger = DiffusionLogger(log_dir=log_dir)

        # Create sampler (no optimizer or dataloader needed!)
        sampler = DiffusionSampler(
            model=model,
            device=device,
            ema=ema,
        )

        logger.info("")
        logger.info(f"Generating {num_samples} samples...")

        # Prepare class labels if conditional generation
        class_labels = None
        if num_classes is not None:
            # Generate balanced samples across all classes
            samples_per_class = num_samples // num_classes
            remainder = num_samples % num_classes
            class_labels = []
            for i in range(num_classes):
                count = samples_per_class + (1 if i < remainder else 0)
                class_labels.extend([i] * count)
            class_labels = torch.tensor(class_labels, device=device)

        # Generate samples using sampler
        samples = sampler.sample(
            num_samples=num_samples,
            class_labels=class_labels,
            guidance_scale=sampling_config["guidance_scale"],
            use_ema=sampling_config["use_ema"],
            show_progress=True,
        )

        # Save generated samples to configured generated directory
        output_dir = resolve_output_path(config, "generated")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs
        from torchvision.utils import save_image

        if output_config["save_grid"]:
            grid_nrow = output_config["grid_nrow"]
            save_image(
                samples,
                output_dir / "generated_samples.png",
                nrow=grid_nrow,
                normalize=True,
            )
            logger.info(
                f"Saved generated grid to: {output_dir / 'generated_samples.png'}"
            )

        # Save individual samples if configured
        if output_config["save_individual"]:
            for i, sample in enumerate(samples):
                save_image(sample, output_dir / f"sample_{i:04d}.png", normalize=True)
            logger.info(f"Saved {len(samples)} individual samples to: {output_dir}")

        metrics_logger.close()
        logger.info("")
        logger.info("Generation completed successfully!")

    else:
        # Training mode
        training_config = config["training"]

        # Create checkpoint directory from output.subdirs
        checkpoint_dir = resolve_output_path(config, "checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {checkpoint_dir}")

        # Initialize dataloader
        data_config = config["data"]
        dataloader = DiffusionDataLoader(
            train_path=data_config["paths"]["train"],
            val_path=data_config["paths"].get("val"),
            batch_size=data_config["loading"]["batch_size"],
            num_workers=data_config["loading"]["num_workers"],
            image_size=derive_image_size_from_model(config),
            horizontal_flip=data_config["augmentation"]["horizontal_flip"],
            rotation_degrees=data_config["augmentation"]["rotation_degrees"],
            color_jitter=data_config["augmentation"]["color_jitter"]["enabled"],
            color_jitter_strength=data_config["augmentation"]["color_jitter"][
                "strength"
            ],
            pin_memory=data_config["loading"]["pin_memory"],
            drop_last=data_config["loading"]["drop_last"],
            shuffle_train=data_config["loading"]["shuffle_train"],
            return_labels=derive_return_labels_from_model(config),
        )

        # Apply performance optimizations
        performance_config = training_config["performance"]

        # Enable TF32 on Ampere+ GPUs (PyTorch 1.7+)
        if performance_config.get("use_tf32", True) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for faster training on Ampere+ GPUs")

        # Enable cuDNN benchmark mode
        if (
            performance_config.get("cudnn_benchmark", True)
            and torch.cuda.is_available()
        ):
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN benchmark mode")

        # Compile model (PyTorch 2.0+)
        if performance_config.get("compile_model", False):
            try:
                model = torch.compile(model)
                logger.info("Compiled model with torch.compile()")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

        # Initialize metrics logger
        tensorboard_config = logging_config.get("metrics", {}).get("tensorboard", {})
        tb_log_dir = (
            resolve_output_path(config, "tensorboard")
            if "tensorboard" in config.get("output", {}).get("subdirs", {})
            else None
        )
        metrics_logger = DiffusionLogger(
            log_dir=log_dir,
            tensorboard_config=tensorboard_config,
            tb_log_dir=tb_log_dir,
        )

        # Initialize optimizer
        optimizer_config = training_config["optimizer"]
        optimizer_name = optimizer_config["type"].lower()

        # Extract optimizer kwargs (excluding type, learning_rate, gradient_clip_norm)
        optimizer_kwargs = {
            k: v
            for k, v in optimizer_config.items()
            if k not in ["type", "learning_rate", "gradient_clip_norm"]
        }

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer_config["learning_rate"],
                **optimizer_kwargs,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_config["learning_rate"],
                **optimizer_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw"
            )

        logger.info(f"Optimizer: {optimizer_name}")
        logger.info(f"Learning rate: {optimizer_config['learning_rate']}")

        # Initialize scheduler if specified
        scheduler = None
        scheduler_config = training_config["scheduler"]
        scheduler_name = scheduler_config.get("type")

        if scheduler_name and scheduler_name.lower() not in ["none", None]:
            # Extract scheduler kwargs (excluding type)
            scheduler_kwargs = {
                k: v for k, v in scheduler_config.items() if k != "type"
            }

            # Handle auto T_max for cosine scheduler
            if scheduler_name.lower() == "cosine":
                if scheduler_kwargs.get("T_max") == "auto":
                    scheduler_kwargs["T_max"] = training_config["epochs"]
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, **scheduler_kwargs
                )
            elif scheduler_name.lower() == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, **scheduler_kwargs
                )
            elif scheduler_name.lower() == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_kwargs
                )
            else:
                raise ValueError(
                    f"Unknown scheduler: {scheduler_name}. "
                    f"Supported: cosine, step, plateau, none"
                )

            logger.info(f"Scheduler: {scheduler_name}")

        # Get configuration sections
        ema_config = training_config["ema"]
        performance_config = training_config["performance"]
        visualization_config = training_config["visualization"]
        optimizer_config = training_config["optimizer"]
        validation_config = training_config["validation"]
        checkpointing_config = training_config["checkpointing"]

        # Initialize trainer (use training.visualization for sampling)
        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=metrics_logger,
            device=device,
            show_progress=True,
            use_ema=ema_config["enabled"],
            ema_decay=ema_config["decay"],
            use_amp=performance_config["use_amp"],
            gradient_clip_norm=optimizer_config.get("gradient_clip_norm"),
            scheduler=scheduler,
            log_images_interval=(
                visualization_config.get("log_images_interval")
                if visualization_config["enabled"]
                else None
            ),
            log_sample_comparison_interval=(
                visualization_config.get("log_sample_comparison_interval")
                if visualization_config["enabled"]
                else None
            ),
            log_denoising_interval=(
                visualization_config.get("log_denoising_interval")
                if visualization_config["enabled"]
                else None
            ),
            samples_per_class=2,  # Will calculate based on num_samples internally
            guidance_scale=visualization_config["guidance_scale"],
            config=config,
        )

        # Train the model
        num_epochs = training_config["epochs"]
        logger.info(f"Starting training for {num_epochs} epochs...")

        try:
            trainer.train(
                num_epochs=num_epochs,
                checkpoint_dir=str(checkpoint_dir),
                save_best=checkpointing_config["save_best_only"],
                checkpoint_frequency=checkpointing_config["save_frequency"],
                save_latest_checkpoint=checkpointing_config.get("save_latest", True),
                validate_frequency=validation_config["frequency"],
                best_metric=validation_config["metric"],
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

        # Close metrics logger
        metrics_logger.close()
        logger.info("Training completed successfully!")


def setup_experiment_gan(config: Dict[str, Any]) -> None:
    """Setup and run GAN experiment.

    Args:
        config: Merged configuration dictionary

    Raises:
        NotImplementedError: This experiment type is not yet implemented
    """
    raise NotImplementedError(
        "GAN experiment is not yet implemented. "
        "This will be added in a future refactoring step."
    )


def main(args: Optional[list] = None) -> None:
    """Main entry point for the application.

    Parses command-line arguments, validates configuration, and dispatches
    to the appropriate experiment implementation.

    Args:
        args: Optional list of argument strings (default: sys.argv)

    Raises:
        ValueError: If configuration is invalid
        NotImplementedError: If experiment type is not implemented
    """
    # Parse arguments and load configuration
    config = parse_args(args)

    # Validate basic config structure
    validate_cli_config(config)

    # Get experiment type
    experiment = config["experiment"]

    # Dispatch to experiment
    try:
        if experiment == "classifier":
            setup_experiment_classifier(config)
        elif experiment == "diffusion":
            setup_experiment_diffusion(config)
        elif experiment == "gan":
            setup_experiment_gan(config)
        else:
            raise ValueError(
                f"Unknown experiment type: {experiment}. "
                f"Supported experiments: classifier, diffusion, gan"
            )
    except Exception as e:
        logger.exception(f"Experiment '{experiment}' failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
