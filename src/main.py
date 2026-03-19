"""Main entry point for AI Image Generation and Statistics Framework.

This module provides the CLI entry point for running different experiments
(classifier, diffusion, GAN) with a unified interface. It dispatches to the
appropriate experiment implementation based on the configuration.

Usage:
    # Train classifier
    python -m src.main configs/examples/classifier.yaml

    # Train diffusion model
    python -m src.main configs/diffusion.yaml

    # Generate synthetic data (for diffusion experiments)
    python -m src.main configs/diffusion-generate.yaml

    # Override config values with dot-notation
    python -m src.main configs/diffusion.yaml --model.architecture.image_size 60
    python -m src.main configs/diffusion.yaml --training.epochs 50 --data.loading.batch_size 16

Generation Mode (Diffusion):
    When mode='generate' in the config, the diffusion experiment uses
    lightweight sampler functions for inference instead of the full trainer.
    This eliminates unnecessary dependencies like optimizer and dataloader,
    providing faster initialization and lower memory usage for generation-only
    workflows.

    Required config sections for generation mode:
    - model: Defines the architecture
    - generation.checkpoint: Path to trained model checkpoint
    - generation.sampling: Sampling parameters (num_samples, use_ema, etc.)
    - generation.output: Output configuration (save_grid, save_individual, etc.)

Note:
    Parameters are specified in the YAML config file. Individual values
    can be overridden via CLI using dot-notation (e.g., --model.architecture.image_size 60).
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dotenv import load_dotenv

from src.utils.cli import parse_args
from src.utils.cli import validate_config as validate_cli_config
from src.utils.experiment import (
    create_experiment_logger,
    run_training,
    setup_experiment_common,
)
from src.utils.notification import notify_error, notify_success
from src.utils.training import create_optimizer, create_scheduler

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
    from src.experiments.classifier.models import (
        InceptionV3Classifier,
        ResNetClassifier,
    )
    from src.experiments.classifier.trainer import ClassifierTrainer
    from src.utils.config import resolve_output_path
    from src.utils.data.loaders import (
        create_train_loader,
        create_val_loader,
    )
    from src.utils.data.loaders import (
        get_class_names as _get_class_names,
    )
    from src.utils.data.transforms import (
        get_train_transforms,
        get_val_transforms,
    )

    # Validate classifier config (strict mode - no defaults)
    validate_classifier_config(config)

    # Create checkpoint directory first (needed before common setup)
    checkpoint_dir = resolve_output_path(config, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Run shared experiment setup (logging, device, seed, config snapshot)
    device, log_dir = setup_experiment_common(config, "CLASSIFIER EXPERIMENT STARTED")

    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Initialize data loaders
    data_config = config["data"]
    split_file = data_config["split_file"]
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

    # Build transforms
    normalize_str = normalize if normalize not in ["none", None] else None
    train_transform = get_train_transforms(
        image_size=image_size,
        crop_size=crop_size,
        horizontal_flip=horizontal_flip,
        color_jitter=color_jitter,
        rotation_degrees=rotation_degrees,
        normalize=normalize_str,
    )
    val_transform = get_val_transforms(
        image_size=image_size,
        crop_size=crop_size,
        normalize=normalize_str,
    )

    # Create data loaders
    compute_config = config.get("compute", {})
    seed = compute_config.get("seed")
    balancing_config = data_config.get("balancing")
    train_loader = create_train_loader(
        split_file=split_file,
        batch_size=batch_size,
        transform=train_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=shuffle_train,
        balancing_config=balancing_config,
        seed=seed,
    )
    val_loader = create_val_loader(
        split_file=split_file,
        batch_size=batch_size,
        transform=val_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Apply synthetic augmentation if configured
    syn_aug_config = data_config.get("synthetic_augmentation", {})
    if syn_aug_config.get("enabled"):
        import random as _random

        import numpy as np
        from torch.utils.data import ConcatDataset, DataLoader

        from src.utils.data.loaders import create_synthetic_augmentation_dataset

        limit_config = syn_aug_config.get("limit", {})
        gen_dataset = create_synthetic_augmentation_dataset(
            split_file=syn_aug_config["split_file"],
            transform=train_transform,
            limit_mode=limit_config.get("mode"),
            max_ratio=limit_config.get("max_ratio"),
            max_samples=limit_config.get("max_samples"),
            real_train_size=len(train_loader.dataset),  # type: ignore[arg-type]
            seed=seed,
        )
        combined_dataset = ConcatDataset([train_loader.dataset, gen_dataset])

        generator = None
        worker_init_fn = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

            def _worker_init_fn(worker_id: int) -> None:
                worker_seed = seed + worker_id  # type: ignore[operator]
                _random.seed(worker_seed)
                np.random.seed(worker_seed)
                torch.manual_seed(worker_seed)

            worker_init_fn = _worker_init_fn

        train_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )
        logger.info(
            f"Synthetic augmentation: added {len(gen_dataset)} generated images "
            f"to {len(combined_dataset) - len(gen_dataset)} real images"
        )

    # Get class names from split file
    class_names = _get_class_names(split_file)

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
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_config,
        valid_types=["adam", "adamw", "sgd"],
    )

    logger.info(f"Optimizer: {optimizer_config['type'].lower()}")
    logger.info(f"Learning rate: {optimizer_config['learning_rate']}")

    # Initialize scheduler if specified
    scheduler_config = training_config.get("scheduler", {})
    scheduler = create_scheduler(
        optimizer,
        scheduler_config,
        num_epochs=training_config["epochs"],
    )
    if scheduler is not None:
        logger.info(f"Scheduler: {scheduler_config.get('type')}")

    # Initialize metrics logger (for training metrics)
    metrics_logger = create_experiment_logger(
        config,
        log_dir,
        subdirs={
            "images": "predictions",
            "confusion_matrices": "confusion_matrices",
        },
    )

    # Initialize trainer
    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        logger=metrics_logger,
        val_loader=val_loader,
        device=device,
        show_progress=True,
        config=config,
    )

    # Set scheduler if available
    if scheduler is not None:
        trainer.scheduler = scheduler

    # Run training with standard error handling
    run_training(
        trainer,
        metrics_logger,
        num_epochs=training_config["epochs"],
        checkpoint_dir=str(checkpoint_dir),
        save_best=training_config["checkpointing"].get("save_best_only", True),
        checkpoint_frequency=training_config["checkpointing"].get("save_frequency", 10),
        save_latest_checkpoint=training_config["checkpointing"].get(
            "save_latest", True
        ),
        validate_frequency=training_config["validation"].get("frequency", 1),
        best_metric=training_config["validation"].get("metric", "accuracy"),
    )


def setup_experiment_diffusion(config: Dict[str, Any]) -> None:
    """Setup and run diffusion experiment.

    This function handles both training and generation modes:

    Training Mode (mode='train'):
        - Initializes full training infrastructure with trainer, dataloader, optimizer
        - Trains the diffusion model and saves checkpoints

    Generation Mode (mode='generate'):
        - Uses lightweight sampler functions for inference
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
    from src.experiments.diffusion.model import create_ddpm
    from src.experiments.diffusion.trainer import DiffusionTrainer
    from src.utils.config import (
        derive_image_size_from_model,
        derive_return_labels_from_model,
        resolve_output_path,
    )
    from src.utils.data.datasets import SplitFileDataset
    from src.utils.data.loaders import create_train_loader, create_val_loader
    from src.utils.data.transforms import (
        get_diffusion_transforms,
        get_diffusion_val_transforms,
    )

    # Validate diffusion config (strict mode - no defaults)
    validate_diffusion_config(config)

    # Get mode (train or generate)
    mode = config["mode"]

    # Run shared experiment setup (logging, device, seed, config snapshot)
    device, log_dir = setup_experiment_common(
        config, f"DIFFUSION EXPERIMENT STARTED - Mode: {mode.upper()}"
    )

    compute_config = config.get("compute", {})

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

    logger.info("Model: DDPM")
    logger.info(f"Image size: {arch_config['image_size']}")
    logger.info(f"Num classes: {cond_config['num_classes']}")
    logger.info(f"Num timesteps: {diff_config['num_timesteps']}")

    # Check if in generation mode
    if mode == "generate":
        # Generation mode: load checkpoint and generate samples
        # Import sampler for inference-only workflow
        from tqdm.contrib.logging import logging_redirect_tqdm

        from src.experiments.diffusion.sampler import sample

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
            state_dict = checkpoint["model_state_dict"]
            # Strip torch.compile prefix if present
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {
                    k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict)
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
        use_ema = sampling_config["use_ema"]
        if use_ema:
            if "ema_state_dict" in checkpoint:
                from src.experiments.diffusion.model import EMA

                ema_decay = sampling_config.get("ema_decay", 0.9999)
                ema = EMA(model, decay=ema_decay, device=device)
                ema.load_state_dict(checkpoint["ema_state_dict"])
                logger.info("Loaded EMA weights from checkpoint")
            else:
                logger.warning("use_ema=True but checkpoint has no EMA weights")
                logger.warning("Falling back to standard model weights")
                use_ema = False

        # Initialize metrics logger for metadata
        metrics_logger = create_experiment_logger(config, log_dir)

        logger.info("")
        logger.info(f"Generating {num_samples} samples...")

        # Prepare class labels if conditional generation
        class_labels = None
        if num_classes is not None:
            class_selection = sampling_config.get("class_selection")
            target_classes = (
                class_selection
                if class_selection is not None
                else list(range(num_classes))
            )
            n = len(target_classes)
            samples_per_class = num_samples // n
            remainder = num_samples % n
            class_labels = []
            for idx, cls in enumerate(target_classes):
                count = samples_per_class + (1 if idx < remainder else 0)
                class_labels.extend([cls] * count)
            class_labels = torch.tensor(class_labels, device=device)
            if class_selection is not None:
                logger.info(f"Class selection: {class_selection}")

        # Generate samples using sampler
        batch_size = sampling_config.get("batch_size", num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        all_samples = []
        # Capture actual console handler levels before logging_redirect_tqdm
        # replaces them. tqdm's replacement handlers default to NOTSET, which
        # leaks DEBUG messages. We restore the original levels inside the
        # context manager.
        # Note: console_level in config is guaranteed to exist (strict config
        # validation), so it always matches what setup_logging() applied.
        original_handler_levels = [
            h.level
            for h in logging.root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        default_console_level = (
            original_handler_levels[0] if original_handler_levels else logging.INFO
        )

        with logging_redirect_tqdm():
            # Restore the original console handler level on the tqdm
            # _TqdmLoggingHandler replacement(s) that default to NOTSET.
            for h in logging.root.handlers:
                if (
                    type(h).__name__ == "_TqdmLoggingHandler"
                    and h.level == logging.NOTSET
                ):
                    h.setLevel(default_console_level)

            for batch_idx, start_idx in enumerate(range(0, num_samples, batch_size), 1):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_labels = (
                    class_labels[start_idx:end_idx]
                    if class_labels is not None
                    else None
                )

                batch_samples = sample(
                    model,
                    device,
                    num_samples=end_idx - start_idx,
                    class_labels=batch_labels,
                    guidance_scale=sampling_config["guidance_scale"],
                    use_ema=use_ema,
                    ema=ema,
                    show_progress=True,
                    progress_desc=f"Denoising batch {batch_idx}/{num_batches}",
                )
                all_samples.append(batch_samples)

        samples = torch.cat(all_samples, dim=0)

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
            for i, img in enumerate(samples):
                save_image(img, output_dir / f"sample_{i:04d}.png", normalize=True)
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

        # Initialize data loaders
        data_config = config["data"]
        balancing_config = data_config.get("balancing")
        seed = compute_config.get("seed")
        image_size = derive_image_size_from_model(config)
        return_labels = derive_return_labels_from_model(config)

        # Build diffusion transforms
        augmentation_config = data_config["augmentation"]
        color_jitter_config = augmentation_config["color_jitter"]
        train_transform = get_diffusion_transforms(
            image_size=image_size,
            horizontal_flip=augmentation_config["horizontal_flip"],
            rotation_degrees=augmentation_config["rotation_degrees"],
            color_jitter=color_jitter_config["enabled"],
            color_jitter_strength=color_jitter_config.get("strength", 0.1),
        )
        val_transform = get_diffusion_val_transforms(image_size=image_size)

        train_loader = create_train_loader(
            split_file=data_config["split_file"],
            batch_size=data_config["loading"]["batch_size"],
            transform=train_transform,
            num_workers=data_config["loading"]["num_workers"],
            pin_memory=data_config["loading"]["pin_memory"],
            drop_last=data_config["loading"]["drop_last"],
            shuffle=data_config["loading"]["shuffle_train"],
            return_labels=return_labels,
            balancing_config=balancing_config,
            seed=seed,
        )
        val_loader = create_val_loader(
            split_file=data_config["split_file"],
            batch_size=data_config["loading"]["batch_size"],
            transform=val_transform,
            num_workers=data_config["loading"]["num_workers"],
            pin_memory=data_config["loading"]["pin_memory"],
            return_labels=return_labels,
        )

        # Log balancing configuration summary
        if balancing_config:
            active = []
            if balancing_config.get("weighted_sampler", {}).get("enabled"):
                active.append("weighted_sampler")
            if balancing_config.get("downsampling", {}).get("enabled"):
                active.append("downsampling")
            if balancing_config.get("upsampling", {}).get("enabled"):
                active.append("upsampling")
            if balancing_config.get("class_weights", {}).get("enabled"):
                active.append("class_weights")
            logger.info(f"Active balancing strategies: {active if active else 'none'}")

        # Compute class weight tensor for weighted loss if enabled
        class_weight_tensor = None
        if balancing_config and balancing_config.get("class_weights", {}).get(
            "enabled"
        ):
            from src.utils.data.samplers import compute_weights_from_config

            cw_config = balancing_config["class_weights"]
            # Need to get targets from dataset to compute weights
            temp_dataset = SplitFileDataset(
                split_file=data_config["split_file"], split="train"
            )
            weights_dict = compute_weights_from_config(
                targets=temp_dataset.targets,
                method=cw_config["method"],
                beta=cw_config.get("beta", 0.999),
                manual_weights=cw_config.get("manual_weights"),
                normalize=cw_config.get("normalize", True),
                num_classes=config["model"]["conditioning"].get("num_classes"),
            )
            num_classes = config["model"]["conditioning"].get("num_classes", 2)
            class_weight_tensor = torch.zeros(num_classes)
            for cls_idx, weight in weights_dict.items():
                class_weight_tensor[cls_idx] = weight
            logger.info(f"Class weight tensor for loss: {class_weight_tensor.tolist()}")

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
        metrics_logger = create_experiment_logger(
            config,
            log_dir,
            subdirs={
                "images": "samples",
                "denoising": "denoising",
            },
        )

        # Initialize optimizer
        optimizer_config = training_config["optimizer"]
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_config,
            valid_types=["adam", "adamw"],
        )

        logger.info(f"Optimizer: {optimizer_config['type'].lower()}")
        logger.info(f"Learning rate: {optimizer_config['learning_rate']}")

        # Initialize scheduler if specified
        scheduler_config = training_config["scheduler"]
        scheduler = create_scheduler(
            optimizer,
            scheduler_config,
            num_epochs=training_config["epochs"],
        )
        if scheduler is not None:
            logger.info(f"Scheduler: {scheduler_config.get('type')}")

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
            train_loader=train_loader,
            optimizer=optimizer,
            logger=metrics_logger,
            val_loader=val_loader,
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
            log_denoising_interval=(
                visualization_config.get("log_denoising_interval")
                if visualization_config["enabled"]
                else None
            ),
            num_samples=visualization_config["num_samples"],
            guidance_scale=visualization_config["guidance_scale"],
            config=config,
            class_weights=class_weight_tensor,
        )

        # Run training with standard error handling
        run_training(
            trainer,
            metrics_logger,
            num_epochs=training_config["epochs"],
            checkpoint_dir=str(checkpoint_dir),
            save_best=checkpointing_config.get("save_best_only", True),
            checkpoint_frequency=checkpointing_config["save_frequency"],
            save_latest_checkpoint=checkpointing_config.get("save_latest", True),
            validate_frequency=validation_config["frequency"],
            best_metric=validation_config["metric"],
        )


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


def setup_experiment_sample_selection(config: Dict[str, Any]) -> None:
    """Setup and run sample selection experiment.

    This experiment compares generated images against real training images in
    feature space, scores each generated sample by its distance to the real
    data manifold, and selects the highest-quality samples.

    Args:
        config: Configuration dictionary with feature_extraction, data, scoring,
                selection, and output sections.

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If data directories or files don't exist
    """
    from src.experiments.sample_selection.config import (
        validate_config as validate_sample_selection_config,
    )
    from src.experiments.sample_selection.selector import run_sample_selection

    validate_sample_selection_config(config)

    device, log_dir = setup_experiment_common(
        config, "SAMPLE SELECTION EXPERIMENT STARTED"
    )

    run_sample_selection(config, device, log_dir)

    logger.info("")
    logger.info("Sample selection completed successfully!")


def setup_experiment_data_preparation(config: Dict[str, Any]) -> None:
    """Setup and run data preparation experiment.

    This experiment scans class directories, performs a stratified train/val split,
    and saves the result as a JSON file for use by training experiments.

    Args:
        config: Configuration dictionary with 'classes' and 'split' sections

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If class directories don't exist
    """
    from src.experiments.data_preparation.config import (
        validate_config as validate_data_preparation_config,
    )
    from src.experiments.data_preparation.prepare import prepare_split

    validate_data_preparation_config(config)
    prepare_split(config)


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
    # Load environment variables from .env file
    load_dotenv()

    # Parse arguments and load configuration
    try:
        config = parse_args(args)
        # Validate basic config structure
        validate_cli_config(config)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    # Get experiment type
    experiment = config["experiment"]

    # Record start time for duration tracking
    start_time = time.time()

    # Dispatch to experiment
    try:
        if experiment == "classifier":
            setup_experiment_classifier(config)
        elif experiment == "diffusion":
            setup_experiment_diffusion(config)
        elif experiment == "gan":
            setup_experiment_gan(config)
        elif experiment == "sample_selection":
            setup_experiment_sample_selection(config)
        elif experiment == "data_preparation":
            setup_experiment_data_preparation(config)
        else:
            raise ValueError(
                f"Unknown experiment type: {experiment}. "
                f"Supported experiments: classifier, diffusion, gan, "
                f"sample_selection, data_preparation"
            )

        # Notify on success
        duration = time.time() - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total duration: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        notify_success(config, duration)

    except NotImplementedError as e:
        logger.error(f"Experiment '{experiment}' is not yet implemented: {e}")
        notify_error(config, e)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Experiment '{experiment}' failed with error: {e}")
        notify_error(config, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
