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

Note:
    All parameters must be specified in the YAML config file.
    CLI parameter overrides are not supported.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from src.utils.cli import parse_args
from src.utils.cli import validate_config as validate_cli_config
from src.utils.device import get_device


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

    # Validate classifier config (strict mode - no defaults)
    validate_classifier_config(config)

    # Set up device
    device_config = config.get("training", {}).get("device", "auto")
    if device_config == "auto":
        device = get_device()
    else:
        device = device_config

    print(f"Using device: {device}")

    # Set random seed if specified
    seed = config.get("training", {}).get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")

    # Create output directories
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    log_dir = Path(config["output"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Log directory: {log_dir}")

    # Save configuration to log directory
    config_save_path = log_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {config_save_path}")

    # Initialize dataloader
    data_config = config["data"]
    dataloader = ClassifierDataLoader(
        train_path=data_config["train_path"],
        val_path=data_config.get("val_path"),
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        image_size=data_config.get("image_size", 256),
        crop_size=data_config.get("crop_size", 224),
        horizontal_flip=data_config.get("horizontal_flip", True),
        color_jitter=data_config.get("color_jitter", False),
        rotation_degrees=data_config.get("rotation_degrees", 0),
        normalize=data_config.get("normalize", "imagenet"),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=data_config.get("drop_last", False),
        shuffle_train=data_config.get("shuffle_train", True),
    )

    # Get class names from dataset
    train_loader = dataloader.get_train_loader()
    if hasattr(train_loader.dataset, "classes"):
        class_names = train_loader.dataset.classes
    else:
        class_names = [f"Class {i}" for i in range(config["model"]["num_classes"])]

    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")

    # Initialize model
    model_config = config["model"]
    model_name = model_config["name"].lower()

    if model_name == "inceptionv3":
        model = InceptionV3Classifier(
            num_classes=model_config["num_classes"],
            pretrained=model_config["pretrained"],
            freeze_backbone=model_config.get("freeze_backbone", False),
            trainable_layers=model_config.get("trainable_layers"),
            dropout=model_config.get("dropout", 0.5),
        )
    elif model_name in ["resnet50", "resnet101", "resnet152"]:
        model = ResNetClassifier(
            variant=model_name,
            num_classes=model_config["num_classes"],
            pretrained=model_config["pretrained"],
            freeze_backbone=model_config.get("freeze_backbone", False),
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: inceptionv3, resnet50, resnet101, resnet152"
        )

    model = model.to(device)
    print(f"Model: {model_name}")
    print(f"Pretrained: {model_config['pretrained']}")
    print(f"Freeze backbone: {model_config.get('freeze_backbone', False)}")

    # Initialize optimizer
    training_config = config["training"]
    optimizer_name = training_config["optimizer"].lower()
    optimizer_kwargs = training_config.get("optimizer_kwargs", {})

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=training_config["learning_rate"], **optimizer_kwargs
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_config["learning_rate"], **optimizer_kwargs
        )
    elif optimizer_name == "sgd":
        # SGD typically needs momentum
        if "momentum" not in optimizer_kwargs:
            optimizer_kwargs["momentum"] = 0.9
        optimizer = torch.optim.SGD(
            model.parameters(), lr=training_config["learning_rate"], **optimizer_kwargs
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw, sgd"
        )

    print(f"Optimizer: {optimizer_name}")
    print(f"Learning rate: {training_config['learning_rate']}")

    # Initialize scheduler if specified
    scheduler = None
    scheduler_name = training_config.get("scheduler", "none")
    if scheduler_name and scheduler_name.lower() != "none":
        scheduler_kwargs = training_config.get("scheduler_kwargs", {})

        if scheduler_name.lower() == "cosine":
            # Use epochs as T_max if not specified
            if "T_max" not in scheduler_kwargs:
                scheduler_kwargs["T_max"] = training_config["epochs"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_kwargs
            )
        elif scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
        elif scheduler_name.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_kwargs
            )
        else:
            raise ValueError(
                f"Unknown scheduler: {scheduler_name}. "
                f"Supported: cosine, step, plateau, none"
            )

        print(f"Scheduler: {scheduler_name}")

    # Initialize logger
    logger = ClassifierLogger(log_dir=log_dir, class_names=class_names)

    # Initialize trainer
    trainer = ClassifierTrainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        logger=logger,
        device=device,
        show_progress=True,
    )

    # Set scheduler if available
    if scheduler is not None:
        trainer.scheduler = scheduler

    # Train the model
    num_epochs = training_config["epochs"]
    print(f"\nStarting training for {num_epochs} epochs...")

    try:
        trainer.train(
            num_epochs=num_epochs,
            checkpoint_dir=str(checkpoint_dir),
            save_best=config["output"].get("save_best_only", True),
            checkpoint_frequency=config["output"].get("save_frequency", 10),
            validate_frequency=config.get("validation", {}).get("frequency", 1),
            best_metric=config.get("validation", {}).get("metric", "accuracy"),
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        logger.close()
        sys.exit(0)
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        logger.close()
        raise

    # Close logger
    logger.close()
    print("\nTraining completed successfully!")


def setup_experiment_diffusion(config: Dict[str, Any]) -> None:
    """Setup and run diffusion experiment.

    Args:
        config: Merged configuration dictionary

    Raises:
        ValueError: If configuration is invalid
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

    # Set up device (now in compute section)
    compute_config = config.get("compute", {})
    device_config = compute_config.get("device", "auto")
    if device_config == "auto":
        device = get_device()
    else:
        device = device_config

    print(f"Using device: {device}")

    # Set random seed if specified (now in compute section)
    seed = compute_config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")

    # Create output directories using V2 structure
    log_dir = resolve_output_path(config, "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Log directory: {log_dir}")

    # Save configuration to log directory
    config_save_path = log_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {config_save_path}")

    # Initialize model using V2 config structure
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

    print(f"Model: DDPM")
    print(f"Image size: {arch_config['image_size']}")
    print(f"Num classes: {cond_config['num_classes']}")
    print(f"Num timesteps: {diff_config['num_timesteps']}")

    # Check if in generation mode
    if mode == "generate":
        # Generation mode: load checkpoint and generate samples
        generation_config = config["generation"]
        checkpoint_path = generation_config.get("checkpoint")
        if not checkpoint_path:
            raise ValueError("generation.checkpoint is required for generation mode")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\nLoading checkpoint: {checkpoint_path}")

        # Load model weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        # Initialize dataloader (needed for class info even in generation mode)
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

        # Initialize logger
        logger = DiffusionLogger(log_dir=log_dir)

        # Create dummy optimizer (required by trainer but not used in generation)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Get generation configuration
        sampling_config = generation_config["sampling"]

        # Initialize trainer for generation mode
        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=device,
            show_progress=True,
            use_ema=sampling_config["use_ema"],
            ema_decay=0.9999,  # Not used in generation, but required by constructor
            use_amp=False,  # No AMP in generation mode
            gradient_clip_norm=None,
            sample_images=False,  # Not used in generation mode
            sample_interval=1,
            samples_per_class=2,  # Will be overridden by generate_samples call
            guidance_scale=sampling_config["guidance_scale"],
        )

        # Get generation parameters
        sampling_config = generation_config["sampling"]
        output_config = generation_config["output"]

        num_samples = sampling_config["num_samples"]
        print(f"\nGenerating {num_samples} samples...")

        # Prepare class labels if conditional generation
        class_labels = None
        if cond_config["num_classes"] is not None:
            # Generate balanced samples across all classes
            samples_per_class = num_samples // cond_config["num_classes"]
            remainder = num_samples % cond_config["num_classes"]
            class_labels = []
            for i in range(cond_config["num_classes"]):
                count = samples_per_class + (1 if i < remainder else 0)
                class_labels.extend([i] * count)
            class_labels = torch.tensor(class_labels, device=device)

        # Generate samples with V2 config
        samples = trainer.generate_samples(
            num_samples=num_samples,
            class_labels=class_labels,
            guidance_scale=sampling_config["guidance_scale"],
            use_ema=sampling_config["use_ema"],
        )

        # Save generated samples to configured generated directory
        output_dir = resolve_output_path(config, "generated")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save outputs according to V2 config
        from torchvision.utils import save_image

        if output_config["save_grid"]:
            grid_nrow = output_config["grid_nrow"]
            save_image(
                samples,
                output_dir / "generated_samples.png",
                nrow=grid_nrow,
                normalize=True,
            )
            print(f"Saved generated grid to: {output_dir / 'generated_samples.png'}")

        # Save individual samples if configured
        if output_config["save_individual"]:
            for i, sample in enumerate(samples):
                save_image(sample, output_dir / f"sample_{i:04d}.png", normalize=True)
            print(f"Saved {len(samples)} individual samples to: {output_dir}")

        logger.close()
        print("\nGeneration completed successfully!")

    else:
        # Training mode
        training_config = config["training"]

        # Create checkpoint directory from output.subdirs
        checkpoint_dir = resolve_output_path(config, "checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")

        # Initialize dataloader with V2 config
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
            print("Enabled TF32 for faster training on Ampere+ GPUs")

        # Enable cuDNN benchmark mode
        if (
            performance_config.get("cudnn_benchmark", True)
            and torch.cuda.is_available()
        ):
            torch.backends.cudnn.benchmark = True
            print("Enabled cuDNN benchmark mode")

        # Compile model (PyTorch 2.0+)
        if performance_config.get("compile_model", False):
            try:
                model = torch.compile(model)
                print("Compiled model with torch.compile()")
            except Exception as e:
                print(f"Warning: Failed to compile model: {e}")

        # Initialize logger
        logger = DiffusionLogger(log_dir=log_dir)

        # Initialize optimizer from V2 config
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

        print(f"Optimizer: {optimizer_name}")
        print(f"Learning rate: {optimizer_config['learning_rate']}")

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

            print(f"Scheduler: {scheduler_name}")

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
            logger=logger,
            device=device,
            show_progress=True,
            use_ema=ema_config["enabled"],
            ema_decay=ema_config["decay"],
            use_amp=performance_config["use_amp"],
            gradient_clip_norm=optimizer_config.get("gradient_clip_norm"),
            scheduler=scheduler,
            sample_images=visualization_config["enabled"],
            sample_interval=visualization_config["interval"],
            samples_per_class=2,  # Will calculate based on num_samples internally
            guidance_scale=visualization_config["guidance_scale"],
        )

        # Train the model
        num_epochs = training_config["epochs"]
        print(f"\nStarting training for {num_epochs} epochs...")

        try:
            trainer.train(
                num_epochs=num_epochs,
                checkpoint_dir=str(checkpoint_dir),
                save_best=checkpointing_config["save_best_only"],
                checkpoint_frequency=checkpointing_config["save_frequency"],
                validate_frequency=validation_config["frequency"],
                best_metric=validation_config["metric"],
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            logger.close()
            sys.exit(0)
        except Exception as e:
            print(f"\n\nTraining failed with error: {e}")
            logger.close()
            raise

        # Close logger
        logger.close()
        print("\nTraining completed successfully!")


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

    print("=" * 60)
    print(f"AI Image Generation and Statistics Framework")
    print("=" * 60)
    print(f"Experiment: {experiment}")
    print(f"Mode: {config.get('mode', 'train')}")
    print("=" * 60)

    # Dispatch to experiment
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


if __name__ == "__main__":
    main()
