"""Main entry point for AI Image Generation and Statistics Framework.

This module provides the CLI entry point for running different experiments
(classifier, diffusion, GAN) with a unified interface. It dispatches to the
appropriate experiment implementation based on the configuration.

Usage:
    # Train with config file
    python -m src.main --experiment classifier --config configs/classifier/baseline.json

    # Train with CLI arguments only
    python -m src.main --experiment classifier --model inceptionv3 --epochs 10

    # Train with config + CLI overrides
    python -m src.main --experiment classifier --config configs/classifier/baseline.json --batch-size 32

    # Generate synthetic data (for diffusion experiments, to be implemented)
    python -m src.main --experiment diffusion --mode generate --checkpoint path/to/model.pth
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

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
    from src.experiments.classifier.config import get_default_config
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
    from src.utils.config import merge_configs

    # Merge with classifier defaults
    classifier_defaults = get_default_config()
    config = merge_configs(classifier_defaults, config)

    # Validate classifier config
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
        NotImplementedError: This experiment type is not yet implemented
    """
    raise NotImplementedError(
        "Diffusion experiment is not yet implemented. "
        "This will be added in a future refactoring step."
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
