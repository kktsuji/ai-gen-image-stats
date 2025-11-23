"""DDPM Training Script

Training script for DDPM models with class-conditional generation support.
"""

import argparse
import csv
import os
import random
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from ddpm import DDPM, EMA, create_ddpm
from util import save_args


def train(
    epochs: int = 200,
    batch_size: int = 8,
    learning_rate: float = 0.00005,
    num_classes: int = 2,
    img_size: int = 40,
    num_timesteps: int = 1000,
    model_channels: int = 64,
    channel_multipliers: Tuple[int, ...] = (1, 2, 4),
    beta_schedule: str = "cosine",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    class_dropout_prob: float = 0.3,
    use_weighted_sampling: bool = True,
    train_data_path: str = "./data/train",
    val_data_path: str = "./data/val",
    out_dir: str = "./out/ddpm",
    num_workers: int = 4,
    seed: Optional[int] = None,
    use_attention: Tuple[bool, ...] = (False, False, True),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Training function for DDPM."""

    # Set random seeds for reproducibility
    if seed is not None:
        print(f"\nSetting random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Enable cuDNN benchmark for performance (fixed input size)
            torch.backends.cudnn.benchmark = True
            # Only use deterministic for strict reproducibility (slower)
            # torch.backends.cudnn.deterministic = True

    # Data transforms (normalize to [-1, 1] for DDPM)
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Scale to [-1, 1]
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),  # Scale to [-1, 1]
        ]
    )

    print("\nLoading datasets...")

    # Training dataset
    train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)

    # Calculate class distribution
    class_counts = [0] * num_classes
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    print(f"Training set: {len(train_dataset)} images")
    print(f"  - Class distribution:")
    for idx, (class_name, count) in enumerate(zip(train_dataset.classes, class_counts)):
        print(
            f"    - {class_name}: {count} images ({count/len(train_dataset)*100:.2f}%)"
        )

    # Setup weighted sampling if enabled
    if use_weighted_sampling:
        print(f"\n  - Weighted sampling: ENABLED")
        # Calculate weights for each class (inverse frequency)
        num_samples = sum(class_counts)
        class_weights = [num_samples / count for count in class_counts]
        print(f"    - Class weights: {[f'{w:.3f}' for w in class_weights]}")

        # Assign weight to each sample based on its class
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )

        # Use sampler (don't use shuffle with sampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        print(f"\n  - Weighted sampling: DISABLED (using random sampling)")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    print(f"  - Number of batches: {len(train_loader)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Classes: {train_dataset.classes}")

    # Validation dataset
    val_dataset = datasets.ImageFolder(val_data_path, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    print(f"\nValidation set: {len(val_dataset)} images")
    print(f"  - Number of batches: {len(val_loader)}")
    print(f"  - Classes: {val_dataset.classes}")

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation datasets have different classes.")

    if len(train_dataset.classes) != num_classes:
        raise ValueError(
            f"Expected {num_classes} classes, but found {len(train_dataset.classes)}"
        )

    # Create class-conditional DDPM model
    print("\n=== Creating Class-Conditional DDPM ===")
    print(f"Using beta schedule: {beta_schedule}")
    model = create_ddpm(
        image_size=img_size,
        in_channels=3,
        model_channels=model_channels,
        channel_multipliers=channel_multipliers,
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        class_dropout_prob=class_dropout_prob,
        use_attention=use_attention,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup EMA for improved sampling quality
    print("\n=== Setting up EMA ===")
    ema = EMA(model, decay=0.9999, device=device)
    print(f"EMA decay rate: 0.9999")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    print("\n=== Starting Training ===")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            loss = model.training_step(images, class_labels=labels, criterion=criterion)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update EMA
            ema.update()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # Validation phase (use EMA weights)
        model.eval()
        ema.apply_shadow()  # Switch to EMA weights for validation

        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                predicted_noise, noise = model(images, class_labels=labels)
                loss = criterion(predicted_noise, noise)

                val_loss += loss.item()
                val_batches += 1

        ema.restore()  # Restore training weights

        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 20 == 0:
            # Save both regular and EMA weights
            torch.save(model.state_dict(), f"{out_dir}/ddpm_epoch{epoch+1}.pth")
            ema.apply_shadow()
            torch.save(model.state_dict(), f"{out_dir}/ddpm_epoch{epoch+1}_ema.pth")
            ema.restore()

        epoch_end_time = time.time()
        epoch_elapsed = epoch_end_time - epoch_start_time
        epoch_minutes = int(epoch_elapsed // 60)
        epoch_seconds = int(epoch_elapsed % 60)

        # Calculate remaining time estimate
        remaining_epochs = epochs - (epoch + 1)
        estimated_remaining = epoch_elapsed * remaining_epochs
        remaining_hours = int(estimated_remaining // 3600)
        remaining_minutes = int((estimated_remaining % 3600) // 60)
        remaining_seconds = int(estimated_remaining % 60)

        print(
            f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Time: {epoch_minutes:02d}:{epoch_seconds:02d}, "
            f"Remaining Time: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}",
            end="\r",
        )

    print("\n\n=== Training Completed ===")

    # Save model (both regular and EMA weights)
    model_path = f"{out_dir}/ddpm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save EMA model
    ema.apply_shadow()
    ema_model_path = f"{out_dir}/ddpm_model_ema.pth"
    torch.save(model.state_dict(), ema_model_path)
    print(f"EMA model saved to {ema_model_path}")
    ema.restore()

    # Save EMA state dict separately for resuming training
    ema_state_path = f"{out_dir}/ema_state.pth"
    torch.save(ema.state_dict(), ema_state_path)
    print(f"EMA state saved to {ema_state_path}")

    # Save training history to CSV
    history_path = f"{out_dir}/training_history.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, (train_loss, val_loss) in enumerate(
            zip(train_losses, val_losses), 1
        ):
            writer.writerow([epoch, train_loss, val_loss])
    print(f"Training history saved to {history_path}")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot loss (log scale) if useful
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training and Validation Loss (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = f"{out_dir}/training_curves.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM Training")
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.00005, help="Learning rate"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (0: Normal, 1: Abnormal)",
    )
    parser.add_argument("--img-size", type=int, default=40, help="Image size (square)")
    parser.add_argument(
        "--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--model-channels",
        type=int,
        default=64,
        help="Base number of channels in U-Net",
    )
    parser.add_argument(
        "--channel-multipliers",
        type=str,
        default="1,2,4",
        help="Comma-separated channel multipliers for each U-Net stage (e.g., '1,2,4' for 3 stages, '1,2,2,4,8' for 5 stages)",
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "quadratic", "sigmoid"],
        help="Type of noise schedule",
    )
    parser.add_argument(
        "--beta-start", type=float, default=0.0001, help="Starting beta value"
    )
    parser.add_argument(
        "--beta-end", type=float, default=0.02, help="Ending beta value"
    )
    parser.add_argument(
        "--class-dropout-prob",
        type=float,
        default=0.3,
        help="Class dropout probability for classifier-free guidance",
    )
    parser.add_argument(
        "--use-weighted-sampling",
        action="store_true",
        default=True,
        help="Enable weighted sampling for class imbalance",
    )
    parser.add_argument(
        "--train-data-path", type=str, default="./data/stats", help="Training data path"
    )
    parser.add_argument(
        "--val-data-path", type=str, default="./data/stats", help="Validation data path"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./out/ddpm", help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers for parallel data loading (default: 4)",
    )
    parser.add_argument(
        "--use-attention",
        type=str,
        default="0,0,1",
        help="Comma-separated boolean values (0 or 1) specifying which U-Net resolution levels should use attention layers. "
        "Format: 'level1,level2,level3' where 1=enabled, 0=disabled. "
        "Default '0,0,1' enables attention only at the coarsest resolution for better feature capture with minimal overhead. "
        "Example: '1,1,1' enables attention at all levels (more computation, potentially better quality).",
    )

    args = parser.parse_args()

    # Print all arguments
    print("\nDDPM Training Script Started")
    print("\n=== Arguments ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Save arguments
    save_args(args, args.out_dir)

    # Check if output directory exists
    if not os.path.exists(args.out_dir):
        print(f"\nError: Output directory '{args.out_dir}' does not exist.")
        print("Please create the directory first or specify a valid output directory.")
        exit(1)

    use_attention = tuple(bool(int(x)) for x in args.use_attention.split(","))
    channel_multipliers = tuple(int(x) for x in args.channel_multipliers.split(","))

    start_time = time.time()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        img_size=args.img_size,
        num_timesteps=args.num_timesteps,
        model_channels=args.model_channels,
        channel_multipliers=channel_multipliers,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        class_dropout_prob=args.class_dropout_prob,
        use_weighted_sampling=args.use_weighted_sampling,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        use_attention=use_attention,
        seed=args.seed,
        device=device,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(
        f"\nTotal execution time for training: {hours:02d}:{minutes:02d}:{seconds:02d}"
    )
