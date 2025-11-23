"""DDPM Generation Script

Generation script for DDPM models with class-conditional sampling support.
"""

import argparse
import os
import random
import time
from typing import Optional

import numpy as np
import torch
from torchvision.utils import save_image

from ddpm import create_ddpm
from util import save_args


def generate(
    model_path: str,
    num_samples: int = 16,
    batch_size: int = 16,
    class_labels: Optional[list] = None,
    guidance_scale: float = 3.0,
    image_size: int = 40,
    num_classes: int = 2,
    model_channels: int = 64,
    num_timesteps: int = 1000,
    beta_schedule: str = "linear",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    out_dir: str = "./out/ddpm/samples",
    save_images: bool = True,
    use_dynamic_threshold: bool = True,
    dynamic_threshold_percentile: float = 0.995,
    use_attention: tuple = (False, False, True),
    seed: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Generate samples using a trained DDPM model.

    Args:
        model_path: Path to the trained model checkpoint (.pth file)
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation (for memory efficiency, will generate in batches)
        class_labels: List of class labels for conditional generation.
                     If None, generates equal samples for each class.
                     Length should match num_samples.
        guidance_scale: Classifier-free guidance scale (0.0 = no guidance, 3.0-7.0 typical)
        image_size: Size of the images (assumes square)
        num_classes: Number of classes in the model
        model_channels: Base number of channels in U-Net
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Type of noise schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
        beta_start: Starting beta value for noise schedule
        beta_end: Ending beta value for noise schedule
        out_dir: Directory to save generated images
        save_images: Whether to save images to disk
        use_dynamic_threshold: Whether to apply dynamic thresholding (default: True)
        dynamic_threshold_percentile: Percentile for dynamic thresholding (default: 0.995)
        seed: Random seed for reproducibility (default: None)
        device: Device to run generation on
    """
    # Set random seeds for reproducibility
    if seed is not None:
        print(f"\nSetting random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    print(f"Loading model from {model_path}...")

    # Create model
    model = create_ddpm(
        image_size=image_size,
        in_channels=3,
        model_channels=model_channels,
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        use_attention=use_attention,
        device=device,
    )

    # Load trained weights (preferably EMA weights if available)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model loaded successfully on {device}")
    print(f"Generating {num_samples} samples with guidance_scale={guidance_scale}...")
    print(f"Dynamic thresholding: {'enabled' if use_dynamic_threshold else 'disabled'}")

    # Prepare class labels
    if class_labels is None:
        # Generate equal samples for each class
        samples_per_class = num_samples // num_classes
        remainder = num_samples % num_classes
        class_labels = []
        for class_idx in range(num_classes):
            count = samples_per_class + (1 if class_idx < remainder else 0)
            class_labels.extend([class_idx] * count)

    if len(class_labels) != num_samples:
        raise ValueError(
            f"Length of class_labels ({len(class_labels)}) must match num_samples ({num_samples})"
        )

    # Convert to tensor
    class_labels_tensor = torch.tensor(class_labels, device=device, dtype=torch.long)

    # Generate samples in batches
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

    print(f"\nGenerating in {num_batches} batch(es) of size {batch_size}...")

    if save_images:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx

            # Get class labels for this batch
            batch_class_labels = class_labels_tensor[start_idx:end_idx]

            # Generate batch with dynamic thresholding
            batch_samples = model.sample(
                batch_size=current_batch_size,
                class_labels=batch_class_labels,
                guidance_scale=guidance_scale,
                use_dynamic_threshold=use_dynamic_threshold,
                dynamic_threshold_percentile=dynamic_threshold_percentile,
            )

            # Save images immediately after generating this batch
            if save_images:
                batch_labels = class_labels[start_idx:end_idx]
                for i, (sample, label) in enumerate(zip(batch_samples, batch_labels)):
                    sample_idx = start_idx + i
                    sample_path = os.path.join(
                        out_dir,
                        f"sample_{sample_idx:03d}_class{label}_guidance{guidance_scale}.png",
                    )
                    sample_normalized = (sample + 1.0) / 2.0
                    save_image(sample_normalized, sample_path, normalize=False)

            print(
                f"  - Batch {batch_idx + 1}/{num_batches}: Generated {current_batch_size} samples",
            )
    print("\nGeneration completed.")

    if save_images:
        print(f"  Saved {num_samples} individual images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM Generation")
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Batch size for generation"
    )
    parser.add_argument(
        "--class-label",
        type=int,
        default=1,
        help="Class label to generate (0: Normal, 1: Abnormal)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument("--img-size", type=int, default=40, help="Image size (square)")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes",
    )
    parser.add_argument(
        "--model-channels",
        type=int,
        default=64,
        help="Base number of channels in U-Net",
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "quadratic", "sigmoid"],
        help="Type of noise schedule",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./out/ddpm/ddpm_model_ema.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./out/ddpm/samples",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--use-dynamic-threshold",
        action="store_true",
        default=True,
        help="Enable dynamic thresholding",
    )
    parser.add_argument(
        "--use-attention",
        type=str,
        default="0,0,1",
        help="Comma-separated boolean values (0 or 1) for attention layers (must match training config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()

    # Print all arguments
    print("\nDDPM Generation Script Started")
    print("\n=== Arguments ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Save arguments
    save_args(args, args.out_dir)

    # Check if model and output directory exist
    if not os.path.exists(args.model_path):
        print(f"\nError: Model path '{args.model_path}' does not exist.")
        print("Please provide a valid model checkpoint path.")
        exit(1)
    if not os.path.exists(args.out_dir):
        print(f"\nError: Output directory '{args.out_dir}' does not exist.")
        print("Please create the directory first or specify a valid output directory.")
        exit(1)

    use_attention = tuple(bool(int(x)) for x in args.use_attention.split(","))

    start_time = time.time()

    generate(
        model_path=args.model_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        class_labels=[args.class_label] * args.num_samples,
        guidance_scale=args.guidance_scale,
        image_size=args.img_size,
        num_classes=args.num_classes,
        model_channels=args.model_channels,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        out_dir=args.out_dir,
        save_images=True,
        use_dynamic_threshold=args.use_dynamic_threshold,
        dynamic_threshold_percentile=0.995,
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
        f"\nTotal execution time for generation: {hours:02d}:{minutes:02d}:{seconds:02d}"
    )
    time_per_sample = elapsed_time / args.num_samples
    minutes_per_sample = int(time_per_sample // 60)
    seconds_per_sample = time_per_sample % 60
    print(f"Time per sample: {minutes_per_sample:02d}:{seconds_per_sample:05.2f}")
