#!/usr/bin/env python3
"""Example script to generate samples using a trained DDPM model."""

import argparse

from ddpm import generate


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples using a trained DDPM model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./out/ddpm/ddpm_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale (0.0 = no guidance, 3.0-7.0 typical)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes in the model",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help="Generate all samples from a specific class (0, 1, etc.). If not set, generates equal samples for each class.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=40,
        help="Image size (assumes square images)",
    )
    parser.add_argument(
        "--model_channels",
        type=int,
        default=64,
        help="Base number of channels in U-Net",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out/ddpm/samples",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run generation on",
    )

    args = parser.parse_args()

    # Prepare class labels if specific class is requested
    class_labels = None
    if args.class_label is not None:
        if args.class_label < 0 or args.class_label >= args.num_classes:
            raise ValueError(
                f"class_label must be between 0 and {args.num_classes - 1}"
            )
        class_labels = [args.class_label] * args.num_samples
        print(f"Generating {args.num_samples} samples for class {args.class_label}")
    else:
        print(
            f"Generating {args.num_samples} samples (balanced across {args.num_classes} classes)"
        )

    # Generate samples
    samples = generate(
        model_path=args.model_path,
        num_samples=args.num_samples,
        class_labels=class_labels,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        num_classes=args.num_classes,
        model_channels=args.model_channels,
        num_timesteps=args.num_timesteps,
        out_dir=args.out_dir,
        save_images=True,
        device=args.device,
    )

    print("\n=== Generation Complete ===")
    print(f"Generated {samples.shape[0]} samples")
    print(f"Images saved to {args.out_dir}")


if __name__ == "__main__":
    main()
