#!/usr/bin/env python3
"""Quick example demonstrating the generate() function."""

import os

import torch

from ddpm import create_ddpm, generate


def main():
    """Quick example of training and generating samples."""

    print("=" * 60)
    print("DDPM Generate Function - Quick Example")
    print("=" * 60)

    # Step 1: Create a small model for demonstration
    print("\n[Step 1] Creating a small DDPM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_ddpm(
        image_size=40,
        model_channels=32,  # Smaller for quick demo
        num_classes=2,
        num_timesteps=10,  # Fewer timesteps for quick demo
        device=device,
    )
    print(f"  Model created on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 2: Save the model
    print("\n[Step 2] Saving model...")
    os.makedirs("./out/ddpm", exist_ok=True)
    model_path = "./out/ddpm/example_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved to {model_path}")

    # Step 3: Generate samples using the saved model
    print("\n[Step 3] Generating samples...")
    print("  Configuration:")
    print("    - Number of samples: 8")
    print("    - Guidance scale: 3.0")
    print("    - Balanced across 2 classes")

    samples = generate(
        model_path=model_path,
        num_samples=8,
        class_labels=None,  # Auto-balance across classes
        guidance_scale=3.0,
        image_size=40,
        num_classes=2,
        model_channels=32,
        num_timesteps=10,
        out_dir="./out/ddpm/example_samples",
        save_images=True,
        device=device,
    )

    print(f"\n  ✓ Generated {samples.shape[0]} samples")
    print(f"    Shape: {samples.shape}")
    print(f"    Range: [{samples.min().item():.3f}, {samples.max().item():.3f}]")
    print(f"    Saved to: ./out/ddpm/example_samples/")

    # Step 4: Generate class-specific samples
    print("\n[Step 4] Generating class-specific samples...")
    print("  Generating 4 samples from class 0 (Normal)")

    samples_class0 = generate(
        model_path=model_path,
        num_samples=4,
        class_labels=[0, 0, 0, 0],  # All from class 0
        guidance_scale=5.0,  # Higher guidance for stronger conditioning
        image_size=40,
        num_classes=2,
        model_channels=32,
        num_timesteps=10,
        out_dir="./out/ddpm/example_samples_class0",
        save_images=True,
        device=device,
    )

    print(f"  ✓ Generated {samples_class0.shape[0]} samples from class 0")
    print(f"    Saved to: ./out/ddpm/example_samples_class0/")

    # Step 5: Show different guidance scales
    print("\n[Step 5] Comparing different guidance scales...")
    for guidance in [0.0, 3.0, 7.0]:
        samples_guidance = generate(
            model_path=model_path,
            num_samples=4,
            class_labels=[0, 0, 1, 1],
            guidance_scale=guidance,
            image_size=40,
            num_classes=2,
            model_channels=32,
            num_timesteps=10,
            out_dir=f"./out/ddpm/example_guidance_{guidance}",
            save_images=True,
            device=device,
        )
        print(f"  ✓ Guidance {guidance}: {samples_guidance.shape[0]} samples")

    print("\n" + "=" * 60)
    print("Example complete! Check ./out/ddpm/ for generated images.")
    print("=" * 60)
    print("\nNote: This example uses a small untrained model for demonstration.")
    print("For production use:")
    print("  1. Train the model: python ddpm.py")
    print("  2. Generate samples: python generate_samples.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
