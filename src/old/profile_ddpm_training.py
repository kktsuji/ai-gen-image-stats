"""Profile DDPM Training to Identify Bottlenecks

This script profiles different components of DDPM training:
- Data loading (I/O)
- Forward pass (GPU compute)
- Backward pass (GPU compute)
- Optimizer step
- Memory transfer (CPU<->GPU)
"""

# Import DDPM components
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, "src")
from ddpm import create_ddpm


def profile_training():
    """Profile one epoch of training to identify bottlenecks."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Training configuration (from your script)
    batch_size = 8
    img_size = 40
    num_classes = 2
    model_channels = 64
    channel_multipliers = (1, 2, 4)
    num_timesteps = 1000
    num_workers = 4

    # Setup data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    print("\n=== Loading Dataset ===")
    dataset = datasets.ImageFolder("./data", transform=train_transform)
    print(f"Total images: {len(dataset)}")

    # Calculate batches per epoch
    batches_per_epoch = len(dataset) // batch_size
    print(f"Batches per epoch: {batches_per_epoch}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print("\n=== Creating Model ===")
    model = create_ddpm(
        image_size=img_size,
        in_channels=3,
        model_channels=model_channels,
        channel_multipliers=channel_multipliers,
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        beta_schedule="cosine",
        class_dropout_prob=0.3,
        use_attention=(False, False, True),
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    criterion = nn.MSELoss()

    # Warmup GPU
    print("\n=== Warming up GPU ===")
    dummy_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    for _ in range(3):
        loss = model.training_step(
            dummy_images, class_labels=dummy_labels, criterion=criterion
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize() if device == "cuda" else None

    # Profile training components
    print("\n=== Profiling Training Components ===")
    model.train()

    # Timing storage
    data_load_times = []
    h2d_transfer_times = []  # Host to Device
    forward_times = []
    backward_times = []
    optimizer_times = []
    total_batch_times = []

    num_profile_batches = min(20, batches_per_epoch)  # Profile first 20 batches
    print(f"Profiling {num_profile_batches} batches...\n")

    batch_start_time = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= num_profile_batches:
            break

        # 1. Data loading time (already measured by iterator)
        data_load_end = time.time()
        data_load_time = data_load_end - batch_start_time
        data_load_times.append(data_load_time)

        # 2. Host to Device transfer
        h2d_start = time.time()
        images = images.to(device)
        labels = labels.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        h2d_end = time.time()
        h2d_transfer_times.append(h2d_end - h2d_start)

        # 3. Forward pass
        optimizer.zero_grad()
        forward_start = time.time()
        loss = model.training_step(images, class_labels=labels, criterion=criterion)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_end = time.time()
        forward_times.append(forward_end - forward_start)

        # 4. Backward pass
        backward_start = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if device == "cuda":
            torch.cuda.synchronize()
        backward_end = time.time()
        backward_times.append(backward_end - backward_start)

        # 5. Optimizer step
        optimizer_start = time.time()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        optimizer_end = time.time()
        optimizer_times.append(optimizer_end - optimizer_start)

        # Total batch time
        batch_end = time.time()
        total_batch_times.append(batch_end - batch_start_time)

        # Print progress
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx + 1}/{num_profile_batches} completed")

        batch_start_time = time.time()

    # Calculate statistics
    print("\n" + "=" * 70)
    print("PROFILING RESULTS (Average per batch)")
    print("=" * 70)

    avg_data_load = np.mean(data_load_times) * 1000
    avg_h2d = np.mean(h2d_transfer_times) * 1000
    avg_forward = np.mean(forward_times) * 1000
    avg_backward = np.mean(backward_times) * 1000
    avg_optimizer = np.mean(optimizer_times) * 1000
    avg_total = np.mean(total_batch_times) * 1000

    print(
        f"\n1. Data Loading (I/O):           {avg_data_load:7.2f} ms  ({avg_data_load/avg_total*100:5.1f}%)"
    )
    print(
        f"2. CPU->GPU Transfer:            {avg_h2d:7.2f} ms  ({avg_h2d/avg_total*100:5.1f}%)"
    )
    print(
        f"3. Forward Pass (GPU Compute):   {avg_forward:7.2f} ms  ({avg_forward/avg_total*100:5.1f}%)"
    )
    print(
        f"4. Backward Pass (GPU Compute):  {avg_backward:7.2f} ms  ({avg_backward/avg_total*100:5.1f}%)"
    )
    print(
        f"5. Optimizer Step:               {avg_optimizer:7.2f} ms  ({avg_optimizer/avg_total*100:5.1f}%)"
    )
    print(f"{'-'*70}")
    print(f"Total per batch:                 {avg_total:7.2f} ms")

    # Calculate epoch time
    epoch_time = avg_total * batches_per_epoch / 1000
    print(f"\nEstimated time per epoch: {epoch_time:.2f} seconds")

    # Bottleneck analysis
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)

    components = {
        "Data Loading (I/O)": avg_data_load,
        "CPU->GPU Transfer": avg_h2d,
        "Forward Pass": avg_forward,
        "Backward Pass": avg_backward,
        "Optimizer Step": avg_optimizer,
    }

    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)

    print("\nComponent ranking (slowest to fastest):")
    for i, (name, time_ms) in enumerate(sorted_components, 1):
        print(f"{i}. {name:30s} {time_ms:7.2f} ms  ({time_ms/avg_total*100:5.1f}%)")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    bottleneck = sorted_components[0]
    bottleneck_name, bottleneck_time = bottleneck

    if "Data Loading" in bottleneck_name:
        print("\n🔴 PRIMARY BOTTLENECK: Data Loading (I/O)")
        print("\nSuggestions:")
        print("  - Increase num_workers (currently: 4)")
        print("  - Use SSD storage if on HDD")
        print("  - Reduce data augmentation complexity")
        print("  - Cache preprocessed images in memory")

    elif "Transfer" in bottleneck_name:
        print("\n🔴 PRIMARY BOTTLENECK: CPU->GPU Memory Transfer")
        print("\nSuggestions:")
        print("  - Increase batch size to amortize transfer overhead")
        print("  - Ensure pin_memory=True (already enabled)")
        print("  - Check PCIe bandwidth")

    elif "Forward Pass" in bottleneck_name or "Backward Pass" in bottleneck_name:
        print("\n🔴 PRIMARY BOTTLENECK: GPU Computation")
        print("\nSuggestions:")
        print("  - Use mixed precision training (AMP) - already enabled")
        print("  - Reduce model size (fewer channels/layers)")
        print("  - Increase batch size for better GPU utilization")
        print("  - Check GPU memory usage")

    elif "Optimizer" in bottleneck_name:
        print("\n🔴 PRIMARY BOTTLENECK: Optimizer Step")
        print("\nSuggestions:")
        print("  - Try different optimizer (AdamW, SGD)")
        print("  - Use fused optimizer implementations")

    # GPU utilization check
    if device == "cuda":
        print("\n" + "=" * 70)
        print("GPU UTILIZATION")
        print("=" * 70)
        print(
            "\nCompute utilization: {:.1f}%".format(
                (avg_forward + avg_backward) / avg_total * 100
            )
        )
        print(
            "I/O overhead: {:.1f}%".format((avg_data_load + avg_h2d) / avg_total * 100)
        )

        if (avg_data_load + avg_h2d) / avg_total > 0.3:
            print("\n⚠️  WARNING: GPU is idle >30% of the time waiting for data!")
            print("   This indicates an I/O bottleneck.")
        else:
            print("\n✅ GPU utilization is good (low I/O overhead)")


if __name__ == "__main__":
    profile_training()
