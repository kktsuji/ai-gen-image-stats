"""
Test script to verify weighted sampling implementation.
This script tests that the weighted sampling correctly balances classes.
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def test_weighted_sampling():
    """Test that weighted sampling balances the classes in batches."""

    # Configuration
    BATCH_SIZE = 8
    NUM_BATCHES_TO_TEST = 100
    IMG_SIZE = 40

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load dataset
    train_data_path = "./data/stats"
    train_dataset = datasets.ImageFolder(train_data_path, transform=transform)

    # Calculate class distribution
    class_counts = [0, 0]
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    print("=" * 60)
    print("ORIGINAL DATASET DISTRIBUTION")
    print("=" * 60)
    print(f"Total samples: {len(train_dataset)}")
    for idx, (class_name, count) in enumerate(zip(train_dataset.classes, class_counts)):
        percentage = count / len(train_dataset) * 100
        print(f"  {class_name}: {count} samples ({percentage:.2f}%)")

    imbalance_ratio = max(class_counts) / min(class_counts)
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

    # Test 1: Random sampling (baseline)
    print("\n" + "=" * 60)
    print("TEST 1: RANDOM SAMPLING (Baseline)")
    print("=" * 60)

    random_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    random_class_distribution = Counter()

    for i, (_, labels) in enumerate(random_loader):
        if i >= NUM_BATCHES_TO_TEST:
            break
        for label in labels:
            random_class_distribution[label.item()] += 1

    total_random_samples = sum(random_class_distribution.values())
    print(f"Samples seen in {NUM_BATCHES_TO_TEST} batches: {total_random_samples}")
    for class_idx, class_name in enumerate(train_dataset.classes):
        count = random_class_distribution.get(class_idx, 0)
        percentage = count / total_random_samples * 100
        print(f"  {class_name}: {count} samples ({percentage:.2f}%)")

    # Test 2: Weighted sampling
    print("\n" + "=" * 60)
    print("TEST 2: WEIGHTED SAMPLING")
    print("=" * 60)

    # Calculate weights
    num_samples = sum(class_counts)
    class_weights = [num_samples / count for count in class_counts]
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights]}")

    # Assign weight to each sample
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    weighted_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    weighted_class_distribution = Counter()

    for i, (_, labels) in enumerate(weighted_loader):
        if i >= NUM_BATCHES_TO_TEST:
            break
        for label in labels:
            weighted_class_distribution[label.item()] += 1

    total_weighted_samples = sum(weighted_class_distribution.values())
    print(f"\nSamples seen in {NUM_BATCHES_TO_TEST} batches: {total_weighted_samples}")
    for class_idx, class_name in enumerate(train_dataset.classes):
        count = weighted_class_distribution.get(class_idx, 0)
        percentage = count / total_weighted_samples * 100
        print(f"  {class_name}: {count} samples ({percentage:.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    random_percentages = [
        random_class_distribution.get(i, 0) / total_random_samples * 100
        for i in range(len(train_dataset.classes))
    ]
    weighted_percentages = [
        weighted_class_distribution.get(i, 0) / total_weighted_samples * 100
        for i in range(len(train_dataset.classes))
    ]

    print("\nClass distribution comparison:")
    print(f"{'Class':<15} {'Original':<15} {'Random':<15} {'Weighted':<15}")
    print("-" * 60)
    for idx, class_name in enumerate(train_dataset.classes):
        original_pct = class_counts[idx] / len(train_dataset) * 100
        print(
            f"{class_name:<15} {original_pct:>6.2f}%        {random_percentages[idx]:>6.2f}%        {weighted_percentages[idx]:>6.2f}%"
        )

    print("\n" + "=" * 60)

    # Check if weighted sampling is more balanced
    random_diff = abs(random_percentages[0] - random_percentages[1])
    weighted_diff = abs(weighted_percentages[0] - weighted_percentages[1])

    print(f"\nClass balance (difference between class percentages):")
    print(f"  Random sampling: {random_diff:.2f}%")
    print(f"  Weighted sampling: {weighted_diff:.2f}%")

    if weighted_diff < random_diff:
        print(f"\n✅ SUCCESS: Weighted sampling is more balanced!")
        print(f"   Improvement: {random_diff - weighted_diff:.2f}% better balance")
    else:
        print(f"\n⚠️  WARNING: Weighted sampling did not improve balance")

    print("=" * 60)


if __name__ == "__main__":
    test_weighted_sampling()
    test_weighted_sampling()
