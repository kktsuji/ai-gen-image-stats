import csv
import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inception_v3 import InceptionV3FeatureTrainer


class UnderSampledImageFolder(datasets.ImageFolder):
    """ImageFolder with automatic under-sampling to balance classes"""

    def __init__(
        self, root, transform=None, target_transform=None, min_samples_per_class=None
    ):
        super().__init__(root, transform, target_transform)
        self.original_samples = self.samples.copy()
        self.original_targets = self.targets.copy()

        # Calculate class distribution
        class_counts = {}
        for _, class_idx in self.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        # Determine minimum samples per class
        if min_samples_per_class is None:
            min_samples_per_class = min(class_counts.values())
        else:
            # Ensure we don't exceed available samples for any class
            min_samples_per_class = min(
                min_samples_per_class, min(class_counts.values())
            )

        print(
            f"Original class distribution: {[class_counts[i] for i in range(len(self.classes))]}"
        )
        print(f"Under-sampling to {min_samples_per_class} samples per class")

        # Under-sample each class
        self._undersample_classes(min_samples_per_class)

        # Update class distribution for verification
        new_class_counts = {}
        for _, class_idx in self.samples:
            new_class_counts[class_idx] = new_class_counts.get(class_idx, 0) + 1
        print(
            f"After under-sampling: {[new_class_counts[i] for i in range(len(self.classes))]}"
        )

    def _undersample_classes(self, min_samples_per_class):
        """Under-sample each class to have exactly min_samples_per_class samples"""

        # Group samples by class
        samples_by_class = {}
        for i, (path, class_idx) in enumerate(self.samples):
            if class_idx not in samples_by_class:
                samples_by_class[class_idx] = []
            samples_by_class[class_idx].append((path, class_idx))

        # Under-sample each class
        new_samples = []
        new_targets = []

        for class_idx in sorted(samples_by_class.keys()):
            class_samples = samples_by_class[class_idx]
            # Randomly sample min_samples_per_class from this class
            random.shuffle(class_samples)
            selected_samples = class_samples[:min_samples_per_class]

            new_samples.extend(selected_samples)
            new_targets.extend([class_idx] * len(selected_samples))

        # Update the dataset
        self.samples = new_samples
        self.targets = new_targets


def _make_dataloader(
    data_path,
    transform,
    batch_size=16,
    under_sampling=False,
    min_samples_per_class=None,
    seed=None,
):
    if under_sampling:
        if seed is not None:
            random.seed(seed)
        dataset = UnderSampledImageFolder(
            data_path, transform=transform, min_samples_per_class=min_samples_per_class
        )
    else:
        dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_CLASSES = 2
    IMG_SIZE_ORIGINAL = 40
    UNDER_SAMPLING = True
    IMG_SIZE = 299  # InceptionV3 input size
    OUT_DIR = "./out"
    SEED = 0
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = InceptionV3FeatureTrainer(model_dir="./models", num_classes=NUM_CLASSES)
    trainer = trainer.to(device)
    print("Model initialized successfully.")

    print("\nTrainable parameters:")
    for param in trainer.get_trainable_parameters():
        print("  -", param.shape)

    # Data paths
    train_data_path = "./data/train"
    val_data_path = "./data/val"

    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMG_SIZE_ORIGINAL, IMG_SIZE_ORIGINAL)
            ),  # make the spacial frequency equal
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("\nLoading datasets...")

    # Training dataloader
    print("\nTraining set:")
    train_loader = _make_dataloader(
        train_data_path,
        transform,
        BATCH_SIZE,
        under_sampling=UNDER_SAMPLING,
        min_samples_per_class=None,
        seed=SEED,
    )
    print("  - Number of batches:", len(train_loader))
    print("  - Batch size:", BATCH_SIZE)
    print("  - Unique classes:", train_loader.dataset.classes)

    # Validation dataloader (no under-sampling for validation)
    print("\nValidation set:")
    val_loader = _make_dataloader(
        val_data_path,
        transform,
        BATCH_SIZE,
        under_sampling=False,
        min_samples_per_class=None,
        seed=SEED,
    )
    print("  - Number of batches:", len(val_loader))
    print("  - Total samples:", len(val_loader.dataset))
    print("  - Unique classes:", val_loader.dataset.classes)

    # Setup optimizer to only train the final layer
    optimizer = torch.optim.Adam(trainer.get_trainable_parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        # Training phase
        trainer.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = trainer(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation phase
        trainer.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = trainer(inputs)
                loss = criterion(outputs, labels)

                # Track metrics
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation metrics
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # Epoch summary
        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

    print("\nTraining completed!")

    save_path = "./models/inception_v3_trained.pth"
    torch.save(trainer.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    csv_path = OUT_DIR + "/training_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(range(1, EPOCHS + 1)))
        writer.writerow(["train_loss"] + [f"{loss:.4f}" for loss in train_loss_list])
        writer.writerow(["train_accuracy"] + [f"{acc:.2f}" for acc in train_acc_list])
        writer.writerow(["val_loss"] + [f"{loss:.4f}" for loss in val_loss_list])
        writer.writerow(["val_accuracy"] + [f"{acc:.2f}" for acc in val_acc_list])

    print(f"Training results saved to {csv_path}")
