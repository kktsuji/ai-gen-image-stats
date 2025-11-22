import argparse
import csv
import os
import random
import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from inception_v3 import InceptionV3FeatureTrainer
from wrn28_cifar10 import WRN28Cifar10Trainer


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
    use_weighted_sampling=False,
):
    if under_sampling:
        if seed is not None:
            random.seed(seed)
        dataset = UnderSampledImageFolder(
            data_path, transform=transform, min_samples_per_class=min_samples_per_class
        )
    else:
        dataset = datasets.ImageFolder(data_path, transform=transform)

    # Setup weighted sampling if enabled
    if use_weighted_sampling:
        from torch.utils.data import WeightedRandomSampler

        print(f"  - Weighted sampling: ENABLED")
        # Calculate class distribution
        class_counts = {}
        for _, class_idx in dataset.samples:
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        # Calculate weights for each class (inverse frequency)
        num_samples = sum(class_counts.values())
        class_weights = [
            num_samples / class_counts[i] for i in sorted(class_counts.keys())
        ]
        print(f"    - Class weights: {[f'{w:.3f}' for w in class_weights]}")

        # Assign weight to each sample based on its class
        sample_weights = [class_weights[label] for _, label in dataset.samples]

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(dataset), replacement=True
        )

        # Use sampler (don't use shuffle with sampler)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        print(f"  - Weighted sampling: DISABLED (using random sampling)")
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(
    use_class_weights=False,
    use_weighted_sampling=False,
    suffix="",
    model_type="inception_v3",
    seed=0,
    epochs=10,
    batch_size=16,
    learning_rate=0.00005,
    num_classes=2,
    img_size_original=40,
    under_sampling=False,
):
    random.seed(seed)
    torch.manual_seed(seed)

    # Set IMG_SIZE based on model type
    if model_type == "wrn28_cifar10":
        IMG_SIZE = 40  # WRN28 uses 40x40 input size
        MODEL_NAME = "wrn28"
    else:
        IMG_SIZE = 299  # InceptionV3 input size
        MODEL_NAME = "inception_v3"

    # SUFFIX = "_no-synth_imbalanced-val_seed0"
    OUT_DIR = f"./out/train-manual-cleansing/train{suffix}_{MODEL_NAME}_seed{seed}" + (
        "-us"
        if under_sampling
        else ("_cw" if use_class_weights else ("-ws" if use_weighted_sampling else ""))
    )
    ONLY_LAST_LAYER = False
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using model: {model_type}")

    # Initialize model based on type
    if model_type == "wrn28_cifar10":
        # CIFAR-10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        trainer = WRN28Cifar10Trainer(model_dir="./models", dropout_rate=0.3)
        if not ONLY_LAST_LAYER:
            # Make post_activ (last layer before fc) trainable
            for param in trainer.features.post_activ.parameters():
                param.requires_grad = True

    else:
        # ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trainer = InceptionV3FeatureTrainer(
            model_dir="./models", num_classes=num_classes
        )
        if not ONLY_LAST_LAYER:
            for param in trainer.Conv2d_1a_3x3.parameters():
                param.requires_grad = True
            for param in trainer.Conv2d_2a_3x3.parameters():
                param.requires_grad = True
            for param in trainer.Conv2d_2b_3x3.parameters():
                param.requires_grad = True
            for param in trainer.Conv2d_3b_1x1.parameters():
                param.requires_grad = True
            for param in trainer.Conv2d_4a_3x3.parameters():
                param.requires_grad = True
            for param in trainer.Mixed_5b.parameters():
                param.requires_grad = True
            for param in trainer.Mixed_5c.parameters():
                param.requires_grad = True
            for param in trainer.Mixed_5d.parameters():
                param.requires_grad = True
            # for param in trainer.Mixed_6a.parameters():
            #     param.requires_grad = True
            # for param in trainer.Mixed_6b.parameters():
            #     param.requires_grad = True
            # for param in trainer.Mixed_6c.parameters():
            #     param.requires_grad = True
            # for param in trainer.Mixed_6d.parameters():
            #     param.requires_grad = True
            # for param in trainer.Mixed_6e.parameters():
            #     param.requires_grad = True
            for param in trainer.Mixed_7a.parameters():
                param.requires_grad = True
            for param in trainer.Mixed_7b.parameters():
                param.requires_grad = True
            for param in trainer.Mixed_7c.parameters():
                param.requires_grad = True

    trainer = trainer.to(device)
    print("Model initialized successfully.")

    print("\nTrainable parameters:")
    for param in trainer.get_trainable_parameters():
        print("  -", param.shape)

    # Data paths
    train_data_path = f"./out/train-manual-cleansing/data/train{suffix}"
    val_data_path = (
        "./out/train-manual-cleansing/data/val_no-synth_imbalanced-val_seed0"
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size_original, img_size_original)
            ),  # make the spacial frequency equal
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size_original, img_size_original)
            ),  # make the spacial frequency equal
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    print("\nLoading datasets...")

    # Training dataloader
    print("\nTraining set:")
    train_loader = _make_dataloader(
        train_data_path,
        train_transform,
        batch_size,
        under_sampling=under_sampling,
        min_samples_per_class=None,
        use_weighted_sampling=use_weighted_sampling,
    )
    print("  - Number of batches:", len(train_loader))
    print("  - Batch size:", batch_size)
    print("  - Unique classes:", train_loader.dataset.classes)

    # Validation dataloader (no under-sampling for validation)
    print("\nValidation set:")
    val_loader = _make_dataloader(
        val_data_path,
        val_transform,
        batch_size,
        under_sampling=False,
        min_samples_per_class=None,
    )
    print("  - Number of batches:", len(val_loader))
    print("  - Total samples:", len(val_loader.dataset))
    print("  - Unique classes:", val_loader.dataset.classes)

    if train_loader.dataset.classes != val_loader.dataset.classes:
        raise ValueError("Train and validation datasets have different classes!")
    if len(train_loader.dataset.classes) != num_classes:
        raise ValueError("Number of classes does not match num_classes!")

    unique_classes = train_loader.dataset.classes

    # Calculate class weights for handling imbalance
    class_weights = None
    if use_class_weights:
        # Count samples per class in training set
        class_sample_count = torch.zeros(num_classes)
        for _, label in train_loader.dataset.samples:
            class_sample_count[label] += 1

        # Calculate weights as inverse of frequency
        # weight = 1 / (samples_per_class / total_samples)
        # = total_samples / samples_per_class
        total_samples = class_sample_count.sum()
        class_weights = total_samples / class_sample_count

        # Normalize weights so they sum to num_classes
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights = class_weights.to(device)

        print(f"\nClass sample distribution: {class_sample_count.tolist()}")
        print(f"Class weights applied: {class_weights.tolist()}")

    # Setup optimizer to only train the final layer
    optimizer = torch.optim.Adam(trainer.get_trainable_parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loss_list = []
    train_acc_list = []
    train_class0_acc_list = []
    train_class1_acc_list = []
    train_class0_loss_list = []
    train_class1_loss_list = []
    train_pr_auc_list = []
    train_roc_auc_list = []
    val_loss_list = []
    val_acc_list = []
    val_class0_acc_list = []
    val_class1_acc_list = []
    val_class0_loss_list = []
    val_class1_loss_list = []
    val_pr_auc_list = []
    val_roc_auc_list = []

    # Training loop
    print("\nStarting training...")
    training_start_time = time.time()

    random.seed(seed)
    torch.manual_seed(seed)

    for epoch in range(epochs):
        # Training phase
        trainer.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class0_correct = 0
        class0_total = 0
        class1_correct = 0
        class1_total = 0
        class0_loss = 0.0
        class1_loss = 0.0
        class0_batch_count = 0
        class1_batch_count = 0

        # For PR-AUC calculation
        all_labels = []
        all_probs = []

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

            # Collect probabilities for PR-AUC (softmax to get probabilities)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(
                probs[:, 1].detach().cpu().numpy()
            )  # Probability of class 1
            all_labels.extend(labels.cpu().numpy())

            # Track per-class metrics
            for i in range(len(labels)):
                label_class = labels[i].item()
                # Calculate per-sample loss for this class
                sample_loss = criterion(
                    outputs[i].unsqueeze(0), labels[i].unsqueeze(0)
                ).item()

                if label_class == 0:
                    class0_total += 1
                    class0_loss += sample_loss
                    class0_batch_count += 1
                    if predicted[i].item() == label_class:
                        class0_correct += 1
                elif label_class == 1:
                    class1_total += 1
                    class1_loss += sample_loss
                    class1_batch_count += 1
                    if predicted[i].item() == label_class:
                        class1_correct += 1

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_class0_acc = (
            100 * class0_correct / class0_total if class0_total > 0 else 0
        )
        train_class1_acc = (
            100 * class1_correct / class1_total if class1_total > 0 else 0
        )
        train_class0_loss = class0_loss / class0_total if class0_total > 0 else 0
        train_class1_loss = class1_loss / class1_total if class1_total > 0 else 0

        # Calculate PR-AUC for training
        train_pr_auc = average_precision_score(all_labels, all_probs)

        # Calculate ROC-AUC for training
        fpr_train_epoch, tpr_train_epoch, _ = roc_curve(all_labels, all_probs)
        train_roc_auc = auc(fpr_train_epoch, tpr_train_epoch)

        # Validation phase
        trainer.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_class0_correct = 0
        val_class0_total = 0
        val_class1_correct = 0
        val_class1_total = 0
        val_class0_loss = 0.0
        val_class1_loss = 0.0

        # For PR-AUC calculation
        val_all_labels = []
        val_all_probs = []

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

                # Collect probabilities for PR-AUC
                probs = torch.nn.functional.softmax(outputs, dim=1)
                val_all_probs.extend(probs[:, 1].detach().cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

                # Track per-class metrics
                for i in range(len(labels)):
                    label_class = labels[i].item()
                    # Calculate per-sample loss for this class
                    sample_loss = criterion(
                        outputs[i].unsqueeze(0), labels[i].unsqueeze(0)
                    ).item()

                    if label_class == 0:
                        val_class0_total += 1
                        val_class0_loss += sample_loss
                        if predicted[i].item() == label_class:
                            val_class0_correct += 1
                    elif label_class == 1:
                        val_class1_total += 1
                        val_class1_loss += sample_loss
                        if predicted[i].item() == label_class:
                            val_class1_correct += 1

        # Calculate validation metrics
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_class0_acc = (
            100 * val_class0_correct / val_class0_total if val_class0_total > 0 else 0
        )
        val_class1_acc = (
            100 * val_class1_correct / val_class1_total if val_class1_total > 0 else 0
        )
        val_class0_loss = (
            val_class0_loss / val_class0_total if val_class0_total > 0 else 0
        )
        val_class1_loss = (
            val_class1_loss / val_class1_total if val_class1_total > 0 else 0
        )

        # Calculate PR-AUC for validation
        val_pr_auc = average_precision_score(val_all_labels, val_all_probs)

        # Calculate ROC-AUC for validation
        fpr_val_epoch, tpr_val_epoch, _ = roc_curve(val_all_labels, val_all_probs)
        val_roc_auc = auc(fpr_val_epoch, tpr_val_epoch)

        # Epoch summary
        print(
            f"Epoch {epoch + 1}: "
            f"\033[92mTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, PR-AUC: {train_pr_auc:.4f}, ROC-AUC: {train_roc_auc:.4f}\033[0m "
            f"(\033[94m{unique_classes[0]}: {train_class0_acc:.2f}% Loss: {train_class0_loss:.4f}\033[0m, "
            f"\033[91m{unique_classes[1]}: {train_class1_acc:.2f}% Loss: {train_class1_loss:.4f}\033[0m) | "
            f"\033[92mVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, PR-AUC: {val_pr_auc:.4f}, ROC-AUC: {val_roc_auc:.4f}\033[0m "
            f"(\033[94m{unique_classes[0]}: {val_class0_acc:.2f}% Loss: {val_class0_loss:.4f}\033[0m, "
            f"\033[91m{unique_classes[1]}: {val_class1_acc:.2f}% Loss: {val_class1_loss:.4f}\033[0m)"
        )

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_class0_acc_list.append(train_class0_acc)
        train_class1_acc_list.append(train_class1_acc)
        train_class0_loss_list.append(train_class0_loss)
        train_class1_loss_list.append(train_class1_loss)
        train_pr_auc_list.append(train_pr_auc)
        train_roc_auc_list.append(train_roc_auc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_class0_acc_list.append(val_class0_acc)
        val_class1_acc_list.append(val_class1_acc)
        val_class0_loss_list.append(val_class0_loss)
        val_class1_loss_list.append(val_class1_loss)
        val_pr_auc_list.append(val_pr_auc)
        val_roc_auc_list.append(val_roc_auc)

    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    print(f"\nTraining completed! Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    # Calculate precision-recall curve for the last epoch
    print("\nCalculating precision-recall curve for final epoch...")
    precision_train, recall_train, thresholds_train = precision_recall_curve(
        all_labels, all_probs
    )
    precision_val, recall_val, thresholds_val = precision_recall_curve(
        val_all_labels, val_all_probs
    )
    print(
        f"  - Training: {len(precision_train)} points on PR curve (PR-AUC: {train_pr_auc:.4f})"
    )
    print(
        f"  - Validation: {len(precision_val)} points on PR curve (PR-AUC: {val_pr_auc:.4f})"
    )

    # Calculate ROC curve for the last epoch
    print("\nCalculating ROC curve for final epoch...")
    fpr_train, tpr_train, roc_thresholds_train = roc_curve(all_labels, all_probs)
    fpr_val, tpr_val, roc_thresholds_val = roc_curve(val_all_labels, val_all_probs)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_val = auc(fpr_val, tpr_val)
    print(
        f"  - Training: {len(fpr_train)} points on ROC curve (ROC-AUC: {roc_auc_train:.4f})"
    )
    print(
        f"  - Validation: {len(fpr_val)} points on ROC curve (ROC-AUC: {roc_auc_val:.4f})"
    )

    # Save precision-recall curve data
    pr_curve_train_path = f"{OUT_DIR}/pr_curve_train.csv"
    with open(pr_curve_train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "threshold"])
        # Note: thresholds array has one fewer element than precision/recall
        for i in range(len(precision_train)):
            threshold = thresholds_train[i] if i < len(thresholds_train) else "N/A"
            writer.writerow(
                [
                    f"{precision_train[i]:.6f}",
                    f"{recall_train[i]:.6f}",
                    threshold if threshold == "N/A" else f"{threshold:.6f}",
                ]
            )
    print(f"\nTraining PR curve saved to {pr_curve_train_path}")

    pr_curve_val_path = f"{OUT_DIR}/pr_curve_val.csv"
    with open(pr_curve_val_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "threshold"])
        for i in range(len(precision_val)):
            threshold = thresholds_val[i] if i < len(thresholds_val) else "N/A"
            writer.writerow(
                [
                    f"{precision_val[i]:.6f}",
                    f"{recall_val[i]:.6f}",
                    threshold if threshold == "N/A" else f"{threshold:.6f}",
                ]
            )
    print(f"Validation PR curve saved to {pr_curve_val_path}")

    # Save ROC curve data
    roc_curve_train_path = f"{OUT_DIR}/roc_curve_train.csv"
    with open(roc_curve_train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for i in range(len(fpr_train)):
            threshold = (
                roc_thresholds_train[i] if i < len(roc_thresholds_train) else "N/A"
            )
            writer.writerow(
                [
                    f"{fpr_train[i]:.6f}",
                    f"{tpr_train[i]:.6f}",
                    threshold if threshold == "N/A" else f"{threshold:.6f}",
                ]
            )
    print(f"Training ROC curve saved to {roc_curve_train_path}")

    roc_curve_val_path = f"{OUT_DIR}/roc_curve_val.csv"
    with open(roc_curve_val_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold"])
        for i in range(len(fpr_val)):
            threshold = roc_thresholds_val[i] if i < len(roc_thresholds_val) else "N/A"
            writer.writerow(
                [
                    f"{fpr_val[i]:.6f}",
                    f"{tpr_val[i]:.6f}",
                    threshold if threshold == "N/A" else f"{threshold:.6f}",
                ]
            )
    print(f"Validation ROC curve saved to {roc_curve_val_path}")

    # Plot precision-recall curves
    print("\nGenerating PR curve plots...")
    plt.figure(figsize=(10, 8))

    # Plot training PR curve
    plt.plot(
        recall_train,
        precision_train,
        label=f"Training (PR-AUC = {train_pr_auc:.4f})",
        linewidth=2,
        color="blue",
    )

    # Plot validation PR curve
    plt.plot(
        recall_val,
        precision_val,
        label=f"Validation (PR-AUC = {val_pr_auc:.4f})",
        linewidth=2,
        color="red",
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve - Final Epoch", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    pr_curve_plot_path = f"{OUT_DIR}/pr_curve.png"
    plt.savefig(pr_curve_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PR curve plot saved to {pr_curve_plot_path}")

    # Plot ROC curves
    print("\nGenerating ROC curve plots...")
    plt.figure(figsize=(10, 8))

    # Plot training ROC curve
    plt.plot(
        fpr_train,
        tpr_train,
        label=f"Training (ROC-AUC = {roc_auc_train:.4f})",
        linewidth=2,
        color="blue",
    )

    # Plot validation ROC curve
    plt.plot(
        fpr_val,
        tpr_val,
        label=f"Validation (ROC-AUC = {roc_auc_val:.4f})",
        linewidth=2,
        color="red",
    )

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Final Epoch", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    roc_curve_plot_path = f"{OUT_DIR}/roc_curve.png"
    plt.savefig(roc_curve_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve plot saved to {roc_curve_plot_path}")

    # Plot training and validation loss
    print("\nGenerating loss plot...")
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, epochs + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_list, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs_range, val_loss_list, "r-", label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs_range,
        train_class0_loss_list,
        "b--",
        label=f"Train {unique_classes[0]}",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        train_class1_loss_list,
        "b:",
        label=f"Train {unique_classes[1]}",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        val_class0_loss_list,
        "r--",
        label=f"Val {unique_classes[0]}",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        val_class1_loss_list,
        "r:",
        label=f"Val {unique_classes[1]}",
        linewidth=2,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Per-Class Loss", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot_path = f"{OUT_DIR}/loss_plot.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")

    # Plot training and validation accuracy
    print("\nGenerating accuracy plot...")
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_list, "b-", label="Training Accuracy", linewidth=2)
    plt.plot(epochs_range, val_acc_list, "r-", label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs_range,
        train_class0_acc_list,
        "b--",
        label=f"Train {unique_classes[0]}",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        train_class1_acc_list,
        "b:",
        label=f"Train {unique_classes[1]}",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        val_class0_acc_list,
        "r--",
        label=f"Val {unique_classes[0]}",
        linewidth=2,
    )
    plt.plot(
        epochs_range,
        val_class1_acc_list,
        "r:",
        label=f"Val {unique_classes[1]}",
        linewidth=2,
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])

    plt.tight_layout()
    accuracy_plot_path = f"{OUT_DIR}/accuracy_plot.png"
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Accuracy plot saved to {accuracy_plot_path}")

    # Plot PR-AUC over epochs
    print("\nGenerating PR-AUC plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs_range, train_pr_auc_list, "b-", label="Training PR-AUC", linewidth=2
    )
    plt.plot(
        epochs_range, val_pr_auc_list, "r-", label="Validation PR-AUC", linewidth=2
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("PR-AUC", fontsize=12)
    plt.title("PR-AUC over Epochs", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    pr_auc_plot_path = f"{OUT_DIR}/pr_auc_plot.png"
    plt.savefig(pr_auc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PR-AUC plot saved to {pr_auc_plot_path}")

    # Plot ROC-AUC over epochs
    print("\nGenerating ROC-AUC plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs_range, train_roc_auc_list, "b-", label="Training ROC-AUC", linewidth=2
    )
    plt.plot(
        epochs_range, val_roc_auc_list, "r-", label="Validation ROC-AUC", linewidth=2
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.title("ROC-AUC over Epochs", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    roc_auc_plot_path = f"{OUT_DIR}/roc_auc_plot.png"
    plt.savefig(roc_auc_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC-AUC plot saved to {roc_auc_plot_path}")

    save_path = f"{OUT_DIR}/{MODEL_NAME}_trained.pth"
    torch.save(trainer.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    csv_path = f"{OUT_DIR}/training_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + list(range(1, epochs + 1)))
        writer.writerow(["train_loss"] + [f"{loss:.4f}" for loss in train_loss_list])
        writer.writerow(["train_accuracy"] + [f"{acc:.2f}" for acc in train_acc_list])
        writer.writerow(
            [f"train_{unique_classes[0]}_accuracy"]
            + [f"{acc:.2f}" for acc in train_class0_acc_list]
        )
        writer.writerow(
            [f"train_{unique_classes[1]}_accuracy"]
            + [f"{acc:.2f}" for acc in train_class1_acc_list]
        )
        writer.writerow(
            [f"train_{unique_classes[0]}_loss"]
            + [f"{loss:.4f}" for loss in train_class0_loss_list]
        )
        writer.writerow(
            [f"train_{unique_classes[1]}_loss"]
            + [f"{loss:.4f}" for loss in train_class1_loss_list]
        )
        writer.writerow(
            ["train_pr_auc"] + [f"{pr_auc:.4f}" for pr_auc in train_pr_auc_list]
        )
        writer.writerow(
            ["train_roc_auc"] + [f"{roc_auc:.4f}" for roc_auc in train_roc_auc_list]
        )
        writer.writerow(["val_loss"] + [f"{loss:.4f}" for loss in val_loss_list])
        writer.writerow(["val_accuracy"] + [f"{acc:.2f}" for acc in val_acc_list])
        writer.writerow(
            [f"val_{unique_classes[0]}_accuracy"]
            + [f"{acc:.2f}" for acc in val_class0_acc_list]
        )
        writer.writerow(
            [f"val_{unique_classes[1]}_accuracy"]
            + [f"{acc:.2f}" for acc in val_class1_acc_list]
        )
        writer.writerow(
            [f"val_{unique_classes[0]}_loss"]
            + [f"{loss:.4f}" for loss in val_class0_loss_list]
        )
        writer.writerow(
            [f"val_{unique_classes[1]}_loss"]
            + [f"{loss:.4f}" for loss in val_class1_loss_list]
        )
        writer.writerow(
            ["val_pr_auc"] + [f"{pr_auc:.4f}" for pr_auc in val_pr_auc_list]
        )
        writer.writerow(
            ["val_roc_auc"] + [f"{roc_auc:.4f}" for roc_auc in val_roc_auc_list]
        )

    print(f"Training results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification models")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.00005, help="Learning rate"
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument(
        "--img-size-original",
        type=int,
        default=40,
        help="Original image size for resizing",
    )
    parser.add_argument(
        "--under-sampling", action="store_true", help="Enable under-sampling"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=["inception_v3", "wrn28_cifar10", "all"],
        help="Model type to train",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_no-synth_imbalanced-val_seed0",
        help="Suffix for output directory",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights in loss function",
    )
    parser.add_argument(
        "--use-weighted-sampling",
        action="store_true",
        help="Use weighted sampling in data loader",
    )
    args = parser.parse_args()

    train(
        use_class_weights=args.use_class_weights,
        use_weighted_sampling=args.use_weighted_sampling,
        suffix=args.suffix,
        model_type=args.model_type,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes,
        img_size_original=args.img_size_original,
        under_sampling=args.under_sampling,
    )
