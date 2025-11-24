import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load all training results
def load_results(base_path, experiment_name):
    results = []
    pattern = os.path.join(base_path, experiment_name, "seed_*", "training_results.csv")
    csv_paths = sorted(glob(pattern))

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # Extract seed number from path
        seed = int(csv_path.split("seed_")[1].split(os.sep)[0])
        df["seed"] = seed
        df["experiment"] = experiment_name
        results.append(df)
    return pd.concat(results, ignore_index=True)


base_path = "./out/split-ratio0.2_seed0"
train_normal = load_results(base_path, "train")
train_synth = load_results(base_path, "train-synth")

# Combine all results
all_results = pd.concat([train_normal, train_synth], ignore_index=True)

# Get final epoch (epoch 10) results for each seed
final_results = all_results[all_results["epoch"] == 10].copy()

# Calculate statistics for each experiment
metrics_to_compare = [
    "val_accuracy",
    "val_pr_auc",
    "val_roc_auc",
    "val_0.Normal_accuracy",
    "val_1.Abnormal_accuracy",
    "train_accuracy",
    "train_pr_auc",
    "train_roc_auc",
]

print("=" * 80)
print("COMPARISON: Normal vs. Abnormal Training")
print("=" * 80)
print("\n1. NORMAL TRAINING (437 normal + 85 abnormal)")
print("2. SYNTH TRAINING (437 normal + 85 abnormal + 30 synthetic abnormal)")
print("\n" + "=" * 80)
print("FINAL EPOCH (10) VALIDATION METRICS - MEAN ± STD ACROSS 5 SEEDS")
print("=" * 80)

comparison_data = []
for metric in metrics_to_compare:
    normal_values = final_results[final_results["experiment"] == "train"][metric].values
    synth_values = final_results[final_results["experiment"] == "train-synth"][
        metric
    ].values

    normal_mean = normal_values.mean()
    normal_std = normal_values.std()
    synth_mean = synth_values.mean()
    synth_std = synth_values.std()

    improvement = ((synth_mean - normal_mean) / normal_mean) * 100

    comparison_data.append(
        {
            "metric": metric,
            "normal_mean": normal_mean,
            "normal_std": normal_std,
            "synth_mean": synth_mean,
            "synth_std": synth_std,
            "improvement_%": improvement,
        }
    )

    print(f"\n{metric}:")
    print(f"  Normal: {normal_mean:.4f} ± {normal_std:.4f}")
    print(f"  Synth:  {synth_mean:.4f} ± {synth_std:.4f}")
    print(f"  Change: {improvement:+.2f}% {'✓' if improvement > 0 else '✗'}")

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Check for improvements
val_metrics = ["val_accuracy", "val_pr_auc", "val_roc_auc", "val_1.Abnormal_accuracy"]
improvements = comparison_df[comparison_df["metric"].isin(val_metrics)]

print("\nValidation Performance:")
for _, row in improvements.iterrows():
    metric_name = row["metric"].replace("val_", "").replace("_", " ").title()
    if row["improvement_%"] > 0:
        print(f"  ✓ {metric_name}: +{row['improvement_%']:.2f}% improvement")
    else:
        print(f"  ✗ {metric_name}: {row['improvement_%']:.2f}% (decreased)")

# Analyze abnormal class specifically
print("\n" + "-" * 80)
print("ABNORMAL CLASS DETECTION (Most Important for Imbalanced Dataset)")
print("-" * 80)
abnormal_normal = final_results[final_results["experiment"] == "train"][
    "val_1.Abnormal_accuracy"
].values
abnormal_synth = final_results[final_results["experiment"] == "train-synth"][
    "val_1.Abnormal_accuracy"
].values

print(f"\nValidation Accuracy on Abnormal Class:")
print(f"  Normal: {abnormal_normal.mean():.2f}% ± {abnormal_normal.std():.2f}%")
print(f"  Synth:  {abnormal_synth.mean():.2f}% ± {abnormal_synth.std():.2f}%")
print(f"  Min/Max Normal: [{abnormal_normal.min():.2f}%, {abnormal_normal.max():.2f}%]")
print(f"  Min/Max Synth:  [{abnormal_synth.min():.2f}%, {abnormal_synth.max():.2f}%]")

# Training stability
print("\n" + "-" * 80)
print("TRAINING STABILITY (Lower std = more stable)")
print("-" * 80)
for metric in ["val_accuracy", "val_pr_auc", "val_roc_auc"]:
    normal_std = final_results[final_results["experiment"] == "train"][metric].std()
    synth_std = final_results[final_results["experiment"] == "train-synth"][
        metric
    ].std()
    print(f"\n{metric}:")
    print(f"  Normal std: {normal_std:.4f}")
    print(f"  Synth std:  {synth_std:.4f}")
    print(f"  {'✓ More stable' if synth_std < normal_std else '✗ Less stable'}")

# Convergence analysis
print("\n" + "=" * 80)
print("CONVERGENCE ANALYSIS")
print("=" * 80)

# Average validation accuracy across epochs
epoch_comparison = (
    all_results.groupby(["experiment", "epoch"])
    .agg(
        {
            "val_accuracy": ["mean", "std"],
            "val_pr_auc": ["mean", "std"],
            "val_1.Abnormal_accuracy": ["mean", "std"],
        }
    )
    .reset_index()
)

print("\nEarly Epochs (1-3) vs Final (10):")
for exp in ["train", "train-synth"]:
    exp_data = all_results[all_results["experiment"] == exp]
    early_acc = exp_data[exp_data["epoch"] <= 3]["val_accuracy"].mean()
    final_acc = exp_data[exp_data["epoch"] == 10]["val_accuracy"].mean()
    exp_name = "Normal" if exp == "train" else "Synth"
    print(f"\n{exp_name}:")
    print(f"  Epochs 1-3: {early_acc:.2f}%")
    print(f"  Epoch 10:   {final_acc:.2f}%")
    print(f"  Improvement: {final_acc - early_acc:.2f}%")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    "Training Comparison: Normal vs. Synth (with 30 synthetic abnormal samples)",
    fontsize=14,
    fontweight="bold",
)

metrics_to_plot = [
    ("val_accuracy", "Validation Accuracy"),
    ("val_pr_auc", "Validation PR-AUC"),
    ("val_roc_auc", "Validation ROC-AUC"),
    ("val_1.Abnormal_accuracy", "Validation Abnormal Class Accuracy"),
    ("train_accuracy", "Training Accuracy"),
    ("train_pr_auc", "Training PR-AUC"),
]

for idx, (metric, title) in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]

    for exp, label, color in [
        ("train", "Normal (437+85)", "blue"),
        ("train-synth", "Synth (437+85+30)", "red"),
    ]:
        exp_data = all_results[all_results["experiment"] == exp]
        grouped = exp_data.groupby("epoch")[metric].agg(["mean", "std"])

        ax.plot(grouped.index, grouped["mean"], label=label, color=color, linewidth=2)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(base_path, "training_comparison.png"),
    dpi=300,
    bbox_inches="tight",
)
print("\n" + "=" * 80)
print(f"Visualization saved to: {os.path.join(base_path, 'training_comparison.png')}")
print("=" * 80)

# Summary
print("\n" + "=" * 80)
print("SUMMARY & CONCLUSION")
print("=" * 80)

val_acc_improvement = comparison_df[comparison_df["metric"] == "val_accuracy"][
    "improvement_%"
].values[0]
val_pr_auc_improvement = comparison_df[comparison_df["metric"] == "val_pr_auc"][
    "improvement_%"
].values[0]
val_roc_auc_improvement = comparison_df[comparison_df["metric"] == "val_roc_auc"][
    "improvement_%"
].values[0]
abnormal_improvement = comparison_df[
    comparison_df["metric"] == "val_1.Abnormal_accuracy"
]["improvement_%"].values[0]

print("\nDoes synthetic data help?")
if val_pr_auc_improvement > 0 and val_roc_auc_improvement > 0:
    print("  ✓ YES - Synthetic data shows positive impact")
else:
    print("  ✗ NO - Synthetic data does not show clear improvement")

print(f"\nKey Metrics Change:")
print(f"  - Validation Accuracy: {val_acc_improvement:+.2f}%")
print(f"  - Validation PR-AUC: {val_pr_auc_improvement:+.2f}%")
print(f"  - Validation ROC-AUC: {val_roc_auc_improvement:+.2f}%")
print(f"  - Abnormal Class Accuracy: {abnormal_improvement:+.2f}%")

print("\n" + "=" * 80)

print("\n" + "=" * 80)
