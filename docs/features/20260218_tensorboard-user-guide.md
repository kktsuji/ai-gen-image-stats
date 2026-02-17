# TensorBoard User Guide

**Date**: February 18, 2026
**Status**: Active

## Overview

TensorBoard provides interactive visualizations for monitoring training progress. It runs alongside the existing CSV-based logging system — both are active simultaneously when TensorBoard is enabled.

## Quick Start

### 1. Enable TensorBoard

In your config file (`configs/classifier.yaml` or `configs/diffusion.yaml`), set:

```yaml
logging:
  metrics:
    tensorboard:
      enabled: true
```

### 2. Run Training

```bash
python -m src.main configs/classifier.yaml
```

### 3. Launch TensorBoard

```bash
tensorboard --logdir outputs/your-experiment/tensorboard
```

### 4. Open in Browser

Navigate to **http://localhost:6006**

---

## Configuration Reference

```yaml
logging:
  metrics:
    csv:
      enabled: true # CSV logging is always active

    tensorboard:
      enabled: false # Set to true to enable TensorBoard
      log_dir: null # null = auto (outputs/<experiment>/tensorboard)
      flush_secs: 30 # How often to flush events to disk (seconds)
      log_images: true # Log confusion matrices, predictions, samples
      log_histograms: false # Log weight/gradient histograms (expensive)
      log_graph: false # Log model computational graph (once at start)
```

| Key              | Type      | Default | Description                                |
| ---------------- | --------- | ------- | ------------------------------------------ |
| `enabled`        | bool      | false   | Master switch for TensorBoard logging      |
| `log_dir`        | str\|null | null    | Custom TensorBoard directory; null = auto  |
| `flush_secs`     | int       | 30      | Flush interval in seconds                  |
| `log_images`     | bool      | true    | Log images (confusion matrices, samples)   |
| `log_histograms` | bool      | false   | Log weight/gradient histograms             |
| `log_graph`      | bool      | false   | Log model computational graph at run start |

---

## Output Directory Structure

TensorBoard event files are written to a `tensorboard/` subdirectory within the experiment output:

```
outputs/experiment_name/
├── logs/                   # Application logs (timestamped .log files)
├── checkpoints/            # Model checkpoint files (.pt)
├── metrics/                # CSV metrics files
├── samples/                # Image samples (PNG)
└── tensorboard/            # TensorBoard event files
    └── events.out.tfevents.*
```

The `log_dir` setting controls where TensorBoard files go:

- `null` (default): `outputs/<experiment>/tensorboard/`
- Custom path: Use any absolute or relative path

---

## Visualizations

### Scalar Metrics (SCALARS tab)

All numeric metrics logged each epoch appear under the **SCALARS** tab, grouped by prefix:

**Classifier**:

- `metrics/train_loss` — training loss per epoch
- `metrics/train_accuracy` — training accuracy per epoch
- `metrics/val_loss` — validation loss per epoch
- `metrics/val_accuracy` — validation accuracy per epoch
- `metrics/learning_rate` — learning rate schedule

**Diffusion**:

- `metrics/train_loss` — training loss per epoch
- `metrics/val_loss` — validation loss per epoch
- `metrics/ema_decay` — EMA decay value

### Images (IMAGES tab)

When `log_images: true` (default), the **IMAGES** tab shows:

**Classifier**:

- `images/predictions` — predicted sample grid each epoch
- `confusion_matrix/matrix` — normalized confusion matrix

**Diffusion**:

- `images/samples` — generated image samples
- `denoising/process` — sampled frames from the denoising sequence
- `denoising/figure` — annotated denoising process figure

### Hyperparameters (HPARAMS tab)

The full experiment configuration is logged at the start of training and appears in the **HPARAMS** tab. This enables comparing hyperparameters across multiple runs.

### Histograms (HISTOGRAMS tab)

When `log_histograms: true`, weight and gradient distributions are logged. This is disabled by default because it adds measurable overhead and increases event file size significantly.

### Model Graph (GRAPHS tab)

When `log_graph: true`, the model computational graph is logged once at the start of training. Useful for verifying architecture. Disabled by default.

---

## Usage Examples

### Monitoring a Single Run

```bash
# Start TensorBoard in the experiment directory
tensorboard --logdir outputs/classifier-test/tensorboard

# Live reload every 5 seconds
tensorboard --logdir outputs/classifier-test/tensorboard --reload_interval 5
```

### Comparing Multiple Experiments

```bash
# Side-by-side comparison with named runs
tensorboard --logdir_spec=baseline:outputs/exp-baseline/tensorboard,augmented:outputs/exp-augmented/tensorboard
```

### Remote Access via SSH Tunnel

```bash
# On remote server
tensorboard --logdir outputs/experiment/tensorboard --bind_all

# On local machine
ssh -L 6006:localhost:6006 user@remote-server

# Then open http://localhost:6006 locally
```

### Custom Port

```bash
tensorboard --logdir outputs/experiment/tensorboard --port 6007
```

---

## Troubleshooting

### TensorBoard not installed

If TensorBoard is enabled in config but the package is not installed, training continues normally with a warning:

```
WARNING | TensorBoard logging enabled but tensorboard package not installed.
         Install with: pip install tensorboard
```

To install:

```bash
pip install tensorboard
```

### No events visible in TensorBoard

1. Verify `enabled: true` is set in your config under `logging.metrics.tensorboard`
2. Check the `tensorboard/` directory exists in your experiment output
3. Ensure you are pointing TensorBoard at the correct `--logdir`
4. Wait for the first flush (default: 30 seconds after training starts)

### Events appear but scalars are empty

Scalars are written at the end of each epoch. Check that at least one epoch has completed.

### TensorBoard shows stale data

Use `--reload_interval` for faster refresh:

```bash
tensorboard --logdir outputs/experiment/tensorboard --reload_interval 5
```

---

## Performance Considerations

| Feature        | Overhead      | Disk Usage  | Recommendation               |
| -------------- | ------------- | ----------- | ---------------------------- |
| Scalar logging | < 1%          | ~1 MB/run   | Always safe to enable        |
| Image logging  | 1–3%          | ~10–50 MB   | Enable for visual debugging  |
| Histograms     | 5–15%         | ~100–500 MB | Enable only when needed      |
| Model graph    | One-time cost | ~1–5 MB     | Enable once per architecture |

With default settings (`log_images: true`, `log_histograms: false`, `log_graph: false`), the performance overhead is less than 3%.

---

## Relationship to CSV Logging

CSV logging and TensorBoard logging are independent and simultaneous:

- **CSV logging**: Always enabled, writes to `metrics/metrics.csv`
- **TensorBoard**: Optional, controlled by `logging.metrics.tensorboard.enabled`

Disabling TensorBoard has no effect on CSV output. Both can be read independently.

---

## See Also

- [TensorBoard Integration Plan](20260218_tensorboard-integration-plan.md) — full implementation design
- [Architecture Specification](../standards/architecture.md) — hybrid logging strategy
- [configs/classifier.yaml](../../configs/classifier.yaml) — classifier config with TensorBoard options
- [configs/diffusion.yaml](../../configs/diffusion.yaml) — diffusion config with TensorBoard options
- [TensorBoard documentation](https://www.tensorflow.org/tensorboard)
