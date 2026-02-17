# TensorBoard Integration Plan

**Date**: February 18, 2026
**Author**: AI Assistant
**Status**: Planning

## Overview

Integrate TensorBoard as a complementary logging mechanism alongside existing CSV-based metrics logging, providing rich interactive visualizations for monitoring training progress.

**Objectives**:

1. Maintain existing CSV logging while adding optional TensorBoard support
2. Enable simultaneous CSV and TensorBoard logging in the same training run
3. Provide interactive plots, histograms, and image visualizations
4. Allow users to enable/disable TensorBoard via YAML config
5. Ensure backward compatibility — zero breaking changes
6. Integrate seamlessly with existing `BaseLogger` architecture

**Design**:

- CSV logging remains unchanged and always enabled
- TensorBoard is optional, controlled by `logging.metrics.tensorboard.enabled` config
- Default: disabled. Graceful degradation if `tensorboard` package not installed
- All TensorBoard utility functions accept `Optional[SummaryWriter]` and handle `None` gracefully

**Architecture**:

```
BaseLogger (Abstract)
    ├── log_metrics()          # Abstract method
    ├── log_images()           # Abstract method
    ├── log_histogram()        # Optional method
    └── log_hyperparams()      # Optional method

ClassifierLogger(BaseLogger)
    ├── _csv_writer            # Existing CSV logging
    ├── _tensorboard_writer    # NEW: Optional TensorBoard writer
    └── log_metrics()          # Writes to BOTH if enabled

DiffusionLogger(BaseLogger)
    ├── _csv_writer            # Existing CSV logging
    ├── _tensorboard_writer    # NEW: Optional TensorBoard writer
    └── log_metrics()          # Writes to BOTH if enabled
```

**Files Changed**:

| File                                    | Action                             |
| --------------------------------------- | ---------------------------------- |
| `requirements.txt`                      | Add `tensorboard>=2.14.0`          |
| `src/utils/tensorboard.py`              | NEW: TensorBoard utility functions |
| `src/experiments/classifier/logger.py`  | Add TensorBoard support            |
| `src/experiments/diffusion/logger.py`   | Add TensorBoard support            |
| `configs/classifier.yaml`               | Add `metrics.tensorboard` config   |
| `configs/diffusion.yaml`                | Add `metrics.tensorboard` config   |
| `src/experiments/classifier/trainer.py` | Pass tensorboard config to logger  |
| `src/experiments/diffusion/trainer.py`  | Pass tensorboard config to logger  |

**Output Directory Structure**:

```
outputs/experiment_name/
├── logs/                   # Application logs (existing)
├── checkpoints/            # Model checkpoints (existing)
├── metrics/                # CSV metrics (existing)
├── samples/                # Image samples (existing)
└── tensorboard/            # NEW: TensorBoard logs
    └── events.out.tfevents.*
```

---

## Implementation Checklist

### Prerequisites

- [x] Review and approve this plan
- [x] Ensure PyTorch 2.0+ and Python 3.8+ available

### Phase 1: Foundation Setup

- [x] 1.1: Add `tensorboard>=2.14.0` to `requirements.txt`
- [x] 1.2: Create `src/utils/tensorboard.py`
- [x] 1.3: Install and verify: `pip install tensorboard>=2.14.0`
- [x] 1.4: Write `tests/utils/test_tensorboard.py`

### Phase 2: Logger Integration

- [x] 2.1: Modify `src/experiments/classifier/logger.py`
  - [x] Add TensorBoard imports
  - [x] Update `__init__` to accept `tensorboard_config`
  - [x] Update `log_metrics` for dual logging
  - [x] Update `log_images` for TensorBoard
  - [x] Update `log_confusion_matrix` for TensorBoard
  - [x] Add `log_hyperparams` method
  - [x] Update `close` method
- [x] 2.2: Modify `src/experiments/diffusion/logger.py`
  - [x] Add TensorBoard imports
  - [x] Update `__init__` to accept `tensorboard_config`
  - [x] Update `log_metrics` for dual logging
  - [x] Update `log_images` for TensorBoard
  - [x] Update `log_denoising_process` for TensorBoard
  - [x] Add `log_hyperparams` method
  - [x] Update `close` method
- [x] 2.3: Run logger unit tests

### Phase 3: Configuration

- [x] 3.1: Add `metrics` section to `configs/classifier.yaml`
- [x] 3.2: Add `metrics` section to `configs/diffusion.yaml`
- [x] 3.3: Validate YAML syntax

### Phase 4: Trainer Integration

- [x] 4.1: Update `src/experiments/classifier/trainer.py`
  - [x] Pass `tensorboard_config` to logger
  - [x] Add optional model graph logging
  - [x] Add hyperparameter logging at training start
- [x] 4.2: Update `src/experiments/diffusion/trainer.py`
  - [x] Pass `tensorboard_config` to logger
  - [x] Add optional model graph logging
  - [x] Add hyperparameter logging at training start
- [x] 4.3: Run trainer unit tests

### Phase 5: Testing

- [x] 5.1: Write `tests/utils/test_tensorboard.py`
- [x] 5.2: Update `tests/experiments/classifier/test_logger.py`
- [x] 5.3: Update `tests/experiments/diffusion/test_logger.py`
- [x] 5.4: Write `tests/integration/test_tensorboard_integration.py`
- [x] 5.5: Run full test suite
- [x] 5.6: Manual testing with sample data

### Phase 6: Documentation

- [x] 6.1: Update `README.md` with TensorBoard section
- [x] 6.2: Create `docs/features/20260218_tensorboard-user-guide.md`
- [x] 6.3: Update `docs/standards/architecture.md`

### Deployment

- [ ] Update requirements in production environment
- [ ] Run smoke tests on production config
- [ ] Monitor first production run
- [ ] Gather user feedback

---

## Phase Details

### Phase 1: Foundation Setup

#### Task 1.1: Add TensorBoard Dependency

**File**: `requirements.txt`

Add under a new "Logging & Visualization" section:

```text
tensorboard>=2.14.0
```

Verify: `pip install tensorboard>=2.14.0 && tensorboard --version`

#### Task 1.2: Create TensorBoard Utility Module

**File**: `src/utils/tensorboard.py` (NEW)

```python
"""TensorBoard Utility Functions

This module provides utility functions for TensorBoard integration including
optional writer creation, safe logging with error handling, and path resolution.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

# Optional import - don't fail if tensorboard not installed
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_tensorboard_available() -> bool:
    """Check if TensorBoard is available.

    Returns:
        True if tensorboard package is installed, False otherwise
    """
    return TENSORBOARD_AVAILABLE


def create_tensorboard_writer(
    log_dir: Union[str, Path],
    flush_secs: int = 30,
    enabled: bool = True,
) -> Optional[SummaryWriter]:
    """Create a TensorBoard SummaryWriter if enabled and available.

    Args:
        log_dir: Directory to save TensorBoard logs
        flush_secs: Flush frequency in seconds
        enabled: Whether TensorBoard logging is enabled

    Returns:
        SummaryWriter instance if enabled and available, None otherwise
    """
    if not enabled:
        logger.info("TensorBoard logging disabled by configuration")
        return None

    if not is_tensorboard_available():
        logger.warning(
            "TensorBoard logging enabled but tensorboard package not installed. "
            "Install with: pip install tensorboard"
        )
        return None

    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
        logger.info(f"TensorBoard logging enabled: {log_dir}")
        return writer
    except Exception as e:
        logger.error(f"Failed to create TensorBoard writer: {e}")
        return None


def safe_log_scalar(
    writer: Optional[SummaryWriter],
    tag: str,
    value: Union[float, int, torch.Tensor],
    step: int,
) -> None:
    """Safely log a scalar value to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        value: Scalar value to log
        step: Global step value
    """
    if writer is None:
        return

    try:
        if isinstance(value, torch.Tensor):
            value = value.item()
        writer.add_scalar(tag, value, step)
    except Exception as e:
        logger.debug(f"Failed to log scalar '{tag}': {e}")


def safe_log_scalars(
    writer: Optional[SummaryWriter],
    main_tag: str,
    tag_scalar_dict: Dict[str, Union[float, int, torch.Tensor]],
    step: int,
) -> None:
    """Safely log multiple scalar values to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        main_tag: Parent name for the group
        tag_scalar_dict: Dictionary of tag-value pairs
        step: Global step value
    """
    if writer is None:
        return

    try:
        processed_dict = {}
        for key, value in tag_scalar_dict.items():
            if isinstance(value, torch.Tensor):
                processed_dict[key] = value.item()
            else:
                processed_dict[key] = value
        writer.add_scalars(main_tag, processed_dict, step)
    except Exception as e:
        logger.debug(f"Failed to log scalars '{main_tag}': {e}")


def safe_log_images(
    writer: Optional[SummaryWriter],
    tag: str,
    img_tensor: torch.Tensor,
    step: int,
    dataformats: str = "NCHW",
) -> None:
    """Safely log images to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        img_tensor: Image tensor (N, C, H, W) or (C, H, W)
        step: Global step value
        dataformats: Format of image tensor (default: NCHW)
    """
    if writer is None:
        return

    try:
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        writer.add_images(tag, img_tensor, step, dataformats=dataformats)
    except Exception as e:
        logger.debug(f"Failed to log images '{tag}': {e}")


def safe_log_histogram(
    writer: Optional[SummaryWriter],
    tag: str,
    values: Union[torch.Tensor, Any],
    step: int,
    bins: str = 'tensorflow',
) -> None:
    """Safely log histogram to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        values: Values to histogram
        step: Global step value
        bins: Binning method ('tensorflow', 'auto', 'fd', etc.)
    """
    if writer is None:
        return

    try:
        writer.add_histogram(tag, values, step, bins=bins)
    except Exception as e:
        logger.debug(f"Failed to log histogram '{tag}': {e}")


def safe_log_figure(
    writer: Optional[SummaryWriter],
    tag: str,
    figure: Any,
    step: int,
    close: bool = True,
) -> None:
    """Safely log matplotlib figure to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        figure: Matplotlib figure object
        step: Global step value
        close: Whether to close figure after logging
    """
    if writer is None:
        return

    try:
        writer.add_figure(tag, figure, step, close=close)
    except Exception as e:
        logger.debug(f"Failed to log figure '{tag}': {e}")


def safe_log_text(
    writer: Optional[SummaryWriter],
    tag: str,
    text_string: str,
    step: int,
) -> None:
    """Safely log text to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        tag: Data identifier
        text_string: Text content
        step: Global step value
    """
    if writer is None:
        return

    try:
        writer.add_text(tag, text_string, step)
    except Exception as e:
        logger.debug(f"Failed to log text '{tag}': {e}")


def safe_log_hparams(
    writer: Optional[SummaryWriter],
    hparam_dict: Dict[str, Any],
    metric_dict: Optional[Dict[str, float]] = None,
) -> None:
    """Safely log hyperparameters to TensorBoard.

    Args:
        writer: SummaryWriter instance (can be None)
        hparam_dict: Dictionary of hyperparameters
        metric_dict: Dictionary of metric values (optional)
    """
    if writer is None:
        return

    try:
        flat_hparams = _flatten_dict(hparam_dict)
        writer.add_hparams(flat_hparams, metric_dict or {})
    except Exception as e:
        logger.debug(f"Failed to log hyperparameters: {e}")


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            if not isinstance(v, (int, float, str, bool)):
                v = str(v)
            items.append((new_key, v))
    return dict(items)


def close_tensorboard_writer(writer: Optional[SummaryWriter]) -> None:
    """Safely close a TensorBoard writer.

    Args:
        writer: SummaryWriter instance (can be None)
    """
    if writer is None:
        return

    try:
        writer.close()
        logger.debug("TensorBoard writer closed")
    except Exception as e:
        logger.error(f"Failed to close TensorBoard writer: {e}")
```

#### Task 1.3: Install and Verify

```bash
pip install tensorboard>=2.14.0 && tensorboard --version
```

#### Task 1.4: Write Unit Tests

**File**: `tests/utils/test_tensorboard.py` (NEW)

Test coverage:

- `is_tensorboard_available()` returns correct boolean
- `create_tensorboard_writer()` with enabled/disabled/missing package
- All `safe_log_*` functions with valid writer, None writer, and error conditions
- `close_tensorboard_writer()` with valid writer and None
- `_flatten_dict()` with nested dictionaries

---

### Phase 2: Logger Integration

Extend ClassifierLogger and DiffusionLogger to optionally write metrics to TensorBoard while preserving all existing CSV logging behavior. TensorBoard logging only occurs when `tb_writer` is not None.

#### Task 2.1: Modify ClassifierLogger

**File**: `src/experiments/classifier/logger.py`

**Add imports:**

```python
from src.utils.tensorboard import (
    create_tensorboard_writer,
    safe_log_scalar,
    safe_log_scalars,
    safe_log_images,
    safe_log_figure,
    safe_log_hparams,
    close_tensorboard_writer,
)
```

**Update `__init__` method:**

```python
def __init__(
    self,
    log_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    tensorboard_config: Optional[Dict[str, Any]] = None,
):
    """Initialize the classifier logger.

    Args:
        log_dir: Directory to save logs and visualizations
        class_names: Names of classes for confusion matrix labels
        tensorboard_config: TensorBoard configuration dict with keys:
            - enabled (bool): Enable TensorBoard logging
            - log_dir (str): Custom TensorBoard directory (optional)
            - flush_secs (int): Flush frequency in seconds
            - log_images (bool): Log images to TensorBoard
            - log_histograms (bool): Log histograms to TensorBoard
            - log_graph (bool): Log model graph to TensorBoard
    """
    self.log_dir = Path(log_dir)
    self.log_dir.mkdir(parents=True, exist_ok=True)

    self.class_names = class_names or []

    # Create subdirectories
    self.metrics_dir = self.log_dir / "metrics"
    self.confusion_dir = self.log_dir / "confusion_matrices"
    self.predictions_dir = self.log_dir / "predictions"

    self.metrics_dir.mkdir(exist_ok=True)
    self.confusion_dir.mkdir(exist_ok=True)
    self.predictions_dir.mkdir(exist_ok=True)

    # Initialize CSV logging (existing)
    self.metrics_file = self.log_dir / "metrics.csv"
    self.csv_initialized = self.metrics_file.exists()
    self.csv_fieldnames = None

    if self.csv_initialized:
        with open(self.metrics_file, "r") as f:
            reader = csv.DictReader(f)
            self.csv_fieldnames = reader.fieldnames

    # Initialize TensorBoard logging (new)
    self.tensorboard_config = tensorboard_config or {}
    self.tb_enabled = self.tensorboard_config.get('enabled', False)
    self.tb_log_images = self.tensorboard_config.get('log_images', True)
    self.tb_log_histograms = self.tensorboard_config.get('log_histograms', False)

    tb_log_dir = self.tensorboard_config.get('log_dir')
    if tb_log_dir is None:
        tb_log_dir = self.log_dir.parent / "tensorboard"
    flush_secs = self.tensorboard_config.get('flush_secs', 30)

    self.tb_writer = create_tensorboard_writer(
        log_dir=tb_log_dir,
        flush_secs=flush_secs,
        enabled=self.tb_enabled,
    )

    self.logged_metrics_history = []
    self.logged_confusion_matrices = []
```

**Update `log_metrics` method:**

```python
def log_metrics(
    self,
    metrics: Dict[str, Union[float, int, torch.Tensor]],
    step: int,
    epoch: Optional[int] = None,
) -> None:
    """Log scalar metrics to CSV and TensorBoard."""
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            processed_metrics[key] = value.item()
        else:
            processed_metrics[key] = value

    log_entry = {"step": step}
    if epoch is not None:
        log_entry["epoch"] = epoch
    log_entry.update(processed_metrics)

    self.logged_metrics_history.append(log_entry)

    # Write to CSV (existing)
    self._write_metrics_to_csv(log_entry)

    # Write to TensorBoard (new)
    if self.tb_writer is not None:
        for key, value in processed_metrics.items():
            safe_log_scalar(self.tb_writer, f"metrics/{key}", value, step)
```

**Update `log_images` method:**

```python
def log_images(
    self,
    images: Union[torch.Tensor, List[torch.Tensor]],
    tag: str,
    step: int,
    epoch: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Log images for visualization to file and TensorBoard."""
    if isinstance(images, list):
        images = torch.stack(images)
    if images.ndim == 3:
        images = images.unsqueeze(0)

    normalize = kwargs.get("normalize", True)
    nrow = kwargs.get("nrow", 8)
    labels = kwargs.get("labels", None)
    predictions = kwargs.get("predictions", None)

    filename_parts = [tag, f"step{step}"]
    if epoch is not None:
        filename_parts.insert(1, f"epoch{epoch}")
    filename = "_".join(filename_parts) + ".png"

    # Save image grid to file (existing)
    image_path = self.predictions_dir / filename
    save_image(images, image_path, normalize=normalize, nrow=nrow)

    # Save to TensorBoard (new)
    if self.tb_writer is not None and self.tb_log_images:
        safe_log_images(self.tb_writer, f"images/{tag}", images, step)

    if labels is not None or predictions is not None:
        self._save_annotated_predictions(
            images, labels, predictions, image_path.with_suffix(".annotated.png")
        )
```

**Update `log_confusion_matrix` method:**

```python
def log_confusion_matrix(
    self,
    confusion_matrix: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    step: int,
    epoch: Optional[int] = None,
    normalize: bool = True,
) -> None:
    """Log confusion matrix visualization."""
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()

    class_names = class_names or self.class_names
    if not class_names:
        num_classes = confusion_matrix.shape[0]
        class_names = [f"Class_{i}" for i in range(num_classes)]

    if normalize:
        confusion_matrix = confusion_matrix.astype(float)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.divide(
            confusion_matrix, row_sums,
            out=np.zeros_like(confusion_matrix), where=row_sums != 0,
        )

    self.logged_confusion_matrices.append({
        "confusion_matrix": confusion_matrix.copy(),
        "class_names": class_names, "step": step, "epoch": epoch,
    })

    fig = self._create_confusion_matrix_figure(confusion_matrix, class_names, normalize)

    # Save to file (existing)
    filename_parts = ["confusion_matrix", f"step{step}"]
    if epoch is not None:
        filename_parts.insert(1, f"epoch{epoch}")
    save_path = self.confusion_dir / ("_".join(filename_parts) + ".png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Save to TensorBoard (new)
    if self.tb_writer is not None and self.tb_log_images:
        safe_log_figure(self.tb_writer, "confusion_matrix/matrix", fig, step, close=False)

    plt.close(fig)
```

**Add `log_hyperparams` method:**

```python
def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
    """Log hyperparameters to TensorBoard."""
    if self.tb_writer is not None:
        safe_log_hparams(self.tb_writer, hyperparams)
```

**Update `close` method:**

```python
def close(self) -> None:
    """Cleanup and finalize logging."""
    close_tensorboard_writer(self.tb_writer)
    self.tb_writer = None
```

#### Task 2.2: Modify DiffusionLogger

**File**: `src/experiments/diffusion/logger.py`

Same imports, `__init__`, `log_metrics`, `log_images`, `log_hyperparams`, and `close` changes as ClassifierLogger (Task 2.1). The only diffusion-specific addition is `log_denoising_process`:

**Update `log_denoising_process` method:**

```python
def log_denoising_process(
    self,
    denoising_sequence: Union[torch.Tensor, List[torch.Tensor]],
    step: int,
    epoch: Optional[int] = None,
    num_steps_to_show: int = 8,
) -> None:
    """Log a visualization of the denoising process."""
    if isinstance(denoising_sequence, list):
        denoising_sequence = torch.stack(denoising_sequence)

    if denoising_sequence.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor (T, C, H, W), got {denoising_sequence.shape}"
        )

    self.logged_denoising_sequences.append({
        "sequence": denoising_sequence.clone(), "step": step, "epoch": epoch,
    })

    # Sample timesteps evenly
    num_timesteps = denoising_sequence.size(0)
    if num_timesteps <= num_steps_to_show:
        indices = list(range(num_timesteps))
    else:
        indices = [
            int(i * (num_timesteps - 1) / (num_steps_to_show - 1))
            for i in range(num_steps_to_show)
        ]
    sampled_images = denoising_sequence[indices]

    fig = self._create_denoising_figure(sampled_images, indices, num_timesteps)

    # Save to file (existing)
    filename_parts = ["denoising", f"step{step}"]
    if epoch is not None:
        filename_parts.insert(1, f"epoch{epoch}")
    save_path = self.denoising_dir / ("_".join(filename_parts) + ".png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # Save to TensorBoard (new)
    if self.tb_writer is not None and self.tb_log_images:
        safe_log_images(self.tb_writer, "denoising/process", sampled_images, step)
        safe_log_figure(self.tb_writer, "denoising/figure", fig, step, close=False)

    plt.close(fig)
```

#### Task 2.3: Run Logger Unit Tests

Verify all existing logger tests still pass with the new optional parameter.

---

### Phase 3: Configuration

Add `metrics` subsection under `logging` in both config files. Insert after the `timezone` line.

#### Task 3.1: Update Classifier Config

**File**: `configs/classifier.yaml` — insert after `timezone: local`:

```yaml
# Metrics Logging Configuration
metrics:
  csv:
    enabled: true

  tensorboard:
    enabled: false # Set to true to enable TensorBoard
    log_dir: null # null = auto ({output.base_dir}/tensorboard)
    flush_secs: 30 # Flush frequency in seconds
    log_images: true # Log confusion matrices, predictions, samples
    log_histograms: false # Log weight/gradient histograms (expensive)
    log_graph: false # Log model computational graph (once at start)
```

#### Task 3.2: Update Diffusion Config

**File**: `configs/diffusion.yaml` — insert after `timezone: local`:

```yaml
# Metrics Logging Configuration
metrics:
  csv:
    enabled: true

  tensorboard:
    enabled: false # Set to true to enable TensorBoard
    log_dir: null # null = auto ({output.base_dir}/tensorboard)
    flush_secs: 30 # Flush frequency in seconds
    log_images: true # Log generated samples, denoising sequences
    log_histograms: false # Log weight/gradient histograms (expensive)
    log_graph: false # Log model computational graph (once at start)
```

#### Task 3.3: Validate YAML Syntax

```bash
python -c "import yaml; yaml.safe_load(open('configs/classifier.yaml')); yaml.safe_load(open('configs/diffusion.yaml')); print('OK')"
```

---

### Phase 4: Trainer Integration

#### Task 4.1: Update Classifier Trainer

**File**: `src/experiments/classifier/trainer.py`

**Extract config and pass to logger:**

```python
def __init__(self, config: Dict[str, Any]):
    # ... existing code ...

    tensorboard_config = config.get('logging', {}).get('metrics', {}).get('tensorboard', {})

    self.logger = ClassifierLogger(
        log_dir=log_dir,
        class_names=class_names,
        tensorboard_config=tensorboard_config,
    )
```

**Optionally log model graph (after model initialization):**

```python
if self.logger.tb_writer is not None and tensorboard_config.get('log_graph', False):
    try:
        dummy_input = torch.randn(1, 3,
            config['data']['preprocessing']['crop_size'],
            config['data']['preprocessing']['crop_size']).to(self.device)
        self.logger.tb_writer.add_graph(self.model, dummy_input)
    except Exception as e:
        logger.warning(f"Failed to log model graph to TensorBoard: {e}")
```

**Log hyperparameters at training start:**

```python
def train(self):
    """Main training loop."""
    self.logger.log_hyperparams(self.config)
    # ... rest of training code ...
```

#### Task 4.2: Update Diffusion Trainer

**File**: `src/experiments/diffusion/trainer.py`

Same changes as Task 4.1, applied to `DiffusionTrainer` and `DiffusionLogger`.

#### Task 4.3: Run Trainer Unit Tests

Verify trainer tests pass with the new config extraction and logger initialization.

---

### Phase 5: Testing

#### Task 5.1: Unit Tests for TensorBoard Utilities

**File**: `tests/utils/test_tensorboard.py` (NEW)

- Test `is_tensorboard_available()`
- Test `create_tensorboard_writer()` with various configs
- Test all `safe_log_*` functions
- Test graceful handling when tensorboard not available
- Test error handling for invalid inputs

#### Task 5.2–5.3: Update Logger Tests

**Files**: `tests/experiments/classifier/test_logger.py`, `tests/experiments/diffusion/test_logger.py`

New test cases:

- Logger with TensorBoard enabled / disabled
- Dual logging (CSV + TensorBoard) produces both outputs
- TensorBoard event files created in correct directory
- Graceful behavior with missing tensorboard package

#### Task 5.4: Integration Tests

**File**: `tests/integration/test_tensorboard_integration.py` (NEW)

Scenarios:

1. **Full training run** with TensorBoard → verify both CSV and TensorBoard logs created and readable
2. **Backward compatibility** with TensorBoard disabled → no TensorBoard files, CSV unchanged
3. **Custom log_dir** → logs in correct location
4. **Graceful degradation** without tensorboard package → training continues, warning logged

#### Task 5.5–5.6: Full Test Suite & Manual Testing

Run `pytest` and manually verify TensorBoard UI with sample data.

---

### Phase 6: Documentation

#### Task 6.1: Update README

**File**: `README.md` — add TensorBoard section:

````markdown
## TensorBoard Integration

The project supports TensorBoard for real-time training visualization alongside traditional CSV logging.

### Quick Start

1. **Enable TensorBoard** in your config file:

```yaml
logging:
  metrics:
    tensorboard:
      enabled: true
```

2. **Run training** as usual:

```bash
python -m src.main --config configs/classifier.yaml
```

3. **Launch TensorBoard**:

```bash
tensorboard --logdir outputs/your-experiment/tensorboard
```

4. **View in browser**: Navigate to http://localhost:6006

### Features

- **Scalar Metrics**: Loss, accuracy, learning rate curves
- **Image Visualization**: Sample predictions, confusion matrices
- **Histogram Analysis**: Weight and gradient distributions (optional)
- **Hyperparameter Tracking**: Log and compare configurations

### Configuration Options

See `configs/classifier.yaml` or `configs/diffusion.yaml` for full options under `logging.metrics.tensorboard`.
````

#### Task 6.2: Create User Guide

**File**: `docs/features/20260218_tensorboard-user-guide.md` (NEW)

Content: configuration reference, usage examples, visualization guide, troubleshooting, performance considerations.

#### Task 6.3: Update Architecture Documentation

**File**: `docs/standards/architecture.md` — update logging section to document hybrid logging strategy.

---

## Risk Mitigation

| Risk                     | Mitigation                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| **Performance overhead** | TensorBoard disabled by default; reasonable `flush_secs` (30s); histogram/graph logging opt-in only |
| **Disk space usage**     | Document expected usage; provide cleanup guidance                                                   |
| **Missing dependencies** | Graceful degradation with warning; clear install instructions                                       |
| **Breaking changes**     | TensorBoard disabled by default; CSV logging unchanged; comprehensive regression tests              |

## Success Criteria

- TensorBoard logging works alongside CSV logging
- Users can enable/disable via single config change
- Logs scalars, images, and hyperparameters
- Zero breaking changes to existing functionality
- Graceful handling of missing tensorboard package
- Code coverage > 80% for new code
- Performance overhead < 5% with TensorBoard enabled

## Timeline Estimate

| Phase                        | Tasks        | Estimated Time  |
| ---------------------------- | ------------ | --------------- |
| Phase 1: Foundation          | 4 tasks      | 2-3 hours       |
| Phase 2: Logger Integration  | 3 tasks      | 4-5 hours       |
| Phase 3: Configuration       | 3 tasks      | 1 hour          |
| Phase 4: Trainer Integration | 3 tasks      | 2-3 hours       |
| Phase 5: Testing             | 6 tasks      | 4-5 hours       |
| Phase 6: Documentation       | 3 tasks      | 2-3 hours       |
| **Total**                    | **22 tasks** | **15-20 hours** |

## Future Enhancements

- TensorBoard profiler plugin for performance analysis
- Remote TensorBoard hosting guide
- Integration with Weights & Biases / MLflow
- Custom TensorBoard plugins for domain-specific visualizations

---

## Reference

### TensorBoard Commands

```bash
# Basic usage
tensorboard --logdir outputs/experiment/tensorboard

# Custom port
tensorboard --logdir outputs/experiment/tensorboard --port 6007

# Multiple experiments comparison
tensorboard --logdir_spec=exp1:outputs/exp1/tensorboard,exp2:outputs/exp2/tensorboard

# Remote access (SSH tunnel)
ssh -L 6006:localhost:6006 user@remote-server
tensorboard --logdir outputs/experiment/tensorboard --bind_all

# Auto-reload for live updates
tensorboard --logdir outputs/experiment/tensorboard --reload_interval 5
```

### Metrics Naming Conventions

**Classifier**: `metrics/train_loss`, `metrics/train_accuracy`, `metrics/val_loss`, `metrics/val_accuracy`, `metrics/learning_rate`, `images/predictions`, `confusion_matrix/matrix`

**Diffusion**: `metrics/train_loss`, `metrics/val_loss`, `metrics/ema_decay`, `images/samples`, `denoising/process`, `denoising/figure`

### Configuration Key Reference

| Key                                          | Type      | Default | Description                      |
| -------------------------------------------- | --------- | ------- | -------------------------------- |
| `logging.metrics.csv.enabled`                | bool      | true    | Enable CSV logging (always true) |
| `logging.metrics.tensorboard.enabled`        | bool      | false   | Enable TensorBoard logging       |
| `logging.metrics.tensorboard.log_dir`        | str\|null | null    | Custom TensorBoard directory     |
| `logging.metrics.tensorboard.flush_secs`     | int       | 30      | Flush frequency (seconds)        |
| `logging.metrics.tensorboard.log_images`     | bool      | true    | Log images to TensorBoard        |
| `logging.metrics.tensorboard.log_histograms` | bool      | false   | Log weight/gradient histograms   |
| `logging.metrics.tensorboard.log_graph`      | bool      | false   | Log model computational graph    |
