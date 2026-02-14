# Classifier Configuration Optimization Plan

**Date:** February 14, 2026  
**Status:** Proposed  
**Related:** [diffusion-config-v2-optimization-plan.md](20260213_diffusion-config-v2-optimization-plan.md)

---

## Executive Summary

This document outlines the proposed optimization changes for the classifier configuration structure. These changes align with the diffusion config V2 optimization principles: eliminate duplication, group related parameters logically, maintain explicit configuration, and provide clear separation of concerns.

**Key Principles:**

- ✅ Eliminate parameter duplication
- ✅ Group related parameters logically
- ✅ Maintain explicit configuration (no "auto" magic)
- ✅ Clear separation of concerns
- ✅ Single source of truth for output directories
- ✅ Consistency with diffusion config structure

---

## Change Summary

### 1. Add Compute Configuration Section

**Issue:** Device and seed are buried in training section, but they're global compute settings.

**Current:**

```yaml
training:
  device: cuda
  seed: null
  # ... other training params
```

**Change:**

```yaml
compute:
  device: cuda # Options: cuda, cpu, auto
  seed: null # Random seed for reproducibility (null to disable)
```

**Impact:**

- Consistent with diffusion config structure
- Clearer scope of these settings
- Code changes: `training["device"]` → `compute["device"]`

---

### 2. Group Optimizer Parameters

**Issue:** Optimizer configuration is split between `learning_rate`, `optimizer`, and `optimizer_kwargs`.

**Current:**

```yaml
training:
  learning_rate: 0.001
  optimizer: adam
  optimizer_kwargs:
    weight_decay: 0.0001
```

**Change:**

```yaml
training:
  optimizer:
    type: adam # Options: adam, adamw, sgd
    learning_rate: 0.001
    weight_decay: 0.0001
    # For SGD: momentum: 0.9
    # For Adam/AdamW: betas: [0.9, 0.999]
    gradient_clip_norm: null # Max gradient norm (null to disable)
```

**Impact:**

- All optimizer settings in one place
- Matches diffusion config structure
- Moved `gradient_clip` into optimizer group
- Code changes in optimizer initialization

---

### 3. Group Scheduler Parameters

**Issue:** Scheduler configuration is split between `scheduler` and `scheduler_kwargs`.

**Current:**

```yaml
training:
  scheduler: cosine
  scheduler_kwargs:
    T_max: 100
    eta_min: 0.000001
```

**Change:**

```yaml
training:
  scheduler:
    type: cosine # Options: cosine, step, plateau, null
    T_max: auto # Auto = epochs, or specify number
    eta_min: 1.0e-6
    # For step: step_size: 30, gamma: 0.1
    # For plateau: mode: min, factor: 0.1, patience: 10
```

**Impact:**

- Consistent structure with optimizer
- Matches diffusion config structure
- Auto-configuration option for T_max
- Code changes in scheduler initialization

---

### 4. Unified Output Directory Structure

**Issue:** Output directories use different base paths, no unified structure.

**Current:**

```yaml
output:
  checkpoint_dir: outputs/checkpoints
  log_dir: outputs/logs
  save_best_only: true
  save_frequency: 10
```

**Change:**

```yaml
output:
  base_dir: outputs
  subdirs:
    logs: logs
    checkpoints: checkpoints

training:
  checkpointing:
    save_frequency: 10
    save_best_only: true
    save_optimizer: true # Include optimizer state in checkpoint
```

**Resolved paths:**

- Logs: `outputs/logs`
- Checkpoints: `outputs/checkpoints`

**Impact:**

- Single source of truth for base output directory
- Consistent with diffusion config
- Checkpointing parameters moved to training section
- Code changes in path construction

---

### 5. Restructure Data Configuration

**Issue:** Data parameters could be better organized into logical groups.

**Current:**

```yaml
data:
  train_path: data/train
  val_path: data/val
  batch_size: 32
  num_workers: 4
  pin_memory: true
  shuffle_train: true
  drop_last: false
  image_size: 256
  crop_size: 224
  normalize: imagenet
  horizontal_flip: true
  color_jitter: false
  rotation_degrees: 0
```

**Change:**

```yaml
data:
  # Paths: Dataset locations
  paths:
    train: data/train # Training data directory (required)
    val: data/val # Validation data directory (null for no validation)

  # Loading: DataLoader parameters
  loading:
    batch_size: 32
    num_workers: 4
    pin_memory: true
    shuffle_train: true
    drop_last: false

  # Preprocessing: Image preprocessing before augmentation
  preprocessing:
    image_size: 256 # Resize to this size
    crop_size: 224 # Crop to this size (224 for ResNet, 299 for InceptionV3)
    normalize: imagenet # Options: imagenet, cifar10, custom, null

  # Augmentation: Data augmentation settings
  augmentation:
    horizontal_flip: true
    rotation_degrees: 0
    color_jitter:
      enabled: false
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
```

**Impact:**

- Logical grouping: paths, loading, preprocessing, augmentation
- Consistent with diffusion config structure
- More detailed color_jitter options
- Code changes in data configuration parsing

---

### 6. Restructure Validation Configuration

**Issue:** Validation should be nested under training for consistency.

**Current:**

```yaml
training:
  # ... training params
  early_stopping_patience: null

validation:
  frequency: 1
  metric: accuracy
```

**Change:**

```yaml
training:
  validation:
    enabled: true # Explicit flag (false if data.paths.val is null)
    frequency: 1
    metric: accuracy # Options: accuracy, loss, f1, precision, recall
    early_stopping_patience: null # Epochs to wait (null to disable)
```

**Impact:**

- Validation scoped to training
- Consistent with diffusion config
- Early stopping moved into validation group
- Explicit enabled flag
- Code changes in validation setup

---

### 7. Restructure Model Configuration

**Issue:** Model parameters could be better organized.

**Current:**

```yaml
model:
  name: resnet50
  pretrained: true
  num_classes: 2
  freeze_backbone: false
  trainable_layers: null
  dropout: 0.5
```

**Change:**

```yaml
model:
  architecture:
    name: resnet50 # Options: resnet50, resnet101, resnet152, inceptionv3, custom
    num_classes: 2

  initialization:
    pretrained: true # Use ImageNet pretrained weights
    freeze_backbone: false # Freeze all backbone layers
    trainable_layers: null # List of layer patterns to unfreeze (null = all)

  regularization:
    dropout: 0.5 # Dropout rate (model-specific, e.g., InceptionV3)
```

**Impact:**

- Clear separation: architecture, initialization, regularization
- Better organization of model settings
- Code changes in model initialization

---

### 8. Add Performance Configuration

**Issue:** Performance settings like mixed precision should be in a dedicated section.

**Current:**

```yaml
training:
  mixed_precision: false
  gradient_clip: null
```

**Change:**

```yaml
training:
  performance:
    use_amp: false # Automatic mixed precision (CUDA only)
    use_tf32: true # Enable TF32 on Ampere+ GPUs
    cudnn_benchmark: true # cuDNN benchmark mode
    compile_model: false # Use torch.compile (PyTorch 2.0+)
```

**Note:** `gradient_clip` moved to `training.optimizer.gradient_clip_norm`

**Impact:**

- Consistent with diffusion config
- New performance optimization options
- Code changes for performance settings

---

### 9. Add Resume Training Configuration

**New feature:** Explicit support for resuming training from checkpoint.

**Addition:**

```yaml
training:
  resume:
    enabled: false # Explicit flag to enable resume mode
    checkpoint: null # Path to checkpoint to resume from (required if enabled)
    reset_optimizer: false
    reset_scheduler: false
```

**Impact:**

- Consistent with diffusion config
- Clear distinction between new training and resuming
- Explicit control over optimizer/scheduler state restoration
- New code required for resume logic

---

### 10. Add Evaluation Mode Configuration

**New feature:** Separate configuration for evaluation/inference mode.

**Addition:**

```yaml
experiment: classifier
mode: train # Options: train, evaluate

evaluation:
  checkpoint: null # Path to trained model checkpoint (required)

  data:
    test_path: data/test # Test data directory
    batch_size: 32 # Can differ from training batch size

  output:
    save_predictions: true # Save predictions to file
    save_confusion_matrix: true # Save confusion matrix
    save_metrics: true # Save detailed metrics
```

**Impact:**

- Explicit evaluation mode
- Clear separation from training
- New code required for evaluation mode

---

## Full V2 Configuration Structure

```yaml
# ==============================================================================
# CLASSIFIER CONFIGURATION (V2 FORMAT)
# ==============================================================================

experiment: classifier
mode: train # Options: train, evaluate

# ==============================================================================
# COMPUTE CONFIGURATION
# ==============================================================================

compute:
  device: cuda # Options: cuda, cpu, auto
  seed: null # Random seed for reproducibility (null to disable)

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

model:
  # Architecture: Model type and structure
  architecture:
    name: resnet50 # Options: resnet50, resnet101, resnet152, inceptionv3
    num_classes: 2

  # Initialization: Pretrained weights and layer freezing
  initialization:
    pretrained: true # Use ImageNet pretrained weights
    freeze_backbone: false # Freeze all backbone layers
    trainable_layers: null # List of layer patterns to unfreeze (null = all)

  # Regularization: Model-specific regularization
  regularization:
    dropout: 0.5 # Dropout rate (InceptionV3 only)

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

data:
  # Paths: Dataset locations
  paths:
    train: data/train # Training data directory (required)
    val: data/val # Validation data directory (null for no validation)

  # Loading: DataLoader parameters
  loading:
    batch_size: 32
    num_workers: 4
    pin_memory: true
    shuffle_train: true
    drop_last: false

  # Preprocessing: Image preprocessing before augmentation
  preprocessing:
    image_size: 256 # Resize to this size
    crop_size: 224 # Crop to this size (224 for ResNet, 299 for InceptionV3)
    normalize: imagenet # Options: imagenet, cifar10, custom, null

  # Augmentation: Data augmentation settings
  augmentation:
    horizontal_flip: true
    rotation_degrees: 0
    color_jitter:
      enabled: false
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

output:
  base_dir: outputs
  subdirs:
    logs: logs
    checkpoints: checkpoints

# ==============================================================================
# TRAINING MODE CONFIGURATION
# ==============================================================================

training:
  # Core Training Parameters
  epochs: 100

  # Optimizer: Optimization algorithm and parameters
  optimizer:
    type: adam # Options: adam, adamw, sgd
    learning_rate: 0.001
    weight_decay: 0.0001
    gradient_clip_norm: null # Max gradient norm (null to disable)
    # For SGD: momentum: 0.9
    # For Adam/AdamW: betas: [0.9, 0.999]

  # Scheduler: Learning rate scheduling
  scheduler:
    type: cosine # Options: cosine, step, plateau, null
    T_max: auto # Auto = epochs, or specify number
    eta_min: 1.0e-6
    # For step: step_size: 30, gamma: 0.1
    # For plateau: mode: min, factor: 0.1, patience: 10

  # Checkpointing: Model checkpoint saving
  checkpointing:
    save_frequency: 10
    save_best_only: true
    save_optimizer: true

  # Validation: Validation during training
  validation:
    enabled: true
    frequency: 1
    metric: accuracy # Options: accuracy, loss, f1, precision, recall
    early_stopping_patience: null # Epochs to wait (null to disable)

  # Performance: Performance optimization settings
  performance:
    use_amp: false # Automatic mixed precision (CUDA only)
    use_tf32: true # Enable TF32 on Ampere+ GPUs
    cudnn_benchmark: true # cuDNN benchmark mode
    compile_model: false # Use torch.compile (PyTorch 2.0+)

  # Resume: Resume training from checkpoint
  resume:
    enabled: false
    checkpoint: null # Path to checkpoint (required if enabled)
    reset_optimizer: false
    reset_scheduler: false

# ==============================================================================
# EVALUATION MODE CONFIGURATION
# ==============================================================================

evaluation:
  checkpoint: null # Path to trained model checkpoint (required)

  data:
    test_path: data/test
    batch_size: 32

  output:
    save_predictions: true
    save_confusion_matrix: true
    save_metrics: true
```

---

## Implementation Checklist

### Phase 1: Configuration File Updates

- [ ] Create new `configs/classifier/default.yaml` with V2 structure
- [ ] Keep old config as `configs/classifier/legacy.yaml` for reference
- [ ] Create example configs for different models (ResNet50, ResNet101, InceptionV3)

### Phase 2: Code Updates - Configuration Loading

**File: `src/utils/config.py`**

- [ ] Add classifier config validation function
- [ ] Add path resolution helper (if not already added for diffusion)
- [ ] Add backward compatibility layer (optional)

**File: `src/main.py` - `setup_experiment_classifier()`**

- [ ] Update device/seed access: `config["compute"]["device"]`
- [ ] Update output path construction using `output.base_dir` + `output.subdirs`
- [ ] Update optimizer initialization to use `training.optimizer.*`
- [ ] Update scheduler initialization to use `training.scheduler.*`
- [ ] Update validation setup to use `training.validation.*`
- [ ] Update checkpointing to use `training.checkpointing.*`
- [ ] Add resume logic for `training.resume.*`
- [ ] Add evaluation mode support

### Phase 3: Code Updates - Data Loading

**File: `src/base/dataloader.py` or `src/data/datasets.py`**

- [ ] Update data config parsing for new structure
- [ ] Update path access: `data.paths.train`, `data.paths.val`
- [ ] Update loading params: `data.loading.*`
- [ ] Update preprocessing: `data.preprocessing.*`
- [ ] Update augmentation: `data.augmentation.*`
- [ ] Add support for expanded color_jitter options

### Phase 4: Code Updates - Model Initialization

**File: `src/experiments/classifier/model.py`**

- [ ] Update model initialization for `model.architecture.*`
- [ ] Update pretrained loading for `model.initialization.*`
- [ ] Update layer freezing logic
- [ ] Update dropout for `model.regularization.*`

### Phase 5: Code Updates - Training

**File: `src/experiments/classifier/trainer.py`**

- [ ] Update checkpoint path construction
- [ ] Update validation setup (enabled, frequency, metric)
- [ ] Update early stopping logic
- [ ] Update performance settings (AMP, TF32, compile, etc.)
- [ ] Add resume functionality
- [ ] Update optimizer to use grouped config

### Phase 6: Code Updates - Evaluation Mode

**File: `src/experiments/classifier/evaluator.py` (may need to create)**

- [ ] Implement evaluation mode
- [ ] Load checkpoint and model
- [ ] Run inference on test data
- [ ] Save predictions, confusion matrix, metrics
- [ ] Generate evaluation report

### Phase 7: Testing

- [ ] Update unit tests in `tests/experiments/classifier/`
- [ ] Update integration tests
- [ ] Test backward compatibility (if implemented)
- [ ] Test all modes: train, evaluate, resume
- [ ] Test with different model architectures

### Phase 8: Documentation

- [ ] Update README.md with new configuration structure
- [ ] Create migration guide from V1 to V2
- [ ] Update configuration examples in documentation
- [ ] Document evaluation mode usage

---

## Migration Path for Users

### Option 1: Manual Migration

Users manually update their configs following the new structure.

### Option 2: Migration Script

Provide a script to automatically convert old configs to new structure:

```bash
python scripts/migrate_classifier_config_v1_to_v2.py --input old_config.yaml --output new_config.yaml
```

### Option 3: Backward Compatibility Layer

Add code to automatically translate old config keys to new keys with deprecation warnings.

**Recommended:** Combination of Option 2 (migration script) + Option 3 (temporary compatibility layer with warnings)

---

## Benefits Summary

| Benefit                    | Impact                                             |
| -------------------------- | -------------------------------------------------- |
| **Consistency**            | Matches diffusion config structure                 |
| **Logical grouping**       | 7 parameter groups restructured                    |
| **Single source of truth** | Unified output directory structure                 |
| **Clearer intent**         | Mode-specific sections (train, evaluate)           |
| **Easier maintenance**     | Related parameters grouped together                |
| **New features**           | Resume training, evaluation mode, performance opts |
| **Better UX**              | One base directory for all outputs                 |

---

## Comparison with Diffusion Config V2

| Aspect          | Diffusion V2               | Classifier V2                |
| --------------- | -------------------------- | ---------------------------- |
| Compute section | ✅ Yes                     | ✅ Yes                       |
| Optimizer group | ✅ Yes                     | ✅ Yes                       |
| Scheduler group | ✅ Yes                     | ✅ Yes                       |
| Output unified  | ✅ Yes                     | ✅ Yes                       |
| Resume support  | ✅ Yes                     | ✅ Yes                       |
| Performance     | ✅ Yes                     | ✅ Yes                       |
| Mode support    | train, generate            | train, evaluate              |
| EMA support     | ✅ Yes (training)          | ❌ No (not typical for cls.) |
| Conditioning    | ✅ Yes (class-conditional) | N/A                          |
| Augmentation    | Basic (flip, rotation)     | Extended (color_jitter)      |

---

## Risks & Mitigation

| Risk                                  | Mitigation                                        |
| ------------------------------------- | ------------------------------------------------- |
| Breaking changes for existing configs | Provide migration script + backward compatibility |
| Code changes across many files        | Comprehensive testing suite                       |
| User confusion during transition      | Clear documentation + examples                    |
| Different structure from old configs  | Migration guide with side-by-side comparison      |

---

## Timeline Estimate

| Phase                    | Estimated Time | Priority |
| ------------------------ | -------------- | -------- |
| Phase 1: Config files    | 2 hours        | HIGH     |
| Phase 2: Config loading  | 3 hours        | HIGH     |
| Phase 3: Data loading    | 2 hours        | HIGH     |
| Phase 4: Model init      | 2 hours        | MEDIUM   |
| Phase 5: Training logic  | 3 hours        | HIGH     |
| Phase 6: Evaluation mode | 3 hours        | MEDIUM   |
| Phase 7: Testing         | 4 hours        | HIGH     |
| Phase 8: Documentation   | 2 hours        | MEDIUM   |
| **Total**                | **21 hours**   | -        |

---

## Next Steps

1. ✅ Document approved changes (this document)
2. ⏳ Review and approve optimization plan
3. ⏳ Create new V2 configuration file
4. ⏳ Implement Phase 1: Configuration file updates
5. ⏳ Implement Phase 2-8: Code changes and testing

---

## References

- [Diffusion V2 Optimization Plan](20260213_diffusion-config-v2-optimization-plan.md)
- [Diffusion Config Migration Guide](20260213_diffusion-config-migration-guide.md)
- Current classifier config: `configs/classifier/default.yaml`
- Current diffusion V2 config: `configs/diffusion/default.yaml`
