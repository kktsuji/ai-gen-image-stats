# Diffusion Configuration V2 Optimization Plan

**Date:** February 13, 2026  
**Status:** Approved for Implementation  
**Related:** [diffusion-config-optimization-report.md](diffusion-config-optimization-report.md)

---

## Executive Summary

This document outlines the approved optimization changes for the diffusion model configuration structure. These changes address parameter organization issues, eliminate duplication, and improve maintainability while keeping the configuration explicit and user-friendly.

**Key Principles:**

- ✅ Eliminate parameter duplication
- ✅ Group related parameters logically
- ✅ Maintain explicit configuration (no "auto" magic)
- ✅ Clear separation of concerns
- ✅ Single source of truth for output directories

**Excluded from scope:**

- ❌ Preset configurations (not needed)
- ❌ Auto-configuration features (keep explicit)

---

## Change Summary

### 1. Consolidate Duplicate Parameters

**Issue:** `image_size` is defined in both `model` and `data` sections, creating a risk of mismatch.

**Current:**

```yaml
model:
  image_size: 40

data:
  image_size: 40 # Must match model.image_size
```

**Change:**

```yaml
model:
  image_size: 40

data:
  # image_size removed - derived from model.image_size in code
```

**Impact:**

- Code must derive `image_size` from `model.image_size`
- Eliminates configuration mismatch errors

---

### 2. Group Related Parameters (Optimizer)

**Issue:** Optimizer configuration is split between multiple keys at the same level.

**Current:**

```yaml
training:
  learning_rate: 0.0001
  optimizer: adam
  optimizer_kwargs:
    weight_decay: 0.0
    betas: [0.9, 0.999]
```

**Change:**

```yaml
training:
  optimizer:
    type: adam
    learning_rate: 0.0001
    weight_decay: 0.0
    betas: [0.9, 0.999]
```

**Impact:**

- Better organization of related parameters
- Easier to understand optimizer configuration at a glance
- Code changes required in optimizer initialization

---

### 3. Group Related Parameters (Scheduler)

**Issue:** Scheduler configuration is split between multiple keys.

**Current:**

```yaml
training:
  scheduler: null
  scheduler_kwargs:
    T_max: 200
    eta_min: 1.0e-6
```

**Change:**

```yaml
training:
  scheduler:
    type: null # Options: cosine, step, plateau, null
    T_max: auto # Auto = epochs
    eta_min: 1.0e-6
```

**Impact:**

- Consistent structure with optimizer configuration
- Code changes required in scheduler initialization

---

### 4. Group Related Parameters (EMA)

**Issue:** EMA parameters are scattered at training level.

**Current:**

```yaml
training:
  use_ema: true
  ema_decay: 0.9999
```

**Change:**

```yaml
training:
  ema:
    enabled: true
    decay: 0.9999
```

**Impact:**

- Consistent naming pattern with other feature groups
- Code changes: `training.get("use_ema")` → `training.get("ema", {}).get("enabled")`

---

### 5. Unified Output Directory Structure

**Issue:** Output directories are scattered across multiple sections with repeated base paths.

**Current:**

```yaml
output:
  log_dir: outputs/logs

training:
  checkpoint_dir: outputs/checkpoints

generation:
  output_dir: null # defaults to log_dir/generated
```

**Change:**

```yaml
output:
  base_dir: outputs
  subdirs:
    logs: logs
    checkpoints: checkpoints
    samples: samples # Training visualization
    generated: generated # Generation mode output
```

**Resolved paths:**

- Logs: `outputs/logs`
- Checkpoints: `outputs/checkpoints`
- Training samples: `outputs/samples`
- Generated images: `outputs/generated`

**Impact:**

- Single source of truth for base output directory
- Easy to change all output locations at once
- Code changes required in all path construction logic

---

### 6. Simplify Conditional Model Configuration

**Issue:** Conditioning-related parameters are scattered and unclear.

**Current:**

```yaml
model:
  num_classes: null
  class_dropout_prob: 0.1

data:
  return_labels: false
```

**Change:**

```yaml
model:
  conditioning:
    type: null # Options: null (unconditional), "class"
    num_classes: null # Required if type="class"
    class_dropout_prob: 0.1 # Only used if type="class"

data:
  # return_labels removed - derived from model.conditioning.type
```

**Impact:**

- Clearer intent of conditioning setup
- Automatic derivation of `return_labels` based on conditioning type
- Code changes in data loader to derive `return_labels`

---

### 7. Restructure Training Checkpointing

**Issue:** Checkpoint-related parameters need better organization.

**Current:**

```yaml
training:
  checkpoint_dir: outputs/checkpoints
  save_best_only: false
  save_frequency: 10
```

**Change:**

```yaml
training:
  checkpointing:
    save_frequency: 10
    save_best_only: false
    save_optimizer: true # Include optimizer state in checkpoint
```

**Note:** Checkpoint directory moved to `output.subdirs.checkpoints`

**Impact:**

- Consistent grouping of checkpoint-related parameters
- New option to control optimizer state saving
- Code changes in checkpoint saving logic

---

### 8. Restructure Training Validation

**Issue:** Validation section at wrong level (should be under training).

**Current:**

```yaml
training:
  # ...other training params...

validation:
  frequency: 1
  metric: loss
```

**Change:**

```yaml
training:
  validation:
    enabled: true
    frequency: 1
    metric: loss
```

**Impact:**

- Validation is clearly scoped to training mode
- New `enabled` flag for explicit control
- Code changes in validation setup logic

---

### 9. Restructure Training Visualization

**Issue:** Sample generation during training is in wrong section (currently under `generation`).

**Current:**

```yaml
generation:
  sample_images: true
  sample_interval: 10
  samples_per_class: 2
  guidance_scale: 3.0
```

**Change:**

```yaml
training:
  visualization:
    enabled: true
    interval: 10
    num_samples: 8 # Total samples (replaces samples_per_class)
    guidance_scale: 3.0
```

**Note:** Samples saved to `output.subdirs.samples`

**Impact:**

- Visualization clearly scoped to training mode
- Renamed parameters for clarity
- Code changes in sample generation during training

---

### 10. Add Resume Training Configuration

**New feature:** Explicit support for resuming training from checkpoint.

**Addition:**

```yaml
training:
  resume:
    enabled: false # EXPLICIT flag to enable resume mode
    checkpoint: null # Path to checkpoint to resume from
    reset_optimizer: false
    reset_scheduler: false
```

**Impact:**

- Clear distinction between starting new training and resuming
- Explicit control over optimizer/scheduler state restoration
- New code required for resume logic

---

### 11. Add Compute/Performance Configuration

**New feature:** Centralize compute and performance settings.

**Current:**

```yaml
device: cuda
seed: null

training:
  use_amp: false
  gradient_clip_norm: null
```

**Change:**

```yaml
compute:
  device: cuda
  seed: null

training:
  optimizer:
    # ...
    gradient_clip_norm: null

  performance:
    use_amp: false
    use_tf32: true # TF32 on Ampere+ GPUs
    cudnn_benchmark: true
    compile_model: false # torch.compile (PyTorch 2.0+)
```

**Impact:**

- Better organization of computation settings
- New performance optimization options
- Code changes to move device/seed access
- New code for performance optimizations

---

### 12. Add Validation Rules Section

**New feature:** Configuration-level validation rules.

**Addition:**

```yaml
validation_rules:
  require_labels_if_conditional: true
  warn_if_batch_size_small: 16

debug:
  profile: false
  log_gradients: false
  detect_anomaly: false
```

**Impact:**

- Early validation of configuration consistency
- New validation code required before training starts
- Debug options for troubleshooting

---

### 13. Restructure Data Configuration

**Issue:** Better organization of data-related parameters.

**Current:**

```yaml
data:
  train_path: data/train
  val_path: null
  batch_size: 32
  num_workers: 4
  # ... augmentation params ...
  pin_memory: true
  drop_last: false
  shuffle_train: true
  return_labels: false
```

**Change:**

```yaml
data:
  paths:
    train: data/train
    val: null

  loading:
    batch_size: 32
    num_workers: 4
    pin_memory: true
    shuffle_train: true
    drop_last: false

  augmentation:
    horizontal_flip: true
    rotation_degrees: 0
    color_jitter:
      enabled: false
      strength: 0.1
```

**Impact:**

- Logical grouping of data parameters
- Easier to find and modify specific settings
- Code changes in data configuration parsing

---

### 14. Restructure Model Configuration

**Issue:** Better organization with architecture vs diffusion parameters.

**Current:**

```yaml
model:
  image_size: 40
  in_channels: 3
  model_channels: 64
  channel_multipliers: [1, 2, 4]
  num_classes: null
  num_timesteps: 1000
  beta_schedule: cosine
  beta_start: 0.0001
  beta_end: 0.02
  class_dropout_prob: 0.1
  use_attention: [false, false, true]
```

**Change:**

```yaml
model:
  architecture:
    image_size: 40
    in_channels: 3
    model_channels: 64
    channel_multipliers: [1, 2, 4]
    use_attention: [false, false, true]

  diffusion:
    num_timesteps: 1000
    beta_schedule: cosine
    beta_start: 0.0001
    beta_end: 0.02

  conditioning:
    type: null
    num_classes: null
    class_dropout_prob: 0.1
```

**Impact:**

- Clear separation of architecture vs diffusion parameters
- Easier to understand model structure
- Code changes in model initialization

---

### 15. Simplify Generation Configuration

**Issue:** Better naming for generation output.

**Current:**

```yaml
generation:
  checkpoint: null
  num_samples: 100
  guidance_scale: 3.0
  use_ema: true
  output_dir: null
  save_grid: true
  grid_nrow: 10
```

**Change:**

```yaml
generation:
  checkpoint: null

  sampling:
    num_samples: 100
    guidance_scale: 3.0
    use_ema: true

  output:
    save_individual: true
    save_grid: true
    grid_nrow: 10
```

**Note:** Output directory moved to `output.subdirs.generated`

**Impact:**

- Logical grouping of generation parameters
- Code changes in generation mode

---

## Implementation Checklist

### Phase 1: Configuration File Updates

- [ ] Create new `configs/diffusion/default.yaml` with V2 structure
- [ ] Keep old config as `configs/diffusion/legacy.yaml` for reference
- [ ] Update all example configs in `tests/fixtures/configs/`

### Phase 2: Code Updates - Configuration Loading

**File: `src/utils/config.py`**

- [ ] Add configuration validation function
- [ ] Add path resolution helper: `resolve_output_path(base_dir, subdir)`
- [ ] Add backward compatibility layer (optional)

**File: `src/main.py` - `setup_experiment_diffusion()`**

- [ ] Update device/seed access: `config["compute"]["device"]`
- [ ] Update output path construction using `output.base_dir` + `output.subdirs`
- [ ] Update optimizer initialization to use `training.optimizer.*`
- [ ] Update scheduler initialization to use `training.scheduler.*`
- [ ] Update EMA initialization to use `training.ema.*`
- [ ] Update validation setup to use `training.validation.*`
- [ ] Update visualization setup to use `training.visualization.*`
- [ ] Add resume logic for `training.resume.*`
- [ ] Update generation mode to use new structure

### Phase 3: Code Updates - Data Loading

**File: `src/base/dataloader.py` or `src/data/datasets.py`**

- [ ] Derive `image_size` from `model.architecture.image_size`
- [ ] Derive `return_labels` from `model.conditioning.type`
- [ ] Update data config parsing for new `data.paths.*`, `data.loading.*`, `data.augmentation.*`

### Phase 4: Code Updates - Model Initialization

**File: `src/experiments/diffusion/model.py`**

- [ ] Update model initialization for `model.architecture.*`
- [ ] Update diffusion parameters for `model.diffusion.*`
- [ ] Update conditioning setup for `model.conditioning.*`

### Phase 5: Code Updates - Training

**File: `src/experiments/diffusion/trainer.py`**

- [ ] Update checkpoint path construction
- [ ] Update validation frequency access
- [ ] Update visualization (sample generation) during training
- [ ] Update performance settings (AMP, TF32, compile, etc.)
- [ ] Add resume functionality

### Phase 6: Testing

- [ ] Update unit tests in `tests/experiments/diffusion/`
- [ ] Update integration tests
- [ ] Test backward compatibility (if implemented)
- [ ] Test all modes: train, generate, resume

### Phase 7: Documentation

- [ ] Update README.md with new configuration structure
- [ ] Create migration guide from V1 to V2
- [ ] Update configuration examples in documentation
- [ ] Update inline documentation/comments

---

## Migration Path for Users

### Option 1: Manual Migration

Users manually update their configs following the new structure.

### Option 2: Migration Script

Provide a script to automatically convert old configs to new structure:

```bash
python scripts/migrate_config_v1_to_v2.py --input old_config.yaml --output new_config.yaml
```

### Option 3: Backward Compatibility Layer

Add code to automatically translate old config keys to new keys with deprecation warnings.

**Recommended:** Combination of Option 2 (migration script) + Option 3 (temporary compatibility layer with warnings)

---

## Benefits Summary

| Benefit                    | Impact                                    |
| -------------------------- | ----------------------------------------- |
| **Single source of truth** | Eliminated 3 duplicate parameters         |
| **Logical grouping**       | 8 parameter groups restructured           |
| **Clearer intent**         | Mode-specific sections properly scoped    |
| **Easier maintenance**     | Related parameters grouped together       |
| **Better UX**              | One base directory for all outputs        |
| **Flexibility**            | Resume training without mode confusion    |
| **Performance**            | New optimization options available        |
| **Validation**             | Early error detection for invalid configs |

---

## Risks & Mitigation

| Risk                                  | Mitigation                                        |
| ------------------------------------- | ------------------------------------------------- |
| Breaking changes for existing configs | Provide migration script + backward compatibility |
| Code changes across many files        | Comprehensive testing suite                       |
| User confusion during transition      | Clear documentation + examples                    |
| Undiscovered edge cases               | Gradual rollout, monitor issues                   |

---

## Timeline Estimate

| Phase                   | Estimated Time | Priority |
| ----------------------- | -------------- | -------- |
| Phase 1: Config files   | 2 hours        | HIGH     |
| Phase 2: Config loading | 4 hours        | HIGH     |
| Phase 3: Data loading   | 2 hours        | HIGH     |
| Phase 4: Model init     | 2 hours        | MEDIUM   |
| Phase 5: Training logic | 4 hours        | HIGH     |
| Phase 6: Testing        | 4 hours        | HIGH     |
| Phase 7: Documentation  | 3 hours        | MEDIUM   |
| **Total**               | **21 hours**   | -        |

---

## Next Steps

1. ✅ Document approved changes (this document)
2. ⏳ Create new V2 configuration file
3. ⏳ Implement Phase 1: Configuration file updates
4. ⏳ Implement Phase 2: Configuration loading updates
5. ⏳ Continue with remaining phases

---

## References

- [Original optimization report](diffusion-config-optimization-report.md)
- [Configuration migration guide](diffusion-config-migration-guide.md) (to be created)
- [User requirements](user-requirements.md)
- [Technical requirements](technical-requirements.md)
