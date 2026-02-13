# Diffusion Configuration Migration Guide

This guide helps users migrate from the old configuration structure to the optimized structure introduced in February 2026.

**Created:** February 13, 2026  
**Status:** Active

---

## Overview

The diffusion configuration has been restructured to better separate common, training-specific, and generation-specific parameters. This improves clarity, reduces duplication, and makes the configuration easier to understand and maintain.

### Key Benefits

- **Clearer organization**: Parameters grouped by their purpose and usage context
- **Better mode separation**: Training and generation parameters clearly separated
- **Logical nesting**: Related parameters (validation, visualization) nested under their parent context
- **Reduced duplication**: Common parameters (device, seed) specified once at top level

---

## Quick Migration Table

| Old Location                   | New Location                               | Notes                     |
| ------------------------------ | ------------------------------------------ | ------------------------- |
| `training.device`              | `device` (top level)                       | Now common parameter      |
| `training.seed`                | `seed` (top level)                         | Now common parameter      |
| `generation.sample_images`     | `training.visualization.sample_images`     | Training visualization    |
| `generation.sample_interval`   | `training.visualization.sample_interval`   | Training visualization    |
| `generation.samples_per_class` | `training.visualization.samples_per_class` | Training visualization    |
| `output.checkpoint_dir`        | `training.checkpoint_dir`                  | Training-specific         |
| `output.save_best_only`        | `training.save_best_only`                  | Training-specific         |
| `output.save_frequency`        | `training.save_frequency`                  | Training-specific         |
| `validation.*` (top level)     | `training.validation.*`                    | Now nested under training |
| `checkpoint` (top level)       | `generation.checkpoint`                    | Generation-specific       |
| `num_samples` (top level)      | `generation.num_samples`                   | Generation-specific       |
| `output_dir` (top level)       | `generation.output_dir`                    | Generation-specific       |

---

## Before and After Examples

### Training Configuration

#### Before (Old Structure)

```yaml
experiment: diffusion
mode: train

model:
  image_size: 64
  # ... model config ...

data:
  train_path: data/train
  # ... data config ...

training:
  device: cuda
  seed: 42
  epochs: 200
  learning_rate: 0.0001
  optimizer: adam
  use_ema: true
  ema_decay: 0.9999

generation:
  sample_images: true
  sample_interval: 10
  samples_per_class: 2
  guidance_scale: 3.0

output:
  checkpoint_dir: outputs/checkpoints
  save_frequency: 10
  save_best_only: false
  log_dir: outputs/logs

validation:
  frequency: 1
  metric: loss
```

#### After (New Structure)

```yaml
experiment: diffusion
mode: train

# Common parameters at top level
device: cuda
seed: 42

model:
  image_size: 64
  # ... model config ...

data:
  train_path: data/train
  # ... data config ...

output:
  log_dir: outputs/logs # Only common output parameter

training:
  # Core training parameters
  epochs: 200
  learning_rate: 0.0001
  optimizer: adam
  use_ema: true
  ema_decay: 0.9999

  # Training checkpointing (moved from output)
  checkpoint_dir: outputs/checkpoints
  save_frequency: 10
  save_best_only: false

  # Nested validation (moved from top level)
  validation:
    frequency: 1
    metric: loss

  # Nested visualization (moved from generation)
  visualization:
    sample_images: true
    sample_interval: 10
    samples_per_class: 2
    guidance_scale: 3.0

# Generation section (for parameters used in generate mode)
generation:
  checkpoint: null # Required when mode=generate
  num_samples: 100
  guidance_scale: 3.0
  use_ema: true
  output_dir: null # Defaults to log_dir/generated
  save_grid: true
  grid_nrow: 10
```

---

### Generation Configuration

#### Before (Old Structure)

```yaml
mode: generate
checkpoint: outputs/checkpoints/model_epoch_200.pth
num_samples: 100
output_dir: outputs/generated

training:
  device: cuda
  use_ema: true

generation:
  guidance_scale: 3.0
```

#### After (New Structure)

```yaml
experiment: diffusion
mode: generate

# Common parameters at top level
device: cuda
seed: null # Optional for generation

model:
  # ... model config (must match training) ...

output:
  log_dir: outputs/logs

generation:
  # Generation input
  checkpoint: outputs/checkpoints/model_epoch_200.pth

  # Generation parameters
  num_samples: 100
  guidance_scale: 3.0
  use_ema: true

  # Generation output
  output_dir: outputs/generated
  save_grid: true
  grid_nrow: 10
```

---

## Step-by-Step Migration Instructions

### For Training Configurations

1. **Move common parameters to top level:**

   ```yaml
   # OLD:
   training:
     device: cuda
     seed: 42

   # NEW:
   device: cuda # Top level
   seed: 42 # Top level
   ```

2. **Move checkpointing parameters to training section:**

   ```yaml
   # OLD:
   output:
     checkpoint_dir: outputs/checkpoints
     save_frequency: 10
     save_best_only: false

   # NEW:
   training:
     checkpoint_dir: outputs/checkpoints
     save_frequency: 10
     save_best_only: false
   ```

3. **Nest validation under training:**

   ```yaml
   # OLD:
   validation:
     frequency: 1
     metric: loss

   # NEW:
   training:
     validation:
       frequency: 1
       metric: loss
   ```

4. **Move training visualization from generation to training:**

   ```yaml
   # OLD:
   generation:
     sample_images: true
     sample_interval: 10
     samples_per_class: 2

   # NEW:
   training:
     visualization:
       sample_images: true
       sample_interval: 10
       samples_per_class: 2
       guidance_scale: 3.0 # Use same as generation.guidance_scale
   ```

5. **Add generation section with defaults:**

   ```yaml
   # NEW (add this entire section):
   generation:
     checkpoint: null
     num_samples: 100
     guidance_scale: 3.0
     use_ema: true
     output_dir: null
     save_grid: true
     grid_nrow: 10
   ```

6. **Update output section:**

   ```yaml
   # OLD:
   output:
     checkpoint_dir: outputs/checkpoints
     save_frequency: 10
     save_best_only: false
     log_dir: outputs/logs

   # NEW:
   output:
     log_dir: outputs/logs  # Only keep log_dir
   ```

### For Generation Configurations

1. **Move common parameters to top level:**

   ```yaml
   # OLD:
   training:
     device: cuda

   # NEW:
   device: cuda # Top level
   seed: null # Optional for generation
   ```

2. **Move generation parameters to generation section:**

   ```yaml
   # OLD:
   checkpoint: path/to/model.pth
   num_samples: 100
   output_dir: outputs/generated

   # NEW:
   generation:
     checkpoint: path/to/model.pth
     num_samples: 100
     output_dir: outputs/generated
     guidance_scale: 3.0
     use_ema: true
     save_grid: true
     grid_nrow: 10
   ```

3. **Ensure all required sections exist:**
   - Model config (must match training)
   - Output config with log_dir
   - Generation config with all parameters

---

## Validation

After migrating your configuration, validate it:

### Method 1: Use validation function

```python
from src.experiments.diffusion.config import validate_config
import yaml

with open('your_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

validate_config(config)  # Will raise error if invalid
print("Configuration is valid!")
```

### Method 2: Try running with your config

```bash
# For training
python -m src.main your_training_config.yaml

# For generation
python -m src.main your_generation_config.yaml
```

---

## Common Migration Issues

### Issue 1: Missing device parameter

**Error:** `KeyError: 'device'`

**Solution:** Move device from `training.device` to top-level `device`:

```yaml
# Add at top level:
device: cuda # or cpu, auto
```

### Issue 2: Missing checkpoint_dir in training

**Error:** `KeyError: 'checkpoint_dir'` during training

**Solution:** Move checkpoint_dir from `output.checkpoint_dir` to `training.checkpoint_dir`:

```yaml
training:
  checkpoint_dir: outputs/checkpoints
```

### Issue 3: Missing validation section

**Error:** `KeyError: 'validation'` in training config

**Solution:** Move validation from top level to nested under training:

```yaml
# OLD:
validation:
  frequency: 1

# NEW:
training:
  validation:
    frequency: 1
```

### Issue 4: Missing visualization section

**Error:** `KeyError: 'visualization'` in training config

**Solution:** Create visualization section under training with parameters from generation:

```yaml
training:
  visualization:
    sample_images: true
    sample_interval: 10
    samples_per_class: 2
    guidance_scale: 3.0
```

### Issue 5: Missing generation.checkpoint

**Error:** `ValueError: generation.checkpoint is required when mode is generate`

**Solution:** Move checkpoint from top level to generation section:

```yaml
# OLD:
checkpoint: path/to/model.pth

# NEW:
generation:
  checkpoint: path/to/model.pth
```

---

## Backward Compatibility

**Important:** The old configuration structure is no longer supported as of February 2026. All configurations must be migrated to the new structure.

If you encounter issues with migration:

1. Check this guide for common issues
2. Compare your config with examples in `configs/diffusion/`
3. Use the validation function to identify specific problems
4. Refer to the complete example: `configs/diffusion/default.yaml`

---

## Additional Resources

- **Complete example config:** [configs/diffusion/default.yaml](../../configs/diffusion/default.yaml)
- **Optimization report:** [diffusion-config-optimization-report.md](./diffusion-config-optimization-report.md)
- **Implementation tasks:** [diffusion-config-implementation-tasks.md](./diffusion-config-implementation-tasks.md)
- **README configuration section:** [README.md](../../README.md#configuration)

---

## Questions or Issues?

If you encounter problems not covered in this guide:

1. Check the test configurations in `tests/fixtures/configs/diffusion/`
2. Review the source code: `src/experiments/diffusion/config.py`
3. Examine integration tests for usage examples: `tests/integration/test_diffusion_pipeline.py`

---

**Last Updated:** February 13, 2026
