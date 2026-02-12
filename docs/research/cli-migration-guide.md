# CLI Migration Guide: Config-Only Mode

**Date:** February 12, 2026  
**Author:** GitHub Copilot  
**Breaking Change:** YES

## Overview

The CLI has been refactored from a hybrid model (CLI params + JSON config + defaults) to a strict config-only model. This guide helps you migrate existing workflows to the new interface.

## What Changed

### Before (Old CLI)

The old CLI supported three input methods with priority: CLI params > JSON config > Code defaults

```bash
# Multiple ways to configure
python -m src.main --experiment classifier --model resnet50 --epochs 10
python -m src.main --experiment classifier --config config.json --batch-size 64
python -m src.main --experiment classifier  # Uses defaults
```

### After (New CLI)

The new CLI requires a JSON config file as a positional argument. No CLI overrides, no defaults.

```bash
# Single way to configure
python -m src.main configs/classifier/baseline.json
python -m src.main configs/diffusion/default.json
```

## Migration Steps

### Step 1: Identify Your Current Usage Patterns

**Pattern A: CLI Arguments Only**

```bash
# Old
python -m src.main --experiment classifier --model resnet50 \
  --epochs 10 --batch-size 32 --lr 0.001
```

**Migration:** Create a config file with all parameters

```bash
# Create config file
cat > my_config.json << EOF
{
  "experiment": "classifier",
  "model": {
    "name": "resnet50",
    "pretrained": true,
    "num_classes": 2
  },
  "training": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "device": "cuda"
  },
  "data": {
    "train_path": "data/train",
    "val_path": "data/val",
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 299,
    "crop_size": 299
  },
  "output": {
    "checkpoint_dir": "outputs/checkpoints",
    "log_dir": "outputs/logs"
  }
}
EOF

# New
python -m src.main my_config.json
```

**Pattern B: Config + CLI Overrides**

```bash
# Old
python -m src.main --experiment classifier \
  --config configs/classifier/baseline.json --batch-size 64
```

**Migration:** Edit config file or create a new one

```bash
# Option 1: Edit existing config file
# Modify configs/classifier/baseline.json to set batch_size: 64

# Option 2: Create variant config
cp configs/classifier/baseline.json configs/classifier/baseline_bs64.json
# Edit the new file to set batch_size: 64

# New
python -m src.main configs/classifier/baseline_bs64.json
```

**Pattern C: Relying on Defaults**

```bash
# Old
python -m src.main --experiment classifier --model resnet50
```

**Migration:** Use or create a complete config file

```bash
# Use existing complete config
python -m src.main configs/classifier/baseline.json

# Or create your own complete config (see Step 2)
```

### Step 2: Create Complete Config Files

All config files must now include all required fields. Use the examples in `configs/` as templates.

**Required fields for classifier:**

```json
{
  "experiment": "classifier",
  "model": {
    "name": "resnet50",
    "pretrained": true,
    "num_classes": 2
  },
  "data": {
    "train_path": "data/0.Normal",
    "val_path": "data/1.Abnormal",
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 299,
    "crop_size": 299
  },
  "training": {
    "epochs": 50,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "device": "cuda"
  },
  "output": {
    "checkpoint_dir": "outputs/checkpoints",
    "log_dir": "outputs/logs"
  }
}
```

**Required fields for diffusion:**

```json
{
  "experiment": "diffusion",
  "model": {
    "image_size": 64,
    "in_channels": 3,
    "model_channels": 128,
    "num_res_blocks": 2,
    "attention_resolutions": [16, 8],
    "dropout": 0.1,
    "channel_mult": [1, 2, 2, 2],
    "num_timesteps": 1000,
    "beta_schedule": "linear"
  },
  "data": {
    "train_path": "data/data-all/0.Normal",
    "batch_size": 16,
    "num_workers": 4,
    "image_size": 64
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "device": "cuda",
    "save_interval": 10,
    "sample_interval": 10
  },
  "output": {
    "checkpoint_dir": "outputs/diffusion-test/checkpoints",
    "log_dir": "outputs/diffusion-test/logs"
  }
}
```

### Step 3: Update Scripts and Workflows

**Shell scripts:**

```bash
# Before
#!/bin/bash
python -m src.main --experiment classifier --model resnet50 --epochs 10

# After
#!/bin/bash
python -m src.main configs/classifier/baseline.json
```

**CI/CD pipelines:**

```yaml
# Before
- name: Train model
  run: python -m src.main --experiment classifier --model resnet50 --epochs 10

# After
- name: Train model
  run: python -m src.main configs/classifier/baseline.json
```

**Documentation and README files:**

Update all command examples to use config files only.

### Step 4: Organize Config Files

Consider organizing configs by use case:

```
configs/
├── classifier/
│   ├── baseline.json           # Standard ResNet50 training
│   ├── inceptionv3.json        # InceptionV3 variant
│   ├── quick_test.json         # Fast testing (few epochs)
│   └── production.json         # Full training run
├── diffusion/
│   ├── default.json            # Standard DDPM training
│   ├── high_res.json           # Higher resolution
│   └── quick_test.json         # Fast testing
└── experiments/
    ├── experiment_001.json     # Custom experiments
    └── experiment_002.json
```

## Common Migration Issues

### Issue 1: Missing Required Fields

**Error:**

```
ValueError: Missing required field: model.name
```

**Solution:** Add all required fields to your config. Check the templates in `configs/` for complete examples.

### Issue 2: Trying to Use CLI Overrides

**Error:**

```
error: unrecognized arguments: --batch-size 64
```

**Solution:** Remove CLI parameters and set values in the config file instead.

### Issue 3: Expecting Default Values

**Error:**

```
ValueError: Missing required field: training.device
```

**Solution:** The new CLI doesn't provide defaults. Add all required fields explicitly.

## Benefits of Config-Only Mode

1. **Reproducibility**: All parameters explicitly stated in version-controlled files
2. **Clarity**: No ambiguity about parameter sources or priority
3. **Simplicity**: Single source of truth for configuration
4. **Validation**: Strict validation catches missing parameters early
5. **Documentation**: Config files serve as self-documenting experiment records

## Quick Reference

### New CLI Format

```bash
python -m src.main <config.json>
```

### Validation

```bash
# The CLI automatically validates configs
# To preview what would run without executing:
python -c "
from src.utils.cli import parse_args
config = parse_args(['configs/classifier/baseline.json'])
print(config)
"
```

### Creating New Configs

```bash
# Copy and modify existing config
cp configs/classifier/baseline.json my_experiment.json
# Edit my_experiment.json with your parameters
python -m src.main my_experiment.json
```

## Getting Help

- See `configs/` directory for complete config examples
- Check [architecture.md](../standards/architecture.md) for CLI documentation
- Run `python -m src.main --help` for usage information

## Rollback Instructions

If you need to temporarily use the old CLI:

```bash
# Switch to the branch before the refactoring
git checkout <previous-commit-hash>

# Or revert the changes
git revert <refactoring-commit-hash>
```

## Questions?

If you encounter issues not covered in this guide, please:

1. Check existing config files in `configs/` for examples
2. Review error messages carefully - they indicate which fields are missing
3. Refer to the validation logic in `src/experiments/*/config.py`
