# Diffusion Configuration Optimization - Implementation Tasks

This document provides a step-by-step implementation plan for optimizing the diffusion configuration structure as outlined in [diffusion-config-optimization-report.md](./diffusion-config-optimization-report.md).

**Created:** February 13, 2026  
**Status:** Ready for Implementation  
**Estimated Total Time:** 10-15 hours

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Implementation Checklist](#pre-implementation-checklist)
3. [Phase 1: Configuration Schema Update](#phase-1-configuration-schema-update)
4. [Phase 2: Code Access Pattern Update](#phase-2-code-access-pattern-update)
5. [Phase 3: Test Suite Updates](#phase-3-test-suite-updates)
6. [Phase 4: Configuration Files Update](#phase-4-configuration-files-update)
7. [Phase 5: Documentation](#phase-5-documentation)
8. [Verification Steps](#verification-steps)
9. [Rollback Plan](#rollback-plan)

---

## Overview

### Objectives

- Restructure diffusion configuration to separate common, training-specific, and generation-specific parameters
- Update all code references to use new configuration paths
- Ensure all tests pass with new structure
- Update configuration files to new format
- Document changes for users

### Key Changes Summary

| Category               | Change                                                         | Impact                        |
| ---------------------- | -------------------------------------------------------------- | ----------------------------- |
| Common Parameters      | Move `device`, `seed` to top level                             | High - affects both modes     |
| Training Parameters    | Move checkpointing to `training` section                       | Medium - training mode only   |
| Training Visualization | Move `sample_*` to `training.visualization`                    | Medium - semantic clarity     |
| Generation Parameters  | Move `checkpoint`, `num_samples`, `output_dir` to `generation` | Medium - generation mode only |
| Validation             | Move `validation` under `training`                             | Low - nesting change          |

---

## Pre-Implementation Checklist

**Before starting, ensure:**

- [ ] All tests currently pass: `pytest tests/`
- [ ] Git repository is clean with committed changes
- [ ] Create implementation branch: `git checkout -b feature/optimize-diffusion-config`
- [ ] Review the optimization report: [diffusion-config-optimization-report.md](./diffusion-config-optimization-report.md)
- [ ] Python environment is active: `source venv/bin/activate`

---

## Phase 1: Configuration Schema Update

**Goal:** Update the default configuration structure in `src/experiments/diffusion/config.py`

**Estimated Time:** 4-6 hours

### Task 1.1: Update `get_default_config()` Function

**File:** `src/experiments/diffusion/config.py` (lines 8-100)

**Changes Required:**

1. **Move common parameters to top level:**
   - Move `training.device` → top-level `device`
   - Move `training.seed` → top-level `seed`

2. **Restructure training section:**
   - Keep core training parameters (epochs, learning_rate, optimizer, scheduler, use_ema, ema_decay, use_amp, gradient_clip_norm)
   - Move `output.checkpoint_dir` → `training.checkpoint_dir`
   - Move `output.save_best_only` → `training.save_best_only`
   - Move `output.save_frequency` → `training.save_frequency`
   - Move top-level `validation` → `training.validation` (nested)
   - Create new `training.visualization` section with:
     - Move `generation.sample_images` → `training.visualization.sample_images`
     - Move `generation.sample_interval` → `training.visualization.sample_interval`
     - Move `generation.samples_per_class` → `training.visualization.samples_per_class`
     - Add `training.visualization.guidance_scale` (copy from generation.guidance_scale)

3. **Restructure generation section:**
   - Move top-level `checkpoint` → `generation.checkpoint`
   - Move `generation.num_samples` → keep in `generation`
   - Move `generation.output_dir` → keep in `generation`
   - Keep `generation.guidance_scale`
   - Add `generation.use_ema` (copy from training.use_ema default)
   - Add optional `generation.save_grid` (default: true)
   - Add optional `generation.grid_nrow` (default: 10)

4. **Update output section:**
   - Keep only `log_dir` (used in both modes)
   - Remove `checkpoint_dir`, `save_best_only`, `save_frequency`

**Implementation Details:**

```python
def get_default_config() -> Dict[str, Any]:
    """Get default configuration for diffusion model experiments.

    Configuration Structure:
    - Common parameters at top level (device, seed)
    - Mode-specific parameters in training/generation sections
    - Logical nesting for related features

    Returns:
        Dictionary containing default configuration values
    """
    return {
        "experiment": "diffusion",
        "mode": "train",  # Options: train, generate

        # Common parameters (used in both modes)
        "device": "cuda",  # Options: cuda, cpu, auto
        "seed": None,  # Random seed for reproducibility

        "model": {
            # ... keep existing model config unchanged
        },

        "data": {
            # ... keep existing data config unchanged
        },

        "output": {
            "log_dir": "outputs/logs",  # Only common output parameter
        },

        "training": {
            # Core training parameters
            "epochs": 200,
            "learning_rate": 0.0001,
            "optimizer": "adam",
            "optimizer_kwargs": {
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": None,
            "scheduler_kwargs": {
                "T_max": 200,
                "eta_min": 1e-6,
            },

            # Advanced training features
            "use_ema": True,
            "ema_decay": 0.9999,
            "use_amp": False,
            "gradient_clip_norm": None,

            # Training checkpointing
            "checkpoint_dir": "outputs/checkpoints",
            "save_best_only": False,
            "save_frequency": 10,

            # Training validation (nested)
            "validation": {
                "frequency": 1,
                "metric": "loss",
            },

            # Training visualization (nested)
            "visualization": {
                "sample_images": True,
                "sample_interval": 10,
                "samples_per_class": 2,
                "guidance_scale": 3.0,
            },
        },

        "generation": {
            # Generation input
            "checkpoint": None,  # Required for generate mode

            # Generation parameters
            "num_samples": 100,
            "guidance_scale": 3.0,
            "use_ema": True,

            # Generation output
            "output_dir": None,  # Defaults to log_dir/generated
            "save_grid": True,
            "grid_nrow": 10,
        },
    }
```

**Verification:**

- [ ] Configuration has all required keys
- [ ] No parameter duplication (except intentional like guidance_scale)
- [ ] All defaults match current behavior

---

### Task 1.2: Update `validate_config()` Function

**File:** `src/experiments/diffusion/config.py` (lines 101-370)

**Changes Required:**

1. **Update common parameter validation:**
   - Add validation for top-level `device` (must be "cuda", "cpu", or "auto")
   - Add validation for top-level `seed` (must be None or positive integer)

2. **Update training section validation:**
   - Remove validation for `training.device` (moved to top level)
   - Remove validation for `training.seed` (moved to top level)
   - Add validation for `training.checkpoint_dir`
   - Add validation for `training.save_best_only`
   - Add validation for `training.save_frequency`
   - Add validation for nested `training.validation` section
   - Add validation for nested `training.visualization` section

3. **Update generation section validation:**
   - Remove validation for `generation.sample_images` (moved to training.visualization)
   - Remove validation for `generation.sample_interval` (moved to training.visualization)
   - Remove validation for `generation.samples_per_class` (moved to training.visualization)
   - Add validation for `generation.checkpoint`
   - Add validation for `generation.use_ema`
   - Add validation for `generation.output_dir`
   - Add validation for `generation.save_grid`
   - Add validation for `generation.grid_nrow`

4. **Update output section validation:**
   - Remove validation for `output.checkpoint_dir` (moved to training)
   - Remove validation for `output.save_best_only` (moved to training)
   - Remove validation for `output.save_frequency` (moved to training)
   - Keep only `output.log_dir` validation

5. **Add mode-aware validation:**
   - When `mode == "generate"`, require `generation.checkpoint` to be set
   - When `mode == "train"`, ensure training parameters are valid
   - Provide helpful error messages for mode-specific requirements

**Implementation Details:**

```python
def validate_config(config: Dict[str, Any]) -> None:
    """Validate diffusion model configuration.

    Performs mode-aware validation with clear error messages.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
        KeyError: If required fields are missing
    """
    # Validate experiment type
    if config.get("experiment") != "diffusion":
        raise ValueError(
            f"Invalid experiment type: {config.get('experiment')}. Must be 'diffusion'"
        )

    # Validate mode
    mode = config.get("mode", "train")
    if mode not in ["train", "generate"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'generate'")

    # Validate common parameters
    device = config.get("device", "cuda")
    valid_devices = ["cuda", "cpu", "auto"]
    if device not in valid_devices:
        raise ValueError(
            f"Invalid device: {device}. Must be one of {valid_devices}"
        )

    seed = config.get("seed")
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be None or a non-negative integer")

    # Validate model section (unchanged logic)
    # ...

    # Validate data section (unchanged logic)
    # ...

    # Validate output section (simplified)
    if "output" not in config:
        raise KeyError("Missing required config key: output")

    output = config["output"]
    if "log_dir" not in output or output["log_dir"] is None:
        raise ValueError("output.log_dir is required and cannot be None")

    # Mode-specific validation
    if mode == "train":
        _validate_training_config(config)
    elif mode == "generate":
        _validate_generation_config(config)


def _validate_training_config(config: Dict[str, Any]) -> None:
    """Validate training-specific configuration."""
    if "training" not in config:
        raise KeyError("Missing required config key: training")

    training = config["training"]

    # Validate core training parameters
    # ... (existing validation logic for epochs, learning_rate, etc.)

    # Validate checkpointing
    if "checkpoint_dir" not in training or training["checkpoint_dir"] is None:
        raise ValueError("training.checkpoint_dir is required for training mode")

    # Validate nested validation section
    if "validation" in training:
        val = training["validation"]
        if "frequency" in val and (not isinstance(val["frequency"], int) or val["frequency"] < 1):
            raise ValueError("training.validation.frequency must be a positive integer")
        if "metric" in val and not isinstance(val["metric"], str):
            raise ValueError("training.validation.metric must be a string")

    # Validate nested visualization section
    if "visualization" in training:
        vis = training["visualization"]
        if "sample_images" in vis and not isinstance(vis["sample_images"], bool):
            raise ValueError("training.visualization.sample_images must be a boolean")
        if "sample_interval" in vis and (not isinstance(vis["sample_interval"], int) or vis["sample_interval"] < 1):
            raise ValueError("training.visualization.sample_interval must be a positive integer")
        if "samples_per_class" in vis and (not isinstance(vis["samples_per_class"], int) or vis["samples_per_class"] < 1):
            raise ValueError("training.visualization.samples_per_class must be a positive integer")
        if "guidance_scale" in vis and (not isinstance(vis["guidance_scale"], (int, float)) or vis["guidance_scale"] < 1.0):
            raise ValueError("training.visualization.guidance_scale must be >= 1.0")


def _validate_generation_config(config: Dict[str, Any]) -> None:
    """Validate generation-specific configuration."""
    if "generation" not in config:
        raise KeyError("Missing required config key: generation")

    generation = config["generation"]

    # Validate checkpoint (required for generation)
    if "checkpoint" not in generation or generation["checkpoint"] is None:
        raise ValueError("generation.checkpoint is required for generate mode")

    # Validate generation parameters
    if "num_samples" in generation:
        if not isinstance(generation["num_samples"], int) or generation["num_samples"] < 1:
            raise ValueError("generation.num_samples must be a positive integer")

    if "guidance_scale" in generation:
        if not isinstance(generation["guidance_scale"], (int, float)) or generation["guidance_scale"] < 1.0:
            raise ValueError("generation.guidance_scale must be >= 1.0")

    if "use_ema" in generation and not isinstance(generation["use_ema"], bool):
        raise ValueError("generation.use_ema must be a boolean")

    if "save_grid" in generation and not isinstance(generation["save_grid"], bool):
        raise ValueError("generation.save_grid must be a boolean")

    if "grid_nrow" in generation:
        if not isinstance(generation["grid_nrow"], int) or generation["grid_nrow"] < 1:
            raise ValueError("generation.grid_nrow must be a positive integer")
```

**Verification:**

- [ ] Default config passes validation
- [ ] Invalid configs raise appropriate errors
- [ ] Error messages are clear and helpful

---

### Task 1.3: Keep `get_resolution_config()` Unchanged

**File:** `src/experiments/diffusion/config.py` (lines 371-434)

**Action:** No changes needed - this function only overrides model and data parameters, which haven't moved.

**Verification:**

- [ ] Function still works with new config structure
- [ ] Resolution-specific overrides apply correctly

---

## Phase 2: Code Access Pattern Update

**Goal:** Update all code that accesses the configuration to use new paths

**Estimated Time:** 2-3 hours

### Task 2.1: Update `setup_experiment_diffusion()` in `src/main.py`

**File:** `src/main.py` (lines 260-560)

**Changes Required:**

1. **Update common parameter access (lines 272-286):**

   ```python
   # OLD:
   device_config = config.get("training", {}).get("device", "auto")
   seed = config.get("training", {}).get("seed")

   # NEW:
   device_config = config.get("device", "auto")
   seed = config.get("seed")
   ```

2. **Update output directories (lines 289-295):**

   ```python
   # OLD:
   checkpoint_dir = Path(config["output"]["checkpoint_dir"])
   log_dir = Path(config["output"]["log_dir"])

   # NEW (only in training mode, move inside training block):
   log_dir = Path(config["output"]["log_dir"])  # Common
   # checkpoint_dir moved to training block
   ```

3. **Update generation mode block (lines 327-445):**

   ```python
   # OLD:
   checkpoint_path = config.get("checkpoint")
   num_samples = config.get("num_samples", 100)
   output_dir = config.get("output_dir", log_dir / "generated")
   use_ema = config.get("training", {}).get("use_ema", True)

   # NEW:
   generation_config = config["generation"]
   checkpoint_path = generation_config["checkpoint"]
   num_samples = generation_config.get("num_samples", 100)
   output_dir = generation_config.get("output_dir")
   if output_dir is None:
       output_dir = log_dir / "generated"
   else:
       output_dir = Path(output_dir)
   use_ema = generation_config.get("use_ema", True)
   guidance_scale = generation_config.get("guidance_scale", 3.0)
   ```

4. **Update trainer initialization in generation mode (lines 372-391):**

   ```python
   # OLD:
   sample_images=generation_config.get("sample_images", True),
   sample_interval=generation_config.get("sample_interval", 10),
   samples_per_class=generation_config.get("samples_per_class", 2),
   guidance_scale=generation_config.get("guidance_scale", 3.0),

   # NEW (these are training params, use dummy/default values for generation):
   sample_images=False,  # Not used in generation mode
   sample_interval=1,
   samples_per_class=2,
   guidance_scale=generation_config.get("guidance_scale", 3.0),
   ```

5. **Update training mode block (lines 447-560):**

   ```python
   # Get training config
   training_config = config["training"]

   # Add checkpoint_dir from training section
   checkpoint_dir = Path(training_config["checkpoint_dir"])
   checkpoint_dir.mkdir(parents=True, exist_ok=True)

   # Update trainer initialization (lines 526-545):
   # OLD:
   sample_images=generation_config.get("sample_images", True),
   sample_interval=generation_config.get("sample_interval", 10),
   samples_per_class=generation_config.get("samples_per_class", 2),
   guidance_scale=generation_config.get("guidance_scale", 3.0),
   validate_frequency=config.get("validation", {}).get("frequency", 1),
   best_metric=config.get("validation", {}).get("metric", "loss"),

   # NEW:
   visualization_config = training_config.get("visualization", {})
   validation_config = training_config.get("validation", {})

   sample_images=visualization_config.get("sample_images", True),
   sample_interval=visualization_config.get("sample_interval", 10),
   samples_per_class=visualization_config.get("samples_per_class", 2),
   guidance_scale=visualization_config.get("guidance_scale", 3.0),
   validate_frequency=validation_config.get("frequency", 1),
   best_metric=validation_config.get("metric", "loss"),

   # Update save parameters:
   save_best=training_config.get("save_best_only", False),
   checkpoint_frequency=training_config.get("save_frequency", 10),
   ```

**Detailed Line-by-Line Changes:**

```python
# Lines 260-295: Initial setup
def setup_experiment_diffusion(config: Dict[str, Any]) -> None:
    """Setup and run diffusion experiment."""
    # ... imports ...

    # Validate diffusion config
    validate_diffusion_config(config)

    # Get mode
    mode = config.get("mode", "train")

    # Set up device (UPDATED - now from top level)
    device_config = config.get("device", "auto")
    if device_config == "auto":
        device = get_device()
    else:
        device = device_config

    print(f"Using device: {device}")

    # Set random seed (UPDATED - now from top level)
    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")

    # Create output directories (UPDATED - log_dir only, checkpoint_dir moved)
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # ... rest of setup ...

# Lines 327-445: Generation mode
    if mode == "generate":
        # Generation mode: load checkpoint and generate samples
        generation_config = config["generation"]  # NEW: get generation config
        checkpoint_path = generation_config.get("checkpoint")  # UPDATED
        if not checkpoint_path:
            raise ValueError("generation.checkpoint is required for generation mode")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\nLoading checkpoint: {checkpoint_path}")

        # ... load checkpoint ...

        # Initialize trainer (trainer params not really used in generation)
        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=device,
            show_progress=True,
            use_ema=generation_config.get("use_ema", True),  # UPDATED
            ema_decay=0.9999,  # Not used in generation
            use_amp=False,
            gradient_clip_norm=None,
            sample_images=False,  # UPDATED: not used in generation
            sample_interval=1,
            samples_per_class=generation_config.get("samples_per_class", 2),
            guidance_scale=generation_config.get("guidance_scale", 3.0),  # UPDATED
        )

        # Generate samples
        num_samples = generation_config.get("num_samples", 100)  # UPDATED
        print(f"\nGenerating {num_samples} samples...")

        # ... generate samples ...

        # Save generated samples
        output_dir = generation_config.get("output_dir")  # UPDATED
        if output_dir is None:
            output_dir = log_dir / "generated"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ... save samples ...

    else:
        # Training mode
        training_config = config["training"]  # Get training config

        # Create checkpoint directory (UPDATED - from training section)
        checkpoint_dir = Path(training_config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")

        # ... initialize optimizer, scheduler ...

        # Initialize trainer (UPDATED paths)
        visualization_config = training_config.get("visualization", {})
        validation_config = training_config.get("validation", {})

        trainer = DiffusionTrainer(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            logger=logger,
            device=device,
            show_progress=True,
            use_ema=training_config.get("use_ema", True),
            ema_decay=training_config.get("ema_decay", 0.9999),
            use_amp=training_config.get("use_amp", False),
            gradient_clip_norm=training_config.get("gradient_clip_norm"),
            scheduler=scheduler,
            sample_images=visualization_config.get("sample_images", True),  # UPDATED
            sample_interval=visualization_config.get("sample_interval", 10),  # UPDATED
            samples_per_class=visualization_config.get("samples_per_class", 2),  # UPDATED
            guidance_scale=visualization_config.get("guidance_scale", 3.0),  # UPDATED
        )

        # Train the model (UPDATED paths)
        num_epochs = training_config["epochs"]
        print(f"\nStarting training for {num_epochs} epochs...")

        try:
            trainer.train(
                num_epochs=num_epochs,
                checkpoint_dir=str(checkpoint_dir),
                save_best=training_config.get("save_best_only", False),  # UPDATED
                checkpoint_frequency=training_config.get("save_frequency", 10),  # UPDATED
                validate_frequency=validation_config.get("frequency", 1),  # UPDATED
                best_metric=validation_config.get("metric", "loss"),  # UPDATED
            )
        except KeyboardInterrupt:
            # ... error handling ...
```

**Verification:**

- [ ] Training mode works with new config paths
- [ ] Generation mode works with new config paths
- [ ] No references to old config paths remain

---

### Task 2.2: Check for Other Config Accesses

**Files to Check:**

- `src/experiments/diffusion/trainer.py` - No changes needed (receives parameters directly)
- `src/experiments/diffusion/dataloader.py` - No changes needed (receives parameters directly)
- `src/experiments/diffusion/logger.py` - No changes needed (receives parameters directly)
- `src/experiments/diffusion/model.py` - No changes needed (receives parameters directly)

**Action:** Verify that these files don't directly access config (they receive parsed parameters)

**Verification:**

- [ ] Trainer doesn't access config directly
- [ ] DataLoader doesn't access config directly
- [ ] Logger doesn't access config directly
- [ ] Model doesn't access config directly

---

### Task 2.3: Update Classifier Experiment (if needed)

**File:** `src/main.py` (lines 50-80 for `setup_experiment_classifier`)

**Changes Required:**

Check if classifier also has `device` and `seed` in training section. If yes, update to match diffusion pattern for consistency.

**Verification:**

- [ ] Classifier experiment still works
- [ ] Pattern is consistent between experiments

---

## Phase 3: Test Suite Updates

**Goal:** Update all tests to use new configuration structure

**Estimated Time:** 3-4 hours

### Task 3.1: Update Configuration Tests

**File:** `tests/experiments/diffusion/test_config.py`

**Changes Required:**

1. **Update `TestGetDefaultConfig` class (lines 27-117):**
   - Add test for top-level `device` parameter
   - Add test for top-level `seed` parameter
   - Update `test_training_defaults()` to check new nested sections
   - Add test for `training.checkpoint_dir`
   - Add test for `training.validation` (nested)
   - Add test for `training.visualization` (nested)
   - Update `test_generation_defaults()` to check new parameters
   - Add test for `generation.checkpoint`
   - Add test for `generation.use_ema`
   - Add test for `generation.output_dir`

2. **Update `TestValidateConfig` class (lines 119-708):**
   - Add tests for top-level `device` validation
   - Add tests for top-level `seed` validation
   - Update tests that reference `training.device` to use top-level `device`
   - Update tests that reference `training.seed` to use top-level `seed`
   - Add tests for `training.checkpoint_dir` validation
   - Add tests for nested `training.validation` validation
   - Add tests for nested `training.visualization` validation
   - Add tests for `generation.checkpoint` validation
   - Add tests for `generation.use_ema` validation
   - Add tests for mode-aware validation

**Example New Tests:**

```python
def test_device_at_top_level(self):
    """Test that device is at top level."""
    config = get_default_config()
    assert "device" in config
    assert isinstance(config["device"], str)

def test_seed_at_top_level(self):
    """Test that seed is at top level."""
    config = get_default_config()
    assert "seed" in config
    # seed can be None or int

def test_training_has_nested_validation(self):
    """Test that validation is nested under training."""
    config = get_default_config()
    assert "validation" in config["training"]
    assert "frequency" in config["training"]["validation"]
    assert "metric" in config["training"]["validation"]

def test_training_has_nested_visualization(self):
    """Test that visualization is nested under training."""
    config = get_default_config()
    assert "visualization" in config["training"]
    assert "sample_images" in config["training"]["visualization"]
    assert "sample_interval" in config["training"]["visualization"]
    assert "samples_per_class" in config["training"]["visualization"]
    assert "guidance_scale" in config["training"]["visualization"]

def test_generation_has_checkpoint(self):
    """Test that generation section has checkpoint parameter."""
    config = get_default_config()
    assert "checkpoint" in config["generation"]

def test_generation_has_use_ema(self):
    """Test that generation section has use_ema parameter."""
    config = get_default_config()
    assert "use_ema" in config["generation"]
    assert isinstance(config["generation"]["use_ema"], bool)

def test_invalid_device(self):
    """Test validation fails with invalid device."""
    config = get_default_config()
    config["device"] = "invalid"

    with pytest.raises(ValueError, match="Invalid device"):
        validate_config(config)

def test_mode_aware_validation_generate_requires_checkpoint(self):
    """Test that generate mode requires checkpoint."""
    config = get_default_config()
    config["mode"] = "generate"
    config["generation"]["checkpoint"] = None

    with pytest.raises(ValueError, match="generation.checkpoint is required"):
        validate_config(config)
```

**Verification:**

- [ ] All existing tests updated to new structure
- [ ] New tests added for new structure elements
- [ ] All tests pass: `pytest tests/experiments/diffusion/test_config.py -v`

---

### Task 3.2: Update Integration Tests

**File:** `tests/integration/test_diffusion_pipeline.py`

**Changes Required:**

Update all test configurations to use new structure (lines 40-1092).

**Example Updates:**

```python
# Lines 48-89: test_full_pipeline_unconditional
config = {
    "experiment": "diffusion",
    "mode": "train",
    "device": TEST_DEVICE,  # NEW: moved to top level
    "seed": None,  # NEW: moved to top level

    "model": { ... },
    "data": { ... },

    "output": {
        "log_dir": str(tmp_path / "logs"),  # Only log_dir remains
    },

    "training": {
        "epochs": 2,
        "learning_rate": 0.0001,
        "optimizer": "adam",
        # "device": removed (now at top level)
        # "seed": removed (now at top level)
        "use_ema": True,
        "ema_decay": 0.999,
        "use_amp": False,

        # NEW: checkpointing in training
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "save_best_only": False,
        "save_frequency": 1,

        # NEW: nested validation
        "validation": {
            "frequency": 1,
            "metric": "loss",
        },

        # NEW: nested visualization
        "visualization": {
            "sample_images": True,
            "sample_interval": 1,
            "samples_per_class": 2,
            "guidance_scale": 0.0,
        },
    },

    # generation section removed (training mode)
}
```

**Files to Update:**

- All test methods in `TestDiffusionPipelineBasic`
- All test methods in `TestDiffusionPipelineCheckpoints`
- All test methods in `TestDiffusionPipelineGeneration`
- All test methods in `TestDiffusionPipelineAdvanced`

**Specific Test Updates:**

1. **Training tests:** Move device/seed to top level, add training.visualization, training.validation, training checkpointing
2. **Generation tests:** Add generation.checkpoint, generation.use_ema, generation.output_dir
3. **Checkpoint tests:** Update paths to training.checkpoint_dir

**Verification:**

- [ ] All integration tests pass: `pytest tests/integration/test_diffusion_pipeline.py -v`

---

### Task 3.3: Update Other Test Files

**Files:**

- `tests/conftest.py` - Update `mock_config_diffusion` fixture (lines 240-255)
- `tests/test_main.py` - Update diffusion test configurations
- `tests/utils/test_cli.py` - Update diffusion CLI test configurations
- `tests/fixtures/configs/diffusion/valid_minimal.yaml` - Update test fixture
- `tests/fixtures/configs/diffusion/invalid_*.yaml` - Update test fixtures

**Changes:** Apply same restructuring as above to all test configurations.

**Verification:**

- [ ] All tests pass: `pytest tests/ -v`
- [ ] No deprecation warnings

---

## Phase 4: Configuration Files Update

**Goal:** Update all YAML configuration files to new structure

**Estimated Time:** 2-3 hours

### Task 4.1: Update Default Configurations

**Files:**

1. `configs/diffusion/default.yaml`
2. `configs/diffusion/default_new.yaml` (already optimized, verify it matches)

**Changes:**

Apply the restructuring to match the new schema. Use `configs/diffusion/default_new.yaml` as reference since it already has the optimized structure.

**For `default.yaml`:**

```yaml
# ==============================================================================
# COMMON PARAMETERS
# ==============================================================================
experiment: diffusion
mode: train

# Common parameters (used in both modes)
device: cuda # MOVED from training.device
seed: null # MOVED from training.seed

# ==============================================================================
# MODEL AND DATA (unchanged)
# ==============================================================================
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

data:
  train_path: data/train
  val_path: null
  batch_size: 32
  num_workers: 4
  image_size: 40
  horizontal_flip: true
  rotation_degrees: 0
  color_jitter: false
  color_jitter_strength: 0.1
  pin_memory: true
  drop_last: false
  shuffle_train: true
  return_labels: false

# ==============================================================================
# OUTPUT (simplified)
# ==============================================================================
output:
  log_dir: outputs/logs # Only common output parameter

# ==============================================================================
# TRAINING MODE CONFIGURATION
# ==============================================================================
training:
  # Core parameters
  epochs: 200
  learning_rate: 0.0001
  optimizer: adam
  optimizer_kwargs:
    weight_decay: 0.0
    betas: [0.9, 0.999]
  scheduler: null
  scheduler_kwargs:
    T_max: 200
    eta_min: 1.0e-6

  # Advanced features
  use_ema: true
  ema_decay: 0.9999
  use_amp: false
  gradient_clip_norm: null

  # Checkpointing (MOVED from output section)
  checkpoint_dir: outputs/checkpoints
  save_best_only: false
  save_frequency: 10

  # Validation (MOVED from top level, now nested)
  validation:
    frequency: 1
    metric: loss

  # Visualization (MOVED from generation, now nested)
  visualization:
    sample_images: true
    sample_interval: 10
    samples_per_class: 2
    guidance_scale: 3.0

# ==============================================================================
# GENERATION MODE CONFIGURATION
# ==============================================================================
generation:
  # Generation input
  checkpoint: null # MOVED from top level

  # Generation parameters
  num_samples: 100 # KEPT here
  guidance_scale: 3.0 # KEPT here
  use_ema: true # NEW

  # Generation output
  output_dir: null # KEPT here
  save_grid: true # NEW
  grid_nrow: 10 # NEW
```

**Verification:**

- [ ] YAML files are valid: `python -c "import yaml; yaml.safe_load(open('configs/diffusion/default.yaml'))"`
- [ ] Config passes validation: Test with updated validation function
- [ ] Both files produce working training runs

---

### Task 4.2: Create Example Configurations

**New Files to Create:**

1. **`configs/diffusion/train_40x40.yaml`** - Example training config for 40x40 images
2. **`configs/diffusion/train_64x64.yaml`** - Example training config for 64x64 images
3. **`configs/diffusion/generate_example.yaml`** - Example generation config

**Content for `configs/diffusion/generate_example.yaml`:**

```yaml
experiment: diffusion
mode: generate

# Common parameters
device: cuda
seed: 42 # For reproducible generation

# Model configuration (must match training)
model:
  image_size: 40
  in_channels: 3
  model_channels: 64
  channel_multipliers: [1, 2, 4]
  num_classes: null # or 2 for conditional
  num_timesteps: 1000
  beta_schedule: cosine
  beta_start: 0.0001
  beta_end: 0.02
  class_dropout_prob: 0.1
  use_attention: [false, false, true]

# Data configuration (needed for class info)
data:
  train_path: data/train # Only for class information
  batch_size: 32
  num_workers: 4
  image_size: 40
  return_labels: false

# Output
output:
  log_dir: outputs/logs

# Generation configuration
generation:
  checkpoint: outputs/checkpoints/epoch_200.pth # Path to trained model
  num_samples: 100
  guidance_scale: 3.0 # Higher = more guided (for conditional)
  use_ema: true # Use EMA weights if available
  output_dir: outputs/generated
  save_grid: true
  grid_nrow: 10
```

**Verification:**

- [ ] Example configs are valid YAML
- [ ] Example configs pass validation
- [ ] Example configs work for their intended purpose

---

### Task 4.3: Delete Old Config Files (if any)

**Action:** Remove any old/deprecated config files that don't follow the new structure.

**Verification:**

- [ ] Only valid configs remain in `configs/diffusion/`

---

## Phase 5: Documentation

**Goal:** Document all changes for users and developers

**Estimated Time:** 2-3 hours

### Task 5.1: Update README

**File:** `README.md`

**Changes:**

Add or update section on diffusion configuration with examples of new structure.

````markdown
### Diffusion Model Configuration

The diffusion configuration is organized into common, training-specific, and generation-specific sections:

#### Common Parameters (used in both modes)

- `device`: Device to use (cuda, cpu, auto)
- `seed`: Random seed for reproducibility

#### Training Mode

```yaml
mode: train
training:
  epochs: 200
  learning_rate: 0.0001
  # ... training parameters ...

  # Nested validation configuration
  validation:
    frequency: 1
    metric: loss

  # Nested visualization configuration
  visualization:
    sample_images: true
    sample_interval: 10
```
````

#### Generation Mode

```yaml
mode: generate
generation:
  checkpoint: path/to/model.pth
  num_samples: 100
  guidance_scale: 3.0
  use_ema: true
```

See `configs/diffusion/` for complete examples.

````

**Verification:**
- [ ] README has correct examples
- [ ] README explains new structure
- [ ] Links to example configs work

---

### Task 5.2: Create Migration Guide

**File:** `docs/research/diffusion-config-migration-guide.md`

**Content:**

```markdown
# Diffusion Configuration Migration Guide

This guide helps users migrate from the old configuration structure to the optimized structure introduced in February 2026.

## Quick Migration Table

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `training.device` | `device` (top level) | Now common parameter |
| `training.seed` | `seed` (top level) | Now common parameter |
| `generation.sample_images` | `training.visualization.sample_images` | Training visualization |
| `generation.sample_interval` | `training.visualization.sample_interval` | Training visualization |
| `generation.samples_per_class` | `training.visualization.samples_per_class` | Training visualization |
| `output.checkpoint_dir` | `training.checkpoint_dir` | Training-specific |
| `output.save_best_only` | `training.save_best_only` | Training-specific |
| `output.save_frequency` | `training.save_frequency` | Training-specific |
| `validation.*` (top level) | `training.validation.*` | Now nested |
| `checkpoint` (top level) | `generation.checkpoint` | Generation-specific |
| `num_samples` (top level) | `generation.num_samples` | Generation-specific |
| `output_dir` (top level) | `generation.output_dir` | Generation-specific |

## Before and After Examples

### Training Configuration

**Before:**
```yaml
training:
  device: cuda
  seed: 42
  epochs: 200
  # ...

generation:
  sample_images: true
  sample_interval: 10

output:
  checkpoint_dir: outputs/checkpoints
  save_frequency: 10

validation:
  frequency: 1
````

**After:**

```yaml
device: cuda # Moved to top level
seed: 42 # Moved to top level

training:
  epochs: 200
  # ...

  checkpoint_dir: outputs/checkpoints # Moved from output
  save_frequency: 10 # Moved from output

  validation: # Nested under training
    frequency: 1

  visualization: # Nested under training
    sample_images: true
    sample_interval: 10
```

### Generation Configuration

**Before:**

```yaml
mode: generate
checkpoint: path/model.pth
num_samples: 100

training:
  device: cuda
  use_ema: true
```

**After:**

```yaml
mode: generate
device: cuda # Moved to top level

generation:
  checkpoint: path/model.pth # Moved from top level
  num_samples: 100
  use_ema: true # Explicit in generation
```

## Automated Migration Script

```python
# scripts/migrate_diffusion_config.py
# TODO: Create script to automate migration
```

````

**Verification:**
- [ ] Migration guide is clear and complete
- [ ] Examples are correct
- [ ] All moved parameters are documented

---

### Task 5.3: Update Code Documentation

**Files:**
- `src/experiments/diffusion/config.py` - Update docstrings
- `src/main.py` - Update comments in setup_experiment_diffusion

**Changes:**
- Update docstrings to reflect new structure
- Add examples with new structure
- Update parameter descriptions

**Verification:**
- [ ] Docstrings are accurate
- [ ] Examples in docstrings work
- [ ] API documentation is correct

---

### Task 5.4: Update CHANGELOG

**File:** `CHANGELOG.md` (or create if doesn't exist)

**Entry:**

```markdown
## [Unreleased]

### Changed
- **BREAKING**: Restructured diffusion configuration for better clarity and organization
  - Moved `device` and `seed` to top level (common parameters)
  - Moved checkpointing parameters to `training` section
  - Nested `validation` under `training`
  - Created new `training.visualization` section for training-time sampling
  - Moved generation-specific parameters to `generation` section
- Updated all configuration files to new structure
- Updated tests to use new configuration structure

### Added
- Mode-aware configuration validation
- New example configuration files for training and generation
- Migration guide for updating existing configurations

### Deprecated
- Old configuration structure (will be removed in future version)

See `docs/research/diffusion-config-migration-guide.md` for migration instructions.
````

**Verification:**

- [ ] CHANGELOG entry is complete
- [ ] Breaking changes are clearly marked
- [ ] Migration path is documented

---

## Verification Steps

**After completing all phases, verify:**

### Phase 1 Verification

- [ ] Run: `python -c "from src.experiments.diffusion.config import get_default_config, validate_config; validate_config(get_default_config())"`
- [ ] No errors from config validation

### Phase 2 Verification

- [ ] Run training with new config: `python -m src.main --config configs/diffusion/default.yaml`
- [ ] Training completes successfully
- [ ] Checkpoints are saved correctly
- [ ] Samples are generated during training

### Phase 3 Verification

- [ ] Run all tests: `pytest tests/ -v`
- [ ] All tests pass
- [ ] No warnings about deprecated config access

### Phase 4 Verification

- [ ] All YAML files are valid
- [ ] All example configs work
- [ ] Generation mode works with new configs

### Phase 5 Verification

- [ ] Documentation is accurate
- [ ] Examples in docs work
- [ ] Migration guide is helpful

### Integration Verification

- [ ] End-to-end training workflow: Train model with new config
- [ ] End-to-end generation workflow: Generate samples with new config
- [ ] Config migration: Old configs can be manually migrated using guide
- [ ] No regression: New structure produces same results as old structure

---

##Rollback Plan

**If issues are discovered after implementation:**

1. **Branch Protection:**
   - Implementation is on feature branch: `feature/optimize-diffusion-config`
   - Main branch remains unchanged until tests pass

2. **Rollback Steps:**

   ```bash
   # If not yet merged to main
   git checkout main
   git branch -D feature/optimize-diffusion-config

   # If already merged to main
   git revert <commit-hash>
   ```

3. **Backward Compatibility:**
   - Consider adding temporary backward compatibility layer if needed
   - Add warnings for deprecated config paths
   - Provide automatic migration in code

4. **Emergency Fixes:**
   - Fix validation issues
   - Fix config access issues
   - Update failing tests

---

## Success Criteria

**Implementation is successful when:**

- [ ] All phases completed
- [ ] All tests passing
- [ ] No breaking changes for existing users with migration path
- [ ] Training and generation work with new configs
- [ ] Documentation is complete and accurate
- [ ] Code review approved
- [ ] Changes merged to main branch

---

## Notes and Tips

### Development Tips

1. **Work Incrementally:**
   - Complete one phase before moving to next
   - Commit after each major task
   - Run tests frequently

2. **Testing Strategy:**
   - Write tests first (TDD) for validation changes
   - Test both valid and invalid configs
   - Test both training and generation modes

3. **Documentation:**
   - Update docs as you code
   - Keep examples synchronized with code
   - Test all documentation examples

### Common Pitfalls to Avoid

1. **Incomplete Updates:**
   - Forgetting to update a test file
   - Missing a config access in code
   - Not updating all example configs

2. **Validation Issues:**
   - Validation too strict (rejects valid configs)
   - Validation too loose (accepts invalid configs)
   - Missing mode-aware validation

3. **Breaking Changes:**
   - No migration path for users
   - Unclear error messages
   - Sudden removal of old structure

### Questions to Ask During Implementation

- Does this change maintain backward compatibility?
- Are error messages clear and helpful?
- Is the new structure more intuitive?
- Have I updated all affected tests?
- Does the documentation reflect reality?

---

## Appendix: File Checklist

### Source Code Files

- [ ] `src/experiments/diffusion/config.py`
- [ ] `src/main.py`
- [ ] `src/experiments/diffusion/trainer.py` (verify only)
- [ ] `src/experiments/diffusion/dataloader.py` (verify only)
- [ ] `src/experiments/diffusion/logger.py` (verify only)
- [ ] `src/experiments/diffusion/model.py` (verify only)

### Test Files

- [ ] `tests/experiments/diffusion/test_config.py`
- [ ] `tests/integration/test_diffusion_pipeline.py`
- [ ] `tests/conftest.py`
- [ ] `tests/test_main.py`
- [ ] `tests/utils/test_cli.py`
- [ ] `tests/fixtures/configs/diffusion/valid_minimal.yaml`
- [ ] `tests/fixtures/configs/diffusion/invalid_*.yaml`

### Configuration Files

- [ ] `configs/diffusion/default.yaml`
- [ ] `configs/diffusion/default_new.yaml`
- [ ] `configs/diffusion/train_40x40.yaml` (new)
- [ ] `configs/diffusion/train_64x64.yaml` (new)
- [ ] `configs/diffusion/generate_example.yaml` (new)

### Documentation Files

- [ ] `README.md`
- [ ] `docs/research/diffusion-config-migration-guide.md` (new)
- [ ] `CHANGELOG.md`
- [ ] Docstrings in `src/experiments/diffusion/config.py`
- [ ] Comments in `src/main.py`

---

## Timeline Estimate

| Phase     | Tasks                | Time            | Dependencies |
| --------- | -------------------- | --------------- | ------------ |
| Phase 1   | Configuration Schema | 4-6 hours       | None         |
| Phase 2   | Code Access Patterns | 2-3 hours       | Phase 1      |
| Phase 3   | Test Suite Updates   | 3-4 hours       | Phase 1, 2   |
| Phase 4   | Configuration Files  | 2-3 hours       | Phase 1      |
| Phase 5   | Documentation        | 2-3 hours       | All previous |
| **Total** | **All phases**       | **13-19 hours** | Sequential   |

**Recommended Schedule:**

- Day 1: Phase 1 (6 hours)
- Day 2: Phase 2 + Phase 3 (7 hours)
- Day 3: Phase 4 + Phase 5 + Verification (6 hours)

---

**END OF IMPLEMENTATION TASKS**
