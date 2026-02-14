# Diffusion Configuration V2 - Implementation Tasks

**Date:** February 13, 2026  
**Status:** Ready for Implementation  
**Related:** [diffusion-config-v2-optimization-plan.md](diffusion-config-v2-optimization-plan.md)

---

## Executive Summary

This document provides a detailed, step-by-step task list for implementing the Diffusion Configuration V2 optimization plan. The tasks are organized by implementation phase and include specific file locations, function names, and expected changes.

**Total Estimated Time:** 21 hours  
**Implementation Order:** Sequential by phase (dependencies exist between phases)

---

## Phase 1: Configuration File Updates (2 hours)

### Task 1.1: Create New V2 Configuration File

**File:** `configs/diffusion/default.yaml`  
**Action:** Create new configuration with V2 structure  
**Changes:**

- Move `device` and `seed` to `compute` section
- Restructure `model` into `architecture`, `diffusion`, `conditioning` subsections
- Restructure `data` into `paths`, `loading`, `augmentation` subsections
- Replace flat `optimizer` parameters with nested `training.optimizer` structure
- Replace flat `scheduler` parameters with nested `training.scheduler` structure
- Replace `use_ema` and `ema_decay` with `training.ema` structure
- Move `checkpoint_dir` to `output.subdirs.checkpoints`
- Create `output.base_dir` and `output.subdirs` structure
- Move `visualization` parameters from `generation` to `training.visualization`
- Add `training.checkpointing` subsection
- Add `training.resume` subsection
- Add `training.performance` subsection
- Add `training.validation` (already exists, ensure correct structure)
- Restructure `generation` into `sampling` and `output` subsections

**Validation:**

- Load config with `yaml.safe_load()` - should not raise errors
- Verify all required keys exist in new structure

---

### Task 1.2: Backup Old Configuration

**File:** `configs/diffusion/default.yaml` → `configs/diffusion/legacy.yaml`  
**Action:** Rename current default.yaml to legacy.yaml  
**Changes:**

- Copy current `default.yaml` to `legacy.yaml` before making changes
- Add comment at top indicating this is the legacy V1 format

**Validation:**

- Verify `legacy.yaml` still loads and validates correctly
- Keep as reference during migration

---

### Task 1.3: Update Test Fixture Configs

**Files:**

- `tests/fixtures/configs/diffusion_minimal.yaml`
- `tests/fixtures/configs/diffusion/valid_minimal.yaml`
- `tests/fixtures/configs/diffusion/invalid_missing_data.yaml`

**Action:** Update all test configs to V2 structure  
**Changes:**

- Apply same structural changes as Task 1.1
- Ensure test cases still cover intended validation scenarios

**Validation:**

- Run existing tests to check fixture compatibility
- Tests should fail initially (expected), to be fixed in Phase 6

---

## Phase 2: Configuration Loading & Validation (4 hours)

### Task 2.1: Update Default Configuration Generator

**File:** `src/experiments/diffusion/config.py`  
**Function:** `get_default_config()`  
**Action:** Update to return V2 structure  
**Changes:**

```python
# OLD:
{
    "device": "cuda",
    "seed": None,
    "model": {
        "image_size": 40,
        "in_channels": 3,
        ...
    },
    ...
}

# NEW:
{
    "compute": {
        "device": "cuda",
        "seed": None
    },
    "model": {
        "architecture": {
            "image_size": 40,
            "in_channels": 3,
            "model_channels": 64,
            "channel_multipliers": [1, 2, 4],
            "use_attention": [False, False, True]
        },
        "diffusion": {
            "num_timesteps": 1000,
            "beta_schedule": "cosine",
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "conditioning": {
            "type": None,  # or "class"
            "num_classes": None,
            "class_dropout_prob": 0.1
        }
    },
    "data": {
        "paths": {
            "train": "data/train",
            "val": None
        },
        "loading": {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "shuffle_train": True,
            "drop_last": False
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation_degrees": 0,
            "color_jitter": {
                "enabled": False,
                "strength": 0.1
            }
        }
    },
    "output": {
        "base_dir": "outputs",
        "subdirs": {
            "logs": "logs",
            "checkpoints": "checkpoints",
            "samples": "samples",
            "generated": "generated"
        }
    },
    "training": {
        "epochs": 200,
        "optimizer": {
            "type": "adam",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "gradient_clip_norm": None
        },
        "scheduler": {
            "type": None,
            "T_max": "auto",
            "eta_min": 1.0e-6
        },
        "ema": {
            "enabled": True,
            "decay": 0.9999
        },
        "checkpointing": {
            "save_frequency": 10,
            "save_best_only": False,
            "save_optimizer": True
        },
        "validation": {
            "enabled": True,
            "frequency": 1,
            "metric": "loss"
        },
        "visualization": {
            "enabled": True,
            "interval": 10,
            "num_samples": 8,
            "guidance_scale": 3.0
        },
        "performance": {
            "use_amp": False,
            "use_tf32": True,
            "cudnn_benchmark": True,
            "compile_model": False
        },
        "resume": {
            "enabled": False,
            "checkpoint": None,
            "reset_optimizer": False,
            "reset_scheduler": False
        }
    },
    "generation": {
        "checkpoint": None,
        "sampling": {
            "num_samples": 100,
            "guidance_scale": 3.0,
            "use_ema": True
        },
        "output": {
            "save_individual": True,
            "save_grid": True,
            "grid_nrow": 10
        }
    }
}
```

**Validation:**

- Call `get_default_config()` and verify structure matches V2
- All tests using `get_default_config()` should be updated

---

### Task 2.2: Update Configuration Validation

**File:** `src/experiments/diffusion/config.py`  
**Function:** `validate_config()`  
**Action:** Update validation logic for V2 structure  
**Changes:**

1. Update device validation: `config.get("compute", {}).get("device")`
2. Update seed validation: `config.get("compute", {}).get("seed")`
3. Add validation for `model.architecture.*` keys
4. Add validation for `model.diffusion.*` keys
5. Add validation for `model.conditioning.type` (must be None or "class")
6. Add validation for `data.paths.*` keys
7. Add validation for `data.loading.*` keys
8. Add validation for `data.augmentation.*` keys
9. Add validation for `output.base_dir` and `output.subdirs.*`
10. Add validation for `training.optimizer.*` keys
11. Add validation for `training.scheduler.*` keys
12. Add validation for `training.ema.*` keys
13. Add validation for `training.checkpointing.*` keys
14. Add validation for `training.validation.*` keys
15. Add validation for `training.visualization.*` keys
16. Add validation for `training.performance.*` keys
17. Add validation for `training.resume.*` keys
18. Add validation for `generation.sampling.*` keys
19. Add validation for `generation.output.*` keys
20. Add cross-validation: if `model.conditioning.type == "class"`, ensure `num_classes` is set
21. Add warning if `training.optimizer.gradient_clip_norm` is very small or large

**Validation:**

- Test with valid V2 config - should pass
- Test with missing required keys - should raise appropriate errors
- Test with invalid values - should raise appropriate errors

---

### Task 2.3: Add Configuration Helper Functions

**File:** `src/utils/config.py`  
**Action:** Add helper functions for V2 config  
**New Functions:**

```python
def resolve_output_path(config: Dict[str, Any], subdir_key: str) -> Path:
    """Resolve output path from base_dir + subdirs.

    Args:
        config: Full configuration dictionary
        subdir_key: Key in output.subdirs (e.g., "logs", "checkpoints")

    Returns:
        Resolved Path object

    Example:
        >>> config = {"output": {"base_dir": "outputs", "subdirs": {"logs": "logs"}}}
        >>> resolve_output_path(config, "logs")
        Path("outputs/logs")
    """
    base_dir = Path(config["output"]["base_dir"])
    subdir = config["output"]["subdirs"][subdir_key]
    return base_dir / subdir

def derive_image_size_from_model(config: Dict[str, Any]) -> int:
    """Derive image_size from model configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Image size from model.architecture.image_size
    """
    return config["model"]["architecture"]["image_size"]

def derive_return_labels_from_model(config: Dict[str, Any]) -> bool:
    """Derive return_labels from model conditioning configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        True if model uses class conditioning, False otherwise
    """
    conditioning_type = config["model"]["conditioning"]["type"]
    return conditioning_type == "class"

def validate_config_consistency(config: Dict[str, Any]) -> None:
    """Validate cross-parameter consistency in configuration.

    Checks:
    - If conditioning.type == "class", num_classes must be set
    - scheduler.T_max == "auto" is replaced with epochs

    Args:
        config: Full configuration dictionary

    Raises:
        ValueError: If configuration has consistency issues
    """
    # Check conditioning consistency
    cond_type = config["model"]["conditioning"]["type"]
    if cond_type == "class":
        num_classes = config["model"]["conditioning"]["num_classes"]
        if num_classes is None or num_classes <= 0:
            raise ValueError(
                "model.conditioning.num_classes must be set when "
                "conditioning.type='class'"
            )

    # Check scheduler T_max
    if "training" in config:
        scheduler_config = config["training"].get("scheduler", {})
        if scheduler_config.get("T_max") == "auto":
            scheduler_config["T_max"] = config["training"]["epochs"]
```

**Validation:**

- Test each helper function independently
- Test with various config structures

---

### Task 2.4: Add Backward Compatibility Layer (Optional)

**File:** `src/utils/config.py`  
**Action:** Add function to convert V1 config to V2  
**New Function:**

```python
def migrate_config_v1_to_v2(v1_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate V1 configuration to V2 structure.

    Args:
        v1_config: Configuration in V1 format

    Returns:
        Configuration in V2 format

    Raises:
        ValueError: If v1_config is already in V2 format
    """
    import warnings

    # Check if already V2 (has compute key)
    if "compute" in v1_config:
        warnings.warn("Configuration appears to be V2 format already")
        return v1_config.copy()

    warnings.warn(
        "Using V1 configuration format. Please migrate to V2 format. "
        "See docs/research/diffusion-config-migration-guide.md"
    )

    v2_config = {}

    # Migrate compute
    v2_config["compute"] = {
        "device": v1_config.get("device", "cuda"),
        "seed": v1_config.get("seed")
    }

    # Migrate model
    v1_model = v1_config["model"]
    v2_config["model"] = {
        "architecture": {
            "image_size": v1_model["image_size"],
            "in_channels": v1_model["in_channels"],
            "model_channels": v1_model["model_channels"],
            "channel_multipliers": v1_model["channel_multipliers"],
            "use_attention": v1_model["use_attention"]
        },
        "diffusion": {
            "num_timesteps": v1_model["num_timesteps"],
            "beta_schedule": v1_model["beta_schedule"],
            "beta_start": v1_model["beta_start"],
            "beta_end": v1_model["beta_end"]
        },
        "conditioning": {
            "type": "class" if v1_model.get("num_classes") else None,
            "num_classes": v1_model.get("num_classes"),
            "class_dropout_prob": v1_model.get("class_dropout_prob", 0.1)
        }
    }

    # Migrate data
    v1_data = v1_config["data"]
    v2_config["data"] = {
        "paths": {
            "train": v1_data["train_path"],
            "val": v1_data.get("val_path")
        },
        "loading": {
            "batch_size": v1_data["batch_size"],
            "num_workers": v1_data["num_workers"],
            "pin_memory": v1_data.get("pin_memory", True),
            "shuffle_train": v1_data.get("shuffle_train", True),
            "drop_last": v1_data.get("drop_last", False)
        },
        "augmentation": {
            "horizontal_flip": v1_data.get("horizontal_flip", True),
            "rotation_degrees": v1_data.get("rotation_degrees", 0),
            "color_jitter": {
                "enabled": v1_data.get("color_jitter", False),
                "strength": v1_data.get("color_jitter_strength", 0.1)
            }
        }
    }

    # Migrate output
    v2_config["output"] = {
        "base_dir": "outputs",
        "subdirs": {
            "logs": "logs",
            "checkpoints": "checkpoints",
            "samples": "samples",
            "generated": "generated"
        }
    }

    # Migrate training
    if "training" in v1_config:
        v1_training = v1_config["training"]
        v2_config["training"] = {
            "epochs": v1_training["epochs"],
            "optimizer": {
                "type": v1_training["optimizer"],
                "learning_rate": v1_training["learning_rate"],
                **v1_training.get("optimizer_kwargs", {})
            },
            "scheduler": {
                "type": v1_training.get("scheduler"),
                **v1_training.get("scheduler_kwargs", {})
            },
            "ema": {
                "enabled": v1_training.get("use_ema", True),
                "decay": v1_training.get("ema_decay", 0.9999)
            },
            "checkpointing": {
                "save_frequency": v1_training.get("save_frequency", 10),
                "save_best_only": v1_training.get("save_best_only", False),
                "save_optimizer": True
            },
            "validation": v1_training.get("validation", {
                "enabled": True,
                "frequency": 1,
                "metric": "loss"
            }),
            "visualization": v1_training.get("visualization", {
                "enabled": True,
                "interval": 10,
                "num_samples": 8,
                "guidance_scale": 3.0
            }),
            "performance": {
                "use_amp": v1_training.get("use_amp", False),
                "use_tf32": True,
                "cudnn_benchmark": True,
                "compile_model": False
            },
            "resume": {
                "enabled": False,
                "checkpoint": None,
                "reset_optimizer": False,
                "reset_scheduler": False
            }
        }

    # Migrate generation
    if "generation" in v1_config:
        v1_gen = v1_config["generation"]
        v2_config["generation"] = {
            "checkpoint": v1_gen.get("checkpoint"),
            "sampling": {
                "num_samples": v1_gen.get("num_samples", 100),
                "guidance_scale": v1_gen.get("guidance_scale", 3.0),
                "use_ema": v1_gen.get("use_ema", True)
            },
            "output": {
                "save_individual": True,
                "save_grid": v1_gen.get("save_grid", True),
                "grid_nrow": v1_gen.get("grid_nrow", 10)
            }
        }

    return v2_config
```

**Validation:**

- Test with actual V1 configs (legacy.yaml)
- Verify output matches expected V2 structure

---

## Phase 3: Data Loading Updates (2 hours)

### Task 3.1: Update DiffusionDataLoader Constructor

**File:** `src/experiments/diffusion/dataloader.py`  
**Class:** `DiffusionDataLoader`  
**Function:** `__init__()`  
**Action:** Update to accept V2 data config structure  
**Changes:**

**Current signature:**

```python
def __init__(
    self,
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 64,
    horizontal_flip: bool = True,
    rotation_degrees: int = 0,
    color_jitter: bool = False,
    color_jitter_strength: float = 0.1,
    pin_memory: bool = True,
    drop_last: bool = False,
    shuffle_train: bool = True,
    return_labels: bool = True,
):
```

**No changes needed to signature** - keep backward compatible  
**Update the calling code in main.py** to pass derived values

**Validation:**

- Existing tests should still pass
- New tests should verify derived values work correctly

---

### Task 3.2: Update DiffusionDataLoader Instantiation in main.py

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update dataloader instantiation for V2 config  
**Changes:**

**Current (lines ~350-362, ~447-459):**

```python
data_config = config["data"]
dataloader = DiffusionDataLoader(
    train_path=data_config["train_path"],
    val_path=data_config.get("val_path"),
    batch_size=data_config["batch_size"],
    num_workers=data_config["num_workers"],
    image_size=data_config["image_size"],
    horizontal_flip=data_config.get("horizontal_flip", True),
    rotation_degrees=data_config.get("rotation_degrees", 0),
    color_jitter=data_config.get("color_jitter", False),
    color_jitter_strength=data_config.get("color_jitter_strength", 0.1),
    pin_memory=data_config.get("pin_memory", True),
    drop_last=data_config.get("drop_last", False),
    shuffle_train=data_config.get("shuffle_train", True),
    return_labels=data_config.get("return_labels", False),
)
```

**New:**

```python
from src.utils.config import derive_image_size_from_model, derive_return_labels_from_model

data_config = config["data"]
dataloader = DiffusionDataLoader(
    train_path=data_config["paths"]["train"],
    val_path=data_config["paths"].get("val"),
    batch_size=data_config["loading"]["batch_size"],
    num_workers=data_config["loading"]["num_workers"],
    image_size=derive_image_size_from_model(config),  # Derived from model
    horizontal_flip=data_config["augmentation"]["horizontal_flip"],
    rotation_degrees=data_config["augmentation"]["rotation_degrees"],
    color_jitter=data_config["augmentation"]["color_jitter"]["enabled"],
    color_jitter_strength=data_config["augmentation"]["color_jitter"]["strength"],
    pin_memory=data_config["loading"]["pin_memory"],
    drop_last=data_config["loading"]["drop_last"],
    shuffle_train=data_config["loading"]["shuffle_train"],
    return_labels=derive_return_labels_from_model(config),  # Derived from model
)
```

**Locations to update:**

1. Generation mode dataloader instantiation (~lines 350-362)
2. Training mode dataloader instantiation (~lines 447-459)

**Validation:**

- Run diffusion experiment in both modes
- Verify dataloader receives correct parameters

---

## Phase 4: Model Initialization Updates (2 hours)

### Task 4.1: Update Model Creation in main.py

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update model initialization for V2 config  
**Changes:**

**Current (lines ~309-322):**

```python
model_config = config["model"]
model = create_ddpm(
    image_size=model_config["image_size"],
    in_channels=model_config["in_channels"],
    model_channels=model_config["model_channels"],
    channel_multipliers=tuple(model_config["channel_multipliers"]),
    num_classes=model_config["num_classes"],
    num_timesteps=model_config["num_timesteps"],
    beta_schedule=model_config["beta_schedule"],
    beta_start=model_config["beta_start"],
    beta_end=model_config["beta_end"],
    class_dropout_prob=model_config["class_dropout_prob"],
    use_attention=tuple(model_config["use_attention"]),
    device=device,
)
```

**New:**

```python
model_config = config["model"]
arch_config = model_config["architecture"]
diff_config = model_config["diffusion"]
cond_config = model_config["conditioning"]

model = create_ddpm(
    image_size=arch_config["image_size"],
    in_channels=arch_config["in_channels"],
    model_channels=arch_config["model_channels"],
    channel_multipliers=tuple(arch_config["channel_multipliers"]),
    num_classes=cond_config["num_classes"],
    num_timesteps=diff_config["num_timesteps"],
    beta_schedule=diff_config["beta_schedule"],
    beta_start=diff_config["beta_start"],
    beta_end=diff_config["beta_end"],
    class_dropout_prob=cond_config["class_dropout_prob"],
    use_attention=tuple(arch_config["use_attention"]),
    device=device,
)
```

**Also update print statements (lines ~324-327):**

```python
print(f"Model: DDPM")
print(f"Image size: {arch_config['image_size']}")
print(f"Num classes: {cond_config['num_classes']}")
print(f"Num timesteps: {diff_config['num_timesteps']}")
```

**Validation:**

- Model should initialize correctly with V2 config
- Parameters should match expected values

---

### Task 4.2: Update Device and Seed Handling

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update device/seed access for V2 config  
**Changes:**

**Current (lines ~273-287):**

```python
# Set up device (now at top level)
device_config = config.get("device", "auto")
if device_config == "auto":
    device = get_device()
else:
    device = device_config

print(f"Using device: {device}")

# Set random seed if specified (now at top level)
seed = config.get("seed")
if seed is not None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
```

**New:**

```python
# Set up device (now in compute section)
compute_config = config.get("compute", {})
device_config = compute_config.get("device", "auto")
if device_config == "auto":
    device = get_device()
else:
    device = device_config

print(f"Using device: {device}")

# Set random seed if specified (now in compute section)
seed = compute_config.get("seed")
if seed is not None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
```

**Validation:**

- Device selection should work correctly
- Seed should be set when specified

---

## Phase 5: Training Logic Updates (4 hours)

### Task 5.1: Update Output Directory Handling

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update output path construction for V2 config  
**Changes:**

**Import at top of file:**

```python
from src.utils.config import resolve_output_path
```

**Current (lines ~295-298):**

```python
# Create output directories
log_dir = Path(config["output"]["log_dir"])
log_dir.mkdir(parents=True, exist_ok=True)

print(f"Log directory: {log_dir}")
```

**New:**

```python
# Create output directories using V2 structure
log_dir = resolve_output_path(config, "logs")
log_dir.mkdir(parents=True, exist_ok=True)

print(f"Log directory: {log_dir}")
```

**Current (lines ~441-444):**

```python
# Create checkpoint directory (now from training section)
checkpoint_dir = Path(training_config["checkpoint_dir"])
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint directory: {checkpoint_dir}")
```

**New:**

```python
# Create checkpoint directory from output.subdirs
checkpoint_dir = resolve_output_path(config, "checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint directory: {checkpoint_dir}")
```

**Current (lines ~416-421):**

```python
# Save generated samples
output_dir = generation_config.get("output_dir")
if output_dir is None:
    output_dir = log_dir / "generated"
else:
    output_dir = Path(output_dir)
```

**New:**

```python
# Save generated samples to configured generated directory
output_dir = resolve_output_path(config, "generated")
```

**Validation:**

- All output paths should resolve correctly
- Directories should be created in expected locations

---

### Task 5.2: Update Optimizer Initialization

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update optimizer initialization for V2 config  
**Changes:**

**Current (lines ~467-487):**

```python
# Initialize optimizer
optimizer_name = training_config["optimizer"].lower()
optimizer_kwargs = training_config.get("optimizer_kwargs", {})

if optimizer_name == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        **optimizer_kwargs,
    )
elif optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        **optimizer_kwargs,
    )
else:
    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw"
    )

print(f"Optimizer: {optimizer_name}")
print(f"Learning rate: {training_config['learning_rate']}")
```

**New:**

```python
# Initialize optimizer from V2 config
optimizer_config = training_config["optimizer"]
optimizer_name = optimizer_config["type"].lower()

# Extract optimizer kwargs (excluding type, learning_rate, gradient_clip_norm)
optimizer_kwargs = {
    k: v for k, v in optimizer_config.items()
    if k not in ["type", "learning_rate", "gradient_clip_norm"]
}

if optimizer_name == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        **optimizer_kwargs,
    )
elif optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        **optimizer_kwargs,
    )
else:
    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. Supported: adam, adamw"
    )

print(f"Optimizer: {optimizer_name}")
print(f"Learning rate: {optimizer_config['learning_rate']}")
```

**Validation:**

- Optimizer should initialize with correct parameters
- Verify weight_decay, betas, etc. are applied

---

### Task 5.3: Update Scheduler Initialization

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update scheduler initialization for V2 config  
**Changes:**

**Current (lines ~492-523):**

```python
# Initialize scheduler if specified
scheduler = None
scheduler_name = training_config.get("scheduler", "none")
if scheduler_name and scheduler_name.lower() not in ["none", None]:
    scheduler_kwargs = training_config.get("scheduler_kwargs", {})

    if scheduler_name.lower() == "cosine":
        # Use epochs as T_max if not specified
        if "T_max" not in scheduler_kwargs:
            scheduler_kwargs["T_max"] = training_config["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_kwargs
        )
    elif scheduler_name.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
    elif scheduler_name.lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_kwargs
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            "Supported: cosine, step, plateau, none"
        )

    print(f"Scheduler: {scheduler_name}")
```

**New:**

```python
# Initialize scheduler if specified
scheduler = None
scheduler_config = training_config["scheduler"]
scheduler_name = scheduler_config.get("type")

if scheduler_name and scheduler_name.lower() not in ["none", None]:
    # Extract scheduler kwargs (excluding type)
    scheduler_kwargs = {
        k: v for k, v in scheduler_config.items()
        if k != "type"
    }

    # Handle auto T_max for cosine scheduler
    if scheduler_name.lower() == "cosine":
        if scheduler_kwargs.get("T_max") == "auto":
            scheduler_kwargs["T_max"] = training_config["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **scheduler_kwargs
        )
    elif scheduler_name.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
    elif scheduler_name.lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_kwargs
        )
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            "Supported: cosine, step, plateau, none"
        )

    print(f"Scheduler: {scheduler_name}")
```

**Validation:**

- Scheduler should initialize with correct parameters
- Auto T_max should work for cosine scheduler

---

### Task 5.4: Update Trainer Initialization (Training Mode)

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update trainer initialization for V2 config  
**Changes:**

**Current (lines ~531-546):**

```python
# Initialize trainer
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
    sample_images=training_config.get("visualization", {}).get("sample_images", True),
    sample_interval=training_config.get("visualization", {}).get("sample_interval", 10),
    samples_per_class=training_config.get("visualization", {}).get("samples_per_class", 2),
    guidance_scale=training_config.get("visualization", {}).get("guidance_scale", 3.0),
)
```

**New:**

```python
# Get configuration sections
ema_config = training_config["ema"]
performance_config = training_config["performance"]
visualization_config = training_config["visualization"]
optimizer_config = training_config["optimizer"]

# Initialize trainer with V2 config
trainer = DiffusionTrainer(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    logger=logger,
    device=device,
    show_progress=True,
    use_ema=ema_config["enabled"],
    ema_decay=ema_config["decay"],
    use_amp=performance_config["use_amp"],
    gradient_clip_norm=optimizer_config.get("gradient_clip_norm"),
    sample_images=visualization_config["enabled"],
    sample_interval=visualization_config["interval"],
    samples_per_class=visualization_config.get("num_samples", 8) // (model_config["conditioning"]["num_classes"] or 1),  # Convert num_samples to per_class
    guidance_scale=visualization_config["guidance_scale"],
)
```

**Note:** The `num_samples` to `samples_per_class` conversion needs careful handling  
**Alternative:** Update trainer to accept `num_samples` directly (see Task 5.7)

**Validation:**

- Trainer should initialize with correct parameters
- EMA, AMP, gradient clipping should work

---

### Task 5.5: Update Trainer Initialization (Generation Mode)

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update trainer initialization in generation mode  
**Changes:**

**Current (lines ~369-384):**

```python
# Initialize trainer (use generation config for EMA)
trainer = DiffusionTrainer(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    logger=logger,
    device=device,
    show_progress=True,
    use_ema=generation_config.get("use_ema", True),
    ema_decay=0.9999,  # Not used in generation
    use_amp=False,
    gradient_clip_norm=None,
    sample_images=False,  # Not used in generation mode
    sample_interval=1,
    samples_per_class=generation_config.get("samples_per_class", 2),
    guidance_scale=generation_config.get("guidance_scale", 3.0),
)
```

**New:**

```python
# Get generation configuration
sampling_config = generation_config["sampling"]

# Initialize trainer for generation mode
trainer = DiffusionTrainer(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    logger=logger,
    device=device,
    show_progress=True,
    use_ema=sampling_config["use_ema"],
    ema_decay=0.9999,  # Not used in generation, but required by constructor
    use_amp=False,  # No AMP in generation mode
    gradient_clip_norm=None,
    sample_images=False,  # Not used in generation mode
    sample_interval=1,
    samples_per_class=2,  # Will be overridden by generate_samples call
    guidance_scale=sampling_config["guidance_scale"],
)
```

**Validation:**

- Generation mode should work correctly
- EMA and guidance_scale should be applied

---

### Task 5.6: Update Generation Output Handling

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update generation output for V2 config  
**Changes:**

**Current (lines ~387-392):**

```python
# Generate samples
num_samples = generation_config.get("num_samples", 100)
print(f"\nGenerating {num_samples} samples...")
```

**New:**

```python
# Get generation parameters
sampling_config = generation_config["sampling"]
output_config = generation_config["output"]

num_samples = sampling_config["num_samples"]
print(f"\nGenerating {num_samples} samples...")
```

**Current (lines ~405-411):**

```python
# Generate samples
samples = trainer.generate_samples(
    num_samples=num_samples,
    class_labels=class_labels,
    guidance_scale=generation_config.get("guidance_scale", 3.0),
    use_ema=generation_config.get("use_ema", True),
)
```

**New:**

```python
# Generate samples with V2 config
samples = trainer.generate_samples(
    num_samples=num_samples,
    class_labels=class_labels,
    guidance_scale=sampling_config["guidance_scale"],
    use_ema=sampling_config["use_ema"],
)
```

**Current (lines ~423-432):**

```python
# Save as grid image
from torchvision.utils import save_image

save_image(
    samples, output_dir / "generated_samples.png", nrow=10, normalize=True
)
print(f"Saved generated samples to: {output_dir / 'generated_samples.png'}")

# Save individual samples
for i, sample in enumerate(samples):
    save_image(sample, output_dir / f"sample_{i:04d}.png", normalize=True)
```

**New:**

```python
# Save outputs according to V2 config
from torchvision.utils import save_image

if output_config["save_grid"]:
    grid_nrow = output_config["grid_nrow"]
    save_image(
        samples, output_dir / "generated_samples.png",
        nrow=grid_nrow, normalize=True
    )
    print(f"Saved generated grid to: {output_dir / 'generated_samples.png'}")

# Save individual samples if configured
if output_config["save_individual"]:
    for i, sample in enumerate(samples):
        save_image(sample, output_dir / f"sample_{i:04d}.png", normalize=True)
    print(f"Saved {len(samples)} individual samples to: {output_dir}")
```

**Validation:**

- Generated images should save according to config
- Grid and individual saves should be optional

---

### Task 5.7: Update Trainer Validation and Visualization Parameters

**File:** `src/experiments/diffusion/trainer.py`  
**Class:** `DiffusionTrainer`  
**Action:** Consider updating to accept `num_samples` instead of `samples_per_class`  
**Changes:**

**Option 1: Keep backward compatibility (Recommended)**

- No changes to trainer
- Handle conversion in main.py

**Option 2: Update trainer signature**

- Change `samples_per_class` parameter to `num_samples`
- Update internal logic to calculate samples per class
- Update all call sites

**Recommendation:** Option 1 for now, Option 2 in future refactor

**Validation:**

- If Option 2: Update all tests using `samples_per_class`

---

### Task 5.8: Update Train Method Call

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Update train() call for V2 config  
**Changes:**

**Current (lines ~554-568):**

```python
# Train the model
num_epochs = training_config["epochs"]
print(f"\nStarting training for {num_epochs} epochs...")

try:
    trainer.train(
        num_epochs=num_epochs,
        checkpoint_dir=str(checkpoint_dir),
        save_best=training_config.get("save_best_only", False),
        checkpoint_frequency=training_config.get("save_frequency", 10),
        validate_frequency=config.get("validation", {}).get("frequency", 1),
        best_metric=config.get("validation", {}).get("metric", "loss"),
    )
except KeyboardInterrupt:
    ...
```

**New:**

```python
# Get configuration sections
checkpointing_config = training_config["checkpointing"]
validation_config = training_config["validation"]

# Train the model
num_epochs = training_config["epochs"]
print(f"\nStarting training for {num_epochs} epochs...")

try:
    trainer.train(
        num_epochs=num_epochs,
        checkpoint_dir=str(checkpoint_dir),
        save_best=checkpointing_config["save_best_only"],
        checkpoint_frequency=checkpointing_config["save_frequency"],
        validate_frequency=validation_config["frequency"],
        best_metric=validation_config["metric"],
    )
except KeyboardInterrupt:
    ...
```

**Validation:**

- Training should run with correct parameters
- Checkpointing and validation should work as configured

---

### Task 5.9: Add Resume Training Support (New Feature)

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Add support for resuming training  
**Changes:**

**Add after trainer initialization (before trainer.train() call):**

```python
# Handle resume training if configured
resume_config = training_config["resume"]
if resume_config["enabled"]:
    checkpoint_path = resume_config.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("training.resume.checkpoint is required when resume.enabled=True")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    print(f"\nResuming training from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state (unless reset requested)
    if not resume_config["reset_optimizer"] and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Restored optimizer state")

    # Load scheduler state (unless reset requested)
    if scheduler is not None and not resume_config["reset_scheduler"]:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("Restored scheduler state")

    # Load EMA state if available
    if ema_config["enabled"] and "ema_state_dict" in checkpoint:
        trainer.ema.load_state_dict(checkpoint["ema_state_dict"])
        print("Restored EMA state")

    # Get starting epoch
    start_epoch = checkpoint.get("epoch", 0) + 1
    print(f"Resuming from epoch {start_epoch}")
else:
    start_epoch = 0
```

**Update trainer.train() call to accept start_epoch:**

```python
trainer.train(
    num_epochs=num_epochs,
    checkpoint_dir=str(checkpoint_dir),
    save_best=checkpointing_config["save_best_only"],
    checkpoint_frequency=checkpointing_config["save_frequency"],
    validate_frequency=validation_config["frequency"],
    best_metric=validation_config["metric"],
    start_epoch=start_epoch,  # New parameter
)
```

**Note:** May need to update BaseTrainer.train() to support start_epoch parameter

**Validation:**

- Resume training from saved checkpoint
- Verify optimizer/scheduler states are restored or reset as configured
- Verify epoch counting continues correctly

---

### Task 5.10: Add Performance Optimizations (New Feature)

**File:** `src/main.py`  
**Function:** `setup_experiment_diffusion()`  
**Action:** Add performance optimization support  
**Changes:**

**Add after model creation and before trainer initialization:**

```python
# Apply performance optimizations
performance_config = training_config["performance"]

# Enable TF32 on Ampere+ GPUs (PyTorch 1.7+)
if performance_config.get("use_tf32", True) and torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("Enabled TF32 for faster training on Ampere+ GPUs")

# Enable cuDNN benchmark mode
if performance_config.get("cudnn_benchmark", True) and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Enabled cuDNN benchmark mode")

# Compile model (PyTorch 2.0+)
if performance_config.get("compile_model", False):
    try:
        import torch._dynamo
        model = torch.compile(model)
        print("Compiled model with torch.compile()")
    except Exception as e:
        print(f"Warning: Failed to compile model: {e}")
```

**Validation:**

- Verify performance flags are applied
- Check for performance improvements with benchmarks
- Ensure compatibility with different PyTorch versions

---

## Phase 6: Testing Updates (4 hours)

### Task 6.1: Update Unit Tests for Config Module

**File:** `tests/experiments/diffusion/test_config.py`  
**Action:** Update all tests for V2 config structure  
**Changes:**

1. **Update `TestGetDefaultConfig` tests:**
   - Update `test_has_required_keys()` to check for new top-level keys: `compute`, updated `model`, etc.
   - Update `test_device_at_top_level()` to check `compute.device`
   - Update `test_seed_at_top_level()` to check `compute.seed`
   - Update `test_model_defaults()` to check nested structure: `architecture`, `diffusion`, `conditioning`
   - Update `test_data_defaults()` to check nested structure: `paths`, `loading`, `augmentation`
   - Add tests for new sections: `output.subdirs`, `training.optimizer.*`, `training.scheduler.*`, etc.

2. **Update `TestValidateConfig` tests:**
   - Update validation tests to use V2 structure
   - Add tests for new validation rules (cross-parameter consistency)
   - Add tests for `validate_config_consistency()`

3. **Add tests for new helper functions:**
   - `test_resolve_output_path()`
   - `test_derive_image_size_from_model()`
   - `test_derive_return_labels_from_model()`
   - `test_validate_config_consistency()`

4. **Add tests for backward compatibility:**
   - `test_migrate_config_v1_to_v2()`
   - Verify V1 configs are correctly migrated

**Example test update:**

```python
def test_model_architecture_section(self):
    """Test model.architecture configuration."""
    config = get_default_config()
    arch = config["model"]["architecture"]

    assert isinstance(arch, dict)
    assert "image_size" in arch
    assert "in_channels" in arch
    assert "model_channels" in arch
    assert "channel_multipliers" in arch
    assert "use_attention" in arch

def test_model_diffusion_section(self):
    """Test model.diffusion configuration."""
    config = get_default_config()
    diff = config["model"]["diffusion"]

    assert isinstance(diff, dict)
    assert "num_timesteps" in diff
    assert "beta_schedule" in diff
    assert "beta_start" in diff
    assert "beta_end" in diff
```

**Validation:**

- Run all config tests: `pytest tests/experiments/diffusion/test_config.py -v`
- All tests should pass

---

### Task 6.2: Update Unit Tests for DataLoader

**File:** `tests/experiments/diffusion/test_dataloader.py`  
**Action:** Add tests for derived parameters  
**Changes:**

1. Add test for derived `image_size`:

```python
def test_dataloader_with_derived_image_size():
    """Test that dataloader works with image_size derived from model config."""
    from src.utils.config import derive_image_size_from_model

    config = get_default_config()
    image_size = derive_image_size_from_model(config)

    dataloader = DiffusionDataLoader(
        train_path="tests/fixtures/mock_data/train",
        batch_size=4,
        image_size=image_size,  # Derived value
        return_labels=False
    )

    assert dataloader.image_size == config["model"]["architecture"]["image_size"]
```

2. Add test for derived `return_labels`:

```python
def test_dataloader_with_derived_return_labels():
    """Test that dataloader works with return_labels derived from model config."""
    from src.utils.config import derive_return_labels_from_model

    # Unconditional case
    config = get_default_config()
    config["model"]["conditioning"]["type"] = None
    return_labels = derive_return_labels_from_model(config)
    assert return_labels == False

    # Conditional case
    config["model"]["conditioning"]["type"] = "class"
    config["model"]["conditioning"]["num_classes"] = 2
    return_labels = derive_return_labels_from_model(config)
    assert return_labels == True
```

**Validation:**

- Run dataloader tests: `pytest tests/experiments/diffusion/test_dataloader.py -v`
- All tests should pass

---

### Task 6.3: Update Integration Tests

**File:** `tests/integration/test_diffusion_training.py` (if exists)  
**Action:** Update integration tests for V2 config  
**Changes:**

1. Update test configs to V2 structure
2. Update assertions that check config values
3. Add integration test for resume training
4. Add integration test for performance optimizations

**Example:**

```python
@pytest.mark.integration
def test_diffusion_training_with_v2_config(tmp_path):
    """Test full diffusion training pipeline with V2 config."""
    config = get_default_config()

    # Override for testing
    config["compute"]["device"] = "cpu"
    config["training"]["epochs"] = 2
    config["output"]["base_dir"] = str(tmp_path / "outputs")
    config["data"]["paths"]["train"] = "tests/fixtures/mock_data/train"
    config["data"]["loading"]["batch_size"] = 4

    # Save config
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    from src.main import setup_experiment_diffusion
    setup_experiment_diffusion(config)

    # Verify outputs
    assert (tmp_path / "outputs" / "logs").exists()
    assert (tmp_path / "outputs" / "checkpoints").exists()
```

**Validation:**

- Run integration tests: `pytest tests/integration/ -v -m integration`
- All tests should pass

---

### Task 6.4: Update Fixture Configs

**Files:**

- `tests/fixtures/configs/diffusion_minimal.yaml`
- `tests/fixtures/configs/diffusion/valid_minimal.yaml`
- `tests/fixtures/configs/diffusion/invalid_missing_data.yaml`

**Action:** Update all fixture configs to V2 structure  
**Changes:**

Apply same transformations as main config file, but keep minimal structure for valid_minimal.yaml

**Example `valid_minimal.yaml`:**

```yaml
experiment: diffusion
mode: train

compute:
  device: cpu
  seed: 42

model:
  architecture:
    image_size: 32
    in_channels: 3
    model_channels: 32
    channel_multipliers: [1, 2]
    use_attention: [false, true]
  diffusion:
    num_timesteps: 100
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
  conditioning:
    type: null
    num_classes: null
    class_dropout_prob: 0.1

data:
  paths:
    train: tests/fixtures/mock_data/train
    val: null
  loading:
    batch_size: 4
    num_workers: 0
    pin_memory: false
    shuffle_train: true
    drop_last: false
  augmentation:
    horizontal_flip: false
    rotation_degrees: 0
    color_jitter:
      enabled: false
      strength: 0.1

output:
  base_dir: outputs/test
  subdirs:
    logs: logs
    checkpoints: checkpoints
    samples: samples
    generated: generated

training:
  epochs: 2
  optimizer:
    type: adam
    learning_rate: 0.001
    weight_decay: 0.0
    betas: [0.9, 0.999]
    gradient_clip_norm: null
  scheduler:
    type: null
  ema:
    enabled: false
    decay: 0.9999
  checkpointing:
    save_frequency: 1
    save_best_only: false
    save_optimizer: true
  validation:
    enabled: false
    frequency: 1
    metric: loss
  visualization:
    enabled: false
    interval: 10
    num_samples: 4
    guidance_scale: 1.0
  performance:
    use_amp: false
    use_tf32: false
    cudnn_benchmark: false
    compile_model: false
  resume:
    enabled: false
    checkpoint: null
    reset_optimizer: false
    reset_scheduler: false

generation:
  checkpoint: null
  sampling:
    num_samples: 10
    guidance_scale: 1.0
    use_ema: false
  output:
    save_individual: true
    save_grid: false
    grid_nrow: 5
```

**Validation:**

- Load each fixture and verify it parses correctly
- Run tests that use these fixtures

---

## Phase 7: Documentation Updates (3 hours)

### Task 7.1: Update Main README

**File:** `README.md`  
**Action:** Update configuration examples for V2  
**Changes:**

1. Update "Configuration" section with V2 structure
2. Update example YAML snippets
3. Add note about V2 vs V1 config format
4. Add link to migration guide

**Example section to add:**

```markdown
### Configuration Structure (V2)

The configuration is organized into logical sections:

- `compute`: Device and seed settings
- `model`: Model architecture, diffusion parameters, and conditioning
  - `architecture`: U-Net architecture parameters
  - `diffusion`: Diffusion process parameters
  - `conditioning`: Conditional generation settings
- `data`: Dataset paths, loading, and augmentation
  - `paths`: Train and validation data paths
  - `loading`: Batch size, workers, memory settings
  - `augmentation`: Data augmentation settings
- `output`: Output directory structure
  - `base_dir`: Base output directory
  - `subdirs`: Subdirectories for logs, checkpoints, samples, generated images
- `training`: Training-specific parameters
  - `optimizer`: Optimizer configuration
  - `scheduler`: Learning rate scheduler
  - `ema`: Exponential moving average
  - `checkpointing`: Checkpoint saving
  - `validation`: Validation settings
  - `visualization`: Training visualization
  - `performance`: Performance optimizations
  - `resume`: Resume training settings
- `generation`: Generation-specific parameters
  - `sampling`: Sampling parameters
  - `output`: Generation output settings

See `configs/diffusion/default.yaml` for a complete example.
```

**Validation:**

- Review README for accuracy
- Verify examples match actual config structure

---

### Task 7.2: Create Migration Guide

**File:** `docs/research/diffusion-config-migration-guide.md`  
**Action:** Create comprehensive migration guide  
**Content:**

````markdown
# Diffusion Configuration Migration Guide: V1 to V2

This guide helps users migrate from V1 to V2 configuration format.

## Overview

Configuration V2 provides better organization and eliminates parameter duplication.

## Key Changes

### 1. Device and Seed → Compute Section

**V1:**

```yaml
device: cuda
seed: 42
```
````

**V2:**

```yaml
compute:
  device: cuda
  seed: 42
```

### 2. Model Section Restructured

**V1:**

```yaml
model:
  image_size: 40
  in_channels: 3
  model_channels: 64
  channel_multipliers: [1, 2, 4]
  num_classes: 2
  num_timesteps: 1000
  beta_schedule: cosine
  beta_start: 0.0001
  beta_end: 0.02
  class_dropout_prob: 0.1
  use_attention: [false, false, true]
```

**V2:**

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
    type: class # or null for unconditional
    num_classes: 2
    class_dropout_prob: 0.1
```

[... continue with all other sections ...]

## Automatic Migration

Use the migration script:

```bash
python scripts/migrate_config_v1_to_v2.py --input old_config.yaml --output new_config.yaml
```

Or use backward compatibility in code (with warnings):

```python
from src.utils.config import migrate_config_v1_to_v2

old_config = load_config("old_config.yaml")
new_config = migrate_config_v1_to_v2(old_config)
```

## Breaking Changes

1. `data.image_size` removed - now derived from `model.architecture.image_size`
2. `data.return_labels` removed - now derived from `model.conditioning.type`
3. `training.checkpoint_dir` moved to `output.subdirs.checkpoints`
4. Optimizer parameters now nested under `training.optimizer`
5. Scheduler parameters now nested under `training.scheduler`
6. EMA parameters now nested under `training.ema`
7. Visualization section moved from `generation` to `training.visualization`

## Benefits

- Single source of truth (no duplicate parameters)
- Logical grouping of related parameters
- Mode-specific sections properly scoped
- Easier to maintain and understand

## Support

For issues or questions, see:

- [V2 Optimization Plan](diffusion-config-v2-optimization-plan.md)
- [Implementation Tasks](diffusion-config-v2-implementation-tasks.md)

````

**Validation:**
- Review migration guide with actual V1 configs
- Verify all changes are documented

---

### Task 7.3: Create Migration Script
**File:** `scripts/migrate_config_v1_to_v2.py`
**Action:** Create standalone migration script
**Content:**

```python
#!/usr/bin/env python3
"""Migrate diffusion configuration from V1 to V2 format.

Usage:
    python scripts/migrate_config_v1_to_v2.py --input old_config.yaml --output new_config.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import migrate_config_v1_to_v2


def main():
    parser = argparse.ArgumentParser(
        description="Migrate diffusion configuration from V1 to V2 format"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to V1 configuration file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to save V2 configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print V2 config without saving",
    )

    args = parser.parse_args()

    # Load V1 config
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading V1 config from: {input_path}")
    with open(input_path, "r") as f:
        v1_config = yaml.safe_load(f)

    # Migrate to V2
    print("Migrating to V2 format...")
    v2_config = migrate_config_v1_to_v2(v1_config)

    # Save or print
    if args.dry_run:
        print("\nMigrated V2 config (dry run):")
        print(yaml.dump(v2_config, default_flow_style=False, sort_keys=False))
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(v2_config, f, default_flow_style=False, sort_keys=False)

        print(f"\nV2 config saved to: {output_path}")
        print("\nPlease review the migrated config and test before using in production.")


if __name__ == "__main__":
    main()
````

**Make executable:**

```bash
chmod +x scripts/migrate_config_v1_to_v2.py
```

**Validation:**

- Test with legacy.yaml
- Verify output matches expected V2 structure

---

### Task 7.4: Update Inline Documentation

**Files:**

- `src/experiments/diffusion/config.py`
- `src/experiments/diffusion/dataloader.py`
- `src/experiments/diffusion/trainer.py`
- `src/main.py`

**Action:** Update docstrings and comments for V2 config  
**Changes:**

1. Update module docstrings to mention V2 config format
2. Update function docstrings with V2 config examples
3. Update inline comments that reference config structure
4. Add deprecation warnings if using V1 format (where applicable)

**Example update in `config.py`:**

```python
"""Diffusion Configuration (V2 Format)

This module provides configuration management for diffusion model experiments
using the V2 configuration format. V2 provides better organization and
eliminates parameter duplication.

Key improvements in V2:
- Logical grouping of related parameters
- Single source of truth (no duplicate parameters)
- Mode-specific sections properly scoped
- Derived parameters (image_size, return_labels)

For migration from V1, see: docs/research/diffusion-config-migration-guide.md
"""
```

**Validation:**

- Review docstrings for accuracy
- Run doctests if present: `pytest --doctest-modules`

---

### Task 7.5: Update Configuration Comments

**File:** `configs/diffusion/default.yaml`  
**Action:** Add comprehensive comments explaining V2 structure  
**Changes:**

Add comments throughout the config file:

```yaml
# ==============================================================================
# DIFFUSION MODEL CONFIGURATION (V2 FORMAT)
# ==============================================================================
# This is the V2 configuration format with improved organization and
# no parameter duplication. For migration guide, see:
# docs/research/diffusion-config-migration-guide.md
# ==============================================================================

experiment: diffusion
mode: train # Options: train, generate

# ==============================================================================
# COMPUTE CONFIGURATION
# ==============================================================================
# Device and reproducibility settings used across all modes

compute:
  device: cuda # Options: cuda, cpu, auto
  seed: null # Random seed for reproducibility (null to disable)

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
# Model architecture, diffusion process, and conditioning settings

model:
  # Architecture: U-Net structure and channels
  architecture:
    image_size: 40 # Size of generated images (H=W)
    in_channels: 3 # Number of input channels (3 for RGB)
    model_channels: 64 # Base number of U-Net channels
    channel_multipliers: [1, 2, 4] # Channel multipliers per stage
    use_attention: [false, false, true] # Attention at each stage

  # Diffusion: Noise scheduling and timesteps
  diffusion:
    num_timesteps: 1000 # Number of diffusion timesteps
    beta_schedule: cosine # Options: linear, cosine, quadratic, sigmoid
    beta_start: 0.0001 # Starting beta value
    beta_end: 0.02 # Ending beta value

  # Conditioning: Class conditioning for guided generation
  conditioning:
    type: null # Options: null (unconditional), "class" (class-conditional)
    num_classes: null # Number of classes (required if type="class")
    class_dropout_prob: 0.1 # Dropout for classifier-free guidance


# [... continue with detailed comments for each section ...]
```

**Validation:**

- Verify comments areaccurate and helpful
- Get feedback from users if possible

---

## Implementation Checklist

### Phase 1: Configuration Files (2 hours) ✅ COMPLETED

- [x] Create new V2 configuration file
- [x] Backup old configuration to legacy.yaml
- [x] Update test fixture configs

### Phase 2: Configuration Loading & Validation (4 hours) ✅ COMPLETED

- [x] Update default configuration generator
- [x] Update configuration validation
- [x] Add configuration helper functions
- [x] Add backward compatibility layer

### Phase 3: Data Loading Updates (2 hours) ✅ COMPLETED

- [x] Update dataloader instantiation in main.py (2 locations)

### Phase 4: Model Initialization Updates (2 hours) ✅ COMPLETED

- [x] Update model creation in main.py
- [x] Update device and seed handling

### Phase 5: Training Logic Updates (4 hours) ✅ COMPLETED

- [x] Update output directory handling
- [x] Update optimizer initialization
- [x] Update scheduler initialization
- [x] Update trainer initialization (training mode)
- [x] Update trainer initialization (generation mode)
- [x] Update generation output handling
- [x] Update train method call
- [x] Add resume training support
- [x] Add performance optimizations

### Phase 6: Testing Updates (4 hours) ✅ COMPLETED

- [x] Update fixture configs for tests
- [x] Update unit tests for config module
- [x] Update unit tests for dataloader
- [x] Update integration tests

### Phase 7: Documentation Updates (3 hours) ✅ COMPLETED

- [x] Update main README
- [x] Create migration guide
- [x] Create migration script
- [x] Update inline documentation
- [x] Update configuration comments

---

## Testing Strategy

### Unit Testing

Run after each phase:

```bash
pytest tests/experiments/diffusion/test_config.py -v
pytest tests/experiments/diffusion/test_dataloader.py -v
pytest tests/experiments/diffusion/test_trainer.py -v
```

### Integration Testing

Run after Phase 5:

```bash
pytest tests/integration/test_diffusion.py -v -m integration
```

### End-to-End Testing

Run after all phases:

```bash
# Test training mode
python -m src.main configs/diffusion/default.yaml

# Test generation mode
python -m src.main configs/diffusion/generate.yaml

# Test resume mode
# (After running training, update config to enable resume)
```

### Backward Compatibility Testing

```bash
# Test V1 config with compatibility layer
python -m src.main configs/diffusion/legacy.yaml
```

---

## Rollback Plan

If critical issues are discovered:

1. **Revert configuration file:**

   ```bash
   cp configs/diffusion/legacy.yaml configs/diffusion/default.yaml
   ```

2. **Revert code changes:**

   ```bash
   git revert <commit-hash>
   ```

3. **Document issues:**
   - Create GitHub issue with details
   - Add to known issues in documentation

---

## Success Criteria

- [x] All unit tests pass
- [x] All integration tests pass
- [x] Training runs successfully with V2 config
- [x] Generation runs successfully with V2 config
- [x] Resume training works correctly (implemented, not tested in full workflow)
- [x] Performance optimizations apply correctly (implemented, not tested in full workflow)
- [x] Backward compatibility works (with warnings) - migrate_config_v1_to_v2 function available
- [x] Migration script successfully converts V1 to V2
- [x] Documentation is comprehensive and accurate

---

## Post-Implementation Tasks

1. **Deprecation notice:**
   - Add deprecation warning for V1 config (effective date)
   - Plan to remove backward compatibility in future version

2. **Performance benchmarks:**
   - Compare training performance before/after
   - Document performance improvements from new optimizations

3. **User feedback:**
   - Gather feedback on new configuration structure
   - Make adjustments based on user experience

4. **Future enhancements:**
   - Consider adding configuration validation schemas (JSON Schema/Pydantic)
   - Consider adding configuration presets for common use cases

---

## Notes

- Each task is designed to be atomic and testable
- Tasks within a phase can sometimes be parallelized, but phases must be sequential
- Estimated times include implementation, testing, and documentation
- Review changes at end of each phase before proceeding

---

**Last Updated:** February 13, 2026  
**Status:** Ready for Implementation
