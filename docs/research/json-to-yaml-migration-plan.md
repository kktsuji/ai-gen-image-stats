# JSON to YAML Migration Plan

**Date:** February 12, 2026  
**Author:** GitHub Copilot  
**Status:** Planning

## Executive Summary

This document outlines the plan to migrate the configuration file format from JSON to YAML across the entire project. This change affects configuration loading, saving, CLI arguments, tests, documentation, and all existing config files.

See the [TODO List](#executable-step-by-step-todo-list) below for the main actionable content.

### Current Behavior vs Target Behavior

**Current (JSON format):**

```bash
python -m src.main configs/classifier/baseline.json
python -m src.main configs/diffusion/default.json
```

**Target (YAML format):**

```bash
python -m src.main configs/classifier/baseline.yaml
python -m src.main configs/diffusion/default.yaml
```

### Quick Reference

| Aspect                | Detail                               |
| --------------------- | ------------------------------------ |
| **Estimated Time**    | 8-10 hours (1-2 sessions)            |
| **Files Modified**    | 30+ files                            |
| **Breaking Change**   | YES (config file format)             |
| **New Config Format** | YAML (`.yaml` extension)             |
| **Dependency**        | PyYAML (already in requirements.txt) |

**Navigation:**

- [Executable TODO List](#executable-step-by-step-todo-list) - Main task list
- [Risk Assessment](#risk-assessment) - Risks and mitigations
- [Timeline & Success Criteria](#estimated-timeline) - Planning info
- [Appendix A](#appendix-a-investigation-summary) - Investigation details
- [Appendix B](#appendix-b-detailed-change-plan) - Technical specifications

---

## Executable Step-by-Step TODO List

### Prerequisites

- [x] Create feature branch: `git checkout -b feature/yaml-config-migration`
- [x] Ensure virtual environment is activated
- [x] Verify PyYAML is installed: `python -c "import yaml; print(yaml.__version__)"`
- [x] Run existing tests to establish baseline: `pytest`

---

### Step 1: Update Core Configuration Utilities

**Goal:** Change config loading/saving from JSON to YAML

#### 1.1 Update `src/utils/config.py`

- [x] 1.1.1 Change import statement:
  - Remove: `import json`
  - Add: `import yaml`

- [x] 1.1.2 Update `load_config()` function:
  - Update docstring: "JSON file" → "YAML file"
  - Change `json.load(f)` → `yaml.safe_load(f)`
  - Update exception handling: `json.JSONDecodeError` → `yaml.YAMLError`
  - Update docstring raises section

- [x] 1.1.3 Update `save_config()` function:
  - Update docstring: "JSON file" → "YAML file"
  - Change `json.dump(config, f, indent=indent)` → `yaml.dump(config, f, default_flow_style=False, sort_keys=False)`
  - Update parameter docs

- [x] 1.1.4 Update `load_and_merge_configs()` docstring:
  - Change "JSON configuration file" → "YAML configuration file"

**Validation:** `python -c "from src.utils.config import load_config, save_config; print('OK')"`

**Effort:** 30 minutes

---

### Step 2: Update CLI Module

**Goal:** Update CLI to accept YAML files

#### 2.1 Update `src/utils/cli.py`

- [x] 2.1.1 Update `create_parser()`:
  - Change help text: "Path to JSON configuration file" → "Path to YAML configuration file"
  - Update docstring examples: `baseline.json` → `baseline.yaml`

- [x] 2.1.2 Update `parse_args()` function:
  - Update docstring examples: `baseline.json` → `baseline.yaml`
  - Update exception docstring: `json.JSONDecodeError` → `yaml.YAMLError`

**Validation:** `python -c "from src.utils.cli import create_parser; print('OK')"`

**Effort:** 20 minutes

---

### Step 3: Update Main Entry Point

**Goal:** Update main.py to use YAML

#### 3.1 Update `src/main.py`

- [x] 3.1.1 Change import statement:
  - Remove: `import json`
  - Add: `import yaml`

- [x] 3.1.2 Update module docstring:
  - Change all `.json` → `.yaml` in usage examples
  - Line 9: `baseline.json` → `baseline.yaml`
  - Line 10: `inceptionv3.json` → `inceptionv3.yaml`
  - Line 13: `default.json` → `default.yaml`
  - Line 16: `generate.json` → `generate.yaml`
  - Update note: "JSON config file" → "YAML config file"

- [x] 3.1.3 Update `setup_experiment_classifier()`:
  - Line ~86: Change `config_save_path = log_dir / "config.json"` → `config.yaml`
  - Line ~88: Change `json.dump(config, f, indent=2)` → `yaml.dump(config, f, default_flow_style=False, sort_keys=False)`

- [x] 3.1.4 Update `setup_experiment_diffusion()`:
  - Line ~299: Change `config_save_path = log_dir / "config.json"` → `config.yaml`
  - Line ~301: Change `json.dump(config, f, indent=2)` → `yaml.dump(config, f, default_flow_style=False, sort_keys=False)`

**Validation:** `python -c "from src.main import main; print('OK')"`

**Effort:** 30 minutes

---

### Step 4: Update Experiment Loggers

**Goal:** Update hyperparameter saving to YAML

#### 4.1 Update `src/experiments/classifier/logger.py`

- [x] 4.1.1 Add import: `import yaml`
- [x] 4.1.2 Update hyperparameter saving (around line 375):
  - Change `hyperparams_file = self.log_dir / "hyperparams.json"` → `hyperparams.yaml`
  - Change `json.dump(hyperparams, f, indent=2)` → `yaml.dump(hyperparams, f, default_flow_style=False, sort_keys=False)`

#### 4.2 Update `src/experiments/diffusion/logger.py`

- [x] 4.2.1 Add import: `import yaml`
- [x] 4.2.2 Update hyperparameter saving (around line 342):
  - Change `hyperparams_file = self.log_dir / "hyperparams.json"` → `hyperparams.yaml`
  - Change `json.dump(hyperparams, f, indent=2)` → `yaml.dump(hyperparams, f, default_flow_style=False, sort_keys=False)`

**Validation:** `python -c "from src.experiments.classifier.logger import ClassifierLogger; from src.experiments.diffusion.logger import DiffusionLogger; print('OK')"`

**Effort:** 20 minutes

---

### Step 5: Convert Configuration Files

**Goal:** Convert all JSON config files to YAML format

#### 5.1 Convert Main Config Files

- [x] 5.1.1 Convert `configs/classifier/baseline.json` → `baseline.yaml`
  - Read JSON file
  - Write as YAML with proper formatting
  - Verify structure is preserved
  - Delete JSON file

- [x] 5.1.2 Convert `configs/classifier/inceptionv3.json` → `inceptionv3.yaml`

- [x] 5.1.3 Convert `configs/diffusion/default.json` → `default.yaml`

#### 5.2 Convert Test Fixture Config Files

- [x] 5.2.1 Convert `tests/fixtures/configs/classifier_minimal.json` → `classifier_minimal.yaml`
- [x] 5.2.2 Convert `tests/fixtures/configs/diffusion_minimal.json` → `diffusion_minimal.yaml`
- [x] 5.2.3 Convert `tests/fixtures/configs/gan_minimal.json` → `gan_minimal.yaml`
- [x] 5.2.4 Convert `tests/fixtures/configs/classifier/valid_minimal.json` → `valid_minimal.yaml`
- [x] 5.2.5 Convert `tests/fixtures/configs/classifier/invalid_missing_model.json` → `invalid_missing_model.yaml`
- [x] 5.2.6 Convert `tests/fixtures/configs/diffusion/valid_minimal.json` → `valid_minimal.yaml`
- [x] 5.2.7 Convert `tests/fixtures/configs/diffusion/invalid_missing_data.json` → `invalid_missing_data.yaml`

**Helper Script:** Create a Python script to automate conversion:

```python
import json
import yaml
from pathlib import Path

def convert_json_to_yaml(json_path: Path):
    """Convert a JSON file to YAML format."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    yaml_path = json_path.with_suffix('.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Converted: {json_path} → {yaml_path}")
    return yaml_path

# Usage:
# convert_json_to_yaml(Path('configs/classifier/baseline.json'))
```

**Validation:** Verify YAML files load correctly with `yaml.safe_load()`

**Effort:** 1 hour (including verification)

---

### Step 6: Update Tests

**Goal:** Update all test files to use YAML format

#### 6.1 Update `tests/utils/test_config.py`

- [x] 6.1.1 Update imports:
  - Change: `import json` → `import yaml`

- [x] 6.1.2 Update `TestLoadConfig` class:
  - Change all `json.dump()` → `yaml.dump()` in test setup
  - Change all `.json` → `.yaml` in filenames
  - Update test names if they reference JSON
  - Change error assertions: `json.JSONDecodeError` → `yaml.YAMLError`
  - Update test `test_load_config_invalid_json` → `test_load_config_invalid_yaml`

- [x] 6.1.3 Update `TestMergeConfigs` class:
  - No changes needed (logic tests only)

- [x] 6.1.4 Update `TestLoadAndMergeConfigs` class:
  - Change all `json.dump()` → `yaml.dump()` in test setup
  - Change all `.json` → `.yaml` in filenames
  - Update fixture path references

- [x] 6.1.5 Update `TestSaveConfig` class:
  - Change all `json.load()` → `yaml.safe_load()` in verification
  - Change all `.json` → `.yaml` in filenames

#### 6.2 Update `tests/utils/test_cli.py`

- [x] 6.2.1 Update imports:
  - Change: `import json` → `import yaml`

- [x] 6.2.2 Update `TestCreateParser` class:
  - Change all examples: `.json` → `.yaml`
  - Line 32: `"configs/test.json"` → `"configs/test.yaml"`
  - Line 44: `"configs/test.json"` → `"configs/test.yaml"`
  - Line 53: `"configs/test.json"` → `"configs/test.yaml"`

- [x] 6.2.3 Update `TestParseArgs` class:
  - Change all `json.dump()` → `yaml.dump()` in test setup
  - Change all `.json` → `.yaml` in filenames
  - Update all config file creation

- [x] 6.2.4 Update `TestValidateConfig` class:
  - Update config file creation if any

- [x] 6.2.5 Update `TestConfigOnlyMode` class:
  - Change all `.json` → `.yaml` in examples

#### 6.3 Update `tests/test_infrastructure.py`

- [x] 6.3.1 Update fixture config references:
  - Line 157: `classifier_minimal.json` → `classifier_minimal.yaml`
  - Line 158: `diffusion_minimal.json` → `diffusion_minimal.yaml`
  - Line 159: `gan_minimal.json` → `gan_minimal.yaml`

#### 6.4 Update Integration Tests

- [x] 6.4.1 Update `tests/integration/test_classifier_pipeline.py`:
  - Search for `.json` and replace with `.yaml`
  - Line 728: `"test_config.json"` → `"test_config.yaml"`
  - Line 820: `"config.json"` → `"config.yaml"`
  - Update `json.dump()` → `yaml.dump()`

- [x] 6.4.2 Update `tests/integration/test_diffusion_pipeline.py`:
  - Line 1003: `"config.json"` → `"config.yaml"`
  - Update `json.dump()` → `yaml.dump()`

**Validation:** Run all tests: `pytest tests/ -v`

**Effort:** 2-3 hours

---

### Step 7: Update Documentation

**Goal:** Update all documentation to reflect YAML format

#### 7.1 Update `README.md`

- [ ] 7.1.1 Update "Configuration Management" section:
  - Line 23: "JSON-based configs" → "YAML-based configs"

- [ ] 7.1.2 Update "Architecture" section:
  - Line 40: "Experiment configurations (JSON)" → "Experiment configurations (YAML)"

- [ ] 7.1.3 Update "Quick Start" section:
  - Line 77: `baseline.json` → `baseline.yaml`
  - Line 80: `inceptionv3.json` → `inceptionv3.yaml`
  - Line 87: `default.json` → `default.yaml`

- [ ] 7.1.4 Update note about configuration:
  - Change references from "JSON" to "YAML"

#### 7.2 Update `docs/standards/architecture.md`

- [ ] 7.2.1 Update "Configuration Driven" section:
  - Line 33: "via JSON files" → "via YAML files"

- [ ] 7.2.2 Update "Directory Structure" section:
  - Line 88: "Config loading (JSON/YAML)" → "Config loading (YAML)"
  - Lines 110-118: Change all `.json` → `.yaml`

- [ ] 7.2.3 Update "CLI Interface" section:
  - Line 181: "JSON configuration file" → "YAML configuration file"
  - Lines 196, 199, 202: Change `.json` → `.yaml` in examples

- [ ] 7.2.4 Update error examples:
  - Lines 213-214: `nonexistent.json` → `nonexistent.yaml`
  - Line 217: `incomplete_config.json` → `incomplete_config.yaml`

#### 7.3 Update `docs/research/cli-migration-guide.md` (if exists)

- [ ] 7.3.1 Update all `.json` → `.yaml` references
- [ ] 7.3.2 Add note about JSON to YAML migration

#### 7.4 Update `docs/research/cli-refactoring-plan.md`

- [ ] 7.4.1 Add note at top indicating JSON→YAML migration completed
- [ ] 7.4.2 Update references throughout (optional, as this is historical)

**Validation:** Review docs for consistency

**Effort:** 1 hour

---

### Step 8: Create Migration Helper Script

**Goal:** Create reusable script for JSON to YAML conversion

- [ ] 8.1 Create `scripts/convert_json_to_yaml.py`:

```python
#!/usr/bin/env python3
"""Convert JSON configuration files to YAML format.

This script converts all JSON config files in the project to YAML format,
preserving the structure and data.

Usage:
    python scripts/convert_json_to_yaml.py
    python scripts/convert_json_to_yaml.py --path configs/
    python scripts/convert_json_to_yaml.py --file configs/classifier/baseline.json
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Optional


def convert_file(json_path: Path, delete_json: bool = False) -> Optional[Path]:
    """Convert a single JSON file to YAML.

    Args:
        json_path: Path to JSON file
        delete_json: Whether to delete JSON file after conversion

    Returns:
        Path to created YAML file, or None if conversion failed
    """
    try:
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Save as YAML
        yaml_path = json_path.with_suffix('.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Converted: {json_path.name} → {yaml_path.name}")

        # Delete JSON if requested
        if delete_json:
            json_path.unlink()
            print(f"  Deleted: {json_path.name}")

        return yaml_path

    except Exception as e:
        print(f"✗ Error converting {json_path}: {e}")
        return None


def convert_directory(dir_path: Path, delete_json: bool = False, recursive: bool = True):
    """Convert all JSON files in a directory.

    Args:
        dir_path: Path to directory
        delete_json: Whether to delete JSON files after conversion
        recursive: Whether to search subdirectories
    """
    pattern = "**/*.json" if recursive else "*.json"
    json_files = list(dir_path.glob(pattern))

    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return

    print(f"Found {len(json_files)} JSON file(s) in {dir_path}")
    print()

    converted = 0
    for json_file in json_files:
        if convert_file(json_file, delete_json):
            converted += 1

    print()
    print(f"Converted {converted}/{len(json_files)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON config files to YAML format"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Directory to convert (default: configs/)",
        default="configs/",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Single file to convert (overrides --path)",
    )
    parser.add_argument(
        "--delete-json",
        action="store_true",
        help="Delete JSON files after successful conversion",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )

    args = parser.parse_args()

    if args.file:
        # Convert single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1

        convert_file(file_path, args.delete_json)

    else:
        # Convert directory
        dir_path = Path(args.path)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            return 1

        convert_directory(dir_path, args.delete_json, not args.no_recursive)

    return 0


if __name__ == "__main__":
    exit(main())
```

- [ ] 8.2 Create `scripts/` directory if it doesn't exist
- [ ] 8.3 Make script executable: `chmod +x scripts/convert_json_to_yaml.py`
- [ ] 8.4 Test script: `python scripts/convert_json_to_yaml.py --file test.json`

**Validation:** Script converts files correctly

**Effort:** 30 minutes

---

### Step 9: Final Testing

**Goal:** Comprehensive validation of all changes

- [ ] 9.1 **Unit Tests:**

  ```bash
  pytest tests/utils/test_config.py -v
  pytest tests/utils/test_cli.py -v
  pytest -m unit -v
  ```

- [ ] 9.2 **Component Tests:**

  ```bash
  pytest -m component -v
  ```

- [ ] 9.3 **Integration Tests:**

  ```bash
  pytest -m integration -v
  ```

- [ ] 9.4 **Full Test Suite:**

  ```bash
  pytest -v
  ```

- [ ] 9.5 **Manual Testing - Classifier:**

  ```bash
  python -m src.main configs/classifier/baseline.yaml
  # Verify: Config loaded, training starts, config.yaml saved to outputs
  ```

- [ ] 9.6 **Manual Testing - Diffusion:**

  ```bash
  python -m src.main configs/diffusion/default.yaml
  # Verify: Config loaded, training starts, config.yaml saved to outputs
  ```

- [ ] 9.7 **Error Cases:**

  ```bash
  # Test missing file
  python -m src.main nonexistent.yaml

  # Test invalid YAML
  echo "invalid: yaml: content:" > invalid.yaml
  python -m src.main invalid.yaml

  # Test JSON file (should fail or warn)
  python -m src.main configs/old_config.json
  ```

- [ ] 9.8 **Verify Generated Files:**
  - Check `outputs/logs/config.yaml` exists and is valid
  - Check `outputs/logs/hyperparams.yaml` exists and is valid

**Validation:** All tests pass, configs load correctly

**Effort:** 1-2 hours

---

### Step 10: Code Review and Cleanup

**Goal:** Final polish and documentation

- [ ] 10.1 Remove any remaining `.json` references:

  ```bash
  grep -r "\.json" src/ tests/ docs/ --include="*.py" --include="*.md"
  ```

- [ ] 10.2 Check for unused imports:

  ```bash
  grep -r "import json" src/ tests/ --include="*.py"
  # Verify all are legitimate uses (not config-related)
  ```

- [ ] 10.3 Format code:

  ```bash
  black src/utils/config.py src/utils/cli.py src/main.py
  black src/experiments/classifier/logger.py src/experiments/diffusion/logger.py
  black tests/utils/test_config.py tests/utils/test_cli.py
  ```

- [ ] 10.4 Review git diff:

  ```bash
  git diff main
  git status
  ```

- [ ] 10.5 Update CHANGELOG.md (if exists):
  - Add entry for JSON → YAML migration
  - Note breaking change
  - Provide migration instructions

**Validation:** Code is clean and consistent

**Effort:** 1 hour

---

### Step 11: Commit and Document

**Goal:** Commit changes with clear documentation

- [ ] 11.1 Stage all changes:

  ```bash
  git add -A
  git status  # Review
  ```

- [ ] 11.2 Commit with descriptive message:

  ```bash
  git commit -m "Migrate config format from JSON to YAML

  - Update config loading/saving to use YAML format
  - Convert all config files from .json to .yaml
  - Update CLI to accept .yaml files
  - Update all tests to use YAML format
  - Update documentation (README, architecture docs)
  - Add conversion script for future migrations

  BREAKING CHANGE: Config files must now be in YAML format
  - All existing .json configs converted to .yaml
  - save_config() now outputs YAML instead of JSON
  - CLI expects .yaml file extension

  Migration: Use scripts/convert_json_to_yaml.py to convert
  custom config files from JSON to YAML format.
  "
  ```

- [ ] 11.3 Create/update migration guide `docs/research/json-to-yaml-migration.md`

- [ ] 11.4 Push to remote:

  ```bash
  git push -u origin feature/yaml-config-migration
  ```

- [ ] 11.5 Create Pull Request with:
  - Link to this planning document
  - Summary of changes
  - Testing performed
  - Migration instructions

**Validation:** Changes committed and documented

**Effort:** 30 minutes

---

## Risk Assessment

### High Risk

- **Breaking Change:** All existing JSON config files become invalid
  - **Mitigation:** Provide conversion script, clear migration guide, convert all existing configs

### Medium Risk

- **YAML Format Edge Cases:** YAML parsing may interpret some values differently than JSON
  - **Mitigation:** Use `yaml.safe_load()` for security, test thoroughly, document any differences
  - **Examples:**
    - `true/false` vs `True/False` (both valid in YAML)
    - Numeric strings may need quoting
    - `null` vs `None` (YAML uses `null` or `~`)

- **Test Coverage:** 30+ files to update
  - **Mitigation:** Systematic testing, update tests incrementally, use helper script

### Low Risk

- **Documentation Drift:** Multiple doc files to update
  - **Mitigation:** 1 hour allocated for docs, checklist approach

- **Third-party Dependencies:** PyYAML API changes
  - **Mitigation:** PyYAML is stable and already in requirements.txt

---

## Estimated Timeline

| Phase     | Task                  | Time           |
| --------- | --------------------- | -------------- |
| 1         | Core config utilities | 30 min         |
| 2         | CLI module            | 20 min         |
| 3         | Main entry point      | 30 min         |
| 4         | Experiment loggers    | 20 min         |
| 5         | Convert config files  | 1 hour         |
| 6         | Update tests          | 2-3 hours      |
| 7         | Update documentation  | 1 hour         |
| 8         | Create helper script  | 30 min         |
| 9         | Final testing         | 1-2 hours      |
| 10        | Code review & cleanup | 1 hour         |
| 11        | Commit & document     | 30 min         |
| **Total** |                       | **8-10 hours** |

---

## Success Criteria

- [ ] All config files converted from JSON to YAML
- [ ] All code uses `yaml.safe_load()` and `yaml.dump()`
- [ ] CLI accepts `.yaml` files
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Conversion script available for custom configs
- [ ] Clear migration guide created
- [ ] No references to `.json` configs remain (except in old/\* folder)

---

## Additional Considerations

### YAML Format Guidelines

When converting JSON to YAML, follow these conventions:

**DO:**

- Use 2-space indentation (consistent with JSON)
- Use `default_flow_style=False` for readable formatting
- Use `sort_keys=False` to preserve logical ordering
- Use lowercase booleans: `true`, `false`, `null`
- Quote strings that might be misinterpreted (e.g., version numbers like "3.10")

**DON'T:**

- Don't use flow style (inline JSON-like syntax)
- Don't use tabs for indentation
- Avoid using YAML anchors/aliases initially (can be added later for DRY configs)

**Example Conversion:**

JSON:

```json
{
  "experiment": "classifier",
  "model": {
    "name": "resnet50",
    "pretrained": true,
    "num_classes": 2
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.001,
    "device": "cuda"
  }
}
```

YAML:

```yaml
experiment: classifier
model:
  name: resnet50
  pretrained: true
  num_classes: 2
training:
  epochs: 100
  learning_rate: 0.001
  device: cuda
```

### Future Enhancements

Once YAML migration is complete, consider:

1. **YAML Anchors for DRY Configs:**

   ```yaml
   base_training: &base_training
     epochs: 100
     learning_rate: 0.001

   experiment_1:
     training:
       <<: *base_training
       batch_size: 32
   ```

2. **Config Validation with JSON Schema:**
   - Create schema files in YAML format
   - Validate configs against schema before training

3. **Environment Variable Interpolation:**
   - Support `${ENV_VAR}` syntax in YAML configs
   - Useful for paths, API keys, etc.

4. **Multi-file Configs:**
   - Split large configs into multiple YAML files
   - Use YAML's `!include` tag or custom loader

---

# Appendices

## Appendix A: Investigation Summary

### Files to Modify

**Core Python Files (9 files):**

1. `src/utils/config.py` - Load/save functions
2. `src/utils/cli.py` - CLI help text
3. `src/main.py` - Examples and config saving
4. `src/experiments/classifier/logger.py` - Hyperparams saving
5. `src/experiments/diffusion/logger.py` - Hyperparams saving
6. `tests/utils/test_config.py` - Config tests
7. `tests/utils/test_cli.py` - CLI tests
8. `tests/test_infrastructure.py` - Fixture references
9. `tests/integration/test_classifier_pipeline.py` - Integration tests
10. `tests/integration/test_diffusion_pipeline.py` - Integration tests

**Config Files (10 files):**

1. `configs/classifier/baseline.json` → `.yaml`
2. `configs/classifier/inceptionv3.json` → `.yaml`
3. `configs/diffusion/default.json` → `.yaml`
4. `tests/fixtures/configs/classifier_minimal.json` → `.yaml`
5. `tests/fixtures/configs/diffusion_minimal.json` → `.yaml`
6. `tests/fixtures/configs/gan_minimal.json` → `.yaml`
7. `tests/fixtures/configs/classifier/valid_minimal.json` → `.yaml`
8. `tests/fixtures/configs/classifier/invalid_missing_model.json` → `.yaml`
9. `tests/fixtures/configs/diffusion/valid_minimal.json` → `.yaml`
10. `tests/fixtures/configs/diffusion/invalid_missing_data.json` → `.yaml`

**Documentation Files (3+ files):**

1. `README.md`
2. `docs/standards/architecture.md`
3. `docs/research/cli-refactoring-plan.md` (optional)
4. `docs/research/cli-migration-guide.md` (if exists)

**Total: 22+ files to modify directly, 10 config files to convert**

### Key Code Patterns to Change

**Pattern 1: Config Loading**

```python
# Before (JSON)
import json
with open(config_path, 'r') as f:
    config = json.load(f)

# After (YAML)
import yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
```

**Pattern 2: Config Saving**

```python
# Before (JSON)
import json
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# After (YAML)
import yaml
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

**Pattern 3: Exception Handling**

```python
# Before (JSON)
try:
    config = load_config(path)
except json.JSONDecodeError:
    # Handle error

# After (YAML)
try:
    config = load_config(path)
except yaml.YAMLError:
    # Handle error
```

**Pattern 4: File Extensions**

```python
# Before (JSON)
config_path = "config.json"
config_save_path = log_dir / "config.json"

# After (YAML)
config_path = "config.yaml"
config_save_path = log_dir / "config.yaml"
```

---

## Appendix B: Detailed Change Plan

### B.1: `src/utils/config.py` Changes

**Current Code (lines 1-35):**

```python
"""Configuration management utilities.

This module provides utilities for loading and merging configuration files.
Priority order: CLI arguments > Config file > Code defaults
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file is not valid JSON
        ValueError: If config_path is empty or None
    """
    if not config_path:
        raise ValueError("Config path cannot be empty or None")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config
```

**New Code:**

```python
"""Configuration management utilities.

This module provides utilities for loading and merging configuration files.
Priority order: CLI arguments > Config file > Code defaults
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the config file is not valid YAML
        ValueError: If config_path is empty or None
    """
    if not config_path:
        raise ValueError("Config path cannot be empty or None")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
```

**Changes for `save_config()` function (lines 120-143):**

```python
def save_config(config: Dict[str, Any], output_path: str, indent: int = 2) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save
        output_path: Path where to save the YAML file
        indent: Number of spaces for indentation (default: 2)

    Raises:
        ValueError: If config is None or output_path is empty
    """
    if config is None:
        raise ValueError("Config cannot be None")

    if not output_path:
        raise ValueError("Output path cannot be empty or None")

    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=indent)
```

---

### B.2: Test File Changes

**Key Test Updates:**

1. **Invalid Format Test:** Change test name and error type

   ```python
   # Before
   def test_load_config_invalid_json(self, tmp_path):
       """Test loading invalid JSON raises JSONDecodeError."""
       config_file = tmp_path / "invalid.json"
       with open(config_file, "w") as f:
           f.write("{ invalid json content }")
       with pytest.raises(json.JSONDecodeError):
           load_config(str(config_file))

   # After
   def test_load_config_invalid_yaml(self, tmp_path):
       """Test loading invalid YAML raises YAMLError."""
       config_file = tmp_path / "invalid.yaml"
       with open(config_file, "w") as f:
           f.write("invalid: yaml: content: [unclosed")
       with pytest.raises(yaml.YAMLError):
           load_config(str(config_file))
   ```

2. **File Creation in Tests:** Update all temporary file creation

   ```python
   # Before
   config_file = tmp_path / "test_config.json"
   with open(config_file, "w") as f:
       json.dump(config_data, f)

   # After
   config_file = tmp_path / "test_config.yaml"
   with open(config_file, "w") as f:
       yaml.dump(config_data, f, default_flow_style=False)
   ```

---

## Appendix C: Sample YAML Configs

### Classifier Config Example

**`configs/classifier/baseline.yaml`:**

```yaml
experiment: classifier
model:
  name: resnet50
  pretrained: true
  num_classes: 2
  freeze_backbone: false
data:
  train_path: data/0.Normal
  val_path: data/1.Abnormal
  batch_size: 32
  num_workers: 4
  image_size: 256
  crop_size: 224
  horizontal_flip: true
  color_jitter: false
  rotation_degrees: 0
  normalize: imagenet
  pin_memory: true
  drop_last: false
  shuffle_train: true
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
  optimizer_kwargs:
    weight_decay: 0.0001
  scheduler: cosine
  scheduler_kwargs:
    T_max: 100
    eta_min: 0.000001
  device: cuda
  mixed_precision: false
  gradient_clip: null
  early_stopping_patience: null
output:
  checkpoint_dir: outputs/checkpoints
  log_dir: outputs/logs
  save_best_only: true
  save_frequency: 10
validation:
  frequency: 1
  metric: accuracy
```

### Minimal Test Config Example

**`tests/fixtures/configs/classifier_minimal.yaml`:**

```yaml
experiment: classifier
model:
  name: resnet50
  pretrained: true
  num_classes: 2
data:
  train_path: tests/fixtures/mock_data/train
  batch_size: 4
  num_workers: 0
  image_size: 64
  crop_size: 32
training:
  epochs: 2
  learning_rate: 0.001
  optimizer: adam
  device: cpu
output:
  checkpoint_dir: /tmp/test_checkpoints
  log_dir: /tmp/test_logs
```

---

**End of Document**
