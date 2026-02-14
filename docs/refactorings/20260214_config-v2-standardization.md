# Configuration V2 Standardization Refactoring Plan

**Date**: 2026-02-14  
**Status**: Planning  
**Goal**: Remove V1 config support and make V2 the standard configuration format

## Executive Summary

This refactoring removes all V1 (legacy) configuration code and makes V2 the standard and only configuration format throughout the codebase. V2 has been in use since February 2026 and provides better organization, eliminates parameter duplication, and has clearer structure.

## Scope

### In Scope

- `src/` - All source code
- `tests/` - All test code
- `README.md` - Documentation

### Out of Scope

- `docs/` - Documentation remains as historical reference

## Investigation Summary

### Files with V1/V2 References

#### Source Code (`src/`)

1. **src/utils/config.py**
   - `migrate_config_v1_to_v2()` function (lines 251-429) - **REMOVE**
   - V2 helper functions with "V2" in names/comments - **KEEP, rename to standard**
   - Comments mentioning "V2" - **UPDATE to standard**

2. **src/experiments/classifier/config.py**
   - `is_v2_config()` function - **REMOVE** (no longer needed)
   - `validate_config_v2()` function - **RENAME to `validate_config()` only**
   - `_validate_config_v1()` function - **REMOVE**
   - `validate_config()` dispatcher function - **REPLACE with V2 validation only**
   - Docstrings mentioning V1/V2 support - **UPDATE**

3. **src/experiments/diffusion/config.py**
   - Comments referring to "V2 format" in multiple locations - **UPDATE**
   - `get_default_config()` docstring mentions V2 - **UPDATE**
   - `validate_config()` docstring mentions V2 - **UPDATE**

4. **src/main.py**
   - Extensive V1/V2 detection logic using `is_v2_config()` - **REMOVE**
   - Conditional branches for V1 vs V2 format - **SIMPLIFY to V2 only**
   - Comments mentioning "V2 config" or "V2 structure" - **UPDATE**
   - Multiple sections with conditional logic based on `is_v2` variable - **SIMPLIFY**

#### Tests (`tests/`)

1. **tests/experiments/classifier/test_config.py**
   - `TestIsV2Config` test class - **REMOVE**
   - `TestValidateConfigV2` test class - **RENAME to `TestValidateConfig`**
   - `get_v2_default_config()` helper function - **RENAME to `get_default_config()`**
   - Uses of `is_v2_config()`, `validate_config_v2()` - **UPDATE**
   - Test names with "v2" suffix - **RENAME**

2. **tests/experiments/diffusion/test_config.py**
   - Comments in test docstrings mentioning "(V2)" - **UPDATE**
   - Multiple test names with "V2" references - **UPDATE**

#### Documentation

1. **README.md**
   - "Configuration Structure (V2)" section heading - **UPDATE to "Configuration Structure"**
   - "Diffusion Model Configuration (V2)" heading - **UPDATE**
   - Comments mentioning "V2 - " - **UPDATE**
   - "Migration from V1" section - **REMOVE**
   - Bulleted list of V2 changes - **REMOVE**

## Detailed Refactoring Plan

### Phase 1: Remove V1 Support Functions

#### Step 1.1: Remove V1 migration function from src/utils/config.py

- **File**: `src/utils/config.py`
- **Action**: Delete entire `migrate_config_v1_to_v2()` function (lines 251-429)
- **Impact**: No other code uses this function directly
- **Verification**: Grep for `migrate_config_v1_to_v2` usage

#### Step 1.2: Update V2 helper function names and comments

- **File**: `src/utils/config.py`
- **Actions**:
  - Update section comment from "# V2 Configuration Helper Functions" to "# Configuration Helper Functions"
  - Update `resolve_output_path()` docstring: Remove "(V2 config)" → "(config)"
  - Update `derive_image_size_from_model()` docstring:
    - Remove "V2" references in description
    - Update "In V2 config" → "In the config"
    - Update parameter description from "(V2 format)" to "(format)"
  - Update `derive_return_labels_from_model()` docstring:
    - Remove "V2" references
    - Update "In V2 config" → "In the config"

### Phase 2: Simplify Classifier Configuration Module

#### Step 2.1: Remove V1 validation function

- **File**: `src/experiments/classifier/config.py`
- **Action**: Delete entire `_validate_config_v1()` function
- **Line count**: ~100 lines
- **Impact**: No longer needed as we only support V2

#### Step 2.2: Remove V2 detection function

- **File**: `src/experiments/classifier/config.py`
- **Action**: Delete `is_v2_config()` function
- **Impact**: Used in main.py and tests - will be updated in subsequent steps

#### Step 2.3: Rename and simplify validation function

- **File**: `src/experiments/classifier/config.py`
- **Actions**:
  - Keep current `validate_config_v2()` as the implementation
  - Replace current `validate_config()` function with just the V2 validation (remove dispatcher logic)
  - Remove import of `warnings` module (no longer needed for V1 deprecation warning)
  - Update docstrings to remove V1/V2 mentions
  - Rename any "V2" references in comments to just standard

#### Step 2.4: Update module docstring

- **File**: `src/experiments/classifier/config.py`
- **Action**: Update module docstring to remove "Supports both V1 (legacy) and V2 configuration formats"

### Phase 3: Update Diffusion Configuration Module

#### Step 3.1: Update docstrings and comments

- **File**: `src/experiments/diffusion/config.py`
- **Actions**:
  - Update module docstring: Remove "V2 Format" from title and V2-specific notes
  - Remove "Key improvements in V2:" section
  - Remove "For migration from V1..." line
  - Update `get_default_config()` docstring: Remove "(V2)" from "Configuration Structure (V2):"
  - Update `validate_config()` docstring: Remove "(V2 format)" references
  - Update helper function docstrings: Remove "(V2)" suffixes
  - Update all internal validation function docstrings

### Phase 4: Simplify Main Entry Point

#### Step 4.1: Remove V1/V2 detection in classifier setup

- **File**: `src/main.py`
- **Function**: `setup_experiment_classifier()`
- **Actions**:
  - Remove `from src.experiments.classifier.config import is_v2_config` import
  - Remove `is_v2 = is_v2_config(config)` line
  - Replace all conditional `if is_v2:` / `else:` blocks with V2 code only
  - Remove V1 code branches
  - Update docstring to remove "Supports both V1 (legacy) and V2 configuration formats"
  - Remove V2-specific comments like "# V2 structure" or "# using V2 config"
  - Simplify variable assignments to assume V2 structure

#### Step 4.2: Remove V1/V2 detection in diffusion setup

- **File**: `src/main.py`
- **Function**: `setup_experiment_diffusion()` (if exists)
- **Actions**: Same as Step 4.1 for diffusion-specific code

#### Step 4.3: Update module docstring

- **File**: `src/main.py`
- **Action**: Remove any V1/V2 compatibility mentions from top-level docstring

### Phase 5: Update Test Suite

#### Step 5.1: Update classifier config tests

- **File**: `tests/experiments/classifier/test_config.py`
- **Actions**:
  - Remove `TestIsV2Config` test class entirely
  - Remove `is_v2_config` from imports
  - Rename `TestValidateConfigV2` → `TestValidateConfig`
  - Rename `get_v2_default_config()` → `get_default_config()` helper function
  - Update all test names: Remove "v2" suffix (e.g., `test_valid_v2_config` → `test_valid_config`)
  - Remove "V2" from test docstrings
  - Update validation function calls from `validate_config_v2()` to `validate_config()`
  - Remove Section comment "# V2 Configuration Tests"
  - Remove any conditional checks for V2 format (e.g., `if is_v2_config(config):`)

#### Step 5.2: Update diffusion config tests

- **File**: `tests/experiments/diffusion/test_config.py`
- **Actions**:
  - Update test docstrings: Remove "(V2)" suffixes
  - Remove comments mentioning V2
  - Ensure all tests assume V2 structure

#### Step 5.3: Search and update other test files

- **Search**: All test files for "v2", "V2", "v1", "V1" references
- **Actions**: Update or remove as appropriate

### Phase 6: Update Documentation

#### Step 6.1: Update README.md

- **File**: `README.md`
- **Actions**:
  - Line ~103: "### Configuration Structure (V2)" → "### Configuration Structure"
  - Line ~134: "### Diffusion Model Configuration (V2)" → "### Diffusion Model Configuration"
  - Line ~169: Comment "# Data configuration (V2 - image_size derived from model)" → "# Data configuration (image_size derived from model)"
  - Lines ~289-300: Remove entire "#### Migration from V1" section and bullet points
  - Update all other "(V2)" or "V2" references to just describe current configuration

### Phase 7: Verification and Testing

#### Step 7.1: Code search verification

- **Action**: Run comprehensive searches to ensure no remaining references:
  ```bash
  grep -r "v1_config" src/ tests/
  grep -r "V1" src/ tests/ README.md
  grep -r "is_v2_config" src/ tests/
  grep -r "validate_config_v2" src/ tests/
  grep -r "_validate_config_v1" src/ tests/
  grep -ri "migrate.*v1.*v2" src/ tests/
  ```
- **Expected**: No matches (except in variable names that happen to contain these strings)

#### Step 7.2: Run test suite

- **Action**: Execute full test suite
  ```bash
  pytest tests/ -v
  ```
- **Expected**: All tests pass

#### Step 7.3: Validate config loading

- **Action**: Test that default configs load without errors
  ```bash
  python -c "from src.experiments.classifier.config import get_default_config; print(get_default_config())"
  python -c "from src.experiments.diffusion.config import get_default_config; print(get_default_config())"
  ```
- **Expected**: Configs load successfully

#### Step 7.4: Manual review

- **Action**: Review key files for any missed references:
  - All files in `src/experiments/*/config.py`
  - `src/main.py`
  - `src/utils/config.py`
  - README.md

## Step-by-Step Todo List

### Phase 1: Remove V1 Support Functions (2 tasks)

- [ ] 1.1: Delete `migrate_config_v1_to_v2()` function from `src/utils/config.py`
- [ ] 1.2: Update V2 helper function names and comments in `src/utils/config.py`

### Phase 2: Simplify Classifier Configuration Module (4 tasks)

- [ ] 2.1: Delete `_validate_config_v1()` function from `src/experiments/classifier/config.py`
- [ ] 2.2: Delete `is_v2_config()` function from `src/experiments/classifier/config.py`
- [ ] 2.3: Rename and simplify validation function in `src/experiments/classifier/config.py`
- [ ] 2.4: Update module docstring in `src/experiments/classifier/config.py`

### Phase 3: Update Diffusion Configuration Module (1 task)

- [ ] 3.1: Update all docstrings and comments in `src/experiments/diffusion/config.py`

### Phase 4: Simplify Main Entry Point (3 tasks)

- [ ] 4.1: Remove V1/V2 detection and simplify classifier setup in `src/main.py`
- [ ] 4.2: Remove V1/V2 detection in diffusion setup in `src/main.py` (if exists)
- [ ] 4.3: Update module docstring in `src/main.py`

### Phase 5: Update Test Suite (3 tasks)

- [ ] 5.1: Update classifier config tests in `tests/experiments/classifier/test_config.py`
- [ ] 5.2: Update diffusion config tests in `tests/experiments/diffusion/test_config.py`
- [ ] 5.3: Search and update other test files for V1/V2 references

### Phase 6: Update Documentation (1 task)

- [ ] 6.1: Update README.md to remove V1/V2 version references

### Phase 7: Verification and Testing (4 tasks)

- [ ] 7.1: Run code search verification for remaining V1/V2 references
- [ ] 7.2: Run full test suite and verify all tests pass
- [ ] 7.3: Validate default config loading works correctly
- [ ] 7.4: Manual review of key files

## Total: 18 Tasks

## Impact Assessment

### Breaking Changes

- **V1 config files**: No longer supported (already deprecated)
- **Migration function**: Removed (users should have migrated by now)

### Non-Breaking Changes

- **V2 behavior**: Unchanged, only naming and documentation cleanup
- **API**: No changes to function signatures for V2 code paths

### Risk Assessment

- **Low Risk**: V1 support was already deprecated with warnings
- **Test Coverage**: Comprehensive tests ensure V2 functionality preserved
- **Rollback**: Simple git revert if issues arise

## Timeline Estimate

- **Phase 1-3**: 2-3 hours (code cleanup and refactoring)
- **Phase 4**: 2-3 hours (main.py simplification requires careful review)
- **Phase 5**: 2-3 hours (test updates)
- **Phase 6**: 1 hour (documentation)
- **Phase 7**: 1-2 hours (verification and testing)

**Total**: 8-12 hours

## Success Criteria

1. ✅ No references to "V1", "v1", "V2", "v2" in version context in src/, tests/, README.md
2. ✅ All tests pass
3. ✅ Default configs load successfully for all experiments
4. ✅ Code is simpler without conditional version branching
5. ✅ Documentation clearly describes current configuration format
6. ✅ No deprecation warnings in code

## Post-Refactoring Cleanup

After completing this refactoring, the following can be considered:

1. Archive migration-related documentation in `docs/refactorings/` to a "legacy" or "archive" subfolder
2. Create a single "Configuration Guide" document that supersedes version-specific guides
3. Consider creating a CHANGELOG.md entry documenting this breaking change

## Notes

- Documentation in `docs/` is out of scope but may reference V1/V2 for historical context
- This refactoring simplifies the codebase significantly by removing ~500+ lines of V1 support code
