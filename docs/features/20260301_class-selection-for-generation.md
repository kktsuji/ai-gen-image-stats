# Class Selection for Diffusion Generation

## Overview

Add a `class_selection` option to the diffusion generation config that restricts which
classes are generated. When set, only the specified class indices are used and samples are
distributed evenly across them. When `null` (the default), all classes are generated as
before.

## Config

```yaml
generation:
  sampling:
    class_selection: null # null = all classes; [0, 1] = only classes 0 and 1
```

## Behaviour

- **`null` (default):** samples are balanced across all `num_classes` classes, unchanged
  from previous behaviour.
- **List of class indices:** samples are distributed evenly across the listed classes only.
  Classes not in the list receive zero samples.
- A log line `Class selection: [...]` is emitted at INFO level when the option is set.

### Example

```yaml
generation:
  sampling:
    num_samples: 10
    class_selection: [0, 1] # 5 samples of class 0, 5 samples of class 1
```

With `num_classes: 4` in the model config, classes 2 and 3 are skipped entirely.

## Validation

`validate_config` enforces the following rules on `generation.sampling.class_selection`:

| Condition                                | Error                                       |
| ---------------------------------------- | ------------------------------------------- |
| Empty list `[]`                          | `must be a non-empty list or null`          |
| Contains non-integers or negative values | `must contain non-negative integers`        |
| Contains duplicate indices               | `must not contain duplicate class indices`  |
| Index ≥ `num_classes`                    | `contains indices [...] >= num_classes (N)` |

## Files Changed

| File                                     | Change                                                            |
| ---------------------------------------- | ----------------------------------------------------------------- |
| `src/experiments/diffusion/default.yaml` | Added `class_selection: null` under `generation.sampling`         |
| `src/experiments/diffusion/config.py`    | Added validation for `class_selection` in `validate_config`       |
| `src/main.py`                            | Reads `class_selection`; filters target classes and logs when set |

## Tests Added

### `tests/experiments/diffusion/test_config.py` (`TestModeAwareValidation`)

9 new unit tests covering validation:

- `test_generation_class_selection_default` — default is `null`
- `test_generation_class_selection_null_passes` — `null` accepted
- `test_generation_class_selection_valid_single` — `[0]` accepted
- `test_generation_class_selection_valid_subset` — `[0, 1]` accepted
- `test_generation_class_selection_empty_list_raises` — `[]` → ValueError
- `test_generation_class_selection_non_integer_raises` — `[0, "a"]` → ValueError
- `test_generation_class_selection_negative_raises` — `[-1]` → ValueError
- `test_generation_class_selection_duplicates_raises` — `[0, 0]` → ValueError
- `test_generation_class_selection_out_of_range_raises` — `[99]` with `num_classes=2` → ValueError

### `tests/test_main.py` (`TestDiffusionGenerationMode`)

4 new unit tests covering generation behaviour:

- `test_generation_mode_class_labels_all_classes` — `null` with `num_samples=4`, `num_classes=2` → 2×class0, 2×class1
- `test_generation_mode_class_selection_single_class` — `[1]` → all 4 labels are 1
- `test_generation_mode_class_selection_subset_balanced` — `[0,1]` with `num_samples=10`, `num_classes=4` → 5×class0, 5×class1, 0×class2, 0×class3
- `test_generation_mode_class_selection_logs_info` — `[0]` → stdout contains "Class selection" and "[0]"

`_create_generation_config` helper updated with `class_selection=None` kwarg.
`_create_mock_checkpoint` helper updated with `num_classes=2` kwarg (backward-compatible).
