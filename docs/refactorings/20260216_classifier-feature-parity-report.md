# Classifier Feature Parity Report

**Date**: February 16, 2026  
**Author**: GitHub Copilot  
**Purpose**: Verify feature parity between deprecated and refactored classifier implementations

## Executive Summary

The refactored classifier implementation in `src/experiments/classifier/` has successfully implemented the core features from the original code in `src/deprecated/`, with several architectural improvements. However, it lacks important features for production use, particularly around per-class metrics, advanced evaluation metrics, and class imbalance handling.

### Status Overview

✅ **Implemented**: Core model architectures, basic training loop, data loading, checkpointing  
⚠️ **Partially Implemented**: Metrics tracking (basic only)  
❌ **Missing**: Per-class metrics, PR-AUC/ROC-AUC, class imbalance handling, comprehensive visualization

---

## 1. Model Architecture Features

### 1.1 InceptionV3

**Original Implementation**: `src/deprecated/inception_v3.py`  
**Refactored Implementation**: `src/experiments/classifier/models/inceptionv3.py`

#### Feature Comparison

| Feature                      | Original | Refactored | Status          | Notes                                  |
| ---------------------------- | -------- | ---------- | --------------- | -------------------------------------- |
| Pretrained ImageNet weights  | ✅       | ✅         | ✅ Complete     |                                        |
| Local model caching          | ✅       | ✅         | ✅ Complete     | `models/inception_v3.pth`              |
| Feature extraction layers    | ✅       | ✅         | ✅ Complete     | Conv2d_1a through Mixed_7c             |
| Global average pooling       | ✅       | ✅         | ✅ Complete     |                                        |
| Custom classification head   | ✅       | ✅         | ✅ Complete     | 2048 → num_classes                     |
| Layer freezing               | ✅       | ✅         | ✅ Complete     | All layers except FC frozen by default |
| `get_trainable_parameters()` | ✅       | ✅         | ✅ Complete     |                                        |
| Selective layer unfreezing   | ❌       | ✅         | ✅ **Enhanced** | `set_trainable_layers()` method        |
| Feature extraction mode      | ❌       | ✅         | ✅ **Enhanced** | `extract_features()` method            |
| Dropout configuration        | ❌       | ✅         | ✅ **Enhanced** | Default 0.5 dropout                    |
| BaseModel inheritance        | ❌       | ✅         | ✅ **Enhanced** | Checkpoint management                  |

#### Code Reference

**Original**:

```python
class InceptionV3FeatureTrainer(nn.Module):
    def __init__(self, num_classes: int, model_dir: str = "./models/"):
        # Freeze all feature extraction layers
        for param in self.parameters():
            param.requires_grad = False
        # Replace final classification layer
        self.fc = nn.Linear(2048, num_classes)
```

**Refactored**:

```python
class InceptionV3Classifier(BaseModel):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5, ...):
        # More flexible layer freezing
        self.set_trainable_layers(trainable_layers)
```

### 1.2 ResNet

**Original Implementation**: `src/deprecated/resnet.py`  
**Refactored Implementation**: `src/experiments/classifier/models/resnet.py`

#### Feature Comparison

| Feature                     | Original | Refactored | Status          | Notes                          |
| --------------------------- | -------- | ---------- | --------------- | ------------------------------ |
| Multiple variants           | ✅       | ✅         | ✅ Complete     | ResNet50, ResNet101, ResNet152 |
| Pretrained ImageNet weights | ✅       | ✅         | ✅ Complete     |                                |
| Local model caching         | ✅       | ✅         | ✅ Complete     | `models/resnet*.pth`           |
| Feature extraction          | ✅       | ✅         | ✅ Complete     | All layers except FC           |
| Global average pooling      | ✅       | ✅         | ✅ Complete     |                                |
| Classification head         | ❌       | ✅         | ✅ **Enhanced** | With dropout                   |
| Selective layer unfreezing  | ❌       | ✅         | ✅ **Enhanced** | `set_trainable_layers()`       |
| Feature extraction mode     | ❌       | ✅         | ✅ **Enhanced** | `extract_features()` method    |
| Freeze backbone support     | ❌       | ✅         | ✅ **Enhanced** | `freeze_backbone` parameter    |
| BaseModel inheritance       | ❌       | ✅         | ✅ **Enhanced** | Checkpoint management          |

---

## 2. Data Loading and Preprocessing

**Original Implementation**: `src/deprecated/train.py` (functions: `UnderSampledImageFolder`, `_make_dataloader`)  
**Refactored Implementation**: `src/experiments/classifier/dataloader.py`

### Feature Comparison

| Feature                        | Original | Refactored | Status          | Notes                                     |
| ------------------------------ | -------- | ---------- | --------------- | ----------------------------------------- |
| ImageFolder dataset loading    | ✅       | ✅         | ✅ Complete     |                                           |
| Train/val transforms           | ✅       | ✅         | ✅ Complete     |                                           |
| Data augmentation              | ✅       | ✅         | ✅ Complete     | Horizontal flip, rotation, color jitter   |
| ImageNet normalization         | ✅       | ✅         | ✅ Complete     |                                           |
| Configurable batch size        | ✅       | ✅         | ✅ Complete     |                                           |
| Configurable num_workers       | ✅       | ✅         | ✅ Complete     |                                           |
| Pin memory optimization        | ✅       | ✅         | ✅ Complete     |                                           |
| Persistent workers             | ✅       | ✅         | ✅ Complete     |                                           |
| Class-balanced under-sampling  | ✅       | ❌         | ❌ **Missing**  | `UnderSampledImageFolder` not implemented |
| Weighted random sampling       | ✅       | ❌         | ❌ **Missing**  | `WeightedRandomSampler` not supported     |
| Multiple normalization schemes | ❌       | ✅         | ✅ **Enhanced** | ImageNet, CIFAR10, custom, none           |
| BaseDataLoader inheritance     | ❌       | ✅         | ✅ **Enhanced** | Consistent interface                      |

### Analysis

#### What's Implemented

The refactored data loader correctly implements:

- Standard PyTorch DataLoader with ImageFolder
- Comprehensive data augmentation pipeline
- Flexible normalization strategies
- Performance optimizations (pin_memory, persistent_workers)

#### What's Missing

```python
# Original implementation had:
class UnderSampledImageFolder(datasets.ImageFolder):
    """ImageFolder with automatic under-sampling to balance classes"""
    def _undersample_classes(self, min_samples_per_class):
        # Balance classes by limiting samples per class
        ...

# And weighted sampling:
if use_weighted_sampling:
    class_counts = torch.tensor([...])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, ...)
```

These features are critical for handling imbalanced datasets, which is common in medical imaging and anomaly detection tasks.

---

## 3. Training Loop and Optimization

**Original Implementation**: `src/deprecated/train.py` (function: `train`)  
**Refactored Implementation**: `src/experiments/classifier/trainer.py`

### Feature Comparison

| Feature                   | Original | Refactored | Status          | Notes                                    |
| ------------------------- | -------- | ---------- | --------------- | ---------------------------------------- |
| Training epoch loop       | ✅       | ✅         | ✅ Complete     |                                          |
| Validation epoch loop     | ✅       | ✅         | ✅ Complete     |                                          |
| Loss computation          | ✅       | ✅         | ✅ Complete     |                                          |
| Backpropagation           | ✅       | ✅         | ✅ Complete     |                                          |
| Overall accuracy tracking | ✅       | ✅         | ✅ Complete     |                                          |
| Per-class accuracy        | ✅       | ❌         | ❌ **Missing**  | Class 0 acc, Class 1 acc                 |
| Per-class loss            | ✅       | ❌         | ❌ **Missing**  | Class 0 loss, Class 1 loss               |
| Class weights in loss     | ✅       | ❌         | ❌ **Missing**  | `CrossEntropyLoss(weight=class_weights)` |
| Random seed setting       | ✅       | ✅         | ✅ Complete     |                                          |
| Device management         | ✅       | ✅         | ✅ Complete     |                                          |
| Model checkpointing       | ✅       | ✅         | ✅ Complete     |                                          |
| Progress bars             | ❌       | ✅         | ✅ **Enhanced** | tqdm integration                         |
| BaseTrainer inheritance   | ❌       | ✅         | ✅ **Enhanced** | Consistent interface                     |
| Mixed precision training  | ❌       | ✅         | ✅ **Enhanced** | Optional AMP support                     |

### Analysis

#### Per-Class Metrics Gap

**Original implementation tracked**:

```python
# Per-class accuracy
class0_correct = (predicted[labels == 0] == 0).sum().item()
class0_total = (labels == 0).sum().item()
class0_acc = class0_correct / class0_total if class0_total > 0 else 0

# Per-class loss
class0_loss = criterion(outputs[labels == 0], labels[labels == 0])
```

**Refactored implementation** only tracks overall metrics:

```python
# Only overall accuracy and loss
correct = (predicted == targets).sum().item()
accuracy = correct / total
```

This is a **critical gap** for imbalanced datasets where overall accuracy can be misleading.

#### Class Weights Gap

**Original implementation**:

```python
if use_class_weights:
    class_counts = torch.tensor([...])
    class_weights = 1.0 / torch.sqrt(class_counts)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

**Refactored implementation**: No class weight support in loss function.

---

## 4. Metrics and Evaluation

**Original Implementation**: `src/deprecated/train.py` (end of training loop)  
**Refactored Implementation**: `src/experiments/classifier/trainer.py` (method: `evaluate`)

### Feature Comparison

| Feature                        | Original | Refactored | Status          | Notes                        |
| ------------------------------ | -------- | ---------- | --------------- | ---------------------------- |
| Overall loss (train/val)       | ✅       | ✅         | ✅ Complete     |                              |
| Overall accuracy (train/val)   | ✅       | ✅         | ✅ Complete     |                              |
| Per-class accuracy (train/val) | ✅       | ❌         | ❌ **Missing**  | Class 0/1 accuracy           |
| Per-class loss (train/val)     | ✅       | ❌         | ❌ **Missing**  | Class 0/1 loss               |
| PR-AUC (Precision-Recall AUC)  | ✅       | ❌         | ❌ **Missing**  | Binary classification metric |
| ROC-AUC                        | ✅       | ❌         | ❌ **Missing**  | Binary classification metric |
| Precision-recall curves        | ✅       | ❌         | ❌ **Missing**  | Saved to CSV                 |
| ROC curves                     | ✅       | ❌         | ❌ **Missing**  | Saved to CSV                 |
| Confusion matrix               | ❌       | ✅         | ✅ **Enhanced** | Added in refactored version  |

### Analysis

#### Missing Advanced Metrics

**Original implementation calculated**:

```python
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc
)

# PR-AUC
train_pr_auc = average_precision_score(all_labels, all_probs)

# ROC-AUC
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# PR Curve
precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
```

**Refactored implementation**: Only basic metrics (loss, accuracy).

These metrics are **essential for binary classification** tasks, especially in medical imaging where:

- Class imbalance is common
- False positives/negatives have different costs
- Threshold selection is critical

---

## 5. Visualization and Reporting

**Original Implementation**: `src/deprecated/train.py` (end of training loop)  
**Refactored Implementation**: `src/experiments/classifier/logger.py`

### Feature Comparison

| Feature                   | Original | Refactored | Status          | Notes                          |
| ------------------------- | -------- | ---------- | --------------- | ------------------------------ |
| Loss plot (overall)       | ✅       | ❌         | ❌ **Missing**  | Train/val loss over epochs     |
| Loss plot (per-class)     | ✅       | ❌         | ❌ **Missing**  | Class 0/1 loss                 |
| Accuracy plot (overall)   | ✅       | ❌         | ❌ **Missing**  | Train/val accuracy over epochs |
| Accuracy plot (per-class) | ✅       | ❌         | ❌ **Missing**  | Class 0/1 accuracy             |
| PR curve plot             | ✅       | ❌         | ❌ **Missing**  | Precision-Recall curve         |
| ROC curve plot            | ✅       | ❌         | ❌ **Missing**  | ROC curve                      |
| PR-AUC over epochs        | ✅       | ❌         | ❌ **Missing**  | PR-AUC evolution               |
| ROC-AUC over epochs       | ✅       | ❌         | ❌ **Missing**  | ROC-AUC evolution              |
| CSV export (PR curve)     | ✅       | ❌         | ❌ **Missing**  | `pr_curve_train.csv`           |
| CSV export (ROC curve)    | ✅       | ❌         | ❌ **Missing**  | `roc_curve_train.csv`          |
| Confusion matrix          | ❌       | ✅         | ✅ **Enhanced** | Added in logger                |

### Analysis

The original implementation generated comprehensive plots using matplotlib:

```python
# 6 subplots:
# 1. Training PR curve
# 2. Validation PR curve
# 3. Training ROC curve
# 4. Validation ROC curve
# 5. PR-AUC over epochs (train/val)
# 6. ROC-AUC over epochs (train/val)

# Plus 2 separate plots:
# - Loss over epochs (overall + per-class, train/val)
# - Accuracy over epochs (overall + per-class, train/val)
```

The refactored logger only has basic logging infrastructure with confusion matrix support.

---

## 6. Configuration and CLI

**Original Implementation**: `src/deprecated/train.py` (CLI arguments)  
**Refactored Implementation**: `configs/classifier/default.yaml` + CLI

### Feature Comparison

| Parameter             | Original | Refactored | Status         | Notes                 |
| --------------------- | -------- | ---------- | -------------- | --------------------- |
| epochs                | ✅       | ✅         | ✅ Complete    |                       |
| batch_size            | ✅       | ✅         | ✅ Complete    |                       |
| learning_rate         | ✅       | ✅         | ✅ Complete    |                       |
| num_classes           | ✅       | ✅         | ✅ Complete    |                       |
| img_size_original     | ✅       | ✅         | ✅ Complete    | As `image_size`       |
| train_data_path       | ✅       | ✅         | ✅ Complete    | In dataset config     |
| val_data_path         | ✅       | ✅         | ✅ Complete    | In dataset config     |
| out_dir               | ✅       | ✅         | ✅ Complete    | As `output_dir`       |
| num_workers           | ✅       | ✅         | ✅ Complete    |                       |
| seed                  | ✅       | ✅         | ✅ Complete    |                       |
| device                | ✅       | ✅         | ✅ Complete    |                       |
| train_layers          | ✅       | ✅         | ✅ Complete    | As `trainable_layers` |
| model_type            | ✅       | ✅         | ✅ Complete    | As `model.name`       |
| under_sampling        | ✅       | ❌         | ❌ **Missing** | No config option      |
| use_class_weights     | ✅       | ❌         | ❌ **Missing** | No config option      |
| use_weighted_sampling | ✅       | ❌         | ❌ **Missing** | No config option      |

---

## 7. Critical Missing Features Summary

### 7.1 Per-Class Metrics (HIGH PRIORITY)

**Impact**: Critical for imbalanced datasets  
**Affected Files**: `src/experiments/classifier/trainer.py`

```python
# Needed in ClassifierTrainer:
def _compute_per_class_metrics(self, outputs, targets, criterion):
    """Compute accuracy and loss for each class separately"""
    per_class_acc = {}
    per_class_loss = {}
    for class_idx in range(self.num_classes):
        mask = targets == class_idx
        if mask.sum() > 0:
            class_outputs = outputs[mask]
            class_targets = targets[mask]
            # Compute metrics...
    return per_class_acc, per_class_loss
```

### 7.2 Advanced Evaluation Metrics (HIGH PRIORITY)

**Impact**: Essential for binary classification evaluation  
**Affected Files**: `src/experiments/classifier/trainer.py`

```python
# Needed in ClassifierTrainer.evaluate():
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# Collect probabilities
probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]

# Compute metrics
pr_auc = average_precision_score(all_labels, all_probs)
roc_auc = roc_auc_score(all_labels, all_probs)
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
```

### 7.3 Class Imbalance Handling (HIGH PRIORITY)

**Impact**: Critical for datasets with class imbalance  
**Affected Files**:

- `src/experiments/classifier/dataloader.py` (sampling)
- `src/experiments/classifier/trainer.py` (class weights)

```python
# Option 1: Class weights in loss function
class_counts = compute_class_distribution(train_dataset)
class_weights = 1.0 / torch.sqrt(class_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Option 2: Weighted random sampling
sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Option 3: Under-sampling
class UnderSampledImageFolder(datasets.ImageFolder):
    def _undersample_classes(self, min_samples_per_class):
        # Balance by limiting samples
```

### 7.4 Visualization (MEDIUM PRIORITY)

**Impact**: Important for model analysis and debugging  
**Affected Files**: `src/experiments/classifier/logger.py`

```python
# Needed in ClassifierLogger:
def log_pr_curve(self, labels, probs, split='train'):
    """Log precision-recall curve"""
    precision, recall, _ = precision_recall_curve(labels, probs)
    # Plot and save

def log_roc_curve(self, labels, probs, split='train'):
    """Log ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    # Plot and save

def log_metrics_over_time(self, history):
    """Plot metrics evolution over epochs"""
    # Loss, accuracy, PR-AUC, ROC-AUC plots
```

### 7.5 CSV Export (LOW PRIORITY)

**Impact**: Nice to have for external analysis  
**Affected Files**: `src/experiments/classifier/logger.py`

```python
def export_curves_to_csv(self, precision, recall, fpr, tpr, output_dir):
    """Export PR and ROC curves to CSV files"""
    # Save pr_curve_*.csv and roc_curve_*.csv
```

---

## 8. Architectural Improvements in Refactored Code

Despite missing features, the refactored code has significant architectural advantages:

### 8.1 Better Separation of Concerns

- ✅ Model definitions in separate files (`models/`)
- ✅ Data loading as a dedicated class (`dataloader.py`)
- ✅ Training logic in trainer class (`trainer.py`)
- ✅ Logging in dedicated logger (`logger.py`)

**Original**: Everything in one 900+ line file

### 8.2 Inheritance from Base Classes

- ✅ `BaseModel` provides common checkpoint management
- ✅ `BaseDataLoader` provides consistent interface
- ✅ `BaseTrainer` provides training loop structure
- ✅ `BaseLogger` provides logging infrastructure

**Original**: No inheritance, lots of code duplication

### 8.3 Configuration Management

- ✅ YAML-based configuration
- ✅ CLI override support
- ✅ Multiple config files for different experiments
- ✅ Better parameter organization

**Original**: Only CLI arguments, no config files

### 8.4 Enhanced Features

- ✅ Selective layer unfreezing with fine control
- ✅ Feature extraction mode for transfer learning
- ✅ Mixed precision training support
- ✅ Better progress tracking with tqdm
- ✅ Confusion matrix logging

**Original**: Limited flexibility

### 8.5 Testing Infrastructure

- ✅ Unit tests in `tests/experiments/classifier/`
- ✅ Fixtures for test data
- ✅ Integration tests

**Original**: No tests

---

## 9. Feature Implementation Roadmap

### Phase 1: Critical Features (Week 1)

#### Task 1.1: Per-Class Metrics

- [ ] Add per-class accuracy computation in `trainer.py`
- [ ] Add per-class loss computation in `trainer.py`
- [ ] Update logging to track per-class metrics
- [ ] Add per-class metrics to history

#### Task 1.2: Class Weights Support

- [ ] Add `use_class_weights` config option
- [ ] Implement class weight calculation in trainer
- [ ] Pass class weights to loss function
- [ ] Add unit tests for class weight computation

#### Task 1.3: Advanced Metrics

- [ ] Add PR-AUC calculation in `evaluate()`
- [ ] Add ROC-AUC calculation in `evaluate()`
- [ ] Store probabilities during evaluation
- [ ] Add metrics to evaluation results

### Phase 2: Important Features (Week 2)

#### Task 2.1: Sampling Strategies

- [ ] Implement `UnderSampledImageFolder` in `dataloader.py`
- [ ] Add weighted sampling support
- [ ] Add config options: `under_sampling`, `use_weighted_sampling`
- [ ] Update dataloader to use samplers

#### Task 2.2: Curve Generation

- [ ] Add PR curve calculation in trainer
- [ ] Add ROC curve calculation in trainer
- [ ] Store curve data (precision, recall, FPR, TPR, thresholds)
- [ ] Return curves from `evaluate()`

### Phase 3: Visualization (Week 3)

#### Task 3.1: Basic Plots

- [ ] Implement `log_pr_curve()` in logger
- [ ] Implement `log_roc_curve()` in logger
- [ ] Add loss plots (overall + per-class)
- [ ] Add accuracy plots (overall + per-class)

#### Task 3.2: Advanced Plots

- [ ] Add PR-AUC over epochs plot
- [ ] Add ROC-AUC over epochs plot
- [ ] Add comprehensive training report plot
- [ ] Add CSV export for curves

### Phase 4: Documentation and Testing (Week 4)

#### Task 4.1: Documentation

- [ ] Update README with new features
- [ ] Add migration guide from deprecated code
- [ ] Document class imbalance handling
- [ ] Add usage examples

#### Task 4.2: Testing

- [ ] Unit tests for per-class metrics
- [ ] Unit tests for sampling strategies
- [ ] Unit tests for metric calculations
- [ ] Integration tests for full training pipeline

---

## 10. Recommendations

### Immediate Actions (This Sprint)

1. **Implement per-class metrics** - Critical for model evaluation
2. **Add class weights support** - Essential for imbalanced datasets
3. **Implement PR-AUC and ROC-AUC** - Standard metrics for binary classification

### Short-term Actions (Next Sprint)

4. **Add sampling strategies** - Important for flexible class balancing
5. **Implement curve visualization** - Helpful for model analysis
6. **Add comprehensive logging** - Better experiment tracking

### Long-term Actions (Future)

7. **Add multi-class support** - Extend beyond binary classification
8. **Implement early stopping** - Based on validation metrics
9. **Add learning rate scheduling** - Improve training dynamics
10. **Support for multiple metrics** - F1, precision, recall, etc.

---

## 11. Validation Checklist

Use this checklist to verify feature parity:

### Core Functionality

- [x] Model can be instantiated and trained
- [x] Checkpoints can be saved and loaded
- [x] Data augmentation works correctly
- [x] Training loop completes successfully
- [x] Validation is performed after each epoch

### Metrics

- [x] Overall accuracy is computed correctly
- [x] Overall loss is computed correctly
- [ ] Per-class accuracy is tracked
- [ ] Per-class loss is tracked
- [ ] PR-AUC is calculated
- [ ] ROC-AUC is calculated
- [ ] Precision-recall curves are generated
- [ ] ROC curves are generated

### Class Imbalance

- [ ] Class weights can be computed
- [ ] Class weights are used in loss function
- [ ] Weighted sampling is supported
- [ ] Under-sampling is supported

### Visualization

- [ ] Loss plots are generated
- [ ] Accuracy plots are generated
- [ ] PR curves are plotted
- [ ] ROC curves are plotted
- [ ] Per-class metrics are plotted

### Configuration

- [x] All training parameters are configurable
- [x] YAML config files are supported
- [x] CLI overrides work correctly
- [ ] Class imbalance config options exist

---

## 12. Conclusion

### Summary

The refactored classifier implementation represents a **significant architectural improvement** over the original code:

**Strengths**:

- ✅ Cleaner code organization
- ✅ Better maintainability
- ✅ Extensible design
- ✅ Enhanced features (layer control, feature extraction)
- ✅ Testing infrastructure

**Gaps**:

- ❌ Missing per-class metrics
- ❌ Missing advanced evaluation metrics
- ❌ Missing class imbalance handling
- ❌ Missing comprehensive visualization

### Assessment

**For production use**: ⚠️ **Not yet ready** - Critical features missing  
**For development/testing**: ✅ **Ready** - Core functionality works  
**Code quality**: ✅ **Excellent** - Better than original  
**Feature completeness**: ⚠️ **~70%** - Important gaps remain

### Next Steps

1. **Prioritize feature implementation** following the roadmap
2. **Add comprehensive tests** for new features
3. **Update documentation** with migration guide
4. **Validate on real datasets** to ensure parity

### Timeline Estimate

- **Phase 1** (Critical): 1 week
- **Phase 2** (Important): 1 week
- **Phase 3** (Visualization): 1 week
- **Phase 4** (Documentation): 1 week

**Total**: ~4 weeks to full feature parity

---

## Appendix A: File Mapping

### Original → Refactored

| Original File                              | Refactored File                                    | Status      |
| ------------------------------------------ | -------------------------------------------------- | ----------- |
| `src/deprecated/inception_v3.py`           | `src/experiments/classifier/models/inceptionv3.py` | ✅ Complete |
| `src/deprecated/resnet.py`                 | `src/experiments/classifier/models/resnet.py`      | ✅ Complete |
| `src/deprecated/wrn28_cifar10.py`          | `src/experiments/classifier/models/wrn.py`         | ✅ Complete |
| `src/deprecated/train.py` (train function) | `src/experiments/classifier/trainer.py`            | ⚠️ Partial  |
| `src/deprecated/train.py` (data loading)   | `src/experiments/classifier/dataloader.py`         | ⚠️ Partial  |
| `src/deprecated/train.py` (plotting)       | `src/experiments/classifier/logger.py`             | ❌ Missing  |
| `src/deprecated/train.py` (CLI)            | `src/main.py` + `configs/classifier/`              | ✅ Complete |

---

## Appendix B: Code Examples

### B.1 Per-Class Metrics Implementation

```python
# In ClassifierTrainer.train_epoch()
def train_epoch(self, epoch: int) -> Dict[str, float]:
    self.model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class tracking
    per_class_correct = {i: 0 for i in range(self.num_classes)}
    per_class_total = {i: 0 for i in range(self.num_classes)}
    per_class_loss = {i: 0.0 for i in range(self.num_classes)}

    for inputs, targets in self.train_loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Overall metrics
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        total_loss += loss.item()

        # Per-class metrics
        for class_idx in range(self.num_classes):
            mask = targets == class_idx
            if mask.sum() > 0:
                class_correct = predicted[mask].eq(targets[mask]).sum().item()
                per_class_correct[class_idx] += class_correct
                per_class_total[class_idx] += mask.sum().item()

                # Per-class loss
                class_loss = self.criterion(outputs[mask], targets[mask])
                per_class_loss[class_idx] += class_loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Compute metrics
    metrics = {
        'loss': total_loss / len(self.train_loader),
        'accuracy': 100. * correct / total,
    }

    # Add per-class metrics
    for class_idx in range(self.num_classes):
        if per_class_total[class_idx] > 0:
            metrics[f'class_{class_idx}_accuracy'] = (
                100. * per_class_correct[class_idx] / per_class_total[class_idx]
            )
            metrics[f'class_{class_idx}_loss'] = (
                per_class_loss[class_idx] / per_class_total[class_idx]
            )

    return metrics
```

### B.2 Class Weights Implementation

```python
# In setup_experiment_classifier() or ClassifierTrainer.__init__()
def compute_class_weights(self, dataset, mode='inverse_sqrt'):
    """Compute class weights for imbalanced datasets"""
    from collections import Counter

    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        labels = [label for _, label in dataset]

    # Count samples per class
    class_counts = Counter(labels)
    class_counts = torch.tensor([class_counts[i] for i in range(self.num_classes)])

    # Compute weights
    if mode == 'inverse':
        class_weights = 1.0 / class_counts.float()
    elif mode == 'inverse_sqrt':
        class_weights = 1.0 / torch.sqrt(class_counts.float())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize
    class_weights = class_weights / class_weights.sum()

    return class_weights.to(self.device)

# Usage:
if self.config.get('use_class_weights', False):
    class_weights = self.compute_class_weights(train_dataset)
    self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### B.3 PR-AUC and ROC-AUC Implementation

```python
# In ClassifierTrainer.evaluate()
def evaluate(self, loader: DataLoader) -> Dict[str, float]:
    self.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # For PR-AUC and ROC-AUC
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item()

            # Collect probabilities for positive class
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Basic metrics
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': 100. * correct / total,
    }

    # Advanced metrics (binary classification)
    if self.num_classes == 2:
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            precision_recall_curve,
            roc_curve
        )

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # PR-AUC and ROC-AUC
        metrics['pr_auc'] = average_precision_score(all_labels, all_probs)
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs)

        # Curves (for plotting)
        precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
        fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)

        metrics['curves'] = {
            'pr': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds},
            'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
        }

    return metrics
```

### B.4 Weighted Sampling Implementation

```python
# In ClassifierDataLoader.get_loader()
def get_loader(self, split: str = 'train') -> DataLoader:
    dataset = self.get_dataset(split)

    # Check if weighted sampling is enabled
    use_weighted_sampling = self.config.get('use_weighted_sampling', False)

    if split == 'train' and use_weighted_sampling:
        # Compute sample weights
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        else:
            labels = [label for _, label in dataset]

        # Compute class weights
        from collections import Counter
        class_counts = Counter(labels)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

        # Assign weight to each sample
        sample_weights = [class_weights[label] for label in labels]

        # Create sampler
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
    else:
        # Standard dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )
```

---

**End of Report**
