# Multi-Dataset Training & Validation Guide

## 🎯 Overview

The **Multi-Dataset Training** feature allows you to:
1. Train models on **combined data from multiple datasets** (e.g., workspace1 + workspace2)
2. Split the combined data into 80% training / 20% testing
3. Validate the trained models on a **completely separate holdout dataset** (e.g., workspace3)
4. Measure true generalization performance on unseen data

This is different from the standard single-dataset workflow where you only use one dataset with cross-validation.

---

## 🚀 Quick Start

### 1. Enable Multi-Dataset Mode

Edit your config file (e.g., `configs/multi_dataset_config.yml`):

```yaml
multi_dataset_training:
  enabled: true  # ✅ Enable multi-dataset mode
  training_datasets:  # Datasets to combine for training
    - "balanced_collected_data_runs_2026_01_15_workspace1_undersample"
    - "balanced_collected_data_runs_2026_01_14_workspace_2_v1_undersample"
  validation_dataset: "balanced_collected_data_runs_2026_01_14_workspace_3_v1_undersample"
  train_test_split: 0.8  # 80% train, 20% test
  random_seed: 42
  stratify: true  # Balance classes in splits

experiments:
  multi_dataset_training:
    enabled: true  # ✅ Enable the experiment
```

### 2. Run the Experiment

```bash
cd acoustic_sensing_starter_kit

# Run with multi-dataset config
python3 run_modular_experiments.py configs/multi_dataset_config.yml
```

---

## 📊 What Happens During Execution

### Step 1: Data Processing (All Datasets)
```
✅ Processing workspace1_undersample: 150 samples
✅ Processing workspace2_undersample: 120 samples
✅ Processing workspace3_undersample: 100 samples
```

### Step 2: Combining Training Datasets
```
📚 Combining workspace1 + workspace2
✅ Combined: 270 samples
   - contact: 90 samples (33.3%)
   - no_contact: 90 samples (33.3%)
   - edge: 90 samples (33.3%)
```

### Step 3: Train/Test Split
```
✅ Training Set: 216 samples (80%)
✅ Test Set: 54 samples (20%)
```

### Step 4: Loading Validation Dataset
```
✅ Validation Set: 100 samples (NEVER SEEN DURING TRAINING)
   - contact: 33 samples
   - no_contact: 34 samples
   - edge: 33 samples
```

### Step 5: Training Multiple Classifiers
```
🤖 Training: Random Forest
   Test Accuracy: 0.8519 | F1: 0.8501
   Validation Accuracy: 0.7800 | F1: 0.7753

🤖 Training: Gradient Boosting
   Test Accuracy: 0.8333 | F1: 0.8312
   Validation Accuracy: 0.7600 | F1: 0.7589

🤖 Training: SVM (RBF)
   Test Accuracy: 0.8704 | F1: 0.8695
   Validation Accuracy: 0.8100 | F1: 0.8088
```

---

## 📁 Output Files

Results are saved to `modular_analysis_results_multi_dataset/multi_dataset_training/`:

```
multi_dataset_training/
├── multi_dataset_training_results.json          # Complete results
├── performance_comparison.png                   # Test vs Validation accuracy
├── confusion_matrices_best_model.png           # CM for best model
├── generalization_analysis.png                 # Generalization gap analysis
└── experiment_log.txt                          # Execution log
```

### Key Metrics in `multi_dataset_training_results.json`:

```json
{
  "config": {
    "training_datasets": ["workspace1", "workspace2"],
    "validation_dataset": "workspace3",
    "num_train_samples": 216,
    "num_test_samples": 54,
    "num_validation_samples": 100
  },
  "models": {
    "Random Forest": {
      "test_accuracy": 0.8519,
      "test_f1": 0.8501,
      "validation_accuracy": 0.7800,
      "validation_f1": 0.7753,
      "test_confusion_matrix": [[...], [...], [...]],
      "validation_confusion_matrix": [[...], [...], [...]]
    }
  }
}
```

---

## 🔍 Understanding the Results

### 1. **Test Accuracy vs Validation Accuracy**
- **Test Accuracy**: Performance on 20% of combined training data
- **Validation Accuracy**: Performance on completely unseen workspace3 data
- **Generalization Gap**: `Test Acc - Validation Acc`
  - **Negative gap** = Model generalizes BETTER to new data (rare, good!)
  - **Positive gap** = Model overfits to training workspaces
  - **Small gap (<5%)** = Good generalization

### 2. **Why Validation Accuracy Might Be Lower**
- Workspace3 has different:
  - Recording conditions
  - Contact patterns
  - Environmental noise
  - Robot movements
- This tests **true real-world generalization**

### 3. **Confusion Matrices**
- **Test CM**: Shows performance on familiar data (workspace1+2)
- **Validation CM**: Shows performance on unfamiliar data (workspace3)
- Compare to identify which classes generalize poorly

---

## 🛠️ Advanced Configuration

### Use All 3 Datasets for Training (No Validation)
```yaml
multi_dataset_training:
  enabled: false  # Disable multi-dataset mode
  # Use standard single-dataset mode instead
```

### Different Train/Test Split
```yaml
multi_dataset_training:
  train_test_split: 0.7  # 70% train, 30% test
```

### Unbalanced Splits (Not Recommended)
```yaml
multi_dataset_training:
  stratify: false  # Don't balance classes in splits
```

### Different Random Seed
```yaml
multi_dataset_training:
  random_seed: 123  # Different random split
```

---

## 📈 Visualization Outputs

### 1. `performance_comparison.png`
- Side-by-side bar charts
- **Left**: Test vs Validation Accuracy
- **Right**: Test vs Validation F1 Score
- Compare all 6 classifiers

### 2. `confusion_matrices_best_model.png`
- **Left**: Test set confusion matrix
- **Right**: Validation set confusion matrix
- Shows which classes are misclassified

### 3. `generalization_analysis.png`
- Horizontal bar chart
- **Green bars**: Model generalizes well (validation > test)
- **Red bars**: Model overfits (test > validation)
- Shows generalization gap for each model

---

## 🔧 Troubleshooting

### Error: "Dataset not found in processed data"
**Solution**: Make sure all 3 datasets are processed first
```bash
# Check data_processing results
ls modular_analysis_results_multi_dataset/data_processing/
```

### Error: "Class mismatch between datasets"
**Solution**: Ensure all datasets have the same classes (contact, no_contact, edge)
- Check `_map_labels_to_groups()` in `data_processing.py`

### Low Validation Accuracy
**Possible reasons**:
1. Workspace3 has very different conditions → Expected!
2. Not enough training data → Add more datasets or samples
3. Features don't generalize → Try different feature sets
4. Model overfitting → Try simpler models or regularization

---

## 🎓 Best Practices

1. **Balance Your Datasets**
   - Use `balance_dataset.py` first to create balanced versions
   - All 3 datasets should have similar class distributions

2. **Choose Representative Datasets**
   - Training: Diverse recording conditions
   - Validation: Different workspace/conditions to test generalization

3. **Monitor Generalization Gap**
   - <5%: Excellent generalization
   - 5-10%: Good generalization
   - >10%: Consider collecting more diverse training data

4. **Iterate on Feature Engineering**
   - If validation accuracy is low, try:
     - Different feature sets (MINIMAL, OPTIMAL, RESEARCH)
     - Motion artifact removal
     - Frequency band filtering

---

## 🔄 Workflow Comparison

### Standard Single-Dataset (Original)
```
Dataset1 → 80% Train / 20% Test → Model → Evaluate on Test
```

### Multi-Dataset Training (NEW)
```
Dataset1 + Dataset2 → 80% Train / 20% Test → Model
                                               ↓
                                          Validate on Dataset3 (Holdout)
```

---

## 📞 Questions?

See the implementation in:
- `src/acoustic_sensing/experiments/multi_dataset_training.py`
- `src/acoustic_sensing/experiments/data_processing.py` (lines 50-85)
- `configs/multi_dataset_config.yml`

**Key advantage**: This workflow tells you if your model can generalize to **completely new workspaces and conditions**, not just reshuffled data from the same workspace!
