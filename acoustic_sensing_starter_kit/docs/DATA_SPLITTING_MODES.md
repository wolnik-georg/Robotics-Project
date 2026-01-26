# Data Splitting Modes

This pipeline supports three different data splitting modes for different evaluation scenarios.

## Mode 1: Standard Mode (Single Dataset Split)

**Use Case:** Quick experimentation with a single dataset or combined datasets.

**Configuration:**
```yaml
datasets:
  - "dataset1"
  - "dataset2"

# Leave these empty or omit them:
# validation_datasets: []
# hyperparameter_tuning_datasets: []
# final_test_datasets: []
```

**Behavior:**
- All datasets are combined
- 80% train / 20% test split
- Cross-validation for model evaluation
- ⚠️ **Limitation:** No holdout validation, may overestimate generalization

---

## Mode 2: 2-Way Split (Train/Test + Validation)

**Use Case:** Evaluating generalization to a completely unseen workspace or condition.

**Configuration:**
```yaml
datasets:
  - "workspace2_dataset1"
  - "workspace2_dataset2"
  - "workspace3_dataset1"

validation_datasets:
  - "workspace1_dataset1"
  - "workspace1_dataset2"
```

**Behavior:**
- Training datasets: 80% train / 20% test (for quick checks)
- Validation datasets: 100% holdout (for generalization evaluation)
- Best model selected based on **validation accuracy**
- ✅ **Advantage:** Tests cross-workspace generalization

**Metrics Reported:**
- **Train Accuracy:** How well model fits training data (overfitting check)
- **Test Accuracy:** Performance on held-out samples from same workspaces
- **Validation Accuracy:** Performance on completely unseen workspace

---

## Mode 3: 3-Way Split (Train + Tuning + Test)

**Use Case:** Rigorous ML evaluation following best practices - separate datasets for training, hyperparameter tuning, and final unbiased testing.

**Configuration:**
```yaml
# Step 1: Training datasets
datasets:
  - "workspace2_dataset1"
  - "workspace2_dataset2"

# Step 2: Tuning datasets (hyperparameter optimization)
hyperparameter_tuning_datasets:
  - "workspace3_dataset1"
  - "workspace3_dataset2"

# Step 3: Final test datasets (unbiased evaluation)
final_test_datasets:
  - "workspace1_dataset1"
  - "workspace1_dataset2"
```

**Behavior:**
1. **Train** models on `datasets`
2. **Select** best model based on `hyperparameter_tuning_datasets`
3. **Report** final unbiased performance on `final_test_datasets`

**Metrics Reported:**
- **Train Accuracy:** Overfitting check
- **Tuning Accuracy:** Hyperparameter selection criterion
- **Test Accuracy:** Final unbiased performance (most important!)

✅ **Advantages:**
- Prevents "peeking" at test set during model selection
- Follows ML best practices (train/val/test split)
- Most reliable estimate of true generalization
- Supports cross-workspace AND cross-condition evaluation

---

## Example Use Cases

### Scenario 1: Cross-Workspace Generalization (2-way)
```yaml
# Train on Workspace 2 & 3, validate on Workspace 1
datasets:
  - "workspace2_all_data"
  - "workspace3_all_data"
validation_datasets:
  - "workspace1_all_data"
```

### Scenario 2: Cross-Condition Generalization (3-way)
```yaml
# Train on pure contact/no-contact, tune on squares, test on workspace1
datasets:
  - "workspace2_pure_contact"
  - "workspace2_pure_no_contact"
hyperparameter_tuning_datasets:
  - "workspace3_squares_cutout"
final_test_datasets:
  - "workspace1_all_conditions"
```

### Scenario 3: Multi-Dataset Robustness (3-way)
```yaml
# Train on multiple datasets, tune on one workspace, test on another
datasets:
  - "workspace2_squares"
  - "workspace2_pure_contact"
  - "workspace3_pure_contact"
hyperparameter_tuning_datasets:
  - "workspace3_squares"
final_test_datasets:
  - "workspace1_squares"
  - "workspace1_pure_contact"
  - "workspace1_pure_no_contact"
```

---

## Output Files

### 2-Way Split Output
```
results_v*/
└── discrimination_analysis/
    └── validation_results/
        ├── discrimination_summary.json
        ├── classifier_performance.png  (train/test/validation bars)
        └── confusion_matrices_*.png
```

### 3-Way Split Output
```
results_v*/
└── discrimination_analysis/
    └── 3way_split_results/
        ├── discrimination_summary.json
        ├── classifier_performance_3way.png  (train/tuning/test bars)
        └── confusion_matrices_3way_*.png
```

---

## Best Practices

1. **Standard Mode:** Only for quick prototyping
2. **2-Way Split:** When you have one clear holdout dataset (e.g., different workspace)
3. **3-Way Split:** When you need rigorous evaluation and have 3+ distinct datasets

**Golden Rule:** The test set should ONLY be used ONCE at the very end for final reporting!
