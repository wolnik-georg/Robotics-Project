# 3-Way Split Pipeline Fix

## Issue Summary

The 3-way split pipeline was **correctly loading and evaluating** on all three dataset splits, but **NOT using the tuning datasets during model training** for validation and early stopping.

### Problem Details

**Before Fix:**
- ❌ GPU-MLP models used internal 15% validation split from training data
- ❌ Tuning datasets (Workspace 3) were only used for post-training evaluation
- ✅ Training datasets (Workspace 2) were correctly used for training
- ✅ Final test datasets (Workspace 1) were correctly used for final evaluation
- ✅ Best model selection was based on tuning accuracy

**Impact:**
- Models were not optimized on the actual tuning workspace during training
- Early stopping was based on training workspace (Workspace 2) instead of tuning workspace (Workspace 3)
- Reduced cross-workspace generalization

## Correct 3-Way Split Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: TRAINING (on Workspace 2 data)                     │
├─────────────────────────────────────────────────────────────┤
│ • Train models on: Training datasets (Workspace 2)         │
│ • Validate during training: Tuning datasets (Workspace 3)  │ ← FIXED!
│ • Early stopping based on: Tuning dataset loss             │ ← FIXED!
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: MODEL SELECTION                                     │
├─────────────────────────────────────────────────────────────┤
│ • Evaluate all models on: Tuning datasets (Workspace 3)    │
│ • Select best model based on: Tuning accuracy              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: FINAL EVALUATION                                    │
├─────────────────────────────────────────────────────────────┤
│ • Report final unbiased performance on:                     │
│   Test datasets (Workspace 1)                               │
└─────────────────────────────────────────────────────────────┘
```

## Changes Made

### 1. Modified `GPUMLPClassifier.fit()` 

**File:** `src/acoustic_sensing/experiments/gpu_classifiers.py`

**Changes:**
```python
def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray] = None,  # NEW
    y_val: Optional[np.ndarray] = None,  # NEW
) -> "GPUMLPClassifier":
```

**Behavior:**
- If `X_val` and `y_val` are provided → Use them for validation (3-way split mode)
- If not provided → Fall back to internal validation split (backward compatible)
- Early stopping now based on actual tuning workspace performance

### 2. Updated `_run_with_3way_split()`

**File:** `src/acoustic_sensing/experiments/discrimination_analysis.py`

**Changes:**
```python
# For GPU-MLP classifiers, pass tuning data as validation
if hasattr(clf, "fit") and "GPU-MLP" in clf_name:
    self.logger.info(
        f"  Using dedicated tuning data for validation ({len(X_tuning_scaled)} samples)"
    )
    clf.fit(
        X_train_scaled, y_train, 
        X_val=X_tuning_scaled,    # Use tuning data for validation
        y_val=y_tuning            # during training
    )
else:
    clf.fit(X_train_scaled, y_train)
```

**Behavior:**
- GPU-MLP models receive tuning data during training
- Other classifiers (sklearn models) use standard fit
- Backward compatible with existing pipelines

## Expected Improvements

### Before Fix
```
GPU-MLP Training Flow:
  Training: Workspace 2 (85%)    ← Train here
  Validation: Workspace 2 (15%)  ← Validate on same workspace ❌
  
Final Evaluation:
  Tuning: Workspace 3            ← Only evaluated, not trained on
  Test: Workspace 1
```

### After Fix
```
GPU-MLP Training Flow:
  Training: Workspace 2 (100%)   ← Train on all training data
  Validation: Workspace 3        ← Validate on different workspace ✅
  
Final Evaluation:
  Tuning: Workspace 3            ← Used during training AND evaluation
  Test: Workspace 1              ← Final unbiased evaluation
```

### Performance Impact

**Expected gains:**
- **Better generalization:** Early stopping based on different workspace
- **More training data:** Use 100% of training data (not 85%)
- **Proper cross-workspace validation:** Models optimize for workspace transfer
- **Accuracy improvement:** Estimated +1-3% on test set due to better generalization

## Verification

Run tests to verify fix:
```bash
cd /home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit

# Test 1: GPU classifier accepts validation data
python3 -c "
import numpy as np
import sys
sys.path.insert(0, 'src')
from acoustic_sensing.experiments.gpu_classifiers import GPUMLPClassifier

X_train = np.random.randn(100, 50)
y_train = np.random.randint(0, 3, 100)
X_val = np.random.randn(30, 50)
y_val = np.random.randint(0, 3, 30)

clf = GPUMLPClassifier(max_epochs=10, verbose=True)
clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
print('✅ GPU classifier accepts validation data')
"

# Test 2: Run full pipeline
python3 run_modular_experiments.py configs/3way_split_config.yml
```

## Backward Compatibility

✅ **All existing code continues to work:**
- Old configs without tuning datasets: Use internal validation split
- Standard sklearn classifiers: No changes required
- GPU-MLP without validation data: Falls back to internal split

❌ **No breaking changes**

## What's Still NOT Implemented

This fix addresses validation data usage but does NOT implement:

1. **Hyperparameter tuning:** No grid search or Bayesian optimization
   - Models use pre-tuned hyperparameters (from Optuna)
   - Could add GridSearchCV/Optuna in future

2. **Per-model tuning:** All models use same training/tuning split
   - Could implement per-classifier hyperparameter optimization

3. **Ensemble methods:** No model averaging or stacking
   - Single best model is selected

These are potential future enhancements but NOT required for correct 3-way split.

## Testing Checklist

- [x] GPU classifier accepts optional validation data
- [x] GPU classifier falls back to internal split when no validation provided
- [x] Discrimination analysis passes tuning data to GPU models
- [x] No syntax errors in modified files
- [x] Backward compatibility maintained
- [ ] Full pipeline runs successfully with contact physics features
- [ ] Validation accuracy improves with tuning dataset validation

## Related Files

- `src/acoustic_sensing/experiments/gpu_classifiers.py`
- `src/acoustic_sensing/experiments/discrimination_analysis.py`
- `configs/3way_split_config.yml`

## Date

January 22, 2026
