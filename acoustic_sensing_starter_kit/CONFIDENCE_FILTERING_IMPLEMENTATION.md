# Confidence-Based Prediction Filtering Implementation

## Summary

✅ **IMPLEMENTED**: You can now access prediction confidence and filter predictions based on a confidence threshold!

All classifiers in your pipeline now:
1. Capture prediction probabilities using `predict_proba()`
2. Calculate confidence scores (maximum probability)
3. Filter predictions based on a configurable threshold
4. Report detailed confidence statistics

## What Was Changed

### 1. Configuration (`configs/multi_dataset_config.yml`)

Added confidence filtering configuration under `discrimination_analysis`:

```yaml
discrimination_analysis:
  enabled: true
  confidence_filtering:
    enabled: false  # Set to true to enable
    threshold: 0.7  # Minimum confidence (0.0-1.0)
    mode: "reject"  # or "default"
    default_class: "no_contact"  # Safe default for robotics
```

### 2. Implementation (`discrimination_analysis.py`)

**Added helper method** (lines ~294-382):
```python
def apply_confidence_filtering(
    self,
    y_true,
    y_pred,
    probabilities,
    threshold=0.7,
    mode="reject",
    default_class=None,
):
    """Filter predictions based on confidence threshold."""
    # Calculates confidence = max(probabilities)
    # Filters based on threshold
    # Returns filtered predictions + statistics
```

**Updated prediction flow** (lines ~1087-1168):
- Captures probabilities: `y_proba = clf.predict_proba(X_scaled)`
- Applies filtering: `apply_confidence_filtering(...)`
- Reports statistics for train, test, and validation sets
- Stores confidence stats in results

## How to Use

### Enable Confidence Filtering

Edit `configs/multi_dataset_config.yml`:

```yaml
confidence_filtering:
  enabled: true  # <- Change from false to true
  threshold: 0.7
  mode: "reject"
  default_class: "no_contact"
```

Then run your experiment:
```bash
python3 run_modular_experiments.py configs/multi_dataset_config.yml
```

### Two Filtering Modes

#### 1. **"reject" Mode** (for analysis)
- Removes low-confidence predictions from evaluation
- Shows accuracy on only high-confidence samples
- **Use case**: "How accurate is the model when it's confident?"

Example output:
```
🔍 Applying confidence filtering to VALIDATION set:
  📊 Confidence Filtering (threshold=0.7):
    Kept: 1300/1520 (85.5%)
    Rejected: 220/1520 (14.5%)
    Mean confidence: 0.798
    Median confidence: 0.812
  Validation Accuracy: 0.8750 | F1: 0.8720
```

#### 2. **"default" Mode** (for robotic control)
- Assigns safe default class to low-confidence predictions
- All samples kept, but uncertain predictions → default class
- **Use case**: "When uncertain, assume 'no_contact' for safety"

Example:
```yaml
confidence_filtering:
  enabled: true
  threshold: 0.8  # High confidence required
  mode: "default"
  default_class: "no_contact"  # Safe assumption
```

## Understanding the Output

### Confidence Statistics

Each classifier result now includes:

```python
results_dict[clf_name]["validation_confidence_stats"] = {
    "total_samples": 1520,
    "high_confidence": 1300,      # Samples >= threshold
    "low_confidence": 220,         # Samples < threshold
    "high_confidence_pct": 85.5,
    "low_confidence_pct": 14.5,
    "mean_confidence": 0.798,
    "median_confidence": 0.812,
    "min_confidence": 0.501,
    "max_confidence": 0.999,
}
```

### What Confidence Means

**Confidence** = Maximum predicted probability for a sample

Example:
- Prediction probabilities: `[0.85, 0.15]` → Confidence = 0.85 (high)
- Prediction probabilities: `[0.52, 0.48]` → Confidence = 0.52 (low, essentially random)

With ~50% validation accuracy, many predictions likely have confidence ~0.5 (random guessing).

## Use Cases

### For Robotic Control (Safety-Critical)

```yaml
confidence_filtering:
  enabled: true
  threshold: 0.8        # Require high confidence
  mode: "default"       # Don't reject, use safe default
  default_class: "no_contact"  # Safe assumption
```

**Behavior**:
- High confidence (≥0.8) → Use model prediction
- Low confidence (<0.8) → Assume "no_contact" for safety
- Robot only acts on confident predictions

### For Model Analysis

```yaml
confidence_filtering:
  enabled: true
  threshold: 0.7
  mode: "reject"
  default_class: "no_contact"  # Not used in reject mode
```

**Behavior**:
- See accuracy on confident predictions only
- Identify if model is uncertain on many samples
- Helps diagnose overfitting vs uncertainty

### Finding Optimal Threshold

Try different thresholds to find the trade-off:

| Threshold | Coverage | Accuracy (example) |
|-----------|----------|-------------------|
| 0.5       | 100%     | 51% (baseline)    |
| 0.6       | 95%      | 55%               |
| 0.7       | 85%      | 62%               |
| 0.8       | 70%      | 72%               |
| 0.9       | 45%      | 85%               |

Higher threshold → Better accuracy but fewer predictions accepted.

## Example Experiment

**Test with current dataset** (hand-crafted features, no edge):

```bash
# 1. Edit config to enable confidence filtering
vim configs/multi_dataset_config.yml
# Set: confidence_filtering.enabled: true

# 2. Run experiment
python3 run_modular_experiments.py configs/multi_dataset_config.yml

# 3. Check confidence statistics in output
# Look for:
#   - What % predictions are high/low confidence?
#   - Does Random Forest have better confidence calibration than K-NN?
#   - Do high-confidence predictions have better accuracy?
```

## Analyzing Results

### Key Questions to Answer:

1. **What % of predictions are low-confidence?**
   - If >30%, model is very uncertain → needs improvement

2. **Do high-confidence predictions have better accuracy?**
   - Compare accuracy in "reject" mode vs standard mode
   - If similar, model is overconfident (wrong but confident)

3. **Which classifier has best confidence calibration?**
   - Compare `validation_confidence_stats` across classifiers
   - Look for: high mean_confidence + high accuracy

4. **Optimal threshold?**
   - Try thresholds: 0.5, 0.6, 0.7, 0.8, 0.9
   - Plot: Coverage vs Accuracy (ROC-like curve)

## Expected Benefits

Given your ~50% validation accuracy:

### Scenario 1: Model is truly random
- Many predictions will have confidence ~0.5-0.6
- High % of predictions rejected at threshold=0.7
- Accuracy on high-confidence subset may not improve much
- **Conclusion**: Model needs better features or domain adaptation

### Scenario 2: Model is uncertain on hard samples
- Some predictions confident (0.8+), some uncertain (0.5-0.6)
- Accuracy improves significantly on high-confidence subset
- **Conclusion**: Model learns some patterns but struggles with domain shift
- **Action**: Use default mode for robotics (safe fallback on uncertain samples)

### Scenario 3: Model is overconfident
- High mean confidence (0.8+) but low accuracy (50%)
- **Conclusion**: Model overfitted to training workspace
- **Action**: Need domain adaptation or more diverse training data

## Next Steps

1. **Enable confidence filtering** in config
2. **Run experiment** with hand-crafted features
3. **Analyze confidence statistics**:
   - Check `low_confidence_pct`
   - Compare accuracy in reject mode vs standard
4. **Choose appropriate mode**:
   - For robotics deployment: `mode="default"` with safe fallback
   - For model analysis: `mode="reject"` to see confident-only accuracy
5. **Tune threshold** based on results

## Files Created/Modified

### Modified:
- ✅ `configs/multi_dataset_config.yml` (added confidence_filtering config)
- ✅ `src/acoustic_sensing/experiments/discrimination_analysis.py` (added helper method + integration)

### Created:
- ✅ `test_confidence_filtering.py` (configuration check script)
- ✅ `confidence_filtering_example.py` (detailed examples and documentation)
- ✅ `CONFIDENCE_FILTERING_IMPLEMENTATION.md` (this file)

## Technical Details

### How Probabilities Work

All your classifiers have `predict_proba()` method:

```python
# Example for binary classification (contact vs no_contact)
X_val = [[feature1, feature2, ...], ...]
y_proba = clf.predict_proba(X_val)

# Output shape: (n_samples, n_classes)
# Example: [[0.85, 0.15],   # Sample 1: 85% confident "contact"
#           [0.52, 0.48],   # Sample 2: 52% confident "contact" (uncertain!)
#           [0.23, 0.77]]   # Sample 3: 77% confident "no_contact"

# Get predictions and confidence
y_pred = clf.predict(X_val)        # ["contact", "contact", "no_contact"]
confidence = np.max(y_proba, axis=1)  # [0.85, 0.52, 0.77]
```

### Filtering Logic

```python
# Reject mode: Remove low-confidence samples
mask = confidence >= threshold
y_filtered = y_true[mask]
y_pred_filtered = y_pred[mask]

# Default mode: Assign default class to low-confidence
y_pred_modified = y_pred.copy()
y_pred_modified[confidence < threshold] = "no_contact"
```

## Questions?

Run the test script to check current configuration:
```bash
python3 test_confidence_filtering.py
```

See detailed examples in:
```bash
less confidence_filtering_example.py
```

---

**Status**: ✅ Fully implemented and ready to use!
**Configuration**: Currently disabled (set `enabled: true` to activate)
**Compatible with**: All existing classifiers (Random Forest, K-NN, MLP, GPU-MLP, CNN, ensembles)
