# Confidence Filtering in Reconstruction Visualizations - Critical Explanation

## Your Question
When confidence filtering is used with `mode=reject`, are ALL predictions mapped to the coordinate system in the visualization, or only the filtered ones that pass the confidence threshold?

## The Answer: **ALL PREDICTIONS ARE MAPPED** (This is Important!)

### Current Implementation Behavior

**The reconstruction visualizations show ALL 2,280 spatial positions, regardless of confidence filtering.**

Here's what happens step-by-step:

### 1. Prediction Generation (Lines 240-250)
```python
# Model makes predictions for ALL 2,280 positions
predictions = self.model.predict(features)
probabilities = self.model.predict_proba(features)
confidences = np.max(probabilities, axis=1)
```
**Result**: 2,280 predictions generated

### 2. Confidence Filtering for Metrics ONLY (Lines 304-314)
```python
if confidence_mode == "reject":
    # Only evaluate on high-confidence predictions
    if high_conf_count > 0:
        filtered_labels = labels[high_conf_mask]  # Only 4 samples
        filtered_preds = predictions[high_conf_mask]  # Only 4 samples
        filtered_accuracy = accuracy_score(filtered_labels, filtered_preds)
        # This calculates 75% accuracy on just 4 samples
```
**Result**: Accuracy metric (75%) computed on only 4 positions

### 3. Visualization Uses ALL Predictions (Lines 349-356)
```python
# predictions_for_viz contains ALL 2,280 predictions!
predictions_for_viz = predictions.copy()  # All predictions
if np.any(excluded_mask):
    predictions_for_viz[excluded_mask] = original_labels[excluded_mask]

# Generate visualizations (use predictions_for_viz which keeps excluded classes as-is)
self._create_all_visualizations(
    dataset_name,
    coords,  # ALL 2,280 coordinates
    true_labels_for_viz,  # ALL 2,280 ground truth
    predictions_for_viz,  # ALL 2,280 predictions <-- KEY POINT
    probabilities,
    accuracy,
    excluded_mask=excluded_mask,
)
```
**Result**: Visualization shows all 2,280 positions with their predictions

### 4. Grid Rendering (Lines 650-680)
```python
# Plot each point as a filled rectangle
for i, (coord, label) in enumerate(zip(coords, labels)):
    x, y = coord
    is_excluded = excluded_mask[i]
    
    color = self.class_colors.get(label, "gray")
    rect = Rectangle(
        (x - dx / 2, y - dy / 2),
        dx,
        dy,
        facecolor=color,
        # ... draws ALL rectangles
    )
    ax.add_patch(rect)
```
**Result**: All 2,280 rectangles drawn on the grid

## The Discrepancy

### What You Expected (Logical Interpretation)
- **4 positions** shown in the reconstruction (only confident predictions)
- Rest of the grid empty or marked as "uncertain"
- Visual representation matching the 0.2% coverage

### What Actually Happens (Current Implementation)
- **All 2,280 positions** shown in the reconstruction
- Each position displays the model's raw prediction (regardless of confidence)
- Confidence filtering ONLY affects the accuracy metric calculation
- The visualization shows what the model predicts everywhere, not what it's confident about

## Why This is Misleading

The current visualization creates a **critical misrepresentation**:

1. **Figure shows**: Full spatial coverage with predictions at all 2,280 positions
2. **Caption claims**: "75% accuracy with threshold 0.7"
3. **Reality**: The 75% accuracy only applies to 4 positions (0.2%), not the full visualization

The reader sees a complete reconstruction and thinks "this model achieves 75% accuracy across this surface," when in reality:
- **75% accuracy**: Applies to only 4 positions (not shown which ones)
- **33% accuracy**: Applies to the remaining 2,276 positions (but reader doesn't know which)
- **Visualization**: Indiscriminately shows all predictions as if they're equally valid

## What SHOULD Happen (Two Options)

### Option A: Show Only Confident Predictions (Recommended)
```python
if confidence_mode == "reject":
    # Create mask for low-confidence predictions
    low_conf_mask = (confidences < confidence_threshold) & included_mask
    
    # Mark low-confidence predictions as "uncertain" class for visualization
    predictions_for_viz = predictions.copy()
    predictions_for_viz[low_conf_mask] = "uncertain"  # New class
    
    # Add "uncertain" color (e.g., light gray or white)
    self.class_colors["uncertain"] = "#E0E0E0"
```

**Result**: 
- 4 positions show colored predictions (contact/no-contact/edge)
- 2,276 positions show as gray/white "uncertain" class
- Visualization matches the 0.2% coverage claim

### Option B: Show Confidence via Transparency
```python
# Use confidence as alpha channel
for i, (coord, label, conf) in enumerate(zip(coords, labels, confidences)):
    alpha = 1.0 if conf >= confidence_threshold else 0.2
    rect = Rectangle(
        (x - dx / 2, y - dy / 2),
        dx,
        dy,
        facecolor=color,
        alpha=alpha,  # Transparent for low confidence
    )
```

**Result**: 
- All predictions shown, but low-confidence ones are nearly transparent
- Visual distinction between confident (opaque) and uncertain (faded) predictions

## Impact on Your Report

### Current Figure Caption States:
> "With confidence filtering (threshold ≥0.7), accuracy increases to 75.00%, but only 4 out of 2,280 positions (0.2% coverage) exceed the confidence threshold."

### But the Figure Shows:
- Complete spatial reconstruction with all 2,280 positions colored
- No visual indication of which 4 positions have high confidence
- No representation of the 99.8% that are below threshold

**This is scientifically misleading** because the visualization does not match the quantitative description in the caption.

## Recommendation

You have three options:

1. **Fix the visualization** (Option A or B above) to visually distinguish confident vs uncertain predictions
2. **Update the caption** to clarify that the figure shows all raw predictions, and only 4 positions had confidence ≥0.7 used for the 75% accuracy metric
3. **Remove the confidence filtering** from the holdout reconstruction entirely and only report the raw 33.03% accuracy (most honest approach)

The current state—showing all predictions while claiming the metric applies to a filtered subset—creates ambiguity about what the reader is actually seeing.

## Code Location for Fix

File: `run_surface_reconstruction.py`
- Lines 300-320: Where confidence filtering happens for metrics
- Lines 349-356: Where `predictions_for_viz` is created
- Lines 650-750: Where grid visualization renders

To implement Option A, add after line 314:
```python
if confidence_mode == "reject" and high_conf_count > 0:
    # Mark low-confidence predictions as "uncertain" for visualization
    low_conf_mask = ~high_conf_mask & included_mask
    predictions_for_viz[low_conf_mask] = "uncertain"
```

And update `class_colors` dictionary to include:
```python
self.class_colors["uncertain"] = "#E0E0E0"  # Light gray
```
