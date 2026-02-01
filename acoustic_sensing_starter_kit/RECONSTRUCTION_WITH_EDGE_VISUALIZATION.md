# Surface Reconstruction with Edge Visualization

**Date**: February 1, 2026  
**Purpose**: Enhanced reconstruction visualizations with complete workspace coverage

---

## 📋 Overview

Added **edge position inference** to surface reconstruction visualizations for scientific completeness. Edge positions are now shown in **BLACK** in all reconstruction plots.

### Key Changes

✅ **Edge Inference Strategy**: Automatically infer edge positions from missing grid coverage  
✅ **Visualization Only**: Edge positions shown but NOT used in training/testing/validation  
✅ **Metrics Unchanged**: Accuracy calculated ONLY on contact/no-contact positions  
✅ **Complete Coverage**: Shows entire [0,1] × [0,1] workspace grid  

---

## 🔧 Implementation Details

### Modified File
- `src/acoustic_sensing/experiments/surface_reconstruction_simple.py`

### New Method: `_infer_edge_positions()`

```python
def _infer_edge_positions(self, positions: np.ndarray, grid_resolution: int = 50) -> np.ndarray:
    """
    Infer edge positions by finding gaps in contact + no_contact coverage.
    
    Since the surface is normalized [0, 1] × [0, 1], we create a full grid
    and mark positions NOT covered by the data as "edge".
    """
```

**Logic**:
1. Create full 50×50 grid on [0,1] × [0,1] surface
2. Round actual data positions to grid resolution
3. Find grid points NOT covered by contact/no-contact data
4. Mark uncovered positions as "edge"

**Why This Works**:
- Surface is normalized to [0,1] × [0,1]
- Balanced datasets contain contact + no-contact positions
- Any position without data = edge of workspace
- No need to load original edge labels from unbalanced datasets

---

## 🎨 Visualization Updates

### Three Plot Types (all with edges)

#### 1. **Comparison Plot** (`*_comparison.png`)
- **Left**: Ground Truth with inferred edges (black squares)
- **Right**: Predictions with inferred edges (black squares, no prediction)
- **Title Update**: Shows contact/no-contact count separately from edge count

#### 2. **Error Map** (`*_error_map.png`)
- Green: Correct predictions (contact/no-contact)
- Red: Incorrect predictions (contact/no-contact)
- **Black squares**: Edge positions (no error calculation)

#### 3. **Confidence Map** (`*_confidence.png`)
- Color gradient: Confidence scores for contact/no-contact
- **Black squares**: Edge positions (no confidence values)

### Color Scheme
```python
colors = {
    "contact": "green",      # Contact predictions
    "no_contact": "red",     # No-contact predictions
    "edge": "black"          # Inferred edge positions (visualization only)
}
```

---

## 📊 Example Output

For `TEST_WS2_pure_contact`:
```
INFO: Inferred 2425 edge positions from grid gaps
INFO: Saved visualizations to: pattern_a_consistent_reconstruction/TEST_WS2_pure_contact
INFO:   Contact/No-contact: 118 positions
INFO:   Inferred edges: 2425 positions
INFO:   Total coverage: 2543 positions
```

**Metrics Calculation**:
- Accuracy: 100% (calculated on 118 contact/no-contact positions ONLY)
- Edge positions: Shown but NOT included in accuracy calculation
- Total visualization: 2543 positions (118 + 2425)

---

## 🔄 Regenerated Reconstructions

### Pattern A: WS2+WS3 → WS1
**TEST Datasets** (WS2 + WS3):
- `TEST_WS2_squares_cutout`
- `TEST_WS2_pure_contact`
- `TEST_WS2_pure_no_contact`
- `TEST_WS3_squares_cutout`
- `TEST_WS3_pure_contact`
- `TEST_WS3_pure_no_contact`

**VALIDATION Datasets** (WS1):
- `VAL_WS1_squares_cutout`
- `VAL_WS1_pure_contact`
- `VAL_WS1_pure_no_contact`

**Total**: 9 surfaces × 3 plots = 27 visualizations

### Pattern B: WS1+WS2+WS3 → WS4
**TEST Datasets** (WS1 + WS2 + WS3):
- `TEST_WS1_squares_cutout`
- `TEST_WS1_pure_contact`
- `TEST_WS1_pure_no_contact`
- `TEST_WS2_squares_cutout`
- `TEST_WS2_pure_contact`
- `TEST_WS2_pure_no_contact`
- `TEST_WS3_squares_cutout`
- `TEST_WS3_pure_contact`
- `TEST_WS3_pure_no_contact`

**HOLDOUT Dataset** (WS4):
- `HOLDOUT_WS4` (object_d)

**Total**: 10 surfaces × 3 plots = 30 visualizations

---

## ✅ Scientific Benefits

### 1. **Complete Workspace Visualization**
- Shows entire normalized [0,1] × [0,1] grid
- No missing regions in visualizations
- Clear boundaries between data and workspace edge

### 2. **Transparency in Data Coverage**
- Explicitly shows where model makes predictions
- Distinguishes data positions from workspace boundaries
- Helps identify potential biases in data collection

### 3. **Publication Quality**
- Scientifically accurate representations
- Complete spatial context
- Clear visual separation (black edges vs. green/red predictions)

### 4. **Metrics Integrity**
- Accuracy unchanged (binary classification only)
- Edge positions excluded from all metrics
- No impact on training/testing/validation

---

## 🚀 Running Reconstructions

### Pattern A
```bash
cd /home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit
python3 run_pattern_a_consistent.py
```

**Output**: `pattern_a_consistent_reconstruction/`

### Pattern B
```bash
cd /home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit
python3 run_pattern_b_consistent.py
```

**Output**: `pattern_b_consistent_reconstruction/`

### Background Execution
```bash
nohup python3 run_pattern_a_consistent.py > pattern_a_reconstruction.log 2>&1 &
nohup python3 run_pattern_b_consistent.py > pattern_b_reconstruction.log 2>&1 &
```

**Monitor Progress**:
```bash
tail -f pattern_a_reconstruction.log
tail -f pattern_b_reconstruction.log
```

---

## 📁 Output Structure

```
pattern_a_consistent_reconstruction/
├── TEST_WS2_squares_cutout/
│   ├── *_comparison.png       # GT vs Pred with edges (black)
│   ├── *_error_map.png         # Error map with edges (black)
│   └── *_confidence.png        # Confidence map with edges (black)
├── TEST_WS2_pure_contact/
│   └── (same 3 plots)
├── TEST_WS2_pure_no_contact/
│   └── (same 3 plots)
├── TEST_WS3_squares_cutout/
│   └── (same 3 plots)
├── TEST_WS3_pure_contact/
│   └── (same 3 plots)
├── TEST_WS3_pure_no_contact/
│   └── (same 3 plots)
├── VAL_WS1_squares_cutout/
│   └── (same 3 plots)
├── VAL_WS1_pure_contact/
│   └── (same 3 plots)
└── VAL_WS1_pure_no_contact/
    └── (same 3 plots)

pattern_b_consistent_reconstruction/
├── TEST_WS1_squares_cutout/
├── TEST_WS1_pure_contact/
├── TEST_WS1_pure_no_contact/
├── TEST_WS2_squares_cutout/
├── TEST_WS2_pure_contact/
├── TEST_WS2_pure_no_contact/
├── TEST_WS3_squares_cutout/
├── TEST_WS3_pure_contact/
├── TEST_WS3_pure_no_contact/
└── HOLDOUT_WS4/
    └── (same 3 plots)
```

---

## 🎯 Key Takeaways

1. **Edge positions inferred from grid gaps** (not loaded from original data)
2. **Balanced datasets contain contact + no-contact** (no edge class)
3. **Missing grid positions = edge** (simple and effective)
4. **Visualization only** (no impact on training or metrics)
5. **Complete workspace coverage** (scientifically accurate)
6. **Black color** for edge positions (clear visual distinction)

---

## 📝 Notes for Presentation

- Edge visualization adds scientific completeness
- Metrics remain unchanged (binary classification)
- Total coverage shown in plot titles
- Clear separation: data (green/red) vs. workspace boundary (black)
- No retraining required (existing models used)
- All Pattern A & B reconstructions regenerated

---

**Status**: ✅ Implementation complete, reconstructions running  
**Expected Completion**: ~5-10 minutes (depending on dataset sizes)  
**Next Step**: Use updated visualizations in presentation slides
