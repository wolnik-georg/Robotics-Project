# Figure Update Plan: 2-Class → 3-Class Transition

## Current Status

We have successfully completed the 3-class workspace rotation experiments with the following results:

### Rotation 1: Train WS1+3 → Validate WS2
- **Directory**: `test_pipeline_3class_v1`
- **CV Accuracy**: 76.9% ± 0.5%
- **Validation Accuracy**: 84.9%
- **Best Classifier**: Random Forest

### Rotation 2: Train WS2+3 → Validate WS1  
- **Directory**: `test_pipeline_3class_rotation2_ws2ws3_train_ws1_val`
- **CV Accuracy**: 75.1%
- **Validation Accuracy**: 60.4%
- **Best Classifier**: Random Forest

### Rotation 3: Train WS1+2 → Validate WS3
- **Directory**: `test_pipeline_3class_rotation3_ws1ws2_train_ws3_val`
- **CV Accuracy**: 79.2%
- **Validation Accuracy**: 34.9%
- **Best Classifier**: Random Forest

### Object Generalization: Train WS1+2+3 → Validate WS4 (Object D)
- **Directory**: `object_generalization_ws4_holdout_3class`
- **CV Accuracy**: 76.6% ± 0.6%
- **Validation Accuracy**: 50.0% (random chance!)
- **Best Classifier**: Random Forest

---

## Figures Currently Referenced in final_report.tex

### Active Figures (Need 3-Class Versions)

1. **`figure11_feature_dimensions.png`** (Line 174)
   - **Current**: Shows feature architecture diagram
   - **Status**: ✅ **NO UPDATE NEEDED** - Architecture didn't change

2. **`figure6_experimental_setup.png`** (Line 205)
   - **Current**: Shows 2-class workspace rotation experimental design
   - **Status**: ⚠️ **UPDATE NEEDED** - Should show 3 rotations with 3-class labels
   - **Action**: Generate new figure showing all 3 rotations with proper labels

3. **`pattern_a_visual_comparison.png`** (Line 224)
   - **Current**: Shows 2-class geometric reconstruction (Rotation 1, WS2 validation)
   - **Status**: ⚠️ **UPDATE NEEDED** - Should show 3-class reconstruction with edge detection
   - **Action**: Generate 3-class reconstruction visualization for Rotation 1 (WS2 validation)

### Commented Out Figures (Were for 2-Class Binary Comparison)

4. **`figure1_v4_vs_v6_main_comparison.png`** (Line 357, commented)
   - **Was**: V4 vs V6 binary comparison  
   - **Status**: ❌ **REMOVE** - No longer relevant

5. **`figure8_confidence_calibration.png`** (Line 364, commented)
   - **Was**: Confidence filtering analysis for binary
   - **Status**: ❌ **REMOVE** - No longer relevant

---

## New Figures to Generate for 3-Class Report

### Priority 1: Essential Figures (Required for Paper)

#### **Figure 1: Workspace Rotation Performance Comparison**
- **Filename**: `figure1_3class_rotation_comparison.png`
- **Content**: Bar chart showing CV and Validation accuracy for all 3 rotations
- **Status**: ✅ **GENERATED**
- **Location**: `ml_analysis_figures/figure1_3class_rotation_comparison.png`

#### **Figure 2: Experimental Setup Diagram**
- **Filename**: `figure6_experimental_setup_3class.png`
- **Content**: Updated diagram showing 3 workspace rotations with 3-class labels
- **Status**: ⚠️ **NEEDS MANUAL CREATION** (or reuse existing if adequate)
- **Current File**: `ml_analysis_figures/figure6_experimental_setup.png`
- **Action**: Check if current file adequately represents 3-class setup

#### **Figure 3: 3-Class Geometric Reconstruction**  
- **Filename**: `pattern_a_3class_reconstruction.png`
- **Content**: Rotation 1 (WS2) validation showing Contact/No-Contact/Edge predictions
- **Status**: ⚠️ **NEEDS GENERATION** from surface reconstruction pipeline
- **Action**: Run reconstruction pipeline on Rotation 1 validation data

### Priority 2: Supporting Figures (Enhance Understanding)

#### **Figure 4: Confusion Matrix Grid**
- **Filename**: `figure2_3class_confusion_matrix_grid.png`
- **Content**: 3×3 confusion matrices for all 3 rotations
- **Status**: ⚠️ **PARTIALLY AVAILABLE** - Individual matrices exist as PNGs
- **Location**: Each rotation has `confusion_matrix_validation.png`
- **Action**: Create composite figure combining all 3, or reference existing ones

#### **Figure 5: CV vs Validation Gap Analysis**
- **Filename**: `figure3_cv_vs_validation_gap.png`
- **Content**: Line plot showing generalization gap across rotations
- **Status**: ⚠️ **NEEDS GENERATION**
- **Action**: Create figure from discrimination summaries

#### **Figure 6: Class-wise Performance**
- **Filename**: `figure4_classwise_performance.png`
- **Content**: Per-class Precision/Recall/F1 for each rotation
- **Status**: ⚠️ **NEEDS GENERATION**
- **Action**: Extract from classifier performance data

### Priority 3: Object Generalization Figures

#### **Figure 7: Object Generalization Results**
- **Filename**: `figure_object_generalization.png`
- **Content**: Bar chart showing complete failure (50% accuracy)
- **Status**: ⚠️ **NEEDS GENERATION**
- **Location**: Data in `object_generalization_ws4_holdout_3class/`

#### **Figure 8: Position vs Object Comparison**
- **Filename**: `figure_position_vs_object.png`
- **Content**: Side-by-side comparison (60% vs 50%)
- **Status**: ⚠️ **NEEDS GENERATION**

---

## Existing Confusion Matrix Images (Already Generated)

Each rotation experiment already has confusion matrices saved:

### Rotation 1 (WS1+3→WS2):
- `test_pipeline_3class_v1/discriminationanalysis/validation_results/confusion_matrix_validation.png`
- `test_pipeline_3class_v1/discriminationanalysis/validation_results/confusion_matrix_cv_vs_validation.png`

### Rotation 2 (WS2+3→WS1):
- `test_pipeline_3class_rotation2_ws2ws3_train_ws1_val/discriminationanalysis/validation_results/confusion_matrix_validation.png`
- `test_pipeline_3class_rotation2_ws2ws3_train_ws1_val/discriminationanalysis/validation_results/confusion_matrix_cv_vs_validation.png`

### Rotation 3 (WS1+2→WS3):
- `test_pipeline_3class_rotation3_ws1ws2_train_ws3_val/discriminationanalysis/validation_results/confusion_matrix_validation.png`
- `test_pipeline_3class_rotation3_ws1ws2_train_ws3_val/discriminationanalysis/validation_results/confusion_matrix_cv_vs_validation.png`

### Object Generalization (WS1+2+3→WS4):
- `object_generalization_ws4_holdout_3class/discriminationanalysis/validation_results/confusion_matrix_validation.png`
- `object_generalization_ws4_holdout_3class/discriminationanalysis/validation_results/confusion_matrix_cv_vs_validation.png`

**Action**: We can use these directly or create a composite grid figure.

---

## Action Items Summary

### Immediate Actions (Before Continuing with Validation):

1. ✅ **DONE**: Generate Figure 1 (Rotation comparison bar chart)

2. ⚠️ **OPTIONAL**: Update Figure 6 (Experimental setup diagram)
   - Check if current diagram adequately shows 3-class setup
   - If not, create new diagram showing 3 rotations with 3 classes

3. ⚠️ **REQUIRED**: Generate Figure 3 (3-Class Reconstruction)
   - Run surface reconstruction on Rotation 1 validation data
   - Create visualization showing Contact/No-Contact/Edge predictions
   - Replace `pattern_a_visual_comparison.png`

4. ⚠️ **OPTIONAL**: Create confusion matrix composite
   - Combine existing confusion matrices into grid figure
   - Or reference individual figures in paper

5. ⚠️ **OPTIONAL**: Generate supporting figures
   - CV vs Validation gap analysis
   - Class-wise performance breakdown
   - Object generalization comparison

### Figure Files to Update in final_report.tex:

```latex
% Line 174: Keep as-is (feature architecture)
\includegraphics[width=\columnwidth]{../ml_analysis_figures/figure11_feature_dimensions.png}

% Line 205: Update or verify adequacy
\includegraphics[width=\textwidth]{../ml_analysis_figures/figure6_experimental_setup.png}
% Consider: figure6_experimental_setup_3class.png if updated

% Line 224: UPDATE REQUIRED - Replace with 3-class reconstruction
\includegraphics[width=\textwidth]{../pattern_a_summary/pattern_a_visual_comparison.png}
% Replace with: ../pattern_a_3class_reconstruction/pattern_a_3class_visual_comparison.png

% Lines 357, 364: Already commented out, can be removed entirely
```

---

## Data Availability Checklist

- ✅ Rotation 1 discrimination results
- ✅ Rotation 2 discrimination results  
- ✅ Rotation 3 discrimination results
- ✅ Object generalization discrimination results
- ✅ All confusion matrices (as PNG files)
- ⚠️ **MISSING**: 3-class reconstruction predictions for visualization
- ⚠️ **MISSING**: Trained model for Rotation 1 reconstruction (if needed)

---

## Next Steps

**User's Question**: "which figures we need to update that were related to the 2 class setup and generate the corresponding figures from the results of the discrimination analysis"

**Answer**: 

### Figures that MUST be updated:
1. **Geometric Reconstruction** (`pattern_a_visual_comparison.png`) - Needs 3-class version with edge detection
   - **Action**: Run reconstruction pipeline on Rotation 1 validation data

### Figures that are RECOMMENDED to update/add:
2. **Experimental Setup** (`figure6_experimental_setup.png`) - Verify it shows 3-class setup clearly
3. **Confusion Matrix Grid** - Combine existing matrices or reference them
4. **Object Generalization Figure** - Show the 50% failure result

### Figures already generated and ready:
- ✅ `figure1_3class_rotation_comparison.png` - Main rotation comparison

### Decision Point:
Before continuing with validation experiments, we need to:
1. **Generate the 3-class reconstruction visualization** (highest priority)
2. **Decide** if we want comprehensive supporting figures or just use existing confusion matrices
3. **Verify** experimental setup diagram is adequate

Would you like me to:
- A) Generate the 3-class reconstruction figure first?
- B) Create a simplified figure generation script that works with existing data?
- C) Proceed with validation experiments and do figures later?
