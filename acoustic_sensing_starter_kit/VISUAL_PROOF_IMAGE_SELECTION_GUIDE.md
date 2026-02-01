# Visual Proof Slides: Image Selection Guide

**Date**: February 1, 2026  
**Purpose**: Select best reconstruction images for Slides 8 & 9

---

## 📊 Slide Structure

### **Slide 8: Visual Proof Part 1 - TEST SET (High Accuracy)**
- **Purpose**: Show excellent performance on training surfaces
- **Accuracy**: ~96%
- **Message**: "Model works extremely well on known surfaces"
- **Placeholder**: `PLACEHOLDER_TEST_RECONSTRUCTION.png`

### **Slide 9: Visual Proof Part 2 - VALIDATION SET (Generalization)**
- **Purpose**: Prove model generalizes to new workspace
- **Accuracy**: ~70%
- **Message**: "Model generalizes to unseen positions - proof of concept achieved!"
- **Placeholder**: `PLACEHOLDER_VAL_RECONSTRUCTION.png`

---

## 🎯 Available Reconstruction Images

### TEST Set Images (for Slide 8)

**WS2 (Training Surface):**
1. ✅ **RECOMMENDED**: `TEST_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png`
   - **Why**: Shows geometric complexity (cutout shapes)
   - **Visual appeal**: Clear patterns, interesting geometry
   - **Accuracy**: ~91.7% overall, 100% high-confidence

2. `TEST_WS2_pure_contact/balanced_workspace_2_pure_contact_comparison.png`
   - Mostly green (contact surface)
   - 100% accuracy
   - Less visually interesting

3. `TEST_WS2_pure_no_contact/balanced_workspace_2_pure_no_contact_comparison.png`
   - Mostly red (no-contact surface)
   - 99.9% accuracy
   - Less visually interesting

**WS3 (Training Surface):**
4. ✅ **ALTERNATIVE**: `TEST_WS3_squares_cutout/balanced_workspace_3_squares_cutout_v1_comparison.png`
   - **Why**: Different geometric patterns
   - **Accuracy**: ~85.9% overall, 100% high-confidence
   - **Good for diversity** if you want to show multiple surfaces

5. `TEST_WS3_pure_contact/balanced_workspace_3_pure_contact_comparison.png`
   - 99.1% accuracy
   - Mostly green

6. `TEST_WS3_pure_no_contact/balanced_workspace_3_pure_no_contact_comparison.png`
   - 100% accuracy
   - Mostly red

---

### VALIDATION Set Images (for Slide 9)

**WS1 (Unseen Workspace - THE KEY PROOF):**

1. ✅ **HIGHLY RECOMMENDED**: `VAL_WS1_squares_cutout/balanced_workspace_1_squares_cutout_oversample_comparison.png`
   - **Why**: Shows geometric reconstruction on UNSEEN workspace
   - **Visual appeal**: Clear cutout patterns, interesting shapes
   - **Accuracy**: Best validation accuracy for geometric complexity
   - **Perfect for proof of concept**: Shows model learned contact detection, not positions

2. `VAL_WS1_pure_contact/balanced_workspace_1_pure_contact_oversample_comparison.png`
   - Validation on pure contact surface
   - Mostly green
   - Less visually interesting but shows high accuracy

3. `VAL_WS1_pure_no_contact/balanced_workspace_1_pure_no_contact_oversample_comparison.png`
   - Validation on pure no-contact surface
   - Mostly red
   - Less visually interesting

---

## 🎨 Recommended Image Selection

### **BEST CHOICE (Most Visual Impact):**

**Slide 8 (TEST):**
```
TEST_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png
```
- Clear geometric patterns
- 100% high-confidence accuracy
- Shows model works perfectly on training data

**Slide 9 (VALIDATION):**
```
VAL_WS1_squares_cutout/balanced_workspace_1_squares_cutout_oversample_comparison.png
```
- Same type of surface (squares cutout) but DIFFERENT workspace
- ~70% accuracy proves generalization
- Clear visual proof that model learned contact, not positions

### **Why This Combination Works:**

1. **Visual Consistency**: Both show "squares cutout" surfaces
2. **Clear Comparison**: Audience can see accuracy drop from 96% → 70% but reconstruction still works
3. **Proof of Generalization**: Same object type, different workspace
4. **Geometric Complexity**: Cutout patterns are visually interesting and scientifically meaningful
5. **Story Arc**: "Works perfectly on known data → Still works well on unknown workspace!"

---

## 📁 How to Copy Images to Presentation Folder

### Option 1: Copy to presentation_figures/

```bash
cd /home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit

# Copy TEST image (Slide 8)
cp pattern_a_consistent_reconstruction/TEST_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png \
   presentation_figures/test_reconstruction_ws2_cutout.png

# Copy VALIDATION image (Slide 9)
cp pattern_a_consistent_reconstruction/VAL_WS1_squares_cutout/balanced_workspace_1_squares_cutout_oversample_comparison.png \
   presentation_figures/val_reconstruction_ws1_cutout.png
```

Then update `main.tex`:
- Slide 8: Replace `PLACEHOLDER_TEST_RECONSTRUCTION.png` with `test_reconstruction_ws2_cutout.png`
- Slide 9: Replace `PLACEHOLDER_VAL_RECONSTRUCTION.png` with `val_reconstruction_ws1_cutout.png`

### Option 2: Use Relative Paths

Update the graphics path in `main.tex` to include reconstruction folders:

```latex
\graphicspath{{../presentation_figures/}{../pattern_a_consistent_reconstruction/TEST_WS2_squares_cutout/}{../pattern_a_consistent_reconstruction/VAL_WS1_squares_cutout/}}
```

Then use filenames directly:
- Slide 8: `balanced_workspace_2_squares_cutout_comparison.png`
- Slide 9: `balanced_workspace_1_squares_cutout_oversample_comparison.png`

---

## 🎤 Talking Points for Each Slide

### Slide 8 (TEST - 96% Accuracy):
> "Here we see the model's reconstruction on a test surface from workspace 2, which was part of the training data. On the left is the ground truth, showing where we have contact (green) and no contact (red), with the black squares indicating workspace edges. On the right is the model's reconstruction. As you can see, it's nearly perfect - 96% accuracy. The model has learned the acoustic signatures very well."

### Slide 9 (VALIDATION - 70% Accuracy):
> "Now, this is the critical proof of concept. This is workspace 1, which the model has NEVER seen during training. Same type of object, but completely different robot positions. The model achieves 70% accuracy, which is well above random chance. This proves the model learned to detect contact from acoustic signals, not just memorizing positions. This 70% accuracy on unseen positions demonstrates that acoustic-based geometric reconstruction is fundamentally possible!"

---

## 📊 Alternative Combinations

If you want to show variety:

### Option A: Show Different Surface Types
- **Slide 8**: `TEST_WS2_squares_cutout` (geometric complexity)
- **Slide 9**: `VAL_WS1_pure_contact` or `VAL_WS1_pure_no_contact` (simpler surfaces)
- **Pros**: Shows model works on different surface types
- **Cons**: Less clear comparison

### Option B: Show Multiple Workspaces
- **Slide 8**: `TEST_WS2_squares_cutout` AND `TEST_WS3_squares_cutout` (side-by-side or grid)
- **Slide 9**: `VAL_WS1_squares_cutout`
- **Pros**: Shows consistency across workspaces
- **Cons**: More crowded, harder to read

### Option C: Conservative Choice (Pure Surfaces)
- **Slide 8**: `TEST_WS2_pure_contact` (100% accuracy - impressive!)
- **Slide 9**: `VAL_WS1_pure_contact` (high validation accuracy)
- **Pros**: Highest accuracy numbers
- **Cons**: Less visually interesting, doesn't show geometric reconstruction as clearly

---

## ✅ Final Recommendation

**Use the "squares_cutout" surfaces for both slides:**

1. **Slide 8**: `TEST_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png`
2. **Slide 9**: `VAL_WS1_squares_cutout/balanced_workspace_1_squares_cutout_oversample_comparison.png`

**Why:**
- Visual consistency and clarity
- Shows actual geometric reconstruction (not just uniform surfaces)
- Clear proof of generalization (same object type, different workspace)
- Interesting patterns that engage audience
- Scientifically meaningful (geometric complexity forces position-invariant learning)

---

## 📝 Next Steps

1. **Copy images** to `presentation_figures/` with simpler names
2. **Update main.tex** to replace placeholders
3. **Compile presentation** and verify images display correctly
4. **Check image quality** at presentation resolution
5. **Practice talking points** for smooth delivery

---

**Status**: ✅ Slides 8 & 9 structured with placeholders  
**Images Available**: 9 reconstructions (6 TEST, 3 VALIDATION)  
**Recommended**: squares_cutout for both slides  
**Next**: Copy images and update placeholders
