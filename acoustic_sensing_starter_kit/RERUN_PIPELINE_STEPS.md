# Pipeline Rerun Steps - Updated Validation Workspace

**Date**: February 1, 2026  
**Reason**: WS1 incomplete, switching to WS3 for validation  
**Changes**: No edge visualization in reconstructions

---

## ✅ Changes Made

### 1. Config Updated (`configs/multi_dataset_config.yml`)

**NEW Configuration:**
- **Training**: WS1 + WS2 (all surfaces: squares_cutout, pure_contact, pure_no_contact)
- **Validation**: WS3 (all surfaces: squares_cutout, pure_contact, pure_no_contact)
- **Output**: `pattern_a_ws1_ws2_train_ws3_validation/`

**OLD Configuration:**
- Training: WS1 + WS3
- Validation: WS2

### 2. Edge Visualization Disabled

**File**: `src/acoustic_sensing/experiments/surface_reconstruction_simple.py`

```python
# DISABLED: Edge visualization removed per user request
# edge_positions = self._infer_edge_positions(viz_positions)
edge_positions = np.array([]).reshape(0, 2)  # Empty array - no edges
```

Now reconstructions will only show:
- ✅ Contact (green)
- ✅ No-contact (red)
- ❌ Edge (REMOVED - no black squares)

---

## 📋 Steps to Complete

### **STEP 1: Wait for Current Pipeline** ⏳
Your current pipeline is running. Let it finish completely.

```bash
# Check if still running
ps aux | grep python3 | grep multi_dataset_config
```

---

### **STEP 2: Run NEW Pipeline** 🚀

Once the current pipeline finishes, run the updated configuration:

```bash
cd /home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit

# Run the pipeline with updated config
python3 run_modular_experiments.py configs/multi_dataset_config.yml
```

**What This Does:**
1. ✅ Loads WS1 + WS2 for training/testing
2. ✅ Loads WS3 for validation
3. ✅ Trains MLP classifier
4. ✅ Tests on WS1+WS2 test split (~96% expected)
5. ✅ Validates on WS3 (~70% expected)
6. ✅ Saves trained model to `pattern_a_ws1_ws2_train_ws3_validation/`

**Expected Output:**
```
pattern_a_ws1_ws2_train_ws3_validation/
├── dataprocessing/
├── discriminationanalysis/
│   ├── models/
│   │   └── mlp_medium.pkl  ← TRAINED MODEL
│   ├── validation_results/
│   │   └── discrimination_summary.json
│   └── ...
└── ...
```

---

### **STEP 3: Run Reconstructions** 🎨

After the pipeline finishes, you need to run reconstructions on the **validation datasets** (WS3).

#### **Option A: Create New Reconstruction Script** (Recommended)

Create `run_ws3_reconstruction.py`:

```python
#!/usr/bin/env python3
"""
Surface Reconstruction for WS3 Validation Set
Reconstructs all WS3 surfaces using trained model from WS1+WS2
"""
import sys
from pathlib import Path
from acoustic_sensing.experiments.surface_reconstruction_simple import SurfaceReconstructionExperiment

# Configuration
EXPERIMENT_DIR = Path("pattern_a_ws1_ws2_train_ws3_validation")
MODEL_PATH = EXPERIMENT_DIR / "discriminationanalysis" / "models" / "mlp_medium.pkl"
OUTPUT_DIR = Path("pattern_a_ws3_reconstruction")

# Validation datasets (WS3 - all surfaces)
VALIDATION_DATASETS = [
    ("data/balanced_workspace_3_squares_cutout_v1", "VAL_WS3_squares_cutout_v1"),
    ("data/balanced_workspace_3_pure_contact", "VAL_WS3_pure_contact"),
    ("data/balanced_workspace_3_pure_no_contact", "VAL_WS3_pure_no_contact"),
]

# Optional: Add test set reconstructions (WS1 or WS2)
TEST_DATASETS = [
    ("data/balanced_workspace_2_squares_cutout", "TEST_WS2_squares_cutout"),
    ("data/balanced_workspace_2_pure_contact", "TEST_WS2_pure_contact"),
    ("data/balanced_workspace_2_pure_no_contact", "TEST_WS2_pure_no_contact"),
]

def main():
    """Run surface reconstructions for all datasets."""
    
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Run the training pipeline first!")
        sys.exit(1)
    
    print(f"✅ Found trained model: {MODEL_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    
    # Initialize experiment
    experiment = SurfaceReconstructionExperiment(
        base_output_dir=OUTPUT_DIR,
        confidence_threshold=0.8,
        logger=None
    )
    
    # Reconstruct VALIDATION set (WS3 - unseen positions)
    print("=" * 80)
    print("VALIDATION SET RECONSTRUCTIONS (WS3 - Unseen Positions)")
    print("=" * 80)
    
    for data_path, name in VALIDATION_DATASETS:
        print(f"\n🔍 Reconstructing: {name}")
        print(f"   Data: {data_path}")
        
        try:
            experiment.run_reconstruction(
                model_path=MODEL_PATH,
                data_path=Path(data_path),
                dataset_name=name,
                sweep_file="sweep.csv"
            )
            print(f"✅ {name} reconstruction complete!")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Reconstruct TEST set (WS2 - training surfaces)
    print("\n" + "=" * 80)
    print("TEST SET RECONSTRUCTIONS (WS2 - Training Surfaces)")
    print("=" * 80)
    
    for data_path, name in TEST_DATASETS:
        print(f"\n🔍 Reconstructing: {name}")
        print(f"   Data: {data_path}")
        
        try:
            experiment.run_reconstruction(
                model_path=MODEL_PATH,
                data_path=Path(data_path),
                dataset_name=name,
                sweep_file="sweep.csv"
            )
            print(f"✅ {name} reconstruction complete!")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 ALL RECONSTRUCTIONS COMPLETE!")
    print("=" * 80)
    print(f"\n📂 Results saved to: {OUTPUT_DIR}")
    print("\nReconstruction images:")
    print("  - VAL_WS3_squares_cutout_v1/balanced_workspace_3_squares_cutout_v1_comparison.png")
    print("  - VAL_WS3_pure_contact/balanced_workspace_3_pure_contact_comparison.png")
    print("  - VAL_WS3_pure_no_contact/balanced_workspace_3_pure_no_contact_comparison.png")
    print("  - TEST_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png")
    print("  - (etc.)")

if __name__ == "__main__":
    main()
```

Then run:

```bash
python3 run_ws3_reconstruction.py
```

#### **Option B: Manual Reconstruction** (If you prefer)

```bash
cd /home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit

# Enable reconstruction in config
# Edit configs/multi_dataset_config.yml and set:
#   reconstruction:
#     enabled: true

# Then rerun (will skip training since models exist)
python3 run_modular_experiments.py configs/multi_dataset_config.yml
```

---

## 📊 Expected Results

### **After Pipeline (Step 2):**

```
📊 VALIDATION RESULTS (WS3):
  - Test Accuracy (WS1+WS2): ~95-96%
  - Validation Accuracy (WS3): ~70-75%
  - Model generalizes to unseen positions!
```

### **After Reconstruction (Step 3):**

```
pattern_a_ws3_reconstruction/
├── VAL_WS3_squares_cutout_v1/
│   ├── balanced_workspace_3_squares_cutout_v1_comparison.png  ← USE THIS FOR SLIDE 9
│   ├── balanced_workspace_3_squares_cutout_v1_confidence.png
│   └── balanced_workspace_3_squares_cutout_v1_error_map.png
├── VAL_WS3_pure_contact/
│   └── balanced_workspace_3_pure_contact_comparison.png
├── VAL_WS3_pure_no_contact/
│   └── balanced_workspace_3_pure_no_contact_comparison.png
├── TEST_WS2_squares_cutout/
│   └── balanced_workspace_2_squares_cutout_comparison.png  ← USE THIS FOR SLIDE 8
└── ...
```

**No edge visualization** - only green (contact) and red (no-contact)!

---

## 🎯 Presentation Updates Needed

After reconstructions complete, update the presentation:

### **Slide 8 (TEST - High Accuracy):**
Replace placeholder with:
```
TEST_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png
```

### **Slide 9 (VALIDATION - Generalization):**
Replace placeholder with:
```
VAL_WS3_squares_cutout_v1/balanced_workspace_3_squares_cutout_v1_comparison.png
```

Copy to presentation:
```bash
cd pattern_a_ws3_reconstruction

# Copy TEST image
cp TEST_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png \
   ../presentation/presentation_figures/test_reconstruction.png

# Copy VALIDATION image
cp VAL_WS3_squares_cutout_v1/balanced_workspace_3_squares_cutout_v1_comparison.png \
   ../presentation/presentation_figures/val_reconstruction.png
```

Update `presentation/main.tex`:
- Slide 8: Change `PLACEHOLDER_TEST_RECONSTRUCTION.png` → `test_reconstruction.png`
- Slide 9: Change `PLACEHOLDER_VAL_RECONSTRUCTION.png` → `val_reconstruction.png`

---

## ⚠️ Important Notes

1. **Wait for current pipeline to finish** before starting Step 2
2. **Edge visualization is DISABLED** - no black squares in new reconstructions
3. **WS3 is now validation** - shows generalization to unseen positions
4. **Model file location**: `pattern_a_ws1_ws2_train_ws3_validation/discriminationanalysis/models/mlp_medium.pkl`
5. **Reconstruction script** needs to point to this new model

---

## 🐛 Troubleshooting

### Pipeline fails with "Dataset not found"
```bash
# Check if datasets exist
ls data/balanced_workspace_1_squares_cutout/
ls data/balanced_workspace_2_squares_cutout/
ls data/balanced_workspace_3_squares_cutout_v1/
```

### Model not found during reconstruction
```bash
# Verify model was saved
ls pattern_a_ws1_ws2_train_ws3_validation/discriminationanalysis/models/
```

### Reconstruction script fails
```bash
# Check sweep.csv exists in each dataset
ls data/balanced_workspace_3_squares_cutout_v1/sweep.csv
```

---

**Status**: ✅ Config updated, edge visualization disabled, ready to rerun!
