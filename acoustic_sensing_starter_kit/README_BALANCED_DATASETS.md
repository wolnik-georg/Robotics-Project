# Fully Balanced Datasets - Complete Pipeline

This directory contains a **ONE-COMMAND** solution to create fully balanced datasets from raw collected data.

## 🎯 What This Does

Takes raw unbalanced datasets → Creates fully balanced rotation-ready datasets in **one command**.

### 3-Level Balancing Applied:
1. **Class Balance**: 33/33/33 (contact/no_contact/edge)
2. **Object Balance**: 33/33/33 (A_cutout/B_empty/C_full)  
3. **Workspace Balance**: 50/50 (for training set combinations)

### Output Structure:
```
data/fully_balanced_datasets/
├─ workspace_1_balanced/          # Individual balanced workspaces
├─ workspace_2_balanced/
├─ workspace_3_balanced/
├─ workspace_4_balanced/          # Holdout (Object D)
├─ rotation1_train/               # WS1+WS3 balanced 50/50
├─ rotation1_val/                 # WS2
├─ rotation2_train/               # WS2+WS3 balanced 50/50
├─ rotation2_val/                 # WS1
├─ rotation3_train/               # WS1+WS2 balanced 50/50
├─ rotation3_val/                 # WS3
├─ balance_summary_report.txt     # Detailed statistics
└─ example_config.yml             # Config template
```

---

## 🚀 Quick Start (Easiest Way)

### **Option 1: Run the Shell Script** (Recommended)

```bash
./run_balance_datasets.sh
```

That's it! Takes ~2-3 minutes, creates everything.

---

### **Option 2: Run Python Script Directly**

```bash
python create_fully_balanced_datasets.py \
  --ws1-cutout "data/collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled" \
  --ws1-contact "data/collected_data_runs_2026_01_15_workspace_1_pure_contact_relabeled" \
  --ws1-no-contact "data/collected_data_runs_2026_01_15_workspace_1_pure_no_contact" \
  --ws2-cutout "data/collected_data_runs_2026_01_15_workspace_2_squares_cutout_relabeled" \
  --ws2-contact "data/collected_data_runs_2026_01_15_workspace_2_pure_contact_relabeled" \
  --ws2-no-contact "data/collected_data_runs_2026_01_15_workspace_2_pure_no_contact" \
  --ws3-cutout "data/collected_data_runs_2025_12_17_v2_workspace_3_squares_cutout_relabeled" \
  --ws3-contact "data/collected_data_runs_2026_01_14_workspace_3_pure_contact_relabeled" \
  --ws3-no-contact "data/collected_data_runs_2026_01_14_workspace_3_pure_no_contact" \
  --ws4-holdout "data/collected_data_runs_2026_01_27_hold_out_dataset_relabeled" \
  --output-dir "data/fully_balanced_datasets"
```

---

## 📊 After Balancing

### 1. Check the Summary Report

```bash
cat data/fully_balanced_datasets/balance_summary_report.txt
```

This shows exact class/object/workspace distributions for all datasets.

### 2. Update Your Config

Use the rotation datasets in `multi_dataset_config.yml`:

```yaml
# Rotation 1: Train WS1+WS3, Validate WS2
datasets:
  - "fully_balanced_datasets/rotation1_train"
validation_datasets:
  - "fully_balanced_datasets/rotation1_val"
```

### 3. Run Experiments

```bash
python run_modular_experiments.py --config configs/multi_dataset_config.yml
```

---

## 🔄 Switching Between Rotations

**Super Easy - Just Edit Config:**

```yaml
# Rotation 1 (default)
datasets: ["fully_balanced_datasets/rotation1_train"]
validation_datasets: ["fully_balanced_datasets/rotation1_val"]

# Change to Rotation 2
# datasets: ["fully_balanced_datasets/rotation2_train"]
# validation_datasets: ["fully_balanced_datasets/rotation2_val"]

# Change to Rotation 3
# datasets: ["fully_balanced_datasets/rotation3_train"]
# validation_datasets: ["fully_balanced_datasets/rotation3_val"]
```

**Time to switch:** 5 seconds (edit config, run experiment)

---

## 📈 Expected Dataset Sizes

| Dataset | Raw Samples | Balanced Samples | Notes |
|---------|-------------|------------------|-------|
| WS1 individual | ~4,429 | ~2,400 | Limited by min class |
| WS2 individual | ~3,500 | ~2,100 | Limited by min class |
| WS3 individual | ~3,200 | ~1,800 | Limited by min class |
| WS4 individual | ~2,500 | ~1,500 | Holdout (Object D) |
| Rotation 1 train | ~6,200 | ~3,600 | WS1+WS3 50/50 balanced |
| Rotation 2 train | ~5,300 | ~3,600 | WS2+WS3 50/50 balanced |
| Rotation 3 train | ~6,500 | ~4,200 | WS1+WS2 50/50 balanced |

**All rotation datasets:**
- ✅ 33/33/33 class balance
- ✅ 33/33/33 object balance
- ✅ 50/50 workspace balance

---

## 🛠️ Advanced Usage

### Create Only Individual Workspaces (Skip Rotations)

```bash
python create_fully_balanced_datasets.py \
  --ws1-cutout ... --ws1-contact ... --ws1-no-contact ... \
  --output-dir "data/balanced_workspaces" \
  --skip-rotations
```

### Use Different Random Seed

```bash
python create_fully_balanced_datasets.py \
  --ws1-cutout ... \
  --seed 12345
```

### Process Only Specific Workspaces

```bash
# Only WS1 and WS2
python create_fully_balanced_datasets.py \
  --ws1-cutout ... --ws1-contact ... --ws1-no-contact ... \
  --ws2-cutout ... --ws2-contact ... --ws2-no-contact ... \
  --output-dir "data/ws1_ws2_balanced"
```

---

## 🔍 Troubleshooting

### "No datasets found for Workspace X"
- Check that dataset paths in `run_balance_datasets.sh` are correct
- Verify `sweep.csv` exists in each dataset directory

### "Warning: Source file not found"
- Usually harmless - means a sweep.csv entry references a missing audio file
- Script will skip missing files and continue

### Output directory already exists
- Script will automatically delete and recreate output directory
- Your raw data is never modified (only reads from source)

---

## 📁 Files in This Directory

- `create_fully_balanced_datasets.py` - Main balancing script
- `run_balance_datasets.sh` - Quick launcher (one command)
- `dataset_paths_config.yml` - Dataset path configuration
- `README_BALANCED_DATASETS.md` - This file

---

## ✅ Validation

After running, verify balance in summary report:

```
ROTATION1_TRAIN
--------------------------------------------------------------
Total Samples: 3,600

Class Balance:
  contact        :  1,200 ( 33.3%)  ✓
  no_contact     :  1,200 ( 33.3%)  ✓
  edge           :  1,200 ( 33.3%)  ✓

Object Balance:
  A_cutout       :  1,200 ( 33.3%)  ✓
  B_empty        :  1,200 ( 33.3%)  ✓
  C_full         :  1,200 ( 33.3%)  ✓

Workspace Balance:
  WS1            :  1,800 ( 50.0%)  ✓
  WS3            :  1,800 ( 50.0%)  ✓
```

All percentages should be 33.3% or 50.0% (±0.1% tolerance due to rounding).

---

## 🎓 Why This Matters

**Before:** Your datasets were 23/54/23 (class imbalance)
- Model achieved 60% by mostly guessing "no_contact"
- Baseline was 57% (majority class), not 33%
- Actual performance: 1.05× over baseline (barely better than random!)

**After:** Datasets are 33/33/33 (truly balanced)
- Model can't exploit class imbalance
- True baseline is 33.3% (random guessing)
- Actual performance will show real generalization capability

---

## 📞 Need Help?

Check the summary report first:
```bash
cat data/fully_balanced_datasets/balance_summary_report.txt
```

This shows exactly what was created and how balanced it is.
