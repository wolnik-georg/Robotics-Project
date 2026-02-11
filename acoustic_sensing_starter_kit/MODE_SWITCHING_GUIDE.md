# Easy Mode Switching Guide

## Quick Reference: 3-Class vs Binary Mode

### **Already Completed (3-Class Mode):**
```bash
# These are DONE - don't need to run again
✅ fully_balanced_rotation1_results/  (3-class: contact, no_contact, edge)
✅ fully_balanced_rotation2_results/  (3-class: contact, no_contact, edge)
✅ fully_balanced_rotation3_results/  (3-class: contact, no_contact, edge)
✅ fully_balanced_rotation1_results_spectogram/ (3-class with spectrograms)
```

### **To Run Binary Mode (NEW):**

**Option 1: Run all 3 rotations at once (~3 hours)**
```bash
./run_all_binary_experiments.sh
```

**Option 2: Run individually**
```bash
# Rotation 1 only
python3 run_modular_experiments.py configs/rotation1_binary.yml

# Rotation 2 only
python3 run_modular_experiments.py configs/rotation2_binary.yml

# Rotation 3 only
python3 run_modular_experiments.py configs/rotation3_binary.yml
```

---

## How Mode Switching Works

### **Method 1: Use Pre-Made Configs (EASIEST)**

Just run different config files:

**3-Class Mode:**
```bash
python3 run_modular_experiments.py configs/multi_dataset_config.yml
```

**Binary Mode:**
```bash
python3 run_modular_experiments.py configs/rotation1_binary.yml
```

### **Method 2: Edit Config File**

Open any config file and change ONE line:

**3-Class Mode (include edge):**
```yaml
class_filtering:
  enabled: false  # Include all 3 classes
```

**Binary Mode (exclude edge):**
```yaml
class_filtering:
  enabled: true   # Exclude edge class
  classes_to_exclude_train: ["edge"]
  classes_to_exclude_validation: ["edge"]
```

**That's it!** No code changes needed.

---

## What Each Mode Does

### **3-Class Mode (`enabled: false`):**
- Classes: contact, no_contact, edge
- Random baseline: 33.3%
- Sample counts: Full dataset (2,430 train / 1,338 val for Rotation 1)
- Normalized performance: `(val_acc - 33.3%) / 33.3%`
- Example: 55.7% val → (55.7 - 33.3) / 33.3 = **67% above random = 1.67× over random**

### **Binary Mode (`enabled: true`):**
- Classes: contact, no_contact (edge samples excluded)
- Random baseline: 50.0%
- Sample counts: Reduced (~1,620 train / ~892 val after removing edge samples)
- Normalized performance: `(val_acc - 50.0%) / 50.0%`
- Example: 60% val → (60 - 50) / 50 = **20% above random = 1.20× over random**

---

## Expected Results

### **Hypothesis:**
3-class mode should have **better normalized performance** despite lower raw accuracy.

### **Why?**
- Binary is easier (50% random vs 33.3%)
- But edge samples contain discriminative information
- Including edges forces model to learn more robust features

### **Comparison:**

| Mode | Val Acc | Random | Normalized | Interpretation |
|------|---------|--------|------------|----------------|
| 3-Class (Rot 1) | 55.7% | 33.3% | **1.67× over random** | Better normalized performance |
| Binary (Rot 1) | ??? | 50.0% | ??? | Higher raw accuracy expected, but normalized? |

---

## After Binary Experiments Complete

1. **Extract results:**
```bash
cat fully_balanced_rotation1_binary/discriminationanalysis/validation_results/discrimination_summary.json
cat fully_balanced_rotation2_binary/discriminationanalysis/validation_results/discrimination_summary.json
cat fully_balanced_rotation3_binary/discriminationanalysis/validation_results/discrimination_summary.json
```

2. **Compare normalized performance:**
- If 3-class > binary (normalized): **Keep Section IV.D** in report
- If binary > 3-class (normalized): **Delete Section IV.D** from report

3. **Update tracking document** with binary results

---

## File Structure

```
configs/
  ├── multi_dataset_config.yml          # 3-class (spectrograms) - used earlier
  ├── rotation1_binary.yml              # Binary Rotation 1 ⭐ NEW
  ├── rotation2_binary.yml              # Binary Rotation 2 ⭐ NEW
  ├── rotation3_binary.yml              # Binary Rotation 3 ⭐ NEW
  └── ... (other configs)

Results:
  ├── fully_balanced_rotation1_results/          # ✅ 3-class complete
  ├── fully_balanced_rotation2_results/          # ✅ 3-class complete
  ├── fully_balanced_rotation3_results/          # ✅ 3-class complete
  ├── fully_balanced_rotation1_results_spectogram/  # ✅ 3-class spectrograms complete
  ├── fully_balanced_rotation1_binary/           # ⏳ Will be created
  ├── fully_balanced_rotation2_binary/           # ⏳ Will be created
  └── fully_balanced_rotation3_binary/           # ⏳ Will be created
```

---

## Summary

✅ **No code changes needed** - just change config file  
✅ **Switch anytime** - `enabled: true/false`  
✅ **Ready to run** - `./run_all_binary_experiments.sh`  
✅ **~3 hours total** for all 3 rotations  
✅ **Compare results** to decide on Section IV.D  

🎯 **Your pipeline is already set up for easy mode switching!**
