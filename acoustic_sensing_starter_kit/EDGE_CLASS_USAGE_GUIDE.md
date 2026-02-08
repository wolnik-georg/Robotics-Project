# Edge Class Inclusion - Usage Guide

## ✅ Changes Applied

The `balance_dataset.py` script has been updated to support both **binary (2-class)** and **3-class** modes with easy switching via command-line flags.

### **Default Behavior: 3-Class Mode**
By default, the script now **includes all three classes** (contact, no_contact, edge).

---

## 🚀 Usage Examples

### **1. 3-Class Mode (Default - Include All Classes)**

```bash
# Basic usage - includes contact, no_contact, AND edge
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_3class_undersample

# With oversampling instead
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_3class_oversample \
    --method oversample

# Explicit flag (same as default)
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_3class \
    --include-all-classes
```

**Output:** Balanced dataset with 3 classes (contact, no_contact, edge)

---

### **2. Binary Mode (Exclude Edge Samples)**

```bash
# Exclude edge class - binary classification
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_binary_undersample \
    --exclude-classes edge

# Short form
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_binary \
    -e edge
```

**Output:** Balanced dataset with 2 classes (contact, no_contact only)

---

### **3. Custom Filtering**

```bash
# Exclude multiple classes (e.g., only edge vs no_contact)
python3 balance_dataset.py \
    --input data/my_dataset \
    --output balanced_edge_vs_nocontact \
    --exclude-classes contact

# Contact vs edge only
python3 balance_dataset.py \
    --input data/my_dataset \
    --output balanced_contact_vs_edge \
    --exclude-classes no_contact
```

---

## 📊 Expected Output

### **3-Class Mode Example:**
```
📊 Original class distribution:
  contact: 575
  no_contact: 1585
  edge: 420

✅ 3-Class mode: Including all classes (contact, no_contact, edge)

🏷️  Found 3 unique class(es): ['contact', 'edge', 'no_contact']

⚖️  Balancing dataset (undersample)...

Class distribution before balancing:
  no_contact: 1585
  contact: 575
  edge: 420

Undersampling to 420 samples per class.

Class distribution after balancing:
  contact: 420
  no_contact: 420
  edge: 420

✅ Balanced dataset saved to: balanced_workspace_1_3class_undersample
   - 1260 audio files in balanced_workspace_1_3class_undersample/data
   - sweep.csv with position info at balanced_workspace_1_3class_undersample/sweep.csv
```

### **Binary Mode Example:**
```
📊 Original class distribution:
  contact: 575
  no_contact: 1585
  edge: 420

🔍 Filtering mode: Excluding classes ['edge']

🔍 CLASS FILTERING:
  Classes to exclude: ['edge']
  Original samples: 2580
  Filtered samples: 2160
  Removed samples: 420
  Breakdown of removed samples:
    - edge: 420

🏷️  Found 2 unique class(es): ['contact', 'no_contact']

⚖️  Balancing dataset (undersample)...

Class distribution before balancing:
  no_contact: 1585
  contact: 575

Undersampling to 575 samples per class.

Class distribution after balancing:
  contact: 575
  no_contact: 575

✅ Balanced dataset saved to: balanced_workspace_1_binary_undersample
   - 1150 audio files in balanced_workspace_1_binary_undersample/data
   - sweep.csv with position info at balanced_workspace_1_binary_undersample/sweep.csv
```

---

## 🔄 Migration from Old Balanced Datasets

### **Issue:** Existing balanced datasets exclude edge samples

All previously created balanced datasets used the old default (`FILTER_CLASSES = True`, `CLASSES_TO_EXCLUDE = ["edge"]`), so they only contain contact and no_contact samples.

### **Solution:** Re-balance datasets with new defaults

To create 3-class datasets, you need to re-run the balancing script on the **original unbalanced data**:

```bash
# Example: Re-balance workspace 1 with all 3 classes
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_3class_undersample

# Example: Re-balance workspace 2 with all 3 classes
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_2 \
    --output balanced_workspace_2_3class_undersample

# Example: Re-balance workspace 3 with all 3 classes  
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_3 \
    --output balanced_workspace_3_3class_undersample
```

---

## ⚙️ Pipeline Configuration

### **Update Config Files for 3-Class Mode**

Create a new config file for 3-class experiments:

```bash
cp configs/multi_dataset_config.yml configs/multi_dataset_3class_config.yml
```

Edit the new config to use 3-class balanced datasets:

```yaml
# configs/multi_dataset_3class_config.yml

datasets:
  # TRAINING: Use 3-class balanced datasets
  - "balanced_workspace_3_3class_undersample"
  - "balanced_workspace_1_3class_undersample"

validation_datasets:
  # VALIDATION: Use 3-class balanced datasets
  - "balanced_workspace_2_3class_undersample"

# No class filtering needed (already balanced with 3 classes)
class_filtering:
  enabled: false
  classes_to_exclude: []  # Include all 3 classes
```

---

## 🧪 Testing the Pipeline

### **Step 1: Create 3-Class Balanced Datasets**

```bash
# Workspace 1 (training)
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_3class_undersample

# Workspace 2 (validation)
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_2 \
    --output balanced_workspace_2_3class_undersample

# Workspace 3 (training)
python3 balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_3 \
    --output balanced_workspace_3_3class_undersample
```

### **Step 2: Run Experiment with 3-Class Data**

```bash
# Run experiment with 3-class configuration
python3 run_modular_experiments.py configs/multi_dataset_3class_config.yml
```

### **Step 3: Compare Binary vs 3-Class Results**

```bash
# Binary mode (existing - 2 classes)
python3 run_modular_experiments.py configs/multi_dataset_config.yml

# 3-class mode (new - 3 classes)
python3 run_modular_experiments.py configs/multi_dataset_3class_config.yml
```

**Expected Differences:**
- **Confusion Matrix:** 2×2 vs 3×3
- **Accuracy:** Likely lower for 3-class (harder problem)
- **F1 Score:** Weighted average across 3 classes
- **Visualizations:** 3 bars instead of 2 in performance plots

---

## 📈 Performance Expectations

### **Binary Mode (2-class: contact vs no_contact)**
- **Classes:** contact, no_contact
- **Current Best:** Random Forest - CV Accuracy **82.07% ± 0.83%**
- **Problem Difficulty:** Moderate
- **Use Case:** Simple contact detection

### **3-Class Mode (contact, no_contact, edge)**
- **Classes:** contact, no_contact, edge
- **Expected Best:** Random Forest - CV Accuracy **~70-75%** (estimate)
- **Problem Difficulty:** Harder (3-way classification)
- **Use Case:** More robust system - can detect uncertain/edge cases
- **Benefit:** Safety - system knows when it's uncertain (edge detection)

### **Why Lower Accuracy is Expected:**
1. **More classes to distinguish** (3 instead of 2)
2. **Edge class may be ambiguous** - shares characteristics with both contact and no_contact
3. **Confusion patterns:** Edge samples might be confused with either contact or no_contact

### **Why 3-Class is Still Valuable:**
1. **Robustness:** System can flag uncertain samples instead of making wrong binary decision
2. **Safety:** For robotics, knowing "I'm not sure" is better than wrong answer
3. **Real-world:** Edge cases exist in practice - better to model them explicitly

---

## 🎯 Quick Reference

| Mode | Command Flag | Output Classes | Use Case |
|------|-------------|----------------|----------|
| **3-Class (Default)** | `--include-all-classes` or no flag | contact, no_contact, edge | Robust detection with uncertainty |
| **Binary** | `--exclude-classes edge` | contact, no_contact | Simple binary classification |
| **Custom** | `--exclude-classes <class1> <class2>` | Remaining classes | Specific experiments |

---

## ✅ Summary

**What Changed:**
- ✅ Default behavior: Include all 3 classes (contact, no_contact, edge)
- ✅ New flag: `--exclude-classes <class_names>` for flexible filtering
- ✅ Backward compatible: `--exclude-classes edge` gives old binary behavior
- ✅ Updated documentation and examples

**Next Steps:**
1. Re-balance datasets with 3-class mode
2. Create new config file pointing to 3-class datasets
3. Run experiments comparing binary vs 3-class performance
4. Analyze confusion matrices to understand edge misclassification patterns
5. Choose mode based on accuracy vs robustness tradeoff
