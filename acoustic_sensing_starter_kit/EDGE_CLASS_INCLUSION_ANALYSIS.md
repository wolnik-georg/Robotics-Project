# Edge Class Inclusion - Code Changes Analysis

## Overview
Currently, the pipeline **excludes edge samples** during the balancing step. To support **3-class classification** (contact, no_contact, edge), we need to make edge samples optional with a flag to easily switch between binary and 3-class modes.

---

## 🔍 Current State: Where Edge Samples Are Excluded

### 1. **Balance Dataset Script** (`balance_dataset.py`)
**Location:** Lines 40-65

**Current Behavior:**
```python
# CLASS FILTERING (applied before balancing)
FILTER_CLASSES = True  # Currently TRUE - excludes edge
CLASSES_TO_EXCLUDE = ["edge"]  # Edge samples are filtered out
```

**What Happens:**
- `filter_classes_df()` (lines 133-170) removes all samples with `label == "edge"`
- This happens BEFORE balancing, so balanced datasets never contain edge samples
- All existing balanced datasets were created without edge samples

**CLI Flag:**
```bash
python balance_dataset.py --exclude-edge  # Currently default behavior
```

---

## ✅ What Already Works (No Changes Needed)

### 1. **Data Processing Pipeline** (`src/acoustic_sensing/experiments/data_processing.py`)
**Line 1144-1156:** Label mapping already supports 3 classes
```python
def _map_labels_to_groups(self, labels):
    """Map raw folder names to grouped classes: 
    surface_* -> contact, no_surface_* -> no_contact, edge_* -> edge."""
    mapped_labels = []
    for label in labels:
        if isinstance(label, str):
            if label.startswith("surface"):
                mapped_labels.append("contact")
            elif label.startswith("no_surface"):
                mapped_labels.append("no_contact")
            elif label.startswith("edge"):
                mapped_labels.append("edge")  # ✅ Already handles edge
```

**Line 1753:** Validation checks for 3 classes
```python
if class_name not in ["contact", "edge", "no_contact"]:
    continue
```

### 2. **Discrimination Analysis** (`src/acoustic_sensing/experiments/discrimination_analysis.py`)
All classifiers are **class-agnostic** and automatically adapt to the number of classes:

**Random Forest, K-NN:** Work with any number of classes out-of-the-box
**MLP (CPU):** Automatically detects number of classes from data
**GPU-MLP:** Line 242 in `gpu_classifiers.py`
```python
num_classes = len(self.classes_)  # Automatically determined from y_train
```

**Confusion Matrix:** Lines 3438-3475
```python
# Get unique labels from actual data
train_labels = sorted(list(set(y_train)))
val_labels = sorted(list(set(y_val)))

cm_val = confusion_matrix(y_val, y_val_pred, labels=val_labels)
# ✅ Automatically handles 2 or 3 classes
```

### 3. **Metrics (Accuracy, F1)**
All use `average="weighted"` which works for any number of classes:
```python
f1_score(y_true, y_pred, average="weighted")  # ✅ Multi-class compatible
```

---

## 🔧 Required Changes

### **CHANGE 1: Balance Dataset Script** - Add Include/Exclude Flag

**File:** `balance_dataset.py`

**Lines 40-65 - Update User Settings:**
```python
# CLASS FILTERING (applied before balancing)
# Set to True to exclude specific classes, False to include all
FILTER_CLASSES = False  # CHANGED: Default to False (include all classes)
CLASSES_TO_EXCLUDE = []  # CHANGED: Empty by default

# EXAMPLES:
# Binary classification (contact vs no_contact only):
#   FILTER_CLASSES = True
#   CLASSES_TO_EXCLUDE = ["edge"]
#
# Full 3-class classification (contact, no_contact, edge):
#   FILTER_CLASSES = False
#   CLASSES_TO_EXCLUDE = []
#
# Custom filtering (e.g., only edge vs no_contact):
#   FILTER_CLASSES = True
#   CLASSES_TO_EXCLUDE = ["contact"]
```

**Lines 380-396 - Update CLI Argument:**
```python
parser.add_argument(
    "--exclude-classes",
    "-e",
    type=str,
    nargs="+",  # CHANGED: Accept multiple classes
    default=[],  # CHANGED: Empty default (include all)
    help="Classes to exclude before balancing (e.g., --exclude-classes edge)",
)
parser.add_argument(
    "--include-all-classes",
    action="store_true",
    default=True,  # CHANGED: Default to including all
    help="Include all classes (contact, no_contact, edge). Overrides --exclude-classes",
)
```

**Lines 420-425 - Update Filtering Logic:**
```python
# Apply class filtering based on arguments
classes_to_exclude = []
if not args.include_all_classes:
    classes_to_exclude = args.exclude_classes

if classes_to_exclude:
    sweep_df = filter_classes_df(sweep_df, classes_to_exclude)
```

**New Usage Examples:**
```bash
# Include ALL classes (3-class mode)
python balance_dataset.py --input data/my_dataset --output balanced_3class

# Exclude edge (binary mode - backward compatible)
python balance_dataset.py --input data/my_dataset --exclude-classes edge --output balanced_binary

# Exclude multiple classes
python balance_dataset.py --input data/my_dataset --exclude-classes edge contact
```

---

### **CHANGE 2: Config File** - Add Class Filtering Options

**File:** `configs/multi_dataset_config.yml`

**Lines 53-65 - Update Class Filtering Section:**
```yaml
# Class Filtering Configuration
class_filtering:
  enabled: false  # Set to true to filter classes
  classes_to_exclude: []  # CHANGED: Empty = include all (3-class mode)
  
  # Options for classes_to_exclude:
  # []              → 3-class: contact, no_contact, edge
  # ["edge"]        → Binary: contact vs no_contact only
  # ["contact"]     → Binary: edge vs no_contact
  # ["no_contact"]  → Binary: contact vs edge
  
  # Notes:
  # 1. Filtering happens BEFORE balancing if done in balance_dataset.py
  # 2. This config is for pipeline-level filtering (if datasets weren't pre-filtered)
  # 3. For clean separation: Filter during balancing, not in pipeline
```

---

### **CHANGE 3: Documentation** - Update README/Comments

**Add to:** `balance_dataset.py` docstring (lines 1-25)
```python
"""
Standalone Script to Balance Collected Data by Undersampling/Oversampling

This script:
1. Loads all WAV files from the collected data directory
2. Reads the sweep.csv to get position information for each audio file
3. Groups labels into classes (contact, no_contact, edge)
4. OPTIONALLY: Filters out specified classes for binary classification
5. Balances classes using undersampling or oversampling
6. Saves balanced WAV files AND a new sweep.csv with position info

CLASS FILTERING OPTIONS:
-----------------------
Binary Classification (contact vs no_contact):
    python balance_dataset.py --exclude-classes edge

3-Class Classification (contact, no_contact, edge):
    python balance_dataset.py --include-all-classes

The default is now 3-class mode. Use --exclude-classes for binary mode.

SWEEP CSV PRESERVATION:
-----------------------
The balanced dataset includes sweep.csv mapping each audio file to its
original position (normalized_x, normalized_y) for surface reconstruction.
"""
```

---

## 📊 Impact Analysis

### **Datasets to Re-Balance**

All existing balanced datasets exclude edge samples. To test 3-class classification, you need to:

1. **Re-run balancing on source datasets** with `--include-all-classes`:
```bash
# Example: Re-balance workspace 1 with all 3 classes
python balance_dataset.py \
    --input data/collected_data_runs_2026_01_27_workspace_1 \
    --output balanced_workspace_1_3class_undersample \
    --method undersample \
    --include-all-classes
```

2. **Update config files** to point to new 3-class datasets:
```yaml
datasets:
  - "balanced_workspace_1_3class_undersample"
  - "balanced_workspace_2_3class_undersample"
  - "balanced_workspace_3_3class_undersample"
```

### **Expected Changes in Pipeline**

| Component | Binary (2-class) | 3-Class | Code Changes |
|-----------|------------------|---------|--------------|
| **Data Processing** | ✅ Works | ✅ Works | None - already supports 3 classes |
| **Feature Extraction** | ✅ Works | ✅ Works | None - class-agnostic |
| **Classifiers** | ✅ Works | ✅ Works | None - auto-detects num_classes |
| **Cross-Validation** | ✅ Works | ✅ Works | None - stratification handles 3 classes |
| **Metrics (Acc, F1)** | ✅ Works | ✅ Works | None - weighted averaging |
| **Confusion Matrix** | 2×2 matrix | 3×3 matrix | None - auto-sizes to labels |
| **Visualizations** | 2 bars/lines | 3 bars/lines | None - auto-adapts |

### **Performance Expectations**

**Binary Mode (current):**
- Classes: `contact`, `no_contact`
- Best CV Accuracy: **82.07% ± 0.83%** (Random Forest)
- Validation Accuracy: **57.58%** (cross-workspace)

**3-Class Mode (expected):**
- Classes: `contact`, `no_contact`, `edge`
- Expected CV Accuracy: **Lower** (harder problem, 3 classes)
- Confusion: Edge samples may be confused with both contact and no_contact
- Benefit: More granular classification for real-world deployment

---

## 🚀 Implementation Steps

### **Step 1: Update balance_dataset.py**
1. Change `FILTER_CLASSES = False` (line 44)
2. Change `CLASSES_TO_EXCLUDE = []` (line 47)
3. Update CLI arguments (lines 380-396)
4. Update filtering logic (lines 420-425)
5. Update docstring (lines 1-25)

### **Step 2: Re-Balance Datasets**
```bash
# Workspace 1 - 3-class mode
python balance_dataset.py \
    --input data/workspace_1_source \
    --output balanced_workspace_1_3class_undersample \
    --method undersample

# Workspace 2 - 3-class mode
python balance_dataset.py \
    --input data/workspace_2_source \
    --output balanced_workspace_2_3class_undersample \
    --method undersample

# Workspace 3 - 3-class mode (repeat for v1, v2, etc.)
python balance_dataset.py \
    --input data/workspace_3_source_v1 \
    --output balanced_workspace_3_3class_v1_undersample \
    --method undersample
```

### **Step 3: Create New Config**
```bash
cp configs/multi_dataset_config.yml configs/multi_dataset_3class_config.yml
# Edit to use 3-class balanced datasets
```

### **Step 4: Test Pipeline**
```bash
# Run experiment with 3-class data
python run_modular_experiments.py configs/multi_dataset_3class_config.yml

# Compare results with binary mode
python run_modular_experiments.py configs/multi_dataset_config.yml
```

### **Step 5: Analyze Results**
- Compare binary vs 3-class accuracy
- Check confusion matrix to see edge misclassification patterns
- Evaluate if edge detection improves system safety/robustness

---

## 🎯 Quick Start Commands

**For Binary Mode (exclude edge):**
```bash
python balance_dataset.py --input data/my_data --exclude-classes edge --output balanced_binary
```

**For 3-Class Mode (include all):**
```bash
python balance_dataset.py --input data/my_data --include-all-classes --output balanced_3class
```

**Switch Between Modes:**
```yaml
# In config file - just point to different balanced datasets
datasets:
  # Binary mode:
  # - "balanced_workspace_1_binary_undersample"
  
  # 3-class mode:
  - "balanced_workspace_1_3class_undersample"
```

---

## 📝 Summary

**Code Changes Required:**
1. ✅ `balance_dataset.py`: Update defaults, add CLI flags, update docstring
2. ✅ `configs/*.yml`: Document class filtering options
3. ✅ No changes to pipeline code - already supports 3 classes!

**Key Insight:** The pipeline is **already multi-class compatible**. Only the data balancing script needs updates to optionally include edge samples. Once edge samples are in the balanced datasets, everything downstream works automatically.

**Testing Strategy:**
1. Create both binary and 3-class balanced datasets
2. Run experiments on both
3. Compare performance, confusion matrices, and deployment suitability
4. Choose based on accuracy vs robustness tradeoff
