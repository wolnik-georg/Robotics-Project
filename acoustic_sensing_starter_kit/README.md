# Acoustic-Based Contact Detection for Robotic Manipulation

**End-to-End Experimental Pipeline for 3-Class Acoustic Contact Sensing**

This repository contains the complete implementation of acoustic sensing for contact detection and geometric reconstruction on rigid robotic manipulators, as described in the paper "Acoustic-Based Contact Detection and Geometric Reconstruction for Robotic Manipulation" (Wolnik, 2026).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Main Results](#reproducing-main-results)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset Structure](#dataset-structure)
- [Configuration Files](#configuration-files)
- [Main Execution Scripts](#main-execution-scripts)
- [Source Code Structure](#source-code-structure)
- [Experimental Results](#experimental-results)
- [Figure Generation](#figure-generation)
- [Documentation](#documentation)
- [Advanced Usage](#advanced-usage)
- [Performance Summary](#performance-summary)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## 🎯 Overview

This work investigates acoustic sensing as a contact detection modality for rigid robotic manipulators. We develop a complete pipeline from data collection through machine learning to 2D contact state mapping with explicit edge detection.

**Key Contributions:**
- First demonstration of **3-class acoustic contact detection** (contact, no-contact, edge) for rigid manipulators
- Systematic **generalization analysis**: position generalization (workspace rotations) and object generalization (novel geometries)
- Multi-seed validation proving **reproducibility** (5 independent seeds, std=0.0%)
- Physics-based **eigenfrequency analysis** explaining generalization failures and successes

**Experimental Platform:**
- Robot: Franka Emika Panda 7-DOF manipulator
- Sensor: Custom acoustic finger with contact microphone
- Objects: 4 wooden boards with different geometries (cutouts, raised shapes, empty)
- Workspaces: 4 different spatial configurations

---

## 🔬 Key Findings

### Proof of Concept (RQ1)
- **69.9% cross-validation accuracy** (2.10× over random baseline)
- Validates feasibility for within-workspace scenarios
- 3-class outperforms binary when normalized (1.04× vs 0.90×)

### Position Generalization (RQ2)
- **Catastrophic workspace-dependent failure**: 23.3–55.7% validation range
- Average 34.5% (barely above 33.3% random baseline)
- Two rotations worse than random (0.70× and 0.73× normalized)
- **Workspace-specific training is mandatory**

### Object Generalization (RQ4)
- **Classifier-dependent results** validated across 5 seeds (std=0.0%)
- Heavily-regularized GPU-MLP: **75.0% validation** (dropout 0.3, weight decay 0.01)
- Unregularized models fail: 35.7–41.7%
- Binary classification collapses to **50% (pure random chance)**
- **Accuracy-coverage tradeoff**: 75% accuracy on only 0.2% of spatial positions

### 3-Class vs Binary (RQ3)
- Binary performs **worse than random guessing** (0.90× normalized)
- 3-class achieves 1.04× over random
- Edge samples contain **essential discriminative information**

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for GPU-MLP)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/wolnik-georg/Robotics-Project.git
cd Robotics-Project/acoustic_sensing_starter_kit
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python run_modular_experiments.py --validate-only
```

### Key Dependencies

**Core Scientific Computing:**
- `numpy>=1.19.0`, `scipy>=1.5.0` - Numerical computing and scientific operations
- `pandas>=0.20.2` - Data manipulation and analysis

**Audio Processing:**
- `librosa>=0.9.0` - Audio feature extraction and spectral analysis
- `soundfile>=0.10.0` - Audio file I/O
- `pyaudio>=0.2.11` - Audio recording

**Machine Learning:**
- `scikit-learn>=0.20.0` - ML classifiers (Random Forest, SVM, LDA, MLP, K-NN)
- `xgboost>=1.5.0` - Gradient boosting classifier
- `imbalanced-learn>=0.8.0` - SMOTE and class balancing

**Deep Learning (GPU Acceleration - Optional):**
- `torch>=2.0.0`, `torchaudio>=2.0.0` - GPU-accelerated neural networks
- `cupy-cuda12x` - GPU-accelerated numpy operations

**Visualization:**
- `matplotlib>=3.3.0`, `seaborn>=0.11.0` - Plotting and visualization

**Hyperparameter Optimization:**
- `optuna>=3.0.0` - Automated hyperparameter tuning

**Configuration & Utilities:**
- `pyyaml>=5.4.0` - Configuration file parsing
- `pillow>=8.0.0`, `imageio>=2.9.0` - Image processing and I/O

---

## 🚀 Quick Start

### Complete Pipeline (One Command)

**Reproduce all main results** with a single command:

```bash
bash run_complete_pipeline.sh
```

This runs the entire pipeline end-to-end (~4-5 hours):
1. ✅ Dataset balancing
2. ✅ Position generalization (3 rotations)
3. ✅ Object generalization (5 seeds)
4. ✅ Figure generation

**For step-by-step execution**, see [Reproducing Main Results](#reproducing-main-results).

---

### Individual Components

#### 1. Dataset Balancing

Create perfectly balanced 3-class datasets (33/33/33 splits):

```bash
bash run_balance_datasets.sh
```

**Output:** `data/fully_balanced_datasets/rotation*_{train,val}/`

#### 2. Position Generalization (3 Workspace Rotations)

Run all 3 workspace rotations:

```bash
bash run_3class_rotations.sh
```

This executes:
- **Rotation 1**: Train WS1+WS3 → Validate WS2
- **Rotation 2**: Train WS2+WS3 → Validate WS1
- **Rotation 3**: Train WS1+WS2 → Validate WS3

**Output:** `fully_balanced_rotation{1,2,3}_results/`

#### 3. Object Generalization (Multi-Seed Validation)

Run object generalization with 5 independent seeds:

```bash
python run_object_generalization_multiseed.py
```

Seeds tested: 42, 123, 456, 789, 1024

**Output:** `object_generalization_ws4_holdout_3class_seed_*/`

#### 4. Generate Figures

Create all reconstruction visualizations and ML analysis figures:

```bash
# Generate all 3 main reconstruction figures
python generate_comprehensive_reconstructions.py

# Generate ML analysis figures (feature architecture, experimental setup)
python generate_ml_analysis_figures.py
```

**Output:** 
- `comprehensive_3class_reconstruction/*.png`
- `ml_analysis_figures/*.png`

---

## 🏗️ Pipeline Architecture

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION                                              │
│    └─ Raw datasets: data/collected_data_runs_*/                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. DATASET BALANCING                                            │
│    Script: create_fully_balanced_datasets.py                    │
│    Config: dataset_paths_config.yml                             │
│    Output: data/fully_balanced_datasets/rotation*_{train,val}  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. TRAINING & VALIDATION                                        │
│    Main Pipeline: run_modular_experiments.py                    │
│    Orchestrator: src/acoustic_sensing/experiments/orchestrator.py│
│                                                                  │
│    Position Generalization:                                     │
│    └─ run_3class_rotations.sh                                  │
│       ├─ Rotation 1: configs/multi_dataset_config.yml          │
│       ├─ Rotation 2: configs/rotation_ws2_ws3_train_ws1_val.yml│
│       └─ Rotation 3: configs/rotation_ws1_ws2_train_ws3_val.yml│
│                                                                  │
│    Object Generalization:                                       │
│    └─ run_object_generalization_multiseed.py                   │
│       └─ configs/object_generalization_3class_seed_*.yml       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. FIGURE GENERATION                                            │
│    ├─ generate_comprehensive_reconstructions.py                │
│    │  └─ 3 main reconstruction figures                         │
│    └─ generate_ml_analysis_figures.py                          │
│       └─ Feature architecture, experimental setup              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. FINAL REPORT                                                 │
│    └─ docs/final_report.tex (IEEE conference format)           │
└─────────────────────────────────────────────────────────────────┘
```

### Core Pipeline Component

**`run_modular_experiments.py`** is the **heart of the pipeline**:

```bash
python run_modular_experiments.py <config_file> [output_dir]

# Example:
python run_modular_experiments.py configs/multi_dataset_config.yml
```

**What it does:**
1. Loads YAML configuration
2. Initializes `ExperimentOrchestrator`
3. Loads data via `geometric_data_loader.py`
4. Extracts features (hand-crafted or spectrograms)
5. Trains classifiers via `multi_dataset_training.py`
6. Evaluates on validation data
7. Generates confusion matrices and metrics
8. (Optional) Performs 2D surface reconstruction

---

## 📁 Dataset Structure

### Raw Data Collection

```
data/
├── collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled/
│   ├── data/
│   │   ├── audio_recordings/
│   │   └── metadata.json
│   └── sweep.csv (spatial positions)
├── collected_data_runs_2026_01_15_workspace_1_pure_contact_relabeled/
├── collected_data_runs_2026_01_15_workspace_1_pure_no_contact/
├── collected_data_runs_2026_01_15_workspace_2_squares_cutout_relabeled/
├── collected_data_runs_2026_01_15_workspace_2_pure_contact_relabeled/
├── collected_data_runs_2026_01_15_workspace_2_pure_no_contact/
├── collected_data_runs_2025_12_17_v2_workspace_3_squares_cutout_relabeled/
├── collected_data_runs_2026_01_14_workspace_3_pure_contact_relabeled/
├── collected_data_runs_2026_01_14_workspace_3_pure_no_contact/
└── collected_data_runs_2026_01_27_hold_out_dataset_relabeled/  # WS4 Object D
```

**Dataset Types:**
- `*_squares_cutout_*`: Object A (wooden board with geometric cutouts)
- `*_pure_contact_*`: Object C (wooden board with raised shapes)
- `*_pure_no_contact`: Object B (empty workspace)
- `*_hold_out_*`: Object D (large square cutout, held-out for object generalization)

### Balanced 3-Class Datasets

Created by `create_fully_balanced_datasets.py`:

```
data/fully_balanced_datasets/
├── rotation1_train/        # WS1 + WS3 combined, balanced
├── rotation1_val/          # WS2, balanced
├── rotation2_train/        # WS2 + WS3 combined
├── rotation2_val/          # WS1
├── rotation3_train/        # WS1 + WS2 combined
├── rotation3_val/          # WS3
└── holdout/                # WS4 Object D, balanced
```

**Each balanced dataset contains:**
- Perfect 33/33/33 class distribution (contact, no-contact, edge)
- `sweep.csv` with spatial position information for reconstruction
- `data/audio_recordings/` with balanced audio samples
- `metadata.json` with dataset information

---

## ⚙️ Configuration Files

### Main Pipeline Configurations

Located in `configs/`:

#### Position Generalization (Workspace Rotations)

| Config File | Training Data | Validation Data | Description |
|-------------|---------------|-----------------|-------------|
| `multi_dataset_config.yml` | WS1 + WS3 | WS2 | **Primary config** for Rotation 1 |
| `rotation_ws2_ws3_train_ws1_val.yml` | WS2 + WS3 | WS1 | Rotation 2 |
| `rotation_ws1_ws2_train_ws3_val.yml` | WS1 + WS2 | WS3 | Rotation 3 |

#### Binary Classification (for comparison)

| Config File | Mode |
|-------------|------|
| `rotation1_binary.yml` | Binary (exclude edge), Rotation 1 |
| `rotation2_binary.yml` | Binary, Rotation 2 |
| `rotation3_binary.yml` | Binary, Rotation 3 |

#### Object Generalization (Multi-Seed)

| Config File | Random Seed |
|-------------|-------------|
| `object_generalization_3class.yml` | Base config |
| `object_generalization_3class_seed_42.yml` | 42 |
| `object_generalization_3class_seed_123.yml` | 123 |
| `object_generalization_3class_seed_456.yml` | 456 |
| `object_generalization_3class_seed_789.yml` | 789 |
| `object_generalization_3class_seed_1024.yml` | 1024 |
| `object_generalization_binary.yml` | Binary mode |

### Configuration Structure

Example `multi_dataset_config.yml`:

```yaml
# Dataset paths
datasets:
  - "fully_balanced_datasets/rotation1_train"
  
validation_datasets:
  - "fully_balanced_datasets/rotation1_val"

# Class filtering (3-class vs binary)
class_filtering:
  enabled: false  # false = 3-class (contact, no_contact, edge)
  classes_to_exclude_train: ["edge"]
  classes_to_exclude_validation: ["edge"]

# Feature extraction
feature_extraction:
  modes:
    - "features"  # Hand-crafted features (80D)
    # - "spectrogram"  # Mel spectrograms (10,240D)
  
  spectrogram:
    n_fft: 512
    hop_length: 128
    n_mels: 80
    time_bins: 128

# Experiments to run
experiments:
  discrimination_analysis:
    enabled: true
    classifiers:
      - RandomForest
      - KNN
      - MLP
      - GPU_MLP
    cv_folds: 5
```

**Key Configuration Options:**

- `datasets`: Training dataset paths (can combine multiple workspaces)
- `validation_datasets`: Held-out validation datasets
- `class_filtering.enabled`: 
  - `false` = 3-class mode (contact, no_contact, edge)
  - `true` = binary mode (exclude edge samples)
- `feature_extraction.modes`: 
  - `"features"` = Hand-crafted features (80D: spectral, MFCCs, temporal, impulse)
  - `"spectrogram"` = Mel-spectrograms (10,240D)
- `experiments`: Which analyses to run (discrimination, reconstruction, etc.)

### Dataset Balancing Configuration

`dataset_paths_config.yml`:

```yaml
workspace_1:
  cutout: "data/collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled"
  contact: "data/collected_data_runs_2026_01_15_workspace_1_pure_contact_relabeled"
  no_contact: "data/collected_data_runs_2026_01_15_workspace_1_pure_no_contact"

workspace_2:
  # ... similar structure

workspace_3:
  # ... similar structure

workspace_4:
  holdout: "data/collected_data_runs_2026_01_27_hold_out_dataset_relabeled"

output:
  directory: "data/fully_balanced_datasets"
```

---

## 🎯 Main Execution Scripts

### Master Pipeline Script

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_complete_pipeline.sh` | **Complete end-to-end pipeline** | `bash run_complete_pipeline.sh` |

**Recommended:** Use this script to reproduce all main results with one command.

Runs: Dataset balancing → Position generalization → Object generalization → Figure generation

---

### Data Preparation

| Script | Purpose | Usage |
|--------|---------|-------|
| `create_fully_balanced_datasets.py` | Create balanced 3-class datasets | `python create_fully_balanced_datasets.py` |
| `run_balance_datasets.sh` | Shell wrapper for balancing | `bash run_balance_datasets.sh` |
| `analyze_dataset_balance.py` | Verify balance and distribution | `python analyze_dataset_balance.py` |

### Training & Validation

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_modular_experiments.py` | **Main pipeline script** | `python run_modular_experiments.py <config> [output]` |
| `run_3class_rotations.sh` | Run all 3 rotations | `bash run_3class_rotations.sh` |
| `run_object_generalization_multiseed.py` | Multi-seed object generalization | `python run_object_generalization_multiseed.py` |
| `run_object_generalization.sh` | Single-seed wrapper | `bash run_object_generalization.sh` |
| `run_all_binary_experiments.sh` | Binary classification experiments | `bash run_all_binary_experiments.sh` |

### Figure Generation

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_comprehensive_reconstructions.py` | All 3 main reconstruction figures | `comprehensive_3class_reconstruction/*.png` |
| `generate_ml_analysis_figures.py` | ML analysis figures | `ml_analysis_figures/*.png` |
| `generate_3class_rotation_figures.py` | Rotation comparison figures | Various |
| `create_combined_reconstruction_figures.py` | Combined panels | Various |
| `regenerate_all_figures_fully_balanced.py` | Regenerate all figures | Various |

### Analysis & Utilities

| Script | Purpose |
|--------|---------|
| `run_surface_reconstruction.py` | 2D spatial reconstruction from trained models |
| `analyze_dataset_balance.py` | Dataset balance verification |

### Deprecated/Development Scripts

**Note:** The repository contains 52 Python scripts in total. The scripts listed above represent the **main pipeline** used for the final results in the paper. Many other scripts (e.g., `test_*.py`, `verify_*.py`, `run_pattern_*.py`, `demo_*.py`, `tune_*.py`, etc.) were created during the research and development process for:
- Exploratory analysis and prototyping
- Testing different approaches
- Debugging and validation
- Ablation studies not included in final report
- Legacy implementations superseded by the modular framework

These scripts are **not required** for reproducing the main experimental results but are preserved for reference and transparency about the research process.

---

## 📦 Source Code Structure

### Main Package: `src/acoustic_sensing/`

```
src/acoustic_sensing/
├── experiments/              # Experiment orchestration and execution
│   ├── orchestrator.py       # Main experiment coordinator
│   ├── multi_dataset_training.py  # Multi-dataset training logic
│   ├── discrimination_analysis.py # ML classifier training/evaluation
│   ├── surface_reconstruction.py  # 2D spatial reconstruction
│   ├── data_processing.py    # Data loading and preprocessing
│   ├── gpu_classifiers.py    # GPU-accelerated MLP implementations
│   └── base_experiment.py    # Base class for all experiments
│
├── features/                 # Feature extraction
│   └── (Feature extraction modules)
│
├── models/                   # Data loading and reconstruction
│   ├── geometric_data_loader.py   # Load data with spatial positions
│   ├── geometric_reconstruction.py # Reconstruct from predictions
│   └── training.py           # Training utilities
│
├── analysis/                 # Analysis modules
│   ├── discrimination_analysis.py # Classifier comparison and metrics
│   ├── batch_analysis.py     # Batch processing utilities
│   └── dimensionality_analysis.py # PCA, t-SNE analysis
│
├── visualization/            # Plotting and figure generation
│   └── (Visualization utilities)
│
└── core/                     # Core utilities
    └── (Core functionality)
```

### Key Modules

#### `experiments/orchestrator.py`
**Main experiment coordinator** that:
- Loads YAML configuration
- Validates configuration
- Coordinates all experiments (data processing, training, analysis)
- Manages output directories
- Handles logging

#### `experiments/multi_dataset_training.py`
Handles training on multiple combined datasets:
- Combines data from multiple workspaces
- Stratified cross-validation (5-fold)
- Class balancing
- Validation on held-out workspaces

#### `experiments/discrimination_analysis.py`
ML classifier training and evaluation:
- Random Forest (100 trees)
- K-Nearest Neighbors
- Multi-Layer Perceptron
- GPU-accelerated MLP (with regularization)
- Ensemble methods
- Confusion matrix generation
- Performance metrics (accuracy, precision, recall, F1)

#### `models/geometric_data_loader.py`
Loads acoustic data with spatial position information:
- Audio file loading
- Feature extraction (hand-crafted or spectrograms)
- Label loading (contact, no_contact, edge)
- Position loading from `sweep.csv`
- Data augmentation (optional)

#### `experiments/surface_reconstruction.py`
2D spatial reconstruction from predictions:
- Maps predictions to 2D spatial coordinates
- Generates ground truth vs prediction visualizations
- Error maps
- Confidence filtering
- Saves reconstruction figures

---

## 📊 Experimental Results

### Results Directory Structure

```
acoustic_sensing_starter_kit/
├── fully_balanced_rotation1_results/          # Rotation 1 (3-class)
│   ├── discriminationanalysis/
│   │   ├── cv_results/                        # Cross-validation results
│   │   └── validation_results/                # Validation results
│   │       ├── classifier_performance.png
│   │       ├── confusion_matrix_*.png
│   │       └── metrics.json
│   └── model.pkl                              # Trained Random Forest
│
├── fully_balanced_rotation2_results/          # Rotation 2 (3-class)
├── fully_balanced_rotation3_results/          # Rotation 3 (3-class)
│
├── fully_balanced_rotation1_binary/           # Rotation 1 (binary)
├── fully_balanced_rotation2_binary/           # Rotation 2 (binary)
├── fully_balanced_rotation3_binary/           # Rotation 3 (binary)
│
├── fully_balanced_rotation1_results_features_rerun/      # Hand-crafted features
├── fully_balanced_rotation1_results_spectogram_rerun/    # Spectrograms
│
├── object_generalization_ws4_holdout_3class_fully_balanced/  # Main 3-class
├── object_generalization_ws4_holdout_3class_seed_42/    # Seed 42
├── object_generalization_ws4_holdout_3class_seed_123/   # Seed 123
├── object_generalization_ws4_holdout_3class_seed_456/   # Seed 456
├── object_generalization_ws4_holdout_3class_seed_789/   # Seed 789
├── object_generalization_ws4_holdout_3class_seed_1024/  # Seed 1024
└── object_generalization_ws4_holdout_binary_fully_balanced/  # Binary
```

### Results Files

Each results directory contains:

- `discriminationanalysis/cv_results/`
  - `confusion_matrix_cv_RandomForest.png` - CV confusion matrix
  - `confusion_matrix_cv_*.png` - Other classifiers
  - `cv_metrics.json` - Cross-validation metrics

- `discriminationanalysis/validation_results/`
  - `classifier_performance.png` - **Main comparison figure**
  - `confusion_matrix_validation_RandomForest.png` - Validation confusion
  - `confusion_matrix_validation_*.png` - Other classifiers
  - `metrics.json` - **Numerical results**
  
- `model.pkl` - Trained Random Forest model (can be loaded for reconstruction)

### Key Performance Metrics

From `metrics.json`:

```json
{
  "cv_accuracy": 0.699,           // Cross-validation accuracy
  "cv_std": 0.008,                // Standard deviation across folds
  "validation_accuracy": 0.345,   // Held-out validation accuracy
  "random_baseline": 0.333,       // Random baseline (3-class)
  "normalized_performance": 1.04, // (val_acc / random_baseline)
  "cv_validation_gap": 0.354      // Performance drop (CV - validation)
}
```

---

## 🎨 Figure Generation

### Main Reconstruction Figures

Generated by `generate_comprehensive_reconstructions.py`:

**Location:** `comprehensive_3class_reconstruction/`

1. **`proof_of_concept_reconstruction_combined.png`**
   - 80/20 train/test split on combined workspaces (WS1+WS2+WS3)
   - Achieves ~93% average accuracy
   - Validates feasibility for within-workspace scenarios

2. **`test_reconstruction_combined.png`**
   - Position generalization: Rotation 1 validation (WS2)
   - Trained on WS1+WS3
   - Achieves 34.89% accuracy (barely above 33.3% random)
   - Demonstrates catastrophic workspace-dependent failure

3. **`holdout_reconstruction_combined.png`**
   - Object generalization: WS4 Object D (holdout)
   - Side-by-side: without vs with confidence filtering
   - Without filtering: 33.03% (random chance)
   - With filtering (threshold 0.7): 75% accuracy on 0.2% coverage
   - Reveals accuracy-coverage tradeoff

### ML Analysis Figures

Generated by `generate_ml_analysis_figures.py`:

**Location:** `ml_analysis_figures/`

1. **`figure6_experimental_setup.png`**
   - Workspace rotation experimental strategy
   - Shows all 3 rotations with train/validation splits

2. **`figure11_feature_dimensions.png`**
   - Hand-crafted feature architecture
   - 80-dimensional feature vector breakdown:
     - 11 spectral features
     - 39 MFCCs (13 + Δ + ΔΔ)
     - 15 temporal features
     - 15 impulse response features

### Feature Comparison Figures

**Hand-crafted features:**
- `fully_balanced_rotation1_results_features_rerun/discriminationanalysis/validation_results/classifier_performance.png`

**Spectrograms:**
- `fully_balanced_rotation1_results_spectogram_rerun/discriminationanalysis/validation_results/classifier_performance.png`

Shows that hand-crafted features (80D) outperform spectrograms (10,240D) by 11 percentage points due to reduced overfitting.

### Setup Images

**Location:** `presentation/`

- `big_setup.jpeg` - Full experimental platform (Franka Panda)
- `mounting_setup.jpeg` - Acoustic finger mounting on gripper
- `close_setup.jpeg` - Contact area close-up (1cm × 0.25cm)

Used in final report as Figure 1.

---

## 📚 Documentation

### Essential Documentation

Located in the main directory:

- **`DATA_COLLECTION_PROTOCOL.md`**
  - Data collection methodology
  - Robot control protocol
  - Acoustic sensor specifications
  - Ground truth labeling procedure
  - Multi-sample recording protocol (5-10 samples per position, 150ms settling time)

- **`PIPELINE_GUIDE.md`**
  - General pipeline usage guide
  - Configuration file structure
  - Common workflows
  - Troubleshooting

- **`PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md`**
  - Physics-based eigenfrequency analysis
  - Explains workspace-specific acoustic signatures
  - Explains object-specific eigenfrequency spectra
  - Theory behind regularization success

- **`RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md`**
  - Comprehensive research findings
  - Detailed experimental analysis
  - Statistical validation
  - Comparative studies

### Final Report

**Location:** `docs/final_report.tex`

IEEE conference format paper with:
- Complete experimental methodology
- All results and figures
- Physics-based interpretation
- Statistical analysis
- Comprehensive literature review

**Compile with:**
```bash
cd docs/
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

---

## 🔬 Reproducing Main Results

This section provides the **exact workflow** to reproduce all main experimental results from the paper.

### Option 1: Complete Reproduction (Recommended)

Run the entire pipeline end-to-end with one command:

```bash
bash run_complete_pipeline.sh
```

**What happens:**
1. ✅ Creates timestamped output directory: `results_YYYYMMDD_HHMMSS/`
2. ✅ Checks if balanced datasets exist → Option to reuse (saves ~5 minutes)
3. ✅ Runs all experiments → Results saved to timestamped directory
4. ✅ **NEVER overwrites existing results** - everything goes to new directory

**Output structure:**
```
results_20260214_143022/           # Timestamped directory
├── rotation1_results/             # Position gen: WS1+WS3 → WS2
├── rotation2_results/             # Position gen: WS2+WS3 → WS1  
├── rotation3_results/             # Position gen: WS1+WS2 → WS3
├── object_generalization_seed_42/
├── object_generalization_seed_123/
├── object_generalization_seed_456/
├── object_generalization_seed_789/
├── object_generalization_seed_1024/
└── figures/
    ├── reconstruction/            # Main figures
    └── analysis/                  # ML analysis figures

data/fully_balanced_datasets/      # Balanced datasets (reusable)
```

**Total estimated time:** ~4-5 hours (or ~4 hours if reusing datasets)

---

### Option 2: Step-by-Step Reproduction

For better understanding, run each step manually:

#### Step 1: Dataset Preparation (~5 minutes)

```bash
# Create perfectly balanced 3-class datasets (33/33/33 splits)
python create_fully_balanced_datasets.py

# Verify balance
python analyze_dataset_balance.py
```

**Verify:** Check that `data/fully_balanced_datasets/` contains:
- `rotation1_train/`, `rotation1_val/`
- `rotation2_train/`, `rotation2_val/`
- `rotation3_train/`, `rotation3_val/`
- Each with 33/33/33 class balance

---

#### Step 2: Position Generalization (~3 hours)

**Run all 3 workspace rotations** to test position generalization:

```bash
bash run_3class_rotations.sh
```

**Or run individually:**
```bash
# Rotation 1: Train WS1+WS3 → Validate WS2
python run_modular_experiments.py configs/multi_dataset_config.yml

# Rotation 2: Train WS2+WS3 → Validate WS1
python run_modular_experiments.py configs/rotation_ws2_ws3_train_ws1_val.yml

# Rotation 3: Train WS1+WS2 → Validate WS3
python run_modular_experiments.py configs/rotation_ws1_ws2_train_ws3_val.yml
```

**Verify results:**
```bash
# Check that results directories exist
ls -d fully_balanced_rotation{1,2,3}_results/

# Check validation accuracy (should average ~34.5%)
cat fully_balanced_rotation1_results/discriminationanalysis/validation_results/metrics.json
cat fully_balanced_rotation2_results/discriminationanalysis/validation_results/metrics.json
cat fully_balanced_rotation3_results/discriminationanalysis/validation_results/metrics.json
```

**Expected validation accuracies:**
- Rotation 1 (WS2): ~55.7%
- Rotation 2 (WS1): ~24.4%
- Rotation 3 (WS3): ~23.3%
- **Average: ~34.5%** (barely above 33.3% random baseline)

---

#### Step 3: Object Generalization (~1.5 hours)

**Run multi-seed validation** to verify reproducibility:

```bash
python run_object_generalization_multiseed.py
```

This runs 5 independent experiments with seeds: 42, 123, 456, 789, 1024

**Verify results:**
```bash
# Check that all 5 seed directories exist
ls -d object_generalization_ws4_holdout_3class_seed_*/

# Check GPU-MLP HighReg performance (should be 75.0% with std=0.0%)
grep -r "GPU_MLP_Medium_HighReg" object_generalization_ws4_holdout_3class_seed_*/discriminationanalysis/validation_results/metrics.json
```

**Expected key finding:**
- GPU-MLP HighReg: **75.0% validation** (dropout=0.3, weight_decay=0.01)
- Standard deviation: **0.0%** across all 5 seeds (perfect reproducibility)

---

#### Step 4: Binary Classification Comparison (~3 hours, optional)

**Compare 3-class vs binary** to validate edge class importance:

```bash
bash run_all_binary_experiments.sh
```

**Verify results:**
```bash
# Check binary results
ls -d fully_balanced_rotation{1,2,3}_binary/

# Compare binary vs 3-class performance
cat fully_balanced_rotation1_binary/discriminationanalysis/validation_results/metrics.json
```

**Expected finding:**
- Binary average: ~45.1% (0.90× vs 50% random baseline)
- 3-class average: ~34.5% (1.04× vs 33.3% random baseline)
- **Binary performs worse than random guessing!**

---

#### Step 5: Figure Generation (~10 minutes)

**Generate all figures** used in the paper:

```bash
# Main reconstruction figures (Figures 7, 8, 9)
python generate_comprehensive_reconstructions.py

# ML analysis figures (Figures 6, 11)
python generate_ml_analysis_figures.py

# Position generalization comparison figures
python generate_3class_rotation_figures.py

# (Optional) Regenerate all figures from scratch
python regenerate_all_figures_fully_balanced.py
```

**Verify outputs:**
```bash
# Check main reconstruction figures
ls comprehensive_3class_reconstruction/*.png

# Check ML analysis figures
ls ml_analysis_figures/*.png
```

**Expected figures:**
1. `proof_of_concept_reconstruction_combined.png` - ~93% accuracy (within-workspace)
2. `test_reconstruction_combined.png` - ~34.89% accuracy (position generalization failure)
3. `holdout_reconstruction_combined.png` - 33% → 75% with confidence filtering
4. `figure6_experimental_setup.png` - Workspace rotation strategy
5. `figure11_feature_dimensions.png` - Feature architecture (80D)

---

### Verification Checklist

After running the complete workflow, verify:

- [ ] **Dataset balance:** All datasets have 33/33/33 class distribution
- [ ] **Position generalization:** Average validation ~34.5% (catastrophic failure)
- [ ] **Object generalization:** GPU-MLP HighReg achieves 75.0% (std=0.0%)
- [ ] **Binary comparison:** Binary performs worse than random (0.90× normalized)
- [ ] **Figures generated:** All main figures present in output directories
- [ ] **Results match paper:** Key metrics align with reported values

### Troubleshooting

**If experiments fail:**
1. Check dataset paths in `dataset_paths_config.yml`
2. Verify balanced datasets exist: `ls data/fully_balanced_datasets/`
3. Check Python environment: `pip install -r requirements.txt`
4. Review experiment logs in results directories

**If figures don't generate:**
1. Check that results directories exist
2. Verify matplotlib backend: `export MPLBACKEND=Agg`
3. Re-run experiments if metrics.json files are missing

---

## 🎯 Advanced Usage

### Custom Experiments

Create a custom config file:

```yaml
# configs/my_experiment.yml
datasets:
  - "data/my_dataset_1"
  - "data/my_dataset_2"

validation_datasets:
  - "data/my_validation_set"

class_filtering:
  enabled: false  # 3-class mode

feature_extraction:
  modes:
    - "features"

experiments:
  discrimination_analysis:
    enabled: true
    classifiers:
      - RandomForest
      - GPU_MLP
    cv_folds: 5
```

Run:
```bash
python run_modular_experiments.py configs/my_experiment.yml my_results/
```

### Feature Extraction Modes

Edit `feature_extraction.modes` in config:

```yaml
feature_extraction:
  modes:
    - "features"           # Hand-crafted (80D)
    - "spectrogram"        # Mel-spectrogram (10,240D)
    - "mfcc"              # MFCC only
    - "magnitude_spectrum" # Raw magnitude spectrum
    - "power_spectrum"     # Power spectrum
    - "chroma"            # Chroma features
```

### GPU Acceleration

For GPU-MLP classifier with regularization:

```yaml
experiments:
  discrimination_analysis:
    classifiers:
      - GPU_MLP_Medium_HighReg  # dropout=0.3, weight_decay=0.01
```

Requires PyTorch with CUDA support.

### Surface Reconstruction

Enable reconstruction in config:

```yaml
reconstruction:
  enabled: true
  output_dir: "reconstruction_results"
```

Or run separately:
```bash
python run_surface_reconstruction.py \
    --model fully_balanced_rotation1_results/model.pkl \
    --dataset data/fully_balanced_datasets/rotation1_val \
    --output reconstruction_output/
```

---

## 📈 Performance Summary

### Position Generalization (3 Workspace Rotations)

| Rotation | Training | Validation | CV Accuracy | Val Accuracy | Normalized |
|----------|----------|------------|-------------|--------------|------------|
| 1 | WS1+WS3 | WS2 | 69.1% | **55.7%** | 1.67× |
| 2 | WS2+WS3 | WS1 | 69.8% | 24.4% | 0.73× |
| 3 | WS1+WS2 | WS3 | 70.7% | **23.3%** | 0.70× |
| **Average** | --- | --- | **69.9%** | **34.5%** | **1.04×** |

*Random baseline: 33.3% (3-class)*

### Object Generalization (5-Seed Validation)

| Classifier | CV Accuracy | Val Accuracy | std | vs Random |
|------------|-------------|--------------|-----|-----------|
| Random Forest | 70.8% ± 0.7% | 41.7% | 0.0% | 1.25× |
| **GPU-MLP HighReg** | **57.2% ± 0.9%** | **75.0%** | **0.0%** | **2.25×** |
| GPU-MLP (no reg) | 48.8% ± 0.9% | 35.7% | 0.0% | 1.07× |
| K-NN | 47.1% ± 0.7% | 33.4% | 0.0% | 1.00× |
| Ensemble | 60.4% ± 0.8% | 30.1% | 0.0% | 0.90× |

*Random baseline: 33.3% (3-class)*

**Binary mode:** All classifiers = 50.0% (pure random chance)

### 3-Class vs Binary

| Mode | Classes | Val Accuracy | Random | Normalized |
|------|---------|--------------|--------|------------|
| 3-Class | contact, no_contact, edge | 34.5% | 33.3% | **1.04×** |
| Binary | contact, no_contact | 45.1% | 50.0% | **0.90×** ⚠️ |

*Binary performs worse than random guessing!*

### Hand-Crafted vs Spectrograms (Rotation 1)

| Feature Type | Dimensions | Val Accuracy | Winner |
|--------------|------------|--------------|--------|
| **Hand-crafted** | 80 | **33.9%** | 5/5 ✓ |
| Spectrograms | 10,240 | 22.9% | 0/5 |
| **Advantage** | --- | **+11.0%** | --- |

*128× more parameters leads to worse performance due to overfitting*

---

## 🐛 Troubleshooting

### Common Issues

**1. Configuration validation failed:**
```bash
python run_modular_experiments.py --validate-only
```
Check dataset paths and config syntax.

**2. GPU not available for GPU-MLP:**
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```
Install PyTorch with CUDA support if needed.

**3. Dataset not found:**
- Verify paths in `dataset_paths_config.yml`
- Ensure balanced datasets exist: `ls data/fully_balanced_datasets/`
- Re-run balancing if needed: `python create_fully_balanced_datasets.py`

**4. Out of memory during training:**
- Reduce batch size in config
- Use smaller feature dimensions
- Disable GPU-MLP (use CPU-based classifiers)

**5. Figures not generating:**
- Check that results directories exist
- Verify matplotlib backend: `export MPLBACKEND=Agg`
- Ensure all dependencies installed: `pip install -r requirements.txt`

---

## 🤝 Contributing

This repository contains the complete experimental pipeline for the research paper. For questions or issues:

1. Review the configuration examples in `configs/` directory
2. Check the troubleshooting section above
3. Refer to the complete workflow and advanced usage sections
4. Contact: georg.wolnik@campus.tu-berlin.de

---

## 📄 License

This project is part of academic research at TU Berlin. Please cite if you use this code.

---

## 📖 Citation

If you use this code or find this work useful, please cite:

```bibtex
@inproceedings{wolnik2026acoustic,
  title={Acoustic-Based Contact Detection and Geometric Reconstruction for Robotic Manipulation},
  author={Wolnik, Georg},
  booktitle={Proceedings of [Conference Name]},
  year={2026},
  organization={Technische Universit{\"a}t Berlin}
}
```

---

## 👤 Author

**Georg Wolnik**  
Robotics and Biology Laboratory  
Technische Universität Berlin  
Berlin, Germany  
wolnik@campus.tu-berlin.de

---

## 🙏 Acknowledgments

- **Advisor:** Paul (Pu Xu), TU Berlin Robotics and Biology Laboratory
- **Platform:** Franka Emika Panda robot manipulator
- **Sensor:** Custom acoustic finger based on Wall et al. (2019)
- **Inspiration:** VibeCheck framework (Zhang et al., 2025) for configuration entanglement analysis

---

## 📊 Repository Statistics

- **Total Lines of Code:** ~15,000+
- **Python Modules:** 40+
- **Configuration Files:** 20+
- **Figures Generated:** 73+
- **Experimental Results:** 15+ directories
- **Documentation:** 4 comprehensive MD files + IEEE paper

---

**Last Updated:** February 13, 2026  
**Version:** 1.0  
