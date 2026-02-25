# Robotics-Project

> **Note:** The actual project lives in the [`acoustic_sensing_starter_kit/`](./acoustic_sensing_starter_kit/) subdirectory. Everything below is copied from there for convenience.

---

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

---

## 📊 Experimental Results

### Results Directory Structure

```
acoustic_sensing_starter_kit/
├── fully_balanced_rotation1_results/
├── fully_balanced_rotation2_results/
├── fully_balanced_rotation3_results/
├── object_generalization_ws4_holdout_3class_seed_42/
├── object_generalization_ws4_holdout_3class_seed_123/
├── object_generalization_ws4_holdout_3class_seed_456/
├── object_generalization_ws4_holdout_3class_seed_789/
└── object_generalization_ws4_holdout_3class_seed_1024/
```

---

## 🎨 Figure Generation

Generated by `generate_comprehensive_reconstructions.py` → `comprehensive_3class_reconstruction/`:

1. **`proof_of_concept_reconstruction_combined.pdf`** — 80/20 split, ~93% average accuracy
2. **`test/`** — Position generalization test data reconstructions
3. **`validation/`** — Position generalization validation reconstructions
4. **`holdout/`** — Object generalization reconstruction (33% = random chance)

---

## 📚 Documentation

- **`DATA_COLLECTION_PROTOCOL.md`** — Data collection methodology
- **`PIPELINE_GUIDE.md`** — Pipeline usage guide
- **`PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md`** — Eigenfrequency analysis
- **`RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md`** — Comprehensive findings
- **`docs/final_report_condensed.tex`** — IEEE conference format paper

---

## 🔬 Reproducing Main Results

```bash
# Full pipeline (recommended)
cd acoustic_sensing_starter_kit/
bash run_complete_pipeline.sh
```

See [`acoustic_sensing_starter_kit/README.md`](./acoustic_sensing_starter_kit/README.md) for detailed step-by-step instructions.

---

## 📈 Performance Summary

| Experiment | Val Accuracy | Random Baseline | Normalized |
|------------|-------------|-----------------|------------|
| Proof of Concept (CV) | 69.9% | 33.3% | 2.10× |
| Position Gen (avg) | 34.5% | 33.3% | 1.04× |
| Object Gen (RF) | 41.7% | 33.3% | 1.25× |
| Object Gen (GPU-MLP HighReg) | 75.0% | 33.3% | 2.25× |
| Binary Classification | 45.1% | 50.0% | 0.90× ⚠️ |

---

## 📖 Citation

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

**Author:** Georg Wolnik — Robotics and Biology Laboratory, TU Berlin  
**Last Updated:** February 25, 2026
