# 🗺️ Stage 3: Surface Reconstruction Pipeline

## Overview

The Surface Reconstruction module is the final stage of the acoustic sensing pipeline that transforms trained ML models into **visual 2D surface maps**. It takes the predictions from trained classifiers and reconstructs a spatial representation of detected contact/object patterns.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SURFACE RECONSTRUCTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Trained    │───▶│    Sweep     │───▶│   Generate   │───▶│   2D      │ │
│  │    Model     │    │    Data      │    │  Predictions │    │   Maps    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                             │
│  Dependencies:                                                              │
│  • multi_dataset_training experiment (provides trained models)              │
│  • sweep.csv with spatial coordinates                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 File Structure

```
src/acoustic_sensing/
├── experiments/
│   ├── surface_reconstruction.py      # Main reconstruction experiment
│   ├── multi_dataset_training.py      # Dependency: model training
│   └── orchestrator.py                # Experiment runner
└── models/
    └── geometric_reconstruction.py    # Helper utilities (feature modes)
```

---

## 🔧 Core Components

### 1. SurfaceReconstructionExperiment (Main Class)

**Location:** `src/acoustic_sensing/experiments/surface_reconstruction.py`

```python
class SurfaceReconstructionExperiment(BaseExperiment):
    """
    Reconstruct surface maps from acoustic sweep data using trained models.
    
    Pipeline:
    1. Load trained model from multi_dataset_training
    2. Load sweep data with spatial coordinates (x, y)
    3. Extract features from sweep audio files
    4. Generate predictions for each spatial position
    5. Create 2D visualizations comparing predictions vs ground truth
    """
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `run()` | Main entry point - orchestrates full reconstruction |
| `_process_sweep_data()` | Extracts features from audio files with coordinates |
| `_create_surface_map()` | Generates single 2D scatter plot |
| `_create_comparison_maps()` | Side-by-side ground truth vs predictions |
| `_create_confidence_map()` | Heatmap of prediction confidence |
| `_create_error_map()` | Highlights misclassifications |
| `_create_model_comparison_maps()` | Compare all trained models |
| `_calculate_reconstruction_metrics()` | Accuracy, F1, confusion matrix |

---

### 2. Dependencies

Surface reconstruction **requires** the `multi_dataset_training` experiment to run first:

```python
def get_dependencies(self) -> List[str]:
    return ["multi_dataset_training"]
```

This provides:
- `trained_models` - Dictionary of model name → trained sklearn model
- `scaler` - StandardScaler for feature normalization
- `feature_names` - List of feature columns to extract
- `classes` - List of class labels (e.g., ["contact", "edge", "no_contact"])

---

## 📊 Input Data Format

### sweep.csv Requirements

The sweep dataset must contain a `sweep.csv` file with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `acoustic_filename` | Path to audio file | `./data/1_edge.wav` |
| `normalized_x` | X coordinate (0-1) | `0.25` |
| `normalized_y` | Y coordinate (0-1) | `0.75` |
| `relabeled_label` | Ground truth class | `contact` |

Example `sweep.csv`:
```csv
acoustic_filename,normalized_x,normalized_y,relabeled_label
./data/1_edge.wav,0.0,0.0,no_contact
./data/2_contact.wav,0.25,0.0,contact
./data/3_edge.wav,0.5,0.0,edge
...
```

---

## 🎨 Output Visualizations

The reconstruction generates **6 visualization types**:

### 1. Ground Truth Surface Map
`ground_truth_surface.png`

Shows the actual spatial distribution of classes based on ground truth labels.

### 2. Predicted Surface Map
`predicted_surface.png`

Model predictions mapped to spatial coordinates using the best-performing model.

### 3. Side-by-Side Comparison
`comparison_surface_maps.png`

Two-panel comparison showing ground truth (left) vs predictions (right) with accuracy overlay.

### 4. Confidence Map
`confidence_map.png`

Heatmap where color intensity represents prediction confidence (max probability across classes).

### 5. Error Map
`error_map.png`

- **Green dots**: Correct predictions
- **Red X markers**: Misclassifications

Includes accuracy and error rate statistics.

### 6. Model Comparison Grid
`model_comparison_maps.png`

Up to 6 models compared side-by-side with individual accuracies.

---

## ⚙️ Configuration

### Enable Surface Reconstruction

In `configs/experiment_config.yml`:

```yaml
experiments:
  surface_reconstruction:
    enabled: true
    sweep_dataset: "collected_data_runs_2025_12_15_v1"  # Dataset with sweep.csv

  multi_dataset_training:
    enabled: true  # Required dependency

multi_dataset_training:
  enabled: true
  training_datasets:
    - "balanced_collected_data_2025_12_15_v2_oversample"
  validation_dataset: "balanced_collected_data_2025_12_16_v3_undersample"
```

### Full Configuration Options

```yaml
experiments:
  surface_reconstruction:
    enabled: true/false
    sweep_dataset: "path_to_dataset"  # Must contain sweep.csv
```

---

## 🔄 Execution Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION SEQUENCE                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. LOAD TRAINED MODELS                                                    │
│     └─ Retrieve from shared_data["trained_models"]                         │
│     └─ Get scaler, feature_names, classes                                  │
│                                                                            │
│  2. LOAD SWEEP DATA                                                        │
│     └─ Read sweep.csv from sweep_dataset folder                            │
│     └─ Columns: acoustic_filename, normalized_x, normalized_y, label       │
│                                                                            │
│  3. FEATURE EXTRACTION                                                     │
│     └─ For each audio file in sweep:                                       │
│         └─ librosa.load(audio_path, sr=48000)                              │
│         └─ GeometricFeatureExtractor.extract_features(audio, "comprehensive")│
│         └─ Collect features, labels, coordinates                           │
│                                                                            │
│  4. PREDICTION                                                             │
│     └─ Scale features using training scaler                                │
│     └─ For each trained model:                                             │
│         └─ model.predict(scaled_features)                                  │
│         └─ model.predict_proba(scaled_features)                            │
│     └─ Select best model by validation accuracy                            │
│                                                                            │
│  5. VISUALIZATION                                                          │
│     └─ Generate 6 visualization types                                      │
│     └─ Save to output_dir/                                                 │
│                                                                            │
│  6. METRICS                                                                │
│     └─ Accuracy per model                                                  │
│     └─ Classification report (precision/recall/F1)                         │
│     └─ Confusion matrix                                                    │
│     └─ Save to surface_reconstruction_results.json                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Running the Reconstruction

### Via Modular Experiments Script

```bash
cd acoustic_sensing_starter_kit

# 1. Edit config to enable surface reconstruction
# 2. Ensure multi_dataset_training is also enabled
# 3. Run:
python run_modular_experiments.py --config configs/experiment_config.yml --output_dir results_reconstruction
```

### Programmatic Usage

```python
from acoustic_sensing.experiments.orchestrator import ExperimentOrchestrator

# Create orchestrator with config that has surface_reconstruction enabled
orchestrator = ExperimentOrchestrator(
    config_path="configs/experiment_config.yml",
    output_dir="results_reconstruction"
)

# Run all enabled experiments (dependency order handled automatically)
results = orchestrator.run_experiments()

# Access reconstruction results
reconstruction_results = results.get("surface_reconstruction", {})
```

---

## 📈 Metrics Output

### surface_reconstruction_results.json

```json
{
  "sweep_dataset": "collected_data_runs_2025_12_15_v1",
  "num_points": 256,
  "best_model": "Random Forest",
  "metrics": {
    "Random Forest": {
      "accuracy": 0.7619,
      "classification_report": {
        "contact": {"precision": 0.82, "recall": 0.75, "f1-score": 0.78},
        "edge": {"precision": 0.68, "recall": 0.71, "f1-score": 0.69},
        "no_contact": {"precision": 0.79, "recall": 0.84, "f1-score": 0.81}
      },
      "confusion_matrix": [[120, 25, 15], [18, 92, 30], [10, 22, 108]]
    }
  }
}
```

---

## 🎯 GeometricReconstructionPipeline (Helper)

**Location:** `src/acoustic_sensing/models/geometric_reconstruction.py`

This module provides **configurable feature modes** for different reconstruction needs:

| Mode | Features | Use Case | Performance |
|------|----------|----------|-------------|
| `MINIMAL` | 2 | Real-time robotic control | <0.1ms, 96.5% |
| `OPTIMAL` | 5 | Production deployment | <0.5ms, 98.0% |
| `RESEARCH` | 8 | Maximum accuracy validation | <1.0ms, 98.0% |

### Usage Example

```python
from acoustic_sensing.models.geometric_reconstruction import GeometricReconstructionPipeline

# Initialize with optimal features
pipeline = GeometricReconstructionPipeline(feature_config="OPTIMAL")

# Train on data
result = pipeline.train(data_path="data/balanced_collected_data")

# Make predictions
prediction = pipeline.predict(features)
```

---

## 🔍 Known Limitations

1. **Currently Disabled**: Surface reconstruction is `enabled: false` in default config
2. **Requires sweep.csv**: Must have spatial coordinate data, not just raw recordings
3. **Dependency Chain**: Requires multi_dataset_training to run first
4. **2D Only**: Current implementation supports 2D surface maps (x, y coordinates)
5. **Fixed Color Scheme**: Uses hardcoded colors (blue=contact, orange=edge, green=no_contact)

---

## 💡 Improvement Opportunities

### Logic Improvements
- [ ] Support 3D surface reconstruction (z-height from confidence)
- [ ] Add spatial smoothing to reduce noisy predictions
- [ ] Implement interpolation for higher-resolution maps
- [ ] Support custom class color schemes

### Visualization Improvements
- [ ] Interactive HTML plots (Plotly)
- [ ] Animated reconstruction showing prediction propagation
- [ ] Overlay on physical surface image
- [ ] Uncertainty boundaries (contour lines)

### Performance Improvements
- [ ] Batch feature extraction (parallel processing)
- [ ] Model caching between runs
- [ ] Incremental reconstruction (new points only)

---

## 📚 Related Documentation

- [📊 Research Findings](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md) - ML analysis results
- [🔧 Data Collection Protocol](./DATA_COLLECTION_PROTOCOL.md) - Stage 1 data gathering
- [🔬 Physics First Principles](./PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md) - Theoretical foundation

---

## 🏷️ Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-XX | Initial documentation |

