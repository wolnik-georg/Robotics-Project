# Guide: Running Single vs Multi-Dataset Pipeline

## 🎯 SINGLE DATASET MODE

### When to Use:
- Analyze individual datasets separately
- Get detailed metrics for one dataset
- Compare performance across different datasets independently

### How to Run:

1. **Create/Edit Config** (one per dataset):
   ```bash
   # Edit the config file
   nano configs/workspace3_squares_cutout.yml
   ```

2. **Update the dataset name**:
   ```yaml
   datasets:
     - "balanced_collected_data_runs_2025_12_15_v2_2_workspace3_squares_cutout_undersample"
   
   output:
     run_name: "workspace3_squares_cutout"  # Change this too
   ```

3. **Run the pipeline**:
   ```bash
   python3 run_modular_experiments.py configs/workspace3_squares_cutout.yml
   ```

4. **Results location**:
   ```
   single_dataset_results/workspace3_squares_cutout/
   ```

### Experiments Available:
- ✅ data_processing (feature extraction)
- ✅ dimensionality_reduction (PCA, t-SNE)
- ✅ discrimination_analysis (classifier comparison)
- ✅ saliency_analysis (feature importance)
- ✅ feature_ablation (feature selection)
- ✅ impulse_response
- ✅ frequency_band_ablation

### Key Config Settings:
```yaml
multi_dataset_training:
  enabled: false  # ❌ MUST be false for single dataset

experiments:
  multi_dataset_training:
    enabled: false  # ❌ MUST be false
  surface_reconstruction:
    enabled: false  # ❌ MUST be false (needs multi-dataset)
```

---

## 🎯 MULTI-DATASET MODE

### When to Use:
- Train on multiple datasets combined
- Test generalization to unseen workspace
- Surface reconstruction on sweep data
- Compare model performance across workspaces

### How to Run:

1. **Edit Multi-Dataset Config**:
   ```bash
   nano configs/multi_dataset_config.yml
   ```

2. **Update dataset names**:
   ```yaml
   multi_dataset_training:
     enabled: true  # ✅ ENABLE
     training_datasets:
       - "balanced_collected_data_runs_2026_01_15_workspace1_undersample"
       - "balanced_collected_data_runs_2026_01_15_workspace2_undersample"
     validation_dataset: "balanced_collected_data_runs_2026_01_14_workspace_3_v1_undersample"
   
   experiments:
     multi_dataset_training:
       enabled: true  # ✅ ENABLE
     surface_reconstruction:
       enabled: true  # ✅ ENABLE if you have sweep data
       sweep_dataset: "collected_data_runs_2026_01_14_workspace_3_v1"
   ```

3. **Run the pipeline**:
   ```bash
   python3 run_modular_experiments.py configs/multi_dataset_config.yml
   ```

4. **Results location**:
   ```
   modular_analysis_results/
   ├── dataprocessing/
   ├── multidatasettraining/
   └── surfacereconstruction/
   ```

### Experiments Available:
- ✅ data_processing (loads all 3 datasets)
- ✅ multi_dataset_training (combines datasets, trains models)
- ✅ surface_reconstruction (maps predictions to spatial coordinates)

### Key Config Settings:
```yaml
multi_dataset_training:
  enabled: true  # ✅ MUST be true
  training_datasets: [...]  # 2+ datasets to combine
  validation_dataset: "..."  # 1 holdout dataset
  train_test_split: 0.8

experiments:
  multi_dataset_training:
    enabled: true  # ✅ MUST be true
  surface_reconstruction:
    enabled: true  # ✅ Optional (needs sweep.csv)
    sweep_dataset: "..."  # Must have sweep.csv file
```

---

## 📋 QUICK REFERENCE

### Run Single Dataset:
```bash
# Option 1: Use template and modify
cp configs/single_dataset_template.yml configs/my_dataset.yml
nano configs/my_dataset.yml  # Edit dataset name
python3 run_modular_experiments.py configs/my_dataset.yml

# Option 2: Use existing config
python3 run_modular_experiments.py configs/workspace3_squares_cutout.yml
```

### Run Multi-Dataset:
```bash
python3 run_modular_experiments.py configs/multi_dataset_config.yml
```

### Process Multiple Single Datasets (Batch):
```bash
# Process dataset 1
python3 run_modular_experiments.py configs/workspace1.yml

# Process dataset 2
python3 run_modular_experiments.py configs/workspace2.yml

# Process dataset 3
python3 run_modular_experiments.py configs/workspace3.yml
```

---

## 🔄 WORKFLOW EXAMPLES

### Example 1: Analyze 3 New Datasets Individually
```bash
# 1. Create configs for each
cp configs/single_dataset_template.yml configs/dataset1.yml
cp configs/single_dataset_template.yml configs/dataset2.yml
cp configs/single_dataset_template.yml configs/dataset3.yml

# 2. Edit each config (change dataset name and run_name)

# 3. Run each
python3 run_modular_experiments.py configs/dataset1.yml
python3 run_modular_experiments.py configs/dataset2.yml
python3 run_modular_experiments.py configs/dataset3.yml

# 4. Results in:
# single_dataset_results/dataset1/
# single_dataset_results/dataset2/
# single_dataset_results/dataset3/
```

### Example 2: Multi-Dataset Training + Surface Reconstruction
```bash
# 1. Update multi_dataset_config.yml with your 3 datasets
# 2. Run multi-dataset pipeline
python3 run_modular_experiments.py configs/multi_dataset_config.yml

# 3. Results in:
# modular_analysis_results/
#   ├── multidatasettraining/  # Model training results
#   └── surfacereconstruction/  # Spatial maps
```

---

## ⚠️ IMPORTANT NOTES

1. **Dataset Names**: Must match folder names in `data/`
2. **Sweep Data**: Only needed for surface_reconstruction
3. **Single vs Multi**: Cannot have both enabled simultaneously
4. **Balanced Data**: Always use balanced datasets (undersample/oversample)
5. **Output Directories**: Single and multi use different output directories

---

## 🎨 WHAT YOU GET

### Single Dataset Output:
- Class distribution plots
- PCA/t-SNE visualizations
- Confusion matrices
- Classification reports
- Feature importance (if enabled)

### Multi-Dataset Output:
- **Training plots**: Performance comparison across 6 models
- **Confusion matrices**: Test and validation sets
- **Generalization analysis**: Test vs validation accuracy
- **Dataset visualizations**: PCA/t-SNE showing dataset clustering
- **Surface maps**: Ground truth, predictions, confidence, errors (if sweep data available)

