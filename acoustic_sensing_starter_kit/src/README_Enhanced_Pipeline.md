# Enhanced Acoustic Geometric Discrimination Analysis

This directory contains a comprehensive ML and data pipeline for acoustic sensing, significantly improved and extended from your original t-SNE analysis. The pipeline includes robust feature extraction, dimensionality reduction, statistical analysis, and publication-ready visualizations.

## What's New

Your original t-SNE analysis:
```python
# Original approach
X_feat = np.array([extract_features(np.load(f"exploration_data/{f}")) for f in os.listdir("exploration_data")])
y_labels = [f.split('_')[1] + "_" + f.split('_')[2] for f in os.listdir("exploration_data")]
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_feat)
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels)
```

Enhanced pipeline:
- ✅ **Robust data loading** from your existing WAV files
- ✅ **Comprehensive feature extraction** optimized for geometric discrimination
- ✅ **Multiple dimensionality reduction** techniques (t-SNE, PCA, UMAP)
- ✅ **Statistical discrimination analysis** with significance testing
- ✅ **Publication-ready visualizations** with proper styling
- ✅ **Comprehensive reporting** proving discrimination capability

## Quick Start

### 1. Test the Pipeline
```bash
cd src
python test_analysis.py
```
This runs a quick test with a small subset of your data to verify everything works.

### 2. Run Full Analysis
```bash
cd src
python run_geometric_analysis.py
```
This performs the complete analysis on all your recorded data.

## Expected Output

The pipeline will create a `geometric_discrimination_results/` directory with:

### Main Results
- `tsne_geometric_separability.png` - Enhanced version of your original t-SNE plot
- `discrimination_analysis_report.txt` - Statistical proof of discrimination capability
- `geometric_discrimination_data.csv` - t-SNE and PCA coordinates with labels

### Additional Analysis
- `tsne_vs_pca_comparison.png` - Side-by-side comparison
- `tsne_perplexity_comparison.png` - Multiple perplexity values tested
- `extracted_features.csv` - All extracted features for further analysis
- `analysis_summary.json` - Summary statistics

## Key Features

### 1. Enhanced Feature Extraction (`feature_extraction.py`)
- **Resonance analysis** (500-800Hz chamber resonance)
- **High-frequency damping** patterns (>1kHz)
- **Contact burst characteristics**
- **Spectral shape descriptors**
- **Multi-band energy analysis**

Based on research: Contact deforms finger chamber, damping high frequencies and shifting resonances.

### 2. Robust Data Loading (`geometric_data_loader.py`)
- Automatically loads from your `soft_finger_batch_*` directories
- Handles multiple batches and contact positions
- Validates and standardizes audio lengths
- Creates proper labels for analysis

### 3. Comprehensive Dimensionality Reduction (`dimensionality_analysis.py`)
- **t-SNE** with multiple perplexity values (including your original 30)
- **PCA** with automatic component selection
- **UMAP** support (if installed)
- **Separability metrics** for each method

### 4. Statistical Discrimination Analysis (`discrimination_analysis.py`)
- **ANOVA testing** for feature significance
- **Classification performance** with multiple algorithms
- **Feature importance** and stability analysis
- **Separability metrics** (silhouette score, Fisher ratio)

## Understanding Results

### Discrimination Evidence
The pipeline looks for multiple forms of evidence:

1. **Statistical significance**: Are features significantly different between classes?
2. **Classification accuracy**: Can ML algorithms distinguish the geometries?
3. **Visual separability**: Are clusters clearly separated in t-SNE/PCA?
4. **Feature consistency**: Are important features stable across bootstrap samples?

### Reading the Report
The final report will conclude with either:
- ✅ **"GEOMETRIC DISCRIMINATION CAPABILITY CONFIRMED"** - Strong evidence
- ⚠️ **"GEOMETRIC DISCRIMINATION CAPABILITY UNCERTAIN"** - Weak evidence

## Customization

### Analyzing Specific Batches
```python
from run_geometric_analysis import run_geometric_discrimination_analysis

results = run_geometric_discrimination_analysis(
    specific_batches=['soft_finger_batch_1', 'soft_finger_batch_2']
)
```

### Testing with Limited Data
```python
results = run_geometric_discrimination_analysis(
    max_samples_per_class=10  # Only 10 samples per contact position
)
```

### Custom Feature Extraction
```python
from feature_extraction import GeometricFeatureExtractor

extractor = GeometricFeatureExtractor(sr=48000)
features = extractor.extract_features(audio, method='resonance')  # Focus on resonance
```

## Data Requirements

Your existing data structure is perfect:
```
data/
├── soft_finger_batch_1/
│   └── data/
│       ├── 1_finger tip.wav
│       ├── 1_finger middle.wav
│       ├── 1_finger bottom.wav
│       └── 1_finger blank.wav
├── soft_finger_batch_2/
└── ...
```

The pipeline automatically:
- Detects all available batches
- Extracts contact positions from filenames
- Loads and standardizes audio data
- Creates proper labels for analysis

## Troubleshooting

### Common Issues
1. **"No data batches found"** - Check that WAV files exist in `../data/soft_finger_batch_*/data/`
2. **"Feature extraction failed"** - Audio files may be corrupted or too short
3. **"t-SNE failed"** - Too few samples or perplexity too high for dataset size

### Dependencies
Required packages (install with pip):
```bash
pip install numpy pandas matplotlib scikit-learn librosa scipy
```

Optional (for enhanced analysis):
```bash
pip install umap-learn seaborn
```

## Contact Positions Analyzed

The pipeline automatically detects and analyzes:
- **tip** (finger tip contact)
- **middle** (finger middle contact) 
- **base** (finger bottom contact)
- **none** (finger blank, no contact)

This maps to your geometric discrimination goals of distinguishing contact locations on the acoustic finger sensor.

---

This enhanced pipeline provides rigorous statistical proof of geometric discrimination capability in your acoustic sensing data, going far beyond the original t-SNE visualization to provide comprehensive evidence for publication or further development.