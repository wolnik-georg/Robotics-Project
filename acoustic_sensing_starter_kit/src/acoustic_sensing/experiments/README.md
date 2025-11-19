# Modular Acoustic Sensing Experiments

This directory contains the modular experiment framework that replaces the monolithic batch analysis approach. Each experiment is implemented as a separate module with clear dependencies and interfaces.

## Architecture

### Base Framework
- **`base_experiment.py`** - Abstract base class defining the common interface for all experiments
- **`orchestrator.py`** - Manages experiment execution, dependencies, and result aggregation
- **`__init__.py`** - Module initialization and exports

### Experiment Modules
Each experiment implements the `BaseExperiment` interface and focuses on a specific analysis:

1. **`data_processing.py`** - Foundation experiment that loads and preprocesses data
2. **`dimensionality_reduction.py`** - PCA and t-SNE analysis for data visualization
3. **`discrimination_analysis.py`** - Multi-classifier material discrimination analysis
4. **`saliency_analysis.py`** - Neural network-based feature importance analysis
5. **`feature_ablation.py`** - Systematic feature importance testing through ablation
6. **`impulse_response.py`** - Deconvolution and transfer function analysis
7. **`frequency_band_ablation.py`** - Frequency-specific contribution analysis

## Usage

### Quick Start
```bash
# Run with default configuration
python run_modular_experiments.py

# Run with custom config
python run_modular_experiments.py configs/my_config.yml

# Validate configuration only
python run_modular_experiments.py --validate-only

# List available experiments
python run_modular_experiments.py --list-experiments
```

### Configuration
Experiments are configured via YAML files (e.g., `configs/experiment_config.yml`):

```yaml
experiments:
  discrimination_analysis:
    enabled: true
    test_multiple_classifiers: true
    include_lda: true
  
  saliency_analysis:
    enabled: true
    neural_network_epochs: 100
    gradient_methods: ["basic", "integrated"]
  
  feature_ablation:
    enabled: false  # Disable slow experiments
```

## Experiment Dependencies

The framework automatically resolves dependencies:

```
data_processing (base)
├── dimensionality_reduction
├── discrimination_analysis
├── saliency_analysis
├── feature_ablation
├── impulse_response
└── frequency_band_ablation
```

## Key Features

### Modular Design
- Each experiment is self-contained with clear inputs/outputs
- Easy to add new experiments or modify existing ones
- Clean separation of concerns

### Dependency Management
- Automatic dependency resolution and execution ordering
- Shared data passing between experiments
- Graceful handling of failed dependencies

### Configuration-Driven
- Enable/disable experiments via configuration
- Fine-tune experiment parameters
- Multiple configuration profiles

### Comprehensive Output
- Each experiment saves its own results and visualizations
- Unified summary with key findings
- Detailed execution logs and error handling

### Preserved Functionality
- All original analysis capabilities maintained
- Improved organization and maintainability
- Enhanced error handling and logging

## Output Structure

```
modular_analysis_results/
├── execution_summary.json          # Overall execution summary
├── experiment_config_used.yml      # Configuration used
├── full_results.pkl               # Complete results (for analysis)
├── data_processing/               # Data loading results
├── discrimination_analysis/       # Classifier comparisons
├── saliency_analysis/            # Feature importance analysis
├── feature_ablation/             # Ablation study results
└── [other_experiments]/          # Additional experiment outputs
```

## Migration from Monolithic Code

This modular framework replaces `batch_analysis.py` while preserving all functionality:

- **Before**: Single large file with intertwined analysis code
- **After**: Clean modular architecture with dependency management
- **Benefits**: Better maintainability, easier testing, flexible execution

## Adding New Experiments

1. Create new experiment class inheriting from `BaseExperiment`
2. Implement required methods: `get_dependencies()` and `run()`
3. Register in `orchestrator.py`
4. Add configuration options
5. Update this documentation

Example:
```python
from .base_experiment import BaseExperiment

class MyNewExperiment(BaseExperiment):
    def get_dependencies(self) -> List[str]:
        return ['data_processing']
    
    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        # Your analysis here
        return results
```