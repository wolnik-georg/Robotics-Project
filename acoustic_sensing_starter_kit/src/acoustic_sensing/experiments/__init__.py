# Experiment Module Initialization
from .base_experiment import BaseExperiment
from .orchestrator import ExperimentOrchestrator

# Import all experiment classes
from .data_processing import DataProcessingExperiment
from .dimensionality_reduction import DimensionalityReductionExperiment
from .discrimination_analysis import DiscriminationAnalysisExperiment
from .saliency_analysis import SaliencyAnalysisExperiment
from .feature_ablation import FeatureAblationExperiment
from .impulse_response import ImpulseResponseExperiment
from .frequency_band_ablation import FrequencyBandAblationExperiment
from .multi_dataset_training import MultiDatasetTrainingExperiment
from .surface_reconstruction import SurfaceReconstructionExperiment

__all__ = [
    "BaseExperiment",
    "ExperimentOrchestrator",
    "DataProcessingExperiment",
    "DimensionalityReductionExperiment",
    "DiscriminationAnalysisExperiment",
    "SaliencyAnalysisExperiment",
    "FeatureAblationExperiment",
    "ImpulseResponseExperiment",
    "FrequencyBandAblationExperiment",
    "MultiDatasetTrainingExperiment",
    "SurfaceReconstructionExperiment",
]
