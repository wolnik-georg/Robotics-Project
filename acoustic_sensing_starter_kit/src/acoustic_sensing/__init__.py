"""
Acoustic Sensing for Geometric Reconstruction

A comprehensive acoustic sensing system for real-time geometric reconstruction
with optimized feature extraction, advanced saliency analysis, and production-ready
real-time capabilities.

Key Components:
- Core: Feature extraction, preprocessing, data management
- Features: Optimized feature sets, ablation analysis, saliency analysis
- Models: Training pipelines, geometric reconstruction
- Sensors: Real-time sensing system
- Analysis: Batch analysis, discrimination analysis
- Visualization: Publication-quality plots and visualizations
- Demo: Integrated demonstration scripts
- Legacy: Original A/B/C record/train/sense scripts

Usage:
    from acoustic_sensing.features import OptimizedFeatureExtractor
    from acoustic_sensing.sensors import OptimizedRealTimeSensor
    from acoustic_sensing.models import ConfigurableTrainingPipeline
"""

__version__ = "1.0.0"
__author__ = "Georg Wolnik"
__email__ = "georg.wolnik@example.com"

# Core imports for easy access
from .core.feature_extraction import extract_features, AudioFeatureExtractor
from .core.preprocessing import preprocess_audio_signal
from .features.optimized_sets import OptimizedFeatureExtractor, FeatureSetConfig
from .sensors.real_time_sensor import OptimizedRealTimeSensor
from .sensors.sensor_config import SensorConfig  
from .models.training import ConfigurableTrainingPipeline

__all__ = [
    'extract_features',
    'AudioFeatureExtractor',
    'preprocess_audio_signal',
    'OptimizedFeatureExtractor',
    'FeatureSetConfig',
    'OptimizedRealTimeSensor',
    'SensorConfig',
    'ConfigurableTrainingPipeline',
]