"""
Feature extraction and analysis

This module contains advanced feature extraction capabilities:
- Optimized feature sets for different use cases
- Feature ablation analysis for validation
- Saliency analysis for ML interpretability
"""

from .optimized_sets import OptimizedFeatureExtractor, FeatureSetConfig

# Note: Other imports commented out until class names are verified
# from .ablation_analysis import FeatureAblationAnalyzer
# from .saliency_analysis import SaliencyAnalyzer

__all__ = [
    'OptimizedFeatureExtractor',
    'FeatureSetConfig'
    # 'FeatureAblationAnalyzer',
    # 'SaliencyAnalyzer'
]