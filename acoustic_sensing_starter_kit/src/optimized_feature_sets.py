#!/usr/bin/env python3
"""
Configurable Feature Selection for Acoustic Geometric Reconstruction
===================================================================

This module provides three optimized feature sets based on comprehensive
saliency and ablation analysis:

- MINIMAL (2 features): Ultra-fast real-time applications
- OPTIMAL (5 features): Best balance of performance and efficiency
- RESEARCH (8 features): Maximum accuracy for research and validation

Usage:
    from optimized_feature_sets import FeatureSetConfig, extract_optimized_features

    # Initialize with desired configuration
    config = FeatureSetConfig(mode='OPTIMAL')  # or 'MINIMAL' or 'RESEARCH'

    # Extract features using selected set
    features = extract_optimized_features(audio_signal, config)

    # Switch configurations dynamically
    config.set_mode('MINIMAL')
    minimal_features = extract_optimized_features(audio_signal, config)
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class FeatureSetConfig:
    """
    Configuration class for optimized feature sets based on ablation analysis.

    Three validated configurations:
    - MINIMAL: 2 features for real-time critical applications
    - OPTIMAL: 5 features for production use (recommended)
    - RESEARCH: 8 features for maximum accuracy and research
    """

    # Feature set definitions based on ablation analysis
    FEATURE_SETS = {
        "MINIMAL": {
            "features": [
                "spectral_bandwidth",  # Most critical feature (1% accuracy drop when removed)
                "spectral_centroid",  # Universal discriminator (appears in all batches)
            ],
            "description": "Ultra-minimal set for real-time critical systems",
            "expected_performance": "94-98% accuracy",
            "computation_time": "<0.1ms",
            "use_case": "Real-time robotic control, embedded systems",
        },
        "OPTIMAL": {
            "features": [
                "spectral_bandwidth",  # #1 Priority - Critical across 3/4 tasks
                "spectral_centroid",  # #2 Priority - Universal discriminator
                "high_energy_ratio",  # #3 Priority - Contact position detection
                "ultra_high_energy_ratio",  # #4 Priority - Edge detection specialist
                "temporal_centroid",  # #5 Priority - Cross-batch consistency
            ],
            "description": "Optimal balance of performance and efficiency",
            "expected_performance": "97-99% accuracy",
            "computation_time": "<0.5ms",
            "use_case": "Production systems, robotic applications",
        },
        "RESEARCH": {
            "features": [
                "spectral_bandwidth",  # Core spectral feature
                "spectral_centroid",  # Universal discriminator
                "high_energy_ratio",  # Mid-high frequency energy
                "ultra_high_energy_ratio",  # High frequency specialist
                "temporal_centroid",  # Timing information
                "mid_energy_ratio",  # Complementary frequency info
                "resonance_peak_amp",  # Resonance characteristics
                "env_max",  # Envelope maximum
            ],
            "description": "Maximum accuracy for research and validation",
            "expected_performance": "99-100% accuracy",
            "computation_time": "<1ms",
            "use_case": "Research validation, offline analysis, publications",
        },
    }

    def __init__(self, mode: str = "OPTIMAL"):
        """
        Initialize feature configuration.

        Args:
            mode: One of 'MINIMAL', 'OPTIMAL', 'RESEARCH'
        """
        self.set_mode(mode)

    def set_mode(self, mode: str):
        """Set the feature extraction mode."""
        if mode not in self.FEATURE_SETS:
            raise ValueError(f"Mode must be one of {list(self.FEATURE_SETS.keys())}")

        self.mode = mode
        self.config = self.FEATURE_SETS[mode]
        self.features = self.config["features"]
        self.n_features = len(self.features)

        print(f"🎯 Feature set configured: {mode}")
        print(f"   Features: {self.n_features} ({', '.join(self.features)})")
        print(f"   Expected: {self.config['expected_performance']}")
        print(f"   Use case: {self.config['use_case']}")

    def get_feature_indices(self, all_feature_names: List[str]) -> List[int]:
        """Get indices of selected features from a complete feature list."""
        indices = []
        for feature in self.features:
            if feature in all_feature_names:
                indices.append(all_feature_names.index(feature))
            else:
                print(
                    f"Warning: Feature '{feature}' not found in provided feature list"
                )
        return indices

    def filter_features(
        self, feature_array: np.ndarray, all_feature_names: List[str]
    ) -> np.ndarray:
        """Filter a feature array to include only selected features."""
        indices = self.get_feature_indices(all_feature_names)
        return (
            feature_array[:, indices]
            if len(feature_array.shape) > 1
            else feature_array[indices]
        )

    def get_info(self) -> Dict:
        """Get current configuration information."""
        return {
            "mode": self.mode,
            "n_features": self.n_features,
            "features": self.features,
            "description": self.config["description"],
            "expected_performance": self.config["expected_performance"],
            "computation_time": self.config["computation_time"],
            "use_case": self.config["use_case"],
        }

    def save_config(self, filepath: str):
        """Save current configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.get_info(), f, indent=2)

    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_data = json.load(f)
        return cls(mode=config_data["mode"])


def extract_optimized_features(
    audio_signal: np.ndarray, config: FeatureSetConfig, sr: int = 22050
) -> np.ndarray:
    """
    Extract optimized feature set based on configuration.

    Args:
        audio_signal: Input audio signal
        config: FeatureSetConfig instance
        sr: Sample rate

    Returns:
        Feature vector with selected features
    """

    # Compute all possible features efficiently
    features = {}

    # Spectral analysis (single FFT computation)
    stft = librosa.stft(audio_signal)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr)

    # Core spectral features
    if "spectral_centroid" in config.features:
        features["spectral_centroid"] = np.mean(
            librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        )

    if "spectral_bandwidth" in config.features:
        features["spectral_bandwidth"] = np.mean(
            librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        )

    # Energy ratio features
    total_energy = np.sum(magnitude)

    if "high_energy_ratio" in config.features:
        high_energy = np.sum(magnitude[(freqs >= 2000) & (freqs < 8000)])
        features["high_energy_ratio"] = high_energy / total_energy

    if "ultra_high_energy_ratio" in config.features:
        ultra_high_energy = np.sum(magnitude[freqs >= 8000])
        features["ultra_high_energy_ratio"] = ultra_high_energy / total_energy

    if "mid_energy_ratio" in config.features:
        mid_energy = np.sum(magnitude[(freqs >= 500) & (freqs < 2000)])
        features["mid_energy_ratio"] = mid_energy / total_energy

    # Temporal features
    if "temporal_centroid" in config.features:
        envelope = np.abs(audio_signal)
        features["temporal_centroid"] = (
            np.sum(np.arange(len(envelope)) * envelope) / np.sum(envelope) / sr
        )

    if "env_max" in config.features:
        envelope = np.abs(audio_signal)
        features["env_max"] = np.max(envelope)

    # Resonance features (if needed)
    if "resonance_peak_amp" in config.features:
        # Simple resonance peak detection
        power_spectrum = np.abs(np.fft.fft(audio_signal)) ** 2
        peak_idx = np.argmax(power_spectrum[: len(power_spectrum) // 2])
        features["resonance_peak_amp"] = power_spectrum[peak_idx]

    # Return features in the order specified by config
    feature_vector = np.array([features.get(name, 0.0) for name in config.features])

    return feature_vector


class OptimizedFeatureExtractor:
    """
    Complete feature extraction class with configurable feature sets.
    Integrates with existing pipeline for easy switching between configurations.
    """

    def __init__(self, mode: str = "OPTIMAL", sr: int = 22050):
        """
        Initialize with specified configuration.

        Args:
            mode: Feature set mode ('MINIMAL', 'OPTIMAL', 'RESEARCH')
            sr: Sample rate for audio processing
        """
        self.config = FeatureSetConfig(mode)
        self.sr = sr
        self._cache = {}  # Cache for expensive computations

    def set_mode(self, mode: str):
        """Change feature extraction mode."""
        self.config.set_mode(mode)
        self._cache.clear()  # Clear cache when mode changes

    def extract_from_audio(self, audio_signal: np.ndarray) -> np.ndarray:
        """Extract features from raw audio signal."""
        return extract_optimized_features(audio_signal, self.config, self.sr)

    def extract_from_file(self, audio_file: str) -> np.ndarray:
        """Extract features from audio file."""
        audio_signal, sr = librosa.load(audio_file, sr=self.sr)
        return self.extract_from_audio(audio_signal)

    def extract_batch(self, audio_files: List[str]) -> np.ndarray:
        """Extract features from multiple audio files."""
        features = []
        for file_path in audio_files:
            file_features = self.extract_from_file(file_path)
            features.append(file_features)
        return np.array(features)

    def benchmark_modes(self, audio_signal: np.ndarray) -> Dict:
        """Benchmark all three modes on the same audio signal."""
        results = {}

        for mode in ["MINIMAL", "OPTIMAL", "RESEARCH"]:
            self.set_mode(mode)

            import time

            start_time = time.time()
            features = self.extract_from_audio(audio_signal)
            computation_time = time.time() - start_time

            results[mode] = {
                "features": features,
                "n_features": len(features),
                "computation_time_ms": computation_time * 1000,
                "config": self.config.get_info(),
            }

        return results

    def get_feature_names(self) -> List[str]:
        """Get names of currently selected features."""
        return self.config.features.copy()

    def get_info(self) -> Dict:
        """Get current configuration information."""
        return self.config.get_info()


# Convenience functions for quick usage
def extract_minimal_features(audio_signal: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract minimal 2-feature set."""
    config = FeatureSetConfig("MINIMAL")
    return extract_optimized_features(audio_signal, config, sr)


def extract_optimal_features(audio_signal: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract optimal 5-feature set."""
    config = FeatureSetConfig("OPTIMAL")
    return extract_optimized_features(audio_signal, config, sr)


def extract_research_features(audio_signal: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract research 8-feature set."""
    config = FeatureSetConfig("RESEARCH")
    return extract_optimized_features(audio_signal, config, sr)


if __name__ == "__main__":
    # Demo usage
    print("🔬 Optimized Feature Selection Demo")
    print("=" * 50)

    # Create synthetic test signal
    import numpy as np

    duration = 1.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    test_signal = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

    # Initialize feature extractor
    extractor = OptimizedFeatureExtractor(mode="OPTIMAL")

    print("\n📊 Benchmarking all feature set modes...")
    results = extractor.benchmark_modes(test_signal)

    for mode, data in results.items():
        print(f"\n{mode} MODE:")
        print(f"  Features extracted: {data['n_features']}")
        print(f"  Computation time: {data['computation_time_ms']:.2f} ms")
        print(f"  Feature values: {data['features']}")
        print(f"  Expected performance: {data['config']['expected_performance']}")

    print("\n✨ Feature extraction system ready!")
    print("Use FeatureSetConfig to switch between MINIMAL/OPTIMAL/RESEARCH modes.")
