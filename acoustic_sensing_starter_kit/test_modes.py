#!/usr/bin/env python3
"""
Test script to verify all three feature extraction modes work correctly.
"""

import sys
import numpy as np
import os

# Add src to path
sys.path.append("/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/src")

from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor


def test_feature_extraction_modes():
    """Test all three feature extraction modes."""

    # Create a dummy audio signal (1 second at 48kHz)
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Create a test signal with some frequency content
    audio = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t)

    # Spectrogram parameters
    spectrogram_params = {
        "n_fft": 2048,
        "hop_length": 256,
        "n_mels": 128,
        "fmin": 50,
        "fmax": 20000,
        "time_bins": 256,
        "use_log_scale": True,
    }

    # Initialize feature extractor
    extractor = GeometricFeatureExtractor(sr=sr)

    print("Testing feature extraction modes...")

    # Test 1: "features" mode
    print("\n1. Testing 'features' mode:")
    try:
        features = extractor.extract_features(audio, method="comprehensive")
        print(f"   ✓ Hand-crafted features: {len(features)} dimensions")
        print(
            f"   ✓ Feature values range: [{features.min():.3f}, {features.max():.3f}]"
        )
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: "spectrogram" mode
    print("\n2. Testing 'spectrogram' mode:")
    try:
        spectrogram = extractor.extract_spectrogram(audio, **spectrogram_params)
        flattened = spectrogram.flatten()
        print(f"   ✓ Spectrogram: {spectrogram.shape} → {len(flattened)} dimensions")
        print(
            f"   ✓ Spectrogram values range: [{flattened.min():.3f}, {flattened.max():.3f}]"
        )
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 3: "both" mode
    print("\n3. Testing 'both' mode:")
    try:
        result = extractor.extract_features_or_spectrogram(
            audio, mode="both", spectrogram_params=spectrogram_params
        )
        hand_features = result["features"]
        spectrogram = result["spectrogram"]
        spec_flattened = spectrogram.flatten()

        print(f"   ✓ Hand-crafted features: {len(hand_features)} dimensions")
        print(
            f"   ✓ Spectrogram: {spectrogram.shape} → {len(spec_flattened)} dimensions"
        )

        # Test simple concatenation (first 200 spectrogram elements)
        combined = np.concatenate([hand_features, spec_flattened[:200]])
        print(f"   ✓ Combined features: {len(combined)} dimensions (80 + 200)")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    print("\n✓ All three modes work correctly!")
    return True


if __name__ == "__main__":
    success = test_feature_extraction_modes()
    sys.exit(0 if success else 1)
