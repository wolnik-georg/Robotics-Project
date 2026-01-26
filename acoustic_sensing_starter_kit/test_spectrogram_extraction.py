#!/usr/bin/env python3
"""
Test script to verify spectrogram extraction works correctly.

This tests the new Phase 1 implementation:
1. Extract hand-crafted features (default mode)
2. Extract spectrograms (new mode)
3. Extract both (hybrid mode)
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor


def test_feature_extraction_modes():
    """Test all three extraction modes."""

    print("=" * 80)
    print("TESTING SPECTROGRAM EXTRACTION (Phase 1)")
    print("=" * 80)

    # Create synthetic audio signal (1 second at 48kHz)
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate contact event: burst + decay
    frequency = 500  # Hz (resonance frequency)
    audio = np.sin(2 * np.pi * frequency * t) * np.exp(-5 * t)
    audio = audio.astype(np.float32)

    print(f"\n✓ Created synthetic audio: {len(audio)} samples @ {sr} Hz")

    # Initialize extractor
    extractor = GeometricFeatureExtractor(
        sr=sr,
        use_workspace_invariant=True,
        use_impulse_features=True,
        use_contact_physics_features=True,
    )

    print("\n" + "=" * 80)
    print("TEST 1: Hand-Crafted Features (mode='features')")
    print("=" * 80)

    features = extractor.extract_features_or_spectrogram(audio, mode="features")
    print(f"✓ Features extracted: shape={features.shape}, dtype={features.dtype}")
    print(f"  Feature vector: {len(features)} dimensions")
    print(
        f"  Sample values: [{features[0]:.4f}, {features[1]:.4f}, ..., {features[-1]:.4f}]"
    )

    print("\n" + "=" * 80)
    print("TEST 2: Mel Spectrogram (mode='spectrogram')")
    print("=" * 80)

    spectrogram_params = {
        "n_fft": 512,
        "hop_length": 128,
        "n_mels": 64,
        "fmin": 0,
        "fmax": 8000,
        "time_bins": 128,
        "use_log_scale": True,
    }

    spectrogram = extractor.extract_features_or_spectrogram(
        audio, mode="spectrogram", spectrogram_params=spectrogram_params
    )
    print(
        f"✓ Spectrogram extracted: shape={spectrogram.shape}, dtype={spectrogram.dtype}"
    )
    print(f"  Frequency bins: {spectrogram.shape[0]} (mel-scaled)")
    print(f"  Time bins: {spectrogram.shape[1]}")
    print(f"  Flattened size: {spectrogram.flatten().shape[0]} dimensions")
    print(f"  Value range: [{spectrogram.min():.2f}, {spectrogram.max():.2f}] dB")

    print("\n" + "=" * 80)
    print("TEST 3: Both Features + Spectrogram (mode='both')")
    print("=" * 80)

    both = extractor.extract_features_or_spectrogram(
        audio, mode="both", spectrogram_params=spectrogram_params
    )
    print(f"✓ Both extracted: {type(both)}")
    print(f"  Features: shape={both['features'].shape}")
    print(f"  Spectrogram: shape={both['spectrogram'].shape}")

    # Test concatenation (as used in data_processing)
    combined = np.concatenate([both["features"], both["spectrogram"].flatten()])
    print(
        f"  Combined: {len(combined)} dimensions ({len(both['features'])} + {both['spectrogram'].size})"
    )

    print("\n" + "=" * 80)
    print("TEST 4: Backward Compatibility")
    print("=" * 80)

    # Test old API still works
    old_features = extractor.extract_features(audio, method="comprehensive")
    print(f"✓ Old API works: shape={old_features.shape}")

    # Verify same as new API with mode="features"
    assert np.allclose(
        features, old_features
    ), "Feature mismatch between old and new API!"
    print(f"✓ New API (mode='features') matches old API ✅")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nPhase 1 Implementation Complete:")
    print("  ✅ Spectrogram extraction added")
    print("  ✅ Mode toggle working (features/spectrogram/both)")
    print("  ✅ Backward compatibility maintained")
    print("  ✅ Ready for pipeline integration")
    print("\nNext Steps:")
    print("  1. Run pipeline with mode='features' (default, should work as before)")
    print("  2. Run pipeline with mode='spectrogram' (test new mode)")
    print("  3. Compare accuracies: features vs spectrogram")
    print("=" * 80)


if __name__ == "__main__":
    test_feature_extraction_modes()
