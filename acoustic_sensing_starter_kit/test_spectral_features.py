#!/usr/bin/env python3
"""
Test script for new spectral feature extraction methods.

This script validates that the newly added MFCC, magnitude spectrum,
power spectrum, and chroma feature extraction methods work correctly.
"""

import numpy as np
import librosa
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor


def create_test_audio(sr=48000, duration=0.1, freq=1000):
    """Create a simple test audio signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Create a sine wave with some noise
    audio = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    return audio.astype(np.float32)


def test_spectral_features():
    """Test all new spectral feature extraction methods."""
    print("Testing new spectral feature extraction methods...")

    # Create test audio
    sr = 48000
    audio = create_test_audio(sr=sr)

    print(f"Test audio shape: {audio.shape}")
    print(f"Test audio duration: {len(audio)/sr:.3f} seconds")

    # Initialize feature extractor
    extractor = GeometricFeatureExtractor(sr=sr)

    # Test MFCC extraction
    print("\n1. Testing MFCC extraction...")
    try:
        mfcc = extractor.extract_mfcc(audio, n_mfcc=13)
        print(f"   MFCC shape: {mfcc.shape}")
        print(f"   MFCC range: [{mfcc.min():.3f}, {mfcc.max():.3f}]")
        assert (
            mfcc.shape[0] == 13
        ), f"Expected 13 MFCC coefficients, got {mfcc.shape[0]}"
        print("   ✓ MFCC extraction successful")
    except Exception as e:
        print(f"   ✗ MFCC extraction failed: {e}")
        return False

    # Test magnitude spectrum
    print("\n2. Testing magnitude spectrum extraction...")
    try:
        mag_spec = extractor.extract_magnitude_spectrum(audio)
        print(f"   Magnitude spectrum shape: {mag_spec.shape}")
        print(
            f"   Magnitude spectrum range: [{mag_spec.min():.3f}, {mag_spec.max():.3f}]"
        )
        expected_freq_bins = 2048 // 2 + 1  # Default n_fft=2048
        assert (
            mag_spec.shape[0] == expected_freq_bins
        ), f"Expected {expected_freq_bins} frequency bins, got {mag_spec.shape[0]}"
        print("   ✓ Magnitude spectrum extraction successful")
    except Exception as e:
        print(f"   ✗ Magnitude spectrum extraction failed: {e}")
        return False

    # Test power spectrum
    print("\n3. Testing power spectrum extraction...")
    try:
        power_spec = extractor.extract_power_spectrum(audio)
        print(f"   Power spectrum shape: {power_spec.shape}")
        print(
            f"   Power spectrum range: [{power_spec.min():.6f}, {power_spec.max():.6f}]"
        )
        expected_freq_bins = 2048 // 2 + 1  # Default n_fft=2048
        assert (
            power_spec.shape[0] == expected_freq_bins
        ), f"Expected {expected_freq_bins} frequency bins, got {power_spec.shape[0]}"
        print("   ✓ Power spectrum extraction successful")
    except Exception as e:
        print(f"   ✗ Power spectrum extraction failed: {e}")
        return False

    # Test chroma features
    print("\n4. Testing chroma feature extraction...")
    try:
        chroma = extractor.extract_chroma_features(audio)
        print(f"   Chroma features shape: {chroma.shape}")
        print(f"   Chroma features range: [{chroma.min():.3f}, {chroma.max():.3f}]")
        assert chroma.shape[0] == 12, f"Expected 12 chroma bins, got {chroma.shape[0]}"
        print("   ✓ Chroma feature extraction successful")
    except Exception as e:
        print(f"   ✗ Chroma feature extraction failed: {e}")
        return False

    # Test with different parameters
    print("\n5. Testing with custom parameters...")
    try:
        # Custom MFCC
        mfcc_custom = extractor.extract_mfcc(audio, n_mfcc=20, n_fft=1024)
        print(f"   Custom MFCC shape: {mfcc_custom.shape}")
        assert (
            mfcc_custom.shape[0] == 20
        ), f"Expected 20 MFCC coefficients, got {mfcc_custom.shape[0]}"

        # Custom magnitude spectrum
        mag_spec_custom = extractor.extract_magnitude_spectrum(
            audio, n_fft=1024, normalize=False
        )
        print(f"   Custom magnitude spectrum shape: {mag_spec_custom.shape}")
        expected_freq_bins = 1024 // 2 + 1
        assert (
            mag_spec_custom.shape[0] == expected_freq_bins
        ), f"Expected {expected_freq_bins} frequency bins, got {mag_spec_custom.shape[0]}"

        print("   ✓ Custom parameter testing successful")
    except Exception as e:
        print(f"   ✗ Custom parameter testing failed: {e}")
        return False

    print("\n✓ All spectral feature extraction methods working correctly!")
    return True


if __name__ == "__main__":
    success = test_spectral_features()
    sys.exit(0 if success else 1)
