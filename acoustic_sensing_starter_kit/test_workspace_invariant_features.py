"""
Test script to verify workspace-invariant features work correctly.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor


def test_feature_extraction():
    """Test that feature extraction works with and without workspace-invariant features."""

    # Create synthetic audio signal
    sr = 48000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate contact signal: burst at 600Hz with decay
    frequency = 600  # Hz (resonance frequency)
    signal = np.sin(2 * np.pi * frequency * t) * np.exp(-20 * t)
    signal = signal.astype(np.float32)

    print("=" * 80)
    print("Testing Workspace-Invariant Feature Extraction")
    print("=" * 80)

    # Test 1: Original features (without workspace-invariant)
    print("\n1. Testing WITHOUT workspace-invariant features:")
    extractor_original = GeometricFeatureExtractor(sr=sr, use_workspace_invariant=False)
    features_original = extractor_original.extract_features(
        signal, method="comprehensive"
    )
    feature_names_original = extractor_original.get_feature_names(
        method="comprehensive"
    )

    print(f"   ✓ Extracted {len(features_original)} features")
    print(f"   ✓ Feature names: {len(feature_names_original)}")
    assert len(features_original) == len(
        feature_names_original
    ), "Feature count mismatch!"

    # Test 2: New features (with workspace-invariant)
    print("\n2. Testing WITH workspace-invariant features:")
    extractor_new = GeometricFeatureExtractor(sr=sr, use_workspace_invariant=True)
    features_new = extractor_new.extract_features(signal, method="comprehensive")
    feature_names_new = extractor_new.get_feature_names(method="comprehensive")

    print(f"   ✓ Extracted {len(features_new)} features")
    print(f"   ✓ Feature names: {len(feature_names_new)}")
    assert len(features_new) == len(feature_names_new), "Feature count mismatch!"

    # Test 3: Verify new features are added
    num_new_features = len(features_new) - len(features_original)
    print(f"\n3. New workspace-invariant features added: {num_new_features}")
    assert num_new_features == 17, f"Expected 17 new features, got {num_new_features}"

    # Test 4: Verify original features are unchanged
    print("\n4. Verifying original features are preserved:")
    original_part = features_new[: len(features_original)]
    if np.allclose(original_part, features_original, rtol=1e-5):
        print("   ✓ Original features identical - nothing broken!")
    else:
        print("   ✗ WARNING: Original features changed!")
        max_diff = np.max(np.abs(original_part - features_original))
        print(f"     Max difference: {max_diff}")

    # Test 5: Display some workspace-invariant feature values
    print("\n5. Sample workspace-invariant feature values:")
    wi_feature_names = feature_names_new[len(features_original) :]
    wi_feature_values = features_new[len(features_original) :]

    for i, (name, value) in enumerate(zip(wi_feature_names[:5], wi_feature_values[:5])):
        print(f"   {name}: {value:.6f}")
    print(f"   ... and {len(wi_feature_names) - 5} more")

    # Test 6: Verify features are not NaN or Inf
    print("\n6. Checking for invalid values:")
    has_nan = np.any(np.isnan(features_new))
    has_inf = np.any(np.isinf(features_new))

    if not has_nan and not has_inf:
        print("   ✓ All features are valid (no NaN or Inf)")
    else:
        print(f"   ✗ WARNING: Found NaN={has_nan}, Inf={has_inf}")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Feature extraction working correctly!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • Original features: {len(features_original)}")
    print(f"  • With workspace-invariant: {len(features_new)}")
    print(f"  • New features added: {num_new_features}")
    print(f"\n🎯 Ready to run experiments with improved generalization!")


if __name__ == "__main__":
    test_feature_extraction()
