#!/usr/bin/env python3
"""Quick test to verify CNN classifiers are working."""

import numpy as np
import sys

print("Testing CNN classifier imports...")

try:
    from src.acoustic_sensing.experiments.gpu_classifiers import (
        SpectrogramCNNClassifier,
        SpectrogramCNN_MLPClassifier,
    )

    print("✓ CNN classifiers imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test with dummy data
print("\nTesting CNN classifier instantiation...")
try:
    clf1 = SpectrogramCNNClassifier(input_shape=(128, 256), max_epochs=2, verbose=False)
    clf2 = SpectrogramCNN_MLPClassifier(
        input_shape=(128, 256), max_epochs=2, verbose=False
    )
    print("✓ CNN classifiers instantiated successfully")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    sys.exit(1)

# Test with small dummy dataset
print("\nTesting CNN classifier fit/predict...")
try:
    # Create dummy spectrogram data (flattened)
    n_samples = 100
    n_mels, time_bins = 128, 256
    X = np.random.randn(n_samples, n_mels * time_bins).astype(np.float32)
    y = np.array(["contact" if i % 2 == 0 else "no_contact" for i in range(n_samples)])

    # Test Pure CNN
    print("  Testing CNN-Spectrogram...")
    clf1.fit(X[:80], y[:80])
    pred1 = clf1.predict(X[80:])
    proba1 = clf1.predict_proba(X[80:])
    print(f"    ✓ Trained on 80 samples, predicted {len(pred1)} labels")
    print(f"    ✓ Predicted classes: {set(pred1)}")
    print(f"    ✓ Probabilities shape: {proba1.shape}")

    # Test CNN + MLP Hybrid
    print("  Testing CNN-MLP-Spectrogram...")
    clf2.fit(X[:80], y[:80])
    pred2 = clf2.predict(X[80:])
    proba2 = clf2.predict_proba(X[80:])
    print(f"    ✓ Trained on 80 samples, predicted {len(pred2)} labels")
    print(f"    ✓ Predicted classes: {set(pred2)}")
    print(f"    ✓ Probabilities shape: {proba2.shape}")

    print("\n✅ All tests passed! CNN classifiers are ready to use.")

except Exception as e:
    print(f"✗ Fit/predict failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
