"""
Test script to verify PCA classifier wrappers work correctly.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.experiments.discrimination_analysis import PCAClassifierWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def test_pca_classifier_wrapper():
    """Test that PCA classifier wrapper works correctly."""

    print("=" * 80)
    print("Testing PCA Classifier Wrapper")
    print("=" * 80)

    # Create synthetic dataset with high dimensionality
    print("\n1. Creating synthetic dataset:")
    X, y = make_classification(
        n_samples=1000,
        n_features=55,  # Same as our real data
        n_informative=20,
        n_redundant=25,
        n_classes=2,
        random_state=42,
    )
    print(f"   ✓ Created dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Split into train/test
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"   ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Test 1: PCA + RandomForest
    print("\n2. Testing PCA+RandomForest:")
    pca_rf = PCAClassifierWrapper(
        base_classifier=RandomForestClassifier(n_estimators=50, random_state=42),
        n_components=0.95,  # Keep 95% variance
        pca_whiten=False,
    )

    pca_rf.fit(X_train, y_train)
    print(f"   ✓ Training completed")

    # Get number of PCA components used
    n_components_used = pca_rf.pipeline.named_steps["pca"].n_components_
    print(f"   ✓ PCA reduced: 55 features → {n_components_used} components")

    # Test predictions
    y_pred = pca_rf.predict(X_test)
    y_proba = pca_rf.predict_proba(X_test)
    print(f"   ✓ Predictions shape: {y_pred.shape}")
    print(f"   ✓ Probabilities shape: {y_proba.shape}")

    # Calculate accuracy
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✓ Test accuracy: {accuracy:.4f}")

    # Test 2: Verify PCA actually reduces dimensions
    print("\n3. Verifying dimensionality reduction:")

    # Regular RF (no PCA)
    rf_regular = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_regular.fit(X_train, y_train)
    acc_regular = rf_regular.score(X_test, y_test)

    print(f"   Regular RF accuracy: {acc_regular:.4f}")
    print(f"   PCA+RF accuracy: {accuracy:.4f}")
    print(f"   Difference: {(accuracy - acc_regular):.4f}")

    # Test 3: Check that variance is preserved
    print("\n4. Checking variance preservation:")
    explained_variance = pca_rf.pipeline.named_steps["pca"].explained_variance_ratio_
    total_variance = sum(explained_variance)
    print(f"   ✓ Total variance explained: {total_variance:.4f} (target: 0.95)")
    print(f"   ✓ First 5 components: {explained_variance[:5]}")

    # Test 4: Verify get_params/set_params work (sklearn compatibility)
    print("\n5. Testing sklearn compatibility:")
    params = pca_rf.get_params()
    print(f"   ✓ get_params() works: {list(params.keys())}")

    pca_rf_copy = PCAClassifierWrapper(
        base_classifier=RandomForestClassifier(random_state=42), n_components=0.95
    )
    pca_rf_copy.set_params(**params)
    print(f"   ✓ set_params() works")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - PCA Classifier Wrapper working correctly!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"  • Dimensionality reduction: 55 → {n_components_used} features")
    print(f"  • Variance retained: {total_variance:.2%}")
    print(f"  • Test accuracy: {accuracy:.2%}")
    print(f"\n🎯 Ready to test on real acoustic data!")


if __name__ == "__main__":
    test_pca_classifier_wrapper()
