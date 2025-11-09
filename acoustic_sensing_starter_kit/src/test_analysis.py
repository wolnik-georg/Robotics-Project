#!/usr/bin/env python3
"""
Quick Test Script for Geometric Discrimination Analysis
=====================================================

This script tests the analysis pipeline with a small subset of data
to verify everything works before running the full analysis.

Usage:
    python test_analysis.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geometric_data_loader import GeometricDataLoader
from feature_extraction import GeometricFeatureExtractor
from dimensionality_analysis import GeometricDimensionalityAnalyzer
from discrimination_analysis import GeometricDiscriminationAnalyzer
import numpy as np


def test_pipeline():
    """Test the analysis pipeline with a small dataset."""
    print("=" * 60)
    print("TESTING GEOMETRIC DISCRIMINATION ANALYSIS PIPELINE")
    print("=" * 60)

    try:
        # 1. Test data loading
        print("\n1. Testing data loader...")
        data_loader = GeometricDataLoader(base_dir="../data", sr=48000)

        batches = data_loader.get_available_batches()
        print(f"Available batches: {batches}")

        if not batches:
            print("ERROR: No data batches found in ../data/")
            print(
                "Make sure you have WAV files in directories like ../data/soft_finger_batch_1/data/"
            )
            return False

        # Load small sample from first batch
        batch_name = batches[0]
        audio_data, labels, metadata = data_loader.load_batch_data(
            batch_name, max_samples_per_class=5, verbose=True
        )

        if len(audio_data) == 0:
            print("ERROR: No audio data loaded")
            return False

        print(f"✓ Data loading successful: {len(audio_data)} samples")

        # 2. Test feature extraction
        print("\n2. Testing feature extraction...")
        feature_extractor = GeometricFeatureExtractor(sr=48000)

        # Test on first audio sample
        test_audio = audio_data[0]
        features = feature_extractor.extract_features(
            test_audio, method="comprehensive"
        )
        feature_names = feature_extractor.get_feature_names(method="comprehensive")

        print(f"✓ Feature extraction successful: {len(features)} features")
        print(f"  Features shape: {features.shape}")
        print(f"  Feature names: {len(feature_names)} names")

        # Extract features for all samples
        X_feat = []
        for audio in audio_data:
            feat = feature_extractor.extract_features(audio, method="comprehensive")
            X_feat.append(feat)
        X_feat = np.array(X_feat)

        # 3. Test dimensionality reduction
        print("\n3. Testing dimensionality reduction...")
        dim_analyzer = GeometricDimensionalityAnalyzer(random_state=42)

        # Test PCA
        pca_embedding, pca_results = dim_analyzer.fit_transform_pca(X_feat)
        print(f"✓ PCA successful: {pca_embedding.shape}")

        # Test t-SNE with single perplexity
        tsne_embedding, tsne_results = dim_analyzer.fit_transform_tsne(
            X_feat, perplexity=[5]
        )
        print(f"✓ t-SNE successful: {tsne_embedding.shape}")

        # 4. Test discrimination analysis
        print("\n4. Testing discrimination analysis...")
        disc_analyzer = GeometricDiscriminationAnalyzer(random_state=42)

        simplified_labels = data_loader.simplify_labels(labels)

        # Test separability analysis
        sep_results = disc_analyzer.analyze_class_separability(
            X_feat, simplified_labels, feature_names
        )
        print(f"✓ Separability analysis successful")
        print(f"  Classes found: {sep_results['n_classes']}")
        print(
            f"  Significant features: {sep_results['significant_features']}/{len(feature_names)}"
        )

        # Test classification
        cls_results = disc_analyzer.evaluate_classification_performance(
            X_feat, simplified_labels, test_multiple_classifiers=False
        )
        print(f"✓ Classification analysis successful")

        # 5. Test visualization
        print("\n5. Testing visualization...")

        # Test basic plot creation (don't save)
        fig = dim_analyzer.plot_2d_embedding(
            tsne_embedding, simplified_labels, title="Test t-SNE Plot"
        )
        print("✓ Visualization successful")

        # Close the figure
        import matplotlib.pyplot as plt

        plt.close(fig)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("The analysis pipeline is ready to use.")
        print("\nTo run the full analysis, execute:")
        print("  python run_geometric_analysis.py")
        print("")
        print("This will:")
        print("• Load all available data")
        print("• Extract comprehensive features")
        print("• Perform t-SNE and PCA analysis")
        print("• Generate statistical discrimination report")
        print("• Create publication-ready visualizations")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pipeline()
    if not success:
        print("\nPlease fix the issues above before running the full analysis.")
        sys.exit(1)
