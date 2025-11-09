#!/usr/bin/env python3
"""
Comprehensive Acoustic Geometric Discrimination Analysis
======================================================

This script performs a complete analysis to prove geometric discrimination
capability in acoustic sensing data, similar to your original t-SNE example
but significantly enhanced.

Usage:
    python geometric_discrimination_analysis.py

Author: Enhanced ML Pipeline for Acoustic Sensing
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geometric_data_loader import GeometricDataLoader, print_dataset_summary
from feature_extraction import GeometricFeatureExtractor
from dimensionality_analysis import GeometricDimensionalityAnalyzer
from discrimination_analysis import GeometricDiscriminationAnalyzer


def run_geometric_discrimination_analysis(
    output_dir: str = "geometric_analysis_results",
    max_samples_per_class: Optional[int] = None,
    specific_batches: Optional[List[str]] = None,
):
    """
    Run comprehensive geometric discrimination analysis.

    This function replicates and enhances your original t-SNE analysis:

    # Original code (enhanced):
    # X_feat = np.array([extract_features(np.load(f"exploration_data/{f}")) for f in os.listdir("exploration_data")])
    # y_labels = [f.split('_')[1] + "_" + f.split('_')[2] for f in os.listdir("exploration_data")]
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_2d = tsne.fit_transform(X_feat)
    # [visualization code]

    Args:
        output_dir: Directory to save results
        max_samples_per_class: Limit samples for testing (None = all samples)
        specific_batches: Specific batches to analyze (None = all available)
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("ACOUSTIC GEOMETRIC DISCRIMINATION ANALYSIS")
    print("Enhanced version of your original t-SNE analysis")
    print("=" * 80)

    # 1. DATA LOADING (replaces your file loading loop)
    print("\n1. LOADING DATA...")
    print("-" * 40)

    data_loader = GeometricDataLoader(base_dir="../data", sr=48000)

    # Get available batches
    available_batches = data_loader.get_available_batches()
    print(f"Available batches: {available_batches}")

    # Use specified batches or all available
    batches_to_use = specific_batches if specific_batches else available_batches

    # Load data (enhanced version of your file loading)
    audio_data, labels, metadata = data_loader.load_multiple_batches(
        batch_names=batches_to_use,
        max_samples_per_class=max_samples_per_class,
        verbose=True,
    )

    print_dataset_summary(audio_data, labels, metadata)

    # Simplify labels (tip, middle, base, none)
    simplified_labels = data_loader.simplify_labels(labels)

    # 2. FEATURE EXTRACTION (enhanced version of your extract_features)
    print("\n2. FEATURE EXTRACTION...")
    print("-" * 40)

    feature_extractor = GeometricFeatureExtractor(sr=48000)

    print("Extracting comprehensive geometric features...")
    X_feat = []
    failed_count = 0

    for i, audio in enumerate(audio_data):
        try:
            # Extract comprehensive features (replaces your extract_features call)
            features = feature_extractor.extract_features(audio, method="comprehensive")
            X_feat.append(features)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(audio_data)} samples")

        except Exception as e:
            print(f"  Warning: Failed to extract features from sample {i}: {e}")
            if X_feat:  # Use zero features as fallback
                X_feat.append(np.zeros_like(X_feat[0]))
            failed_count += 1

    X_feat = np.array(X_feat)
    feature_names = feature_extractor.get_feature_names(method="comprehensive")

    print(f"Feature extraction complete: {X_feat.shape}")
    print(f"Failed extractions: {failed_count}")

    # 3. t-SNE ANALYSIS (enhanced version of your original t-SNE)
    print("\n3. t-SNE ANALYSIS...")
    print("-" * 40)

    dim_analyzer = GeometricDimensionalityAnalyzer(random_state=42)

    # Test multiple perplexity values (including your original 30)
    print("Testing multiple perplexity values for robustness...")
    perplexity_values = [5, 10, 20, 30, 50]  # Including your original 30

    # Enhanced t-SNE analysis
    tsne_embedding, tsne_results = dim_analyzer.fit_transform_tsne(
        X_feat, perplexity=perplexity_values
    )

    optimal_perplexity = tsne_results["optimal_perplexity"]
    print(f"Optimal perplexity: {optimal_perplexity} (you used 30)")

    # 4. PCA ANALYSIS (additional analysis)
    print("\n4. PCA ANALYSIS...")
    print("-" * 40)

    pca_embedding, pca_results = dim_analyzer.fit_transform_pca(X_feat)
    print(
        f"PCA: {pca_results['n_components']} components explain {pca_results['total_explained_variance']:.3f} variance"
    )

    # 5. VISUALIZATION (enhanced version of your plotting code)
    print("\n5. CREATING VISUALIZATIONS...")
    print("-" * 40)

    # Main t-SNE plot (similar to your original but enhanced)
    print("Creating main t-SNE plot (enhanced version of your original)...")

    fig_main = dim_analyzer.plot_2d_embedding(
        tsne_embedding,
        simplified_labels,
        title="t-SNE: Are Geometries Separable? (Enhanced Analysis)",
        figsize=(12, 8),
        save_path=output_path / "tsne_geometric_separability.png",
    )
    plt.close(fig_main)

    # Comparison plot (t-SNE vs PCA)
    print("Creating comparison visualization...")

    comparison_embeddings = {
        f"t-SNE (perplexity={optimal_perplexity})": tsne_embedding,
        "PCA (first 2 components)": pca_embedding[:, :2],
    }

    fig_comparison = dim_analyzer.create_comparison_plot(
        comparison_embeddings,
        simplified_labels,
        figsize=(16, 6),
        save_path=output_path / "tsne_vs_pca_comparison.png",
    )
    plt.close(fig_comparison)

    # Perplexity comparison (if multiple values tested)
    if len(perplexity_values) > 1:
        print("Creating perplexity comparison...")
        from dimensionality_analysis import compare_perplexity_values

        fig_perp = compare_perplexity_values(
            X_feat, simplified_labels, perplexity_values, figsize=(20, 8)
        )
        fig_perp.savefig(
            output_path / "tsne_perplexity_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig_perp)

    # 6. STATISTICAL ANALYSIS (proves discrimination capability)
    print("\n6. DISCRIMINATION ANALYSIS...")
    print("-" * 40)

    disc_analyzer = GeometricDiscriminationAnalyzer(random_state=42)

    # Comprehensive separability analysis
    print("Analyzing class separability...")
    sep_results = disc_analyzer.analyze_class_separability(
        X_feat, simplified_labels, feature_names
    )

    # Classification performance
    print("Evaluating classification performance...")
    cls_results = disc_analyzer.evaluate_classification_performance(
        X_feat, simplified_labels
    )

    # Analyze embedding separability
    print("Analyzing t-SNE and PCA embedding separability...")
    tsne_sep = dim_analyzer.analyze_separability(
        tsne_embedding, simplified_labels, "t-SNE"
    )
    pca_sep = dim_analyzer.analyze_separability(
        pca_embedding[:, :2], simplified_labels, "PCA"
    )

    # Generate comprehensive report
    print("Generating discrimination report...")
    report = disc_analyzer.generate_discrimination_report()

    # 7. SAVE RESULTS
    print("\n7. SAVING RESULTS...")
    print("-" * 40)

    # Save the main results (like your original but enhanced)
    results_data = {
        "tsne_coordinates": {
            "x": tsne_embedding[:, 0].tolist(),
            "y": tsne_embedding[:, 1].tolist(),
        },
        "pca_coordinates": {
            "x": pca_embedding[:, 0].tolist(),
            "y": pca_embedding[:, 1].tolist(),
        },
        "labels": simplified_labels.tolist(),
        "original_labels": labels.tolist(),
    }

    # Save as CSV (like your original data would be)
    results_df = pd.DataFrame(
        {
            "tsne_1": tsne_embedding[:, 0],
            "tsne_2": tsne_embedding[:, 1],
            "pca_1": pca_embedding[:, 0],
            "pca_2": pca_embedding[:, 1],
            "label": simplified_labels,
            "original_label": labels,
        }
    )
    results_df.to_csv(output_path / "geometric_discrimination_data.csv", index=False)

    # Save features matrix
    features_df = pd.DataFrame(X_feat, columns=feature_names)
    features_df["label"] = simplified_labels
    features_df.to_csv(output_path / "extracted_features.csv", index=False)

    # Save discrimination report
    with open(output_path / "discrimination_analysis_report.txt", "w") as f:
        f.write(report)

    # Save analysis summary (convert NumPy types to JSON-serializable types)
    summary = {
        "dataset_info": {
            "total_samples": int(len(audio_data)),
            "n_features": int(X_feat.shape[1]),
            "n_classes": int(len(np.unique(simplified_labels))),
            "classes": [str(x) for x in np.unique(simplified_labels).tolist()],
            "batches_analyzed": batches_to_use,
        },
        "tsne_info": {
            "optimal_perplexity": float(optimal_perplexity),
            "separability_score": float(tsne_sep["silhouette_score"]),
            "separation_ratio": float(tsne_sep["separation_ratio"]),
        },
        "pca_info": {
            "n_components": int(pca_results["n_components"]),
            "explained_variance": float(pca_results["total_explained_variance"]),
            "separability_score": float(pca_sep["silhouette_score"]),
        },
        "discrimination_evidence": {
            "significant_features_ratio": float(
                sep_results["significant_feature_ratio"]
            ),
            "fisher_discriminant_ratio": float(
                sep_results["fisher_discriminant_ratio"]
            ),
            "best_classification_accuracy": float(
                cls_results["best_classifier"]["cv_accuracy"]
                if "best_classifier" in cls_results
                else 0
            ),
        },
    }

    with open(output_path / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 8. FINAL REPORT
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {output_path}")
    print("\nKey findings:")
    print(
        f"• Dataset: {len(audio_data)} samples, {X_feat.shape[1]} features, {len(np.unique(simplified_labels))} classes"
    )
    print(f"• t-SNE optimal perplexity: {optimal_perplexity}")
    print(f"• t-SNE separability score: {tsne_sep['silhouette_score']:.3f}")
    print(f"• PCA explained variance: {pca_results['total_explained_variance']:.3f}")
    print(
        f"• Significant features: {sep_results['significant_feature_ratio']*100:.1f}%"
    )

    # Determine if discrimination is proven
    discrimination_proven = (
        sep_results["significant_feature_ratio"] > 0.2
        and tsne_sep["silhouette_score"] > 0.1
        and (cls_results.get("best_classifier", {}).get("cv_accuracy", 0) > 0.6)
    )

    if discrimination_proven:
        print("\n✓ GEOMETRIC DISCRIMINATION CAPABILITY CONFIRMED!")
        print("  The acoustic signal contains sufficient information to distinguish")
        print("  between different contact geometries (tip, middle, base positions).")
    else:
        print("\n⚠ GEOMETRIC DISCRIMINATION CAPABILITY UNCERTAIN")
        print("  Limited evidence for reliable geometric discrimination.")
        print("  Consider more data or different feature engineering approaches.")

    print(f"\nMain visualization: {output_path}/tsne_geometric_separability.png")
    print(f"Full report: {output_path}/discrimination_analysis_report.txt")
    print(f"Data for further analysis: {output_path}/geometric_discrimination_data.csv")

    return {
        "tsne_embedding": tsne_embedding,
        "pca_embedding": pca_embedding,
        "labels": simplified_labels,
        "features": X_feat,
        "discrimination_proven": discrimination_proven,
        "summary": summary,
    }


def main():
    """Main execution function."""
    print("Starting comprehensive geometric discrimination analysis...")
    print("This enhances your original t-SNE analysis with:")
    print("• Robust feature extraction optimized for geometric discrimination")
    print("• Multiple perplexity testing and PCA comparison")
    print("• Statistical significance testing")
    print("• Classification performance evaluation")
    print("• Publication-ready visualizations")
    print("")

    # Run the analysis
    results = run_geometric_discrimination_analysis(
        output_dir="geometric_discrimination_results",
        max_samples_per_class=None,  # Use all available data
        specific_batches=None,  # Use all available batches
    )

    return results


if __name__ == "__main__":
    main()
