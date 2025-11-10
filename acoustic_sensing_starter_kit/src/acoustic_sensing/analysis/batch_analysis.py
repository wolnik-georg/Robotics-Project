#!/usr/bin/env python3
"""
Batch-Specific Acoustic Geometric Discrimination Analysis
========================================================

This script performs separate analysis for each experimental batch,
recognizing that different batches test different research questions:

Batch 1-2: Contact position (tip/middle/base/blank)
Batch 3:   Edge detection (contact_edge/no_contact)
Batch 4:   Material detection (metal/no_metal)

Author: Enhanced ML Pipeline for Acoustic Sensing
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add the src directory to the path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from acoustic_sensing.models.geometric_data_loader import (
    GeometricDataLoader,
    print_dataset_summary,
)
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor
from acoustic_sensing.analysis.dimensionality_analysis import (
    GeometricDimensionalityAnalyzer,
)
from acoustic_sensing.analysis.discrimination_analysis import (
    GeometricDiscriminationAnalyzer,
)
from acoustic_sensing.features.saliency_analysis import (
    AcousticSaliencyAnalyzer,
)
from acoustic_sensing.features.ablation_analysis import (
    FeatureAblationAnalyzer,
)
from acoustic_sensing.analysis.impulse_response_analysis import (
    ImpulseResponseAnalyzer,
)


class BatchSpecificAnalyzer:
    """
    Analyzer that handles different experimental conditions per batch.
    """

    def __init__(
        self, base_dir: str = "data", output_base: str = "batch_analysis_results"
    ):
        self.base_dir = Path(base_dir)
        self.output_base = Path(output_base)
        self.output_base.mkdir(exist_ok=True, parents=True)

        # Define batch-specific experimental conditions
        self.batch_configs = {
            "soft_finger_batch_1": {
                "experiment": "contact_position",
                "description": "Contact Position Discrimination",
                "expected_classes": [
                    "finger tip",
                    "finger middle",
                    "finger bottom",
                    "finger blank",
                ],
                "research_question": "Can acoustic signals distinguish WHERE on the finger contact occurs?",
                "class_mapping": {
                    "finger tip": "tip",
                    "finger middle": "middle",
                    "finger bottom": "base",
                    "finger blank": "none",
                },
            },
            "soft_finger_batch_2": {
                "experiment": "contact_position",
                "description": "Contact Position Discrimination (Batch 2)",
                "expected_classes": [
                    "finger tip",
                    "finger middle",
                    "finger bottom",
                    "finger blank",
                ],
                "research_question": "Can acoustic signals distinguish WHERE on the finger contact occurs?",
                "class_mapping": {
                    "finger tip": "tip",
                    "finger middle": "middle",
                    "finger bottom": "base",
                    "finger blank": "none",
                },
            },
            "soft_finger_batch_3": {
                "experiment": "edge_detection",
                "description": "Geometric Edge Detection",
                "expected_classes": ["contact edge", "no contact"],
                "research_question": "Can acoustic signals detect geometric edges/transitions?",
                "class_mapping": {"contact edge": "edge", "no contact": "no_edge"},
            },
            "soft_finger_batch_4": {
                "experiment": "material_detection",
                "description": "Material Property Detection",
                "expected_classes": [
                    "no metal",
                    "metal",
                ],  # no metal first to avoid substring matching
                "research_question": "Can acoustic signals distinguish material properties?",
                "class_mapping": {"_metal": "metal", "_no metal": "no_metal"},
            },
        }

    def detect_actual_classes(self, batch_name: str) -> List[str]:
        """Detect what classes actually exist in a batch."""
        batch_path = self.base_dir / batch_name / "data"

        if not batch_path.exists():
            return []

        # Get all WAV files and extract unique class names
        wav_files = list(batch_path.glob("*.wav"))
        classes = set()

        for wav_file in wav_files:
            if "sweep" in wav_file.name:
                continue

            # Extract class from filename (everything after number and underscore)
            parts = wav_file.stem.split("_", 1)
            if len(parts) > 1:
                class_name = parts[1]
                classes.add(class_name)

        return sorted(list(classes))

    def analyze_batch(
        self, batch_name: str, max_samples_per_class: Optional[int] = None
    ) -> Dict:
        """
        Perform comprehensive analysis on a single batch.
        """
        print("=" * 80)
        print(f"ANALYZING BATCH: {batch_name}")
        print("=" * 80)

        # Get batch configuration
        config = self.batch_configs.get(
            batch_name,
            {
                "experiment": "unknown",
                "description": "Unknown Experiment",
                "expected_classes": [],
                "research_question": "Unknown research question",
                "class_mapping": {},
            },
        )

        # Detect actual classes and use configuration order
        detected_classes = self.detect_actual_classes(batch_name)
        detected_set = set(detected_classes)

        # Use expected order but only include detected classes
        actual_classes = [
            cls for cls in config["expected_classes"] if cls in detected_set
        ]

        # Add any detected classes not in expected list (fallback)
        for cls in detected_classes:
            if cls not in actual_classes:
                actual_classes.append(cls)

        print(f"Expected classes: {config['expected_classes']}")
        print(f"Detected classes: {detected_classes}")
        print(f"Using class order: {actual_classes}")
        print(f"Research question: {config['research_question']}")

        if not actual_classes:
            print(f"❌ No data found in batch {batch_name}")
            return None

        # Create output directory for this batch
        batch_output = self.output_base / batch_name
        batch_output.mkdir(exist_ok=True, parents=True)

        # Load data with detected classes
        print(f"\n1. LOADING DATA FROM {batch_name}...")
        print("-" * 40)

        data_loader = GeometricDataLoader(base_dir=str(self.base_dir), sr=48000)

        try:
            audio_data, labels, metadata = data_loader.load_batch_data(
                batch_name,
                contact_positions=actual_classes,
                max_samples_per_class=max_samples_per_class,
                verbose=True,
            )

            if len(audio_data) == 0:
                print(f"❌ No audio data loaded from {batch_name}")
                return None

            print_dataset_summary(audio_data, labels, metadata)

            # Apply class mapping if available
            simplified_labels = labels.copy()
            if config["class_mapping"]:
                simplified_labels = np.array(
                    [config["class_mapping"].get(label, label) for label in labels]
                )

        except Exception as e:
            print(f"❌ Failed to load data from {batch_name}: {e}")
            return None

        # Feature extraction
        print(f"\n2. FEATURE EXTRACTION...")
        print("-" * 40)

        feature_extractor = GeometricFeatureExtractor(sr=48000)

        X_feat = []
        failed_count = 0

        for i, audio in enumerate(audio_data):
            try:
                features = feature_extractor.extract_features(
                    audio, method="comprehensive"
                )
                X_feat.append(features)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(audio_data)} samples")

            except Exception as e:
                print(f"  Warning: Failed to extract features from sample {i}: {e}")
                if X_feat:
                    X_feat.append(np.zeros_like(X_feat[0]))
                failed_count += 1

        X_feat = np.array(X_feat)
        feature_names = feature_extractor.get_feature_names(method="comprehensive")

        print(f"Feature extraction complete: {X_feat.shape}")
        print(f"Failed extractions: {failed_count}")

        # Impulse response analysis (deconvolution-based features)
        print(f"\n2.5. IMPULSE RESPONSE ANALYSIS...")
        print("-" * 40)

        impulse_analyzer = ImpulseResponseAnalyzer(sr=48000)
        impulse_features = []
        impulse_failed_count = 0

        # Find sweep signal file
        sweep_path = None
        data_dir = self.base_dir / batch_name / "data"
        if data_dir.exists():
            # Look for sweep file (usually named with 'sweep' or index 0)
            for wav_file in data_dir.glob("*.wav"):
                if "sweep" in wav_file.name.lower() or wav_file.name.startswith("0_"):
                    sweep_path = wav_file
                    break

        if sweep_path:
            print(f"Found sweep signal: {sweep_path}")

            for i, audio in enumerate(audio_data):
                try:
                    # Create temporary response file path for analysis
                    temp_response_path = data_dir / f"temp_response_{i}.wav"
                    # Save current audio as temporary WAV for analysis
                    import soundfile as sf

                    # Ensure audio is a proper float array (convert from object array)
                    audio_float = np.asarray(audio, dtype=np.float32)
                    sf.write(temp_response_path, audio_float, 48000)

                    # Analyze impulse response
                    features = impulse_analyzer.analyze_measurement(
                        sweep_path, temp_response_path
                    )

                    if features:  # Only append if we got valid features
                        # Filter out metadata fields before converting to list
                        filtered_features = {
                            k: v for k, v in features.items() if not k.startswith("_")
                        }
                        impulse_features.append(list(filtered_features.values()))
                    else:
                        # If this is the first sample, we don't know the expected length yet
                        # Append a placeholder that we'll fix later
                        impulse_features.append(None)

                    # Clean up temp file
                    temp_response_path.unlink(missing_ok=True)

                    if (i + 1) % 50 == 0:
                        print(
                            f"  Processed {i + 1}/{len(audio_data)} impulse responses"
                        )

                except Exception as e:
                    print(
                        f"  Warning: Failed to extract impulse features from sample {i}: {e}"
                    )
                    impulse_features.append(None)
                    impulse_failed_count += 1

            if impulse_features:
                # Find the first valid feature set to determine expected length
                expected_length = None
                valid_features = []
                impulse_feature_names = []

            # Process impulse features to match the number of regular features
            if impulse_features:
                # Find the first valid feature set to determine expected length
                expected_length = None
                impulse_feature_names = []

                # First pass: find expected length and feature names
                for feat in impulse_features:
                    if feat is not None and len(feat) > 0:
                        expected_length = len(feat)
                        # Create dummy impulse response to get feature names (already filtered)
                        dummy_ir = np.random.randn(10000).astype(np.float32)
                        all_feature_names = list(
                            impulse_analyzer.extract_impulse_features(dummy_ir).keys()
                        )
                        impulse_feature_names = [
                            name
                            for name in all_feature_names
                            if not name.startswith("_")
                        ]
                        break

                if expected_length is None:
                    print(
                        "No valid impulse features found, using regular features only"
                    )
                else:
                    # Second pass: create valid_features array matching X_feat shape
                    valid_features = []
                    for i, feat in enumerate(impulse_features):
                        if feat is not None and len(feat) > 0:
                            if len(feat) == expected_length:
                                valid_features.append(feat)
                            else:
                                # Length mismatch - pad or truncate to expected length
                                print(
                                    f"Warning: Feature length mismatch {len(feat)} vs {expected_length}, adjusting..."
                                )
                                if len(feat) < expected_length:
                                    feat.extend([0.0] * (expected_length - len(feat)))
                                else:
                                    feat = feat[:expected_length]
                                valid_features.append(feat)
                        else:
                            # Replace invalid/None with zeros
                            valid_features.append([0.0] * expected_length)

                    # Ensure we have the same number of impulse features as regular features
                    if len(valid_features) != len(X_feat):
                        print(
                            f"Warning: Impulse features ({len(valid_features)}) != regular features ({len(X_feat)}), padding..."
                        )
                        while len(valid_features) < len(X_feat):
                            valid_features.append([0.0] * expected_length)

                    X_impulse = np.array(
                        valid_features[: len(X_feat)]
                    )  # Truncate if necessary
                    print(f"Impulse response analysis complete: {X_impulse.shape}")
                    print(f"Failed impulse extractions: {impulse_failed_count}")

                    # Combine regular and impulse features
                    X_feat_combined = np.concatenate([X_feat, X_impulse], axis=1)
                    feature_names_combined = feature_names + impulse_feature_names
                    print(
                        f"Combined features: {X_feat_combined.shape} (regular + impulse)"
                    )

                    # Use combined features for subsequent analysis
                    X_feat = X_feat_combined
                    feature_names = feature_names_combined
            else:
                print("No impulse features extracted, using regular features only")
        else:
            print("No sweep signal found, skipping impulse response analysis")

        # Dimensionality reduction
        print(f"\n3. DIMENSIONALITY REDUCTION...")
        print("-" * 40)

        dim_analyzer = GeometricDimensionalityAnalyzer(random_state=42)

        # PCA Analysis
        pca_embedding, pca_results = dim_analyzer.fit_transform_pca(X_feat)
        print(
            f"PCA: {pca_results['n_components']} components explain {pca_results['total_explained_variance']:.3f} variance"
        )

        # t-SNE Analysis
        perplexity_values = [5, 10, 20, 30, 50]
        tsne_embedding, tsne_results = dim_analyzer.fit_transform_tsne(
            X_feat, perplexity=perplexity_values
        )
        optimal_perplexity = tsne_results["optimal_perplexity"]
        print(f"t-SNE: Optimal perplexity = {optimal_perplexity}")

        # Statistical analysis
        print(f"\n4. DISCRIMINATION ANALYSIS...")
        print("-" * 40)

        # Check if we have multiple classes for discrimination analysis
        unique_classes = np.unique(simplified_labels)

        if len(unique_classes) < 2:
            print(f"⚠️ Single class dataset detected: {unique_classes[0]}")
            print("Cannot perform discrimination analysis with only one class.")
            print("Generating descriptive statistics instead...")

            # Create dummy results for single-class case
            sep_results = {
                "n_classes": 1,
                "class_names": unique_classes.tolist(),
                "class_distribution": {str(unique_classes[0]): len(simplified_labels)},
                "significant_features": 0,
                "significant_feature_ratio": 0.0,
                "fisher_discriminant_ratio": 0.0,
                "lda": None,
            }

            cls_results = {"best_classifier": None, "classifiers": {}}

            # Create dummy separability results
            tsne_sep = {
                "silhouette_score": 0.0,
                "separation_ratio": 0.0,
                "method": "t-SNE (single class)",
            }
            pca_sep = {
                "silhouette_score": 0.0,
                "separation_ratio": 0.0,
                "method": "PCA (single class)",
            }

            print(f"✓ Single-class analysis completed for: {unique_classes[0]}")

        else:
            # Normal multi-class discrimination analysis
            print(f"Analyzing {len(unique_classes)} classes: {list(unique_classes)}")

            disc_analyzer = GeometricDiscriminationAnalyzer(random_state=42)

            # Separability analysis
            sep_results = disc_analyzer.analyze_class_separability(
                X_feat, simplified_labels, feature_names
            )

            # Classification performance
            cls_results = disc_analyzer.evaluate_classification_performance(
                X_feat, simplified_labels, test_multiple_classifiers=True
            )

            # Analyze embedding separability
            tsne_sep = dim_analyzer.analyze_separability(
                tsne_embedding, simplified_labels, "t-SNE"
            )
            pca_sep = dim_analyzer.analyze_separability(
                pca_embedding[:, :2], simplified_labels, "PCA"
            )

        # Create enhanced visualizations
        print(f"\n5. CREATING VISUALIZATIONS...")
        print("-" * 40)

        self._create_enhanced_visualizations(
            batch_name,
            config,
            tsne_embedding,
            pca_embedding,
            simplified_labels,
            X_feat,
            feature_names,
            pca_results,
            batch_output,
            audio_data,
        )

        # Generate comprehensive report
        report = self._generate_batch_report(
            batch_name,
            config,
            sep_results,
            cls_results,
            tsne_sep,
            pca_sep,
            len(audio_data),
            X_feat.shape[1],
        )

        # Save core results FIRST (before advanced analysis needs them)
        print(f"\n6. SAVING CORE RESULTS...")
        print("-" * 40)

        self._save_batch_results(
            batch_name,
            config,
            batch_output,
            audio_data,
            labels,
            simplified_labels,
            X_feat,
            feature_names,
            tsne_embedding,
            pca_embedding,
            sep_results,
            cls_results,
            report,
        )

        # Advanced feature analysis (now files exist)
        print(f"\n7. ADVANCED FEATURE ANALYSIS...")
        print("-" * 40)

        # Saliency analysis (neural network feature importance)
        print("Running saliency analysis...")
        try:
            saliency_analyzer = AcousticSaliencyAnalyzer(
                self.batch_configs, str(self.base_dir)
            )
            saliency_results = saliency_analyzer.analyze_batch_saliency(batch_name)
            saliency_analyzer.visualize_saliency_maps(batch_name, batch_output)
            print(f"✅ Saliency analysis completed for {batch_name}")
        except Exception as e:
            print(f"⚠️ Saliency analysis failed for {batch_name}: {e}")
            saliency_results = None

        # Ablation analysis (systematic feature testing)
        print("Running ablation analysis...")
        try:
            ablation_analyzer = FeatureAblationAnalyzer(
                self.batch_configs, str(self.base_dir)
            )
            ablation_results = ablation_analyzer.analyze_batch_ablation(batch_name)
            print(f"✅ Ablation analysis completed for {batch_name}")
        except Exception as e:
            print(f"⚠️ Ablation analysis failed for {batch_name}: {e}")
            ablation_results = None

        return {
            "batch_name": batch_name,
            "config": config,
            "n_samples": len(audio_data),
            "n_features": X_feat.shape[1],
            "classes": list(np.unique(simplified_labels)),
            "tsne_separability": tsne_sep["silhouette_score"] if tsne_sep else 0.0,
            "pca_explained_variance": pca_results["total_explained_variance"],
            "best_accuracy": (
                (cls_results.get("best_classifier") or {}).get("cv_accuracy", 0)
                if cls_results
                else 0
            ),
            "significant_features_ratio": sep_results["significant_feature_ratio"],
            "saliency_results": saliency_results,
            "ablation_results": ablation_results,
        }

    def _create_enhanced_visualizations(
        self,
        batch_name: str,
        config: Dict,
        tsne_embedding: np.ndarray,
        pca_embedding: np.ndarray,
        labels: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        pca_results: Dict,
        output_dir: Path,
        audio_data: List[np.ndarray],
    ):
        """Create enhanced visualizations with detailed dimension labels."""

        # 1. Enhanced t-SNE plot with detailed labels
        fig_tsne = plt.figure(figsize=(14, 10))

        # Main t-SNE plot
        ax_main = plt.subplot(2, 2, (1, 2))
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax_main.scatter(
                tsne_embedding[mask, 0],
                tsne_embedding[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7,
                s=60,
                edgecolors="black",
                linewidth=0.5,
            )

        ax_main.set_title(
            f'{config["description"]}\nt-SNE Visualization',
            fontsize=14,
            fontweight="bold",
        )
        ax_main.set_xlabel(
            "t-SNE Dimension 1\n(Non-linear combination of acoustic features)",
            fontsize=12,
        )
        ax_main.set_ylabel(
            "t-SNE Dimension 2\n(Non-linear combination of acoustic features)",
            fontsize=12,
        )
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax_main.grid(True, alpha=0.3)

        # Add research question as text
        ax_main.text(
            0.02,
            0.98,
            f'Research Question:\n{config["research_question"]}',
            transform=ax_main.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Feature importance subplot
        ax_feat = plt.subplot(2, 2, 3)
        if "feature_importance" in locals():
            importance = np.random.random(len(feature_names))  # Placeholder
        else:
            # Get feature importance from random forest
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)
            importance = rf.feature_importances_

        # Show top 10 most important features
        top_indices = np.argsort(importance)[-10:]
        ax_feat.barh(range(len(top_indices)), importance[top_indices])
        ax_feat.set_yticks(range(len(top_indices)))
        ax_feat.set_yticklabels([feature_names[i] for i in top_indices], fontsize=8)
        ax_feat.set_xlabel("Feature Importance")
        ax_feat.set_title("Top 10 Discriminative Features")
        ax_feat.grid(True, alpha=0.3)

        # PCA variance explanation
        ax_pca = plt.subplot(2, 2, 4)
        explained_var = pca_results["explained_variance_ratio"][
            :10
        ]  # First 10 components
        ax_pca.bar(range(1, len(explained_var) + 1), explained_var)
        ax_pca.set_xlabel("Principal Component")
        ax_pca.set_ylabel("Explained Variance Ratio")
        ax_pca.set_title("PCA Explained Variance")
        ax_pca.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_tsne.savefig(
            output_dir / f"{batch_name}_comprehensive_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_tsne)

        # 2. PCA vs t-SNE comparison with enhanced labels
        fig_comp = plt.figure(figsize=(16, 6))

        # PCA plot
        ax_pca = plt.subplot(1, 2, 1)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax_pca.scatter(
                pca_embedding[mask, 0],
                pca_embedding[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7,
                s=50,
            )

        ax_pca.set_title("PCA: Linear Dimensionality Reduction")
        pca_var_1 = pca_results["explained_variance_ratio"][0] * 100
        pca_var_2 = pca_results["explained_variance_ratio"][1] * 100
        ax_pca.set_xlabel(
            f"PC1 ({pca_var_1:.1f}% variance)\n(Linear combination of acoustic features)"
        )
        ax_pca.set_ylabel(
            f"PC2 ({pca_var_2:.1f}% variance)\n(Linear combination of acoustic features)"
        )
        ax_pca.legend()
        ax_pca.grid(True, alpha=0.3)

        # t-SNE plot
        ax_tsne = plt.subplot(1, 2, 2)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax_tsne.scatter(
                tsne_embedding[mask, 0],
                tsne_embedding[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=0.7,
                s=50,
            )

        ax_tsne.set_title("t-SNE: Non-linear Dimensionality Reduction")
        ax_tsne.set_xlabel("t-SNE 1\n(Non-linear manifold dimension)")
        ax_tsne.set_ylabel("t-SNE 2\n(Non-linear manifold dimension)")
        ax_tsne.legend()
        ax_tsne.grid(True, alpha=0.3)

        plt.suptitle(f'{config["description"]} - Method Comparison', fontsize=16)
        plt.tight_layout()
        fig_comp.savefig(
            output_dir / f"{batch_name}_method_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_comp)

        # Impulse response visualization (if sweep signal available)
        data_dir = self.base_dir / batch_name / "data"
        sweep_path = None
        if data_dir.exists():
            for wav_file in data_dir.glob("*.wav"):
                if "sweep" in wav_file.name.lower() or wav_file.name.startswith("0_"):
                    sweep_path = wav_file
                    break

        if sweep_path:
            try:
                print("  Creating impulse response visualization...")
                impulse_analyzer = ImpulseResponseAnalyzer(sr=48000)

                # Use first response file as example for single visualization
                response_files = [f for f in data_dir.glob("*.wav") if f != sweep_path]
                if response_files:
                    response_path = response_files[0]

                    # Load signals
                    sweep = impulse_analyzer.load_sweep_signal(sweep_path)
                    response = impulse_analyzer.load_response_signal(response_path)

                    # Generate inverse filter and deconvolve
                    inv_filter = impulse_analyzer.generate_inverse_filter(sweep)
                    impulse_response = impulse_analyzer.deconvolve_response(
                        response, inv_filter
                    )

                    # Create single example visualization
                    impulse_analyzer.visualize_analysis(
                        sweep,
                        response,
                        impulse_response,
                        save_path=output_dir
                        / f"{batch_name}_impulse_response_analysis.png",
                    )
                    print(f"  ✓ Saved impulse response visualization")

                # Create per-class transfer function comparison
                print("  Creating per-class transfer function analysis...")
                class_responses = {}

                # Group responses by class
                for audio, label in zip(audio_data, labels):
                    class_name = label
                    if class_name not in class_responses:
                        class_responses[class_name] = []

                    # Convert audio to proper format and add to class
                    audio_float = np.asarray(audio, dtype=np.float32)
                    class_responses[class_name].append(audio_float)

                # Limit to first 10 samples per class for efficiency
                for class_name in class_responses:
                    class_responses[class_name] = class_responses[class_name][:10]

                if class_responses and len(class_responses) > 1:
                    # Load sweep signal
                    sweep = impulse_analyzer.load_sweep_signal(sweep_path)

                    # Create per-class transfer function visualization
                    impulse_analyzer.visualize_class_transfer_functions(
                        sweep,
                        class_responses,
                        save_path=output_dir
                        / f"{batch_name}_class_transfer_functions.png",
                    )
                    print(f"  ✓ Saved per-class transfer function analysis")
                else:
                    print("  ⚠️ Not enough classes for transfer function comparison")

            except Exception as e:
                print(f"  ⚠️ Failed to create impulse response visualizations: {e}")

        print(f"  ✓ Saved enhanced visualizations to {output_dir}")

    def _generate_batch_report(
        self,
        batch_name: str,
        config: Dict,
        sep_results: Dict,
        cls_results: Dict,
        tsne_sep: Dict,
        pca_sep: Dict,
        n_samples: int,
        n_features: int,
    ) -> str:
        """Generate a comprehensive report for this batch."""

        report_lines = [
            "=" * 80,
            f"BATCH-SPECIFIC ACOUSTIC DISCRIMINATION ANALYSIS",
            f"Batch: {batch_name}",
            "=" * 80,
            "",
            f"EXPERIMENT: {config['description']}",
            f"Research Question: {config['research_question']}",
            "",
            "DATASET SUMMARY",
            "-" * 40,
            f"Samples: {n_samples}",
            f"Features: {n_features}",
            f"Classes: {len(sep_results['class_names'])} ({', '.join(sep_results['class_names'])})",
            "",
            "Class Distribution:",
        ]

        for class_name, count in sep_results["class_distribution"].items():
            percentage = (int(count) / n_samples) * 100
            report_lines.append(f"  {class_name}: {count} samples ({percentage:.1f}%)")

        report_lines.extend(
            [
                "",
                "STATISTICAL ANALYSIS",
                "-" * 40,
                f"Statistically significant features: {sep_results['significant_features']}/{n_features} ({sep_results['significant_feature_ratio']*100:.1f}%)",
                f"Fisher discriminant ratio: {sep_results['fisher_discriminant_ratio']:.3f}",
            ]
        )

        if sep_results["lda"]:
            lda = sep_results["lda"]
            report_lines.extend(
                [
                    f"LDA explained variance: {lda['total_explained_variance']:.3f}",
                    f"Number of discriminant functions: {lda['n_discriminant_functions']}",
                ]
            )

        report_lines.extend(
            [
                "",
                "CLASSIFICATION PERFORMANCE",
                "-" * 40,
            ]
        )

        if (
            "best_classifier" in cls_results
            and cls_results["best_classifier"] is not None
        ):
            best = cls_results["best_classifier"]
            report_lines.extend(
                [
                    f"Best classifier: {best['name']}",
                    f"Cross-validation accuracy: {best['cv_accuracy']:.3f}",
                    f"Cross-validation F1-score: {best['cv_f1']:.3f}",
                    "",
                    "All Classifiers:",
                ]
            )

            for name, results in cls_results["classifiers"].items():
                if "error" not in results:
                    report_lines.append(
                        f"  {name}: Accuracy={results['cv_accuracy_mean']:.3f}±{results['cv_accuracy_std']:.3f}, "
                        f"F1={results['cv_f1_mean']:.3f}±{results['cv_f1_std']:.3f}"
                    )
        else:
            report_lines.append(
                "Single-class dataset - no classification metrics available"
            )

        report_lines.extend(
            [
                "",
                "EMBEDDING ANALYSIS",
                "-" * 40,
            ]
        )

        if tsne_sep is not None:
            report_lines.extend(
                [
                    f"t-SNE separability score: {tsne_sep['silhouette_score']:.3f}",
                    f"PCA separability score: {pca_sep['silhouette_score']:.3f}",
                    f"t-SNE separation ratio: {tsne_sep['separation_ratio']:.3f}",
                ]
            )
        else:
            report_lines.append(
                "Single-class dataset - no separability analysis available"
            )

        report_lines.extend(
            [
                "",
                "CONCLUSIONS",
                "-" * 40,
            ]
        )

        # Determine discrimination capability
        if cls_results.get("best_classifier") is not None:
            accuracy = cls_results["best_classifier"]["cv_accuracy"]
        else:
            accuracy = 0
        sig_features = sep_results["significant_feature_ratio"]

        if accuracy > 0.9 and sig_features > 0.3:
            report_lines.extend(
                [
                    f"✓ STRONG DISCRIMINATION CAPABILITY CONFIRMED",
                    "",
                    "Evidence:",
                    f"  • High classification accuracy: {accuracy:.3f}",
                    f"  • Significant features: {sig_features*100:.1f}%",
                    f"  • Research question: {config['research_question']}",
                    "  • ANSWER: YES - Acoustic signals reliably discriminate the tested conditions",
                ]
            )
        elif accuracy > 0.7 and sig_features > 0.2:
            report_lines.extend(
                [
                    f"✓ MODERATE DISCRIMINATION CAPABILITY",
                    "",
                    "Evidence:",
                    f"  • Moderate classification accuracy: {accuracy:.3f}",
                    f"  • Some significant features: {sig_features*100:.1f}%",
                    "  • Acoustic signals show discriminative potential but may need improvement",
                ]
            )
        else:
            report_lines.extend(
                [
                    f"⚠ LIMITED DISCRIMINATION CAPABILITY",
                    "",
                    f"  • Classification accuracy: {accuracy:.3f}",
                    f"  • Significant features: {sig_features*100:.1f}%",
                    "  • Consider different features, more data, or experimental modifications",
                ]
            )

        report_lines.extend(["", "=" * 80])

        return "\n".join(report_lines)

    def _save_batch_results(
        self,
        batch_name: str,
        config: Dict,
        output_dir: Path,
        audio_data: np.ndarray,
        original_labels: np.ndarray,
        simplified_labels: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        tsne_embedding: np.ndarray,
        pca_embedding: np.ndarray,
        sep_results: Dict,
        cls_results: Dict,
        report: str,
    ):
        """Save all results for this batch."""

        # Save report
        with open(output_dir / f"{batch_name}_analysis_report.txt", "w") as f:
            f.write(report)

        # Save data as CSV
        results_df = pd.DataFrame(
            {
                "tsne_1": tsne_embedding[:, 0],
                "tsne_2": tsne_embedding[:, 1],
                "pca_1": pca_embedding[:, 0],
                "pca_2": pca_embedding[:, 1],
                "simplified_label": simplified_labels,
                "original_label": original_labels,
            }
        )
        results_df.to_csv(output_dir / f"{batch_name}_embeddings.csv", index=False)

        # Save features
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df["simplified_label"] = simplified_labels
        features_df["original_label"] = original_labels
        features_df.to_csv(output_dir / f"{batch_name}_features.csv", index=False)

        # Save configuration and summary
        summary = {
            "batch_info": config,
            "dataset_summary": {
                "n_samples": len(audio_data),
                "n_features": features.shape[1],
                "classes": list(np.unique(simplified_labels)),
                "class_distribution": {
                    str(k): int(v) for k, v in sep_results["class_distribution"].items()
                },
            },
            "performance_summary": {
                "significant_features_ratio": float(
                    sep_results["significant_feature_ratio"]
                ),
                "fisher_discriminant_ratio": float(
                    sep_results["fisher_discriminant_ratio"]
                ),
                "best_classification_accuracy": float(
                    (cls_results.get("best_classifier") or {}).get("cv_accuracy", 0)
                    if cls_results
                    else 0
                ),
            },
        }

        with open(output_dir / f"{batch_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Results saved to {output_dir}")

    def analyze_all_batches(self, max_samples_per_class: Optional[int] = None) -> Dict:
        """Analyze all available batches separately."""

        print("🔍 BATCH-SPECIFIC ACOUSTIC DISCRIMINATION ANALYSIS")
        print("=" * 80)
        print(
            "Analyzing each experimental batch separately to address different research questions:"
        )
        print("")

        results = {}

        for batch_name in self.batch_configs.keys():
            batch_path = self.base_dir / batch_name
            print(
                f"Checking batch: {batch_name} at {batch_path} (exists: {batch_path.exists()})"
            )
            if batch_path.exists():
                try:
                    batch_result = self.analyze_batch(batch_name, max_samples_per_class)
                    if batch_result:
                        results[batch_name] = batch_result
                        print(f"✅ {batch_name} analysis completed")
                    else:
                        print(f"⚠️ {batch_name} analysis skipped (no data)")
                except Exception as e:
                    print(f"❌ {batch_name} analysis failed: {e}")
                    import traceback

                    traceback.print_exc()

                print("")

        # Generate combined summary
        self._generate_combined_summary(results)

        return results

    def _generate_combined_summary(self, results: Dict):
        """Generate a summary across all analyzed batches."""

        summary_path = self.output_base / "combined_analysis_summary.txt"

        with open(summary_path, "w") as f:
            f.write("COMBINED BATCH ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for batch_name, result in results.items():
                config = result["config"]
                f.write(f"BATCH: {batch_name}\n")
                f.write(f"Experiment: {config['description']}\n")
                f.write(f"Research Question: {config['research_question']}\n")
                f.write(
                    f"Samples: {result['n_samples']}, Features: {result['n_features']}\n"
                )
                f.write(f"Classes: {result['classes']}\n")
                f.write(f"Best Accuracy: {result['best_accuracy']:.3f}\n")
                f.write(
                    f"Significant Features: {result['significant_features_ratio']*100:.1f}%\n"
                )
                f.write(f"t-SNE Separability: {result['tsne_separability']:.3f}\n")
                f.write("\n")

        print(f"📊 Combined summary saved to: {summary_path}")


def main():
    """Main execution function."""

    analyzer = BatchSpecificAnalyzer()

    print("This analysis will separately examine each experimental batch:")
    print("• Batch 1-2: Contact position discrimination (tip/middle/base/blank)")
    print("• Batch 3:   Edge detection (edge/no_edge)")
    print("• Batch 4:   Material detection (metal/no_metal)")
    print("")

    # Run analysis on all batches
    results = analyzer.analyze_all_batches(max_samples_per_class=None)

    print("🎉 COMPREHENSIVE ACOUSTIC ANALYSIS COMPLETE!")
    print(f"Results saved in: {analyzer.output_base}")
    print("")
    print("Key outputs:")
    print("• Core analysis: discrimination, dimensionality reduction, visualizations")
    print("• Advanced analysis: saliency maps, ablation testing, feature importance")
    print("• Individual batch analysis reports and enhanced plots")
    print("• Separate CSV files for each experimental condition")
    print("• Combined summary across all experiments")

    return results


if __name__ == "__main__":
    main()
