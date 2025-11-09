#!/usr/bin/env python3
"""
Training Integration with Configurable Feature Sets
==================================================

This module shows how to integrate the optimized feature sets with your
existing training pipeline. You can easily switch between MINIMAL/OPTIMAL/RESEARCH
configurations during training and evaluation.

Usage Examples:
    # Quick training with different feature sets
    python training_integration.py --mode MINIMAL
    python training_integration.py --mode OPTIMAL
    python training_integration.py --mode RESEARCH

    # Compare all modes
    python training_integration.py --compare-all
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import time
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Import our optimized feature sets
from optimized_feature_sets import FeatureSetConfig, OptimizedFeatureExtractor

# Import existing pipeline components
try:
    from batch_specific_analysis import BatchSpecificAnalyzer
    from geometric_data_loader import GeometricDataLoader
except ImportError:
    print("Warning: Could not import existing pipeline components")


class ConfigurableTrainingPipeline:
    """
    Training pipeline that can switch between different feature configurations.
    Integrates with your existing batch analysis system.
    """

    def __init__(self, batch_configs: Dict, base_data_dir: str):
        self.batch_configs = batch_configs
        self.base_data_dir = base_data_dir
        self.results_dir = Path("batch_analysis_results")
        self.results_dir.mkdir(exist_ok=True)

        # Feature extraction configurations
        self.feature_configs = {
            "MINIMAL": FeatureSetConfig("MINIMAL"),
            "OPTIMAL": FeatureSetConfig("OPTIMAL"),
            "RESEARCH": FeatureSetConfig("RESEARCH"),
        }

        self.current_mode = "OPTIMAL"

    def set_feature_mode(self, mode: str):
        """Switch feature extraction mode."""
        if mode not in self.feature_configs:
            raise ValueError(f"Mode must be one of {list(self.feature_configs.keys())}")

        self.current_mode = mode
        print(f"\n🔄 Switched to {mode} feature set")
        print(f"   Features: {self.feature_configs[mode].n_features}")
        print(
            f"   Expected: {self.feature_configs[mode].config['expected_performance']}"
        )

    def load_and_filter_features(
        self, batch_name: str, mode: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load existing features and filter to selected set."""
        if mode is None:
            mode = self.current_mode

        # Load from existing feature analysis results
        features_file = self.results_dir / batch_name / f"{batch_name}_features.csv"

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        df = pd.read_csv(features_file)

        # Extract labels and features
        y = df["simplified_label"].values
        X = df.drop(["simplified_label", "original_label"], axis=1)

        # Get all feature names
        all_feature_names = X.columns.tolist()

        # Filter to selected features
        config = self.feature_configs[mode]
        selected_indices = config.get_feature_indices(all_feature_names)

        if len(selected_indices) != config.n_features:
            missing_features = set(config.features) - set(
                [all_feature_names[i] for i in selected_indices]
            )
            print(f"Warning: Missing features in {batch_name}: {missing_features}")

        X_filtered = X.iloc[:, selected_indices]

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        print(
            f"  Loaded {batch_name}: {X_filtered.shape[0]} samples, {X_filtered.shape[1]} features"
        )

        return X_filtered.values, y_encoded

    def train_single_batch(self, batch_name: str, mode: str = None) -> Dict:
        """Train models on a single batch with specified feature set."""
        if mode is None:
            mode = self.current_mode

        print(f"\n🚀 Training {batch_name} with {mode} features...")

        # Load and filter features
        X, y = self.load_and_filter_features(batch_name, mode)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train multiple models
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(kernel="rbf", random_state=42),
        }

        results = {
            "batch_name": batch_name,
            "mode": mode,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": self.feature_configs[mode].features,
            "models": {},
        }

        for model_name, model in models.items():
            start_time = time.time()

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)

            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

            training_time = time.time() - start_time

            results["models"][model_name] = {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "training_time_ms": training_time * 1000,
            }

            print(
                f"  {model_name}: Test={test_acc:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
            )

        return results

    def compare_feature_modes(self, batch_name: str) -> Dict:
        """Compare all three feature modes on the same batch."""
        print(f"\n🔍 Comparing feature modes for {batch_name}")
        print("=" * 60)

        comparison_results = {}

        for mode in ["MINIMAL", "OPTIMAL", "RESEARCH"]:
            try:
                results = self.train_single_batch(batch_name, mode)
                comparison_results[mode] = results

                # Extract best model performance
                best_model = max(
                    results["models"].items(), key=lambda x: x[1]["cv_mean"]
                )

                print(
                    f"\n{mode:8s}: {results['n_features']} features, "
                    f"Best CV: {best_model[1]['cv_mean']:.3f}±{best_model[1]['cv_std']:.3f} "
                    f"({best_model[0]})"
                )

            except Exception as e:
                print(f"Error with {mode} mode: {str(e)}")

        return comparison_results

    def train_all_batches(self, mode: str = None) -> Dict:
        """Train all batches with specified feature mode."""
        if mode is None:
            mode = self.current_mode

        print(f"\n🎯 Training all batches with {mode} feature set")
        print("=" * 60)

        all_results = {}

        for batch_name in self.batch_configs.keys():
            try:
                results = self.train_single_batch(batch_name, mode)
                all_results[batch_name] = results

            except Exception as e:
                print(f"Error training {batch_name}: {str(e)}")

        # Save results
        self.save_training_results(all_results, mode)

        return all_results

    def save_training_results(self, results: Dict, mode: str):
        """Save training results to file."""
        output_file = self.results_dir / f"training_results_{mode.lower()}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")

    def create_comparison_plot(self, comparison_results: Dict, batch_name: str):
        """Create visualization comparing feature modes."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Feature Mode Comparison: {batch_name}", fontsize=16, fontweight="bold"
        )

        modes = list(comparison_results.keys())

        # 1. Accuracy comparison
        rf_accuracies = []
        n_features = []

        for mode in modes:
            if mode in comparison_results:
                rf_acc = comparison_results[mode]["models"]["RandomForest"]["cv_mean"]
                n_feat = comparison_results[mode]["n_features"]
                rf_accuracies.append(rf_acc)
                n_features.append(n_feat)

        bars1 = ax1.bar(
            modes, rf_accuracies, color=["lightcoral", "lightblue", "lightgreen"]
        )
        ax1.set_ylabel("Cross-Validation Accuracy")
        ax1.set_title("Random Forest Accuracy by Feature Set")
        ax1.set_ylim([0.8, 1.0])

        # Add accuracy values on bars
        for bar, acc in zip(bars1, rf_accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        # 2. Number of features
        bars2 = ax2.bar(
            modes, n_features, color=["lightcoral", "lightblue", "lightgreen"]
        )
        ax2.set_ylabel("Number of Features")
        ax2.set_title("Feature Count by Mode")

        for bar, n_feat in zip(bars2, n_features):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(n_feat),
                ha="center",
                va="bottom",
            )

        # 3. Training time comparison
        training_times = []
        for mode in modes:
            if mode in comparison_results:
                time_ms = comparison_results[mode]["models"]["RandomForest"][
                    "training_time_ms"
                ]
                training_times.append(time_ms)

        bars3 = ax3.bar(
            modes, training_times, color=["lightcoral", "lightblue", "lightgreen"]
        )
        ax3.set_ylabel("Training Time (ms)")
        ax3.set_title("Training Time by Feature Set")

        # 4. Accuracy vs Features scatter
        ax4.scatter(
            n_features, rf_accuracies, s=100, c=["red", "blue", "green"], alpha=0.7
        )
        for i, mode in enumerate(modes):
            ax4.annotate(
                mode,
                (n_features[i], rf_accuracies[i]),
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax4.set_xlabel("Number of Features")
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Accuracy vs Feature Count")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = self.results_dir / f"{batch_name}_feature_mode_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Comparison plot saved to: {plot_file}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train with configurable feature sets")
    parser.add_argument(
        "--mode",
        choices=["MINIMAL", "OPTIMAL", "RESEARCH"],
        default="OPTIMAL",
        help="Feature set mode",
    )
    parser.add_argument("--batch", type=str, help="Specific batch to train")
    parser.add_argument(
        "--compare-all", action="store_true", help="Compare all feature modes"
    )
    parser.add_argument(
        "--compare-batch", type=str, help="Compare modes for specific batch"
    )

    args = parser.parse_args()

    # Initialize with existing batch configuration
    try:
        analyzer = BatchSpecificAnalyzer()
        batch_configs = analyzer.batch_configs
        base_data_dir = analyzer.base_dir
    except:
        print("Error: Could not load batch configurations")
        return

    # Create training pipeline
    pipeline = ConfigurableTrainingPipeline(batch_configs, base_data_dir)

    if args.compare_all:
        # Compare all modes for all batches
        for batch_name in batch_configs.keys():
            try:
                comparison = pipeline.compare_feature_modes(batch_name)
                pipeline.create_comparison_plot(comparison, batch_name)
            except Exception as e:
                print(f"Error comparing {batch_name}: {e}")

    elif args.compare_batch:
        # Compare modes for specific batch
        comparison = pipeline.compare_feature_modes(args.compare_batch)
        pipeline.create_comparison_plot(comparison, args.compare_batch)

    elif args.batch:
        # Train specific batch with specified mode
        pipeline.set_feature_mode(args.mode)
        pipeline.train_single_batch(args.batch)

    else:
        # Train all batches with specified mode
        pipeline.set_feature_mode(args.mode)
        pipeline.train_all_batches()


if __name__ == "__main__":
    main()
