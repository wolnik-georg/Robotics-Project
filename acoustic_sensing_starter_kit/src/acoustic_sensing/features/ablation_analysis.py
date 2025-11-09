#!/usr/bin/env python3
"""
Feature Ablation Analysis for Acoustic Geometric Discrimination
=============================================================

This module performs comprehensive ablation testing to validate feature importance
by systematically removing features and testing different combinations to measure
their actual impact on classification performance.

Ablation tests included:
1. Single feature removal (leave-one-out)
2. Top feature removal (remove most important)
3. Feature group removal (spectral, temporal, energy ratios)
4. Cumulative feature addition (start with best, add one by one)
5. Random feature combinations for baseline comparison
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from itertools import combinations
import random

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import time


class FeatureAblationAnalyzer:
    """
    Comprehensive feature ablation analysis to validate feature importance.

    Tests different feature combinations to understand:
    - Which features are truly essential
    - Which features are redundant
    - How feature combinations interact
    - Optimal minimal feature sets for different accuracy targets
    """

    def __init__(self, batch_configs: Dict, base_data_dir: str):
        self.batch_configs = batch_configs
        self.base_data_dir = base_data_dir
        self.results_dir = Path("batch_analysis_results")
        self.results_dir.mkdir(exist_ok=True)

        # Ablation results storage
        self.ablation_results = {}

        # Feature groupings for systematic testing
        self.feature_groups = {
            "spectral": [
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "spectral_flatness",
                "spectral_contrast_mean",
            ],
            "energy_ratios": [
                "ultra_high_energy_ratio",
                "high_energy_ratio",
                "mid_energy_ratio",
                "low_energy_ratio",
                "ultra_high_ratio",
                "high_ratio",
                "mid_ratio",
                "low_ratio",
            ],
            "temporal": [
                "temporal_centroid",
                "env_mean",
                "env_std",
                "env_skew",
                "env_kurtosis",
                "attack_time",
                "decay_time",
            ],
            "frequency_bands": ["low_mid_ratio", "mid_high_ratio", "low_high_ratio"],
            "burst_analysis": ["burst_rms", "burst_count", "burst_density"],
            "damping": ["damping_ratio", "q_factor"],
            "mfcc": [f"mfcc_{i}" for i in range(1, 13)],  # MFCC features if present
        }

    def load_batch_data(
        self, batch_name: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load feature data for a specific batch."""
        print(f"Loading data for {batch_name}...")

        # Load from the feature analysis results
        features_file = self.results_dir / batch_name / f"{batch_name}_features.csv"

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        df = pd.read_csv(features_file)

        # The last two columns are labels (simplified_label, original_label)
        # Use simplified_label as our target
        if "simplified_label" in df.columns:
            y = df["simplified_label"].values
            X = df.drop(["simplified_label", "original_label"], axis=1)
        elif "label" in df.columns:
            y = df["label"].values
            X = df.drop(["label"], axis=1)
        elif "class" in df.columns:
            y = df["class"].values
            X = df.drop(["class"], axis=1)
        else:
            # Assume last column is labels
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1]

        feature_names = X.columns.tolist()
        X = X.values

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        print(
            f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes"
        )
        print(f"  Classes: {list(le.classes_)}")

        return X, y, feature_names

    def get_baseline_performance(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Dict:
        """Get baseline performance with all features."""
        print("Computing baseline performance (all features)...")

        # Use multiple classifiers for robust baseline
        classifiers = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(kernel="rbf", random_state=42),
        }

        baseline_results = {}

        for clf_name, clf in classifiers.items():
            # Cross-validation
            cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            baseline_results[clf_name] = {
                "mean_accuracy": cv_scores.mean(),
                "std_accuracy": cv_scores.std(),
                "cv_scores": cv_scores.tolist(),
            }

        baseline_results["n_features"] = X.shape[1]
        baseline_results["feature_names"] = feature_names

        print(
            f"  Baseline RF accuracy: {baseline_results['RandomForest']['mean_accuracy']:.3f} ± {baseline_results['RandomForest']['std_accuracy']:.3f}"
        )

        return baseline_results

    def leave_one_out_ablation(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Dict:
        """Test removing each feature one at a time."""
        print("\nRunning leave-one-out feature ablation...")

        loo_results = {}

        # Use Random Forest as primary classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        for i, feature_name in enumerate(feature_names):
            # Create feature set without this feature
            feature_mask = np.ones(len(feature_names), dtype=bool)
            feature_mask[i] = False

            X_ablated = X[:, feature_mask]

            # Cross-validation
            cv_scores = cross_val_score(clf, X_ablated, y, cv=5, scoring="accuracy")

            loo_results[feature_name] = {
                "mean_accuracy": cv_scores.mean(),
                "std_accuracy": cv_scores.std(),
                "accuracy_drop": None,  # Will be computed after baseline
                "feature_index": i,
            }

            if i % 10 == 0:
                print(f"  Processed {i+1}/{len(feature_names)} features")

        print(f"  Completed leave-one-out for {len(feature_names)} features")

        return loo_results

    def top_features_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        top_feature_indices: List[int],
        n_top: int = 10,
    ) -> Dict:
        """Test removing combinations of top features."""
        print(f"\nTesting removal of top {n_top} features...")

        top_ablation_results = {}

        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Test removing 1, 2, 3, ... top features
        for n_remove in range(1, min(n_top + 1, len(top_feature_indices) + 1)):
            # Remove top n features
            features_to_remove = top_feature_indices[:n_remove]

            feature_mask = np.ones(len(feature_names), dtype=bool)
            feature_mask[features_to_remove] = False

            X_ablated = X[:, feature_mask]

            # Cross-validation
            cv_scores = cross_val_score(clf, X_ablated, y, cv=5, scoring="accuracy")

            removed_features = [feature_names[idx] for idx in features_to_remove]

            top_ablation_results[f"remove_top_{n_remove}"] = {
                "mean_accuracy": cv_scores.mean(),
                "std_accuracy": cv_scores.std(),
                "removed_features": removed_features,
                "n_remaining_features": X_ablated.shape[1],
            }

            print(
                f"  Removed top {n_remove}: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )

        return top_ablation_results

    def feature_group_ablation(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Dict:
        """Test removing entire feature groups."""
        print("\nTesting feature group ablation...")

        group_results = {}

        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        for group_name, group_features in self.feature_groups.items():
            # Find indices of features in this group
            group_indices = []
            for feature in group_features:
                if feature in feature_names:
                    group_indices.append(feature_names.index(feature))

            if not group_indices:
                continue

            # Remove this feature group
            feature_mask = np.ones(len(feature_names), dtype=bool)
            feature_mask[group_indices] = False

            X_ablated = X[:, feature_mask]

            # Cross-validation
            cv_scores = cross_val_score(clf, X_ablated, y, cv=5, scoring="accuracy")

            group_results[group_name] = {
                "mean_accuracy": cv_scores.mean(),
                "std_accuracy": cv_scores.std(),
                "removed_features": [feature_names[idx] for idx in group_indices],
                "n_features_removed": len(group_indices),
                "n_remaining_features": X_ablated.shape[1],
            }

            print(
                f"  Removed {group_name} ({len(group_indices)} features): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            )

        return group_results

    def cumulative_feature_addition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        feature_ranking: List[int],
    ) -> Dict:
        """Start with best feature, add one by one."""
        print("\nTesting cumulative feature addition...")

        cumulative_results = {}

        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        for n_features in range(
            1, min(21, len(feature_ranking) + 1)
        ):  # Test up to 20 features
            # Use top n features
            selected_features = feature_ranking[:n_features]

            X_selected = X[:, selected_features]

            # Cross-validation
            cv_scores = cross_val_score(clf, X_selected, y, cv=5, scoring="accuracy")

            selected_feature_names = [feature_names[idx] for idx in selected_features]

            cumulative_results[f"top_{n_features}"] = {
                "mean_accuracy": cv_scores.mean(),
                "std_accuracy": cv_scores.std(),
                "selected_features": selected_feature_names,
                "n_features": n_features,
            }

            if n_features <= 10 or n_features % 5 == 0:
                print(
                    f"  Top {n_features} features: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                )

        return cumulative_results

    def random_feature_combinations(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str], n_trials: int = 20
    ) -> Dict:
        """Test random feature combinations for baseline comparison."""
        print(f"\nTesting {n_trials} random feature combinations...")

        random_results = {}

        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Test different numbers of random features
        for n_features in [5, 10, 15, 20]:
            if n_features >= len(feature_names):
                continue

            trial_results = []

            for trial in range(n_trials):
                # Random feature selection
                random.seed(42 + trial)  # Reproducible randomness
                selected_indices = random.sample(range(len(feature_names)), n_features)

                X_random = X[:, selected_indices]

                # Cross-validation
                cv_scores = cross_val_score(
                    clf, X_random, y, cv=3, scoring="accuracy"
                )  # Faster CV
                trial_results.append(cv_scores.mean())

            random_results[f"random_{n_features}"] = {
                "mean_accuracy": np.mean(trial_results),
                "std_accuracy": np.std(trial_results),
                "min_accuracy": np.min(trial_results),
                "max_accuracy": np.max(trial_results),
                "n_features": n_features,
                "n_trials": n_trials,
            }

            print(
                f"  Random {n_features} features: {np.mean(trial_results):.3f} ± {np.std(trial_results):.3f} (range: {np.min(trial_results):.3f}-{np.max(trial_results):.3f})"
            )

        return random_results

    def analyze_batch_ablation(
        self, batch_name: str, save_results: bool = True
    ) -> Dict:
        """Run complete ablation analysis for a batch."""
        print(f"\n{'='*60}")
        print(f"FEATURE ABLATION ANALYSIS: {batch_name}")
        print(f"{'='*60}")

        # Load data
        X, y, feature_names = self.load_batch_data(batch_name)

        # Load existing feature importance ranking if available
        importance_file = (
            self.results_dir / batch_name / f"{batch_name}_feature_saliency.csv"
        )
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            # Assuming the CSV has 'feature_name' and 'rf_builtin' columns
            feature_ranking = []
            for _, row in importance_df.iterrows():
                if row["feature_name"] in feature_names:
                    feature_ranking.append(feature_names.index(row["feature_name"]))
        else:
            # Fallback: compute quick feature importance
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)
            importance_scores = clf.feature_importances_
            feature_ranking = np.argsort(importance_scores)[::-1].tolist()

        batch_results = {
            "batch_name": batch_name,
            "dataset_info": {
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "feature_names": feature_names,
            },
        }

        # 1. Baseline performance
        batch_results["baseline"] = self.get_baseline_performance(X, y, feature_names)

        # 2. Leave-one-out ablation
        batch_results["leave_one_out"] = self.leave_one_out_ablation(
            X, y, feature_names
        )

        # Compute accuracy drops for leave-one-out
        baseline_acc = batch_results["baseline"]["RandomForest"]["mean_accuracy"]
        for feature_name in batch_results["leave_one_out"]:
            acc_without_feature = batch_results["leave_one_out"][feature_name][
                "mean_accuracy"
            ]
            batch_results["leave_one_out"][feature_name]["accuracy_drop"] = (
                baseline_acc - acc_without_feature
            )

        # 3. Top features ablation
        batch_results["top_features_ablation"] = self.top_features_ablation(
            X, y, feature_names, feature_ranking
        )

        # 4. Feature group ablation
        batch_results["group_ablation"] = self.feature_group_ablation(
            X, y, feature_names
        )

        # 5. Cumulative feature addition
        batch_results["cumulative_addition"] = self.cumulative_feature_addition(
            X, y, feature_names, feature_ranking
        )

        # 6. Random combinations baseline
        batch_results["random_combinations"] = self.random_feature_combinations(
            X, y, feature_names
        )

        # Store results
        self.ablation_results[batch_name] = batch_results

        # Save results
        if save_results:
            self.save_ablation_results(batch_name, batch_results)
            self.create_ablation_visualizations(batch_name, batch_results)

        return batch_results

    def save_ablation_results(self, batch_name: str, results: Dict):
        """Save ablation results to files."""
        batch_dir = self.results_dir / batch_name
        batch_dir.mkdir(exist_ok=True)

        # Save JSON summary
        results_file = batch_dir / f"{batch_name}_ablation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save detailed CSV files
        self.save_ablation_csvs(batch_name, results, batch_dir)

        # Save summary report
        self.save_ablation_report(batch_name, results, batch_dir)

    def save_ablation_csvs(self, batch_name: str, results: Dict, batch_dir: Path):
        """Save detailed CSV files for different ablation tests."""

        # Leave-one-out results
        loo_data = []
        for feature_name, data in results["leave_one_out"].items():
            loo_data.append(
                {
                    "feature_name": feature_name,
                    "accuracy_without_feature": data["mean_accuracy"],
                    "accuracy_std": data["std_accuracy"],
                    "accuracy_drop": data["accuracy_drop"],
                    "feature_index": data["feature_index"],
                }
            )
        loo_df = pd.DataFrame(loo_data).sort_values("accuracy_drop", ascending=False)
        loo_df.to_csv(batch_dir / f"{batch_name}_leave_one_out.csv", index=False)

        # Cumulative addition results
        cum_data = []
        for key, data in results["cumulative_addition"].items():
            cum_data.append(
                {
                    "n_features": data["n_features"],
                    "accuracy": data["mean_accuracy"],
                    "accuracy_std": data["std_accuracy"],
                    "selected_features": ", ".join(data["selected_features"]),
                }
            )
        cum_df = pd.DataFrame(cum_data)
        cum_df.to_csv(batch_dir / f"{batch_name}_cumulative_addition.csv", index=False)

    def save_ablation_report(self, batch_name: str, results: Dict, batch_dir: Path):
        """Generate human-readable ablation report."""

        report_file = batch_dir / f"{batch_name}_ablation_report.txt"

        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"FEATURE ABLATION ANALYSIS REPORT: {batch_name}\n")
            f.write("=" * 80 + "\n\n")

            # Dataset info
            info = results["dataset_info"]
            f.write(f"DATASET SUMMARY\n")
            f.write(f"Samples: {info['n_samples']}\n")
            f.write(f"Features: {info['n_features']}\n")
            f.write(f"Classes: {info['n_classes']}\n\n")

            # Baseline performance
            baseline = results["baseline"]["RandomForest"]
            f.write(f"BASELINE PERFORMANCE (All Features)\n")
            f.write(
                f"Random Forest Accuracy: {baseline['mean_accuracy']:.4f} ± {baseline['std_accuracy']:.4f}\n\n"
            )

            # Most important features (by leave-one-out)
            loo = results["leave_one_out"]
            sorted_features = sorted(
                loo.items(), key=lambda x: x[1]["accuracy_drop"], reverse=True
            )

            f.write(f"MOST CRITICAL FEATURES (Top 10 by accuracy drop when removed)\n")
            f.write("-" * 60 + "\n")
            for i, (feature_name, data) in enumerate(sorted_features[:10]):
                f.write(
                    f"{i+1:2d}. {feature_name:25s} | Drop: {data['accuracy_drop']:.4f} | Acc without: {data['mean_accuracy']:.4f}\n"
                )

            f.write(f"\nMINIMAL FEATURE SETS\n")
            f.write("-" * 30 + "\n")

            # Find minimal feature sets for different accuracy targets
            baseline_acc = baseline["mean_accuracy"]
            cum_results = results["cumulative_addition"]

            for target_acc in [0.95, 0.90, 0.85]:
                if baseline_acc < target_acc:
                    continue

                for key, data in cum_results.items():
                    if data["mean_accuracy"] >= target_acc * baseline_acc:
                        f.write(
                            f"≥{target_acc*100:.0f}% baseline: {data['n_features']} features "
                            f"(accuracy: {data['mean_accuracy']:.4f})\n"
                        )
                        break

            # Feature group analysis
            f.write(f"\nFEATURE GROUP IMPORTANCE\n")
            f.write("-" * 40 + "\n")
            group_results = results["group_ablation"]
            sorted_groups = sorted(
                group_results.items(),
                key=lambda x: baseline["mean_accuracy"] - x[1]["mean_accuracy"],
                reverse=True,
            )

            for group_name, data in sorted_groups:
                acc_drop = baseline["mean_accuracy"] - data["mean_accuracy"]
                f.write(
                    f"{group_name:20s} | Drop: {acc_drop:.4f} | Features removed: {data['n_features_removed']}\n"
                )

    def create_ablation_visualizations(self, batch_name: str, results: Dict):
        """Create comprehensive ablation visualizations."""
        batch_dir = self.results_dir / batch_name

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Feature Ablation Analysis: {batch_name}", fontsize=16, fontweight="bold"
        )

        # 1. Leave-one-out accuracy drops
        loo = results["leave_one_out"]
        sorted_features = sorted(
            loo.items(), key=lambda x: x[1]["accuracy_drop"], reverse=True
        )

        feature_names = [x[0] for x in sorted_features[:15]]
        accuracy_drops = [x[1]["accuracy_drop"] for x in sorted_features[:15]]

        ax1.barh(range(len(accuracy_drops)), accuracy_drops, color="coral")
        ax1.set_yticks(range(len(accuracy_drops)))
        ax1.set_yticklabels(feature_names, fontsize=8)
        ax1.set_xlabel("Accuracy Drop When Removed")
        ax1.set_title("Most Critical Features (Leave-One-Out)")
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative feature addition
        cum_results = results["cumulative_addition"]
        n_features = [data["n_features"] for data in cum_results.values()]
        accuracies = [data["mean_accuracy"] for data in cum_results.values()]
        stds = [data["std_accuracy"] for data in cum_results.values()]

        ax2.errorbar(
            n_features, accuracies, yerr=stds, marker="o", linewidth=2, markersize=6
        )
        ax2.axhline(
            y=results["baseline"]["RandomForest"]["mean_accuracy"],
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Baseline (all features)",
        )
        ax2.set_xlabel("Number of Features")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Cumulative Feature Addition")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Feature group comparison
        group_results = results["group_ablation"]
        baseline_acc = results["baseline"]["RandomForest"]["mean_accuracy"]

        group_names = list(group_results.keys())
        group_accs = [group_results[name]["mean_accuracy"] for name in group_names]
        group_drops = [baseline_acc - acc for acc in group_accs]

        bars = ax3.bar(
            range(len(group_names)), group_drops, color="lightblue", alpha=0.7
        )
        ax3.set_xticks(range(len(group_names)))
        ax3.set_xticklabels(group_names, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Accuracy Drop")
        ax3.set_title("Feature Group Ablation")
        ax3.grid(True, alpha=0.3)

        # 4. Random vs. Optimal comparison
        random_results = results["random_combinations"]

        n_feat_optimal = []
        acc_optimal = []
        for key, data in cum_results.items():
            if data["n_features"] in [5, 10, 15, 20]:
                n_feat_optimal.append(data["n_features"])
                acc_optimal.append(data["mean_accuracy"])

        n_feat_random = []
        acc_random = []
        acc_random_std = []
        for key, data in random_results.items():
            n_feat_random.append(data["n_features"])
            acc_random.append(data["mean_accuracy"])
            acc_random_std.append(data["std_accuracy"])

        if n_feat_optimal:
            ax4.plot(
                n_feat_optimal,
                acc_optimal,
                "o-",
                label="Optimal Features",
                linewidth=2,
                markersize=8,
            )
        if n_feat_random:
            ax4.errorbar(
                n_feat_random,
                acc_random,
                yerr=acc_random_std,
                fmt="x--",
                label="Random Features",
                alpha=0.7,
                linewidth=2,
                markersize=8,
            )

        ax4.set_xlabel("Number of Features")
        ax4.set_ylabel("Accuracy")
        ax4.set_title("Optimal vs. Random Feature Selection")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(
            batch_dir / f"{batch_name}_ablation_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def analyze_all_batches(self, save_results: bool = True) -> Dict:
        """Run ablation analysis for all batches."""
        print(f"\n🔬 Starting Feature Ablation Analysis for All Batches")
        print(f"=" * 80)

        all_results = {}

        for batch_name in self.batch_configs.keys():
            try:
                results = self.analyze_batch_ablation(batch_name, save_results)
                all_results[batch_name] = results

                print(f"\n✅ Completed ablation analysis for {batch_name}")

            except Exception as e:
                print(f"\n❌ Error analyzing {batch_name}: {str(e)}")
                continue

        # Create combined summary
        if save_results and all_results:
            self.create_combined_ablation_summary(all_results)

        return all_results

    def create_combined_ablation_summary(self, all_results: Dict):
        """Create a combined summary across all batches."""

        summary_file = self.results_dir / "combined_ablation_summary.txt"

        with open(summary_file, "w") as f:
            f.write("COMBINED FEATURE ABLATION ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("MOST CONSISTENTLY IMPORTANT FEATURES ACROSS BATCHES\n")
            f.write("-" * 60 + "\n")

            # Collect all feature importance scores
            feature_importance_by_batch = {}

            for batch_name, results in all_results.items():
                if "leave_one_out" in results:
                    feature_importance_by_batch[batch_name] = results["leave_one_out"]

            # Find features that appear as important across multiple batches
            all_features = set()
            for batch_data in feature_importance_by_batch.values():
                all_features.update(batch_data.keys())

            feature_cross_batch_importance = {}
            for feature in all_features:
                appearances = []
                importance_scores = []

                for batch_name, batch_data in feature_importance_by_batch.items():
                    if feature in batch_data:
                        drop = batch_data[feature].get("accuracy_drop", 0)
                        if drop > 0.001:  # Only consider meaningful drops
                            appearances.append(batch_name)
                            importance_scores.append(drop)

                if len(appearances) >= 2:  # Appears in at least 2 batches
                    feature_cross_batch_importance[feature] = {
                        "batches": appearances,
                        "mean_importance": np.mean(importance_scores),
                        "n_appearances": len(appearances),
                    }

            # Sort by number of appearances, then by mean importance
            sorted_cross_batch = sorted(
                feature_cross_batch_importance.items(),
                key=lambda x: (x[1]["n_appearances"], x[1]["mean_importance"]),
                reverse=True,
            )

            for feature_name, data in sorted_cross_batch[:15]:
                f.write(
                    f"{feature_name:25s} | Batches: {data['n_appearances']} | "
                    f"Avg importance: {data['mean_importance']:.4f}\n"
                )
                f.write(f"    Appears in: {', '.join(data['batches'])}\n")

            f.write(f"\nOPTIMAL MINIMAL FEATURE SETS BY BATCH\n")
            f.write("-" * 50 + "\n")

            for batch_name, results in all_results.items():
                if "cumulative_addition" in results:
                    baseline_acc = results["baseline"]["RandomForest"]["mean_accuracy"]
                    cum_results = results["cumulative_addition"]

                    f.write(f"\n{batch_name}:\n")
                    f.write(f"  Baseline accuracy: {baseline_acc:.4f}\n")

                    # Find minimal sets for different accuracy targets
                    for target_pct in [0.95, 0.90]:
                        target_acc = target_pct * baseline_acc
                        for key, data in cum_results.items():
                            if data["mean_accuracy"] >= target_acc:
                                f.write(
                                    f"  {target_pct*100:.0f}% baseline ({target_acc:.3f}): "
                                    f"{data['n_features']} features\n"
                                )
                                break

        print(f"\n📊 Combined ablation summary saved to: {summary_file}")


if __name__ == "__main__":
    # Run feature ablation analysis
    from batch_specific_analysis import BatchSpecificAnalyzer

    # Get batch configurations
    analyzer = BatchSpecificAnalyzer()
    batch_configs = analyzer.batch_configs
    base_data_dir = analyzer.base_dir

    # Create ablation analyzer
    ablation_analyzer = FeatureAblationAnalyzer(batch_configs, base_data_dir)

    # Run ablation analysis for all batches
    print("🔬 Starting comprehensive feature ablation analysis...")
    results = ablation_analyzer.analyze_all_batches(save_results=True)

    print("\n🎉 Feature ablation analysis completed!")
    print("Key outputs:")
    print("- Leave-one-out importance rankings")
    print("- Minimal feature sets for different accuracy targets")
    print("- Feature group ablation results")
    print("- Optimal vs. random feature comparison")
    print("- Cross-batch feature consistency analysis")
    print("\nCheck the batch_analysis_results directories for detailed results!")
