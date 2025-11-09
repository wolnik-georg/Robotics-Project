"""
Acoustic Feature Saliency Analysis for Geometric Reconstruction
=============================================================

This module provides feature importance and saliency analysis using scikit-learn
to understand which acoustic features are most important for geometric discrimination.

This version works with your existing pipeline and doesn't require PyTorch.
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

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    permutation_test_score,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

# Try to import SHAP for advanced explanations
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print(
        "Warning: SHAP not available. Install with 'pip install shap' for advanced explanations."
    )

# Try to import LIME for model-agnostic explanations
try:
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print(
        "Warning: LIME not available. Install with 'pip install lime' for additional explanations."
    )


class SklearnFeatureSaliencyAnalyzer:
    """
    Feature importance and saliency analysis using scikit-learn models.
    """

    def __init__(self, batch_configs: Dict, base_data_dir: Path):
        self.batch_configs = batch_configs
        self.base_data_dir = base_data_dir
        self.models = {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = {}
        self.saliency_results = {}

    def load_batch_features(
        self, batch_name: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load engineered features for a specific batch."""
        results_dir = Path(f"batch_analysis_results/{batch_name}")
        features_file = results_dir / f"{batch_name}_features.csv"

        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        df = pd.read_csv(features_file)
        feature_columns = [
            col
            for col in df.columns
            if col not in ["simplified_label", "original_label"]
        ]

        X = df[feature_columns].values
        y = df["simplified_label"].values

        return X, y, feature_columns

    def train_interpretable_models(self, batch_name: str) -> Dict:
        """
        Train multiple interpretable models and extract feature importance.
        """
        print(f"\n🧠 TRAINING INTERPRETABLE MODELS FOR {batch_name}")
        print("-" * 50)

        # Load data
        X, y, feature_names = self.load_batch_features(batch_name)
        print(f"Data shape: {X.shape}, Classes: {np.unique(y)}")

        # Store feature names
        self.feature_names[batch_name] = feature_names

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders[batch_name] = le

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[batch_name] = scaler

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Train multiple models
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
            "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=10),
            "svm_linear": SVC(kernel="linear", random_state=42),
        }

        results = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)

            print(f"  Train accuracy: {train_score:.3f}")
            print(f"  Test accuracy: {test_score:.3f}")
            print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            # Extract feature importance
            importance_dict = self._extract_feature_importance(
                model, model_name, X_train, y_train, feature_names
            )

            results[model_name] = {
                "model": model,
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "cv_accuracy_mean": cv_scores.mean(),
                "cv_accuracy_std": cv_scores.std(),
                "feature_importance": importance_dict,
                "test_data": (X_test, y_test),
            }

        self.models[batch_name] = results
        return results

    def _extract_feature_importance(
        self,
        model,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """Extract feature importance from different model types."""
        importance_dict = {}

        # Built-in feature importance (if available)
        if hasattr(model, "feature_importances_"):
            importance_dict["builtin"] = dict(
                zip(feature_names, model.feature_importances_)
            )

        if hasattr(model, "coef_"):
            # For linear models, use absolute coefficients
            if len(model.coef_.shape) == 1:
                coeffs = np.abs(model.coef_)
            else:
                # Multi-class: use mean absolute coefficients
                coeffs = np.mean(np.abs(model.coef_), axis=0)
            importance_dict["coefficients"] = dict(zip(feature_names, coeffs))

        # Permutation importance (model-agnostic)
        try:
            perm_importance = permutation_importance(
                model, X_train, y_train, n_repeats=10, random_state=42
            )
            importance_dict["permutation"] = dict(
                zip(feature_names, perm_importance.importances_mean)
            )
            importance_dict["permutation_std"] = dict(
                zip(feature_names, perm_importance.importances_std)
            )
        except Exception as e:
            print(f"  Warning: Permutation importance failed for {model_name}: {e}")

        return importance_dict

    def compute_statistical_significance(
        self, batch_name: str, model_name: str = "random_forest"
    ) -> Dict:
        """
        Compute statistical significance of features using permutation tests.
        """
        if batch_name not in self.models:
            raise ValueError(f"Models not trained for {batch_name}")

        print(f"\n📊 COMPUTING STATISTICAL SIGNIFICANCE FOR {batch_name}")
        print("-" * 50)

        # Load data
        X, y, feature_names = self.load_batch_features(batch_name)
        scaler = self.scalers[batch_name]
        le = self.label_encoders[batch_name]

        X_scaled = scaler.transform(X)
        y_encoded = le.transform(y)

        # Get model
        model = self.models[batch_name][model_name]["model"]

        # Baseline score
        baseline_score = model.score(X_scaled, y_encoded)

        # Feature-wise permutation tests
        feature_significance = {}

        for i, feature_name in enumerate(feature_names):
            scores = []
            for _ in range(100):  # 100 permutations
                X_perm = X_scaled.copy()
                np.random.shuffle(X_perm[:, i])  # Permute this feature
                score = model.score(X_perm, y_encoded)
                scores.append(score)

            # Calculate p-value (proportion of permuted scores >= baseline)
            p_value = np.mean(np.array(scores) >= baseline_score)
            score_drop = baseline_score - np.mean(scores)

            feature_significance[feature_name] = {
                "p_value": p_value,
                "score_drop": score_drop,
                "is_significant": p_value < 0.05,
                "permuted_scores_mean": np.mean(scores),
                "permuted_scores_std": np.std(scores),
            }

            if i % 10 == 0:
                print(f"  Processed {i+1}/{len(feature_names)} features")

        # Sort by significance
        significant_features = {
            k: v for k, v in feature_significance.items() if v["is_significant"]
        }
        print(
            f"\n✅ Found {len(significant_features)}/{len(feature_names)} statistically significant features"
        )

        return feature_significance

    def compute_shap_values(
        self, batch_name: str, model_name: str = "random_forest"
    ) -> Dict:
        """
        Compute SHAP values for model interpretability.
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with 'pip install shap'")
            return {}

        print(f"\n🔍 COMPUTING SHAP VALUES FOR {batch_name}")
        print("-" * 50)

        # Load data
        X, y, feature_names = self.load_batch_features(batch_name)
        scaler = self.scalers[batch_name]
        X_scaled = scaler.transform(X)

        # Get model
        model = self.models[batch_name][model_name]["model"]

        # Create SHAP explainer
        if model_name == "random_forest" or model_name == "decision_tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
        else:
            # For linear models
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)

        # Process SHAP values
        if isinstance(shap_values, list):
            # Multi-class: average absolute values across classes
            shap_importance = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)

        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "feature_importance": dict(zip(feature_names, shap_importance)),
        }

    def compute_lime_explanations(
        self, batch_name: str, model_name: str = "random_forest", num_samples: int = 10
    ) -> Dict:
        """
        Compute LIME explanations for individual predictions.
        """
        if not LIME_AVAILABLE:
            print("LIME not available. Install with 'pip install lime'")
            return {}

        print(f"\n🍋 COMPUTING LIME EXPLANATIONS FOR {batch_name}")
        print("-" * 50)

        # Load data
        X, y, feature_names = self.load_batch_features(batch_name)
        scaler = self.scalers[batch_name]
        X_scaled = scaler.transform(X)

        # Get model and test data
        model = self.models[batch_name][model_name]["model"]
        X_test, y_test = self.models[batch_name][model_name]["test_data"]

        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X_scaled,
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=True,
        )

        # Get explanations for multiple samples
        explanations = []
        feature_importance_sum = np.zeros(len(feature_names))

        for i in range(min(num_samples, len(X_test))):
            explanation = explainer.explain_instance(
                X_test[i], model.predict_proba, num_features=len(feature_names)
            )

            explanations.append(explanation)

            # Aggregate feature importance
            for feature_name, importance in explanation.as_list():
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    feature_importance_sum[idx] += abs(importance)

        # Average feature importance across samples
        avg_importance = feature_importance_sum / num_samples

        return {
            "explanations": explanations,
            "avg_feature_importance": dict(zip(feature_names, avg_importance)),
        }

    def analyze_batch_saliency(self, batch_name: str) -> Dict:
        """
        Comprehensive feature saliency analysis for a batch.
        """
        print(f"\n📊 COMPREHENSIVE SALIENCY ANALYSIS FOR {batch_name}")
        print("=" * 60)

        config = self.batch_configs[batch_name]

        results = {
            "batch_name": batch_name,
            "config": config,
            "model_results": {},
            "feature_importance": {},
            "statistical_significance": {},
            "interpretability": {},
        }

        try:
            # Train interpretable models
            model_results = self.train_interpretable_models(batch_name)
            results["model_results"] = {
                k: {
                    "train_accuracy": v["train_accuracy"],
                    "test_accuracy": v["test_accuracy"],
                    "cv_accuracy_mean": v["cv_accuracy_mean"],
                    "cv_accuracy_std": v["cv_accuracy_std"],
                }
                for k, v in model_results.items()
            }

            # Extract feature importance from all models
            all_importance = {}
            for model_name, model_data in model_results.items():
                all_importance[model_name] = model_data["feature_importance"]

            results["feature_importance"] = all_importance

            # Statistical significance testing
            try:
                significance = self.compute_statistical_significance(
                    batch_name, "random_forest"
                )
                results["statistical_significance"] = significance
            except Exception as e:
                print(f"Warning: Statistical significance testing failed: {e}")

            # SHAP analysis (if available)
            if SHAP_AVAILABLE:
                try:
                    shap_results = self.compute_shap_values(batch_name, "random_forest")
                    results["interpretability"]["shap"] = shap_results.get(
                        "feature_importance", {}
                    )
                except Exception as e:
                    print(f"Warning: SHAP analysis failed: {e}")

            # LIME analysis (if available)
            if LIME_AVAILABLE:
                try:
                    lime_results = self.compute_lime_explanations(
                        batch_name, "random_forest"
                    )
                    results["interpretability"]["lime"] = lime_results.get(
                        "avg_feature_importance", {}
                    )
                except Exception as e:
                    print(f"Warning: LIME analysis failed: {e}")

        except Exception as e:
            print(f"Error in saliency analysis for {batch_name}: {e}")
            import traceback

            traceback.print_exc()

        self.saliency_results[batch_name] = results
        return results

    def visualize_feature_importance(self, batch_name: str, save_dir: Path = None):
        """
        Create comprehensive feature importance visualizations.
        """
        if batch_name not in self.saliency_results:
            print(f"No saliency results for {batch_name}")
            return

        results = self.saliency_results[batch_name]
        feature_names = self.feature_names[batch_name]

        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'Feature Importance Analysis: {batch_name}\n{results["config"]["description"]}',
            fontsize=16,
            fontweight="bold",
        )

        # 1. Random Forest Feature Importance
        ax1 = axes[0, 0]
        if (
            "random_forest" in results["feature_importance"]
            and "builtin" in results["feature_importance"]["random_forest"]
        ):
            rf_importance = results["feature_importance"]["random_forest"]["builtin"]
            importance_values = [rf_importance.get(name, 0) for name in feature_names]

            # Sort by importance
            sorted_indices = np.argsort(importance_values)[-20:]  # Top 20
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_values = [importance_values[i] for i in sorted_indices]

            bars = ax1.barh(
                range(len(sorted_values)), sorted_values, color="skyblue", alpha=0.7
            )
            ax1.set_yticks(range(len(sorted_values)))
            ax1.set_yticklabels(sorted_names, fontsize=8)
            ax1.set_xlabel("Feature Importance")
            ax1.set_title("Random Forest Feature Importance (Top 20)")

            # Color top 5 differently
            for i in range(max(0, len(bars) - 5), len(bars)):
                bars[i].set_color("orange")

        # 2. Permutation Importance
        ax2 = axes[0, 1]
        if (
            "random_forest" in results["feature_importance"]
            and "permutation" in results["feature_importance"]["random_forest"]
        ):
            perm_importance = results["feature_importance"]["random_forest"][
                "permutation"
            ]
            importance_values = [perm_importance.get(name, 0) for name in feature_names]

            sorted_indices = np.argsort(importance_values)[-20:]
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_values = [importance_values[i] for i in sorted_indices]

            bars = ax2.barh(
                range(len(sorted_values)), sorted_values, color="lightgreen", alpha=0.7
            )
            ax2.set_yticks(range(len(sorted_values)))
            ax2.set_yticklabels(sorted_names, fontsize=8)
            ax2.set_xlabel("Permutation Importance")
            ax2.set_title("Permutation Feature Importance (Top 20)")

            for i in range(max(0, len(bars) - 5), len(bars)):
                bars[i].set_color("red")

        # 3. Statistical Significance
        ax3 = axes[1, 0]
        if "statistical_significance" in results:
            significance = results["statistical_significance"]
            p_values = [
                significance.get(name, {}).get("p_value", 1.0) for name in feature_names
            ]
            score_drops = [
                significance.get(name, {}).get("score_drop", 0.0)
                for name in feature_names
            ]

            # Only show significant features
            significant_mask = np.array(p_values) < 0.05
            if significant_mask.any():
                sig_names = [
                    name for i, name in enumerate(feature_names) if significant_mask[i]
                ]
                sig_drops = [
                    score_drops[i]
                    for i in range(len(score_drops))
                    if significant_mask[i]
                ]

                sorted_indices = np.argsort(sig_drops)
                sorted_names = [sig_names[i] for i in sorted_indices]
                sorted_values = [sig_drops[i] for i in sorted_indices]

                ax3.barh(
                    range(len(sorted_values)), sorted_values, color="coral", alpha=0.7
                )
                ax3.set_yticks(range(len(sorted_values)))
                ax3.set_yticklabels(sorted_names, fontsize=8)
                ax3.set_xlabel("Score Drop (Significance)")
                ax3.set_title(
                    f"Statistically Significant Features ({len(sorted_names)})"
                )
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No statistically\nsignificant features\nfound",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                )
                ax3.set_title("Statistical Significance")

        # 4. Comparison of Methods
        ax4 = axes[1, 1]
        try:
            # Compare different importance methods
            methods = []
            all_scores = []

            if (
                "random_forest" in results["feature_importance"]
                and "builtin" in results["feature_importance"]["random_forest"]
            ):
                rf_scores = [
                    results["feature_importance"]["random_forest"]["builtin"].get(
                        name, 0
                    )
                    for name in feature_names
                ]
                methods.append("RF Built-in")
                all_scores.append(rf_scores)

            if (
                "random_forest" in results["feature_importance"]
                and "permutation" in results["feature_importance"]["random_forest"]
            ):
                perm_scores = [
                    results["feature_importance"]["random_forest"]["permutation"].get(
                        name, 0
                    )
                    for name in feature_names
                ]
                methods.append("Permutation")
                all_scores.append(perm_scores)

            if len(methods) >= 2:
                # Correlation heatmap
                correlation_matrix = np.corrcoef(all_scores)
                im = ax4.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                ax4.set_title("Feature Importance Method Correlation")
                ax4.set_xticks(range(len(methods)))
                ax4.set_yticks(range(len(methods)))
                ax4.set_xticklabels(methods, rotation=45)
                ax4.set_yticklabels(methods)

                # Add correlation values
                for i in range(len(methods)):
                    for j in range(len(methods)):
                        ax4.text(
                            j,
                            i,
                            f"{correlation_matrix[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color=(
                                "white"
                                if abs(correlation_matrix[i, j]) > 0.5
                                else "black"
                            ),
                        )

                plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "Need multiple methods\nfor comparison",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.set_title("Method Comparison")

        except Exception as e:
            ax4.text(
                0.5,
                0.5,
                f"Comparison failed:\n{str(e)[:50]}...",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Method Comparison")

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            plt.savefig(
                save_dir / f"{batch_name}_feature_importance.png",
                dpi=300,
                bbox_inches="tight",
            )
            self._save_saliency_results(batch_name, save_dir)

        plt.show()

    def _save_saliency_results(self, batch_name: str, save_dir: Path):
        """Save detailed saliency results to files."""
        results = self.saliency_results[batch_name]
        feature_names = self.feature_names[batch_name]

        # Create comprehensive results DataFrame
        saliency_df = pd.DataFrame({"feature_name": feature_names})

        # Add different importance scores
        if "random_forest" in results["feature_importance"]:
            rf_importance = results["feature_importance"]["random_forest"]

            if "builtin" in rf_importance:
                saliency_df["rf_builtin"] = [
                    rf_importance["builtin"].get(name, 0) for name in feature_names
                ]

            if "permutation" in rf_importance:
                saliency_df["rf_permutation"] = [
                    rf_importance["permutation"].get(name, 0) for name in feature_names
                ]
                if "permutation_std" in rf_importance:
                    saliency_df["rf_permutation_std"] = [
                        rf_importance["permutation_std"].get(name, 0)
                        for name in feature_names
                    ]

        # Add statistical significance
        if "statistical_significance" in results:
            significance = results["statistical_significance"]
            saliency_df["p_value"] = [
                significance.get(name, {}).get("p_value", 1.0) for name in feature_names
            ]
            saliency_df["score_drop"] = [
                significance.get(name, {}).get("score_drop", 0.0)
                for name in feature_names
            ]
            saliency_df["is_significant"] = [
                significance.get(name, {}).get("is_significant", False)
                for name in feature_names
            ]

        # Add interpretability results
        if "interpretability" in results:
            if "shap" in results["interpretability"]:
                shap_importance = results["interpretability"]["shap"]
                saliency_df["shap_importance"] = [
                    shap_importance.get(name, 0) for name in feature_names
                ]

            if "lime" in results["interpretability"]:
                lime_importance = results["interpretability"]["lime"]
                saliency_df["lime_importance"] = [
                    lime_importance.get(name, 0) for name in feature_names
                ]

        # Sort by most important feature (RF built-in if available)
        sort_column = (
            "rf_builtin"
            if "rf_builtin" in saliency_df.columns
            else saliency_df.columns[1]
        )
        saliency_df = saliency_df.sort_values(sort_column, ascending=False)

        # Save to CSV
        saliency_df.to_csv(save_dir / f"{batch_name}_feature_saliency.csv", index=False)

        # Save summary JSON
        summary = {
            "batch_name": batch_name,
            "config": results["config"],
            "model_performance": results["model_results"],
            "top_features": {},
            "analysis_summary": {},
        }

        # Add top features for different methods
        for method in ["rf_builtin", "rf_permutation", "score_drop"]:
            if method in saliency_df.columns:
                top_10 = saliency_df.nlargest(10, method)
                summary["top_features"][method] = {
                    "feature_names": top_10["feature_name"].tolist(),
                    "values": top_10[method].tolist(),
                }

        # Analysis summary
        if "statistical_significance" in results:
            n_significant = (
                saliency_df["is_significant"].sum()
                if "is_significant" in saliency_df.columns
                else 0
            )
            summary["analysis_summary"]["significant_features"] = {
                "count": int(n_significant),
                "percentage": float(n_significant / len(feature_names) * 100),
            }

        with open(save_dir / f"{batch_name}_saliency_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def analyze_all_batches(self, save_results: bool = True) -> Dict:
        """
        Run comprehensive saliency analysis for all available batches.
        """
        print("🧪 COMPREHENSIVE FEATURE SALIENCY ANALYSIS")
        print("=" * 80)
        print("This analysis will identify which acoustic features are most important")
        print("for geometric discrimination in each experimental condition.")
        print("")

        all_results = {}

        for batch_name in self.batch_configs.keys():
            # Check if feature data exists
            results_dir = Path(f"batch_analysis_results/{batch_name}")
            features_file = results_dir / f"{batch_name}_features.csv"

            if features_file.exists():
                try:
                    results = self.analyze_batch_saliency(batch_name)
                    all_results[batch_name] = results

                    if save_results:
                        save_dir = Path(f"batch_analysis_results/{batch_name}")
                        self.visualize_feature_importance(batch_name, save_dir)

                    print(f"✅ {batch_name} saliency analysis completed")
                except Exception as e:
                    print(f"❌ {batch_name} saliency analysis failed: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                print(f"⚠️ {batch_name} skipped - no feature data found")

        # Generate combined summary
        if save_results and all_results:
            self._generate_combined_saliency_summary(all_results)

        return all_results

    def _generate_combined_saliency_summary(self, all_results: Dict):
        """Generate a combined summary of all saliency analyses."""
        summary_path = Path(
            "batch_analysis_results/combined_feature_saliency_summary.txt"
        )

        with open(summary_path, "w") as f:
            f.write("COMBINED FEATURE SALIENCY ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write("This report identifies the most important acoustic features for\n")
            f.write("geometric discrimination across all experimental conditions.\n\n")

            # Overall feature ranking
            feature_votes = {}
            for batch_name, result in all_results.items():
                if (
                    "feature_importance" in result
                    and "random_forest" in result["feature_importance"]
                ):
                    rf_importance = result["feature_importance"]["random_forest"]
                    if "builtin" in rf_importance:
                        # Get top 10 features for this batch
                        sorted_features = sorted(
                            rf_importance["builtin"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:10]

                        for feature, score in sorted_features:
                            if feature not in feature_votes:
                                feature_votes[feature] = []
                            feature_votes[feature].append((batch_name, score))

            # Find features that appear in multiple batches
            consistent_features = {
                k: v for k, v in feature_votes.items() if len(v) >= 2
            }

            f.write("MOST CONSISTENT IMPORTANT FEATURES ACROSS BATCHES:\n")
            f.write("-" * 60 + "\n")

            for feature, batch_scores in sorted(
                consistent_features.items(), key=lambda x: len(x[1]), reverse=True
            ):
                f.write(f"\n{feature}:\n")
                f.write(f"  Appears in {len(batch_scores)} batches\n")
                for batch, score in batch_scores:
                    f.write(f"    {batch}: {score:.4f}\n")

            f.write(f"\n\nBATCH-SPECIFIC ANALYSIS:\n")
            f.write("=" * 60 + "\n")

            for batch_name, result in all_results.items():
                config = result["config"]
                f.write(f"\nBATCH: {batch_name}\n")
                f.write(f"Experiment: {config['description']}\n")
                f.write(f"Research Question: {config['research_question']}\n")

                if (
                    "model_results" in result
                    and "random_forest" in result["model_results"]
                ):
                    rf_acc = result["model_results"]["random_forest"]["test_accuracy"]
                    f.write(f"Random Forest Accuracy: {rf_acc:.3f}\n")

                if "statistical_significance" in result:
                    significance = result["statistical_significance"]
                    n_significant = sum(
                        1
                        for v in significance.values()
                        if v.get("is_significant", False)
                    )
                    total_features = len(significance)
                    f.write(
                        f"Statistically Significant Features: {n_significant}/{total_features} ({n_significant/total_features*100:.1f}%)\n"
                    )

                # Top 5 features
                if (
                    "feature_importance" in result
                    and "random_forest" in result["feature_importance"]
                ):
                    rf_importance = result["feature_importance"]["random_forest"]
                    if "builtin" in rf_importance:
                        top_features = sorted(
                            rf_importance["builtin"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:5]
                        f.write("Top 5 Features:\n")
                        for i, (feature, score) in enumerate(top_features):
                            f.write(f"  {i+1}. {feature}: {score:.4f}\n")

                f.write("\n")

        print(f"\n📋 Combined feature saliency summary saved to: {summary_path}")


if __name__ == "__main__":
    # Example usage
    from batch_specific_analysis import BatchSpecificAnalyzer

    # Get batch configurations
    analyzer = BatchSpecificAnalyzer()
    batch_configs = analyzer.batch_configs
    base_data_dir = analyzer.base_dir

    # Create saliency analyzer
    saliency_analyzer = SklearnFeatureSaliencyAnalyzer(batch_configs, base_data_dir)

    # Run analysis for all batches
    results = saliency_analyzer.analyze_all_batches(save_results=True)

    print("\n🎉 Feature saliency analysis completed!")
    print("Check the batch_analysis_results directories for detailed results.")
    print("\nKey outputs:")
    print("- Feature importance visualizations (.png)")
    print("- Detailed feature saliency data (.csv)")
    print("- Analysis summaries (.json)")
    print("- Combined summary report (.txt)")
