from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import combinations
import os


class FeatureAblationExperiment(BaseExperiment):
    """
    Experiment for systematic feature ablation analysis to understand feature importance.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing and optionally discrimination analysis for baseline."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive feature ablation analysis.

        Args:
            shared_data: Dictionary containing loaded features and labels

        Returns:
            Dictionary containing ablation results and feature importance rankings
        """
        self.logger.info("Starting feature ablation experiment...")

        # Load per-batch data from previous experiment
        batch_results = self.load_shared_data(shared_data, "batch_results")

        # For feature ablation, combine all batch data for comprehensive analysis
        all_X = []
        all_y = []

        for batch_name, batch_data in batch_results.items():
            X_batch = batch_data["features"]
            y_batch = batch_data["labels"]

            all_X.append(X_batch)
            all_y.append(y_batch)

        # Combine all batches
        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        self.logger.info(
            f"Combined data: {len(X)} samples, {X.shape[1]} features across {len(np.unique(y))} classes"
        )

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get baseline performance with all features
        baseline_performance = self._get_baseline_performance(X_scaled, y)
        self.logger.info(
            f"Baseline accuracy with all features: {baseline_performance:.4f}"
        )

        results = {"baseline_performance": baseline_performance, "scaler": scaler}

        # Perform different types of ablation studies
        if self.config.get("include_leave_one_out", True):
            self.logger.info("Performing leave-one-out ablation...")
            results["leave_one_out"] = self._perform_leave_one_out_ablation(
                X_scaled, y, baseline_performance
            )

        if self.config.get("include_cumulative_addition", True):
            self.logger.info("Performing cumulative feature addition...")
            results["cumulative_addition"] = self._perform_cumulative_addition(
                X_scaled, y
            )

        if self.config.get("include_group_ablation", True):
            self.logger.info("Performing feature group ablation...")
            results["group_ablation"] = self._perform_group_ablation(
                X_scaled, y, baseline_performance
            )

        if self.config.get("include_random_combinations", True):
            self.logger.info("Performing random feature combination analysis...")
            results["random_combinations"] = self._perform_random_combinations(
                X_scaled, y
            )

        # Analyze and synthesize results
        synthesis = self._synthesize_ablation_results(results)
        results["synthesis"] = synthesis

        # Create visualizations
        self._create_ablation_visualizations(results)

        # Save summary
        self._save_ablation_summary(results)

        self.logger.info("Feature ablation experiment completed")
        return results

    def _get_baseline_performance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Get baseline performance with all features."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
        return scores.mean()

    def _perform_leave_one_out_ablation(
        self, X: np.ndarray, y: np.ndarray, baseline: float
    ) -> dict:
        """Perform leave-one-out feature ablation."""
        n_features = X.shape[1]
        feature_importance = []

        for i in range(n_features):
            # Create feature set without feature i
            X_ablated = np.delete(X, i, axis=1)

            # Evaluate performance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=42
            )  # Reduced folds for speed
            scores = cross_val_score(rf, X_ablated, y, cv=cv, scoring="accuracy")
            ablated_performance = scores.mean()

            # Calculate importance as performance drop
            importance = baseline - ablated_performance
            feature_importance.append(
                {
                    "feature_index": i,
                    "ablated_performance": ablated_performance,
                    "importance_score": importance,
                    "performance_drop": importance,
                }
            )

            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{n_features} features")

        # Sort by importance
        feature_importance.sort(key=lambda x: x["importance_score"], reverse=True)

        return {
            "feature_importance": feature_importance,
            "most_important_features": [
                f["feature_index"] for f in feature_importance[:10]
            ],
            "least_important_features": [
                f["feature_index"] for f in feature_importance[-10:]
            ],
            "max_performance_drop": max(
                f["importance_score"] for f in feature_importance
            ),
            "min_performance_drop": min(
                f["importance_score"] for f in feature_importance
            ),
        }

    def _perform_cumulative_addition(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Perform cumulative feature addition analysis."""
        n_features = X.shape[1]

        # Start with Random Forest feature importance to guide order
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_order = np.argsort(rf.feature_importances_)[::-1]

        cumulative_performance = []

        # Add features one by one in order of importance
        for i in range(
            1, min(n_features + 1, 51)
        ):  # Limit to first 50 features for speed
            selected_features = feature_order[:i]
            X_selected = X[:, selected_features]

            # Evaluate performance
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(rf, X_selected, y, cv=cv, scoring="accuracy")
            performance = scores.mean()

            cumulative_performance.append(
                {
                    "num_features": i,
                    "features_used": selected_features.tolist(),
                    "performance": performance,
                }
            )

            if i % 5 == 0:
                self.logger.info(
                    f"Cumulative addition: {i} features, accuracy: {performance:.4f}"
                )

        # Find optimal number of features (elbow point)
        optimal_num_features = self._find_optimal_features(cumulative_performance)

        return {
            "cumulative_performance": cumulative_performance,
            "feature_order": feature_order.tolist(),
            "optimal_num_features": optimal_num_features,
            "performance_at_optimal": (
                cumulative_performance[optimal_num_features - 1]["performance"]
                if optimal_num_features > 0
                else 0
            ),
        }

    def _perform_group_ablation(
        self, X: np.ndarray, y: np.ndarray, baseline: float
    ) -> dict:
        """Perform ablation on feature groups (acoustic vs impulse response)."""
        # Define feature groups (assuming first 38 are acoustic, rest are impulse response)
        acoustic_features = list(range(38))
        impulse_features = list(range(38, X.shape[1]))

        groups = {
            "acoustic_only": acoustic_features,
            "impulse_only": impulse_features,
            "without_acoustic": impulse_features,
            "without_impulse": acoustic_features,
        }

        group_results = {}

        for group_name, feature_indices in groups.items():
            if not feature_indices:  # Skip empty groups
                continue

            X_group = X[:, feature_indices]

            # Evaluate performance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(rf, X_group, y, cv=cv, scoring="accuracy")
            performance = scores.mean()

            group_results[group_name] = {
                "features": feature_indices,
                "num_features": len(feature_indices),
                "performance": performance,
                "performance_vs_baseline": performance - baseline,
            }

            self.logger.info(
                f"Group {group_name}: {performance:.4f} accuracy "
                f"({performance - baseline:+.4f} vs baseline)"
            )

        return group_results

    def _perform_random_combinations(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Perform analysis with random feature combinations."""
        n_features = X.shape[1]

        # Test different subset sizes
        subset_sizes = [5, 10, 15, 20, 25, 30, min(40, n_features)]
        random_results = {}

        for size in subset_sizes:
            if size >= n_features:
                continue

            performances = []
            feature_combinations = []

            # Test multiple random combinations
            for _ in range(10):  # 10 random trials per size
                # Randomly select features
                selected_features = np.random.choice(n_features, size, replace=False)
                X_random = X[:, selected_features]

                # Evaluate performance
                rf = RandomForestClassifier(
                    n_estimators=50, random_state=42
                )  # Reduced estimators for speed
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(rf, X_random, y, cv=cv, scoring="accuracy")
                performance = scores.mean()

                performances.append(performance)
                feature_combinations.append(selected_features.tolist())

            random_results[f"size_{size}"] = {
                "subset_size": size,
                "performances": performances,
                "mean_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "best_performance": np.max(performances),
                "best_combination": feature_combinations[np.argmax(performances)],
            }

            self.logger.info(
                f"Random size {size}: {np.mean(performances):.4f} ± {np.std(performances):.4f}"
            )

        return random_results

    def _find_optimal_features(self, cumulative_performance: List[dict]) -> int:
        """Find optimal number of features using elbow method."""
        if not cumulative_performance:
            return 0

        performances = [p["performance"] for p in cumulative_performance]

        # Simple elbow detection: find point where improvement becomes minimal
        improvements = [
            performances[i] - performances[i - 1] if i > 0 else 0
            for i in range(len(performances))
        ]

        # Find where improvement drops below threshold
        improvement_threshold = 0.005  # 0.5% improvement threshold
        for i, improvement in enumerate(improvements):
            if i > 10 and improvement < improvement_threshold:  # At least 10 features
                return i

        # If no clear elbow, return point with 90% of max performance improvement
        max_perf = max(performances)
        min_perf = min(performances)
        target_perf = min_perf + 0.9 * (max_perf - min_perf)

        for i, perf in enumerate(performances):
            if perf >= target_perf:
                return i + 1

        return len(performances)

    def _synthesize_ablation_results(self, results: Dict[str, Any]) -> dict:
        """Synthesize results from all ablation methods."""
        synthesis = {
            "top_features_consensus": [],
            "feature_importance_rankings": {},
            "key_insights": [],
        }

        # Combine rankings from different methods
        all_rankings = {}

        # Leave-one-out rankings
        if "leave_one_out" in results:
            loo_features = results["leave_one_out"]["most_important_features"]
            all_rankings["leave_one_out"] = loo_features
            synthesis["feature_importance_rankings"]["leave_one_out"] = loo_features

        # Cumulative addition rankings
        if "cumulative_addition" in results:
            cum_features = results["cumulative_addition"]["feature_order"][:10]
            all_rankings["cumulative_addition"] = cum_features
            synthesis["feature_importance_rankings"][
                "cumulative_addition"
            ] = cum_features

        # Find consensus features (appearing in multiple rankings)
        if len(all_rankings) > 1:
            feature_counts = {}
            for ranking in all_rankings.values():
                for feature in ranking:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1

            # Features appearing in multiple rankings
            consensus_features = [f for f, count in feature_counts.items() if count > 1]
            synthesis["top_features_consensus"] = sorted(
                consensus_features, key=lambda x: feature_counts[x], reverse=True
            )

        # Generate insights
        insights = []

        if "group_ablation" in results:
            group_results = results["group_ablation"]
            if "acoustic_only" in group_results and "impulse_only" in group_results:
                acoustic_perf = group_results["acoustic_only"]["performance"]
                impulse_perf = group_results["impulse_only"]["performance"]

                if acoustic_perf > impulse_perf:
                    insights.append(
                        f"Acoustic features ({acoustic_perf:.4f}) outperform impulse response features ({impulse_perf:.4f})"
                    )
                else:
                    insights.append(
                        f"Impulse response features ({impulse_perf:.4f}) outperform acoustic features ({acoustic_perf:.4f})"
                    )

        if "cumulative_addition" in results:
            optimal_num = results["cumulative_addition"]["optimal_num_features"]
            total_features = len(results["cumulative_addition"]["feature_order"])
            insights.append(
                f"Optimal performance achieved with {optimal_num} features out of {total_features} total"
            )

        if "random_combinations" in results:
            random_results = results["random_combinations"]
            best_random = max([r["best_performance"] for r in random_results.values()])
            baseline = results["baseline_performance"]
            if best_random < baseline * 0.95:  # If best random is < 95% of baseline
                insights.append(
                    "Careful feature selection is critical - random combinations perform poorly"
                )

        synthesis["key_insights"] = insights

        return synthesis

    def _create_ablation_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive ablation analysis visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Leave-one-out importance
        if "leave_one_out" in results:
            loo_results = results["leave_one_out"]["feature_importance"]
            feature_indices = [f["feature_index"] for f in loo_results[:20]]  # Top 20
            importance_scores = [f["importance_score"] for f in loo_results[:20]]

            axes[0, 0].barh(range(len(feature_indices)), importance_scores)
            axes[0, 0].set_yticks(range(len(feature_indices)))
            axes[0, 0].set_yticklabels([f"Feature {idx}" for idx in feature_indices])
            axes[0, 0].set_xlabel("Importance Score")
            axes[0, 0].set_title("Top 20 Features (Leave-One-Out)")

        # 2. Cumulative addition curve
        if "cumulative_addition" in results:
            cum_results = results["cumulative_addition"]["cumulative_performance"]
            num_features = [r["num_features"] for r in cum_results]
            performances = [r["performance"] for r in cum_results]

            axes[0, 1].plot(num_features, performances, "bo-")
            axes[0, 1].axhline(
                y=results["baseline_performance"],
                color="r",
                linestyle="--",
                label="Baseline (All Features)",
            )
            axes[0, 1].set_xlabel("Number of Features")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].set_title("Cumulative Feature Addition")
            axes[0, 1].legend()

            # Mark optimal point
            optimal_num = results["cumulative_addition"]["optimal_num_features"]
            if optimal_num <= len(performances):
                axes[0, 1].axvline(
                    x=optimal_num,
                    color="g",
                    linestyle=":",
                    label=f"Optimal ({optimal_num} features)",
                )
                axes[0, 1].legend()

        # 3. Group ablation comparison
        if "group_ablation" in results:
            group_results = results["group_ablation"]
            group_names = list(group_results.keys())
            group_performances = [
                group_results[name]["performance"] for name in group_names
            ]

            bars = axes[0, 2].bar(range(len(group_names)), group_performances)
            axes[0, 2].axhline(
                y=results["baseline_performance"],
                color="r",
                linestyle="--",
                label="Baseline",
            )
            axes[0, 2].set_xticks(range(len(group_names)))
            axes[0, 2].set_xticklabels(group_names, rotation=45)
            axes[0, 2].set_ylabel("Accuracy")
            axes[0, 2].set_title("Feature Group Performance")
            axes[0, 2].legend()

        # 4. Random combinations analysis
        if "random_combinations" in results:
            random_results = results["random_combinations"]
            sizes = []
            means = []
            stds = []

            for result in random_results.values():
                sizes.append(result["subset_size"])
                means.append(result["mean_performance"])
                stds.append(result["std_performance"])

            axes[1, 0].errorbar(sizes, means, yerr=stds, marker="o", capsize=5)
            axes[1, 0].axhline(
                y=results["baseline_performance"],
                color="r",
                linestyle="--",
                label="Baseline",
            )
            axes[1, 0].set_xlabel("Random Subset Size")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].set_title("Random Feature Combinations")
            axes[1, 0].legend()

        # 5. Feature importance distribution (leave-one-out)
        if "leave_one_out" in results:
            importance_scores = [
                f["importance_score"]
                for f in results["leave_one_out"]["feature_importance"]
            ]
            axes[1, 1].hist(importance_scores, bins=20, edgecolor="black")
            axes[1, 1].set_xlabel("Importance Score")
            axes[1, 1].set_ylabel("Number of Features")
            axes[1, 1].set_title("Feature Importance Distribution")
            axes[1, 1].axvline(
                x=np.mean(importance_scores),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(importance_scores):.4f}",
            )
            axes[1, 1].legend()

        # 6. Consensus features
        if "synthesis" in results and results["synthesis"]["top_features_consensus"]:
            consensus_features = results["synthesis"]["top_features_consensus"][:15]

            # Count how many methods agree on each feature
            feature_counts = {}
            if "leave_one_out" in results:
                for f in results["leave_one_out"]["most_important_features"][:10]:
                    feature_counts[f] = feature_counts.get(f, 0) + 1
            if "cumulative_addition" in results:
                for f in results["cumulative_addition"]["feature_order"][:10]:
                    feature_counts[f] = feature_counts.get(f, 0) + 1

            consensus_counts = [feature_counts.get(f, 0) for f in consensus_features]

            axes[1, 2].barh(range(len(consensus_features)), consensus_counts)
            axes[1, 2].set_yticks(range(len(consensus_features)))
            axes[1, 2].set_yticklabels([f"Feature {f}" for f in consensus_features])
            axes[1, 2].set_xlabel("Number of Methods Agreeing")
            axes[1, 2].set_title("Feature Importance Consensus")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "feature_ablation_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_ablation_summary(self, results: Dict[str, Any]):
        """Save feature ablation summary."""
        summary = {
            "baseline_performance": results["baseline_performance"],
            "top_consensus_features": results["synthesis"]["top_features_consensus"][
                :10
            ],
            "key_insights": results["synthesis"]["key_insights"],
        }

        # Add method-specific top features
        if "leave_one_out" in results:
            summary["leave_one_out_top_features"] = results["leave_one_out"][
                "most_important_features"
            ]
            summary["max_performance_drop"] = results["leave_one_out"][
                "max_performance_drop"
            ]

        if "cumulative_addition" in results:
            summary["optimal_num_features"] = results["cumulative_addition"][
                "optimal_num_features"
            ]
            summary["cumulative_top_features"] = results["cumulative_addition"][
                "feature_order"
            ][:10]

        if "group_ablation" in results:
            summary["group_performance"] = {
                name: result["performance"]
                for name, result in results["group_ablation"].items()
            }

        self.save_results(summary, "feature_ablation_summary.json")
