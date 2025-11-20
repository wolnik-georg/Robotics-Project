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

        # Perform feature ablation analysis for each batch separately
        per_batch_results = {}

        for batch_name, batch_data in batch_results.items():
            self.logger.info(f"Analyzing feature ablation for batch: {batch_name}")

            X = batch_data["features"]
            y = batch_data["labels"]

            # Skip batches with insufficient data
            if len(X) < 10:
                self.logger.warning(
                    f"Skipping {batch_name}: insufficient samples ({len(X)})"
                )
                continue

            # Skip single-class batches
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                self.logger.warning(
                    f"Skipping {batch_name}: single class ({unique_classes[0]})"
                )
                continue

            self.logger.info(
                f"Batch {batch_name}: {len(X)} samples, {X.shape[1]} features across {len(unique_classes)} classes"
            )

            # Standardize features for this batch
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform ablation analysis for this batch
            batch_ablation_results = self._perform_batch_ablation_analysis(
                X_scaled, y, batch_name, scaler
            )

            per_batch_results[batch_name] = batch_ablation_results

        # Aggregate results across batches
        aggregated_results = self._aggregate_batch_results(per_batch_results)

        results = {
            "per_batch_results": per_batch_results,
            "aggregated_results": aggregated_results,
            "total_batches_analyzed": len(per_batch_results),
            "skipped_batches": len(batch_results) - len(per_batch_results),
        }

        # Save summary
        self._save_ablation_summary(per_batch_results, aggregated_results)

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

    def _save_ablation_summary(
        self, per_batch_results: Dict[str, Any], aggregated_results: Dict[str, Any]
    ):
        """Save feature ablation summary for both per-batch and aggregated results."""

        # Save detailed per-batch results
        self.save_results(per_batch_results, "feature_ablation_per_batch_results.json")

        # Create overall summary
        summary = {
            "overall_baseline_performance": aggregated_results.get(
                "overall_baseline_performance", 0
            ),
            "baseline_std": aggregated_results.get("baseline_std", 0),
            "best_batch_performance": aggregated_results.get(
                "best_batch_performance", 0
            ),
            "worst_batch_performance": aggregated_results.get(
                "worst_batch_performance", 0
            ),
            "top_consensus_features": aggregated_results.get(
                "top_features_consensus", []
            ),
            "batch_summaries": aggregated_results.get("batch_summaries", {}),
            "num_batches_analyzed": len(per_batch_results),
        }

        # Add batch-specific insights if available
        batch_insights = {}
        for batch_name, batch_results in per_batch_results.items():
            if "synthesis" in batch_results:
                batch_insights[batch_name] = {
                    "baseline_performance": batch_results["baseline_performance"],
                    "key_insights": batch_results["synthesis"].get("key_insights", []),
                    "top_features": batch_results["synthesis"].get(
                        "top_features_consensus", []
                    )[:5],
                }

        if batch_insights:
            summary["per_batch_insights"] = batch_insights

        self.save_results(summary, "feature_ablation_summary.json")

    def _perform_batch_ablation_analysis(
        self, X: np.ndarray, y: np.ndarray, batch_name: str, scaler
    ) -> dict:
        """Perform comprehensive ablation analysis for a single batch."""
        # Create batch-specific output directory
        batch_output_dir = os.path.join(self.experiment_output_dir, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        # Get baseline performance with all features
        baseline_performance = self._get_baseline_performance(X, y)
        self.logger.info(
            f"Batch {batch_name} baseline accuracy: {baseline_performance:.4f}"
        )

        results = {
            "baseline_performance": baseline_performance,
            "scaler": scaler,
            "batch_name": batch_name,
        }

        # Perform different types of ablation studies
        if self.config.get("include_leave_one_out", True):
            results["leave_one_out"] = self._perform_leave_one_out_ablation(
                X, y, baseline_performance
            )

        if self.config.get("include_cumulative_addition", True):
            results["cumulative_addition"] = self._perform_cumulative_addition(X, y)

        if self.config.get("include_group_ablation", True):
            results["group_ablation"] = self._perform_group_ablation(
                X, y, baseline_performance
            )

        if self.config.get("include_random_combinations", True):
            results["random_combinations"] = self._perform_random_combinations(X, y)

        # Analyze and synthesize results
        synthesis = self._synthesize_ablation_results(results)
        results["synthesis"] = synthesis

        # Save batch-specific results
        self._save_batch_specific_results(results, batch_name, batch_output_dir)

        # Generate batch-specific plots
        self._create_batch_plots(results, batch_name, batch_output_dir)

        return results

    def _save_batch_specific_results(
        self, results: dict, batch_name: str, output_dir: str
    ):
        """Save detailed results for a specific batch."""
        import json

        # Save full batch results
        batch_results_path = os.path.join(
            output_dir, f"{batch_name}_feature_ablation_results.json"
        )

        # Create a serializable version of the results
        serializable_results = {}
        for key, value in results.items():
            if key == "scaler":
                continue  # Skip non-serializable scaler
            elif hasattr(value, "tolist"):  # Handle numpy arrays
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

        with open(batch_results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Create batch summary
        batch_summary = {
            "batch_name": batch_name,
            "baseline_performance": results["baseline_performance"],
            "num_features": len(
                results.get("leave_one_out", {}).get("feature_importance", [])
            ),
            "classes_analyzed": results.get("classes_analyzed", "unknown"),
            "analysis_methods": list(results.keys()),
        }

        # Add top features if available
        if "leave_one_out" in results:
            batch_summary["top_features"] = results["leave_one_out"][
                "most_important_features"
            ][:10]
            batch_summary["max_performance_drop"] = results["leave_one_out"][
                "max_performance_drop"
            ]

        if "synthesis" in results:
            batch_summary["key_insights"] = results["synthesis"]["key_insights"]
            batch_summary["consensus_top_features"] = results["synthesis"][
                "top_features_consensus"
            ][:5]

        # Save batch summary
        summary_path = os.path.join(output_dir, f"{batch_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2, default=str)

        self.logger.info(f"Batch {batch_name} results saved to: {output_dir}")

    def _create_batch_plots(self, results: dict, batch_name: str, output_dir: str):
        """Create visualization plots for a specific batch."""
        try:
            # Feature importance plot
            if "leave_one_out" in results:
                self._create_feature_importance_plot(
                    results["leave_one_out"], batch_name, output_dir
                )

            # Cumulative addition plot
            if "cumulative_addition" in results:
                self._create_cumulative_performance_plot(
                    results["cumulative_addition"], batch_name, output_dir
                )

            # Random feature subset plot
            if "random_subsets" in results:
                self._create_random_subset_plot(
                    results["random_subsets"], batch_name, output_dir
                )

            # Comprehensive summary plot
            self._create_ablation_summary_plot(results, batch_name, output_dir)

        except Exception as e:
            self.logger.warning(f"Failed to create plots for batch {batch_name}: {e}")

    def _create_feature_importance_plot(
        self, loo_results: dict, batch_name: str, output_dir: str
    ):
        """Create feature importance visualization."""
        import matplotlib.pyplot as plt

        feature_importance = loo_results["feature_importance"]
        top_features = feature_importance[:20]  # Top 20 features

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Top features performance drop
        indices = [f["feature_index"] for f in top_features]
        drops = [f["performance_drop"] for f in top_features]

        ax1.barh(range(len(indices)), drops, alpha=0.7, color="steelblue")
        ax1.set_yticks(range(len(indices)))
        ax1.set_yticklabels([f"Feature {idx}" for idx in indices])
        ax1.set_xlabel("Performance Drop (Importance)")
        ax1.set_title(f"Top 20 Most Important Features - {batch_name}")
        ax1.grid(axis="x", alpha=0.3)

        # Performance drop distribution
        all_drops = [f["performance_drop"] for f in feature_importance]
        ax2.hist(all_drops, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
        ax2.axvline(
            np.mean(all_drops),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(all_drops):.3f}",
        )
        ax2.set_xlabel("Performance Drop")
        ax2.set_ylabel("Number of Features")
        ax2.set_title(f"Feature Importance Distribution - {batch_name}")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_feature_importance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_cumulative_performance_plot(
        self, cumulative_results: dict, batch_name: str, output_dir: str
    ):
        """Create cumulative feature addition performance plot."""
        import matplotlib.pyplot as plt

        if "performance_progression" not in cumulative_results:
            return

        progression = cumulative_results["performance_progression"]
        feature_counts = [p["n_features"] for p in progression]
        accuracies = [p["accuracy"] for p in progression]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(
            feature_counts,
            accuracies,
            "o-",
            linewidth=2,
            markersize=6,
            color="darkgreen",
        )
        ax.axhline(
            y=cumulative_results.get("baseline_performance", 0),
            color="red",
            linestyle="--",
            label="Baseline (All Features)",
        )

        ax.set_xlabel("Number of Features Used")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Cumulative Feature Addition Performance - {batch_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Annotate the elbow point if it exists
        if len(accuracies) > 3:
            # Find elbow point (where improvement slows down)
            diffs = np.diff(accuracies)
            if len(diffs) > 2:
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(second_diffs) + 2
                if elbow_idx < len(feature_counts):
                    ax.annotate(
                        f"Elbow: {feature_counts[elbow_idx]} features",
                        xy=(feature_counts[elbow_idx], accuracies[elbow_idx]),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                    )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_cumulative_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_random_subset_plot(
        self, random_results: dict, batch_name: str, output_dir: str
    ):
        """Create random feature subset performance plot."""
        import matplotlib.pyplot as plt

        sizes = []
        means = []
        stds = []

        for size_key, stats in random_results.items():
            if size_key.startswith("size_"):
                size = int(size_key.split("_")[1])
                sizes.append(size)
                means.append(stats["mean_accuracy"])
                stds.append(stats["std_accuracy"])

        if not sizes:
            return

        # Sort by size
        sorted_data = sorted(zip(sizes, means, stds))
        sizes, means, stds = zip(*sorted_data)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.errorbar(
            sizes,
            means,
            yerr=stds,
            fmt="o-",
            capsize=5,
            linewidth=2,
            markersize=8,
            color="purple",
            label="Random subsets",
        )
        ax.axhline(
            y=random_results.get("baseline_performance", 0),
            color="red",
            linestyle="--",
            label="Baseline (All Features)",
        )

        ax.set_xlabel("Number of Random Features")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Random Feature Subset Performance - {batch_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_random_subsets.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_ablation_summary_plot(
        self, results: dict, batch_name: str, output_dir: str
    ):
        """Create comprehensive ablation analysis summary plot."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Top features bar plot
        ax1 = fig.add_subplot(gs[0, :])
        if "leave_one_out" in results:
            top_features = results["leave_one_out"]["feature_importance"][:15]
            indices = [f["feature_index"] for f in top_features]
            drops = [f["performance_drop"] for f in top_features]

            bars = ax1.bar(range(len(indices)), drops, alpha=0.7, color="steelblue")
            ax1.set_xticks(range(len(indices)))
            ax1.set_xticklabels([f"F{idx}" for idx in indices], rotation=45)
            ax1.set_ylabel("Performance Drop")
            ax1.set_title(f"Top 15 Most Important Features - {batch_name}")
            ax1.grid(axis="y", alpha=0.3)

            # Color code the bars by importance
            max_drop = max(drops)
            for bar, drop in zip(bars, drops):
                intensity = drop / max_drop
                bar.set_color(plt.cm.RdYlBu_r(intensity))

        # 2. Performance comparison
        ax2 = fig.add_subplot(gs[1, 0])
        methods = []
        performances = []

        if "baseline_performance" in results:
            methods.append("Baseline\n(All Features)")
            performances.append(results["baseline_performance"])

        if (
            "cumulative_addition" in results
            and "best_performance" in results["cumulative_addition"]
        ):
            methods.append("Best Subset\n(Cumulative)")
            performances.append(results["cumulative_addition"]["best_performance"])

        if "random_subsets" in results:
            # Get best random performance
            best_random = 0
            for key, stats in results["random_subsets"].items():
                if key.startswith("size_") and "mean_accuracy" in stats:
                    best_random = max(best_random, stats["mean_accuracy"])
            if best_random > 0:
                methods.append("Best Random\nSubset")
                performances.append(best_random)

        if methods and performances:
            bars = ax2.bar(
                methods,
                performances,
                alpha=0.7,
                color=["green", "blue", "orange"][: len(methods)],
            )
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Performance Comparison")
            ax2.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, perf in zip(bars, performances):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.001,
                    f"{perf:.3f}",
                    ha="center",
                    va="bottom",
                )

        # 3. Cumulative performance if available
        ax3 = fig.add_subplot(gs[1, 1])
        if (
            "cumulative_addition" in results
            and "performance_progression" in results["cumulative_addition"]
        ):
            progression = results["cumulative_addition"]["performance_progression"]
            feature_counts = [p["n_features"] for p in progression]
            accuracies = [p["accuracy"] for p in progression]

            ax3.plot(
                feature_counts,
                accuracies,
                "o-",
                linewidth=2,
                markersize=4,
                color="darkgreen",
            )
            ax3.set_xlabel("Number of Features")
            ax3.set_ylabel("Accuracy")
            ax3.set_title("Cumulative Addition Performance")
            ax3.grid(True, alpha=0.3)

        # 4. Feature importance distribution
        ax4 = fig.add_subplot(gs[2, :])
        if "leave_one_out" in results:
            all_drops = [
                f["performance_drop"]
                for f in results["leave_one_out"]["feature_importance"]
            ]
            ax4.hist(
                all_drops, bins=25, alpha=0.7, color="lightcoral", edgecolor="black"
            )
            ax4.axvline(
                np.mean(all_drops),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(all_drops):.4f}",
            )
            ax4.axvline(
                np.median(all_drops),
                color="blue",
                linestyle="--",
                label=f"Median: {np.median(all_drops):.4f}",
            )
            ax4.set_xlabel("Performance Drop (Feature Importance)")
            ax4.set_ylabel("Number of Features")
            ax4.set_title("Feature Importance Distribution")
            ax4.legend()
            ax4.grid(alpha=0.3)

        plt.suptitle(
            f"Feature Ablation Analysis Summary - {batch_name}", fontsize=16, y=0.98
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_ablation_summary.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _aggregate_batch_results(self, per_batch_results: dict) -> dict:
        """Aggregate results across all batches."""
        if not per_batch_results:
            return {}

        aggregated = {
            "overall_baseline_performance": 0,
            "top_features_consensus": [],
            "performance_statistics": {},
            "batch_summaries": {},
        }

        # Calculate average baseline performance
        baseline_performances = []
        all_top_features = []

        for batch_name, batch_results in per_batch_results.items():
            baseline_performances.append(batch_results["baseline_performance"])

            # Collect top features from each batch
            if "leave_one_out" in batch_results:
                top_features = batch_results["leave_one_out"][
                    "most_important_features"
                ][:5]
                all_top_features.extend(top_features)

            # Store batch summary
            aggregated["batch_summaries"][batch_name] = {
                "baseline_performance": batch_results["baseline_performance"],
                "num_features": len(
                    batch_results.get("leave_one_out", {}).get("feature_importance", [])
                ),
                "top_feature": batch_results.get("leave_one_out", {}).get(
                    "most_important_features", [None]
                )[0],
            }

        # Calculate overall statistics
        aggregated["overall_baseline_performance"] = np.mean(baseline_performances)
        aggregated["baseline_std"] = np.std(baseline_performances)
        aggregated["best_batch_performance"] = max(baseline_performances)
        aggregated["worst_batch_performance"] = min(baseline_performances)

        # Find consensus top features (most commonly important across batches)
        if all_top_features:
            from collections import Counter

            feature_counts = Counter(all_top_features)
            aggregated["top_features_consensus"] = [
                feature for feature, count in feature_counts.most_common(10)
            ]

        return aggregated
