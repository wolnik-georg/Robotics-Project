from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import os


class DimensionalityReductionExperiment(BaseExperiment):
    """
    Experiment for dimensionality reduction analysis using PCA and t-SNE.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform dimensionality reduction analysis.

        Args:
            shared_data: Dictionary containing loaded features and labels

        Returns:
            Dictionary containing PCA and t-SNE results
        """
        self.logger.info("Starting dimensionality reduction experiment...")

        # Load per-batch data from previous experiment
        batch_results = self.load_shared_data(shared_data, "batch_results")

        per_batch_results = {}

        for batch_name, batch_data in batch_results.items():
            self.logger.info(f"Processing dimensionality reduction for {batch_name}...")

            X_batch = batch_data["features"]
            y_batch = batch_data["labels"]

            # Skip batch if insufficient data
            if len(X_batch) < 10:
                self.logger.warning(
                    f"Skipping {batch_name}: insufficient data ({len(X_batch)} samples)"
                )
                continue

            unique_classes = np.unique(y_batch)
            if len(unique_classes) < 2:
                self.logger.warning(f"Skipping {batch_name}: single class data")
                continue

            self.logger.info(
                f"Batch {batch_name}: {len(X_batch)} samples, {X_batch.shape[1]} features, {len(unique_classes)} classes"
            )

            try:
                # Perform per-batch dimensionality reduction
                batch_reduction_results = self._perform_batch_dimensionality_reduction(
                    X_batch, y_batch, batch_name, batch_data
                )
                per_batch_results[batch_name] = batch_reduction_results

            except Exception as e:
                self.logger.error(f"Error processing {batch_name}: {str(e)}")
                continue

        # Aggregate results across batches
        aggregated_results = self._aggregate_reduction_results(per_batch_results)

        results = {
            "per_batch_results": per_batch_results,
            "aggregated_results": aggregated_results,
            "total_batches_analyzed": len(per_batch_results),
            "skipped_batches": len(batch_results) - len(per_batch_results),
        }

        # Create cross-batch visualizations
        if per_batch_results:
            self._create_cross_batch_reduction_visualizations(per_batch_results)

        # Save summary
        self._save_dimensionality_reduction_summary(
            per_batch_results, aggregated_results
        )

        self.logger.info("Dimensionality reduction experiment completed")
        return results

    def _perform_batch_dimensionality_reduction(
        self, X: np.ndarray, y: np.ndarray, batch_name: str, batch_data: dict
    ) -> dict:
        """Perform dimensionality reduction analysis for a single batch."""
        batch_results = {"batch_name": batch_name}

        # PCA Analysis
        if self.config.get("pca_enabled", True):
            self.logger.info(f"Performing PCA analysis for {batch_name}...")
            batch_results.update(
                self._perform_pca_analysis(X, y, {batch_name: batch_data})
            )

        # t-SNE Analysis
        if self.config.get("tsne_enabled", True):
            self.logger.info(f"Performing t-SNE analysis for {batch_name}...")
            batch_results.update(
                self._perform_tsne_analysis(X, y, {batch_name: batch_data})
            )

        # UMAP Analysis (if enabled)
        if self.config.get("include_umap", False):
            self.logger.info(f"Performing UMAP analysis for {batch_name}...")
            batch_results.update(
                self._perform_umap_analysis(X, y, {batch_name: batch_data})
            )

        # Save batch-specific results
        self._save_batch_reduction_results(batch_results, batch_name, X, y)

        return batch_results

    def _save_batch_reduction_results(
        self, batch_results: dict, batch_name: str, X: np.ndarray, y: np.ndarray
    ):
        """Save detailed dimensionality reduction results for a specific batch."""
        import json

        # Create batch-specific output directory
        batch_output_dir = os.path.join(self.experiment_output_dir, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        # Generate batch-specific plots
        self._create_batch_plots(batch_results, batch_name, X, y, batch_output_dir)

        # Create a serializable version of the results
        serializable_results = {
            "batch_name": batch_name,
            "num_samples": len(X),
            "original_dimensions": X.shape[1],
            "num_classes": len(np.unique(y)),
            "classes": np.unique(y).tolist(),
        }

        # Handle PCA results
        if "pca_results" in batch_results:
            pca_res = batch_results["pca_results"]
            serializable_results["pca_analysis"] = {
                "explained_variance_ratio": pca_res[
                    "explained_variance_ratio"
                ].tolist(),
                "cumulative_variance": pca_res["cumulative_variance"].tolist(),
                "components_for_95_variance": int(pca_res["n_components_95_variance"]),
                "n_components_used": int(pca_res["pca_model"].n_components_),
                "total_explained_variance": float(
                    np.sum(pca_res["explained_variance_ratio"])
                ),
            }

        # Handle t-SNE results
        if "tsne_results" in batch_results:
            tsne_res = batch_results["tsne_results"]
            tsne_summary = {
                "methods_tested": list(tsne_res.keys()),
                "best_perplexity": tsne_res.get("best_perplexity", None),
                "best_silhouette_score": tsne_res.get("silhouette_score", None),
            }

            # Add perplexity details
            perplexity_details = {}
            for method, data in tsne_res.items():
                if isinstance(data, dict) and "silhouette_score" in data:
                    perplexity_details[method] = {
                        "perplexity": data["perplexity"],
                        "silhouette_score": data["silhouette_score"],
                        "kl_divergence": data["kl_divergence"],
                    }

            tsne_summary["perplexity_results"] = perplexity_details
            serializable_results["tsne_analysis"] = tsne_summary

        # Handle UMAP results
        if "umap_results" in batch_results:
            umap_res = batch_results["umap_results"]
            serializable_results["umap_analysis"] = {
                "silhouette_score": umap_res.get("silhouette_score", None)
            }

        # Save full batch results
        results_path = os.path.join(
            batch_output_dir, f"{batch_name}_reduction_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Create batch summary
        batch_summary = {
            "batch_name": batch_name,
            "num_samples": len(X),
            "original_dimensions": X.shape[1],
            "num_classes": len(np.unique(y)),
            "analysis_methods": [],
        }

        if "pca_results" in batch_results:
            pca_info = batch_results["pca_results"]
            batch_summary["analysis_methods"].append("PCA")
            batch_summary["pca_components_95"] = int(
                pca_info["n_components_95_variance"]
            )
            batch_summary["first_component_variance"] = float(
                pca_info["explained_variance_ratio"][0]
            )

        if "tsne_results" in batch_results:
            batch_summary["analysis_methods"].append("t-SNE")
            batch_summary["best_tsne_silhouette"] = batch_results["tsne_results"].get(
                "silhouette_score", None
            )

        if "umap_results" in batch_results:
            batch_summary["analysis_methods"].append("UMAP")

        # Save batch summary
        summary_path = os.path.join(
            batch_output_dir, f"{batch_name}_reduction_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2, default=str)

        self.logger.info(
            f"Batch {batch_name} reduction results saved to: {batch_output_dir}"
        )

    def _create_batch_plots(
        self,
        batch_results: dict,
        batch_name: str,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str,
    ):
        """Create visualization plots for a specific batch."""
        # Create single-batch info structure for plotting methods
        batch_info = {batch_name: {"indices": np.arange(len(X))}}

        # Create PCA plots if PCA was performed
        if "pca_results" in batch_results:
            pca_results = batch_results["pca_results"]
            X_pca = pca_results["pca_components"]
            explained_var = pca_results["explained_variance_ratio"]
            cumulative_var = pca_results["cumulative_variance"]

            self._create_batch_pca_plot(
                X_pca, y, explained_var, cumulative_var, batch_name, output_dir
            )

        # Create t-SNE plots if t-SNE was performed
        if "tsne_results" in batch_results:
            tsne_results = batch_results["tsne_results"]
            for method_name, tsne_data in tsne_results.items():
                if isinstance(tsne_data, dict) and "embedding" in tsne_data:
                    X_tsne = tsne_data["embedding"]
                    perplexity = tsne_data.get("perplexity", "unknown")
                    self._create_batch_tsne_plot(
                        X_tsne, y, perplexity, batch_name, output_dir
                    )

        # Create UMAP plots if UMAP was performed
        if "umap_results" in batch_results:
            umap_results = batch_results["umap_results"]
            if "embedding" in umap_results:
                X_umap = umap_results["embedding"]
                self._create_batch_umap_plot(X_umap, y, batch_name, output_dir)

    def _create_batch_pca_plot(
        self,
        X_pca: np.ndarray,
        y: np.ndarray,
        explained_var: np.ndarray,
        cumulative_var: np.ndarray,
        batch_name: str,
        output_dir: str,
    ):
        """Create PCA cluster projection visualization for a single batch."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Fixed color map for base class names (strip numbers after underscore)
        color_map = {
            "contact": "red",
            "no_contact": "blue",
            "edge": "green",
        }
        default_color = "gray"  # For any other labels

        # 2D PCA plot colored by class
        unique_classes = np.unique(y)
        for cls in unique_classes:
            # Extract base class name (everything before first underscore)
            base_cls = cls.split("_")[0] if "_" in cls else cls
            mask = y == cls
            color = color_map.get(base_cls, default_color)
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=color,
                label=base_cls,  # Use base class name in legend
                alpha=0.7,
                s=50,
            )
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")
        ax.set_title(f"PCA Projection - {batch_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_pca_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_batch_tsne_plot(
        self,
        X_tsne: np.ndarray,
        y: np.ndarray,
        perplexity: int,
        batch_name: str,
        output_dir: str,
    ):
        """Create t-SNE visualization for a single batch."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Fixed color map for base class names (strip numbers after underscore)
        color_map = {
            "contact": "red",
            "no_contact": "blue",
            "edge": "green",
        }
        default_color = "gray"  # For any other labels

        # Plot colored by class
        unique_classes = np.unique(y)
        for cls in unique_classes:
            # Extract base class name (everything before first underscore)
            base_cls = cls.split("_")[0] if "_" in cls else cls
            mask = y == cls
            color = color_map.get(base_cls, default_color)
            ax.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=color,
                label=base_cls,  # Use base class name in legend
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(f"t-SNE Visualization (perplexity={perplexity}) - {batch_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_tsne_perplexity_{perplexity}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_batch_umap_plot(
        self, X_umap: np.ndarray, y: np.ndarray, batch_name: str, output_dir: str
    ):
        """Create UMAP visualization for a single batch."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Fixed color map for base class names (strip numbers after underscore)
        color_map = {
            "surface": "red",
            "no_surface": "blue",
            "edge": "green",
        }
        default_color = "gray"  # For any other labels

        # Plot colored by class
        unique_classes = np.unique(y)
        for cls in unique_classes:
            # Extract base class name (everything before first underscore)
            base_cls = cls.split("_")[0] if "_" in cls else cls
            mask = y == cls
            color = color_map.get(base_cls, default_color)
            ax.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                c=color,
                label=base_cls,  # Use base class name in legend
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_title(f"UMAP Visualization - {batch_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_umap_projection.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _aggregate_reduction_results(self, per_batch_results: dict) -> dict:
        """Aggregate dimensionality reduction results across batches."""
        if not per_batch_results:
            return {}

        aggregated = {"pca_summary": {}, "tsne_summary": {}, "batch_summaries": {}}

        # Collect PCA results across batches
        pca_explained_variances = []
        pca_components_needed = []

        # Collect t-SNE results
        tsne_silhouette_scores = []

        for batch_name, batch_results in per_batch_results.items():
            # Store batch summary
            aggregated["batch_summaries"][batch_name] = {
                "num_samples": batch_results.get("num_samples", 0),
                "original_dimensions": batch_results.get("original_dimensions", 0),
            }

            # Aggregate PCA results
            if "pca_results" in batch_results:
                pca_res = batch_results["pca_results"]
                if "explained_variance_ratio" in pca_res:
                    cumsum_var = np.cumsum(pca_res["explained_variance_ratio"])
                    # Find components needed for 95% variance
                    components_95 = np.argmax(cumsum_var >= 0.95) + 1
                    pca_components_needed.append(components_95)
                    pca_explained_variances.append(
                        pca_res["explained_variance_ratio"][:10]
                    )  # Top 10

                    aggregated["batch_summaries"][batch_name][
                        "pca_components_95"
                    ] = components_95
                    aggregated["batch_summaries"][batch_name][
                        "first_component_variance"
                    ] = pca_res["explained_variance_ratio"][0]

            # Aggregate t-SNE results
            if (
                "tsne_results" in batch_results
                and "silhouette_score" in batch_results["tsne_results"]
            ):
                score = batch_results["tsne_results"]["silhouette_score"]
                tsne_silhouette_scores.append(score)
                aggregated["batch_summaries"][batch_name]["tsne_silhouette"] = score

        # Calculate overall PCA statistics
        if pca_explained_variances:
            aggregated["pca_summary"] = {
                "avg_components_for_95_variance": (
                    np.mean(pca_components_needed) if pca_components_needed else 0
                ),
                "std_components_for_95_variance": (
                    np.std(pca_components_needed) if pca_components_needed else 0
                ),
                "min_components_needed": (
                    np.min(pca_components_needed) if pca_components_needed else 0
                ),
                "max_components_needed": (
                    np.max(pca_components_needed) if pca_components_needed else 0
                ),
            }

        # Calculate overall t-SNE statistics
        if tsne_silhouette_scores:
            aggregated["tsne_summary"] = {
                "avg_silhouette_score": np.mean(tsne_silhouette_scores),
                "std_silhouette_score": np.std(tsne_silhouette_scores),
                "best_silhouette_score": np.max(tsne_silhouette_scores),
                "worst_silhouette_score": np.min(tsne_silhouette_scores),
            }

        return aggregated

    def _create_cross_batch_reduction_visualizations(self, per_batch_results: dict):
        """Create visualizations comparing reduction results across batches."""
        try:
            import matplotlib.pyplot as plt

            # PCA variance comparison
            batch_names = list(per_batch_results.keys())
            components_needed = []
            first_component_vars = []

            for batch_name in batch_names:
                batch_res = per_batch_results[batch_name]
                if (
                    "pca_results" in batch_res
                    and "explained_variance_ratio" in batch_res["pca_results"]
                ):
                    explained_var = batch_res["pca_results"]["explained_variance_ratio"]
                    cumsum_var = np.cumsum(explained_var)
                    components_95 = np.argmax(cumsum_var >= 0.95) + 1
                    components_needed.append(components_95)
                    first_component_vars.append(explained_var[0])
                else:
                    components_needed.append(0)
                    first_component_vars.append(0)

            # Plot components needed for 95% variance
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.bar(batch_names, components_needed)
            plt.title("PCA Components Needed for 95% Variance")
            plt.ylabel("Number of Components")
            plt.xlabel("Batch")
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            plt.bar(batch_names, first_component_vars)
            plt.title("First Principal Component Variance Explained")
            plt.ylabel("Variance Ratio")
            plt.xlabel("Batch")
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.experiment_output_dir, "cross_batch_pca_comparison.png"
                )
            )
            plt.close()

        except Exception as e:
            self.logger.warning(f"Failed to create cross-batch visualizations: {e}")

    def _save_dimensionality_reduction_summary(
        self, per_batch_results: dict, aggregated_results: dict
    ):
        """Save dimensionality reduction summary."""

        # Save detailed per-batch results
        self.save_results(
            per_batch_results, "dimensionality_reduction_per_batch_results.json"
        )

        # Create overall summary
        summary = {
            "pca_analysis": aggregated_results.get("pca_summary", {}),
            "tsne_analysis": aggregated_results.get("tsne_summary", {}),
            "batch_summaries": aggregated_results.get("batch_summaries", {}),
            "num_batches_analyzed": len(per_batch_results),
        }

        # Add batch-specific insights
        batch_insights = {}
        for batch_name, batch_results in per_batch_results.items():
            insights = {
                "original_dimensions": batch_results.get("original_dimensions", 0),
                "num_samples": batch_results.get("num_samples", 0),
            }

            if "pca_results" in batch_results:
                pca_res = batch_results["pca_results"]
                if "explained_variance_ratio" in pca_res:
                    cumsum_var = np.cumsum(pca_res["explained_variance_ratio"])
                    components_95 = np.argmax(cumsum_var >= 0.95) + 1
                    insights["pca_components_for_95_variance"] = components_95
                    insights["first_component_variance"] = pca_res[
                        "explained_variance_ratio"
                    ][0]

            if (
                "tsne_results" in batch_results
                and "silhouette_score" in batch_results["tsne_results"]
            ):
                insights["tsne_silhouette_score"] = batch_results["tsne_results"][
                    "silhouette_score"
                ]

            batch_insights[batch_name] = insights

        if batch_insights:
            summary["per_batch_insights"] = batch_insights

        self.save_results(summary, "dimensionality_reduction_summary.json")

    def _perform_pca_analysis(
        self, X: np.ndarray, y: np.ndarray, batch_info: dict
    ) -> dict:
        """Perform PCA analysis."""
        # Standardize features
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit PCA with all components first to analyze variance
        pca_full = PCA()
        pca_full.fit(X_scaled)

        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # Find number of components for specified variance threshold
        variance_threshold = self.config.get("pca_variance_threshold", 0.95)
        n_components_95 = np.argmax(cumulative_variance >= variance_threshold) + 1

        self.logger.info(
            f"Components needed for {variance_threshold*100}% variance: {n_components_95}"
        )

        # Fit PCA with reduced components
        pca_reduced = PCA(n_components=min(n_components_95, 50))  # Cap at 50 components
        X_pca = pca_reduced.fit_transform(X_scaled)

        # Create visualizations (temporarily disabled for per-batch compatibility)
        # self._create_pca_visualizations(
        #     X_pca, y, pca_reduced, cumulative_variance, batch_results
        # )

        # Analyze component contributions
        feature_importance = self._analyze_pca_components(
            pca_reduced, n_top_features=10
        )

        pca_results = {
            "pca_components": X_pca,
            "pca_model": pca_reduced,
            "explained_variance_ratio": pca_reduced.explained_variance_ratio_,
            "cumulative_variance": cumulative_variance,
            "n_components_95_variance": n_components_95,
            "feature_importance": feature_importance,
            "scaler": scaler,
        }

        # Save PCA summary
        pca_summary = {
            "n_components": pca_reduced.n_components_,
            "explained_variance_ratio": pca_reduced.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(
                np.sum(pca_reduced.explained_variance_ratio_)
            ),
            "components_for_95_variance": int(n_components_95),
        }

        # Add metadata for per-batch tracking
        pca_results["num_samples"] = len(X)
        pca_results["original_dimensions"] = X.shape[1]

        # Save batch-specific PCA summary if we have batch info
        if len(batch_info) == 1:  # Single batch analysis
            batch_name = list(batch_info.keys())[0]
            self.save_results(pca_summary, f"pca_summary_{batch_name}.json")
        else:
            self.save_results(pca_summary, "pca_summary.json")

        return {"pca_results": pca_results}

    def _perform_tsne_analysis(
        self, X: np.ndarray, y: np.ndarray, batch_info: dict
    ) -> dict:
        """Perform t-SNE analysis with multiple perplexity values."""
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use PCA preprocessing for t-SNE if data is high-dimensional
        if X_scaled.shape[1] > 50:
            pca = PCA(n_components=50)
            X_for_tsne = pca.fit_transform(X_scaled)
            self.logger.info(
                "Applied PCA preprocessing for t-SNE (reduced to 50 dimensions)"
            )
        else:
            X_for_tsne = X_scaled

        perplexity_values = self.config.get(
            "tsne_perplexity_values", [5, 10, 20, 30, 50]
        )
        tsne_results = {}

        for perplexity in perplexity_values:
            # Skip if perplexity is too large for the dataset
            if perplexity >= len(X) / 3:
                self.logger.warning(
                    f"Skipping perplexity {perplexity} (too large for dataset)"
                )
                continue

            self.logger.info(f"Running t-SNE with perplexity {perplexity}...")

            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    max_iter=1000,
                    verbose=0,
                )
                X_tsne = tsne.fit_transform(X_for_tsne)

                tsne_results[f"perplexity_{perplexity}"] = {
                    "embedding": X_tsne,
                    "perplexity": perplexity,
                    "kl_divergence": tsne.kl_divergence_,
                }

                # Calculate silhouette score for this embedding
                try:
                    from sklearn.metrics import silhouette_score

                    if (
                        len(np.unique(y)) > 1
                    ):  # Need at least 2 classes for silhouette score
                        silhouette = silhouette_score(X_tsne, y)
                        tsne_results[f"perplexity_{perplexity}"][
                            "silhouette_score"
                        ] = silhouette

                except Exception as e:
                    self.logger.warning(f"Could not calculate silhouette score: {e}")

                # Create visualization for this perplexity (temporarily disabled)
                # self._create_tsne_visualization(X_tsne, y, perplexity, batch_results)

            except Exception as e:
                self.logger.error(
                    f"Error running t-SNE with perplexity {perplexity}: {str(e)}"
                )
                continue

        # Find best perplexity based on silhouette score
        best_silhouette = -1
        best_perplexity = None
        for pkey, presult in tsne_results.items():
            if (
                "silhouette_score" in presult
                and presult["silhouette_score"] > best_silhouette
            ):
                best_silhouette = presult["silhouette_score"]
                best_perplexity = presult["perplexity"]

        # Add metadata
        tsne_final = {"tsne_results": tsne_results}
        if best_perplexity is not None:
            tsne_final["tsne_results"]["best_perplexity"] = best_perplexity
            tsne_final["tsne_results"]["silhouette_score"] = best_silhouette

        # Save batch-specific t-SNE summary if we have batch info
        tsne_summary = {
            "best_perplexity": best_perplexity,
            "best_silhouette_score": best_silhouette,
            "perplexities_tested": list(tsne_results.keys()),
        }

        if len(batch_info) == 1:  # Single batch analysis
            batch_name = list(batch_info.keys())[0]
            self.save_results(tsne_summary, f"tsne_summary_{batch_name}.json")
        else:
            self.save_results(tsne_summary, "tsne_summary.json")

        return tsne_final

    def _perform_umap_analysis(
        self, X: np.ndarray, y: np.ndarray, batch_info: dict
    ) -> dict:
        """Perform UMAP analysis if available."""
        try:
            import umap
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # UMAP with default parameters
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(X_scaled)

            # Create visualization
            # self._create_umap_visualization(X_umap, y, batch_results)

            umap_result = {"umap_embedding": X_umap, "umap_model": reducer}

            # Calculate silhouette score for UMAP embedding
            try:
                from sklearn.metrics import silhouette_score

                if len(np.unique(y)) > 1:
                    silhouette = silhouette_score(X_umap, y)
                    umap_result["silhouette_score"] = silhouette
            except Exception as e:
                self.logger.warning(f"Could not calculate UMAP silhouette score: {e}")

            return {"umap_results": umap_result}

        except ImportError:
            self.logger.warning(
                "UMAP not available. Install umap-learn for UMAP analysis."
            )
            return {}

    def _create_pca_visualizations(
        self,
        X_pca: np.ndarray,
        y: np.ndarray,
        pca_model,
        cumulative_variance: np.ndarray,
        batch_info: dict,
    ):
        """Create PCA visualizations."""
        plt.style.use("default")

        # 1. Explained Variance Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Variance explained by each component
        axes[0, 0].bar(
            range(1, len(pca_model.explained_variance_ratio_) + 1),
            pca_model.explained_variance_ratio_,
        )
        axes[0, 0].set_xlabel("Principal Component")
        axes[0, 0].set_ylabel("Explained Variance Ratio")
        axes[0, 0].set_title("Explained Variance by Component")

        # Cumulative variance
        axes[0, 1].plot(
            range(1, len(cumulative_variance) + 1), cumulative_variance, "bo-"
        )
        axes[0, 1].axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
        axes[0, 1].set_xlabel("Number of Components")
        axes[0, 1].set_ylabel("Cumulative Explained Variance")
        axes[0, 1].set_title("Cumulative Explained Variance")
        axes[0, 1].legend()

        # 2D PCA plot colored by class
        unique_classes = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

        for i, cls in enumerate(unique_classes):
            mask = y == cls
            axes[1, 0].scatter(
                X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], label=cls, alpha=0.6
            )
        axes[1, 0].set_xlabel("First Principal Component")
        axes[1, 0].set_ylabel("Second Principal Component")
        axes[1, 0].set_title("PCA Projection (Colored by Material)")
        axes[1, 0].legend()

        # 2D PCA plot colored by batch
        batch_colors = plt.cm.Set3(np.linspace(0, 1, len(batch_info)))
        for i, (batch_name, info) in enumerate(batch_info.items()):
            indices = info["indices"]
            axes[1, 1].scatter(
                X_pca[indices, 0],
                X_pca[indices, 1],
                c=[batch_colors[i]],
                label=batch_name,
                alpha=0.6,
            )
        axes[1, 1].set_xlabel("First Principal Component")
        axes[1, 1].set_ylabel("Second Principal Component")
        axes[1, 1].set_title("PCA Projection (Colored by Batch)")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "pca_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_tsne_visualization(
        self, X_tsne: np.ndarray, y: np.ndarray, perplexity: int, batch_info: dict
    ):
        """Create t-SNE visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot colored by class
        unique_classes = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

        for i, cls in enumerate(unique_classes):
            mask = y == cls
            axes[0].scatter(
                X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], label=cls, alpha=0.6
            )
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")
        axes[0].set_title(f"t-SNE (perplexity={perplexity}) - By Material")
        axes[0].legend()

        # Plot colored by batch
        batch_colors = plt.cm.Set3(np.linspace(0, 1, len(batch_info)))
        for i, (batch_name, info) in enumerate(batch_info.items()):
            indices = info["indices"]
            axes[1].scatter(
                X_tsne[indices, 0],
                X_tsne[indices, 1],
                c=[batch_colors[i]],
                label=batch_name,
                alpha=0.6,
            )
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")
        axes[1].set_title(f"t-SNE (perplexity={perplexity}) - By Batch")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.experiment_output_dir, f"tsne_perplexity_{perplexity}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_umap_visualization(
        self, X_umap: np.ndarray, y: np.ndarray, batch_info: dict
    ):
        """Create UMAP visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot colored by class
        unique_classes = np.unique(y)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

        for i, cls in enumerate(unique_classes):
            mask = y == cls
            axes[0].scatter(
                X_umap[mask, 0], X_umap[mask, 1], c=[colors[i]], label=cls, alpha=0.6
            )
        axes[0].set_xlabel("UMAP 1")
        axes[0].set_ylabel("UMAP 2")
        axes[0].set_title("UMAP Projection - By Material")
        axes[0].legend()

        # Plot colored by batch
        batch_colors = plt.cm.Set3(np.linspace(0, 1, len(batch_info)))
        for i, (batch_name, info) in enumerate(batch_info.items()):
            indices = info["indices"]
            axes[1].scatter(
                X_umap[indices, 0],
                X_umap[indices, 1],
                c=[batch_colors[i]],
                label=batch_name,
                alpha=0.6,
            )
        axes[1].set_xlabel("UMAP 1")
        axes[1].set_ylabel("UMAP 2")
        axes[1].set_title("UMAP Projection - By Batch")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "umap_projection.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _analyze_pca_components(self, pca_model, n_top_features: int = 10) -> dict:
        """Analyze which original features contribute most to each PC."""
        components = pca_model.components_
        feature_importance = {}

        for i in range(min(5, len(components))):  # Analyze first 5 components
            # Get absolute loadings for this component
            loadings = np.abs(components[i])

            # Get indices of top features
            top_indices = np.argsort(loadings)[-n_top_features:][::-1]

            feature_importance[f"PC{i+1}"] = {
                "top_features": top_indices.tolist(),
                "loadings": loadings[top_indices].tolist(),
            }

        return feature_importance
