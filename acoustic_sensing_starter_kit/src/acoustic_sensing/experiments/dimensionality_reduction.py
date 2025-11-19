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

        # For dimensionality reduction, combine all batch data for comprehensive analysis
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

        results = {}

        # PCA Analysis
        if self.config.get("pca_enabled", True):
            self.logger.info("Performing PCA analysis...")
            results.update(self._perform_pca_analysis(X, y, batch_results))

        # t-SNE Analysis
        if self.config.get("tsne_enabled", True):
            self.logger.info("Performing t-SNE analysis...")
            results.update(self._perform_tsne_analysis(X, y, batch_results))

        # UMAP Analysis (if enabled)
        if self.config.get("include_umap", False):
            self.logger.info("Performing UMAP analysis...")
            results.update(self._perform_umap_analysis(X, y, batch_results))

        self.logger.info("Dimensionality reduction experiment completed")
        return results

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
        self.save_results(pca_summary, "pca_summary.json")

        return pca_results

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

                # Create visualization for this perplexity (temporarily disabled)
                # self._create_tsne_visualization(X_tsne, y, perplexity, batch_results)

            except Exception as e:
                self.logger.error(
                    f"Error running t-SNE with perplexity {perplexity}: {str(e)}"
                )
                continue

        return {"tsne_results": tsne_results}

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

            return {"umap_embedding": X_umap, "umap_model": reducer}

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
