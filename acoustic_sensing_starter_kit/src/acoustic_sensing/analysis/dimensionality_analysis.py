"""
Dimensionality Reduction and Visualization for Geometric Discrimination
=====================================================================

This module provides comprehensive t-SNE, PCA, and other dimensionality
reduction techniques specifically designed for analyzing acoustic geometric
discrimination capability. It includes:

1. Multiple dimensionality reduction algorithms (t-SNE, PCA, UMAP)
2. Cluster analysis and separability metrics
3. Publication-ready visualizations
4. Cross-validation and robustness analysis
5. Feature importance analysis

Author: Enhanced for geometric discrimination analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn(
        "UMAP not available. Install with 'pip install umap-learn' for additional analysis options."
    )


class GeometricDimensionalityAnalyzer:
    """
    Advanced dimensionality reduction and analysis for geometric discrimination.

    Provides multiple reduction techniques, visualization tools, and statistical
    analysis to prove geometric discrimination capability in acoustic sensing.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}

    def fit_transform_pca(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
        explained_variance_threshold: float = 0.95,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform PCA analysis with comprehensive statistics.

        Args:
            X: Feature matrix (n_samples, n_features)
            n_components: Number of components. If None, chosen to explain variance threshold
            explained_variance_threshold: Minimum explained variance ratio

        Returns:
            X_pca: Transformed data
            results: Dictionary with PCA statistics and model
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Initial PCA to determine optimal components
        if n_components is None:
            pca_full = PCA(random_state=self.random_state)
            pca_full.fit(X_scaled)

            # Find number of components for threshold
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum_var >= explained_variance_threshold) + 1
            n_components = min(n_components, min(X.shape) - 1)  # Ensure valid

        # Final PCA with optimal components
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        results = {
            "model": pca,
            "scaler": self.scaler,
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "total_explained_variance": np.sum(pca.explained_variance_ratio_),
            "components": pca.components_,
            "singular_values": pca.singular_values_,
            "feature_importance": np.abs(pca.components_).mean(axis=0),
        }

        self.results["pca"] = results
        return X_pca, results

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Apply pre-fitted PCA transformation to new data.

        Args:
            X: Feature matrix to transform (n_samples, n_features)

        Returns:
            X_pca: Transformed data using the fitted PCA model
        """
        if "pca" not in self.results:
            raise ValueError("PCA model not fitted. Call fit_transform_pca first.")

        # Use the stored scaler and PCA model
        X_scaled = self.results["pca"]["scaler"].transform(X)
        X_pca = self.results["pca"]["model"].transform(X_scaled)

        return X_pca

    def fit_transform_tsne(
        self,
        X: np.ndarray,
        n_components: int = 2,
        perplexity: Union[float, List[float]] = [30.0],
        n_iter: int = 1000,
        learning_rate: str = "auto",
        early_exaggeration: float = 12.0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform t-SNE analysis with multiple perplexity values for robustness.

        Args:
            X: Feature matrix (n_samples, n_features)
            n_components: Number of output dimensions (usually 2)
            perplexity: Perplexity value(s) to test
            n_iter: Number of iterations
            learning_rate: Learning rate ('auto' or float)
            early_exaggeration: Early exaggeration factor

        Returns:
            X_tsne: Transformed data (for first perplexity if multiple)
            results: Dictionary with t-SNE results for all perplexity values
        """
        # Ensure perplexity is a list
        if isinstance(perplexity, (int, float)):
            perplexity = [perplexity]

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        results = {
            "perplexity_results": {},
            "optimal_perplexity": None,
            "best_embedding": None,
        }

        best_stress = float("inf")
        best_perp = perplexity[0]

        for perp in perplexity:
            try:
                # Check sklearn version compatibility for t-SNE parameters
                import sklearn

                sklearn_version = tuple(map(int, sklearn.__version__.split(".")[:2]))

                if sklearn_version >= (1, 2):
                    # Newer sklearn versions use max_iter instead of n_iter
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=min(
                            perp, (len(X_scaled) - 1) / 3
                        ),  # Ensure valid perplexity
                        max_iter=n_iter,
                        learning_rate=learning_rate,
                        early_exaggeration=early_exaggeration,
                        random_state=self.random_state,
                        init="pca",
                    )
                else:
                    # Older sklearn versions use n_iter
                    tsne = TSNE(
                        n_components=n_components,
                        perplexity=min(
                            perp, (len(X_scaled) - 1) / 3
                        ),  # Ensure valid perplexity
                        n_iter=n_iter,
                        learning_rate=learning_rate,
                        early_exaggeration=early_exaggeration,
                        random_state=self.random_state,
                        init="pca",
                    )

                X_tsne = tsne.fit_transform(X_scaled)

                # Calculate stress (KL divergence)
                stress = tsne.kl_divergence_

                # Get final iteration count (different attribute names in different versions)
                n_iter_final = getattr(
                    tsne, "n_iter_", getattr(tsne, "n_iter_final_", n_iter)
                )
                learning_rate_final = getattr(tsne, "learning_rate_", learning_rate)

                results["perplexity_results"][perp] = {
                    "embedding": X_tsne,
                    "stress": stress,
                    "n_iter_final": n_iter_final,
                    "learning_rate_final": learning_rate_final,
                }

                # Track best result
                if stress < best_stress:
                    best_stress = stress
                    best_perp = perp

            except Exception as e:
                warnings.warn(f"t-SNE failed for perplexity {perp}: {e}")
                continue

        if results["perplexity_results"]:
            results["optimal_perplexity"] = best_perp
            results["best_embedding"] = results["perplexity_results"][best_perp][
                "embedding"
            ]
            best_embedding = results["best_embedding"]
        else:
            raise RuntimeError("All t-SNE runs failed")

        self.results["tsne"] = results
        return best_embedding, results

    def fit_transform_umap(
        self,
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform UMAP analysis (if available).

        Args:
            X: Feature matrix (n_samples, n_features)
            n_components: Number of output dimensions
            n_neighbors: Number of neighbors for manifold approximation
            min_dist: Minimum distance between points in embedding
            metric: Distance metric

        Returns:
            X_umap: Transformed data
            results: Dictionary with UMAP results
        """
        if not HAS_UMAP:
            raise ImportError(
                "UMAP not available. Install with 'pip install umap-learn'"
            )

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_state,
        )

        X_umap = umap_model.fit_transform(X_scaled)

        results = {
            "model": umap_model,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
        }

        self.results["umap"] = results
        return X_umap, results

    def analyze_separability(
        self, X_reduced: np.ndarray, labels: np.ndarray, method_name: str = "unknown"
    ) -> Dict:
        """
        Comprehensive separability analysis for geometric discrimination.

        Args:
            X_reduced: Reduced-dimension data
            labels: Class labels
            method_name: Name of reduction method for reporting

        Returns:
            Dictionary with separability metrics
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        results = {
            "method": method_name,
            "n_classes": n_classes,
            "n_samples": len(labels),
            "class_distribution": {
                str(label): np.sum(labels == label) for label in unique_labels
            },
        }

        # 1. Silhouette Score
        if len(unique_labels) > 1:
            silhouette = silhouette_score(X_reduced, labels)
            results["silhouette_score"] = silhouette
        else:
            results["silhouette_score"] = -1

        # 2. Intra-class vs Inter-class distances
        intra_distances = []
        inter_distances = []

        for label in unique_labels:
            class_indices = labels == label
            class_data = X_reduced[class_indices]

            if len(class_data) > 1:
                # Intra-class distances
                intra_dist = pdist(class_data)
                intra_distances.extend(intra_dist)

                # Inter-class distances
                for other_label in unique_labels:
                    if other_label != label:
                        other_indices = labels == other_label
                        other_data = X_reduced[other_indices]

                        for point in class_data:
                            for other_point in other_data:
                                dist = np.linalg.norm(point - other_point)
                                inter_distances.append(dist)

        if intra_distances and inter_distances:
            results["mean_intra_distance"] = np.mean(intra_distances)
            results["mean_inter_distance"] = np.mean(inter_distances)
            results["separation_ratio"] = np.mean(inter_distances) / np.mean(
                intra_distances
            )
        else:
            results["mean_intra_distance"] = 0
            results["mean_inter_distance"] = 0
            results["separation_ratio"] = 0

        # 3. Class-wise centroid distances
        centroids = {}
        for label in unique_labels:
            class_data = X_reduced[labels == label]
            centroids[label] = np.mean(class_data, axis=0)

        centroid_distances = {}
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j:
                    dist = np.linalg.norm(centroids[label1] - centroids[label2])
                    centroid_distances[f"{label1}_to_{label2}"] = dist

        results["centroid_distances"] = centroid_distances
        results["mean_centroid_distance"] = (
            np.mean(list(centroid_distances.values())) if centroid_distances else 0
        )

        # 4. Statistical separability (ANOVA-like)
        if X_reduced.shape[1] >= 1:
            separability_stats = []
            for dim in range(X_reduced.shape[1]):
                groups = [X_reduced[labels == label, dim] for label in unique_labels]
                if all(len(group) > 0 for group in groups):
                    f_stat, p_value = f_oneway(*groups)
                    separability_stats.append(
                        {"dimension": dim, "f_statistic": f_stat, "p_value": p_value}
                    )

            results["anova_results"] = separability_stats
            if separability_stats:
                results["mean_f_statistic"] = np.mean(
                    [s["f_statistic"] for s in separability_stats]
                )
                results["mean_p_value"] = np.mean(
                    [s["p_value"] for s in separability_stats]
                )
            else:
                results["mean_f_statistic"] = 0
                results["mean_p_value"] = 1

        return results

    def plot_2d_embedding(
        self,
        X_reduced: np.ndarray,
        labels: np.ndarray,
        title: str = "Dimensionality Reduction",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None,
        show_centroids: bool = True,
        alpha: float = 0.7,
        point_size: float = 50,
    ) -> plt.Figure:
        """
        Create publication-ready 2D embedding visualization.

        Args:
            X_reduced: 2D reduced data
            labels: Class labels
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure (optional)
            show_centroids: Whether to show class centroids
            alpha: Point transparency
            point_size: Size of scatter points

        Returns:
            matplotlib Figure object
        """
        if X_reduced.shape[1] != 2:
            raise ValueError("This function requires 2D embeddings")

        fig, ax = plt.subplots(figsize=figsize)

        # Set style for publication quality
        plt.style.use("default")

        # Color palette
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        # Plot each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                c=[colors[i]],
                label=label,
                alpha=alpha,
                s=point_size,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add centroids
            if show_centroids:
                centroid = np.mean(X_reduced[mask], axis=0)
                ax.scatter(
                    centroid[0],
                    centroid[1],
                    c="black",
                    marker="x",
                    s=100,
                    linewidth=3,
                    label=f"{label} centroid" if i == 0 else "",
                )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio for better visualization
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_pca_analysis(
        self,
        pca_results: Dict,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create comprehensive PCA analysis visualization.

        Args:
            pca_results: Results from fit_transform_pca
            feature_names: Names of original features
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Explained variance ratio
        explained_var = pca_results["explained_variance_ratio"]
        cumulative_var = np.cumsum(explained_var)

        axes[0, 0].bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7)
        axes[0, 0].plot(
            range(1, len(explained_var) + 1), cumulative_var, "ro-", linewidth=2
        )
        axes[0, 0].set_xlabel("Principal Component")
        axes[0, 0].set_ylabel("Explained Variance Ratio")
        axes[0, 0].set_title("PCA Explained Variance")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(
            y=0.95, color="red", linestyle="--", alpha=0.7, label="95% threshold"
        )
        axes[0, 0].legend()

        # 2. Feature importance in first few components
        n_components_show = min(4, pca_results["n_components"])
        components = pca_results["components"][:n_components_show]

        im = axes[0, 1].imshow(components, aspect="auto", cmap="RdBu")
        axes[0, 1].set_title("Component Loadings (Feature Weights)")
        axes[0, 1].set_xlabel("Features")
        axes[0, 1].set_ylabel("Principal Components")
        if (
            feature_names and len(feature_names) <= 20
        ):  # Only show names if not too many
            axes[0, 1].set_xticks(range(len(feature_names)))
            axes[0, 1].set_xticklabels(feature_names, rotation=45, ha="right")
        plt.colorbar(im, ax=axes[0, 1])

        # 3. Feature importance overall
        feature_importance = pca_results["feature_importance"]
        if feature_names:
            top_indices = np.argsort(feature_importance)[-15:]  # Top 15 features
            axes[1, 0].barh(range(len(top_indices)), feature_importance[top_indices])
            axes[1, 0].set_yticks(range(len(top_indices)))
            axes[1, 0].set_yticklabels([feature_names[i] for i in top_indices])
            axes[1, 0].set_xlabel("Importance")
            axes[1, 0].set_title("Top Feature Importances")
        else:
            axes[1, 0].plot(feature_importance)
            axes[1, 0].set_xlabel("Feature Index")
            axes[1, 0].set_ylabel("Importance")
            axes[1, 0].set_title("Feature Importances")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Singular values
        singular_values = pca_results["singular_values"]
        axes[1, 1].semilogy(range(1, len(singular_values) + 1), singular_values, "bo-")
        axes[1, 1].set_xlabel("Principal Component")
        axes[1, 1].set_ylabel("Singular Value (log scale)")
        axes[1, 1].set_title("PCA Singular Values")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("Principal Component Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_comparison_plot(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        figsize: Tuple[int, int] = (18, 6),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create side-by-side comparison of different reduction methods.

        Args:
            embeddings_dict: Dictionary mapping method names to 2D embeddings
            labels: Class labels
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        n_methods = len(embeddings_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)

        if n_methods == 1:
            axes = [axes]

        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, (method_name, embedding) in enumerate(embeddings_dict.items()):
            ax = axes[i]

            for j, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors[j]],
                    label=label,
                    alpha=0.7,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )

            ax.set_title(f"{method_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", adjustable="box")

            if i == 0:  # Only show legend for first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.suptitle(
            "Dimensionality Reduction Comparison", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def compare_perplexity_values(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity_values: List[float] = [5, 10, 20, 30, 50],
    figsize: Tuple[int, int] = (20, 8),
) -> plt.Figure:
    """
    Compare t-SNE results across different perplexity values.

    Args:
        X: Feature matrix
        labels: Class labels
        perplexity_values: List of perplexity values to test
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    n_perp = len(perplexity_values)
    fig, axes = plt.subplots(1, n_perp, figsize=figsize)

    if n_perp == 1:
        axes = [axes]

    analyzer = GeometricDimensionalityAnalyzer()

    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    for i, perp in enumerate(perplexity_values):
        try:
            embedding, results = analyzer.fit_transform_tsne(X, perplexity=[perp])
            stress = results["perplexity_results"][perp]["stress"]

            ax = axes[i]

            for j, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[colors[j]],
                    label=label,
                    alpha=0.7,
                    s=30,
                    edgecolors="black",
                    linewidth=0.5,
                )

            ax.set_title(f"Perplexity = {perp}\nStress = {stress:.3f}", fontsize=12)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        except Exception as e:
            axes[i].text(
                0.5,
                0.5,
                f"Failed\nPerplexity: {perp}\nError: {str(e)[:50]}...",
                transform=axes[i].transAxes,
                ha="center",
                va="center",
            )
            axes[i].set_title(f"Perplexity = {perp} (Failed)")

    plt.suptitle("t-SNE Perplexity Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig
