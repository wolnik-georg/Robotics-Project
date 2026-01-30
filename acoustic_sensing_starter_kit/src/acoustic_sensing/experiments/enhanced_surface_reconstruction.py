"""
Enhanced Surface Reconstruction Experiment

Improved visualizations for surface reconstruction:
1. Grid-based heatmaps (solid cell coloring)
2. Overlay comparison (ground truth + predictions with transparency)
3. Confidence-weighted visualization
4. Professional presentation-ready figures

Author: Georg Wolnik
Date: January 2026
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from acoustic_sensing.experiments.surface_reconstruction import (
    SurfaceReconstructionExperiment,
)


class EnhancedSurfaceReconstructionExperiment(SurfaceReconstructionExperiment):
    """
    Enhanced surface reconstruction with improved visualizations.

    Improvements over base class:
    - Grid-based heatmaps instead of scatter plots
    - Overlay comparisons with transparency
    - Presentation-ready figure styling
    - Physical surface boundary overlay
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)
        self.logger = logging.getLogger(__name__)

        # Enhanced color scheme (colorblind-friendly)
        self.class_colors = {
            "contact": "#2166ac",  # Blue
            "edge": "#f4a582",  # Orange/Salmon
            "no_contact": "#4dac26",  # Green
        }

        # Numeric mapping for grid visualization
        self.class_to_num = {
            "contact": 0,
            "edge": 1,
            "no_contact": 2,
        }

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced surface reconstruction."""
        self.logger.info("=" * 80)
        self.logger.info("🗺️  Starting ENHANCED Surface Reconstruction")
        self.logger.info("=" * 80)

        # Run base reconstruction to get predictions
        base_results = super().run(shared_data)

        if "error" in base_results:
            return base_results

        # Generate enhanced visualizations
        self.logger.info("\n🎨 Generating enhanced visualizations...")

        coords = base_results["sweep_coordinates"]
        true_labels = base_results["sweep_labels"]
        predictions = base_results["predictions"]
        best_model = base_results["best_model"]
        best_predictions = predictions[best_model]
        classes = shared_data.get("classes", list(self.class_colors.keys()))

        # 1. Grid-based heatmap (ground truth)
        self._create_grid_heatmap(
            coords,
            true_labels,
            classes,
            "Ground Truth Surface (Grid View)",
            "enhanced_ground_truth_grid.png",
        )

        # 2. Grid-based heatmap (predictions)
        self._create_grid_heatmap(
            coords,
            best_predictions,
            classes,
            f"Predicted Surface - {best_model} (Grid View)",
            "enhanced_predicted_grid.png",
        )

        # 3. Side-by-side grid comparison
        self._create_grid_comparison(
            coords, true_labels, best_predictions, classes, best_model
        )

        # 4. Overlay comparison (key improvement!)
        self._create_overlay_comparison(
            coords, true_labels, best_predictions, classes, best_model
        )

        # 5. Error heatmap with magnitude
        self._create_enhanced_error_map(
            coords, true_labels, best_predictions, classes, best_model
        )

        # 6. Presentation-ready summary figure
        self._create_presentation_summary(
            coords,
            true_labels,
            best_predictions,
            classes,
            best_model,
            base_results.get("metrics", {}),
        )

        self.logger.info("\n✅ Enhanced Surface Reconstruction Complete!")
        return base_results

    def _create_grid_heatmap(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        classes: List[str],
        title: str,
        filename: str,
        ax: Optional[plt.Axes] = None,
        show_colorbar: bool = True,
    ):
        """
        Create a grid-based heatmap with filled cells.

        Instead of scatter points, creates a proper grid visualization
        where each measurement position is shown as a colored cell.
        """
        save_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.get_figure()

        # Convert labels to numeric
        label_nums = np.array([self.class_to_num.get(l, -1) for l in labels])

        # Determine grid resolution from data
        unique_x = np.unique(np.round(coords[:, 0], 3))
        unique_y = np.unique(np.round(coords[:, 1], 3))

        # Calculate cell size
        if len(unique_x) > 1:
            cell_width = np.min(np.diff(np.sort(unique_x))) * 0.9
        else:
            cell_width = 0.1
        if len(unique_y) > 1:
            cell_height = np.min(np.diff(np.sort(unique_y))) * 0.9
        else:
            cell_height = 0.1

        # Create colormap
        colors = [self.class_colors[c] for c in classes if c in self.class_colors]
        cmap = ListedColormap(colors)

        # Draw filled rectangles for each point
        for i, (x, y) in enumerate(coords):
            label_num = label_nums[i]
            if label_num >= 0 and label_num < len(colors):
                color = colors[label_num]
                rect = Rectangle(
                    (x - cell_width / 2, y - cell_height / 2),
                    cell_width,
                    cell_height,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.85,
                )
                ax.add_patch(rect)

        # Set axis limits with padding
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        padding = max(cell_width, cell_height)
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Create legend
        legend_patches = [
            mpatches.Patch(
                color=self.class_colors[c], label=c.replace("_", " ").title()
            )
            for c in classes
            if c in self.class_colors
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper right",
            fontsize=10,
            framealpha=0.9,
            edgecolor="gray",
        )

        ax.set_xlabel("Normalized X Position", fontsize=12)
        ax.set_ylabel("Normalized Y Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")

        # Add surface boundary box
        ax.add_patch(
            Rectangle(
                (0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2, linestyle="-"
            )
        )

        if save_fig:
            plt.tight_layout()
            self.save_plot(fig, filename)
            plt.close()

    def _create_grid_comparison(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
        model_name: str,
    ):
        """Side-by-side grid comparison: Ground Truth vs Predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Ground truth
        self._create_grid_heatmap(
            coords,
            true_labels,
            classes,
            "Ground Truth",
            None,
            ax=axes[0],
            show_colorbar=False,
        )

        # Predictions
        self._create_grid_heatmap(
            coords,
            pred_labels,
            classes,
            f"Predictions ({model_name})",
            None,
            ax=axes[1],
            show_colorbar=False,
        )

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)

        fig.suptitle(
            f"Surface Reconstruction Comparison\nAccuracy: {accuracy:.1%}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        self.save_plot(fig, "enhanced_grid_comparison.png")
        plt.close()

    def _create_overlay_comparison(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
        model_name: str,
    ):
        """
        Overlay visualization showing predictions on top of ground truth.

        - Ground truth shown as filled background
        - Predictions shown as circles with borders
        - Mismatches are immediately visible
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Determine grid resolution
        unique_x = np.unique(np.round(coords[:, 0], 3))
        unique_y = np.unique(np.round(coords[:, 1], 3))

        if len(unique_x) > 1:
            cell_width = np.min(np.diff(np.sort(unique_x))) * 0.9
        else:
            cell_width = 0.1
        if len(unique_y) > 1:
            cell_height = np.min(np.diff(np.sort(unique_y))) * 0.9
        else:
            cell_height = 0.1

        # Draw ground truth as background rectangles
        for i, (x, y) in enumerate(coords):
            true_label = true_labels[i]
            if true_label in self.class_colors:
                rect = Rectangle(
                    (x - cell_width / 2, y - cell_height / 2),
                    cell_width,
                    cell_height,
                    facecolor=self.class_colors[true_label],
                    edgecolor="none",
                    alpha=0.4,  # Semi-transparent background
                )
                ax.add_patch(rect)

        # Draw predictions as circles on top
        for i, (x, y) in enumerate(coords):
            pred_label = pred_labels[i]
            true_label = true_labels[i]

            is_correct = pred_label == true_label

            if pred_label in self.class_colors:
                # Circle for prediction
                circle = plt.Circle(
                    (x, y),
                    radius=min(cell_width, cell_height) * 0.35,
                    facecolor=self.class_colors[pred_label],
                    edgecolor="black" if is_correct else "red",
                    linewidth=1 if is_correct else 2.5,
                    alpha=0.9,
                )
                ax.add_patch(circle)

        # Set axis limits
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        padding = max(cell_width, cell_height) * 1.5
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Legend
        legend_elements = [
            mpatches.Patch(
                color=self.class_colors[c],
                alpha=0.4,
                label=f"{c.replace('_', ' ').title()} (Ground Truth)",
            )
            for c in classes
            if c in self.class_colors
        ]
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
                label="Correct Prediction",
            )
        )
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="red",
                markeredgewidth=2.5,
                label="Wrong Prediction",
            )
        )
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.95
        )

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        n_errors = np.sum(true_labels != pred_labels)

        ax.set_xlabel("Normalized X Position", fontsize=12)
        ax.set_ylabel("Normalized Y Position", fontsize=12)
        ax.set_title(
            f"Overlay Comparison: Ground Truth (Background) + Predictions (Circles)\n"
            f"Model: {model_name} | Accuracy: {accuracy:.1%} | Errors: {n_errors}",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")

        # Surface boundary
        ax.add_patch(
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2)
        )

        plt.tight_layout()
        self.save_plot(fig, "enhanced_overlay_comparison.png")
        plt.close()

    def _create_enhanced_error_map(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
        model_name: str,
    ):
        """
        Enhanced error visualization with confusion details.

        Shows what the model predicted vs what it should have predicted.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Separate correct and incorrect
        correct_mask = true_labels == pred_labels
        incorrect_mask = ~correct_mask

        # Determine cell size
        unique_x = np.unique(np.round(coords[:, 0], 3))
        unique_y = np.unique(np.round(coords[:, 1], 3))
        if len(unique_x) > 1:
            cell_size = np.min(np.diff(np.sort(unique_x))) * 0.4
        else:
            cell_size = 0.05

        # Plot correct predictions (green, small)
        ax.scatter(
            coords[correct_mask, 0],
            coords[correct_mask, 1],
            c="#2ca02c",  # Green
            s=80,
            alpha=0.5,
            marker="o",
            edgecolors="darkgreen",
            linewidths=0.5,
            label=f"Correct ({np.sum(correct_mask)})",
        )

        # Plot incorrect predictions (red, with error details)
        for i in np.where(incorrect_mask)[0]:
            x, y = coords[i]
            true_label = true_labels[i]
            pred_label = pred_labels[i]

            # Draw X marker
            ax.scatter(
                x,
                y,
                c="#d62728",  # Red
                s=200,
                alpha=0.9,
                marker="X",
                edgecolors="darkred",
                linewidths=1.5,
                zorder=10,
            )

            # Add text annotation for error type
            ax.annotate(
                f"{pred_label[0].upper()}→{true_label[0].upper()}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.7,
                color="darkred",
            )

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        n_errors = np.sum(incorrect_mask)

        # Legend
        ax.scatter(
            [],
            [],
            c="#d62728",
            s=200,
            marker="X",
            label=f"Errors ({n_errors})",
            edgecolors="darkred",
        )
        ax.legend(loc="upper right", fontsize=11, framealpha=0.95)

        # Axis settings
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        padding = 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        ax.set_xlabel("Normalized X Position", fontsize=12)
        ax.set_ylabel("Normalized Y Position", fontsize=12)
        ax.set_title(
            f"Error Map: {model_name}\n"
            f"Accuracy: {accuracy:.1%} | Error Rate: {1-accuracy:.1%}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Surface boundary
        ax.add_patch(
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2)
        )

        # Add error type legend
        error_text = "Error Labels: C=Contact, E=Edge, N=No_Contact\n"
        error_text += "Format: Predicted→Actual"
        ax.text(
            0.02,
            0.02,
            error_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        self.save_plot(fig, "enhanced_error_map.png")
        plt.close()

    def _create_presentation_summary(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
        model_name: str,
        metrics: Dict[str, Any],
    ):
        """
        Create a single presentation-ready summary figure.

        4-panel layout:
        - Ground Truth | Predictions
        - Overlay      | Confusion Matrix
        """
        fig = plt.figure(figsize=(16, 14))

        # Create 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # Panel 1: Ground Truth Grid
        self._create_grid_heatmap(
            coords,
            true_labels,
            classes,
            "A) Ground Truth",
            None,
            ax=ax1,
            show_colorbar=False,
        )

        # Panel 2: Predictions Grid
        self._create_grid_heatmap(
            coords,
            pred_labels,
            classes,
            "B) Model Predictions",
            None,
            ax=ax2,
            show_colorbar=False,
        )

        # Panel 3: Mini Overlay
        self._create_mini_overlay(ax3, coords, true_labels, pred_labels, classes)
        ax3.set_title("C) Overlay (Errors in Red)", fontsize=12, fontweight="bold")

        # Panel 4: Confusion Matrix
        self._create_confusion_matrix_panel(ax4, true_labels, pred_labels, classes)
        ax4.set_title("D) Confusion Matrix", fontsize=12, fontweight="bold")

        # Overall title
        accuracy = accuracy_score(true_labels, pred_labels)
        fig.suptitle(
            f"Surface Reconstruction Summary\n"
            f"Model: {model_name} | Accuracy: {accuracy:.1%} | N={len(true_labels)} points",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self.save_plot(fig, "presentation_summary.png")
        plt.close()

        self.logger.info(f"📊 Created presentation summary figure")

    def _create_mini_overlay(
        self,
        ax: plt.Axes,
        coords: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
    ):
        """Compact overlay for summary figure."""
        # Determine cell size
        unique_x = np.unique(np.round(coords[:, 0], 3))
        if len(unique_x) > 1:
            cell_size = np.min(np.diff(np.sort(unique_x))) * 0.8
        else:
            cell_size = 0.1

        # Draw all points
        for i, (x, y) in enumerate(coords):
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            is_correct = true_label == pred_label

            color = self.class_colors.get(pred_label, "gray")
            edge_color = "black" if is_correct else "red"
            edge_width = 0.5 if is_correct else 2

            circle = plt.Circle(
                (x, y),
                radius=cell_size * 0.4,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width,
                alpha=0.8,
            )
            ax.add_patch(circle)

        # Axis settings
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        padding = cell_size
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

    def _create_confusion_matrix_panel(
        self,
        ax: plt.Axes,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
    ):
        """Create confusion matrix visualization."""
        cm = confusion_matrix(true_labels, pred_labels, labels=classes)

        # Normalize
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        im = ax.imshow(
            cm_normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1
        )

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom", fontsize=10)

        # Labels
        tick_marks = np.arange(len(classes))
        class_labels = [c.replace("_", "\n").title() for c in classes]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_labels, fontsize=9)
        ax.set_yticklabels(class_labels, fontsize=9)

        # Add text annotations
        thresh = 0.5
        for i in range(len(classes)):
            for j in range(len(classes)):
                color = "white" if cm_normalized[i, j] > thresh else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm_normalized[i, j]:.0%})",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)


# Register with orchestrator
def register_enhanced_reconstruction():
    """Helper to register this experiment with the orchestrator."""
    return {"enhanced_surface_reconstruction": EnhancedSurfaceReconstructionExperiment}
