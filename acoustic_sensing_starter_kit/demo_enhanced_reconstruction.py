#!/usr/bin/env python3
"""
Quick Demo: Enhanced Surface Reconstruction Visualizations

This script demonstrates the enhanced visualization capabilities
without requiring the full ML pipeline. It creates sample visualizations
using mock data to show what the reconstruction output looks like.

Usage:
    python demo_enhanced_reconstruction.py

Output:
    Creates visualization files in ./demo_reconstruction_output/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import ListedColormap
from pathlib import Path
import os

# Output directory
OUTPUT_DIR = Path("demo_reconstruction_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Color scheme (colorblind-friendly)
CLASS_COLORS = {
    "contact": "#2166ac",  # Blue
    "edge": "#f4a582",  # Orange/Salmon
    "no_contact": "#4dac26",  # Green
}

CLASS_TO_NUM = {"contact": 0, "edge": 1, "no_contact": 2}


def generate_sample_surface_data(grid_size: int = 10) -> tuple:
    """
    Generate realistic sample data mimicking the squares_cutout surface.

    Returns:
        coords: Nx2 array of (x, y) coordinates
        true_labels: Ground truth labels
        pred_labels: Simulated predictions (with some errors)
    """
    # Create grid
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Create ground truth pattern (squares_cutout surface)
    true_labels = []
    for xi, yi in coords:
        # Define 4 "holes" (no_contact regions)
        hole_positions = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]

        is_hole = False
        is_edge = False

        for hx, hy in hole_positions:
            dist = np.sqrt((xi - hx) ** 2 + (yi - hy) ** 2)
            if dist < 0.12:
                is_hole = True
            elif dist < 0.18:
                is_edge = True

        # Surface boundary is also edge
        if xi < 0.05 or xi > 0.95 or yi < 0.05 or yi > 0.95:
            is_edge = True
            is_hole = False

        if is_hole:
            true_labels.append("no_contact")
        elif is_edge:
            true_labels.append("edge")
        else:
            true_labels.append("contact")

    true_labels = np.array(true_labels)

    # Simulate predictions (75% accuracy, typical for position generalization)
    pred_labels = true_labels.copy()
    n_errors = int(len(pred_labels) * 0.25)  # 25% error rate
    error_indices = np.random.choice(len(pred_labels), n_errors, replace=False)

    classes = list(CLASS_COLORS.keys())
    for idx in error_indices:
        # Replace with random wrong class
        current = pred_labels[idx]
        wrong_classes = [c for c in classes if c != current]
        pred_labels[idx] = np.random.choice(wrong_classes)

    return coords, true_labels, pred_labels


def create_grid_heatmap(coords, labels, title, filename, ax=None):
    """Create grid-based heatmap visualization."""
    save_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    # Determine cell size
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

    # Draw filled rectangles
    for i, (x, y) in enumerate(coords):
        label = labels[i]
        if label in CLASS_COLORS:
            rect = Rectangle(
                (x - cell_width / 2, y - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=CLASS_COLORS[label],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
            ax.add_patch(rect)

    # Axis limits
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # Legend
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS[c], label=c.replace("_", " ").title())
        for c in CLASS_COLORS.keys()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.9)

    ax.set_xlabel("Normalized X Position", fontsize=12)
    ax.set_ylabel("Normalized Y Position", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")

    # Surface boundary
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2))

    if save_fig:
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved {filename}")


def create_overlay_comparison(coords, true_labels, pred_labels, model_name, filename):
    """Create overlay visualization with predictions on ground truth."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Cell size
    unique_x = np.unique(np.round(coords[:, 0], 3))
    if len(unique_x) > 1:
        cell_width = np.min(np.diff(np.sort(unique_x))) * 0.9
    else:
        cell_width = 0.1
    cell_height = cell_width

    # Draw ground truth as background
    for i, (x, y) in enumerate(coords):
        true_label = true_labels[i]
        if true_label in CLASS_COLORS:
            rect = Rectangle(
                (x - cell_width / 2, y - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=CLASS_COLORS[true_label],
                edgecolor="none",
                alpha=0.35,
            )
            ax.add_patch(rect)

    # Draw predictions as circles
    for i, (x, y) in enumerate(coords):
        pred_label = pred_labels[i]
        true_label = true_labels[i]
        is_correct = pred_label == true_label

        if pred_label in CLASS_COLORS:
            circle = Circle(
                (x, y),
                radius=cell_width * 0.35,
                facecolor=CLASS_COLORS[pred_label],
                edgecolor="black" if is_correct else "red",
                linewidth=1 if is_correct else 2.5,
                alpha=0.9,
            )
            ax.add_patch(circle)

    # Axis limits
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # Legend
    legend_elements = [
        mpatches.Patch(
            color=CLASS_COLORS[c],
            alpha=0.35,
            label=f"{c.replace('_', ' ').title()} (GT)",
        )
        for c in CLASS_COLORS.keys()
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
            label="Correct Pred",
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
            label="Error",
        )
    )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.95)

    # Metrics
    accuracy = np.mean(true_labels == pred_labels)
    n_errors = np.sum(true_labels != pred_labels)

    ax.set_xlabel("Normalized X Position", fontsize=12)
    ax.set_ylabel("Normalized Y Position", fontsize=12)
    ax.set_title(
        f"Overlay: Ground Truth (Background) + Predictions (Circles)\n"
        f"Model: {model_name} | Accuracy: {accuracy:.1%} | Errors: {n_errors}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def create_error_map(coords, true_labels, pred_labels, model_name, filename):
    """Create error visualization."""
    fig, ax = plt.subplots(figsize=(12, 10))

    correct_mask = true_labels == pred_labels
    incorrect_mask = ~correct_mask

    # Correct predictions (green)
    ax.scatter(
        coords[correct_mask, 0],
        coords[correct_mask, 1],
        c="#2ca02c",
        s=80,
        alpha=0.5,
        marker="o",
        edgecolors="darkgreen",
        linewidths=0.5,
        label=f"Correct ({np.sum(correct_mask)})",
    )

    # Errors (red X)
    for i in np.where(incorrect_mask)[0]:
        x, y = coords[i]
        ax.scatter(
            x,
            y,
            c="#d62728",
            s=200,
            alpha=0.9,
            marker="X",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=10,
        )

        # Error annotation
        ax.annotate(
            f"{pred_labels[i][0].upper()}→{true_labels[i][0].upper()}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.7,
            color="darkred",
        )

    ax.scatter(
        [],
        [],
        c="#d62728",
        s=200,
        marker="X",
        label=f"Errors ({np.sum(incorrect_mask)})",
        edgecolors="darkred",
    )

    # Axis
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    accuracy = np.mean(correct_mask)
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
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2))

    ax.text(
        0.02,
        0.02,
        "Error Labels: C=Contact, E=Edge, N=No_Contact\nFormat: Predicted→Actual",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def create_presentation_summary(coords, true_labels, pred_labels, model_name, filename):
    """Create 4-panel presentation summary."""
    from sklearn.metrics import confusion_matrix

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel 1: Ground Truth
    create_grid_heatmap(coords, true_labels, "A) Ground Truth", None, ax=ax1)

    # Panel 2: Predictions
    create_grid_heatmap(coords, pred_labels, "B) Model Predictions", None, ax=ax2)

    # Panel 3: Mini overlay
    unique_x = np.unique(np.round(coords[:, 0], 3))
    cell_size = np.min(np.diff(np.sort(unique_x))) * 0.8 if len(unique_x) > 1 else 0.1

    for i, (x, y) in enumerate(coords):
        is_correct = true_labels[i] == pred_labels[i]
        color = CLASS_COLORS.get(pred_labels[i], "gray")
        circle = Circle(
            (x, y),
            radius=cell_size * 0.4,
            facecolor=color,
            edgecolor="black" if is_correct else "red",
            linewidth=0.5 if is_correct else 2,
            alpha=0.8,
        )
        ax3.add_patch(circle)

    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.2)
    ax3.set_xlabel("X", fontsize=10)
    ax3.set_ylabel("Y", fontsize=10)
    ax3.set_title("C) Overlay (Errors in Red)", fontsize=12, fontweight="bold")

    # Panel 4: Confusion Matrix
    classes = list(CLASS_COLORS.keys())
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax4.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom", fontsize=10)

    ax4.set_xticks(range(len(classes)))
    ax4.set_yticks(range(len(classes)))
    ax4.set_xticklabels([c.replace("_", "\n").title() for c in classes], fontsize=9)
    ax4.set_yticklabels([c.replace("_", "\n").title() for c in classes], fontsize=9)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax4.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax4.set_xlabel("Predicted", fontsize=11)
    ax4.set_ylabel("Actual", fontsize=11)
    ax4.set_title("D) Confusion Matrix", fontsize=12, fontweight="bold")

    # Overall title
    accuracy = np.mean(true_labels == pred_labels)
    fig.suptitle(
        f"Surface Reconstruction Summary\n"
        f"Model: {model_name} | Accuracy: {accuracy:.1%} | N={len(true_labels)} points",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def create_side_by_side_comparison(
    coords, true_labels, pred_labels, model_name, filename
):
    """Create side-by-side grid comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    create_grid_heatmap(coords, true_labels, "Ground Truth", None, ax=axes[0])
    create_grid_heatmap(
        coords, pred_labels, f"Predictions ({model_name})", None, ax=axes[1]
    )

    accuracy = np.mean(true_labels == pred_labels)
    fig.suptitle(
        f"Surface Reconstruction Comparison\nAccuracy: {accuracy:.1%}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {filename}")


def main():
    """Generate all demo visualizations."""
    print("=" * 60)
    print("🗺️  Enhanced Surface Reconstruction Demo")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}\n")

    # Generate sample data
    print("📊 Generating sample surface data...")
    np.random.seed(42)  # For reproducibility
    coords, true_labels, pred_labels = generate_sample_surface_data(grid_size=10)
    model_name = "Random Forest"

    accuracy = np.mean(true_labels == pred_labels)
    print(f"   Grid: 10x10 = {len(coords)} points")
    print(f"   Simulated accuracy: {accuracy:.1%}\n")

    # Generate visualizations
    print("🎨 Creating visualizations...")

    # 1. Grid heatmaps
    create_grid_heatmap(
        coords,
        true_labels,
        "Ground Truth Surface (Grid View)",
        "1_ground_truth_grid.png",
    )
    create_grid_heatmap(
        coords,
        pred_labels,
        f"Predicted Surface - {model_name} (Grid View)",
        "2_predicted_grid.png",
    )

    # 2. Side-by-side comparison
    create_side_by_side_comparison(
        coords, true_labels, pred_labels, model_name, "3_side_by_side_comparison.png"
    )

    # 3. Overlay comparison (key visualization!)
    create_overlay_comparison(
        coords, true_labels, pred_labels, model_name, "4_overlay_comparison.png"
    )

    # 4. Error map
    create_error_map(coords, true_labels, pred_labels, model_name, "5_error_map.png")

    # 5. Presentation summary (4-panel)
    create_presentation_summary(
        coords, true_labels, pred_labels, model_name, "6_presentation_summary.png"
    )

    print(f"\n✅ Demo complete! Check {OUTPUT_DIR}/ for outputs.")
    print("\nVisualization files created:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   📄 {f.name}")


if __name__ == "__main__":
    main()
