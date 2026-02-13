#!/usr/bin/env python3
"""
Generate Position Generalization (Workspace Rotation) Comparison Figures

Creates comparative visualizations across all 3 workspace rotations to analyze
position generalization performance and identify workspace-specific patterns.

USAGE:
------
    python generate_3class_rotation_figures.py

WHAT IT DOES:
-------------
    Analyzes results from all 3 rotations:
        - Rotation 1: Train WS1+WS3 → Validate WS2 (55.7% validation)
        - Rotation 2: Train WS2+WS3 → Validate WS1 (24.4% validation)
        - Rotation 3: Train WS1+WS2 → Validate WS3 (23.3% validation)

    Generates comparison figures:
        1. Performance across rotations (CV vs validation)
        2. 3×3 confusion matrix grid (all rotations)
        3. Class-wise accuracy breakdown
        4. Normalized performance vs random baseline

OUTPUTS:
--------
    Various comparison figures showing:
        - Catastrophic workspace-dependent failure
        - Average 34.5% validation (barely above 33.3% random)
        - Two rotations worse than random (0.70× and 0.73× normalized)

DEMONSTRATES:
-------------
    - Position generalization failure
    - Workspace-specific acoustic signatures
    - Need for workspace-specific training

See README.md Section "Performance Summary → Position Generalization" for details.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
from typing import Dict, Any, List, Tuple
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Color scheme
COLORS = {
    "ws1": "#e74c3c",  # Red
    "ws2": "#2ecc71",  # Green
    "ws3": "#3498db",  # Blue
    "cv": "#9b59b6",  # Purple
    "validation": "#e67e22",  # Orange
    "random": "#95a5a6",  # Gray
    "contact": "#27ae60",  # Dark green
    "no_contact": "#c0392b",  # Dark red
    "edge": "#f39c12",  # Bright orange
}

# Experiment directories
EXPERIMENTS = {
    "rotation1": {
        "dir": "test_pipeline_3class_v1",
        "label": "Rotation 1: WS1+3→WS2",
        "val_ws": "WS2",
        "train_ws": "WS1+3",
    },
    "rotation2": {
        "dir": "test_pipeline_3class_rotation2_ws2ws3_train_ws1_val",
        "label": "Rotation 2: WS2+3→WS1",
        "val_ws": "WS1",
        "train_ws": "WS2+3",
    },
    "rotation3": {
        "dir": "test_pipeline_3class_rotation3_ws1ws2_train_ws3_val",
        "label": "Rotation 3: WS1+2→WS3",
        "val_ws": "WS3",
        "train_ws": "WS1+2",
    },
}

# Object generalization experiment
OBJECT_GEN_EXPERIMENT = {
    "dir": "object_generalization_ws4_holdout_3class",
    "label": "Object Gen: WS1+2+3→WS4",
    "val_ws": "WS4 (Object D)",
    "train_ws": "WS1+2+3 (Objects A,B,C)",
}

OUTPUT_DIR = Path("ml_analysis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_discrimination_summary(experiment_dir: str) -> Dict[str, Any]:
    """Load discrimination summary from experiment directory."""
    summary_path = (
        Path(experiment_dir)
        / "discriminationanalysis"
        / "validation_results"
        / "discrimination_summary.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(f"No discrimination summary found at {summary_path}")

    with open(summary_path) as f:
        return json.load(f)


def generate_rotation_comparison_figure(all_results: Dict[str, Dict]) -> None:
    """
    Figure 1: Workspace Rotation Performance Comparison
    Shows CV and validation accuracy for all 3 rotations with random baseline.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    rotations = ["rotation1", "rotation2", "rotation3"]
    rotation_labels = [
        "Rotation 1\n(WS1+3→WS2)",
        "Rotation 2\n(WS2+3→WS1)",
        "Rotation 3\n(WS1+2→WS3)",
    ]

    x = np.arange(len(rotations))
    width = 0.35

    cv_accs = []
    val_accs = []

    for rot in rotations:
        results = all_results[rot]
        best_clf = results["best_classifier"]
        cv_accs.append(best_clf["cv_test_accuracy"] * 100)
        val_accs.append(best_clf["validation_accuracy"] * 100)

    # Plot bars
    bars1 = ax.bar(
        x - width / 2,
        cv_accs,
        width,
        label="Cross-Validation",
        color=COLORS["cv"],
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        val_accs,
        width,
        label="Validation",
        color=COLORS["validation"],
        alpha=0.8,
    )

    # Add random baseline
    ax.axhline(
        y=33.3,
        color=COLORS["random"],
        linestyle="--",
        linewidth=2,
        label="Random Baseline (33.3%)",
        zorder=0,
    )

    # Formatting
    ax.set_xlabel("Workspace Rotation", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        "3-Class Position Generalization: Workspace Rotation Performance",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rotation_labels)
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_ylim([0, 100])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Add average line for validation
    avg_val = np.mean(val_accs)
    ax.axhline(
        y=avg_val,
        color=COLORS["validation"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.5,
        label=f"Avg Validation ({avg_val:.1f}%)",
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure1_3class_rotation_comparison.png")
    plt.savefig(OUTPUT_DIR / "figure1_3class_rotation_comparison.pdf")
    print(f"✓ Saved: figure1_3class_rotation_comparison.png/pdf")
    plt.close()


def generate_confusion_matrix_grid(all_results: Dict[str, Dict]) -> None:
    """
    Figure 2: 3x3 Confusion Matrix Grid
    Shows validation confusion matrices for all 3 rotations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    class_names = ["Contact", "No-Contact", "Edge"]
    rotations = ["rotation1", "rotation2", "rotation3"]

    for idx, rot in enumerate(rotations):
        ax = axes[idx]
        results = all_results[rot]

        # Get confusion matrix
        cm = np.array(results["best_classifier"]["validation_confusion_matrix"])

        # Normalize to percentages
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        # Plot
        im = ax.imshow(cm_normalized, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        # Set ticks
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(
                    j,
                    i,
                    f"{cm_normalized[i, j]:.1f}%\n({cm[i, j]})",
                    ha="center",
                    va="center",
                    color="black" if cm_normalized[i, j] > 50 else "white",
                    fontsize=9,
                    fontweight="bold",
                )

        # Labels
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")

        # Title with validation accuracy
        val_acc = results["best_classifier"]["validation_accuracy"] * 100
        ax.set_title(
            f'{EXPERIMENTS[rot]["label"]}\nVal Acc: {val_acc:.1f}%',
            fontsize=13,
            fontweight="bold",
        )

    # Add colorbar
    fig.colorbar(
        im,
        ax=axes,
        orientation="horizontal",
        pad=0.15,
        fraction=0.046,
        label="Classification Accuracy (%)",
    )

    plt.suptitle(
        "3-Class Confusion Matrices: Workspace Rotation Validation Results",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure2_3class_confusion_matrix_grid.png")
    plt.savefig(OUTPUT_DIR / "figure2_3class_confusion_matrix_grid.pdf")
    print(f"✓ Saved: figure2_3class_confusion_matrix_grid.png/pdf")
    plt.close()


def generate_cv_vs_validation_comparison(all_results: Dict[str, Dict]) -> None:
    """
    Figure 3: CV vs Validation Accuracy Gap Analysis
    Shows the generalization gap for each rotation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    rotations = ["rotation1", "rotation2", "rotation3"]
    rotation_labels = ["Rotation 1", "Rotation 2", "Rotation 3"]

    cv_accs = []
    val_accs = []
    gaps = []

    for rot in rotations:
        results = all_results[rot]
        best_clf = results["best_classifier"]
        cv_acc = best_clf["cv_test_accuracy"] * 100
        val_acc = best_clf["validation_accuracy"] * 100

        cv_accs.append(cv_acc)
        val_accs.append(val_acc)
        gaps.append(cv_acc - val_acc)

    x = np.arange(len(rotations))

    # Plot lines
    ax.plot(
        x,
        cv_accs,
        "o-",
        label="Cross-Validation",
        color=COLORS["cv"],
        linewidth=2.5,
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="white",
    )
    ax.plot(
        x,
        val_accs,
        "s-",
        label="Validation",
        color=COLORS["validation"],
        linewidth=2.5,
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="white",
    )

    # Shade the gap
    for i in range(len(rotations)):
        ax.fill_between(
            [x[i] - 0.15, x[i] + 0.15],
            [val_accs[i], val_accs[i]],
            [cv_accs[i], cv_accs[i]],
            alpha=0.3,
            color="red",
        )
        # Add gap annotation
        ax.text(
            x[i] + 0.25,
            (cv_accs[i] + val_accs[i]) / 2,
            f"Gap:\n{gaps[i]:.1f}%",
            fontsize=9,
            ha="left",
            va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    # Add random baseline
    ax.axhline(
        y=33.3,
        color=COLORS["random"],
        linestyle="--",
        linewidth=2,
        label="Random Baseline",
        zorder=0,
    )

    # Formatting
    ax.set_xlabel("Workspace Rotation", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Generalization Gap Analysis: CV vs Validation Performance",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rotation_labels)
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_ylim([25, 90])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure3_cv_vs_validation_gap.png")
    plt.savefig(OUTPUT_DIR / "figure3_cv_vs_validation_gap.pdf")
    print(f"✓ Saved: figure3_cv_vs_validation_gap.png/pdf")
    plt.close()


def generate_classwise_performance(all_results: Dict[str, Dict]) -> None:
    """
    Figure 4: Class-wise Performance Breakdown
    Shows per-class precision/recall/F1 for each rotation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    class_names = ["Contact", "No-Contact", "Edge"]
    rotations = ["rotation1", "rotation2", "rotation3"]

    for idx, rot in enumerate(rotations):
        ax = axes[idx]
        results = all_results[rot]

        # Get per-class metrics
        clf_report = results["best_classifier"]["validation_classification_report"]

        precisions = []
        recalls = []
        f1s = []

        for class_name in class_names:
            class_key = class_name.lower().replace("-", "_")
            if class_key in clf_report:
                precisions.append(clf_report[class_key]["precision"] * 100)
                recalls.append(clf_report[class_key]["recall"] * 100)
                f1s.append(clf_report[class_key]["f1-score"] * 100)
            else:
                precisions.append(0)
                recalls.append(0)
                f1s.append(0)

        x = np.arange(len(class_names))
        width = 0.25

        bars1 = ax.bar(
            x - width, precisions, width, label="Precision", color="#3498db", alpha=0.8
        )
        bars2 = ax.bar(x, recalls, width, label="Recall", color="#e74c3c", alpha=0.8)
        bars3 = ax.bar(
            x + width, f1s, width, label="F1-Score", color="#2ecc71", alpha=0.8
        )

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 1,
                        f"{height:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xlabel("Class", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title(f'{EXPERIMENTS[rot]["label"]}', fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(loc="lower right")
        ax.set_ylim([0, 110])
        ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)

    plt.suptitle(
        "Class-wise Performance Analysis: Precision, Recall, F1-Score",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure4_classwise_performance.png")
    plt.savefig(OUTPUT_DIR / "figure4_classwise_performance.pdf")
    print(f"✓ Saved: figure4_classwise_performance.png/pdf")
    plt.close()


def generate_summary_table_figure(all_results: Dict[str, Dict]) -> None:
    """
    Figure 5: Summary Table as Image
    Creates a publication-ready table summarizing all results.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("tight")
    ax.axis("off")

    # Prepare data
    table_data = []
    table_data.append(
        [
            "Rotation",
            "Training",
            "Validation",
            "CV Acc",
            "Val Acc",
            "Gap",
            "vs Random",
            "Best F1",
        ]
    )

    rotations = ["rotation1", "rotation2", "rotation3"]

    for rot in rotations:
        results = all_results[rot]
        best_clf = results["best_classifier"]

        cv_acc = best_clf["cv_test_accuracy"] * 100
        val_acc = best_clf["validation_accuracy"] * 100
        gap = cv_acc - val_acc
        vs_random = val_acc / 33.3

        # Get weighted F1
        clf_report = best_clf["validation_classification_report"]
        f1 = clf_report.get("weighted avg", {}).get("f1-score", 0) * 100

        row = [
            EXPERIMENTS[rot]["label"].split(":")[0],  # Rotation name
            EXPERIMENTS[rot]["train_ws"],
            EXPERIMENTS[rot]["val_ws"],
            f"{cv_acc:.1f}%",
            f"{val_acc:.1f}%",
            f"{gap:.1f}%",
            f"{vs_random:.2f}×",
            f"{f1:.1f}%",
        ]
        table_data.append(row)

    # Add average row
    avg_cv = np.mean(
        [
            all_results[rot]["best_classifier"]["cv_test_accuracy"] * 100
            for rot in rotations
        ]
    )
    avg_val = np.mean(
        [
            all_results[rot]["best_classifier"]["validation_accuracy"] * 100
            for rot in rotations
        ]
    )
    avg_gap = avg_cv - avg_val
    avg_vs_random = avg_val / 33.3

    avg_f1s = []
    for rot in rotations:
        clf_report = all_results[rot]["best_classifier"][
            "validation_classification_report"
        ]
        avg_f1s.append(clf_report.get("weighted avg", {}).get("f1-score", 0) * 100)
    avg_f1 = np.mean(avg_f1s)

    table_data.append(
        [
            "AVERAGE",
            "---",
            "---",
            f"{avg_cv:.1f}%",
            f"{avg_val:.1f}%",
            f"{avg_gap:.1f}%",
            f"{avg_vs_random:.2f}×",
            f"{avg_f1:.1f}%",
        ]
    )

    # Create table
    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.12, 0.12, 0.10, 0.10, 0.10, 0.12, 0.10],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor("#3498db")
        cell.set_text_props(weight="bold", color="white")

    # Style average row
    for i in range(len(table_data[0])):
        cell = table[(len(table_data) - 1, i)]
        cell.set_facecolor("#f39c12")
        cell.set_text_props(weight="bold")

    # Color code validation accuracy cells
    for row_idx in range(1, len(table_data) - 1):
        val_acc_cell = table[(row_idx, 4)]
        val_acc = float(table_data[row_idx][4].strip("%"))

        if val_acc > 70:
            val_acc_cell.set_facecolor("#2ecc71")  # Green for good
        elif val_acc > 50:
            val_acc_cell.set_facecolor("#f39c12")  # Orange for moderate
        else:
            val_acc_cell.set_facecolor("#e74c3c")  # Red for poor
        val_acc_cell.set_text_props(weight="bold", color="white")

    plt.title(
        "3-Class Workspace Rotation: Complete Performance Summary",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.savefig(OUTPUT_DIR / "figure5_summary_table.png")
    plt.savefig(OUTPUT_DIR / "figure5_summary_table.pdf")
    print(f"✓ Saved: figure5_summary_table.png/pdf")
    plt.close()


def main():
    """Generate all 3-class rotation figures."""
    print("=" * 80)
    print("3-CLASS WORKSPACE ROTATION FIGURE GENERATION")
    print("=" * 80)
    print()

    # Load all results
    print("Loading discrimination results...")
    all_results = {}

    for rot_key, rot_info in EXPERIMENTS.items():
        exp_dir = rot_info["dir"]
        try:
            results = load_discrimination_summary(exp_dir)
            all_results[rot_key] = results
            print(f"✓ Loaded {rot_key}: {rot_info['label']}")
        except FileNotFoundError as e:
            print(f"✗ Error loading {rot_key}: {e}")
            return

    print()
    print("Generating figures...")
    print("-" * 80)

    # Generate all figures
    generate_rotation_comparison_figure(all_results)
    generate_confusion_matrix_grid(all_results)
    generate_cv_vs_validation_comparison(all_results)
    generate_classwise_performance(all_results)
    generate_summary_table_figure(all_results)

    print()
    print("=" * 80)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    print()
    print("Generated figures:")
    print(
        "  1. figure1_3class_rotation_comparison.png/pdf - Main performance comparison"
    )
    print("  2. figure2_3class_confusion_matrix_grid.png/pdf - All confusion matrices")
    print("  3. figure3_cv_vs_validation_gap.png/pdf - Generalization gap analysis")
    print("  4. figure4_classwise_performance.png/pdf - Per-class metrics")
    print("  5. figure5_summary_table.png/pdf - Complete summary table")
    print()


if __name__ == "__main__":
    main()
