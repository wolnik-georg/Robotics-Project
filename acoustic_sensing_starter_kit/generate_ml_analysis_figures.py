#!/usr/bin/env python3
"""
Generate ML Analysis Figures for Final Report

Creates the supporting figures used in the paper to explain the experimental
setup and feature architecture:
    - Figure 6: Experimental Setup (workspace rotation strategy)
    - Figure 11: Feature Architecture (80D hand-crafted features breakdown)

USAGE:
------
    python generate_ml_analysis_figures.py

WHAT IT DOES:
-------------
    1. Creates workspace rotation experimental strategy visualization
       - Shows 3 workspace rotations with train/validation splits
       - Illustrates position generalization approach

    2. Creates hand-crafted feature architecture diagram
       - 80-dimensional feature vector breakdown:
         • 11 spectral features (centroid, bandwidth, rolloff, etc.)
         • 39 MFCCs (13 + Δ + ΔΔ)
         • 15 temporal features (ZCR, RMS, etc.)
         • 15 impulse response features

OUTPUTS:
--------
    ml_analysis_figures/
        figure6_experimental_setup.png      # Workspace rotation strategy
        figure11_feature_dimensions.png     # Feature architecture breakdown

USED IN:
--------
    Final report (docs/final_report.tex):
        - Figure 6: Illustrates position generalization experimental design
        - Figure 11: Explains hand-crafted feature extraction pipeline

See README.md Section "ML Analysis Figures" for details.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
from typing import Dict, Any, Tuple
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
    "train": "#2ecc71",  # Green
    "test": "#3498db",  # Blue
    "validation": "#e74c3c",  # Red
    "holdout": "#9b59b6",  # Purple
    "success": "#27ae60",  # Dark green
    "failure": "#c0392b",  # Dark red
    "neutral": "#7f8c8d",  # Gray
    "random_chance": "#e67e22",  # Orange
}


def load_results(base_dir: str) -> Dict[str, Any]:
    """Load all results from an experiment directory."""
    base_path = Path(base_dir)
    results = {}

    # Load discrimination summary
    disc_summary_path = (
        base_path
        / "discriminationanalysis"
        / "validation_results"
        / "discrimination_summary.json"
    )
    if disc_summary_path.exists():
        with open(disc_summary_path) as f:
            results["discrimination"] = json.load(f)

    # Load dimensionality reduction summary
    dim_summary_path = (
        base_path / "dimensionalityreduction" / "dimensionality_reduction_summary.json"
    )
    if dim_summary_path.exists():
        with open(dim_summary_path) as f:
            results["dimensionality"] = json.load(f)

    # Load execution summary
    exec_summary_path = base_path / "execution_summary.json"
    if exec_summary_path.exists():
        with open(exec_summary_path) as f:
            results["execution"] = json.load(f)

    # Load full results pickle if exists
    full_results_path = base_path / "full_results.pkl"
    if full_results_path.exists():
        with open(full_results_path, "rb") as f:
            results["full"] = pickle.load(f)

    return results


def create_main_comparison_figure(v4_results: Dict, v6_results: Dict, output_dir: Path):
    """
    Figure 1: Main V4 vs V6 Comparison
    Shows the key finding: Position generalization works, Object generalization fails.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Get classifier performance
    v4_classifiers = v4_results["discrimination"]["classifier_performance"]
    v6_classifiers = v6_results["discrimination"]["classifier_performance"]

    # Use Random Forest as the main classifier for comparison
    rf_v4 = v4_classifiers.get("Random Forest", {})
    rf_v6 = v6_classifiers.get("Random Forest", {})

    # === Left Panel: V4 Position Generalization ===
    ax1 = axes[0]
    categories = ["Train", "Test", "Validation"]
    v4_values = [
        rf_v4.get("train_accuracy", 0) * 100,
        rf_v4.get("test_accuracy", 0) * 100,
        rf_v4.get("validation_accuracy", 0) * 100,
    ]
    colors_v4 = [COLORS["train"], COLORS["test"], COLORS["validation"]]

    bars1 = ax1.bar(
        categories, v4_values, color=colors_v4, edgecolor="black", linewidth=1.5
    )
    ax1.axhline(
        y=50,
        color=COLORS["random_chance"],
        linestyle="--",
        linewidth=2,
        label="Random Chance (50%)",
    )

    # Add value labels
    for bar, val in zip(bars1, v4_values):
        height = bar.get_height()
        ax1.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
        )

    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title(
        "V4: Position Generalization\n(Same Object, Different Position)",
        fontweight="bold",
        color=COLORS["success"],
    )
    ax1.set_ylim(0, 110)
    ax1.legend(loc="lower right")

    # Add success badge
    ax1.text(
        0.95,
        0.95,
        "✓ SUCCESS",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["success"],
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="#d5f4e6", edgecolor=COLORS["success"]),
    )

    # === Right Panel: V6 Object Generalization ===
    ax2 = axes[1]
    v6_values = [
        rf_v6.get("train_accuracy", 0) * 100,
        rf_v6.get("test_accuracy", 0) * 100,
        rf_v6.get("validation_accuracy", 0) * 100,
    ]
    categories_v6 = ["Train", "Test", "Holdout\n(New Object)"]
    colors_v6 = [COLORS["train"], COLORS["test"], COLORS["holdout"]]

    bars2 = ax2.bar(
        categories_v6, v6_values, color=colors_v6, edgecolor="black", linewidth=1.5
    )
    ax2.axhline(
        y=50,
        color=COLORS["random_chance"],
        linestyle="--",
        linewidth=2,
        label="Random Chance (50%)",
    )

    # Add value labels
    for bar, val in zip(bars2, v6_values):
        height = bar.get_height()
        ax2.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
        )

    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.set_title(
        "V6: Object Generalization\n(New Object, New Position)",
        fontweight="bold",
        color=COLORS["failure"],
    )
    ax2.set_ylim(0, 110)
    ax2.legend(loc="lower right")

    # Add failure badge
    ax2.text(
        0.95,
        0.95,
        "✗ FAILURE",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["failure"],
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="#fadbd8", edgecolor=COLORS["failure"]),
    )

    plt.suptitle(
        "Key Result: Position Generalization Works, Object Generalization Fails",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path = output_dir / "figure1_v4_vs_v6_main_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_all_classifiers_comparison(
    v4_results: Dict, v6_results: Dict, output_dir: Path
):
    """
    Figure 2: All Classifiers Performance Comparison
    Shows that the failure is consistent across all ML methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    v4_classifiers = v4_results["discrimination"]["classifier_performance"]
    v6_classifiers = v6_results["discrimination"]["classifier_performance"]

    # Common classifiers
    classifiers = [
        "Random Forest",
        "K-NN",
        "MLP (Medium)",
        "GPU-MLP (Medium-HighReg)",
        "Ensemble (Top3-MLP)",
    ]
    short_names = ["RF", "K-NN", "MLP", "GPU-MLP", "Ensemble"]

    x = np.arange(len(classifiers))
    width = 0.25

    # === Left Panel: V4 ===
    ax1 = axes[0]
    train_v4 = [
        v4_classifiers.get(c, {}).get("train_accuracy", 0) * 100 for c in classifiers
    ]
    test_v4 = [
        v4_classifiers.get(c, {}).get("test_accuracy", 0) * 100 for c in classifiers
    ]
    val_v4 = [
        v4_classifiers.get(c, {}).get("validation_accuracy", 0) * 100
        for c in classifiers
    ]

    bars1 = ax1.bar(
        x - width,
        train_v4,
        width,
        label="Train",
        color=COLORS["train"],
        edgecolor="black",
    )
    bars2 = ax1.bar(
        x, test_v4, width, label="Test", color=COLORS["test"], edgecolor="black"
    )
    bars3 = ax1.bar(
        x + width,
        val_v4,
        width,
        label="Validation",
        color=COLORS["validation"],
        edgecolor="black",
    )

    ax1.axhline(
        y=50, color=COLORS["random_chance"], linestyle="--", linewidth=2, alpha=0.7
    )
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title("V4: Position Generalization - All Classifiers", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=0)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="lower right")

    # Add validation accuracy labels
    for i, val in enumerate(val_v4):
        ax1.annotate(
            f"{val:.1f}%",
            xy=(x[i] + width, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # === Right Panel: V6 ===
    ax2 = axes[1]
    train_v6 = [
        v6_classifiers.get(c, {}).get("train_accuracy", 0) * 100 for c in classifiers
    ]
    test_v6 = [
        v6_classifiers.get(c, {}).get("test_accuracy", 0) * 100 for c in classifiers
    ]
    val_v6 = [
        v6_classifiers.get(c, {}).get("validation_accuracy", 0) * 100
        for c in classifiers
    ]

    bars4 = ax2.bar(
        x - width,
        train_v6,
        width,
        label="Train",
        color=COLORS["train"],
        edgecolor="black",
    )
    bars5 = ax2.bar(
        x, test_v6, width, label="Test", color=COLORS["test"], edgecolor="black"
    )
    bars6 = ax2.bar(
        x + width,
        val_v6,
        width,
        label="Holdout",
        color=COLORS["holdout"],
        edgecolor="black",
    )

    ax2.axhline(
        y=50,
        color=COLORS["random_chance"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Random (50%)",
    )
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.set_title("V6: Object Generalization - All Classifiers", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=0)
    ax2.set_ylim(0, 110)
    ax2.legend(loc="lower right")

    # Add holdout accuracy labels
    for i, val in enumerate(val_v6):
        ax2.annotate(
            f"{val:.1f}%",
            xy=(x[i] + width, val),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["failure"],
        )

    plt.suptitle(
        "Classifier Comparison: Object Generalization Fails Across ALL Methods",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path = output_dir / "figure2_all_classifiers_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_generalization_gap_figure(
    v4_results: Dict, v6_results: Dict, output_dir: Path
):
    """
    Figure 3: Generalization Gap Visualization
    Highlights the dramatic difference between test and validation performance.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    v4_disc = v4_results["discrimination"]["classifier_performance"]["Random Forest"]
    v6_disc = v6_results["discrimination"]["classifier_performance"]["Random Forest"]

    # Data
    experiments = ["V4: Position\nGeneralization", "V6: Object\nGeneralization"]
    test_acc = [v4_disc["test_accuracy"] * 100, v6_disc["test_accuracy"] * 100]
    val_acc = [
        v4_disc["validation_accuracy"] * 100,
        v6_disc["validation_accuracy"] * 100,
    ]
    gaps = [test_acc[0] - val_acc[0], test_acc[1] - val_acc[1]]

    x = np.arange(len(experiments))
    width = 0.35

    # Bars
    bars1 = ax.bar(
        x - width / 2,
        test_acc,
        width,
        label="Test Accuracy",
        color=COLORS["test"],
        edgecolor="black",
        linewidth=2,
    )
    bars2 = ax.bar(
        x + width / 2,
        val_acc,
        width,
        label="Validation/Holdout",
        color=[COLORS["validation"], COLORS["holdout"]],
        edgecolor="black",
        linewidth=2,
    )

    # Random chance line
    ax.axhline(
        y=50,
        color=COLORS["random_chance"],
        linestyle="--",
        linewidth=3,
        label="Random Chance (50%)",
    )

    # Gap annotations with arrows
    for i in range(len(experiments)):
        # Draw arrow showing gap
        ax.annotate(
            "",
            xy=(x[i] + width / 2, val_acc[i] + 2),
            xytext=(x[i] - width / 2, test_acc[i] - 2),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2),
        )

        # Gap label
        mid_y = (test_acc[i] + val_acc[i]) / 2
        gap_color = COLORS["success"] if gaps[i] < 30 else COLORS["failure"]
        ax.text(
            x[i],
            mid_y,
            f"Gap: {gaps[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=gap_color,
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor=gap_color, alpha=0.9
            ),
        )

    # Value labels on bars
    for bar, val in zip(bars1, test_acc):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    for bar, val in zip(bars2, val_acc):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=14)
    ax.set_title(
        "Generalization Gap: Test vs Validation Performance",
        fontweight="bold",
        fontsize=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=13)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=11)

    # Add insight box
    insight_text = "Key Insight:\n• V4: 25% gap → Moderate generalization\n• V6: 50% gap → Complete failure (random chance)"
    ax.text(
        0.02,
        0.98,
        insight_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="gray"
        ),
    )

    plt.tight_layout()
    output_path = output_dir / "figure3_generalization_gap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_sample_distribution_figure(
    v4_results: Dict, v6_results: Dict, output_dir: Path
):
    """
    Figure 4: Dataset Sample Distribution
    Shows the data split used in each experiment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # V4 data
    v4_disc = v4_results["discrimination"]
    v4_train = v4_disc.get("num_train_samples", 0)
    v4_test = v4_disc.get("num_test_samples", 0)
    v4_val = v4_disc.get("num_val_samples", 0)

    # V6 data
    v6_disc = v6_results["discrimination"]
    v6_train = v6_disc.get("num_train_samples", 0)
    v6_test = v6_disc.get("num_test_samples", 0)
    v6_val = v6_disc.get("num_val_samples", 0)

    # === V4 Pie ===
    ax1 = axes[0]
    sizes_v4 = [v4_train, v4_test, v4_val]
    labels_v4 = [
        f"Train\n({v4_train:,})",
        f"Test\n({v4_test:,})",
        f"Validation\n({v4_val:,})",
    ]
    colors_v4 = [COLORS["train"], COLORS["test"], COLORS["validation"]]
    explode_v4 = (0, 0, 0.1)  # Highlight validation

    wedges1, texts1, autotexts1 = ax1.pie(
        sizes_v4,
        explode=explode_v4,
        labels=labels_v4,
        colors=colors_v4,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
    )
    ax1.set_title(
        "V4: Position Generalization\n(WS2+WS3 → WS1)", fontweight="bold", fontsize=14
    )

    # Add total count
    total_v4 = sum(sizes_v4)
    ax1.text(
        0,
        -1.4,
        f"Total: {total_v4:,} samples",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    # === V6 Pie ===
    ax2 = axes[1]
    sizes_v6 = [v6_train, v6_test, v6_val]
    labels_v6 = [
        f"Train\n({v6_train:,})",
        f"Test\n({v6_test:,})",
        f"Holdout\n({v6_val:,})",
    ]
    colors_v6 = [COLORS["train"], COLORS["test"], COLORS["holdout"]]
    explode_v6 = (0, 0, 0.1)  # Highlight holdout

    wedges2, texts2, autotexts2 = ax2.pie(
        sizes_v6,
        explode=explode_v6,
        labels=labels_v6,
        colors=colors_v6,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
    )
    ax2.set_title(
        "V6: Object Generalization\n(WS1+WS2+WS3 → WS4 New Object)",
        fontweight="bold",
        fontsize=14,
    )

    # Add total count
    total_v6 = sum(sizes_v6)
    ax2.text(
        0,
        -1.4,
        f"Total: {total_v6:,} samples",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    plt.suptitle(
        "Dataset Distribution: Training vs Validation Split",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_path = output_dir / "figure4_sample_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_key_metrics_summary(v4_results: Dict, v6_results: Dict, output_dir: Path):
    """
    Figure 5: Key Metrics Summary Table
    A clean summary of all important numbers.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # Prepare data
    v4_disc = v4_results["discrimination"]
    v6_disc = v6_results["discrimination"]

    v4_rf = v4_disc["classifier_performance"]["Random Forest"]
    v6_rf = v6_disc["classifier_performance"]["Random Forest"]

    v4_best = v4_disc["best_classifier"]
    v6_best = v6_disc["best_classifier"]

    # Table data
    headers = ["Metric", "V4: Position Gen.", "V6: Object Gen.", "Interpretation"]

    data = [
        [
            "Experiment Type",
            "Same object,\ndifferent position",
            "New object,\nnew position",
            "",
        ],
        [
            "Training Samples",
            f"{v4_disc['num_train_samples']:,}",
            f"{v6_disc['num_train_samples']:,}",
            "",
        ],
        [
            "Validation Samples",
            f"{v4_disc['num_val_samples']:,}",
            f"{v6_disc['num_val_samples']:,}",
            "",
        ],
        ["", "", "", ""],  # Spacer
        [
            "Train Accuracy (RF)",
            f"{v4_rf['train_accuracy']*100:.1f}%",
            f"{v6_rf['train_accuracy']*100:.1f}%",
            "Both overfit",
        ],
        [
            "Test Accuracy (RF)",
            f"{v4_rf['test_accuracy']*100:.1f}%",
            f"{v6_rf['test_accuracy']*100:.1f}%",
            "Both excellent",
        ],
        [
            "Validation Accuracy (RF)",
            f"{v4_rf['validation_accuracy']*100:.1f}%",
            f"{v6_rf['validation_accuracy']*100:.1f}%",
            "V4 ✓, V6 ✗",
        ],
        ["", "", "", ""],  # Spacer
        ["Best Classifier", v4_best["name"], v6_best["name"], ""],
        [
            "Best Val Accuracy",
            f"{v4_best['validation_accuracy']*100:.1f}%",
            f"{v6_best['validation_accuracy']*100:.1f}%",
            "",
        ],
        ["", "", "", ""],  # Spacer
        [
            "Generalization Gap",
            f"{(v4_rf['test_accuracy'] - v4_rf['validation_accuracy'])*100:.1f}%",
            f"{(v6_rf['test_accuracy'] - v6_rf['validation_accuracy'])*100:.1f}%",
            "V6 gap = 50%!",
        ],
        ["Random Chance", "50%", "50%", "Binary classification"],
        ["Above Random?", "✓ YES (+25%)", "✗ NO (0%)", "Critical finding"],
    ]

    # Create table
    table = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor("#3498db")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color code results
    for i, row in enumerate(data, start=1):
        if "Validation Accuracy" in row[0]:
            table[(i, 1)].set_facecolor("#d5f4e6")  # Green for V4
            table[(i, 2)].set_facecolor("#fadbd8")  # Red for V6
        if "Above Random" in row[0]:
            table[(i, 1)].set_facecolor("#d5f4e6")
            table[(i, 2)].set_facecolor("#fadbd8")
        if row[0] == "":  # Spacer rows
            for j in range(4):
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title(
        "Key Metrics Summary: V4 vs V6 Experiments",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    output_path = output_dir / "figure5_key_metrics_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_training_datasets_figure(
    v4_results: Dict, v6_results: Dict, output_dir: Path
):
    """
    Figure 6: Workspace Rotation Experimental Strategy
    Shows the 3 rotation experiments for position generalization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # === Rotation 1: Train WS1+WS3, Validate WS2 ===
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")

    # Training workspaces
    ax1.add_patch(
        plt.Rectangle(
            (0.5, 5.5),
            3.5,
            3.5,
            facecolor=COLORS["train"],
            edgecolor="black",
            linewidth=2,
        )
    )
    ax1.text(
        2.25,
        7.25,
        "WS1 + WS3\n(Training)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax1.text(
        2.25,
        6,
        "7,290 samples",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Arrow
    ax1.annotate(
        "",
        xy=(6, 7.25),
        xytext=(4.5, 7.25),
        arrowprops=dict(arrowstyle="->", lw=3, color="black"),
    )
    ax1.text(5.25, 8, "Train", ha="center", fontsize=11, fontweight="bold")

    # Validation workspace
    ax1.add_patch(
        plt.Rectangle(
            (6, 5.5),
            3.5,
            3.5,
            facecolor=COLORS["validation"],
            edgecolor="black",
            linewidth=2,
        )
    )
    ax1.text(
        7.75,
        7.25,
        "WS2\n(Validate)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
    )
    ax1.text(
        7.75,
        6,
        "1,338 samples",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
    )

    # Result
    ax1.add_patch(
        plt.Rectangle(
            (2, 1),
            6,
            2.5,
            facecolor="#fadbd8",
            edgecolor=COLORS["failure"],
            linewidth=2,
        )
    )
    ax1.text(
        5,
        2.25,
        "CV: 69.1%\nVal: 55.7%",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax1.set_title(
        "Rotation 1: Train WS1+WS3 → Validate WS2",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    # === Rotation 2: Train WS2+WS3, Validate WS1 ===
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    # Training workspaces
    ax2.add_patch(
        plt.Rectangle(
            (0.5, 5.5),
            3.5,
            3.5,
            facecolor=COLORS["train"],
            edgecolor="black",
            linewidth=2,
        )
    )
    ax2.text(
        2.25,
        7.25,
        "WS2 + WS3\n(Training)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax2.text(
        2.25,
        6,
        "7,290 samples",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Arrow
    ax2.annotate(
        "",
        xy=(6, 7.25),
        xytext=(4.5, 7.25),
        arrowprops=dict(arrowstyle="->", lw=3, color="black"),
    )
    ax2.text(5.25, 8, "Train", ha="center", fontsize=11, fontweight="bold")

    # Validation workspace
    ax2.add_patch(
        plt.Rectangle(
            (6, 5.5),
            3.5,
            3.5,
            facecolor=COLORS["validation"],
            edgecolor="black",
            linewidth=2,
        )
    )
    ax2.text(
        7.75,
        7.25,
        "WS1\n(Validate)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
    )
    ax2.text(
        7.75,
        6,
        "1,362 samples",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
    )

    # Result
    ax2.add_patch(
        plt.Rectangle(
            (2, 1),
            6,
            2.5,
            facecolor="#fadbd8",
            edgecolor=COLORS["failure"],
            linewidth=2,
        )
    )
    ax2.text(
        5,
        2.25,
        "CV: 69.8%\nVal: 24.4%",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax2.set_title(
        "Rotation 2: Train WS2+WS3 → Validate WS1",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    # === Rotation 3: Train WS1+WS2, Validate WS3 ===
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")

    # Training workspaces
    ax3.add_patch(
        plt.Rectangle(
            (0.5, 5.5),
            3.5,
            3.5,
            facecolor=COLORS["train"],
            edgecolor="black",
            linewidth=2,
        )
    )
    ax3.text(
        2.25,
        7.25,
        "WS1 + WS2\n(Training)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax3.text(
        2.25,
        6,
        "8,028 samples",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Arrow
    ax3.annotate(
        "",
        xy=(6, 7.25),
        xytext=(4.5, 7.25),
        arrowprops=dict(arrowstyle="->", lw=3, color="black"),
    )
    ax3.text(5.25, 8, "Train", ha="center", fontsize=11, fontweight="bold")

    # Validation workspace
    ax3.add_patch(
        plt.Rectangle(
            (6, 5.5),
            3.5,
            3.5,
            facecolor=COLORS["validation"],
            edgecolor="black",
            linewidth=2,
        )
    )
    ax3.text(
        7.75,
        7.25,
        "WS3\n(Validate)",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
    )
    ax3.text(
        7.75,
        6,
        "1,215 samples",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
    )

    # Result
    ax3.add_patch(
        plt.Rectangle(
            (2, 1),
            6,
            2.5,
            facecolor="#fadbd8",
            edgecolor=COLORS["failure"],
            linewidth=2,
        )
    )
    ax3.text(
        5,
        2.25,
        "CV: 70.7%\nVal: 23.3%",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax3.set_title(
        "Rotation 3: Train WS1+WS2 → Validate WS3",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    plt.suptitle(
        "Workspace Rotation Experimental Strategy",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_path = output_dir / "figure6_experimental_setup.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_entanglement_concept_figure(output_dir: Path):
    """
    Figure 7: The Entanglement Problem Concept
    Visual explanation of why object generalization fails.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        7, 9.5, "The Entanglement Problem", ha="center", fontsize=20, fontweight="bold"
    )
    ax.text(
        7,
        8.8,
        "Why Acoustic Signatures Cannot Separate Contact from Object Properties",
        ha="center",
        fontsize=14,
        style="italic",
        color="gray",
    )

    # === Left side: What we want ===
    ax.add_patch(
        plt.Rectangle(
            (0.5, 4), 5, 4, facecolor="#e8f6f3", edgecolor="#1abc9c", linewidth=2
        )
    )
    ax.text(
        3,
        7.5,
        "What We Want",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#1abc9c",
    )

    # Contact box
    ax.add_patch(
        plt.Rectangle((1, 5.5), 1.8, 1.2, facecolor="#3498db", edgecolor="black")
    )
    ax.text(
        1.9,
        6.1,
        "Contact\nState",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    # Plus sign
    ax.text(3, 6.1, "+", ha="center", va="center", fontsize=24, fontweight="bold")

    # Object box
    ax.add_patch(
        plt.Rectangle((3.5, 5.5), 1.8, 1.2, facecolor="#9b59b6", edgecolor="black")
    )
    ax.text(
        4.4,
        6.1,
        "Object\nProps",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    # Equals
    ax.text(3, 4.8, "(Separable)", ha="center", fontsize=11, style="italic")

    # === Right side: What we get ===
    ax.add_patch(
        plt.Rectangle(
            (8.5, 4), 5, 4, facecolor="#fdedec", edgecolor="#e74c3c", linewidth=2
        )
    )
    ax.text(
        11,
        7.5,
        "What We Get",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#e74c3c",
    )

    # Entangled blob
    from matplotlib.patches import FancyBboxPatch

    entangled = FancyBboxPatch(
        (9.5, 5.2),
        3,
        1.8,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        facecolor="#8e44ad",
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(entangled)
    ax.text(
        11,
        6.1,
        "Contact ⊗ Object\n(Entangled)",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    ax.text(
        11,
        4.8,
        "(Inseparable)",
        ha="center",
        fontsize=11,
        style="italic",
        color="#e74c3c",
    )

    # Arrow between
    ax.annotate(
        "",
        xy=(8, 6),
        xytext=(6, 6),
        arrowprops=dict(arrowstyle="->", lw=3, color="gray"),
    )
    ax.text(7, 6.5, "Reality", ha="center", fontsize=12, color="gray")

    # === Bottom: Explanation ===
    explanation = """
    Physical Reality:
    • Acoustic signal S(t) = f(Contact, Object_Material, Object_Mass, Object_Geometry, ...)
    • These factors are multiplicatively coupled (⊗), not additively separable (+)
    • Model learns: "Object A sounds like THIS when touched" (instance-specific)
    • Model cannot learn: "Contact sounds like THIS regardless of object" (category-level)
    
    Result:
    • Same object, different position → Features still correlate → 75% accuracy ✓
    • Different object → Completely different feature space → 50% accuracy ✗
    """

    ax.text(
        7,
        2,
        explanation,
        ha="center",
        va="center",
        fontsize=11,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9
        ),
    )

    output_path = output_dir / "figure7_entanglement_concept.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def main():
    """Generate all ML analysis figures."""
    print("\n" + "=" * 70)
    print("📊 GENERATING ML ANALYSIS FIGURES")
    print("=" * 70)

    # Paths
    base_dir = Path("/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit")
    v4_dir = (
        base_dir
        / "training_truly_without_edge_with_handcrafted_features_with_threshold_v4"
    )
    v6_dir = (
        base_dir
        / "training_truly_without_edge_with_handcrafted_features_with_threshold_v6"
    )
    output_dir = base_dir / "ml_analysis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")

    # Load results
    print("\n📥 Loading experiment results...")
    v4_results = load_results(str(v4_dir))
    v6_results = load_results(str(v6_dir))

    if not v4_results.get("discrimination") or not v6_results.get("discrimination"):
        print("❌ Error: Could not load discrimination results")
        return

    print("  ✅ V4 results loaded")
    print("  ✅ V6 results loaded")

    # Generate figures
    print("\n🎨 Generating figures...")

    figures = []
    # figures.append(create_main_comparison_figure(v4_results, v6_results, output_dir))  # DISABLED: v4_vs_v6_main_comparison
    # figures.append(
    #     create_all_classifiers_comparison(v4_results, v6_results, output_dir)
    # )  # DISABLED: all_classifiers_comparison
    # figures.append(create_generalization_gap_figure(v4_results, v6_results, output_dir))  # DISABLED: generalization_gap
    figures.append(
        create_sample_distribution_figure(v4_results, v6_results, output_dir)
    )
    # figures.append(create_key_metrics_summary(v4_results, v6_results, output_dir))  # DISABLED: key_metrics_summary
    figures.append(create_training_datasets_figure(v4_results, v6_results, output_dir))
    figures.append(create_entanglement_concept_figure(output_dir))

    print("\n" + "=" * 70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Total figures: {len(figures)}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  • {f.name}")

    print("\n💡 Use these figures in your presentation and documentation!")


if __name__ == "__main__":
    main()
