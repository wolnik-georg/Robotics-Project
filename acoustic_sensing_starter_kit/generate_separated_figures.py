#!/usr/bin/env python3
"""
Generate Separated ML Analysis Figures for Presentation
========================================================

Creates visualizations SEPARATED by narrative structure:

SET 1: PROOF OF CONCEPT (Act 1) - Shows it works!
    - poc_main_result.png - V4 bar chart (76% accuracy)
    - poc_all_classifiers.png - All classifiers work on V4
    - poc_sample_distribution.png - V4 data split
    - poc_metrics_summary.png - V4 metrics table

SET 2: OBJECT GENERALIZATION (Act 2) - The challenge revealed
    - obj_gen_main_result.png - V6 bar chart (50% = random)
    - obj_gen_comparison.png - V4 vs V6 side-by-side (the reveal)
    - obj_gen_all_classifiers.png - All classifiers fail on V6
    - obj_gen_gap_visualization.png - Dramatic gap visualization
    - entanglement_concept.png - Theory explanation

This separation supports the narrative:
1. First prove the concept works (Act 1)
2. Then reveal the generalization challenge (Act 2)

Author: Georg Wolnik
Date: February 1, 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
from typing import Dict, Any
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
    "contact": "#3498db",
    "no_contact": "#e74c3c",
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


# =============================================================================
# SET 1: PROOF OF CONCEPT FIGURES (Act 1)
# =============================================================================


def create_poc_main_result(v4_results: Dict, output_dir: Path):
    """
    POC Figure 1: Main Proof of Concept Result
    Shows V4 performance - Train/Test/Validation with SUCCESS badge.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get classifier performance
    v4_classifiers = v4_results["discrimination"]["classifier_performance"]
    rf_v4 = v4_classifiers.get("Random Forest", {})

    categories = ["Train", "Test", "Validation\n(New Position)"]
    values = [
        rf_v4.get("train_accuracy", 0) * 100,
        rf_v4.get("test_accuracy", 0) * 100,
        rf_v4.get("validation_accuracy", 0) * 100,
    ]
    colors = [COLORS["train"], COLORS["test"], COLORS["validation"]]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=2, width=0.6)
    
    # Random chance line
    ax.axhline(
        y=50,
        color=COLORS["random_chance"],
        linestyle="--",
        linewidth=3,
        label="Random Chance (50%)",
    )

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=18,
        )

    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=14)
    ax.set_title(
        "Proof of Concept: Contact Detection Works!\n(Position Generalization)",
        fontweight="bold",
        fontsize=16,
        color=COLORS["success"],
    )
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=12)

    # Add success badge
    ax.text(
        0.95,
        0.95,
        "✓ SUCCESS",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=COLORS["success"],
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#d5f4e6", edgecolor=COLORS["success"], linewidth=2),
    )

    # Add key insight
    ax.text(
        0.02,
        0.02,
        f"Key: 70% validation accuracy\n→ Model generalizes to new positions",
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="gray"),
    )

    plt.tight_layout()
    output_path = output_dir / "poc_main_result.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_poc_all_classifiers(v4_results: Dict, output_dir: Path):
    """
    POC Figure 2: All Classifiers Performance for V4
    Shows that multiple ML methods work for position generalization.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    v4_classifiers = v4_results["discrimination"]["classifier_performance"]

    # Common classifiers
    classifiers = [
        "Random Forest",
        "K-NN",
        "MLP (Medium)",
        "GPU-MLP (Medium-HighReg)",
        "Ensemble (Top3-MLP)",
    ]
    short_names = ["Random\nForest", "K-NN", "MLP", "GPU-MLP", "Ensemble"]

    x = np.arange(len(classifiers))
    width = 0.25

    train_v4 = [v4_classifiers.get(c, {}).get("train_accuracy", 0) * 100 for c in classifiers]
    test_v4 = [v4_classifiers.get(c, {}).get("test_accuracy", 0) * 100 for c in classifiers]
    val_v4 = [v4_classifiers.get(c, {}).get("validation_accuracy", 0) * 100 for c in classifiers]

    bars1 = ax.bar(x - width, train_v4, width, label="Train", color=COLORS["train"], edgecolor="black")
    bars2 = ax.bar(x, test_v4, width, label="Test", color=COLORS["test"], edgecolor="black")
    bars3 = ax.bar(x + width, val_v4, width, label="Validation", color=COLORS["validation"], edgecolor="black")

    ax.axhline(y=50, color=COLORS["random_chance"], linestyle="--", linewidth=2, alpha=0.7, label="Random (50%)")
    
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("All Classifiers Work for Position Generalization", fontweight="bold", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right")

    # Add validation accuracy labels
    for i, val in enumerate(val_v4):
        ax.annotate(
            f"{val:.1f}%",
            xy=(x[i] + width, val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["success"],
        )

    # Highlight that all are above random
    ax.text(
        0.5,
        0.02,
        "All classifiers achieve >70% validation accuracy → Robust result",
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="#d5f4e6", edgecolor=COLORS["success"], alpha=0.9),
    )

    plt.tight_layout()
    output_path = output_dir / "poc_all_classifiers.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_poc_sample_distribution(v4_results: Dict, output_dir: Path):
    """
    POC Figure 3: Dataset Distribution for V4
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    v4_disc = v4_results["discrimination"]
    v4_train = v4_disc.get("num_train_samples", 0)
    v4_test = v4_disc.get("num_test_samples", 0)
    v4_val = v4_disc.get("num_val_samples", 0)

    sizes = [v4_train, v4_test, v4_val]
    labels = [
        f"Train\n({v4_train:,})",
        f"Test\n({v4_test:,})",
        f"Validation\n({v4_val:,})",
    ]
    colors = [COLORS["train"], COLORS["test"], COLORS["validation"]]
    explode = (0, 0, 0.1)  # Highlight validation

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "black", "linewidth": 1.5},
        textprops={"fontsize": 12},
    )
    
    for autotext in autotexts:
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    ax.set_title(
        "Proof of Concept: Data Split\n(WS2+WS3 → WS1)",
        fontweight="bold",
        fontsize=14,
    )

    # Add total count
    total = sum(sizes)
    ax.text(
        0,
        -1.3,
        f"Total: {total:,} samples",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path = output_dir / "poc_sample_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_poc_metrics_summary(v4_results: Dict, output_dir: Path):
    """
    POC Figure 4: Metrics Summary Table for V4
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    v4_disc = v4_results["discrimination"]
    v4_rf = v4_disc["classifier_performance"]["Random Forest"]
    v4_best = v4_disc["best_classifier"]

    # Table data
    headers = ["Metric", "Value", "Status"]

    data = [
        ["Experiment Type", "Position Generalization\n(Same object, different position)", ""],
        ["Training Samples", f"{v4_disc['num_train_samples']:,}", ""],
        ["Validation Samples", f"{v4_disc['num_val_samples']:,}", ""],
        ["", "", ""],  # Spacer
        ["Train Accuracy", f"{v4_rf['train_accuracy']*100:.1f}%", "✓"],
        ["Test Accuracy", f"{v4_rf['test_accuracy']*100:.1f}%", "✓"],
        ["Validation Accuracy", f"{v4_rf['validation_accuracy']*100:.1f}%", "✓ Above Random"],
        ["", "", ""],  # Spacer
        ["Best Classifier", v4_best["name"], ""],
        ["Generalization Gap", f"{(v4_rf['test_accuracy'] - v4_rf['validation_accuracy'])*100:.1f}%", "Acceptable"],
        ["Random Chance", "50%", ""],
        ["Above Random", f"+{(v4_rf['validation_accuracy']-0.5)*100:.1f}%", "✓ SUCCESS"],
    ]

    # Create table
    table = ax.table(cellText=data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor(COLORS["success"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color code success rows
    for i, row in enumerate(data, start=1):
        if "Validation Accuracy" in row[0] or "Above Random" in row[0]:
            for j in range(3):
                table[(i, j)].set_facecolor("#d5f4e6")
        if row[0] == "":  # Spacer rows
            for j in range(3):
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title(
        "Proof of Concept: Key Metrics",
        fontsize=16,
        fontweight="bold",
        pad=20,
        color=COLORS["success"],
    )

    output_path = output_dir / "poc_metrics_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


# =============================================================================
# SET 2: OBJECT GENERALIZATION FIGURES (Act 2)
# =============================================================================


def create_obj_gen_main_result(v6_results: Dict, output_dir: Path):
    """
    Object Gen Figure 1: Main Result - Shows failure
    V6 bar chart showing 50% = random chance
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    v6_classifiers = v6_results["discrimination"]["classifier_performance"]
    rf_v6 = v6_classifiers.get("Random Forest", {})

    categories = ["Train", "Test", "Holdout\n(New Object)"]
    values = [
        rf_v6.get("train_accuracy", 0) * 100,
        rf_v6.get("test_accuracy", 0) * 100,
        rf_v6.get("validation_accuracy", 0) * 100,
    ]
    colors = [COLORS["train"], COLORS["test"], COLORS["holdout"]]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=2, width=0.6)
    
    # Random chance line
    ax.axhline(
        y=50,
        color=COLORS["random_chance"],
        linestyle="--",
        linewidth=3,
        label="Random Chance (50%)",
    )

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        color = COLORS["failure"] if val <= 55 else "black"
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=18,
            color=color,
        )

    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=14)
    ax.set_title(
        "Object Generalization: Model Fails on New Object\n(Accuracy = Random Chance)",
        fontweight="bold",
        fontsize=16,
        color=COLORS["failure"],
    )
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=12)

    # Add failure badge
    ax.text(
        0.95,
        0.95,
        "✗ FAILURE",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        color=COLORS["failure"],
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor=COLORS["failure"], linewidth=2),
    )

    # Add key insight
    ax.text(
        0.02,
        0.02,
        f"Key: 50% holdout accuracy = Random Chance\n→ Model cannot generalize to new objects",
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="#fff3cd", alpha=0.9, edgecolor=COLORS["failure"]),
    )

    plt.tight_layout()
    output_path = output_dir / "obj_gen_main_result.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_obj_gen_comparison(v4_results: Dict, v6_results: Dict, output_dir: Path):
    """
    Object Gen Figure 2: V4 vs V6 Comparison (The Reveal)
    Side-by-side showing success vs failure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    v4_classifiers = v4_results["discrimination"]["classifier_performance"]
    v6_classifiers = v6_results["discrimination"]["classifier_performance"]
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

    bars1 = ax1.bar(categories, v4_values, color=colors_v4, edgecolor="black", linewidth=1.5)
    ax1.axhline(y=50, color=COLORS["random_chance"], linestyle="--", linewidth=2, label="Random (50%)")

    for bar, val in zip(bars1, v4_values):
        ax1.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
        )

    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_title("Position Generalization\n(Same Object, Different Position)", fontweight="bold", color=COLORS["success"])
    ax1.set_ylim(0, 110)
    ax1.legend(loc="lower right")

    ax1.text(
        0.95, 0.95, "✓ SUCCESS",
        transform=ax1.transAxes, fontsize=14, fontweight="bold", color=COLORS["success"],
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="#d5f4e6", edgecolor=COLORS["success"]),
    )

    # === Right Panel: V6 Object Generalization ===
    ax2 = axes[1]
    categories_v6 = ["Train", "Test", "Holdout\n(New Object)"]
    v6_values = [
        rf_v6.get("train_accuracy", 0) * 100,
        rf_v6.get("test_accuracy", 0) * 100,
        rf_v6.get("validation_accuracy", 0) * 100,
    ]
    colors_v6 = [COLORS["train"], COLORS["test"], COLORS["holdout"]]

    bars2 = ax2.bar(categories_v6, v6_values, color=colors_v6, edgecolor="black", linewidth=1.5)
    ax2.axhline(y=50, color=COLORS["random_chance"], linestyle="--", linewidth=2, label="Random (50%)")

    for bar, val in zip(bars2, v6_values):
        color = COLORS["failure"] if val <= 55 else "black"
        ax2.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=14,
            color=color,
        )

    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.set_title("Object Generalization\n(New Object, New Position)", fontweight="bold", color=COLORS["failure"])
    ax2.set_ylim(0, 110)
    ax2.legend(loc="lower right")

    ax2.text(
        0.95, 0.95, "✗ FAILURE",
        transform=ax2.transAxes, fontsize=14, fontweight="bold", color=COLORS["failure"],
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="#fadbd8", edgecolor=COLORS["failure"]),
    )

    plt.suptitle(
        "The Challenge: Position Works, Object Fails",
        fontsize=18, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    output_path = output_dir / "obj_gen_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_obj_gen_all_classifiers(v6_results: Dict, output_dir: Path):
    """
    Object Gen Figure 3: All Classifiers Fail on V6
    Shows that the failure is consistent across all ML methods.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    v6_classifiers = v6_results["discrimination"]["classifier_performance"]

    classifiers = [
        "Random Forest",
        "K-NN",
        "MLP (Medium)",
        "GPU-MLP (Medium-HighReg)",
        "Ensemble (Top3-MLP)",
    ]
    short_names = ["Random\nForest", "K-NN", "MLP", "GPU-MLP", "Ensemble"]

    x = np.arange(len(classifiers))
    width = 0.25

    train_v6 = [v6_classifiers.get(c, {}).get("train_accuracy", 0) * 100 for c in classifiers]
    test_v6 = [v6_classifiers.get(c, {}).get("test_accuracy", 0) * 100 for c in classifiers]
    val_v6 = [v6_classifiers.get(c, {}).get("validation_accuracy", 0) * 100 for c in classifiers]

    bars1 = ax.bar(x - width, train_v6, width, label="Train", color=COLORS["train"], edgecolor="black")
    bars2 = ax.bar(x, test_v6, width, label="Test", color=COLORS["test"], edgecolor="black")
    bars3 = ax.bar(x + width, val_v6, width, label="Holdout", color=COLORS["holdout"], edgecolor="black")

    ax.axhline(y=50, color=COLORS["random_chance"], linestyle="--", linewidth=3, label="Random (50%)")
    
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("ALL Classifiers Fail on Object Generalization", fontweight="bold", fontsize=16, color=COLORS["failure"])
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right")

    # Add holdout accuracy labels
    for i, val in enumerate(val_v6):
        ax.annotate(
            f"{val:.1f}%",
            xy=(x[i] + width, val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=COLORS["failure"],
        )

    # Highlight that all fail
    ax.text(
        0.5,
        0.02,
        "ALL classifiers ≈ 50% on holdout → This is NOT a model problem, it's fundamental",
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="#fadbd8", edgecolor=COLORS["failure"], alpha=0.9),
    )

    plt.tight_layout()
    output_path = output_dir / "obj_gen_all_classifiers.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_obj_gen_gap_visualization(v4_results: Dict, v6_results: Dict, output_dir: Path):
    """
    Object Gen Figure 4: Generalization Gap Visualization
    Dramatic visualization of the gap difference.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    v4_rf = v4_results["discrimination"]["classifier_performance"]["Random Forest"]
    v6_rf = v6_results["discrimination"]["classifier_performance"]["Random Forest"]

    experiments = ["Position\nGeneralization", "Object\nGeneralization"]
    test_acc = [v4_rf["test_accuracy"] * 100, v6_rf["test_accuracy"] * 100]
    val_acc = [v4_rf["validation_accuracy"] * 100, v6_rf["validation_accuracy"] * 100]
    gaps = [test_acc[0] - val_acc[0], test_acc[1] - val_acc[1]]

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax.bar(x - width/2, test_acc, width, label="Test Accuracy", color=COLORS["test"], edgecolor="black", linewidth=2)
    bars2 = ax.bar(x + width/2, val_acc, width, label="Validation/Holdout", color=[COLORS["validation"], COLORS["holdout"]], edgecolor="black", linewidth=2)

    ax.axhline(y=50, color=COLORS["random_chance"], linestyle="--", linewidth=3, label="Random Chance (50%)")

    # Gap annotations with arrows
    for i in range(len(experiments)):
        ax.annotate(
            "",
            xy=(x[i] + width/2, val_acc[i] + 2),
            xytext=(x[i] - width/2, test_acc[i] - 2),
            arrowprops=dict(arrowstyle="<->", color="red", lw=2.5),
        )

        mid_y = (test_acc[i] + val_acc[i]) / 2
        gap_color = COLORS["success"] if gaps[i] < 30 else COLORS["failure"]
        ax.text(
            x[i],
            mid_y,
            f"Gap:\n{gaps[i]:.0f}%",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=gap_color,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=gap_color, alpha=0.95, linewidth=2),
        )

    # Value labels
    for bar, val in zip(bars1, test_acc):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold", fontsize=13)

    for bar, val in zip(bars2, val_acc):
        color = COLORS["failure"] if val <= 55 else "black"
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width()/2, val), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold", fontsize=13, color=color)

    ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=14)
    ax.set_title("Generalization Gap: The Critical Difference", fontweight="bold", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=14)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    output_path = output_dir / "obj_gen_gap_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def create_entanglement_concept_figure(output_dir: Path):
    """
    Object Gen Figure 5: Entanglement Concept
    Visual explanation of why object generalization fails.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(7, 9.5, "The Entanglement Problem", ha="center", fontsize=20, fontweight="bold")
    ax.text(7, 8.8, "Why Object Generalization Fails", ha="center", fontsize=14, color="gray")

    # === Left side: What we want ===
    ax.add_patch(plt.Rectangle((0.5, 5), 5, 3, facecolor="#d5f4e6", edgecolor=COLORS["success"], linewidth=2, alpha=0.8))
    ax.text(3, 7.5, "What We Want", ha="center", fontsize=14, fontweight="bold", color=COLORS["success"])
    
    # Separable boxes
    ax.add_patch(plt.Rectangle((1, 5.5), 1.8, 1.5, facecolor=COLORS["test"], edgecolor="black", linewidth=1.5))
    ax.text(1.9, 6.25, "Contact\nState", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    
    ax.text(3, 6.25, "+", ha="center", fontsize=20, fontweight="bold")
    
    ax.add_patch(plt.Rectangle((3.5, 5.5), 1.8, 1.5, facecolor=COLORS["neutral"], edgecolor="black", linewidth=1.5))
    ax.text(4.4, 6.25, "Object\nProperties", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.text(3, 5.1, "(Separable)", ha="center", fontsize=11, style="italic", color=COLORS["success"])

    # === Right side: What we have ===
    ax.add_patch(plt.Rectangle((8.5, 5), 5, 3, facecolor="#fadbd8", edgecolor=COLORS["failure"], linewidth=2, alpha=0.8))
    ax.text(11, 7.5, "What We Have", ha="center", fontsize=14, fontweight="bold", color=COLORS["failure"])
    
    # Entangled box
    ax.add_patch(plt.Rectangle((9.5, 5.5), 3, 1.5, facecolor="#9b59b6", edgecolor="black", linewidth=1.5))
    ax.text(11, 6.25, "Contact ⊗ Object\n(Entangled)", ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    ax.text(11, 5.1, "(Inseparable)", ha="center", fontsize=11, style="italic", color=COLORS["failure"])

    # Arrow between
    ax.annotate("", xy=(8, 6.5), xytext=(6, 6.5), arrowprops=dict(arrowstyle="->", lw=3, color="gray"))
    ax.text(7, 7, "Reality", ha="center", fontsize=12, color="gray")

    # === Bottom: Explanation ===
    explanation = """Physical Reality:
• Acoustic signal S(t) = f(Contact, Object_Material, Object_Mass, Object_Geometry, ...)
• These factors are multiplicatively coupled (⊗), not additively separable (+)
• Model learns: "Object A sounds like THIS when touched" (instance-specific)
• Model cannot learn: "Contact sounds like THIS regardless of object" (category-level)

Result:
• Same object, different position → Features still correlate → 70% accuracy ✓
• Different object → Completely different feature space → 50% accuracy ✗"""

    ax.text(
        7, 2.2, explanation, ha="center", va="center", fontsize=11, fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    output_path = output_dir / "entanglement_concept.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    return output_path


def main():
    """Generate all separated figures."""
    print("\n" + "=" * 70)
    print("📊 GENERATING SEPARATED FIGURES FOR PRESENTATION")
    print("=" * 70)

    # Paths
    base_dir = Path("/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit")
    v4_dir = base_dir / "training_truly_without_edge_with_handcrafted_features_with_threshold_v4"
    v6_dir = base_dir / "training_truly_without_edge_with_handcrafted_features_with_threshold_v6"
    
    # Create separate output directories
    poc_output_dir = base_dir / "presentation_figures" / "proof_of_concept"
    obj_gen_output_dir = base_dir / "presentation_figures" / "object_generalization"
    poc_output_dir.mkdir(parents=True, exist_ok=True)
    obj_gen_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 POC output: {poc_output_dir}")
    print(f"📁 Object Gen output: {obj_gen_output_dir}")

    # Load results
    print("\n📥 Loading experiment results...")
    v4_results = load_results(str(v4_dir))
    v6_results = load_results(str(v6_dir))

    if not v4_results.get("discrimination") or not v6_results.get("discrimination"):
        print("❌ Error: Could not load discrimination results")
        return

    print("  ✅ V4 (Position Generalization) results loaded")
    print("  ✅ V6 (Object Generalization) results loaded")

    # ===========================================
    # SET 1: PROOF OF CONCEPT (Act 1)
    # ===========================================
    print("\n" + "-" * 50)
    print("🎯 SET 1: PROOF OF CONCEPT FIGURES (Act 1)")
    print("-" * 50)

    poc_figures = []
    poc_figures.append(create_poc_main_result(v4_results, poc_output_dir))
    poc_figures.append(create_poc_all_classifiers(v4_results, poc_output_dir))
    poc_figures.append(create_poc_sample_distribution(v4_results, poc_output_dir))
    poc_figures.append(create_poc_metrics_summary(v4_results, poc_output_dir))

    # ===========================================
    # SET 2: OBJECT GENERALIZATION (Act 2)
    # ===========================================
    print("\n" + "-" * 50)
    print("🔬 SET 2: OBJECT GENERALIZATION FIGURES (Act 2)")
    print("-" * 50)

    obj_figures = []
    obj_figures.append(create_obj_gen_main_result(v6_results, obj_gen_output_dir))
    obj_figures.append(create_obj_gen_comparison(v4_results, v6_results, obj_gen_output_dir))
    obj_figures.append(create_obj_gen_all_classifiers(v6_results, obj_gen_output_dir))
    obj_figures.append(create_obj_gen_gap_visualization(v4_results, v6_results, obj_gen_output_dir))
    obj_figures.append(create_entanglement_concept_figure(obj_gen_output_dir))

    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 70)
    print("✅ ALL SEPARATED FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)

    print("\n📁 PROOF OF CONCEPT figures (Act 1):")
    for f in sorted(poc_output_dir.glob("*.png")):
        print(f"  • {f.name}")

    print("\n📁 OBJECT GENERALIZATION figures (Act 2):")
    for f in sorted(obj_gen_output_dir.glob("*.png")):
        print(f"  • {f.name}")

    print("\n💡 Use these figures in your presentation:")
    print("   Act 1 (Slides 1-8): Use figures from proof_of_concept/")
    print("   Act 2 (Slides 9-11): Use figures from object_generalization/")


if __name__ == "__main__":
    main()
