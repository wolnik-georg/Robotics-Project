#!/usr/bin/env python3
"""
Generate a Proof of Concept figure showing ONLY Train and Test accuracy.
No validation - that comes later in the generalization slides.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Colors
COLORS = {
    "train": "#2ecc71",  # Green
    "test": "#3498db",  # Blue
    "random_chance": "#e67e22",  # Orange
}


def create_poc_train_test_figure():
    """Create POC figure with only Train/Test (no validation)."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data: Train and Test only (from 80/20 split of same distribution)
    # Actual values from experiments: Train 100%, Test 99.6%
    categories = ["Training\n(80%)", "Test\n(20%)"]
    values = [100, 99.6]  # Actual values from experiments
    colors = [COLORS["train"], COLORS["test"]]

    bars = ax.bar(
        categories, values, color=colors, edgecolor="black", linewidth=2, width=0.5
    )

    # Random chance line
    ax.axhline(
        y=50,
        color=COLORS["random_chance"],
        linestyle="--",
        linewidth=2,
        label="Random Chance (50%)",
    )

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
        )

    # Styling
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.set_title(
        "Proof of Concept: Classification on Known Data\n(All 3 Workspaces, 80/20 Split)",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=12)

    # Add annotation
    ax.annotate(
        "Same distribution\n(known surfaces & configurations)",
        xy=(0.5, 0.15),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    # Save
    output_path = (
        "../presentation_figures/proof_of_concept/poc_main_result_train_test_only.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    create_poc_train_test_figure()
