#!/usr/bin/env python3
"""
Create side-by-side TEST vs HOLDOUT visualization for presentation.
Shows the clear generalization gap between training workspaces and unseen workspace.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def main():
    base_dir = Path(__file__).parent
    comparison_dir = base_dir / "test_vs_holdout_comparison"
    output_dir = base_dir / "presentation_test_vs_holdout_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Create 2x2 comparison: TEST (best) vs HOLDOUT comparison plots
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 Creating TEST vs HOLDOUT Comparison Visualization")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top row: Best TEST case (WS2 - 100% accuracy)
    test_img = mpimg.imread(
        comparison_dir
        / "TEST_WS2_squares"
        / "balanced_workspace_2_squares_cutout_comparison.png"
    )
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title(
        "TEST: Workspace 2 (100% Accuracy)",
        fontsize=14,
        fontweight="bold",
        color="green",
    )
    axes[0, 0].axis("off")

    test_error = mpimg.imread(
        comparison_dir
        / "TEST_WS2_squares"
        / "balanced_workspace_2_squares_cutout_error_map.png"
    )
    axes[0, 1].imshow(test_error)
    axes[0, 1].set_title(
        "TEST: Error Map (No Errors)", fontsize=14, fontweight="bold", color="green"
    )
    axes[0, 1].axis("off")

    # Bottom row: HOLDOUT (58.4% accuracy - near random)
    holdout_img = mpimg.imread(
        comparison_dir / "HOLDOUT_WS4" / "balanced_holdout_oversample_comparison.png"
    )
    axes[1, 0].imshow(holdout_img)
    axes[1, 0].set_title(
        "HOLDOUT: Workspace 4 (58.4% Accuracy)",
        fontsize=14,
        fontweight="bold",
        color="red",
    )
    axes[1, 0].axis("off")

    holdout_error = mpimg.imread(
        comparison_dir / "HOLDOUT_WS4" / "balanced_holdout_oversample_error_map.png"
    )
    axes[1, 1].imshow(holdout_error)
    axes[1, 1].set_title(
        "HOLDOUT: Error Map (Many Errors)", fontsize=14, fontweight="bold", color="red"
    )
    axes[1, 1].axis("off")

    plt.suptitle(
        "Generalization Gap: Training vs Unseen Workspace\n"
        + "Model trained on WS1+WS2+WS3, tested on unseen WS4",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = output_dir / "test_vs_holdout_comparison_2x2.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Saved: {output_path}")

    # =========================================================================
    # 2. Create all workspaces comparison (1 row per workspace)
    # =========================================================================
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))

    datasets = [
        (
            "TEST_WS1_squares",
            "Workspace 1 (84.6%)",
            "green",
            "balanced_workspace_1_squares_cutout_oversample",
        ),
        (
            "TEST_WS2_squares",
            "Workspace 2 (100%)",
            "green",
            "balanced_workspace_2_squares_cutout",
        ),
        (
            "TEST_WS3_squares",
            "Workspace 3 (100%)",
            "green",
            "balanced_workspace_3_squares_cutout_v1",
        ),
        (
            "HOLDOUT_WS4",
            "Workspace 4 - HOLDOUT (58.4%)",
            "red",
            "balanced_holdout_oversample",
        ),
    ]

    for row, (folder, title, color, dataset_name) in enumerate(datasets):
        folder_path = comparison_dir / folder

        # Comparison
        comp_path = folder_path / f"{dataset_name}_comparison.png"
        if comp_path.exists():
            img = mpimg.imread(comp_path)
            axes[row, 0].imshow(img)
        axes[row, 0].set_title(
            f"{title}\nGround Truth vs Prediction", fontsize=11, color=color
        )
        axes[row, 0].axis("off")

        # Error map
        error_path = folder_path / f"{dataset_name}_error_map.png"
        if error_path.exists():
            img = mpimg.imread(error_path)
            axes[row, 1].imshow(img)
        axes[row, 1].set_title(f"{title}\nError Map", fontsize=11, color=color)
        axes[row, 1].axis("off")

        # Confidence
        conf_path = folder_path / f"{dataset_name}_confidence.png"
        if conf_path.exists():
            img = mpimg.imread(conf_path)
            axes[row, 2].imshow(img)
        axes[row, 2].set_title(f"{title}\nConfidence Map", fontsize=11, color=color)
        axes[row, 2].axis("off")

    plt.suptitle(
        "Surface Reconstruction: All Workspaces Comparison\n"
        + "WS1-3: Training Data | WS4: Completely Unseen Holdout",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / "all_workspaces_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Saved: {output_path}")

    # =========================================================================
    # 3. Create focused comparison: Best TEST vs HOLDOUT only
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Best TEST (WS2)
    test_img = mpimg.imread(
        comparison_dir
        / "TEST_WS2_squares"
        / "balanced_workspace_2_squares_cutout_comparison.png"
    )
    axes[0].imshow(test_img)
    axes[0].set_title(
        "TEST: Workspace 2\n100% Reconstruction Accuracy",
        fontsize=14,
        fontweight="bold",
        color="green",
    )
    axes[0].axis("off")

    # HOLDOUT
    holdout_img = mpimg.imread(
        comparison_dir / "HOLDOUT_WS4" / "balanced_holdout_oversample_comparison.png"
    )
    axes[1].imshow(holdout_img)
    axes[1].set_title(
        "HOLDOUT: Workspace 4 (Unseen)\n58.4% Accuracy (Near Random)",
        fontsize=14,
        fontweight="bold",
        color="red",
    )
    axes[1].axis("off")

    plt.suptitle(
        "Generalization Gap: 41.6% Drop on Unseen Workspace",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    output_path = output_dir / "test_vs_holdout_sidebyside.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ Saved: {output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("📁 All visualizations saved to:", output_dir)
    print("=" * 70)
    print("\nFiles created:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  📊 {f}")

    print("\n💡 Key visualizations for presentation:")
    print("  1. test_vs_holdout_sidebyside.png - Simple 2-panel comparison")
    print("  2. test_vs_holdout_comparison_2x2.png - Comparison + Error maps")
    print("  3. all_workspaces_comparison.png - Complete 4-workspace overview")


if __name__ == "__main__":
    main()
