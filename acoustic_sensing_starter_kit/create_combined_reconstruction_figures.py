#!/usr/bin/env python3
"""
Create combined reconstruction figures for the final report.

This script combines individual reconstruction visualizations into three figures:
1. Test data (WS1 - proof of concept, best performance)
2. Validation data (WS2 - position generalization)
3. Holdout data (WS4 - object generalization failure)
"""

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

# Base directory
BASE_DIR = Path(__file__).resolve().parent
RECON_DIR = BASE_DIR / "comprehensive_3class_reconstruction"
OUTPUT_DIR = RECON_DIR


def load_comparison_image(dataset_path):
    """Load the 03_comparison.png from a dataset reconstruction."""
    # The reconstruction creates a nested directory with the same name
    dataset_name = dataset_path.name
    comparison_path = dataset_path / dataset_name / "03_comparison.png"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison image not found: {comparison_path}")
    return Image.open(comparison_path)


def create_test_figure():
    """Create combined figure for test data (WS2 objects A, B, C)."""
    print("\n" + "=" * 80)
    print("Creating TEST DATA combined figure (Proof of Concept)")
    print("=" * 80)

    # Use WS2 validation data (best performance workspace - 84.9% accuracy)
    datasets = [
        "validation/balanced_workspace_2_3class_squares_cutout",
        "validation/balanced_workspace_2_3class_pure_no_contact",
        "validation/balanced_workspace_2_3class_pure_contact",
    ]

    labels = [
        "Object A (Cutout)",
        "Object B (Empty)",
        "Object C (Full Contact)",
    ]

    # Smaller figure size to fit on one page
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    fig.suptitle(
        "Test Data Reconstruction (WS2 - Position Generalization, 84.9% Accuracy)",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (dataset, label) in enumerate(zip(datasets, labels)):
        dataset_path = RECON_DIR / dataset
        img = load_comparison_image(dataset_path)

        axes[idx].imshow(img)
        axes[idx].set_title(label, fontsize=12, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()

    output_png = OUTPUT_DIR / "test_reconstruction_combined.png"
    output_pdf = OUTPUT_DIR / "test_reconstruction_combined.pdf"

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()

    print(f"✓ Created: {output_png}")
    print(f"✓ Created: {output_pdf}")

    return output_png


def create_validation_figure():
    """Create combined figure for validation data (WS1 objects A, B, C)."""
    print("\n" + "=" * 80)
    print("Creating VALIDATION DATA combined figure (Position Generalization)")
    print("=" * 80)

    # Use WS1 data (held out during training in Rotation 2)
    datasets = [
        "test/balanced_workspace_1_3class_squares_cutout",
        "test/balanced_workspace_1_3class_pure_no_contact",
        "test/balanced_workspace_1_3class_pure_contact",
    ]

    labels = [
        "Object A (Cutout)",
        "Object B (Empty)",
        "Object C (Full Contact)",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(
        "Validation Data Reconstruction (WS1 - Position Generalization, 60.4% Accuracy)",
        fontsize=16,
        fontweight="bold",
    )

    for idx, (dataset, label) in enumerate(zip(datasets, labels)):
        dataset_path = RECON_DIR / dataset
        img = load_comparison_image(dataset_path)

        axes[idx].imshow(img)
        axes[idx].set_title(label, fontsize=14, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()

    output_png = OUTPUT_DIR / "validation_reconstruction_combined.png"
    output_pdf = OUTPUT_DIR / "validation_reconstruction_combined.pdf"

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()

    print(f"✓ Created: {output_png}")
    print(f"✓ Created: {output_pdf}")

    return output_png


def create_holdout_figure():
    """Create figure for holdout data (WS4 Object D - novel geometry)."""
    print("\n" + "=" * 80)
    print("Creating HOLDOUT DATA figure (Object Generalization Failure)")
    print("=" * 80)

    dataset_path = RECON_DIR / "holdout/balanced_holdout_3class"
    img = load_comparison_image(dataset_path)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(
        "Holdout Data Reconstruction (WS4 Object D - Novel Geometry, 50% Accuracy = Random Chance)",
        fontsize=16,
        fontweight="bold",
    )

    ax.imshow(img)
    ax.set_title(
        "Object D (Novel Geometry - Generalization Failure)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()

    output_png = OUTPUT_DIR / "holdout_reconstruction_combined.png"
    output_pdf = OUTPUT_DIR / "holdout_reconstruction_combined.pdf"

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()

    print(f"✓ Created: {output_png}")
    print(f"✓ Created: {output_pdf}")

    return output_png


def main():
    """Create all three combined figures."""
    print("\n" + "=" * 80)
    print("CREATING COMBINED RECONSTRUCTION FIGURES")
    print("=" * 80)

    # Create all three figures
    test_fig = create_test_figure()
    validation_fig = create_validation_figure()
    holdout_fig = create_holdout_figure()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nAll combined figures created in: {OUTPUT_DIR}")
    print(f"\n1. Test data (proof of concept): {test_fig.name}")
    print(f"2. Validation data (position gen): {validation_fig.name}")
    print(f"3. Holdout data (object gen): {holdout_fig.name}")
    print("\nThese figures are now referenced in final_report.tex")
    print("Ready to compile the LaTeX document!")


if __name__ == "__main__":
    main()
