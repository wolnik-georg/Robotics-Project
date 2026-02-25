#!/usr/bin/env python3
"""
Create combined figure showing GPU-MLP HighReg accuracy-coverage tradeoff.
Left: No confidence filtering (33.03% accuracy, all 2,280 samples)
Right: Confidence threshold 0.7 (75% accuracy, only 4/2,280 samples = 0.2%)
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def create_tradeoff_comparison():
    """Create side-by-side comparison of unfiltered vs filtered reconstruction."""

    # Paths to individual reconstruction summary figures
    no_filter_path = Path(
        "comprehensive_3class_reconstruction/holdout_reconstruction/gpu_mlp_highreg_no_threshold/fully_balanced_datasets/workspace_4_balanced/06_presentation_summary.png"
    )
    filtered_path = Path(
        "comprehensive_3class_reconstruction/holdout_reconstruction/gpu_mlp_highreg_seed42_threshold07/fully_balanced_datasets/workspace_4_balanced/06_presentation_summary.png"
    )

    # Verify both files exist
    if not no_filter_path.exists():
        raise FileNotFoundError(f"No filter reconstruction not found: {no_filter_path}")
    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered reconstruction not found: {filtered_path}")

    # Load images
    img_no_filter = mpimg.imread(no_filter_path)
    img_filtered = mpimg.imread(filtered_path)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Left panel: No confidence filtering
    axes[0].imshow(img_no_filter)
    axes[0].axis("off")
    axes[0].set_title(
        "No Confidence Filtering: 33.03% Accuracy (Random Chance)\n"
        "All 2,280 spatial positions evaluated",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Right panel: Confidence filtering threshold 0.7
    axes[1].imshow(img_filtered)
    axes[1].axis("off")
    axes[1].set_title(
        "Confidence Filtering (≥0.7): 75.00% Accuracy\n"
        "Only 4/2,280 positions (0.2%) above threshold",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Overall title
    fig.suptitle(
        "Object Generalization: Accuracy-Coverage Tradeoff for GPU-MLP (Medium-HighReg)\n"
        "Trained on Objects A, B, C (WS1+WS2+WS3) — Validated on Object D (WS4)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save outputs
    output_dir = Path("comprehensive_3class_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "holdout_reconstruction_combined.png"
    pdf_path = output_dir / "holdout_reconstruction_combined.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"✅ Created combined holdout reconstruction figure:")
    print(f"   PNG: {png_path} ({png_path.stat().st_size / 1024:.1f} KB)")
    print(f"   PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    print(f"\n📊 Key Finding:")
    print(
        f"   - Without filtering: 33.03% accuracy (random chance) on all 2,280 positions"
    )
    print(
        f"   - With threshold 0.7: 75.00% accuracy on only 4 positions (0.2% coverage)"
    )
    print(
        f"   - Reveals fundamental accuracy-coverage tradeoff for object generalization"
    )

    plt.close()


if __name__ == "__main__":
    create_tradeoff_comparison()
