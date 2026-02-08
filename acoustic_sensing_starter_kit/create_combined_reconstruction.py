#!/usr/bin/env python3
"""
Create Combined 3-Object Visual Comparison for 3-Class Reconstruction

Combines the individual reconstruction visualizations for all 3 WS2 objects
(squares_cutout, pure_no_contact, pure_contact) into a single comparison figure
showing ground truth and predictions side-by-side.

This replicates the pattern_a_visual_comparison.png format but with 3-class labels.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# Input paths
RECON_DIR = Path("pattern_a_3class_reconstruction")
DATASETS = {
    "squares_cutout": "Object A: Squares Cutout",
    "pure_no_contact": "Object B: Empty Workspace",
    "pure_contact": "Object C: Full Contact Surface",
}

# Output path
OUTPUT_FILE = RECON_DIR / "pattern_a_visual_comparison.png"


def load_comparison_image(dataset_name: str) -> np.ndarray:
    """Load the comparison image for a dataset."""
    image_path = (
        RECON_DIR
        / dataset_name
        / f"balanced_workspace_2_3class_{dataset_name}"
        / "03_comparison.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(f"Comparison image not found: {image_path}")

    return mpimg.imread(image_path)


def create_combined_figure():
    """Create the combined 3-object comparison figure."""
    print("Creating combined 3-class reconstruction figure...")
    print("=" * 80)

    # Load all comparison images
    images = {}
    for dataset_name, label in DATASETS.items():
        print(f"Loading: {label}...")
        try:
            images[dataset_name] = load_comparison_image(dataset_name)
            print(f"  ✓ Loaded {dataset_name}")
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            return False

    # Create figure with 3 rows (one per object)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        "3-Class Acoustic Reconstruction: Workspace 2 Validation (Rotation 1)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Plot each object's comparison
    for idx, (dataset_name, label) in enumerate(DATASETS.items()):
        ax = axes[idx]
        ax.imshow(images[dataset_name])
        ax.axis("off")
        ax.set_title(label, fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_FILE.with_suffix(".pdf"), bbox_inches="tight")

    print()
    print("=" * 80)
    print(f"✓ Created combined figure:")
    print(f"  {OUTPUT_FILE}")
    print(f"  {OUTPUT_FILE.with_suffix('.pdf')}")
    print("=" * 80)

    # Also copy to ml_analysis_figures
    ml_fig_dir = Path("ml_analysis_figures")
    ml_fig_dir.mkdir(exist_ok=True)

    import shutil

    shutil.copy(OUTPUT_FILE, ml_fig_dir / "pattern_a_3class_reconstruction.png")
    print(f"✓ Copied to: {ml_fig_dir / 'pattern_a_3class_reconstruction.png'}")

    return True


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("3-CLASS RECONSTRUCTION: COMBINED FIGURE GENERATION")
    print("=" * 80 + "\n")

    if create_combined_figure():
        print("\n✅ SUCCESS: Combined 3-class reconstruction figure ready!")
        print("\nNext step:")
        print("  Update final_report.tex Line 224 to:")
        print(
            "  \\includegraphics[width=\\textwidth]{../pattern_a_3class_reconstruction/pattern_a_visual_comparison.png}"
        )
        print()
    else:
        print("\n❌ ERROR: Failed to create combined figure")
        print("Check that all individual reconstructions completed successfully.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
