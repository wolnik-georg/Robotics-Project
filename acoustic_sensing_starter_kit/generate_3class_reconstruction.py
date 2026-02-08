#!/usr/bin/env python3
"""
Generate 3-Class Reconstruction Visualization for Rotation 1

Runs surface reconstruction using the trained Random Forest model from
test_pipeline_3class_v1 (Rotation 1: WS1+3 train, WS2 validate) and
creates the 3-class geometric reconstruction figure showing contact,
no-contact, and edge predictions.

This replaces the old binary classification reconstruction with the
full 3-class version including edge detection.

Output: pattern_a_3class_reconstruction/pattern_a_visual_comparison.png
"""

import subprocess
import sys
from pathlib import Path
import shutil

# Paths
ROTATION1_DIR = Path("test_pipeline_3class_v1")
MODEL_PATH = (
    ROTATION1_DIR
    / "discriminationanalysis/trained_models/model_rank1_random_forest.pkl"
)
OUTPUT_DIR = Path("pattern_a_3class_reconstruction")

# Use the balanced datasets directly (they have sweep.csv with spatial coordinates)
# Don't include "data/" prefix since run_surface_reconstruction.py adds it
WS2_DATASETS = [
    "balanced_workspace_2_3class_squares_cutout",
    "balanced_workspace_2_3class_pure_no_contact",
    "balanced_workspace_2_3class_pure_contact",
]


def check_model_exists():
    """Verify the trained model exists."""
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("\nAvailable models:")
        model_dir = ROTATION1_DIR / "discriminationanalysis/trained_models"
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                print(f"  - {model_file.name}")
        sys.exit(1)
    print(f"✓ Found model: {MODEL_PATH}")


def run_reconstruction_for_dataset(dataset_name: str):
    """Run reconstruction for a single balanced dataset (already has sweep.csv)."""
    print(f"\n{'='*80}")
    print(f"Running reconstruction: {dataset_name}")
    print(f"{'='*80}\n")

    cmd = [
        "python3",
        "run_surface_reconstruction.py",
        "--model",
        str(MODEL_PATH),
        "--dataset",
        dataset_name,  # Already includes path relative to cwd
        "--output",
        str(OUTPUT_DIR / dataset_name.replace("balanced_workspace_2_3class_", "")),
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Reconstruction failed for {dataset_name}")
        return False

    print(f"✓ Completed reconstruction for {dataset_name}")
    return True


def create_combined_visualization():
    """Create the combined 3-object visual comparison figure."""
    print(f"\n{'='*80}")
    print("Creating combined 3-class reconstruction figure...")
    print(f"{'='*80}\n")

    # This will be created by the reconstruction script
    # We just need to check if it exists

    output_image = OUTPUT_DIR / "pattern_a_visual_comparison.png"

    if output_image.exists():
        print(f"✓ Created combined figure: {output_image}")

        # Copy to ml_analysis_figures for easy reference
        ml_fig_dir = Path("ml_analysis_figures")
        ml_fig_dir.mkdir(exist_ok=True)
        shutil.copy(output_image, ml_fig_dir / "pattern_a_3class_reconstruction.png")
        print(f"✓ Copied to: {ml_fig_dir / 'pattern_a_3class_reconstruction.png'}")
    else:
        print("WARNING: Combined figure not found. You may need to create it manually.")


def main():
    """Main execution."""
    print("=" * 80)
    print("3-CLASS RECONSTRUCTION GENERATION (ROTATION 1: WS2 VALIDATION)")
    print("=" * 80)

    # Check model
    check_model_exists()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")

    # Run reconstruction for each WS2 dataset
    print(f"\nProcessing {len(WS2_DATASETS)} validation datasets...")

    successes = 0
    for dataset in WS2_DATASETS:
        if run_reconstruction_for_dataset(dataset):
            successes += 1

    print(f"\n{'='*80}")
    print(f"RECONSTRUCTION SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {successes}/{len(WS2_DATASETS)} datasets")

    # Create combined visualization
    create_combined_visualization()

    print(f"\n{'='*80}")
    print("✓ 3-CLASS RECONSTRUCTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nNext step: Update final_report.tex to use:")
    print(
        f"  \\includegraphics{{../pattern_a_3class_reconstruction/pattern_a_visual_comparison.png}}"
    )
    print()


if __name__ == "__main__":
    main()
