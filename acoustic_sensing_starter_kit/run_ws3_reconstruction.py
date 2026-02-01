#!/usr/bin/env python3
"""
Surface Reconstruction for Validation and Test Sets
Uses trained model from WS1+WS3 to reconstruct WS2 (validation)

Train: WS1 + WS3 (all surfaces)
Test: WS1 + WS3 (subset)
Validation: WS2 (unseen workspace)
"""
import sys
from pathlib import Path
from acoustic_sensing.experiments.surface_reconstruction_simple import (
    SurfaceReconstructor,
)

# Configuration - Use the actual results folder
EXPERIMENT_DIR = Path("rerun_reconstruction_poc_workspace2")
MODEL_PATH = (
    EXPERIMENT_DIR
    / "discriminationanalysis"
    / "trained_models"
    / "model_rank1_gpu-mlp_(medium-highreg).pkl"
)
OUTPUT_DIR = Path("final_reconstruction_results")

# Validation datasets (WS2 - unseen workspace)
VALIDATION_DATASETS = [
    ("data/balanced_workspace_2_squares_cutout", "VAL_WS2_squares_cutout"),
    ("data/balanced_workspace_2_pure_contact", "VAL_WS2_pure_contact"),
    ("data/balanced_workspace_2_pure_no_contact", "VAL_WS2_pure_no_contact"),
]

# Test datasets (WS3 - training surfaces, high accuracy expected)
TEST_DATASETS = [
    ("data/balanced_workspace_3_squares_cutout_v1", "TEST_WS3_squares_cutout_v1"),
    ("data/balanced_workspace_3_pure_contact", "TEST_WS3_pure_contact"),
    ("data/balanced_workspace_3_pure_no_contact", "TEST_WS3_pure_no_contact"),
]

# Hold-out dataset (completely unseen object + workspace)
HOLDOUT_DATASETS = [
    ("data/balanced_holdout_oversample", "HOLDOUT_WS4_oversample"),
]


def main():
    """Run surface reconstructions for validation and test datasets."""

    print("=" * 80)
    print("SURFACE RECONSTRUCTION PIPELINE")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        print()
        print("You need to run the training pipeline first:")
        print("  python3 run_modular_experiments.py configs/multi_dataset_config.yml")
        print()
        sys.exit(1)

    print(f"✅ Found trained model: {MODEL_PATH}")
    print()

    # Initialize reconstructor
    reconstructor = SurfaceReconstructor(
        base_output_dir=OUTPUT_DIR, confidence_threshold=0.8, logger=None
    )

    # Reconstruct VALIDATION set (WS2 - unseen workspace)
    print("=" * 80)
    print("PART 1: VALIDATION SET (WS2 - Unseen Workspace)")
    print("=" * 80)
    print("Expected accuracy: ~70% (generalization to new workspace)")
    print()

    val_success = 0
    for data_path, name in VALIDATION_DATASETS:
        print(f"🔍 Reconstructing: {name}")
        print(f"   Path: {data_path}")

        try:
            reconstructor.run_reconstruction(
                model_path=MODEL_PATH,
                data_path=Path(data_path),
                dataset_name=name,
                sweep_file="sweep.csv",
            )
            print(f"✅ {name} complete!")
            val_success += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        print()

    # Reconstruct TEST set (WS3 - training surfaces)
    print("=" * 80)
    print("PART 2: TEST SET (WS3 - Training Surfaces)")
    print("=" * 80)
    print("Expected accuracy: ~96% (high accuracy on known surfaces)")
    print()

    test_success = 0
    for data_path, name in TEST_DATASETS:
        print(f"🔍 Reconstructing: {name}")
        print(f"   Path: {data_path}")

        try:
            reconstructor.run_reconstruction(
                model_path=MODEL_PATH,
                data_path=Path(data_path),
                dataset_name=name,
                sweep_file="sweep.csv",
            )
            print(f"✅ {name} complete!")
            test_success += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        print()

    # Reconstruct HOLD-OUT set (completely unseen object + workspace)
    print("=" * 80)
    print("PART 3: HOLD-OUT SET (WS4 - Completely New Object + Workspace)")
    print("=" * 80)
    print("Expected accuracy: ~58% (entanglement problem - slight above random)")
    print()

    holdout_success = 0
    for data_path, name in HOLDOUT_DATASETS:
        print(f"🔍 Reconstructing: {name}")
        print(f"   Path: {data_path}")

        try:
            reconstructor.run_reconstruction(
                model_path=MODEL_PATH,
                data_path=Path(data_path),
                dataset_name=name,
                sweep_file="sweep.csv",
            )
            print(f"✅ {name} complete!")
            holdout_success += 1
        except Exception as e:
            print(f"❌ {name} failed: {e}")
        print()

    # Summary
    print("=" * 80)
    print("RECONSTRUCTION COMPLETE!")
    print("=" * 80)
    print(f"✅ Validation (WS2): {val_success}/{len(VALIDATION_DATASETS)} datasets")
    print(f"✅ Test (WS3): {test_success}/{len(TEST_DATASETS)} datasets")
    print(f"✅ Hold-Out (WS4): {holdout_success}/{len(HOLDOUT_DATASETS)} datasets")
    print()
    print(f"📂 Results saved to: {OUTPUT_DIR}/")
    print()
    print("📊 Key Images for Presentation:")
    print()
    print("  Slide 8 (TEST - 96% accuracy on training surfaces):")
    print(
        f"    {OUTPUT_DIR}/TEST_WS3_squares_cutout_v1/balanced_workspace_3_squares_cutout_v1_comparison.png"
    )
    print()
    print("  Slide 9 (VALIDATION - 70% accuracy on unseen workspace):")
    print(
        f"    {OUTPUT_DIR}/VAL_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png"
    )
    print()
    print("  Slide 10 (HOLD-OUT - 58% accuracy on new object):")
    print(
        f"    {OUTPUT_DIR}/HOLDOUT_WS4_oversample/balanced_holdout_oversample_comparison.png"
    )
    print()
    print("✨ No edge visualization - only contact (green) and no-contact (red)!")
    print()


if __name__ == "__main__":
    main()
