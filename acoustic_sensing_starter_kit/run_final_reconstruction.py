#!/usr/bin/env python3
"""
Final Surface Reconstruction for Presentation
Uses trained model from WS1+WS3 to reconstruct all surfaces

Train: WS1 + WS3 (all surfaces)
Test: WS3 (high accuracy expected ~96%)
Validation: WS2 (generalization to unseen workspace ~70%)
Hold-Out: WS4 (completely new object ~58%)
"""
import sys
from pathlib import Path
from acoustic_sensing.experiments.surface_reconstruction_simple import (
    SurfaceReconstructor,
)

# Configuration
EXPERIMENT_DIR = Path("rerun_reconstruction_poc_workspace2")
MODEL_PATH = (
    EXPERIMENT_DIR
    / "discriminationanalysis"
    / "trained_models"
    / "model_rank1_gpu-mlp_(medium-highreg).pkl"
)
OUTPUT_BASE_DIR = Path("final_reconstruction_results")

# Datasets to reconstruct
DATASETS = [
    # VALIDATION (WS2 - unseen workspace, ~70% expected)
    (
        "data/balanced_workspace_2_squares_cutout",
        "VAL_WS2_squares_cutout",
        "validation",
    ),
    ("data/balanced_workspace_2_pure_contact", "VAL_WS2_pure_contact", "validation"),
    (
        "data/balanced_workspace_2_pure_no_contact",
        "VAL_WS2_pure_no_contact",
        "validation",
    ),
    # TEST (WS3 - training surfaces, ~96% expected)
    (
        "data/balanced_workspace_3_squares_cutout_v1",
        "TEST_WS3_squares_cutout_v1",
        "test",
    ),
    ("data/balanced_workspace_3_pure_contact", "TEST_WS3_pure_contact", "test"),
    ("data/balanced_workspace_3_pure_no_contact", "TEST_WS3_pure_no_contact", "test"),
    # HOLD-OUT (WS4 - new object + workspace, ~58% expected)
    ("data/balanced_holdout_oversample", "HOLDOUT_WS4_oversample", "holdout"),
]


def main():
    """Run surface reconstructions for all datasets."""

    print("=" * 80)
    print("FINAL SURFACE RECONSTRUCTION FOR PRESENTATION")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_BASE_DIR}")
    print()

    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        print()
        sys.exit(1)

    print(f"✅ Found trained model")
    print()

    # Initialize reconstructor
    reconstructor = SurfaceReconstructor(
        model_path=str(MODEL_PATH),
        sr=48000,
        confidence_config={"enabled": True, "threshold": 0.8, "mode": "reject"},
        position_aggregation="highest_confidence",
        logger=None,
    )

    results = {}
    for data_path, name, category in DATASETS:
        output_dir = OUTPUT_BASE_DIR / name

        print(f"🔍 Reconstructing: {name} ({category})")
        print(f"   Data: {data_path}")
        print(f"   Output: {output_dir}")

        try:
            result = reconstructor.reconstruct_dataset(
                dataset_path=data_path,
                output_dir=str(output_dir),
                feature_extractor=None,  # Will create internally
            )

            accuracy = result.get("accuracy", 0.0) * 100
            print(f"   ✅ Accuracy: {accuracy:.1f}%")
            results[name] = {"accuracy": accuracy, "category": category}

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback

            traceback.print_exc()
            results[name] = {"accuracy": None, "category": category, "error": str(e)}

        print()

    # Summary
    print("=" * 80)
    print("RECONSTRUCTION COMPLETE!")
    print("=" * 80)
    print()

    # Group by category
    for category in ["test", "validation", "holdout"]:
        cat_results = {k: v for k, v in results.items() if v["category"] == category}
        if cat_results:
            print(f"{category.upper()}:")
            for name, res in cat_results.items():
                if res["accuracy"] is not None:
                    print(f"  ✅ {name}: {res['accuracy']:.1f}%")
                else:
                    print(f"  ❌ {name}: FAILED")
            print()

    print(f"📂 Results saved to: {OUTPUT_BASE_DIR}/")
    print()
    print("📊 Key Images for Presentation:")
    print()
    print("  Slide 8 (TEST - High accuracy on training surfaces):")
    print(
        f"    {OUTPUT_BASE_DIR}/TEST_WS3_squares_cutout_v1/balanced_workspace_3_squares_cutout_v1_comparison.png"
    )
    print()
    print("  Slide 9 (VALIDATION - Generalization to unseen workspace):")
    print(
        f"    {OUTPUT_BASE_DIR}/VAL_WS2_squares_cutout/balanced_workspace_2_squares_cutout_comparison.png"
    )
    print()
    print("  Slide 10 (HOLD-OUT - New object failure):")
    print(
        f"    {OUTPUT_BASE_DIR}/HOLDOUT_WS4_oversample/balanced_holdout_oversample_comparison.png"
    )
    print()
    print("✨ No edge visualization - only contact (green) and no-contact (red)!")
    print()


if __name__ == "__main__":
    main()
