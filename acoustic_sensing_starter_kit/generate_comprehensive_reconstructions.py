#!/usr/bin/env python3
"""
Generate All 3 Main Reconstruction Figures for the Paper

Creates the three key 2D spatial reconstruction figures used in the final report:
    1. Proof of Concept (80/20 train/test on combined WS1+WS2+WS3)
    2. Position Generalization (Rotation 1 validation on WS2)
    3. Object Generalization (WS4 Object D holdout, with/without confidence filtering)

USAGE:
------
    python generate_comprehensive_reconstructions.py

WHAT IT DOES:
-------------
    Calls specialized reconstruction scripts for each figure:
        - generate_proof_of_concept_reconstruction.py → Figure: Proof of Concept
        - run_final_reconstruction.py → Figure: Position Generalization (WS2)
        - run_holdout_reconstruction.py → Figure: Object Generalization (WS4)

    Each script loads trained models, applies them to spatial grid positions,
    and generates ground truth vs prediction visualizations.

OUTPUTS:
--------
    comprehensive_3class_reconstruction/
        proof_of_concept_reconstruction_combined.png  # ~93% accuracy
        test_reconstruction_combined.png              # 34.89% accuracy (failure)
        holdout_reconstruction_combined.png           # Without/with confidence filtering

DEMONSTRATES:
-------------
    - Within-workspace success (~93% on proof of concept)
    - Position generalization failure (34.89% on WS2)
    - Object generalization failure (33% → 75% with confidence filtering tradeoff)

See README.md Section "Figure Generation" for complete documentation.
"""

import subprocess
from pathlib import Path
import json

# Base directories
# Script is in acoustic_sensing_starter_kit/, experiments are in the same directory
BASE_DIR = Path(__file__).resolve().parent
ROTATION1_DIR = BASE_DIR / "test_pipeline_3class_v1"
OBJECT_GEN_DIR = BASE_DIR / "object_generalization_ws4_holdout_3class"

# Verify paths exist
if not ROTATION1_DIR.exists():
    raise FileNotFoundError(f"Rotation 1 directory not found: {ROTATION1_DIR}")
if not OBJECT_GEN_DIR.exists():
    raise FileNotFoundError(
        f"Object generalization directory not found: {OBJECT_GEN_DIR}"
    )

# Model paths
ROTATION1_MODEL = (
    ROTATION1_DIR
    / "discriminationanalysis/trained_models/model_rank1_random_forest.pkl"
)
OBJECT_GEN_MODEL = (
    OBJECT_GEN_DIR
    / "discriminationanalysis/trained_models/model_rank1_random_forest.pkl"
)

# Output directory - save in acoustic_sensing_starter_kit for easy access
OUTPUT_DIR = BASE_DIR / "comprehensive_3class_reconstruction"

# Test datasets from Rotation 1 (WS1 + WS3 - training data for proof of concept)
TEST_DATASETS = [
    "balanced_workspace_1_3class_squares_cutout",  # Object A - WS1
    "balanced_workspace_1_3class_pure_no_contact",  # Object B - WS1
    "balanced_workspace_1_3class_pure_contact",  # Object C - WS1
    "balanced_workspace_3_3class_squares_cutout_v1",  # Object A - WS3
    "balanced_workspace_3_3class_pure_no_contact",  # Object B - WS3
    "balanced_workspace_3_3class_pure_contact",  # Object C - WS3
]

# Validation datasets from Rotation 1 (WS2 - same distribution)
VALIDATION_DATASETS = [
    "balanced_workspace_2_3class_squares_cutout",  # Object A
    "balanced_workspace_2_3class_pure_no_contact",  # Object B
    "balanced_workspace_2_3class_pure_contact",  # Object C
]

# Holdout dataset (WS4 - novel object D)
HOLDOUT_DATASET = "balanced_holdout_3class"


def run_surface_reconstruction(model_path, dataset_name, output_subdir, description):
    """Run surface reconstruction for a single dataset."""

    output_path = OUTPUT_DIR / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    # run_surface_reconstruction.py is in the same directory as this script
    reconstruction_script = (
        Path(__file__).resolve().parent / "run_surface_reconstruction.py"
    )

    cmd = [
        "python3",
        str(reconstruction_script),
        "--model",
        str(model_path),
        "--dataset",
        dataset_name,
        "--output",
        str(output_path),
    ]

    print(f"\n{'='*80}")
    print(f"Running reconstruction: {description}")
    print(f"Model: {model_path.name}")
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Generate all reconstructions."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE 3-CLASS SURFACE RECONSTRUCTION")
    print("=" * 80)

    # Track results
    results = {"test": {}, "validation": {}, "holdout": {}}

    # Part 1: Test datasets (training data - should work excellently)
    print("\n" + "=" * 80)
    print("PART 1: TEST DATA (Training Distribution - Proof of Concept)")
    print("Training: WS1+WS3 | Test on: WS1+WS3 | Objects: A, B, C")
    print("Expected: Very high accuracy (~95%+)")
    print("=" * 80)

    test_object_names = {
        "balanced_workspace_1_3class_squares_cutout": "WS1 - Object A (Cutout)",
        "balanced_workspace_1_3class_pure_no_contact": "WS1 - Object B (Empty)",
        "balanced_workspace_1_3class_pure_contact": "WS1 - Object C (Full Contact)",
        "balanced_workspace_3_3class_squares_cutout_v1": "WS3 - Object A (Cutout)",
        "balanced_workspace_3_3class_pure_no_contact": "WS3 - Object B (Empty)",
        "balanced_workspace_3_3class_pure_contact": "WS3 - Object C (Full Contact)",
    }

    for dataset in TEST_DATASETS:
        obj_name = test_object_names[dataset]
        success = run_surface_reconstruction(
            ROTATION1_MODEL,
            dataset,
            f"test/{dataset}",
            f"Test - {obj_name}",
        )
        results["test"][obj_name] = success

    # Part 2: Validation datasets (same distribution - should work well)
    print("\n" + "=" * 80)
    print("PART 2: VALIDATION DATA (Same Distribution - Position Generalization)")
    print("Training: WS1+WS3 | Validation: WS2 | Objects: A, B, C")
    print("Expected: High accuracy (~85%)")
    print("=" * 80)

    validation_object_names = {
        "balanced_workspace_2_3class_squares_cutout": "WS2 - Object A (Cutout)",
        "balanced_workspace_2_3class_pure_no_contact": "WS2 - Object B (Empty)",
        "balanced_workspace_2_3class_pure_contact": "WS2 - Object C (Full Contact)",
    }

    for dataset in VALIDATION_DATASETS:
        obj_name = validation_object_names[dataset]
        success = run_surface_reconstruction(
            ROTATION1_MODEL,
            dataset,
            f"validation/{dataset}",
            f"Validation - {obj_name}",
        )
        results["validation"][obj_name] = success

    # Part 3: Holdout dataset (novel object - should fail)
    print("\n" + "=" * 80)
    print("PART 3: HOLDOUT DATA (Novel Object - Object Generalization)")
    print("Training: WS1+WS2+WS3 Objects A,B,C | Validation: WS4 Object D")
    print("Expected: Random chance (~33% for 3-class)")
    print("=" * 80)

    success = run_surface_reconstruction(
        OBJECT_GEN_MODEL,
        HOLDOUT_DATASET,
        f"holdout/{HOLDOUT_DATASET}",
        "Holdout - WS4 Object D (Novel Geometry)",
    )
    results["holdout"]["WS4 - Object D (Novel)"] = success

    # Summary
    print("\n" + "=" * 80)
    print("RECONSTRUCTION SUMMARY")
    print("=" * 80)

    print("\nTest Data (Training Distribution - Proof of Concept):")
    for obj_name, success in results["test"].items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {obj_name}")

    print("\nValidation Data (Same Distribution - Position Generalization):")
    for obj_name, success in results["validation"].items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {obj_name}")

    print("\nHoldout Data (Novel Object - Object Generalization):")
    for obj_name, success in results["holdout"].items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {obj_name}")

    print(f"\nAll reconstructions saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Create combined test figure (proof of concept - best performance)")
    print("2. Create combined validation figure (position generalization)")
    print("3. Create holdout figure (object generalization failure)")
    print("4. Update final_report.tex with all three figures")

    # Save results
    results_file = OUTPUT_DIR / "reconstruction_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
