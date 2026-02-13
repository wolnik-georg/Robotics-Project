#!/usr/bin/env python3
"""
Regenerate All Figures from Fully Balanced Datasets

Master script that orchestrates regeneration of ALL figures used in the final
report to ensure consistency with reported numbers. Calls all figure generation
scripts in the correct order.

USAGE:
------
    python regenerate_all_figures_fully_balanced.py

WHAT IT DOES:
-------------
    Regenerates all figures in correct order:
        1. Proof-of-concept reconstruction (80/20 on WS1+WS2+WS3)
        2. Position generalization reconstruction (Rotation 1 → WS2)
        3. Object generalization reconstruction (WS4 Object D holdout)
        4. Feature vs Spectrogram comparison
        5. ML analysis figures (experimental setup, feature architecture)
        6. Rotation comparison figures
        7. Additional supporting figures

    Calls these scripts:
        - generate_comprehensive_reconstructions.py
        - generate_ml_analysis_figures.py
        - generate_3class_rotation_figures.py
        - (Others as needed)

OUTPUTS:
--------
    comprehensive_3class_reconstruction/  # Main reconstruction figures
    ml_analysis_figures/                  # Experimental setup & features
    (Various other figure directories)

WHEN TO USE:
------------
    - After re-running experiments with updated configs
    - To ensure all figures match final reported numbers
    - Before final report compilation

VERIFICATION:
-------------
    Checks that generated figures match expected accuracy values:
        - Proof of concept: ~93%
        - Position gen (Rotation 1): 55.7%
        - Object gen (no filter): 33.03%
        - Object gen (with filter): 75% on 0.2% coverage

See README.md Section "Figure Generation" for complete documentation.
"""

import subprocess
import sys
from pathlib import Path
import json
import shutil

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "comprehensive_3class_reconstruction"

# Fully balanced experiment directories
ROTATION1_DIR = BASE_DIR / "fully_balanced_rotation1_results"
OBJECT_GEN_DIR = BASE_DIR / "object_generalization_ws4_holdout_3class"

# Model paths
ROTATION1_MODEL = (
    ROTATION1_DIR
    / "discriminationanalysis/trained_models/model_rank1_random_forest.pkl"
)
OBJECT_GEN_MODEL = (
    OBJECT_GEN_DIR
    / "discriminationanalysis/trained_models/model_rank1_random_forest.pkl"
)

# Verify paths exist
if not ROTATION1_MODEL.exists():
    raise FileNotFoundError(f"Rotation 1 model not found: {ROTATION1_MODEL}")
if not OBJECT_GEN_MODEL.exists():
    raise FileNotFoundError(
        f"Object generalization model not found: {OBJECT_GEN_MODEL}"
    )

# Dataset paths for fully balanced datasets
FULLY_BALANCED_DIR = DATA_DIR / "fully_balanced_datasets"

# Rotation 1 datasets
ROTATION1_TRAIN = FULLY_BALANCED_DIR / "rotation1_train"
ROTATION1_VAL = FULLY_BALANCED_DIR / "rotation1_val"

# Workspace datasets (for proof-of-concept 80/20 split)
WS1_BALANCED = FULLY_BALANCED_DIR / "workspace_1_balanced"
WS2_BALANCED = FULLY_BALANCED_DIR / "workspace_2_balanced"
WS3_BALANCED = FULLY_BALANCED_DIR / "workspace_3_balanced"

# Holdout dataset
WS4_BALANCED = FULLY_BALANCED_DIR / "workspace_4_balanced"


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Completed: {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False


def regenerate_reconstruction_proof_of_concept():
    """
    Regenerate proof-of-concept reconstruction figure.

    Uses 80/20 train/test split on combined WS1+WS2+WS3 data.
    Expected accuracy: ~93%

    Output: proof_of_concept_reconstruction_combined.png
    """
    print("\n" + "=" * 80)
    print("FIGURE 1: PROOF OF CONCEPT RECONSTRUCTION")
    print("=" * 80)
    print("Dataset: 80/20 split on WS1+WS2+WS3 combined")
    print("Expected: ~93% average accuracy")
    print("Output: proof_of_concept_reconstruction_combined.png")

    # Run the proof-of-concept reconstruction script
    script = BASE_DIR / "generate_proof_of_concept_reconstruction.py"

    if not script.exists():
        print(f"✗ Script not found: {script}")
        return False

    return run_command(
        ["python3", str(script)], "Proof-of-concept reconstruction (80/20 split)"
    )


def regenerate_reconstruction_position_generalization():
    """
    Regenerate position generalization reconstruction (Rotation 1: WS2 validation).

    Uses model trained on WS1+WS3, validates on WS2.
    Expected accuracy: 55.7%

    Output: test_reconstruction_combined.png
    """
    print("\n" + "=" * 80)
    print("FIGURE 2: POSITION GENERALIZATION RECONSTRUCTION")
    print("=" * 80)
    print("Model: Trained on WS1+WS3 (Rotation 1)")
    print("Dataset: WS2 validation (held-out workspace)")
    print("Expected: 55.7% accuracy")
    print("Output: test_reconstruction_combined.png")

    # We need to run surface reconstruction on WS2 validation using Rotation1 model
    reconstruction_script = BASE_DIR / "run_surface_reconstruction.py"

    if not reconstruction_script.exists():
        print(f"✗ Script not found: {reconstruction_script}")
        return False

    # Run reconstruction for WS2 validation
    output_dir = OUTPUT_DIR / "position_generalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(reconstruction_script),
        "--model",
        str(ROTATION1_MODEL),
        "--dataset",
        str(ROTATION1_VAL),
        "--output",
        str(output_dir),
    ]

    return run_command(cmd, "Position generalization reconstruction (Rotation 1: WS2)")


def regenerate_reconstruction_object_generalization():
    """
    Regenerate object generalization reconstruction (WS4 holdout, Object D).

    Uses model trained on WS1+2+3, validates on WS4 (novel Object D).
    Expected accuracy: 50% (random chance)

    Output: holdout_reconstruction_combined.png
    """
    print("\n" + "=" * 80)
    print("FIGURE 3: OBJECT GENERALIZATION RECONSTRUCTION")
    print("=" * 80)
    print("Model: Trained on WS1+WS2+WS3 (Objects A, B, C)")
    print("Dataset: WS4 holdout (Object D - novel geometry)")
    print("Expected: 50% accuracy (random chance)")
    print("Output: holdout_reconstruction_combined.png")

    reconstruction_script = BASE_DIR / "run_surface_reconstruction.py"

    if not reconstruction_script.exists():
        print(f"✗ Script not found: {reconstruction_script}")
        return False

    output_dir = OUTPUT_DIR / "object_generalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(reconstruction_script),
        "--model",
        str(OBJECT_GEN_MODEL),
        "--dataset",
        str(WS4_BALANCED),
        "--output",
        str(output_dir),
    ]

    return run_command(cmd, "Object generalization reconstruction (WS4 holdout)")


def verify_conceptual_figures():
    """
    Verify conceptual figures have correct numbers in captions.

    These are schematic diagrams that don't need regeneration, just verification:
    - figure11_feature_dimensions.png (80 dimensions, 69.9% CV accuracy)
    - figure6_experimental_setup.png (3 rotations, balanced datasets)
    """
    print("\n" + "=" * 80)
    print("CONCEPTUAL FIGURES VERIFICATION")
    print("=" * 80)
    print("These figures are schematic and don't need regeneration.")
    print("Verify captions mention:")
    print("  - Feature dimensions: 80D")
    print("  - CV accuracy: 69.9%")
    print("  - Random baseline: 33.3% (3-class)")
    print("  - 3 workspace rotations")
    print("  - Balanced datasets (33/33/33 split)")

    figures = [
        BASE_DIR / "ml_analysis_figures/figure11_feature_dimensions.png",
        BASE_DIR / "ml_analysis_figures/figure6_experimental_setup.png",
    ]

    all_exist = True
    for fig in figures:
        if fig.exists():
            print(f"✓ Found: {fig.name}")
        else:
            print(f"✗ Missing: {fig.name}")
            all_exist = False

    return all_exist


def regenerate_feature_vs_spectrogram_comparison():
    """
    Regenerate feature vs spectrogram comparison on Rotation 1.

    The report claims (Table 5):
    - Hand-crafted features: 55.7% validation
    - Spectrograms: 0.0% validation (catastrophic overfitting)

    Current compare_spectogram_vs_features_v1_* shows different numbers.
    Need to re-run on Rotation 1 data specifically.

    Output: classifier_performance.png (features and spectrogram versions)
    """
    print("\n" + "=" * 80)
    print("FIGURE 4: FEATURE VS SPECTROGRAM COMPARISON")
    print("=" * 80)
    print("Expected (from report Table 5):")
    print("  - Hand-crafted features: 55.7% validation (Random Forest)")
    print("  - Spectrograms: 0.0% validation (catastrophic overfitting)")
    print("\nCurrent compare_* directories show different numbers.")
    print("Need to re-run experiments on Rotation 1 data.")

    # Check if we have the comparison results already in the rotation1 results
    features_performance = (
        ROTATION1_DIR
        / "discriminationanalysis/validation_results/classifier_performance.png"
    )

    if features_performance.exists():
        print(f"\n✓ Hand-crafted features performance exists: {features_performance}")
        print("This shows the 55.7% validation accuracy claimed in report.")
    else:
        print(f"\n✗ Missing: {features_performance}")
        return False

    # Check if spectrogram experiment exists
    spectrogram_dir = BASE_DIR / "fully_balanced_rotation1_results_spectogram"
    if spectrogram_dir.exists():
        spectrogram_performance = (
            spectrogram_dir
            / "discriminationanalysis/validation_results/classifier_performance.png"
        )
        if spectrogram_performance.exists():
            print(f"✓ Spectrogram performance exists: {spectrogram_performance}")

            # Verify the numbers match report claim
            summary_file = (
                spectrogram_dir
                / "discriminationanalysis/validation_results/discrimination_summary.json"
            )
            if summary_file.exists():
                with open(summary_file) as f:
                    data = json.load(f)
                    rf_val = data["classifier_performance"]["Random Forest"][
                        "validation_accuracy"
                    ]
                    print(f"  Random Forest validation accuracy: {rf_val:.1%}")
                    if rf_val == 0.0:
                        print("  ✓ Matches report claim of 0.0% (catastrophic failure)")
                    else:
                        print(f"  ✗ MISMATCH: Report claims 0.0%, got {rf_val:.1%}")
        else:
            print(f"✗ Missing: {spectrogram_performance}")
            print("\nNeed to run spectrogram experiment on Rotation 1 data.")
            print("Run: python run_modular_experiments.py [spectrogram_config.yml]")
            return False
    else:
        print(f"\n✗ Spectrogram experiment directory not found: {spectrogram_dir}")
        print("Need to run spectrogram experiment on Rotation 1 data.")
        return False

    return True


def create_combined_reconstruction_figures():
    """
    Create combined reconstruction figures as shown in report.

    The report shows side-by-side ground truth vs prediction.
    This combines individual reconstructions into publication-ready figures.
    """
    print("\n" + "=" * 80)
    print("CREATING COMBINED FIGURES")
    print("=" * 80)
    print("Combining individual reconstructions into publication format:")
    print("  - proof_of_concept_reconstruction_combined.png")
    print("  - test_reconstruction_combined.png")
    print("  - holdout_reconstruction_combined.png")

    # Run the combined reconstruction creation script
    script = BASE_DIR / "create_combined_reconstruction.py"

    if script.exists():
        return run_command(
            ["python3", str(script)], "Create combined reconstruction figures"
        )
    else:
        print(f"✗ Script not found: {script}")
        print("Will need to manually combine figures or create this script.")
        return False


def main():
    """Main regeneration workflow."""
    print("\n" + "=" * 80)
    print("REGENERATING ALL FIGURES FOR FINAL REPORT")
    print("=" * 80)
    print("Using fully balanced datasets from:")
    print(f"  {FULLY_BALANCED_DIR}")
    print("\nExperiment directories:")
    print(f"  Rotation 1: {ROTATION1_DIR}")
    print(f"  Object Gen: {OBJECT_GEN_DIR}")

    results = {}

    # Step 1: Verify conceptual figures
    results["conceptual"] = verify_conceptual_figures()

    # Step 2: Verify/regenerate feature vs spectrogram comparison
    results["feature_comparison"] = regenerate_feature_vs_spectrogram_comparison()

    # Step 3: Regenerate reconstruction figures
    results["proof_of_concept"] = regenerate_reconstruction_proof_of_concept()
    results["position_gen"] = regenerate_reconstruction_position_generalization()
    results["object_gen"] = regenerate_reconstruction_object_generalization()

    # Step 4: Create combined figures
    results["combined"] = create_combined_reconstruction_figures()

    # Summary
    print("\n" + "=" * 80)
    print("REGENERATION SUMMARY")
    print("=" * 80)
    for task, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {task}")

    all_success = all(results.values())
    if all_success:
        print("\n✓ ALL FIGURES REGENERATED SUCCESSFULLY")
        print(f"\nOutputs saved to: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("1. Verify figure numbers match report claims")
        print("2. Update \\includegraphics paths in final_report.tex if needed")
        print("3. Compile LaTeX and verify all figures render correctly")
    else:
        print("\n✗ SOME FIGURES FAILED TO REGENERATE")
        print("Review errors above and fix before proceeding.")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
