#!/usr/bin/env python3
"""
Run surface reconstruction comparison between TEST (training workspaces) and HOLDOUT (new workspace).

This script demonstrates the generalization gap:
- TEST: Near-perfect reconstruction on data from training workspaces (WS1/WS2/WS3)
- HOLDOUT: Random-chance reconstruction on completely unseen workspace (WS4)
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.experiments.surface_reconstruction_simple import (
    SurfaceReconstructor,
)
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    base_dir = Path(__file__).parent

    # Paths
    model_path = (
        base_dir
        / "training_all_workspaces_holdout_val/discriminationanalysis/trained_models/model_rank1_random_forest.pkl"
    )
    output_dir = base_dir / "test_vs_holdout_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create feature extractor (same as training)
    feature_extractor = GeometricFeatureExtractor(
        use_workspace_invariant=True, use_impulse_features=True, sr=48000
    )

    # Datasets to compare
    # TEST: From training workspaces (should have high accuracy)
    # HOLDOUT: From completely new workspace (WS4)
    datasets = [
        # From WS1 (training) - squares cutout (undersample has sweep.csv)
        ("TEST_WS1_squares", "data/balanced_workspace_1_squares_cutout_oversample"),
        # From WS2 (training) - squares cutout (undersample)
        ("TEST_WS2_squares", "data/balanced_workspace_2_squares_cutout"),
        # From WS3 (training) - squares cutout (undersample)
        ("TEST_WS3_squares", "data/balanced_workspace_3_squares_cutout_v1"),
        # HOLDOUT (completely unseen WS4)
        ("HOLDOUT_WS4", "data/balanced_holdout_oversample"),
    ]

    print("\n" + "=" * 80)
    print("🔬 TEST vs HOLDOUT RECONSTRUCTION COMPARISON")
    print("=" * 80)
    print("\nModel trained on: WS1 + WS2 + WS3 (10 balanced datasets)")
    print("Testing on: Individual workspaces to show generalization gap\n")

    results_summary = {}

    for name, dataset_path in datasets:
        full_path = base_dir / dataset_path

        if not full_path.exists():
            print(f"⚠️  Skipping {name}: Dataset not found at {full_path}")
            continue

        print(f"\n{'='*60}")
        print(f"📊 {name}")
        print(f"   Path: {dataset_path}")
        print(f"{'='*60}")

        try:
            # Create reconstructor
            reconstructor = SurfaceReconstructor(
                model_path=str(model_path),
                sr=48000,
                confidence_config={"enabled": False},
                position_aggregation="highest_confidence",
                logger=logger,
            )

            # Run reconstruction
            result = reconstructor.reconstruct_dataset(
                str(full_path),
                str(output_dir / name),
                feature_extractor=feature_extractor,
            )

            accuracy = result.get("accuracy", 0)
            n_positions = result.get("n_positions", 0)
            mean_conf = result.get("mean_confidence", 0)

            results_summary[name] = {
                "accuracy": accuracy,
                "n_positions": n_positions,
                "mean_confidence": mean_conf,
            }

            is_holdout = "HOLDOUT" in name
            icon = "🔴" if is_holdout and accuracy < 0.6 else "🟢"

            print(f"\n   {icon} Accuracy: {accuracy:.1%}")
            print(f"      Positions: {n_positions}")
            print(f"      Mean Confidence: {mean_conf:.1%}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results_summary[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY: TEST vs HOLDOUT")
    print("=" * 80)

    test_accs = []
    holdout_accs = []

    for name, result in results_summary.items():
        if "error" not in result:
            acc = result["accuracy"]
            if "HOLDOUT" in name:
                holdout_accs.append(acc)
                print(f"  🔴 {name}: {acc:.1%} (UNSEEN WORKSPACE)")
            else:
                test_accs.append(acc)
                print(f"  🟢 {name}: {acc:.1%}")

    if test_accs and holdout_accs:
        avg_test = sum(test_accs) / len(test_accs)
        avg_holdout = sum(holdout_accs) / len(holdout_accs)
        gap = avg_test - avg_holdout

        print(f"\n  📈 Average TEST accuracy:    {avg_test:.1%}")
        print(f"  📉 Average HOLDOUT accuracy: {avg_holdout:.1%}")
        print(f"  ⚠️  Generalization Gap:       {gap:.1%}")

        print("\n  💡 INTERPRETATION:")
        if avg_holdout < 0.55:
            print("     The model does NOT generalize to unseen workspaces.")
            print("     ~50% accuracy = random guessing on binary classification.")
            print(
                "     Models learn workspace-specific patterns, not contact detection."
            )

    print(f"\n✅ Results saved to: {output_dir}")

    return results_summary


if __name__ == "__main__":
    main()
