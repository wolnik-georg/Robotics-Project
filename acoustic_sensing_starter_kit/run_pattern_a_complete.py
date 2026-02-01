#!/usr/bin/env python3
"""
Pattern A Complete Reconstruction with Edges
=============================================

Model: Trained on WS2+WS3 (all surfaces, no edges)
- TEST: Reconstruct all 3 surfaces from WS2 and WS3
- VALIDATION: Reconstruct all 3 surfaces from WS1 (unseen workspace)

All visualizations include edge positions (from original data) for completeness,
but edges were NOT used in training/testing/validation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.experiments.surface_reconstruction_simple import (
    SurfaceReconstructor,
)
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_edge_positions(original_dataset_path: Path) -> tuple:
    """
    Load edge positions from the original (unbalanced) dataset.

    Returns:
        Tuple of (edge_x, edge_y) arrays, or (None, None) if no edges found
    """
    sweep_path = original_dataset_path / "sweep.csv"
    if not sweep_path.exists():
        return None, None

    df = pd.read_csv(sweep_path)

    # Check for edge labels
    if "original_label" in df.columns:
        edge_mask = df["original_label"] == "edge"
    elif "label" in df.columns:
        edge_mask = df["label"] == "edge"
    else:
        return None, None

    if edge_mask.sum() == 0:
        return None, None

    edge_df = df[edge_mask]

    # Get positions
    x = edge_df.get("normalized_x", edge_df.get("x", None))
    y = edge_df.get("normalized_y", edge_df.get("y", None))

    if x is None or y is None:
        return None, None

    # Aggregate to unique positions
    positions = np.column_stack([x.values, y.values])
    unique_positions = np.unique(positions, axis=0)

    return unique_positions[:, 0], unique_positions[:, 1]


def create_reconstruction_with_edges(
    reconstructor: SurfaceReconstructor,
    dataset_path: str,
    original_dataset_path: str,
    output_dir: Path,
    dataset_name: str,
    feature_extractor,
) -> dict:
    """
    Run reconstruction and create visualizations with edge overlay.
    """
    dataset_path = Path(dataset_path)
    original_dataset_path = Path(original_dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get edge positions from original dataset
    edge_x, edge_y = get_edge_positions(original_dataset_path)
    has_edges = edge_x is not None and len(edge_x) > 0

    if has_edges:
        logger.info(f"  Found {len(edge_x)} edge positions for overlay")
    else:
        logger.info(f"  No edge positions found for overlay")

    # Run reconstruction
    result = reconstructor.reconstruct_dataset(
        str(dataset_path), str(output_dir), feature_extractor=feature_extractor
    )

    accuracy = result.get("accuracy", 0)
    n_positions = result.get("n_positions", 0)

    # If we have edges, create enhanced visualizations
    if has_edges:
        # Load the sweep.csv to get positions and labels
        sweep_df = pd.read_csv(dataset_path / "sweep.csv")
        label_col = (
            "relabeled_label"
            if "relabeled_label" in sweep_df.columns
            else "original_label"
        )

        # Create enhanced comparison plot with edges
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        colors = {"contact": "green", "no_contact": "red"}

        # Get unique positions from balanced data
        x = sweep_df["normalized_x"].values
        y = sweep_df["normalized_y"].values
        labels = sweep_df[label_col].values

        # Ground truth
        for label in ["contact", "no_contact"]:
            mask = labels == label
            axes[0].scatter(
                x[mask], y[mask], c=colors[label], label=label, alpha=0.7, s=50
            )

        # Add edges
        axes[0].scatter(
            edge_x,
            edge_y,
            c="orange",
            label="edge",
            alpha=0.8,
            s=60,
            marker="s",
            edgecolors="black",
        )

        axes[0].set_xlabel("Normalized X")
        axes[0].set_ylabel("Normalized Y")
        axes[0].set_title(f"Ground Truth with Edges ({n_positions} positions)")
        axes[0].legend()
        axes[0].set_xlim(-0.05, 1.05)
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_aspect("equal")

        # Just show the edge overlay on the ground truth for now
        axes[1].scatter(
            edge_x,
            edge_y,
            c="orange",
            label="edge (not predicted)",
            alpha=0.8,
            s=60,
            marker="s",
            edgecolors="black",
        )

        # Add a note about predictions
        axes[1].text(
            0.5,
            0.5,
            f"Accuracy: {accuracy:.1%}\n(See comparison.png\nfor predictions)",
            ha="center",
            va="center",
            fontsize=14,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        axes[1].set_xlabel("Normalized X")
        axes[1].set_ylabel("Normalized Y")
        axes[1].set_title(f"Edge Positions (shown for completeness)")
        axes[1].legend()
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_aspect("equal")

        plt.suptitle(
            f"Surface with Edges: {dataset_name}\n(Edges NOT used in training/testing)",
            fontsize=14,
        )
        plt.tight_layout()

        edge_plot_path = output_dir / f"{dataset_name}_with_edges.png"
        plt.savefig(edge_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        result["edge_plot"] = str(edge_plot_path)
        result["n_edge_positions"] = len(edge_x)

    return result


def main():
    base_dir = Path(__file__).parent

    # Model from Pattern A training (WS2+WS3 train → WS1 validation)
    model_path = (
        base_dir
        / "training_for_reconstruction_v1/discriminationanalysis/trained_models/model_rank1_random_forest.pkl"
    )

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Output directory
    output_dir = base_dir / "pattern_a_complete_reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature extractor
    feature_extractor = GeometricFeatureExtractor(
        use_workspace_invariant=True, use_impulse_features=True, sr=48000
    )

    print("\n" + "=" * 80)
    print("🔬 PATTERN A: Complete Reconstruction with Edges")
    print("=" * 80)
    print("Model: WS2+WS3 trained → WS1 validation")
    print("Edges shown in visualizations but NOT used in training/testing")
    print("=" * 80)

    # Define all datasets for Pattern A
    # Format: (name, balanced_dataset, original_dataset_for_edges, is_validation)
    datasets = [
        # TEST: WS2 surfaces (used in training)
        (
            "WS2_squares_cutout",
            "data/balanced_workspace_2_squares_cutout",
            "data/collected_data_runs_2026_01_15_workspace_2_squares_cutout_relabeled",
            False,
        ),
        (
            "WS2_pure_contact",
            "data/balanced_workspace_2_pure_contact",
            "data/collected_data_runs_2026_01_15_workspace_2_pure_contact_relabeled",
            False,
        ),
        (
            "WS2_pure_no_contact",
            "data/balanced_workspace_2_pure_no_contact",
            "data/collected_data_runs_2026_01_15_workspace_2_pure_no_contact",
            False,
        ),
        # TEST: WS3 surfaces (used in training)
        (
            "WS3_squares_cutout",
            "data/balanced_workspace_3_squares_cutout_v1",
            "data/collected_data_runs_2025_12_15_v2_2_workspace3_squares_cutout_relabeled",
            False,
        ),
        (
            "WS3_pure_contact",
            "data/balanced_workspace_3_pure_contact",
            "data/collected_data_runs_2026_01_14_workspace_3_pure_contact_relabeled",
            False,
        ),
        (
            "WS3_pure_no_contact",
            "data/balanced_workspace_3_pure_no_contact",
            "data/collected_data_runs_2026_01_14_workspace_3_pure_no_contact",
            False,
        ),
        # VALIDATION: WS1 surfaces (unseen workspace, same object)
        (
            "VAL_WS1_squares_cutout",
            "data/balanced_workspace_1_squares_cutout_oversample",
            "data/collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled",
            True,
        ),
        (
            "VAL_WS1_pure_contact",
            "data/balanced_workspace_1_pure_contact_oversample",
            "data/collected_data_runs_2026_01_15_workspace_1_pure_contact_relabeled",
            True,
        ),
        (
            "VAL_WS1_pure_no_contact",
            "data/balanced_workspace_1_pure_no_contact_oversample",
            "data/collected_data_runs_2026_01_15_workspace_1_pure_no_contact",
            True,
        ),
    ]

    results_summary = {}

    for name, balanced_path, original_path, is_validation in datasets:
        balanced_full = base_dir / balanced_path
        original_full = base_dir / original_path

        if not balanced_full.exists():
            print(f"\n⚠️  Skipping {name}: Dataset not found at {balanced_full}")
            continue

        print(f"\n{'='*60}")
        print(f"📊 {name}")
        print(f"{'='*60}")

        try:
            reconstructor = SurfaceReconstructor(
                model_path=str(model_path),
                sr=48000,
                confidence_config={"enabled": False},
                position_aggregation="highest_confidence",
                logger=logger,
            )

            result = create_reconstruction_with_edges(
                reconstructor=reconstructor,
                dataset_path=str(balanced_full),
                original_dataset_path=str(original_full),
                output_dir=output_dir / name,
                dataset_name=name,
                feature_extractor=feature_extractor,
            )

            accuracy = result.get("accuracy", 0)
            n_positions = result.get("n_positions", 0)

            icon = "🔴" if is_validation and accuracy < 0.8 else "🟢"

            print(f"\n   {icon} Accuracy: {accuracy:.1%}")
            print(f"      Positions: {n_positions}")

            results_summary[name] = {
                "accuracy": accuracy,
                "n_positions": n_positions,
                "is_validation": is_validation,
            }

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback

            traceback.print_exc()
            results_summary[name] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("📋 PATTERN A RECONSTRUCTION SUMMARY")
    print("=" * 80)

    test_accs = []
    val_accs = []

    for name, result in results_summary.items():
        if "error" not in result:
            acc = result["accuracy"]
            if result.get("is_validation"):
                val_accs.append(acc)
                print(f"  🔴 {name}: {acc:.1%} (VALIDATION)")
            else:
                test_accs.append(acc)
                print(f"  🟢 {name}: {acc:.1%} (TEST)")

    if test_accs:
        print(f"\n  📈 Average TEST accuracy: {np.mean(test_accs):.1%}")
    if val_accs:
        print(f"  📉 Average VALIDATION accuracy: {np.mean(val_accs):.1%}")
    if test_accs and val_accs:
        print(f"  ⚠️  Generalization Gap: {np.mean(test_accs) - np.mean(val_accs):.1%}")

    print(f"\n✅ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
