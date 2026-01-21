#!/usr/bin/env python3
"""
Helper Script: Run Single Dataset Analysis

Usage:
    python3 run_single_dataset.py <dataset_name> [output_name]

Example:
    python3 run_single_dataset.py balanced_collected_data_runs_2025_12_15_v2_2_workspace3_squares_cutout_undersample
    python3 run_single_dataset.py balanced_collected_data_runs_2025_12_15_v2_2_workspace3_squares_cutout_undersample workspace3_analysis
"""

import sys
import os
import yaml
import subprocess
from pathlib import Path

# Template config for single dataset analysis
SINGLE_DATASET_CONFIG = {
    "base_data_dir": "data",
    "output": {"base_dir": "single_dataset_results", "run_name": "dataset_analysis"},
    "multi_dataset_training": {"enabled": False},
    "datasets": [],
    "experiments": {
        "data_processing": {
            "enabled": True,
            "include_impulse_features": True,
            "apply_audio_smoothing": False,
            "motion_artifact_removal": {"enabled": False},
        },
        "dimensionality_reduction": {
            "enabled": True,
            "pca_enabled": True,
            "tsne_enabled": True,
            "tsne_perplexity_values": [30],
        },
        "discrimination_analysis": {"enabled": True},
        "saliency_analysis": {"enabled": False},
        "feature_ablation": {"enabled": False},
        "impulse_response": {"enabled": False},
        "frequency_band_ablation": {"enabled": False},
        "multi_dataset_training": {"enabled": False},
        "surface_reconstruction": {"enabled": False},
    },
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_single_dataset.py <dataset_name> [output_name]")
        print("\nExample:")
        print(
            "  python3 run_single_dataset.py balanced_collected_data_runs_2025_12_15_v2_2_workspace3_squares_cutout_undersample"
        )
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_name = (
        sys.argv[2]
        if len(sys.argv) > 2
        else dataset_name.replace("balanced_collected_data_runs_", "")
        .replace("_undersample", "")
        .replace("_oversample", "")
    )

    # Check if dataset exists
    dataset_path = Path("data") / dataset_name
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print(f"\nAvailable datasets in data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for item in sorted(data_dir.iterdir()):
                if item.is_dir() and item.name.startswith("balanced_"):
                    print(f"  - {item.name}")
        sys.exit(1)

    # Create config
    config = SINGLE_DATASET_CONFIG.copy()
    config["datasets"] = [dataset_name]
    config["output"]["run_name"] = output_name

    # Save config
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / f"temp_{output_name}.yml"

    print(f"📋 Creating config: {config_file}")
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"📂 Dataset: {dataset_name}")
    print(f"📊 Output: single_dataset_results/{output_name}/")
    print(f"\n🚀 Running pipeline...\n")

    # Run pipeline
    try:
        result = subprocess.run(
            ["python3", "run_modular_experiments.py", str(config_file)], check=True
        )

        print(f"\n✅ Analysis complete!")
        print(f"📁 Results saved to: single_dataset_results/{output_name}/")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with error code {e.returncode}")
        sys.exit(1)
    finally:
        # Optionally clean up temp config
        # config_file.unlink()
        pass


if __name__ == "__main__":
    main()
