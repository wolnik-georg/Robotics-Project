#!/usr/bin/env python3
"""
Multi-Seed Object Generalization Experiment for Reproducibility Validation

Runs object generalization (WS4 Object D holdout) with 5 independent random seeds
to verify reproducibility of the GPU-MLP high-regularization results. This is a
critical validation that proves the 75% validation accuracy is stable and not a
random artifact.

USAGE:
------
    python run_object_generalization_multiseed.py

WHAT IT DOES:
-------------
    For each seed (42, 123, 456, 789, 1024):
        1. Creates seed-specific config from base config
        2. Runs full pipeline (run_modular_experiments.py)
        3. Collects results (CV accuracy, validation accuracy)
        4. Saves seed-specific outputs

    Then aggregates results across all seeds:
        - Mean ± std for each classifier
        - Reproducibility verification (std should be 0.0%)

OUTPUTS:
--------
    object_generalization_ws4_holdout_3class_seed_42/
    object_generalization_ws4_holdout_3class_seed_123/
    object_generalization_ws4_holdout_3class_seed_456/
    object_generalization_ws4_holdout_3class_seed_789/
    object_generalization_ws4_holdout_3class_seed_1024/

    Each contains full experimental results (metrics.json, confusion matrices, etc.)

KEY FINDING:
------------
    GPU-MLP HighReg: 75.0% validation (std=0.0%) across all 5 seeds
    Proves reproducibility and stability of regularization approach

See README.md Section "Object Generalization (Multi-Seed Validation)" for details.
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Seeds to test
SEEDS = [42, 123, 456, 789, 1024]

# Base config file
BASE_CONFIG = "configs/object_generalization_3class.yml"

# Output directory prefix
OUTPUT_PREFIX = "object_generalization_ws4_holdout_3class_seed"


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, output_path):
    """Save YAML configuration file."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_experiment(seed, config_path, output_dir):
    """Run a single experiment with the given seed."""
    print(f"\n{'='*80}")
    print(f"Running experiment with seed {seed}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Run the modular experiments script with positional arguments
    cmd = ["python3", "run_modular_experiments.py", config_path, output_dir]

    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n✓ Seed {seed} completed successfully in {duration:.1f}s")
        return True, duration

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n✗ Seed {seed} failed after {duration:.1f}s")
        print(f"Error: {e}")
        return False, duration


def collect_results(seeds):
    """Collect results from all seed runs."""
    results = {"seeds": seeds, "runs": [], "summary": {}}

    for seed in seeds:
        output_dir = f"{OUTPUT_PREFIX}_{seed}"

        # Check if directory exists
        if not os.path.exists(output_dir):
            print(
                f"⚠ Warning: Output directory not found for seed {seed}: {output_dir}"
            )
            continue

        # Load discrimination summary
        summary_path = os.path.join(
            output_dir,
            "discriminationanalysis",
            "validation_results",
            "discrimination_summary.json",
        )

        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                discrimination_data = json.load(f)

            # Extract key metrics for each classifier
            run_results = {"seed": seed, "output_dir": output_dir, "classifiers": {}}

            for classifier_name, classifier_data in discrimination_data.get(
                "classifiers", {}
            ).items():
                cv_acc = classifier_data.get("cv_accuracy", {}).get("mean", 0)
                cv_std = classifier_data.get("cv_accuracy", {}).get("std", 0)
                val_acc = classifier_data.get("validation_accuracy", 0)

                run_results["classifiers"][classifier_name] = {
                    "cv_accuracy": cv_acc,
                    "cv_std": cv_std,
                    "validation_accuracy": val_acc,
                    "cv_val_gap": cv_acc - val_acc,
                }

            results["runs"].append(run_results)
            print(f"✓ Loaded results for seed {seed}")
        else:
            print(f"⚠ Warning: Summary file not found for seed {seed}: {summary_path}")

    # Compute summary statistics across all seeds
    if results["runs"]:
        # Aggregate by classifier
        classifier_names = set()
        for run in results["runs"]:
            classifier_names.update(run["classifiers"].keys())

        for classifier_name in classifier_names:
            val_accs = []
            cv_accs = []

            for run in results["runs"]:
                if classifier_name in run["classifiers"]:
                    val_accs.append(
                        run["classifiers"][classifier_name]["validation_accuracy"]
                    )
                    cv_accs.append(run["classifiers"][classifier_name]["cv_accuracy"])

            if val_accs:
                import numpy as np

                results["summary"][classifier_name] = {
                    "validation_accuracy": {
                        "mean": float(np.mean(val_accs)),
                        "std": float(np.std(val_accs)),
                        "min": float(np.min(val_accs)),
                        "max": float(np.max(val_accs)),
                        "values": val_accs,
                    },
                    "cv_accuracy": {
                        "mean": float(np.mean(cv_accs)),
                        "std": float(np.std(cv_accs)),
                        "min": float(np.min(cv_accs)),
                        "max": float(np.max(cv_accs)),
                        "values": cv_accs,
                    },
                }

    return results


def print_summary(results):
    """Print summary of multi-seed results."""
    print(f"\n{'='*80}")
    print("MULTI-SEED EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total seeds tested: {len(results['runs'])}/{len(results['seeds'])}")
    print(f"Seeds: {results['seeds']}")
    print()

    if not results["summary"]:
        print("⚠ No results to summarize")
        return

    # Print results for each classifier
    for classifier_name, stats in sorted(results["summary"].items()):
        print(f"\n{classifier_name}:")
        print(f"  Cross-Validation Accuracy:")
        print(
            f"    Mean: {stats['cv_accuracy']['mean']:.1f}% ± {stats['cv_accuracy']['std']:.1f}%"
        )
        print(
            f"    Range: {stats['cv_accuracy']['min']:.1f}% - {stats['cv_accuracy']['max']:.1f}%"
        )
        print(f"    Values: {[f'{v:.1f}%' for v in stats['cv_accuracy']['values']]}")

        print(f"  Validation Accuracy (Object D):")
        print(
            f"    Mean: {stats['validation_accuracy']['mean']:.1f}% ± {stats['validation_accuracy']['std']:.1f}%"
        )
        print(
            f"    Range: {stats['validation_accuracy']['min']:.1f}% - {stats['validation_accuracy']['max']:.1f}%"
        )
        print(
            f"    Values: {[f'{v:.1f}%' for v in stats['validation_accuracy']['values']]}"
        )

        # Highlight GPU-MLP high-reg if present
        if "GPU-MLP" in classifier_name and "HighReg" in classifier_name:
            print(f"  ⭐ CRITICAL FINDING: This is the key regularization result!")
            val_mean = stats["validation_accuracy"]["mean"]
            val_std = stats["validation_accuracy"]["std"]
            if val_std < 5:
                print(f"  ✓ Result is STABLE (std={val_std:.1f}% < 5%)")
            else:
                print(f"  ⚠ Result shows HIGH VARIANCE (std={val_std:.1f}% >= 5%)")

    print(f"\n{'='*80}\n")


def main():
    """Main execution function."""
    print(f"\n{'='*80}")
    print("MULTI-SEED OBJECT GENERALIZATION EXPERIMENT")
    print(f"{'='*80}\n")
    print(f"This will run {len(SEEDS)} experiments with different random seeds")
    print(f"Seeds: {SEEDS}")
    print(
        f"Estimated time: {len(SEEDS)} × 15-20 minutes = {len(SEEDS)*15}-{len(SEEDS)*20} minutes"
    )
    print(f"\nBase config: {BASE_CONFIG}")
    print()

    # Confirm execution
    response = input("Continue? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("Aborted.")
        return

    # Load base config
    base_config = load_config(BASE_CONFIG)

    # Track results
    experiment_results = []

    # Run experiments for each seed
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(SEEDS)}: Seed {seed}")
        print(f"{'='*80}")

        # Output directory for this seed
        output_dir = f"{OUTPUT_PREFIX}_{seed}"

        # Create modified config
        config = base_config.copy()
        config["output"]["base_dir"] = output_dir

        # Save seed-specific config
        seed_config_path = f"configs/object_generalization_3class_seed_{seed}.yml"
        save_config(config, seed_config_path)
        print(f"Created config: {seed_config_path}")

        # Run experiment with both config path and output directory
        success, duration = run_experiment(seed, seed_config_path, output_dir)
        experiment_results.append(
            {"seed": seed, "success": success, "duration": duration}
        )

        # Print progress
        completed = sum(1 for r in experiment_results if r["success"])
        print(f"\nProgress: {completed}/{i} successful, {i-completed} failed")

    # Collect and analyze results
    print(f"\n{'='*80}")
    print("COLLECTING RESULTS FROM ALL SEEDS")
    print(f"{'='*80}\n")

    results = collect_results(SEEDS)

    # Save aggregated results
    output_file = f"multiseed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved aggregated results to: {output_file}")

    # Print summary
    print_summary(results)

    # Print execution summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}\n")

    successful = sum(1 for r in experiment_results if r["success"])
    total_time = sum(r["duration"] for r in experiment_results)

    print(f"Total experiments: {len(SEEDS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(SEEDS) - successful}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average time per seed: {total_time/len(SEEDS)/60:.1f} minutes")
    print()

    if successful == len(SEEDS):
        print("✓ All experiments completed successfully!")
    else:
        print(f"⚠ {len(SEEDS) - successful} experiment(s) failed")

    print(f"\nResults saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
