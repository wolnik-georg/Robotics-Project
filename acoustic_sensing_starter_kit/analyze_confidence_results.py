#!/usr/bin/env python3
"""
Analyze confidence statistics from experiment results.

Usage:
    python3 analyze_confidence_results.py <results_dir>

Example:
    python3 analyze_confidence_results.py modular_analysis_results_v12
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np


def load_results(results_dir):
    """Load results from the experiment output directory."""
    results_path = Path(results_dir)

    # Try to find results.pkl or results.json
    pkl_file = results_path / "discrimination_analysis" / "results.pkl"
    json_file = results_path / "discrimination_analysis" / "results.json"

    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            return pickle.load(f)
    elif json_file.exists():
        with open(json_file) as f:
            return json.load(f)
    else:
        print(f"❌ No results file found in {results_path}")
        return None


def analyze_confidence_stats(results):
    """Analyze and display confidence statistics."""

    clf_results = results.get("classifier_results", {})

    if not clf_results:
        print("❌ No classifier results found")
        return

    print("=" * 80)
    print("CONFIDENCE STATISTICS ANALYSIS")
    print("=" * 80)

    # Check if confidence filtering was enabled
    any_conf_stats = any(
        clf_data.get("validation_confidence_stats") is not None
        for clf_data in clf_results.values()
    )

    if not any_conf_stats:
        print("\n⚠️  Confidence filtering was NOT enabled in this experiment")
        print("   Enable it in config: confidence_filtering.enabled = true")
        return

    print("\n✅ Confidence filtering was ENABLED")
    print()

    # Analyze each classifier
    for clf_name, clf_data in clf_results.items():
        print("-" * 80)
        print(f"Classifier: {clf_name}")
        print("-" * 80)

        # Get confidence stats for different splits
        train_stats = clf_data.get("train_confidence_stats")
        test_stats = clf_data.get("test_confidence_stats")
        val_stats = clf_data.get("validation_confidence_stats")

        # Display stats for each split
        for split_name, stats in [
            ("TRAIN", train_stats),
            ("TEST", test_stats),
            ("VALIDATION", val_stats),
        ]:
            if stats is None:
                continue

            print(f"\n{split_name} Set:")
            print(f"  Total samples: {stats['total_samples']}")
            print(
                f"  High confidence: {stats['high_confidence']} ({stats['high_confidence_pct']:.1f}%)"
            )
            print(
                f"  Low confidence: {stats['low_confidence']} ({stats['low_confidence_pct']:.1f}%)"
            )
            print(f"  Mean confidence: {stats['mean_confidence']:.3f}")
            print(f"  Median confidence: {stats['median_confidence']:.3f}")
            print(f"  Min confidence: {stats['min_confidence']:.3f}")
            print(f"  Max confidence: {stats['max_confidence']:.3f}")

        # Get accuracies
        train_acc = clf_data.get("train_accuracy", 0)
        test_acc = clf_data.get("test_accuracy", 0)
        val_acc = clf_data.get("validation_accuracy", 0)

        print(f"\nAccuracies (on filtered predictions):")
        print(f"  Train: {train_acc:.4f}")
        print(f"  Test: {test_acc:.4f}")
        print(f"  Validation: {val_acc:.4f}")

        # Analyze overfitting
        if train_stats and val_stats:
            train_overfit = train_acc - val_acc
            conf_gap = train_stats["mean_confidence"] - val_stats["mean_confidence"]

            print(f"\nOverfitting Analysis:")
            print(f"  Accuracy gap (train - val): {train_overfit:+.4f}")
            print(f"  Confidence gap (train - val): {conf_gap:+.3f}")

            if train_overfit > 0.1:
                print(f"  ⚠️  Significant overfitting detected!")
            if conf_gap > 0.1:
                print(f"  ⚠️  Model more confident on training data (overconfident)")

        print()

    # Summary comparison across classifiers
    print("=" * 80)
    print("CLASSIFIER COMPARISON (Validation Set)")
    print("=" * 80)
    print()

    comparison = []
    for clf_name, clf_data in clf_results.items():
        val_stats = clf_data.get("validation_confidence_stats")
        if val_stats:
            comparison.append(
                {
                    "name": clf_name,
                    "accuracy": clf_data.get("validation_accuracy", 0),
                    "mean_conf": val_stats["mean_confidence"],
                    "high_conf_pct": val_stats["high_confidence_pct"],
                    "low_conf_pct": val_stats["low_confidence_pct"],
                }
            )

    # Sort by accuracy
    comparison.sort(key=lambda x: x["accuracy"], reverse=True)

    print(
        f"{'Classifier':<25} {'Accuracy':>10} {'Mean Conf':>11} {'High %':>9} {'Low %':>8}"
    )
    print("-" * 80)
    for item in comparison:
        print(
            f"{item['name']:<25} {item['accuracy']:>10.4f} {item['mean_conf']:>11.3f} {item['high_conf_pct']:>8.1f}% {item['low_conf_pct']:>7.1f}%"
        )

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Analyze patterns and give recommendations
    if comparison:
        best = comparison[0]

        print(f"\n🏆 Best classifier: {best['name']}")
        print(f"   Validation accuracy: {best['accuracy']:.4f}")
        print(f"   Mean confidence: {best['mean_conf']:.3f}")
        print(f"   High confidence: {best['high_conf_pct']:.1f}%")

        # Check if accuracy is low
        if best["accuracy"] < 0.6:
            print("\n⚠️  Low validation accuracy (<60%)")
            print("   Possible causes:")
            print("   1. Domain shift between training and validation workspaces")
            print("   2. Features don't generalize well")
            print("   3. Classes are inherently difficult to separate")

            if best["low_conf_pct"] > 30:
                print(
                    f"\n   High % of low-confidence predictions ({best['low_conf_pct']:.1f}%)"
                )
                print("   → Model is uncertain about many samples")
                print(
                    "   → Consider: domain adaptation, better features, or temporal models"
                )
            else:
                print(
                    f"\n   Model is confident ({best['mean_conf']:.3f} mean confidence)"
                )
                print("   → Model is overconfident (confident but wrong)")
                print("   → Consider: more diverse training data or regularization")

        # Check confidence calibration
        avg_conf = np.mean([item["mean_conf"] for item in comparison])
        avg_acc = np.mean([item["accuracy"] for item in comparison])

        print(f"\nOverall Statistics:")
        print(f"   Average accuracy: {avg_acc:.4f}")
        print(f"   Average confidence: {avg_conf:.3f}")

        if avg_conf - avg_acc > 0.2:
            print("   ⚠️  Models are overconfident (confidence > accuracy)")
            print("   → Predictions are more confident than they should be")
            print("   → For robotics: Use 'default' mode for safety")
        elif avg_acc - avg_conf > 0.1:
            print("   ✅ Models are well-calibrated or under-confident")
            print("   → Safe for robotics applications")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_confidence_results.py <results_dir>")
        print("\nExample:")
        print("  python3 analyze_confidence_results.py modular_analysis_results_v12")
        return

    results_dir = sys.argv[1]
    results = load_results(results_dir)

    if results:
        analyze_confidence_stats(results)


if __name__ == "__main__":
    main()
