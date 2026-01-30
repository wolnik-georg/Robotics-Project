#!/usr/bin/env python3
"""
Quick validation script to check if edge samples exist in balanced datasets.

Usage: python3 validate_no_edge.py
"""

import os
from pathlib import Path
from collections import Counter


def check_dataset(dataset_path):
    """Check a single dataset for edge samples."""
    data_dir = Path(dataset_path) / "data"

    if not data_dir.exists():
        return None, f"Directory not found: {data_dir}"

    # Get all WAV files
    wav_files = list(data_dir.glob("*.wav"))

    if len(wav_files) == 0:
        return None, "No WAV files found"

    # Extract labels from filenames (format: number_class.wav)
    labels = []
    for f in wav_files:
        parts = f.stem.split("_", 1)
        if len(parts) >= 2:
            labels.append(parts[1])

    # Count classes
    counts = Counter(labels)

    return counts, None


def main():
    print("=" * 70)
    print("Validating Balanced Datasets - Checking for Edge Samples")
    print("=" * 70)

    # List of datasets to check (add your dataset names here)
    datasets = [
        "balanced_collected_data_runs_2026_01_15_workspace_2_squares_cutout_relabeled_undersample",
        "balanced_collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled_undersample",
        "balanced_collected_data_runs_2025_12_15_v2_2_workspace3_squares_cutout_relabeled_undersample",
        "balanced_collected_data_runs_2025_12_17_v2_workspace_3_squares_cutout_relabeled_undersample",
        "balanced_collected_data_runs_2026_01_27_hold_out_dataset_relabeled_undersample",
    ]

    base_dir = Path("data")

    all_clean = True

    for dataset_name in datasets:
        dataset_path = base_dir / dataset_name

        print(f"\n📁 {dataset_name}")
        print("-" * 70)

        counts, error = check_dataset(dataset_path)

        if error:
            print(f"  ⚠️  {error}")
            continue

        # Display counts
        print(f"  Total samples: {sum(counts.values())}")
        print("  Class distribution:")
        for cls, count in sorted(counts.items()):
            print(f"    {cls}: {count}")

        # Check for edge
        if "edge" in counts:
            print(f"\n  ❌ ERROR: Found {counts['edge']} edge samples!")
            all_clean = False
        else:
            print(f"\n  ✅ SUCCESS: No edge samples found")

    # Final summary
    print("\n" + "=" * 70)
    if all_clean:
        print("🎉 ALL DATASETS CLEAN - No edge samples found in any dataset!")
    else:
        print("⚠️  SOME DATASETS CONTAIN EDGE SAMPLES - Review needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
