#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Script to Balance Collected Data by Undersampling

This script:
1. Loads all WAV files from the collected data directory
2. Reads the sweep.csv to get position information for each audio file
3. Groups labels into classes (contact, no_contact, edge)
4. OPTIONALLY: Filters out specified classes (e.g., edge) to create cleaner datasets
5. Balances classes using undersampling
6. Saves balanced WAV files AND a new sweep.csv with position info to the output folder

NEW FEATURE: Sweep CSV Preservation
-----------------------------------
The balanced dataset now includes a sweep.csv that maps each balanced audio file
to its original position (normalized_x, normalized_y). This enables:
- Surface reconstruction directly from balanced datasets
- Consistent feature extraction between training and reconstruction
- No need for complex file matching or hash-based lookups

Usage:
    python balance_dataset.py                                    # Use default settings
    python balance_dataset.py --input <path> --output <path>     # Specify paths
    python balance_dataset.py --input data/my_dataset --exclude-edge  # Filter edges
"""

import numpy as np
import pandas as pd
import librosa
import os
import shutil
import argparse
from collections import Counter
import random

# ==================
# USER SETTINGS
# ==================
DATA_DIR = os.path.join(
    "data",
    "collected_data_runs_2026_01_27_hold_out_dataset_relabeled",
)  # Input data folder (relative path) - parent folder containing data/ and sweep.csv
OUTPUT_DIR = "balanced_collected_data_runs_2026_01_27_hold_out_dataset_relabeled"  # Output folder for balanced data
SR = 48000
RANDOM_SEED = 42
CLASSES = ["contact", "no_contact", "edge"]

# CLASS FILTERING (applied before balancing)
FILTER_CLASSES = True  # Set to True to exclude specific classes before balancing
CLASSES_TO_EXCLUDE = [
    "edge"
]  # Classes to filter out (e.g., ["edge"] to create binary contact/no_contact datasets)

# EXAMPLES:
# Binary classification (contact vs no_contact only):
#   FILTER_CLASSES = True
#   CLASSES_TO_EXCLUDE = ["edge"]
#
# Full 3-class classification (contact, no_contact, edge):
#   FILTER_CLASSES = False
#   CLASSES_TO_EXCLUDE = []
#
# Custom filtering (e.g., only edge vs no_contact):
#   FILTER_CLASSES = True
#   CLASSES_TO_EXCLUDE = ["contact"]
# ==================


def load_data_with_positions(data_dir):
    """
    Load WAV files and position info from sweep.csv.

    Args:
        data_dir: Path to dataset folder (containing data/ subfolder and sweep.csv)

    Returns:
        Tuple of (file_paths, sweep_df) where sweep_df contains all position info
    """
    # Load sweep.csv
    sweep_path = os.path.join(data_dir, "sweep.csv")
    if not os.path.exists(sweep_path):
        raise FileNotFoundError(f"sweep.csv not found at {sweep_path}")

    sweep_df = pd.read_csv(sweep_path)
    print(f"Loaded sweep.csv with {len(sweep_df)} entries")

    # Extract just the filename from acoustic_filename (e.g., "./data/1_contact.wav" -> "1_contact.wav")
    sweep_df["filename"] = sweep_df["acoustic_filename"].apply(
        lambda x: os.path.basename(x) if pd.notna(x) else None
    )

    # Get file paths that actually exist
    data_subdir = os.path.join(data_dir, "data")
    file_paths = []

    for _, row in sweep_df.iterrows():
        if pd.isna(row["filename"]):
            continue
        audio_path = os.path.join(data_subdir, row["filename"])
        if os.path.exists(audio_path):
            file_paths.append(audio_path)
        else:
            print(f"Warning: File not found: {audio_path}")

    print(f"Found {len(file_paths)} audio files")
    return file_paths, sweep_df


def get_labels_from_sweep(sweep_df):
    """
    Get labels from sweep dataframe using relabeled_label column.

    Returns list of labels aligned with sweep_df rows.
    """
    # Use relabeled_label if available, otherwise original_label
    if "relabeled_label" in sweep_df.columns:
        labels = sweep_df["relabeled_label"].tolist()
    elif "original_label" in sweep_df.columns:
        labels = sweep_df["original_label"].tolist()
    else:
        # Fallback: extract from filename
        labels = []
        for filename in sweep_df["filename"]:
            parts = filename.replace(".wav", "").split("_")
            if len(parts) >= 2:
                labels.append("_".join(parts[1:]))
            else:
                labels.append("unknown")
    return labels


def filter_classes_df(sweep_df, classes_to_exclude):
    """
    Filter out samples belonging to specified classes from sweep dataframe.

    Args:
        sweep_df: DataFrame with sweep data including labels
        classes_to_exclude: List of class names to filter out (e.g., ["edge"])

    Returns:
        Filtered DataFrame
    """
    if not classes_to_exclude:
        return sweep_df

    # Get label column
    label_col = (
        "relabeled_label" if "relabeled_label" in sweep_df.columns else "original_label"
    )

    original_count = len(sweep_df)

    # Filter out excluded classes
    mask = ~sweep_df[label_col].isin(classes_to_exclude)
    filtered_df = sweep_df[mask].copy()

    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count

    print(f"\n🔍 CLASS FILTERING:")
    print(f"  Classes to exclude: {classes_to_exclude}")
    print(f"  Original samples: {original_count}")
    print(f"  Filtered samples: {filtered_count}")
    print(f"  Removed samples: {removed_count}")

    if removed_count > 0:
        print(f"  Breakdown of removed samples:")
        for cls in classes_to_exclude:
            count = (sweep_df[label_col] == cls).sum()
            if count > 0:
                print(f"    - {cls}: {count}")

    return filtered_df


def balance_data_undersample(sweep_df):
    """
    Balance data by undersampling to the minority class.

    Args:
        sweep_df: DataFrame with sweep data including labels

    Returns:
        DataFrame with balanced samples (undersampled)
    """
    # Get label column
    label_col = (
        "relabeled_label" if "relabeled_label" in sweep_df.columns else "original_label"
    )

    # Get class counts
    class_counts = sweep_df[label_col].value_counts()
    print(f"\nClass distribution before balancing:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # Find minimum count
    min_count = class_counts.min()
    print(f"\nUndersampling to {min_count} samples per class.")

    # Sample each class
    random.seed(RANDOM_SEED)
    balanced_dfs = []

    for cls in class_counts.index:
        class_df = sweep_df[sweep_df[label_col] == cls]
        if len(class_df) > min_count:
            # Undersample
            sampled_df = class_df.sample(n=min_count, random_state=RANDOM_SEED)
        else:
            sampled_df = class_df
        balanced_dfs.append(sampled_df)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(
        drop=True
    )

    print(f"\nClass distribution after balancing:")
    for cls, count in balanced_df[label_col].value_counts().items():
        print(f"  {cls}: {count}")

    return balanced_df


def save_balanced_data_with_sweep(balanced_df, data_dir, output_dir):
    """
    Save balanced WAV files and a new sweep.csv to output folder.

    Args:
        balanced_df: DataFrame with balanced sweep data
        data_dir: Source data directory (parent folder with data/ subfolder)
        output_dir: Output directory for balanced dataset
    """
    # Create output directories
    output_data_dir = os.path.join(output_dir, "data")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_data_dir, exist_ok=True)

    # Get label column for naming
    label_col = (
        "relabeled_label"
        if "relabeled_label" in balanced_df.columns
        else "original_label"
    )

    # Prepare new sweep data
    new_sweep_rows = []

    # Copy files and build new sweep CSV
    for idx, (_, row) in enumerate(balanced_df.iterrows()):
        # Source file
        src_filename = row["filename"]
        src_path = os.path.join(data_dir, "data", src_filename)

        # New filename with sequential counter
        label = row[label_col]
        new_filename = f"{idx}_{label}.wav"
        dst_path = os.path.join(output_data_dir, new_filename)

        # Copy file
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Source file not found: {src_path}")
            continue

        # Create new sweep row
        new_row = row.copy()
        new_row["acoustic_filename"] = f"./data/{new_filename}"
        new_row["filename"] = new_filename
        new_sweep_rows.append(new_row)

    # Save new sweep.csv
    new_sweep_df = pd.DataFrame(new_sweep_rows)
    sweep_output_path = os.path.join(output_dir, "sweep.csv")
    new_sweep_df.to_csv(sweep_output_path, index=False)

    print(f"\n✅ Balanced dataset saved to: {output_dir}")
    print(f"   - {len(new_sweep_rows)} audio files in {output_data_dir}")
    print(f"   - sweep.csv with position info at {sweep_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Balance collected dataset by undersampling with position preservation"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DATA_DIR,
        help="Input dataset folder (containing data/ and sweep.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output folder for balanced dataset (default: balanced_<input_name>)",
    )
    parser.add_argument(
        "--exclude-edge",
        action="store_true",
        default=FILTER_CLASSES,
        help="Exclude edge class before balancing",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set paths
    data_dir = args.input
    if args.output:
        output_dir = args.output
    else:
        # Auto-generate output name
        input_name = os.path.basename(data_dir.rstrip("/"))
        output_dir = f"balanced_{input_name}"

    # Update random seed
    random.seed(args.seed)

    print("=" * 60)
    print("Balancing Collected Data (with Position Preservation)")
    print("=" * 60)

    # Load data with positions from sweep.csv
    print(f"\n📁 Loading data from: {data_dir}")
    file_paths, sweep_df = load_data_with_positions(data_dir)

    # Get label column
    label_col = (
        "relabeled_label" if "relabeled_label" in sweep_df.columns else "original_label"
    )

    # Show original distribution
    print("\n📊 Original class distribution:")
    for cls, count in sweep_df[label_col].value_counts().items():
        print(f"  {cls}: {count}")

    # Apply class filtering if enabled
    if args.exclude_edge:
        sweep_df = filter_classes_df(sweep_df, ["edge"])

    # Check if we have any data
    if len(sweep_df) == 0:
        print("❌ Error: No data found after filtering!")
        return

    # Check number of unique classes
    unique_classes = sweep_df[label_col].unique()
    print(
        f"\n🏷️  Found {len(unique_classes)} unique class(es): {sorted(unique_classes)}"
    )

    # Balance using undersampling
    print("\n⚖️  Balancing dataset (undersampling)...")
    balanced_df = balance_data_undersample(sweep_df)

    # Save balanced data with sweep.csv
    print("\n💾 Saving balanced dataset...")
    save_balanced_data_with_sweep(balanced_df, data_dir, output_dir)

    print(f"\n🎉 Done! Balanced dataset created at: {output_dir}")
    print("   This dataset includes sweep.csv with position info for reconstruction.")


if __name__ == "__main__":
    main()
