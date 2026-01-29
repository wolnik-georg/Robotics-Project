#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Script to Balance Collected Data by Undersampling Majority Class

This script:
1. Loads all WAV files from the collected data directory
2. Groups labels into 2 classes (contact, no_contact)
3. Balances classes using undersampling or oversampling
4. Saves balanced WAV files to a new folder structure

Usage: python balance_dataset.py
"""

import numpy as np
import librosa
import os
import shutil
from collections import Counter
import random

# ==================
# USER SETTINGS
# ==================
DATA_DIR = os.path.join(
    "data",
    "collected_data_runs_2026_01_27_hold_out_dataset_relabeled",
    "data",
)  # Input data folder (relative path)
OUTPUT_DIR_UNDERSAMPLE = "balanced_collected_data_runs_2026_01_27_hold_out_dataset_relabeled_undersample"  # Output folder for undersampled balanced data
OUTPUT_DIR_OVERSAMPLE = "balanced_collected_data_runs_2026_01_27_hold_out_dataset_relabeled_oversample"  # Output folder for oversampled balanced data
SR = 48000
RANDOM_SEED = 42
CLASSES = ["contact", "no_contact", "edge"]
BALANCE_METHOD = "both"  # "undersample", "oversample", or "both"
# ==================


def load_data(data_dir):
    """Load WAV files and extract labels from filenames."""
    sounds = []
    labels = []
    file_paths = []

    # Get all WAV files directly in the data_dir
    wav_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    for audio_path in wav_files:
        try:
            sound = librosa.load(audio_path, sr=SR)[0]
            sounds.append(sound)

            # Extract label from filename (e.g., "1_contact.wav" -> "contact")
            filename = os.path.basename(audio_path)
            # Assuming format: number_class.wav, e.g., "1_contact.wav"
            parts = filename.replace(".wav", "").split("_")
            if len(parts) >= 2:
                label = "_".join(
                    parts[1:]
                )  # Take everything after the first underscore
            else:
                label = "unknown"

            labels.append(label)
            file_paths.append(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")

    return sounds, labels, file_paths


def group_labels(labels):
    """Group raw labels into 3 classes: contact, no_contact, and edge."""
    grouped = []
    for label in labels:
        # Remove common prefixes
        clean_label = label.replace("finger_", "").replace("finger", "")

        if clean_label.startswith("surface") or clean_label == "contact":
            grouped.append("contact")
        elif clean_label.startswith("no_surface") or clean_label == "no_contact":
            grouped.append("no_contact")
        elif clean_label.startswith("edge") or clean_label == "edge":
            grouped.append("edge")
        else:
            # Default: treat unknown labels as no_contact or skip
            print(f"Warning: Unknown label '{clean_label}' - treating as no_contact")
            grouped.append("no_contact")
    return grouped


def balance_data(labels, file_paths, method="undersample"):
    """Balance data using undersampling or oversampling."""
    y = np.array(labels)
    file_paths = np.array(file_paths)

    # Get indices for each class
    contact_indices = np.where(y == "contact")[0]
    no_contact_indices = np.where(y == "no_contact")[0]
    edge_indices = np.where(y == "edge")[0]

    counts = {
        "contact": len(contact_indices),
        "no_contact": len(no_contact_indices),
        "edge": len(edge_indices),
    }

    # Filter out classes with 0 samples
    available_classes = {k: v for k, v in counts.items() if v > 0}

    if len(available_classes) == 0:
        raise ValueError("No valid samples found in any class!")

    if len(available_classes) == 1:
        # Only one class present - return all samples
        class_name = list(available_classes.keys())[0]
        print(
            f"Warning: Only '{class_name}' class found with {available_classes[class_name]} samples."
        )
        print(f"Returning all samples without balancing.")
        if class_name == "contact":
            return list(contact_indices)
        elif class_name == "edge":
            return list(edge_indices)
        else:
            return list(no_contact_indices)

    # Multiple classes present - proceed with balancing
    if method == "undersample":
        # Undersample to minimum count
        min_count = min(available_classes.values())
        print(f"Undersampling to {min_count} samples per class.")

        random.seed(RANDOM_SEED)
        sampled_contact_indices = (
            random.sample(list(contact_indices), min_count)
            if counts["contact"] > 0
            else []
        )
        sampled_no_contact_indices = (
            random.sample(list(no_contact_indices), min_count)
            if counts["no_contact"] > 0
            else []
        )
        sampled_edge_indices = (
            random.sample(list(edge_indices), min_count) if counts["edge"] > 0 else []
        )

        balanced_indices = (
            list(sampled_contact_indices)
            + list(sampled_no_contact_indices)
            + list(sampled_edge_indices)
        )

    elif method == "oversample":
        # Oversample to maximum count by duplicating
        max_count = max(available_classes.values())
        print(f"Oversampling to {max_count} samples per class by duplication.")

        random.seed(RANDOM_SEED)

        if counts["contact"] > 0:
            sampled_contact_indices = list(contact_indices) + random.choices(
                list(contact_indices), k=max_count - len(contact_indices)
            )
        else:
            sampled_contact_indices = []

        if counts["no_contact"] > 0:
            sampled_no_contact_indices = list(no_contact_indices) + random.choices(
                list(no_contact_indices), k=max_count - len(no_contact_indices)
            )
        else:
            sampled_no_contact_indices = []

        if counts["edge"] > 0:
            sampled_edge_indices = list(edge_indices) + random.choices(
                list(edge_indices), k=max_count - len(edge_indices)
            )
        else:
            sampled_edge_indices = []

        balanced_indices = (
            list(sampled_contact_indices)
            + list(sampled_no_contact_indices)
            + list(sampled_edge_indices)
        )

    random.shuffle(balanced_indices)
    return balanced_indices


def save_balanced_data(file_paths, balanced_indices, grouped_labels, output_dir):
    """Save balanced WAV files to a single data subfolder."""
    data_dir = os.path.join(output_dir, "data")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Use sequential counter for unique filenames
    counter = 0

    for idx in balanced_indices:
        src_path = file_paths[idx]
        grouped_label = grouped_labels[idx]

        # Create filename with sequential counter and grouped label
        new_filename = f"{counter}_{grouped_label}.wav"

        dst_path = os.path.join(data_dir, new_filename)
        shutil.copy2(src_path, dst_path)

        counter += 1

    print(f"Balanced data saved to {data_dir}")


def main():
    print("Balancing Collected Data")
    print("=" * 40)

    # Load and group data
    sounds, raw_labels, file_paths = load_data(DATA_DIR)
    grouped_labels = group_labels(raw_labels)

    # Show original distribution
    original_counts = Counter(grouped_labels)
    print("Original class distribution:")
    for cls, count in sorted(original_counts.items()):
        print(f"  {cls}: {count}")

    # Check if we have any data
    if len(grouped_labels) == 0:
        print("Error: No data found!")
        return

    # Check number of unique classes
    unique_classes = set(grouped_labels)
    print(f"\nFound {len(unique_classes)} unique class(es): {sorted(unique_classes)}")

    methods = []
    if BALANCE_METHOD in ["undersample", "both"]:
        methods.append(("undersample", OUTPUT_DIR_UNDERSAMPLE))
    if BALANCE_METHOD in ["oversample", "both"]:
        methods.append(("oversample", OUTPUT_DIR_OVERSAMPLE))

    for method, output_dir in methods:
        print(f"\n--- Balancing with {method} ---")

        # Balance
        balanced_indices = balance_data(grouped_labels, file_paths, method)
        balanced_labels = [grouped_labels[i] for i in balanced_indices]

        # Show balanced distribution
        balanced_counts = Counter(balanced_labels)
        print(f"\n{method.capitalize()} balanced class distribution:")
        for cls, count in sorted(balanced_counts.items()):
            print(f"  {cls}: {count}")

        # Save
        save_balanced_data(file_paths, balanced_indices, grouped_labels, output_dir)

        print(
            f"{method.capitalize()} balanced dataset created with {len(balanced_indices)} samples in {output_dir}"
        )


if __name__ == "__main__":
    main()
