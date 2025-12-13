#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Script to Balance Collected Data by Undersampling Majority Class

This script:
1. Loads all WAV files from the collected data directory
2. Groups labels into 3 classes (contact, no_contact, edge)
3. Undersamples the majority class (edge) to balance
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
    "data", "collected_data_all", "data"
)  # Input data folder (relative path)
OUTPUT_DIR = "balanced_collected_data"  # Output folder for balanced data
SR = 48000
RANDOM_SEED = 42
CLASSES = ["contact", "edge", "no_contact"]
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
    """Group raw labels into 3 classes."""
    grouped = []
    for label in labels:
        if label.startswith("surface"):
            grouped.append("contact")
        elif label.startswith("no_surface"):
            grouped.append("no_contact")
        elif label.startswith("edge"):
            grouped.append("edge")
        else:
            grouped.append(label)
    return grouped


def balance_data(labels, file_paths, target_edge_count=None):
    """Undersample all classes to the minimum count."""
    y = np.array(labels)
    file_paths = np.array(file_paths)

    # Get indices for each class
    edge_indices = np.where(y == "edge")[0]
    contact_indices = np.where(y == "contact")[0]
    no_contact_indices = np.where(y == "no_contact")[0]

    # Find the minimum count
    min_count = min(len(edge_indices), len(contact_indices), len(no_contact_indices))
    print(
        f"Minimum class count: {min_count}. Balancing all classes to {min_count} samples."
    )

    random.seed(RANDOM_SEED)
    sampled_edge_indices = random.sample(list(edge_indices), min_count)
    sampled_contact_indices = random.sample(list(contact_indices), min_count)
    sampled_no_contact_indices = random.sample(list(no_contact_indices), min_count)

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

    for idx in balanced_indices:
        src_path = file_paths[idx]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(data_dir, filename)
        shutil.copy2(src_path, dst_path)

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

    # Balance
    balanced_indices = balance_data(grouped_labels, file_paths)
    balanced_labels = [grouped_labels[i] for i in balanced_indices]

    # Show balanced distribution
    balanced_counts = Counter(balanced_labels)
    print("\nBalanced class distribution:")
    for cls, count in sorted(balanced_counts.items()):
        print(f"  {cls}: {count}")

    # Save
    save_balanced_data(file_paths, balanced_indices, grouped_labels, OUTPUT_DIR)

    print(f"\nBalanced dataset created with {len(balanced_indices)} samples")


if __name__ == "__main__":
    main()
