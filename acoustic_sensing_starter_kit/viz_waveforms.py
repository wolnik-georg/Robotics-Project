#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Script to Visualize Waveforms of WAV Recordings

This script:
1. Selects 5 random WAV files from each class (contact, no_contact, edge)
2. Plots their waveforms to visualize oscillations/transients
3. Displays the plots for two datasets

Usage: python viz_waveforms.py
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import random

# ==================
# USER SETTINGS
# ==================
DATA_DIR1 = "data/balanced_collected_data/data"  # First dataset folder
DATA_DIR2 = "data/reduce_signal_time_experiment/data"  # Second dataset folder (specify your second dataset here)
DATASET_NAMES = ["Balanced Dataset", "Original Dataset"]  # Names for labeling
CLASSES = ["contact", "edge", "no_contact"]
NUM_SAMPLES_PER_CLASS = 5  # Number of random samples to visualize per class
SR = 48000  # Sampling rate
# ==================


def load_and_plot_waveforms(data_dir, dataset_name, classes, num_samples, sr):
    """Load and plot waveforms for random samples from each class."""
    for class_name in classes:
        # Find WAV files for this class (assuming filenames contain class name, e.g., "contact_001.wav")
        all_files = [
            f for f in os.listdir(data_dir) if f.endswith(".wav") and class_name in f
        ]

        if len(all_files) < num_samples:
            print(
                f"Warning: Only {len(all_files)} files for {class_name} in {dataset_name}, selecting all."
            )
            selected_files = all_files
        else:
            selected_files = random.sample(all_files, num_samples)

        # Plot waveforms
        fig, axes = plt.subplots(
            num_samples, 1, figsize=(12, 3 * num_samples), sharex=True
        )
        if num_samples == 1:
            axes = [axes]

        for i, filename in enumerate(selected_files):
            filepath = os.path.join(data_dir, filename)
            try:
                y, sr_loaded = librosa.load(filepath, sr=sr)
                times = np.arange(len(y)) / sr_loaded

                axes[i].plot(times, y, color="blue", alpha=0.7)
                axes[i].set_title(f"{dataset_name} - {class_name} - {filename}")
                axes[i].set_ylabel("Amplitude")
                axes[i].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error: {e}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.suptitle(
            f"Waveforms for {dataset_name} - {class_name}", fontsize=14, y=0.98
        )
        plt.show()  # Display the plot
        print(f"Displayed plot for {dataset_name} - {class_name}")


def main():
    print("Visualizing Waveforms for Two Datasets")
    print("=" * 40)

    # Visualize first dataset
    print(f"\nProcessing {DATASET_NAMES[0]}...")
    load_and_plot_waveforms(
        DATA_DIR1, DATASET_NAMES[0], CLASSES, NUM_SAMPLES_PER_CLASS, SR
    )

    # Visualize second dataset
    print(f"\nProcessing {DATASET_NAMES[1]}...")
    load_and_plot_waveforms(
        DATA_DIR2, DATASET_NAMES[1], CLASSES, NUM_SAMPLES_PER_CLASS, SR
    )

    print("\nAll plots displayed. Close the windows to continue.")


if __name__ == "__main__":
    main()
