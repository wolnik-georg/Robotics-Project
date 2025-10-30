#!/usr/bin/env python3
"""
Script to validate recorded acoustic sensing data by visualizing spectra from different classes.
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from glob import glob

# Configuration
MODEL_NAME = "material_tapping_demo"
DATA_DIR = f"./{MODEL_NAME}"


def load_sample(file_path):
    """Load a .wav file and return the audio data."""
    y, sr = librosa.load(file_path, sr=48000)
    return y, sr


def plot_spectra_comparison():
    """Plot spectra of samples from each class for comparison."""
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found. Run A_record.py first.")
        return

    # Get files for each class
    tap_files = glob(f"{DATA_DIR}/*_tap.wav")
    no_tap_files = glob(f"{DATA_DIR}/*_no_tap.wav")

    print(f"Found {len(tap_files)} tap samples and {len(no_tap_files)} no_tap samples")

    if len(tap_files) == 0 or len(no_tap_files) == 0:
        print("No samples found. Run A_record.py to collect data.")
        return

    # Load a few samples from each class
    num_samples = min(5, len(tap_files), len(no_tap_files))

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        # Load tap sample
        tap_y, sr = load_sample(tap_files[i])
        tap_spectrum = np.abs(librosa.stft(tap_y, n_fft=1024))
        tap_spectrum_db = librosa.amplitude_to_db(tap_spectrum, ref=np.max)

        # Load no_tap sample
        no_tap_y, sr = load_sample(no_tap_files[i])
        no_tap_spectrum = np.abs(librosa.stft(no_tap_y, n_fft=1024))
        no_tap_spectrum_db = librosa.amplitude_to_db(no_tap_spectrum, ref=np.max)

        # Plot tap spectrum
        axes[i][0].imshow(
            tap_spectrum_db,
            aspect="auto",
            origin="lower",
            extent=[0, len(tap_y) / sr, 0, sr / 2],
        )
        axes[i][0].set_title(f"Tap Sample {i+1}")
        axes[i][0].set_ylabel("Frequency (Hz)")
        axes[i][0].set_xlabel("Time (s)")

        # Plot no_tap spectrum
        axes[i][1].imshow(
            no_tap_spectrum_db,
            aspect="auto",
            origin="lower",
            extent=[0, len(no_tap_y) / sr, 0, sr / 2],
        )
        axes[i][1].set_title(f"No-Tap Sample {i+1}")
        axes[i][1].set_ylabel("Frequency (Hz)")
        axes[i][1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def check_file_sizes():
    """Check if files have reasonable sizes (not empty or too small)."""
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found.")
        return

    files = glob(f"{DATA_DIR}/*.wav")
    print(f"Checking {len(files)} files:")

    for file in files[:10]:  # Check first 10
        size = os.path.getsize(file)
        print(f"{os.path.basename(file)}: {size} bytes")

    if len(files) > 10:
        print(f"... and {len(files)-10} more files")


if __name__ == "__main__":
    print("=== Acoustic Sensing Data Validation ===")
    check_file_sizes()
    print("\nGenerating spectrogram comparison plots...")
    plot_spectra_comparison()
