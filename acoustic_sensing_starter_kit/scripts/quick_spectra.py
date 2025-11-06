#!/usr/bin/env python3
"""
Quick spectral analysis to check finger position differences
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import sys

# Add src to path
sys.path.append("src")
from preprocessing import load_audio

SR = 48000


def plot_spectra_comparison():
    """Plot spectra for different finger positions"""
    data_dir = "data/soft_finger_batch_3/data"

    # Get one sample from each position
    samples = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav") and not filename.startswith("0_"):  # Skip sweep
            parts = filename.split("_")
            if len(parts) >= 2:
                label = "_".join(parts[1:]).replace(".wav", "")
                if label not in samples:
                    samples[label] = os.path.join(data_dir, filename)

    print("Comparing spectra for:", list(samples.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (label, filepath) in enumerate(samples.items()):
        if i >= 4:
            break

        # Load and analyze
        audio = load_audio(filepath, sr=SR)
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / SR)

        # Plot full spectrum
        axes[i].plot(freqs, 20 * np.log10(spec + 1e-10))
        axes[i].set_title(f"{label}")
        axes[i].set_xlabel("Frequency (Hz)")
        axes[i].set_ylabel("Magnitude (dB)")
        axes[i].set_xlim(0, 5000)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/results/spectra_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Spectra comparison saved to: data/results/spectra_comparison.png")


if __name__ == "__main__":
    plot_spectra_comparison()
