#!/usr/bin/env python3
"""
Finger Position Verification Script
===================================

This script analyzes acoustic recordings to verify that different finger touch positions
(tip, middle, bottom, blank) produce distinguishable acoustic signatures.

Features analyzed:
- Peak amplitude around 500Hz (resonant frequency)
- High-frequency energy (>1000Hz)
- RMS energy of initial contact burst (first 0.5s)

Usage:
    python scripts/verify_finger_positions.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import librosa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, LeaveOneOut
import sys

# Add src to path for imports
sys.path.append("src")
from preprocessing import load_audio
import preprocessing

# Configuration
SR = 48000  # Sample rate
DATA_DIRS = ["data/soft_finger_batch_3/data"]


def analyze_sample(filepath):
    """Extract acoustic features from a single audio sample."""
    try:
        # Load audio
        audio = load_audio(filepath, sr=SR)

        # Use the same feature extraction as the model (STFT by default)
        features = preprocessing.audio_to_features(audio)

        # For analysis, we'll look at key frequency bands
        # STFT features are summed power per frequency bin
        freqs = features.index

        # 1. Peak amplitude around 500Hz (finger resonance)
        mask_500hz = (freqs > 450) & (freqs < 550)
        if np.any(mask_500hz):
            peak_idx = np.argmax(features.values[mask_500hz])
            peak_amp = features.values[mask_500hz][peak_idx]
        else:
            peak_amp = 0

        # 2. High-frequency energy (>1000Hz)
        mask_high = freqs > 1000
        high_energy = np.sum(features.values[mask_high]) if np.any(mask_high) else 0

        # 3. RMS of first 0.5s (contact burst)
        burst_samples = int(0.5 * SR)
        if len(audio) >= burst_samples:
            rms_burst = np.sqrt(np.mean(audio[:burst_samples] ** 2))
        else:
            rms_burst = np.sqrt(np.mean(audio**2))

        return peak_amp, high_energy, rms_burst

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0, 0, 0


def collect_data():
    """Collect all audio samples and their labels."""
    results = {}
    all_features = []
    all_labels = []

    for data_dir in DATA_DIRS:
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} does not exist, skipping...")
            continue

        print(f"Processing directory: {data_dir}")

        for filename in os.listdir(data_dir):
            if not filename.endswith(".wav"):
                continue

            filepath = os.path.join(data_dir, filename)

            # Extract label from filename (format: N_label.wav)
            parts = filename.split("_")
            if len(parts) >= 2:
                label = "_".join(parts[1:]).replace(".wav", "")

                # Map common variations to standard labels
                label_map = {
                    "finger_tip": "tip",
                    "finger_middle": "middle",
                    "finger_bottom": "bottom",
                    "finger_blank": "blank",
                    "tip": "tip",
                    "middle": "middle",
                    "bottom": "bottom",
                    "base": "bottom",  # alias
                    "back": "bottom",  # alias
                    "no_tap": "blank",
                    "none": "blank",
                    "void": "blank",
                    "blank": "blank",
                }
                label = label_map.get(label, label)

                # Analyze sample
                features = analyze_sample(filepath)
                results.setdefault(label, []).append(features)

                # Collect for LDA
                all_features.append(features)
                all_labels.append(label)

    return results, np.array(all_features), np.array(all_labels)


def print_statistics(results):
    """Print statistical summary of features by label."""
    print("\n" + "=" * 80)
    print("FINGER POSITION VERIFICATION - FEATURE STATISTICS")
    print("=" * 80)
    print(
        "LABEL          PEAK_AMP     STD    HIGH_ENERGY    STD    RMS_BURST    STD    SAMPLES"
    )
    print("-" * 80)

    for label in sorted(results.keys()):
        samples = results[label]
        if not samples:
            continue

        # Extract features
        peak_amps = [s[0] for s in samples]
        high_energies = [s[1] for s in samples]
        rms_bursts = [s[2] for s in samples]

        # Calculate statistics
        n_samples = len(samples)
        peak_mean = np.mean(peak_amps)
        peak_std = np.std(peak_amps)
        high_mean = np.mean(high_energies)
        high_std = np.std(high_energies)
        rms_mean = np.mean(rms_bursts)
        rms_std = np.std(rms_bursts)

        print(
            f"{label:<15}{peak_mean:>12.4f}{peak_std:>8.4f}{high_mean:>15.1f}{high_std:>8.1f}{rms_mean:>12.6f}{rms_std:>8.6f}{n_samples:>8d}"
        )


def plot_feature_distributions(results):
    """Create plots showing feature distributions for each label."""
    labels = sorted(results.keys())
    features = ["Peak Amplitude (500Hz)", "High-Freq Energy", "RMS Burst"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Finger Position Feature Distributions", fontsize=16)

    for i, feature_name in enumerate(features):
        ax = axes[i]
        feature_data = []

        for label in labels:
            samples = results[label]
            if samples:
                feature_values = [s[i] for s in samples]
                feature_data.append(feature_values)

        if feature_data:
            ax.boxplot(feature_data, tick_labels=labels)
            ax.set_title(feature_name)
            ax.set_ylabel("Amplitude")
            ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "data/results/finger_position_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()  # Close without showing


def run_lda_classification(features, labels):
    """Run LDA classification with leave-one-out cross-validation."""
    if len(features) < 4:
        print("Warning: Need at least 4 samples for meaningful LDA analysis")
        return

    print("\n" + "=" * 80)
    print("LINEAR DISCRIMINANT ANALYSIS - CLASSIFICATION PERFORMANCE")
    print("=" * 80)

    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(features, labels)

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    scores = cross_val_score(lda, features, labels, cv=loo)

    print(".3f")
    print(".3f")
    print(f"Standard Deviation: {np.std(scores):.3f}")

    # Confusion matrix would be good but with small sample sizes it's not meaningful
    print(f"\nIndividual Fold Accuracies: {scores}")

    return np.mean(scores)


def main():
    """Main analysis pipeline."""
    print("🔍 Finger Position Verification Analysis")
    print("========================================")

    # Collect data
    results, all_features, all_labels = collect_data()

    if not results:
        print("❌ No data found! Check that audio files exist in the data directories.")
        return

    # Print statistics
    print_statistics(results)

    # Plot distributions
    try:
        plot_feature_distributions(results)
        print(
            "📊 Feature distribution plots saved to: data/results/finger_position_analysis.png"
        )
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

    # Run LDA classification
    if len(all_features) >= 4:
        accuracy = run_lda_classification(all_features, all_labels)
        if accuracy > 0.8:
            print(
                "✅ Excellent separability! Finger positions are clearly distinguishable."
            )
        elif accuracy > 0.6:
            print("⚠️  Moderate separability. Some positions may be confused.")
        else:
            print("❌ Poor separability. Finger positions are not well distinguished.")
    else:
        print("⚠️  Insufficient data for LDA analysis (need ≥4 samples)")

    print("\n" + "=" * 80)
    print("Analysis complete! Check the plots and statistics above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
