"""
Motion Artifact Removal for Acoustic Sensing

This module provides state-of-the-art functions to remove robot motion artifacts from acoustic recordings
using a multi-stage VibeCheck pipeline: static reference subtraction and LMS adaptive filtering.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
import librosa
from pathlib import Path
from scipy.signal import butter, filtfilt, lfilter

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def _static_subtraction(
    sweep_audio: np.ndarray, static_template: np.ndarray
) -> np.ndarray:
    """
    VibeCheck Technique 1: Subtract mean static reference.

    Args:
        sweep_audio: Audio signal to clean
        static_template: Static reference template

    Returns:
        Cleaned audio signal
    """
    min_len = min(len(sweep_audio), len(static_template))
    clean = sweep_audio[:min_len] - static_template[:min_len]

    # RMS normalize
    rms = np.sqrt(np.mean(clean**2))
    if rms > 1e-8:
        clean /= rms
    return clean


def _lms_adaptive_filter(
    sweep_audio: np.ndarray,
    static_reference: np.ndarray,
    mu: float = 0.01,
    filter_length: int = 32,
) -> np.ndarray:
    """
    VibeCheck Technique 2: Least Mean Squares adaptive filter.

    Args:
        sweep_audio: Audio signal to clean
        static_reference: Static reference for adaptation
        mu: Learning rate
        filter_length: Length of adaptive filter

    Returns:
        Cleaned audio signal
    """
    w = np.zeros(filter_length)
    clean = np.zeros(len(sweep_audio))
    static_padded = np.pad(
        static_reference, (0, len(sweep_audio) - len(static_reference)), "constant"
    )

    for i in range(filter_length, len(sweep_audio)):
        x = static_padded[i - filter_length : i][::-1]
        y = sweep_audio[i]
        clean[i] = y - np.dot(w, x)
        w = w + mu * clean[i] * x

    # Normalize
    rms = np.sqrt(np.mean(clean**2))
    if rms > 1e-8:
        clean /= rms
    return clean


def _resonance_tracking_filter(
    audio: np.ndarray, resonance_band: Tuple[float, float] = (500, 800), sr: int = 48000
) -> np.ndarray:
    """
    VibeCheck Technique 3: Bandpass around finger resonance (500–800 Hz).

    Args:
        audio: Audio signal to filter
        resonance_band: Frequency band for resonance tracking
        sr: Sample rate

    Returns:
        Filtered audio signal
    """
    low = resonance_band[0] / (sr / 2)
    high = resonance_band[1] / (sr / 2)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, audio)

    # Normalize
    rms = np.sqrt(np.mean(filtered**2))
    if rms > 1e-8:
        filtered /= rms
    return filtered


def compute_static_templates(
    static_dir: Path, sr: int = 48000
) -> Dict[str, np.ndarray]:
    """
    Compute mean static template for each class from static reference recordings.

    Args:
        static_dir: Path to the static reference data directory
        sr: Sample rate for audio loading

    Returns:
        Dictionary with class names as keys and mean templates as values
    """
    templates = {}
    classes = ["contact", "edge", "no_contact"]

    for class_name in classes:
        pattern = f"*_{class_name}.wav"
        files = list(static_dir.glob(pattern))
        if not files:
            logger.warning(
                f"No static files found for {class_name} with pattern {pattern}"
            )
            continue

        audios = []
        for f in files:
            try:
                audio = librosa.load(f, sr=sr)[0]
                audios.append(audio)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
                continue

        if not audios:
            logger.warning(f"No valid audio files for {class_name}")
            continue

        # Find minimum length across all recordings
        min_len = min(len(a) for a in audios)
        truncated = [a[:min_len] for a in audios]
        mean_template = np.mean(truncated, axis=0)
        templates[class_name] = mean_template
        logger.info(
            f"Computed template for {class_name}: {len(audios)} samples, length {min_len}"
        )

    return templates


def remove_motion_artifacts(
    audio_data: np.ndarray, labels: np.ndarray, templates: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Apply state-of-the-art VibeCheck motion artifact removal pipeline:
    1. Static reference subtraction
    2. LMS adaptive filtering

    Args:
        audio_data: Array of audio signals (n_samples, signal_length)
        labels: Array of string labels
        templates: Dictionary containing class templates

    Returns:
        Cleaned audio data array
    """
    cleaned_audio = []

    for i, (audio, label) in enumerate(zip(audio_data, labels)):
        if label in templates:
            template = templates[label]

            # VibeCheck Pipeline Stage 1: Static subtraction
            clean1 = _static_subtraction(audio, template)

            # VibeCheck Pipeline Stage 2: LMS adaptive filtering
            clean_final = _lms_adaptive_filter(clean1, template)

            # Pad back to original length if necessary
            if len(clean_final) < len(audio):
                clean_final = np.pad(
                    clean_final, (0, len(audio) - len(clean_final)), "constant"
                )

            cleaned_audio.append(clean_final.astype(np.float32))
        else:
            # No template for this label, keep unchanged
            logger.warning(f"No template found for label {label}, keeping unchanged")
            cleaned_audio.append(audio)

    num_processed = sum(1 for label in labels if label in templates)
    logger.info(
        f"Applied VibeCheck motion artifact removal to {num_processed}/{len(labels)} recordings"
    )

    return np.array(cleaned_audio)


def apply_motion_artifact_removal(
    audio_data: np.ndarray, labels: np.ndarray, static_dir: Path, sr: int = 48000
) -> np.ndarray:
    """
    Convenience function to apply motion artifact removal in one step.

    Args:
        audio_data: Array of audio signals (n_samples, signal_length)
        labels: Array of string labels
        static_dir: Path to static reference directory
        sr: Sample rate

    Returns:
        Cleaned audio data array
    """
    templates = compute_static_templates(static_dir, sr)
    if not templates:
        logger.warning("No templates computed, returning original data")
        return audio_data
    return remove_motion_artifacts(audio_data, labels, templates)
