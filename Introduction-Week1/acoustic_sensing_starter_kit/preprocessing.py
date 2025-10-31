"""
Preprocessing utilities for the Acoustic Sensing ML Pipeline.

This module provides functions for loading audio and extracting features,
ensuring consistency across training and inference.
"""

import librosa
import numpy as np
import pandas

SR = 48000  # Default sample rate


def load_audio(file_path, sr=SR):
    """
    Load audio from file.

    Args:
        file_path (str): Path to the audio file.
        sr (int): Sample rate.

    Returns:
        np.ndarray: Audio waveform.
    """
    return librosa.load(file_path, sr=sr)[0]


def audio_to_features(audio, method="stft", n_fft=4096, in_dB=False, sr=SR):
    """
    Extract features from audio waveform.

    Args:
        audio (np.ndarray): Audio waveform.
        method (str): Feature extraction method ('stft' for now).
        n_fft (int): FFT size for STFT.
        in_dB (bool): Convert to dB scale.
        sr (int): Sample rate.

    Returns:
        pandas.Series: Feature series with frequency index.
    """
    if method == "stft":
        spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft))
        features = spectrogram.sum(axis=1)
        if in_dB:
            features = librosa.amplitude_to_db(features, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        index = pandas.Index(freqs)
        return pandas.Series(features, index=index)
    else:
        raise ValueError(f"Unsupported method: {method}")
