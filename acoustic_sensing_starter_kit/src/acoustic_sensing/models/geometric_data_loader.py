"""
Geometric Data Loader for Acoustic Sensing Analysis
==================================================

This module loads recorded WAV files from the acoustic sensing experiments
and prepares them for geometric discrimination analysis. It handles:

1. Loading WAV files from multiple experimental batches
2. Extracting contact position labels (tip, middle, base, blank)
3. Organizing data for t-SNE, PCA, and discrimination analysis
4. Batch management and cross-batch analysis

Compatible with data recorded using A_record.py script.

Author: Enhanced for geometric discrimination analysis
"""

import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
from tqdm import tqdm


class GeometricDataLoader:
    """
    Data loader for acoustic sensing geometric discrimination experiments.

    Loads WAV files from recorded experimental batches and organizes them
    for analysis of geometric discrimination capability.
    """

    def __init__(self, base_dir: str = "../data", sr: int = 48000):
        """
        Initialize the data loader.

        Args:
            base_dir: Base directory containing experimental data
            sr: Sample rate for audio loading
        """
        self.base_dir = Path(base_dir)
        self.sr = sr
        self.data_cache = {}

    def get_available_batches(self) -> List[str]:
        """Get list of available experimental batches."""
        batches = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and "soft_finger_batch" in item.name:
                batches.append(item.name)
        return sorted(batches)

    def load_batch_data(
        self,
        batch_name: str,
        contact_positions: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load audio data from a specific experimental batch.

        Args:
            batch_name: Name of the batch (e.g., 'soft_finger_batch_1')
            contact_positions: List of contact positions to include
                             Default: ['finger tip', 'finger middle', 'finger bottom', 'finger blank']
            max_samples_per_class: Maximum samples to load per class (for testing)
            verbose: Show loading progress

        Returns:
            audio_data: Array of audio signals (n_samples, signal_length)
            labels: Array of string labels
            metadata: Dictionary with batch information
        """
        if contact_positions is None:
            contact_positions = [
                "finger tip",
                "finger middle",
                "finger bottom",
                "finger blank",
            ]

        batch_path = self.base_dir / batch_name / "data"

        if not batch_path.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_path}")

        audio_data = []
        labels = []
        file_info = []

        # Get all WAV files
        wav_files = list(batch_path.glob("*.wav"))

        if verbose:
            print(f"Loading batch: {batch_name}")
            print(f"Found {len(wav_files)} WAV files")

        # Group files by contact position
        files_by_position = {pos: [] for pos in contact_positions}

        for wav_file in wav_files:
            # Skip the sweep file
            if "sweep" in wav_file.name:
                continue

            # Extract contact position from filename
            # Sort contact_positions by length (descending) to match longer strings first
            # This prevents "contact" from matching "no contact" files
            sorted_positions = sorted(contact_positions, key=len, reverse=True)
            for pos in sorted_positions:
                if pos in wav_file.name:
                    files_by_position[pos].append(wav_file)
                    break

        # Load data with optional sample limiting
        for position, files in files_by_position.items():
            if max_samples_per_class:
                files = files[:max_samples_per_class]

            if verbose:
                print(f"  Loading {len(files)} files for '{position}'")

            for wav_file in tqdm(files, disable=not verbose, desc=f"  {position}"):
                try:
                    # Load audio
                    audio, _ = librosa.load(wav_file, sr=self.sr, mono=True)

                    audio_data.append(audio)
                    labels.append(position)
                    file_info.append(
                        {
                            "filename": wav_file.name,
                            "batch": batch_name,
                            "position": position,
                            "sample_number": wav_file.stem.split("_")[0],
                            "duration": len(audio) / self.sr,
                        }
                    )

                except Exception as e:
                    warnings.warn(f"Failed to load {wav_file}: {e}")
                    continue

        # Convert to arrays
        audio_data = np.array(audio_data, dtype=object)  # Variable length signals
        labels = np.array(labels)

        metadata = {
            "batch_name": batch_name,
            "total_samples": len(audio_data),
            "contact_positions": contact_positions,
            "samples_per_position": {
                pos: np.sum(labels == pos) for pos in contact_positions
            },
            "sample_rate": self.sr,
            "file_info": file_info,
        }

        if verbose:
            print(f"Loaded {len(audio_data)} samples")
            for pos in contact_positions:
                count = metadata["samples_per_position"][pos]
                print(f"  {pos}: {count} samples")

        return audio_data, labels, metadata

    def load_multiple_batches(
        self,
        batch_names: Optional[List[str]] = None,
        contact_positions: Optional[List[str]] = None,
        max_samples_per_class: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load data from multiple experimental batches.

        Args:
            batch_names: List of batch names to load. If None, loads all available batches
            contact_positions: List of contact positions to include
            max_samples_per_class: Maximum samples per class per batch
            verbose: Show loading progress

        Returns:
            audio_data: Combined audio data from all batches
            labels: Combined labels with batch information
            metadata: Combined metadata
        """
        if batch_names is None:
            batch_names = self.get_available_batches()

        if contact_positions is None:
            contact_positions = [
                "finger tip",
                "finger middle",
                "finger bottom",
                "finger blank",
            ]

        all_audio_data = []
        all_labels = []
        all_metadata = {
            "batches": {},
            "total_samples": 0,
            "contact_positions": contact_positions,
            "sample_rate": self.sr,
        }

        for batch_name in batch_names:
            try:
                audio_data, labels, metadata = self.load_batch_data(
                    batch_name, contact_positions, max_samples_per_class, verbose
                )

                all_audio_data.extend(audio_data)
                all_labels.extend(labels)
                all_metadata["batches"][batch_name] = metadata

            except Exception as e:
                warnings.warn(f"Failed to load batch {batch_name}: {e}")
                continue

        # Convert to arrays
        all_audio_data = np.array(all_audio_data, dtype=object)
        all_labels = np.array(all_labels)

        all_metadata["total_samples"] = len(all_audio_data)
        all_metadata["samples_per_position"] = {
            pos: np.sum(all_labels == pos) for pos in contact_positions
        }

        if verbose:
            print(f"\nCombined dataset:")
            print(f"Total samples: {all_metadata['total_samples']}")
            for pos in contact_positions:
                count = all_metadata["samples_per_position"][pos]
                print(f"  {pos}: {count} samples")

        return all_audio_data, all_labels, all_metadata

    def standardize_audio_length(
        self,
        audio_data: np.ndarray,
        method: str = "pad_truncate",
        target_length: Optional[int] = None,
    ) -> np.ndarray:
        """
        Standardize audio signal lengths for consistent feature extraction.

        Args:
            audio_data: Array of audio signals
            method: Standardization method
                   - 'pad_truncate': Pad short signals, truncate long ones
                   - 'interpolate': Resample to target length
            target_length: Target length in samples. If None, uses median length

        Returns:
            Standardized audio data array
        """
        if target_length is None:
            lengths = [len(audio) for audio in audio_data]
            target_length = int(np.median(lengths))

        standardized_data = []

        for audio in audio_data:
            if method == "pad_truncate":
                if len(audio) < target_length:
                    # Pad with zeros
                    padded = np.zeros(target_length)
                    padded[: len(audio)] = audio
                    standardized_data.append(padded)
                else:
                    # Truncate
                    standardized_data.append(audio[:target_length])

            elif method == "interpolate":
                # Resample to target length
                from scipy import signal

                resampled = signal.resample(audio, target_length)
                standardized_data.append(resampled)

        return np.array(standardized_data)

    def create_cross_batch_labels(
        self, labels: np.ndarray, metadata: Dict
    ) -> np.ndarray:
        """
        Create labels that include batch information for cross-batch analysis.

        Args:
            labels: Original position labels
            metadata: Metadata containing batch information

        Returns:
            Enhanced labels with batch information
        """
        enhanced_labels = []

        sample_idx = 0
        for batch_name, batch_info in metadata["batches"].items():
            batch_samples = batch_info["total_samples"]

            for i in range(batch_samples):
                original_label = labels[sample_idx + i]
                enhanced_label = f"{batch_name}_{original_label}"
                enhanced_labels.append(enhanced_label)

            sample_idx += batch_samples

        return np.array(enhanced_labels)

    def get_position_mapping(self) -> Dict[str, str]:
        """Get mapping from full labels to simplified position names."""
        return {
            "finger tip": "tip",
            "finger middle": "middle",
            "finger bottom": "base",
            "finger blank": "none",
        }

    def simplify_labels(self, labels: np.ndarray) -> np.ndarray:
        """Convert full position names to simplified labels."""
        mapping = self.get_position_mapping()
        return np.array([mapping.get(label, label) for label in labels])


# Utility functions for data analysis
def print_dataset_summary(audio_data: np.ndarray, labels: np.ndarray, metadata: Dict):
    """Print a comprehensive summary of the loaded dataset."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Total samples: {len(audio_data)}")
    print(f"Sample rate: {metadata['sample_rate']} Hz")

    # Audio characteristics
    lengths = [len(audio) for audio in audio_data]
    print(
        f"Audio duration: {np.mean(lengths)/metadata['sample_rate']:.3f} ± {np.std(lengths)/metadata['sample_rate']:.3f} seconds"
    )
    print(f"Signal length: {np.mean(lengths):.0f} ± {np.std(lengths):.0f} samples")

    # Class distribution
    print("\nClass distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = count / len(labels) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")

    # Batch information
    if "batches" in metadata:
        print(f"\nBatches: {len(metadata['batches'])}")
        for batch_name, batch_info in metadata["batches"].items():
            print(f"  {batch_name}: {batch_info['total_samples']} samples")


if __name__ == "__main__":
    # Example usage
    loader = GeometricDataLoader()

    # Load single batch for testing
    print("Available batches:", loader.get_available_batches())

    # Load first batch
    if loader.get_available_batches():
        batch_name = loader.get_available_batches()[0]
        audio_data, labels, metadata = loader.load_batch_data(
            batch_name, max_samples_per_class=50  # Limit for testing
        )

        print_dataset_summary(audio_data, labels, metadata)
