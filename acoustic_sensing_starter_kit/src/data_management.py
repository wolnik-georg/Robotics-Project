"""
Data Management and Loading System for Acoustic Geometric Discrimination
========================================================================

This module provides robust data loading and management for acoustic sensing
experiments with multiple experimental conditions:

- Signal types: sweep, pulse, white_noise
- Touch locations: tip, middle, base
- Geometries: flat, void_5mm, edge_45°, recess_2mm
- Forces: 3N, 6N, 9N

Features:
- Intelligent filename parsing
- Metadata extraction and validation
- Batch loading with progress tracking
- Data organization and filtering
- Experimental condition handling
- Cache management for large datasets

Author: Enhanced for geometric discrimination experiments
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import librosa
import json
import warnings
from dataclasses import dataclass, asdict
from collections import defaultdict
import glob
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for a single acoustic experiment."""

    filename: str
    filepath: str
    sample_id: str
    signal_type: Optional[str] = None
    touch_location: Optional[str] = None
    geometry: Optional[str] = None
    force: Optional[str] = None
    frequency: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetadata":
        """Create from dictionary."""
        return cls(**data)


class FilenameParser:
    """Intelligent parser for acoustic sensing filenames."""

    def __init__(self):
        """Initialize parser with common patterns."""
        # Common filename patterns
        self.patterns = {
            # Standard format: ID_condition.wav
            "standard": re.compile(r"(\d+)_(.+)\.wav$"),
            # Frequency format: ID_frequency.wav
            "frequency": re.compile(r"(\d+)_(\d+)\s*Hz\.wav$"),
            # Complex format: ID_signal_location_geometry_force.wav
            "complex": re.compile(r"(\d+)_(.+?)_(.+?)_(.+?)_(.+?)\.wav$"),
            # Sweep format: ID_sweep.wav
            "sweep": re.compile(r"(\d+)_(long_sweep|sweep)\.wav$"),
        }

        # Signal type mapping
        self.signal_types = {
            "sweep": "sweep",
            "long_sweep": "sweep",
            "pulse": "pulse",
            "burst": "pulse",
            "noise": "white_noise",
            "white_noise": "white_noise",
        }

        # Touch location mapping
        self.touch_locations = {
            "tip": "tip",
            "finger_tip": "tip",
            "middle": "middle",
            "finger_middle": "middle",
            "base": "base",
            "bottom": "base",
            "finger_bottom": "base",
            "finger_base": "base",
            "back": "base",
            "blank": "no_contact",
            "no_tap": "no_contact",
            "none": "no_contact",
            "void": "no_contact",
            "no contact": "no_contact",
        }

        # Geometry mapping
        self.geometries = {
            "flat": "flat",
            "contact": "flat",
            "edge": "edge_45°",
            "edge_45": "edge_45°",
            "void": "void_5mm",
            "void_5mm": "void_5mm",
            "void_1mm": "void_1mm",
            "void_10mm": "void_10mm",
            "recess": "recess_2mm",
            "recess_2mm": "recess_2mm",
            "recess_1mm": "recess_1mm",
            "recess_5mm": "recess_5mm",
            "no contact": "air_gap",
            "no_contact": "air_gap",
        }

        # Force mapping (extract from filenames if present)
        self.forces = {
            "3n": "3N",
            "6n": "6N",
            "9n": "9N",
            "low": "3N",
            "medium": "6N",
            "med": "6N",
            "high": "9N",
        }

    def parse_filename(self, filename: str) -> ExperimentMetadata:
        """
        Parse filename to extract experimental metadata.

        Args:
            filename: Audio filename to parse

        Returns:
            ExperimentMetadata object with extracted information
        """
        filepath = filename if os.path.isabs(filename) else os.path.basename(filename)
        basename = os.path.basename(filename)

        # Initialize metadata
        metadata = ExperimentMetadata(
            filename=basename,
            filepath=filepath,
            sample_id="unknown",
            additional_info={},
        )

        # Try different parsing patterns
        for pattern_name, pattern in self.patterns.items():
            match = pattern.match(basename)
            if match:
                metadata.sample_id = match.group(1)

                if pattern_name == "frequency":
                    # Frequency-specific file (e.g., "1_500 Hz.wav")
                    frequency = float(match.group(2))
                    metadata.frequency = frequency
                    metadata.signal_type = "pulse"  # Assume single-frequency is pulse

                elif pattern_name == "sweep":
                    # Sweep file
                    metadata.signal_type = "sweep"

                elif pattern_name in ["standard", "complex"]:
                    # Parse the condition part
                    condition_parts = (
                        match.group(2).split("_")
                        if pattern_name == "standard"
                        else match.groups()[1:]
                    )
                    self._parse_condition_parts(metadata, condition_parts)

                break

        # Fallback: try to extract info from any underscore-separated parts
        if metadata.signal_type is None and metadata.touch_location is None:
            parts = basename.replace(".wav", "").split("_")[
                1:
            ]  # Skip first part (sample ID)
            self._parse_condition_parts(metadata, parts)

        return metadata

    def _parse_condition_parts(
        self, metadata: ExperimentMetadata, parts: List[str]
    ) -> None:
        """Parse condition parts and update metadata."""
        for part in parts:
            part_lower = part.lower().strip()

            # Check for signal type
            if part_lower in self.signal_types:
                metadata.signal_type = self.signal_types[part_lower]
                continue

            # Check for touch location
            if part_lower in self.touch_locations:
                metadata.touch_location = self.touch_locations[part_lower]
                continue

            # Check for geometry
            if part_lower in self.geometries:
                metadata.geometry = self.geometries[part_lower]
                continue

            # Check for force
            if part_lower in self.forces:
                metadata.force = self.forces[part_lower]
                continue

            # Check for frequency (Hz in the name)
            freq_match = re.search(r"(\d+)\s*hz", part_lower)
            if freq_match:
                metadata.frequency = float(freq_match.group(1))
                if metadata.signal_type is None:
                    metadata.signal_type = "pulse"
                continue

            # Check for force with unit (e.g., "3n", "6n")
            force_match = re.search(r"(\d+)n", part_lower)
            if force_match:
                metadata.force = f"{force_match.group(1)}N"
                continue

            # Store unrecognized parts in additional_info
            if part_lower not in ["wav", ""]:
                if "unrecognized" not in metadata.additional_info:
                    metadata.additional_info["unrecognized"] = []
                metadata.additional_info["unrecognized"].append(part)


class AcousticDataLoader:
    """
    Comprehensive data loader for acoustic sensing experiments.

    Supports multiple data directories, intelligent caching, and flexible
    filtering based on experimental conditions.
    """

    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        sr: int = 48000,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize the data loader.

        Args:
            data_dirs: Directory or list of directories containing audio data
            sr: Sample rate for audio loading
            cache_dir: Directory for caching processed data
            use_cache: Whether to use caching for faster loading
        """
        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        self.sr = sr
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".acoustic_cache")

        # Initialize components
        self.parser = FilenameParser()
        self.metadata_cache = {}
        self.audio_cache = {}

        # Create cache directory
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Scan and index all available files
        self.file_index = self._build_file_index()
        logger.info(
            f"Found {len(self.file_index)} audio files across {len(self.data_dirs)} directories"
        )

    def _build_file_index(self) -> List[ExperimentMetadata]:
        """Build index of all audio files and their metadata."""
        file_index = []

        for data_dir in self.data_dirs:
            if not os.path.exists(data_dir):
                logger.warning(f"Data directory not found: {data_dir}")
                continue

            # Find all WAV files
            wav_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(".wav"):
                        filepath = os.path.join(root, file)
                        wav_files.append(filepath)

            # Parse metadata for each file
            for filepath in wav_files:
                try:
                    metadata = self.parser.parse_filename(filepath)
                    metadata.filepath = filepath
                    file_index.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to parse {filepath}: {e}")

        return file_index

    def get_file_summary(self) -> pd.DataFrame:
        """Get summary of all indexed files as DataFrame."""
        data = []
        for metadata in self.file_index:
            data.append(metadata.to_dict())

        df = pd.DataFrame(data)
        return df

    def filter_files(
        self,
        signal_type: Optional[Union[str, List[str]]] = None,
        touch_location: Optional[Union[str, List[str]]] = None,
        geometry: Optional[Union[str, List[str]]] = None,
        force: Optional[Union[str, List[str]]] = None,
        frequency_range: Optional[Tuple[float, float]] = None,
    ) -> List[ExperimentMetadata]:
        """
        Filter files based on experimental conditions.

        Args:
            signal_type: Signal type(s) to include
            touch_location: Touch location(s) to include
            geometry: Geometry type(s) to include
            force: Force level(s) to include
            frequency_range: Frequency range (min, max) to include

        Returns:
            List of metadata for filtered files
        """
        filtered = self.file_index.copy()

        # Apply filters
        if signal_type is not None:
            signal_types = (
                [signal_type] if isinstance(signal_type, str) else signal_type
            )
            filtered = [f for f in filtered if f.signal_type in signal_types]

        if touch_location is not None:
            touch_locations = (
                [touch_location] if isinstance(touch_location, str) else touch_location
            )
            filtered = [f for f in filtered if f.touch_location in touch_locations]

        if geometry is not None:
            geometries = [geometry] if isinstance(geometry, str) else geometry
            filtered = [f for f in filtered if f.geometry in geometries]

        if force is not None:
            forces = [force] if isinstance(force, str) else force
            filtered = [f for f in filtered if f.force in forces]

        if frequency_range is not None:
            min_freq, max_freq = frequency_range
            filtered = [
                f
                for f in filtered
                if f.frequency is not None and min_freq <= f.frequency <= max_freq
            ]

        return filtered

    def load_audio(self, filepath: str) -> np.ndarray:
        """
        Load audio file with caching.

        Args:
            filepath: Path to audio file

        Returns:
            Audio waveform as numpy array
        """
        # Check cache first
        if self.use_cache and filepath in self.audio_cache:
            return self.audio_cache[filepath]

        # Load audio
        try:
            audio, _ = librosa.load(filepath, sr=self.sr)

            # Cache if enabled
            if self.use_cache:
                self.audio_cache[filepath] = audio

            return audio
        except Exception as e:
            logger.error(f"Failed to load audio from {filepath}: {e}")
            raise

    def load_batch(
        self,
        metadata_list: Optional[List[ExperimentMetadata]] = None,
        max_files: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tuple[List[np.ndarray], List[ExperimentMetadata]]:
        """
        Load batch of audio files.

        Args:
            metadata_list: List of metadata for files to load (if None, loads all)
            max_files: Maximum number of files to load
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (audio_list, metadata_list)
        """
        if metadata_list is None:
            metadata_list = self.file_index

        if max_files is not None:
            metadata_list = metadata_list[:max_files]

        audio_list = []
        valid_metadata = []

        iterator = (
            tqdm(metadata_list, desc="Loading audio files")
            if show_progress
            else metadata_list
        )

        for metadata in iterator:
            try:
                audio = self.load_audio(metadata.filepath)
                audio_list.append(audio)
                valid_metadata.append(metadata)
            except Exception as e:
                logger.warning(f"Skipping {metadata.filename}: {e}")

        logger.info(
            f"Successfully loaded {len(audio_list)} out of {len(metadata_list)} files"
        )
        return audio_list, valid_metadata

    def get_condition_summary(self) -> Dict[str, Any]:
        """Get summary of experimental conditions in dataset."""
        summary = {
            "total_files": len(self.file_index),
            "signal_types": defaultdict(int),
            "touch_locations": defaultdict(int),
            "geometries": defaultdict(int),
            "forces": defaultdict(int),
            "frequencies": [],
        }

        for metadata in self.file_index:
            if metadata.signal_type:
                summary["signal_types"][metadata.signal_type] += 1
            if metadata.touch_location:
                summary["touch_locations"][metadata.touch_location] += 1
            if metadata.geometry:
                summary["geometries"][metadata.geometry] += 1
            if metadata.force:
                summary["forces"][metadata.force] += 1
            if metadata.frequency:
                summary["frequencies"].append(metadata.frequency)

        # Convert defaultdicts to regular dicts
        for key in ["signal_types", "touch_locations", "geometries", "forces"]:
            summary[key] = dict(summary[key])

        # Frequency statistics
        if summary["frequencies"]:
            freq_array = np.array(summary["frequencies"])
            summary["frequency_stats"] = {
                "min": float(np.min(freq_array)),
                "max": float(np.max(freq_array)),
                "mean": float(np.mean(freq_array)),
                "unique_count": len(np.unique(freq_array)),
            }
        else:
            summary["frequency_stats"] = None

        return summary

    def save_metadata(self, filepath: str) -> None:
        """Save metadata index to JSON file."""
        data = [metadata.to_dict() for metadata in self.file_index]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved metadata for {len(self.file_index)} files to {filepath}")

    def load_metadata(self, filepath: str) -> None:
        """Load metadata index from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.file_index = [ExperimentMetadata.from_dict(item) for item in data]
        logger.info(f"Loaded metadata for {len(self.file_index)} files from {filepath}")


class ExperimentDataset:
    """
    High-level dataset interface for acoustic sensing experiments.

    Provides organized access to data by experimental conditions with
    automatic train/test splits and cross-validation support.
    """

    def __init__(self, data_dirs: Union[str, List[str]], **loader_kwargs):
        """
        Initialize experiment dataset.

        Args:
            data_dirs: Data directories
            **loader_kwargs: Additional arguments for data loader
        """
        self.loader = AcousticDataLoader(data_dirs, **loader_kwargs)
        self._data_cache = {}

    def get_geometric_discrimination_data(
        self,
        geometries: List[str],
        signal_type: str = "sweep",
        balance_classes: bool = True,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get data specifically for geometric discrimination analysis.

        Args:
            geometries: List of geometry types to include
            signal_type: Signal type to use
            balance_classes: Whether to balance class sizes

        Returns:
            Tuple of (audio_list, labels)
        """
        audio_list = []
        labels = []

        for geometry in geometries:
            # Filter files for this geometry
            filtered_metadata = self.loader.filter_files(
                geometry=geometry, signal_type=signal_type
            )

            if not filtered_metadata:
                logger.warning(f"No files found for geometry: {geometry}")
                continue

            # Load audio for this geometry
            geometry_audio, _ = self.loader.load_batch(filtered_metadata)

            # Balance classes if requested
            if balance_classes and len(geometry_audio) > 0:
                min_samples = min(
                    [
                        len(
                            self.loader.filter_files(
                                geometry=g, signal_type=signal_type
                            )
                        )
                        for g in geometries
                        if self.loader.filter_files(geometry=g, signal_type=signal_type)
                    ]
                )
                geometry_audio = geometry_audio[:min_samples]

            audio_list.extend(geometry_audio)
            labels.extend([geometry] * len(geometry_audio))

        return audio_list, labels

    def print_dataset_summary(self) -> None:
        """Print comprehensive dataset summary."""
        summary = self.loader.get_condition_summary()

        print("🔍 ACOUSTIC SENSING DATASET SUMMARY")
        print("=" * 50)
        print(f"Total Files: {summary['total_files']}")
        print()

        if summary["signal_types"]:
            print("📡 Signal Types:")
            for signal_type, count in summary["signal_types"].items():
                print(f"  {signal_type}: {count} files")
            print()

        if summary["touch_locations"]:
            print("👆 Touch Locations:")
            for location, count in summary["touch_locations"].items():
                print(f"  {location}: {count} files")
            print()

        if summary["geometries"]:
            print("🔺 Geometries:")
            for geometry, count in summary["geometries"].items():
                print(f"  {geometry}: {count} files")
            print()

        if summary["forces"]:
            print("⚡ Forces:")
            for force, count in summary["forces"].items():
                print(f"  {force}: {count} files")
            print()

        if summary["frequency_stats"]:
            print("🌊 Frequency Information:")
            stats = summary["frequency_stats"]
            print(f"  Range: {stats['min']:.0f} - {stats['max']:.0f} Hz")
            print(f"  Mean: {stats['mean']:.1f} Hz")
            print(f"  Unique frequencies: {stats['unique_count']}")


# Convenience functions for backward compatibility
def load_audio(file_path: str, sr: int = 48000) -> np.ndarray:
    """Load audio file (backward compatibility)."""
    return librosa.load(file_path, sr=sr)[0]


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    data_dirs = [
        "data/diverse_frequency_sensing_demo/data",
        "data/soft_finger_batch_3/data",
        "data/material_tapping_demo/data",
    ]

    # Create dataset
    dataset = ExperimentDataset(data_dirs)

    # Print summary
    dataset.print_dataset_summary()

    # Example: Get data for geometric discrimination
    geometries = ["flat", "edge_45°", "void_5mm"]
    audio_list, labels = dataset.get_geometric_discrimination_data(geometries)

    print(f"\n🎯 Geometric Discrimination Data:")
    print(f"Total samples: {len(audio_list)}")
    for geometry in geometries:
        count = labels.count(geometry)
        print(f"  {geometry}: {count} samples")
