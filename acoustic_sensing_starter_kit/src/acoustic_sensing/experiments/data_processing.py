from .base_experiment import BaseExperiment
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os
from scipy.signal import butter, filtfilt  # Add for audio smoothing
from pathlib import Path
import librosa
from acoustic_sensing.core.motion_artifact_removal import apply_motion_artifact_removal
from acoustic_sensing.core.feature_extraction import HAS_GPU_SUPPORT


class PerSampleNormalizer:
    """
    Per-sample feature normalization for improved cross-workspace generalization.

    Unlike global normalization (StandardScaler), this normalizes each sample
    independently, making features invariant to:
    - Recording gain/volume differences
    - Background noise levels
    - Microphone sensitivity variations

    Methods:
    - 'zscore': Per-sample z-score (mean=0, std=1 for each sample)
    - 'minmax': Per-sample min-max scaling (0-1 range for each sample)
    - 'robust': Per-sample robust scaling using median and IQR
    - 'l2': L2 normalization (unit norm for each sample)
    """

    def __init__(self, method: str = "zscore", epsilon: float = 1e-10):
        """
        Initialize the per-sample normalizer.

        Args:
            method: Normalization method ('zscore', 'minmax', 'robust', 'l2')
            epsilon: Small value to avoid division by zero
        """
        self.method = method.lower()
        self.epsilon = epsilon

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize a single feature vector (per-sample).

        Args:
            features: 1D numpy array of features for one sample

        Returns:
            Normalized feature vector
        """
        features = np.asarray(features, dtype=np.float64)

        if self.method == "zscore":
            # Per-sample z-score normalization
            mean = np.mean(features)
            std = np.std(features)
            if std < self.epsilon:
                std = 1.0
            return (features - mean) / std

        elif self.method == "minmax":
            # Per-sample min-max normalization to [0, 1]
            min_val = np.min(features)
            max_val = np.max(features)
            range_val = max_val - min_val
            if range_val < self.epsilon:
                range_val = 1.0
            return (features - min_val) / range_val

        elif self.method == "robust":
            # Per-sample robust normalization using median and IQR
            median = np.median(features)
            q75, q25 = np.percentile(features, [75, 25])
            iqr = q75 - q25
            if iqr < self.epsilon:
                iqr = 1.0
            return (features - median) / iqr

        elif self.method == "l2":
            # L2 normalization (unit norm)
            norm = np.linalg.norm(features)
            if norm < self.epsilon:
                norm = 1.0
            return features / norm

        else:
            # No normalization (pass through)
            return features

    def normalize_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize a batch of samples (each row is normalized independently).

        Args:
            X: 2D numpy array of shape (n_samples, n_features)

        Returns:
            Normalized array of same shape
        """
        return np.array([self.normalize(row) for row in X])


class AudioAugmenter:
    """
    Audio data augmentation for improving model robustness.

    Applies various transformations to training audio to help models
    generalize better across different workspaces and recording conditions.
    """

    def __init__(self, sr: int = 48000, random_state: int = 42, enhanced: bool = False):
        self.sr = sr
        self.rng = np.random.RandomState(random_state)
        self.enhanced = enhanced  # Use stronger augmentation techniques

    def augment(self, audio: np.ndarray, augment_type: str = "all") -> np.ndarray:
        """
        Apply augmentation to audio signal.

        Args:
            audio: Input audio signal
            augment_type: Type of augmentation ('noise', 'time_shift', 'pitch', 'gain', 'time_stretch', 'all')

        Returns:
            Augmented audio signal
        """
        audio = audio.astype(np.float32)

        if augment_type == "all":
            # Apply random subset of augmentations
            augmented = audio.copy()
            if self.rng.random() < 0.5:
                augmented = self._add_noise(augmented)
            if self.rng.random() < 0.5:
                augmented = self._time_shift(augmented)
            if self.rng.random() < 0.4:  # Increased probability for pitch
                augmented = self._pitch_shift(augmented)
            if self.rng.random() < 0.5:
                augmented = self._gain_variation(augmented)
            if self.enhanced and self.rng.random() < 0.4:  # NEW: Time stretch
                augmented = self._time_stretch(augmented)
            return augmented
        elif augment_type == "noise":
            return self._add_noise(audio)
        elif augment_type == "time_shift":
            return self._time_shift(audio)
        elif augment_type == "pitch":
            return self._pitch_shift(audio)
        elif augment_type == "gain":
            return self._gain_variation(audio)
        elif augment_type == "time_stretch":
            return self._time_stretch(audio)
        else:
            return audio

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at random SNR."""
        # SNR between 20-40 dB (subtle noise)
        snr_db = self.rng.uniform(20, 40)

        signal_power = np.mean(audio**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), len(audio))

        return (audio + noise).astype(np.float32)

    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Shift audio in time (circular shift)."""
        # Shift by up to 5% of signal length
        max_shift = int(0.05 * len(audio))
        shift = self.rng.randint(-max_shift, max_shift)

        return np.roll(audio, shift).astype(np.float32)

    def _pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Slightly shift pitch (simulates different materials/tensions)."""
        # Enhanced: Shift by up to ±3 semitones for more diversity
        n_steps = self.rng.uniform(
            -3 if self.enhanced else -2, 3 if self.enhanced else 2
        )

        try:
            shifted = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
            return shifted.astype(np.float32)
        except Exception:
            return audio

    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """
        Time stretch audio without changing pitch.
        Simulates different contact speeds/durations.
        """
        # Stretch rate between 0.85x and 1.15x (more aggressive if enhanced)
        if self.enhanced:
            rate = self.rng.uniform(0.85, 1.15)
        else:
            rate = self.rng.uniform(0.9, 1.1)

        try:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            # Ensure same length by padding or trimming
            if len(stretched) > len(audio):
                stretched = stretched[: len(audio)]
            elif len(stretched) < len(audio):
                stretched = np.pad(
                    stretched, (0, len(audio) - len(stretched)), mode="constant"
                )
            return stretched.astype(np.float32)
        except Exception:
            return audio

    def _gain_variation(self, audio: np.ndarray) -> np.ndarray:
        """Apply random gain (simulates different recording levels)."""
        # Gain variation of ±3 dB
        gain_db = self.rng.uniform(-3, 3)
        gain = 10 ** (gain_db / 20)

        return (audio * gain).astype(np.float32)

    def _frequency_mask(self, audio: np.ndarray) -> np.ndarray:
        """Apply random frequency masking (simulates room acoustics)."""
        # Apply a random bandpass filter
        from scipy.signal import butter, filtfilt

        # Random cutoff frequencies
        low_cut = self.rng.uniform(50, 200)
        high_cut = self.rng.uniform(8000, 15000)

        try:
            nyquist = self.sr / 2
            low = low_cut / nyquist
            high = high_cut / nyquist
            b, a = butter(2, [low, high], btype="band")
            filtered = filtfilt(b, a, audio)
            return filtered.astype(np.float32)
        except Exception:
            return audio


class DataProcessingExperiment(BaseExperiment):
    """
    Experiment for loading and preprocessing acoustic sensing data.
    This serves as the foundation for all other experiments.
    """

    def get_dependencies(self) -> List[str]:
        """No dependencies - this is the base experiment."""
        return []

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and preprocess data for all experiments.

        Args:
            shared_data: Empty initially

        Returns:
            Dictionary containing loaded features, labels, and metadata
        """
        self.logger.info("Starting data processing experiment...")

        # Import the necessary modules from the existing codebase
        import sys

        sys.path.append(
            "/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/src"
        )

        from acoustic_sensing.analysis.batch_analysis import BatchSpecificAnalyzer
        from acoustic_sensing.models.geometric_data_loader import GeometricDataLoader
        from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor

        # Initialize the analyzer and data loader exactly like the working code
        base_data_dir = self.config.get("base_data_dir", "data")
        analyzer = BatchSpecificAnalyzer(base_dir=base_data_dir)
        data_loader = GeometricDataLoader(base_dir=base_data_dir, sr=48000)

        # Check if multi-dataset mode is enabled
        multi_dataset_config = self.config.get("multi_dataset_training", {})
        multi_dataset_enabled = multi_dataset_config.get("enabled", False)

        # Get validation datasets from config (supports both single and multiple datasets)
        validation_datasets_config = self.config.get("validation_datasets", [])
        if not validation_datasets_config:
            validation_datasets_config = []
        elif isinstance(validation_datasets_config, str):
            # Convert single string to list
            validation_datasets_config = [validation_datasets_config]

        if multi_dataset_enabled:
            # Multi-dataset mode: process training datasets + optional validation dataset
            self.logger.info("🔄 MULTI-DATASET MODE ENABLED")
            training_datasets = multi_dataset_config.get("training_datasets", [])
            validation_dataset = multi_dataset_config.get("validation_dataset", None)

            if not training_datasets:
                raise ValueError(
                    "Multi-dataset mode requires 'training_datasets' to be specified in config"
                )

            # Combine all datasets to process
            if validation_dataset:
                all_datasets = training_datasets + [validation_dataset]
                self.logger.info(f"Training datasets: {training_datasets}")
                self.logger.info(f"Validation dataset: {validation_dataset}")
                # Store validation info for later use
                validation_datasets_config = [validation_dataset]
            else:
                all_datasets = training_datasets
                self.logger.info(f"Training datasets: {training_datasets}")
                self.logger.info("No validation dataset - will use train/test split")
                validation_datasets_config = []

            available_batches = all_datasets
        else:
            # Standard mode: Use datasets from config
            self.logger.info("📊 STANDARD DATASET MODE")

            # Get datasets from config
            datasets_from_config = self.config.get("datasets", [])

            if datasets_from_config:
                # IMPORTANT: Load ALL datasets (training + validation)
                # Combine datasets and validation_datasets to get all batches to load
                available_batches = datasets_from_config.copy()

                # Add validation datasets if specified
                if validation_datasets_config:
                    # Add validation datasets to the list of batches to load
                    for val_dataset in validation_datasets_config:
                        if val_dataset not in available_batches:
                            available_batches.append(val_dataset)

                    self.logger.info(
                        f"✓ Validation datasets specified: {validation_datasets_config}"
                    )
                    self.logger.info(f"  Training datasets: {datasets_from_config}")
                    self.logger.info(f"  All datasets to load: {available_batches}")
                else:
                    self.logger.info(
                        "✓ No validation datasets - will combine all datasets for train/test split"
                    )
                    self.logger.info(
                        f"Using datasets from config: {datasets_from_config}"
                    )
            else:
                # Fallback: scan directory (legacy behavior)
                self.logger.info(
                    "No datasets specified in config, scanning directory..."
                )
                data_dir_path = f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}"
                available_batches = [
                    d
                    for d in os.listdir(data_dir_path)
                    if (
                        d
                        == "balanced_collected_data_runs_2026_01_14_workspace_3_v1_undersample"
                    )
                    and os.path.isdir(os.path.join(data_dir_path, d))
                ]

        self.logger.info(f"Found {len(available_batches)} batches: {available_batches}")

        # Process each batch separately and store results
        batch_results = {}

        for batch_name in available_batches:
            try:
                self.logger.info(f"Processing {batch_name}...")

                # Use the same approach as the working batch_analysis.py
                # Detect actual classes for this batch
                actual_classes = analyzer.detect_actual_classes(batch_name)

                if not actual_classes:
                    self.logger.warning(f"No classes detected for {batch_name}")
                    continue

                # Load batch data using the same method that works
                audio_data, labels, metadata = data_loader.load_batch_data(
                    batch_name,
                    contact_positions=actual_classes,
                    max_samples_per_class=None,
                    verbose=False,  # Keep it quiet for modular execution
                )

                if len(audio_data) > 0:
                    # Apply motion artifact removal if enabled (disabled by default)
                    motion_removal_config = self.config.get(
                        "motion_artifact_removal", {"enabled": False}
                    )
                    if motion_removal_config.get("enabled", False):
                        self.logger.info("Motion artifact removal is ENABLED")
                        static_dir = Path(
                            f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}/collected_data_runs_validate_2025_12_16_v3_static_reference/data"
                        )

                        # Check if static reference directory exists
                        if not static_dir.exists():
                            self.logger.warning(
                                f"Static reference directory not found: {static_dir}"
                            )
                            self.logger.warning("Skipping motion artifact removal")
                        else:
                            self.logger.info("Applying motion artifact removal...")
                            audio_data = apply_motion_artifact_removal(
                                audio_data, np.array(labels), static_dir, sr=48000
                            )
                            self.logger.info("Motion artifact removal completed")

                            # Create motion artifact removal comparison plots
                            self._create_motion_artifact_comparison_plots(
                                audio_data,
                                labels,
                                static_dir,
                                batch_name,
                                self.experiment_output_dir,
                            )
                    else:
                        self.logger.info(
                            "Motion artifact removal is DISABLED (default)"
                        )

                    self.logger.info(
                        f"Extracting features from {len(audio_data)} audio samples..."
                    )

                    # Check for optional audio smoothing
                    apply_smoothing = (
                        False  # Disabled by default to preserve frequency sweeps
                    )
                    if apply_smoothing:
                        self.logger.info(
                            "Audio smoothing enabled - applying high-pass filter"
                        )
                        cutoff_freq = self.config.get("smoothing_cutoff_freq", 500)
                    else:
                        self.logger.info("Audio smoothing disabled")

                    # Feature extraction configuration
                    # Support workspace-invariant features for better cross-workspace generalization
                    use_workspace_invariant = self.config.get(
                        "use_workspace_invariant_features", True  # Default to enabled
                    )
                    if use_workspace_invariant:
                        self.logger.info(
                            "✓ Workspace-invariant features ENABLED (better cross-workspace generalization)"
                        )
                    else:
                        self.logger.info(
                            "✓ Workspace-invariant features DISABLED (using original features only)"
                        )

                    # Support impulse response / transfer function features
                    use_impulse_features = self.config.get(
                        "use_impulse_features",
                        False,  # Default to disabled for backward compatibility
                    )
                    if use_impulse_features:
                        self.logger.info(
                            "✓ Impulse response features ENABLED (transfer function analysis)"
                        )
                    else:
                        self.logger.info("✓ Impulse response features DISABLED")

                    # Support data augmentation for training data
                    use_augmentation = self.config.get(
                        "use_data_augmentation", False  # Default to disabled
                    )
                    augmentation_factor = self.config.get(
                        "augmentation_factor", 2  # How many augmented copies per sample
                    )
                    enhanced_augmentation = self.config.get(
                        "enhanced_augmentation",
                        False,  # Use stronger augmentation (pitch ±3, time stretch)
                    )
                    # Only augment training data, not validation
                    is_validation_batch = batch_name in validation_datasets_config

                    if use_augmentation and not is_validation_batch:
                        aug_mode = "enhanced" if enhanced_augmentation else "standard"
                        self.logger.info(
                            f"✓ Data augmentation ENABLED (mode={aug_mode}, factor={augmentation_factor}x for training)"
                        )
                        augmenter = AudioAugmenter(
                            sr=48000, enhanced=enhanced_augmentation
                        )
                    else:
                        if use_augmentation and is_validation_batch:
                            self.logger.info(
                                "✓ Data augmentation SKIPPED for validation data"
                            )
                        augmenter = None

                    # Get feature extraction mode(s) from config
                    feature_extraction_config = self.config.get(
                        "feature_extraction", {}
                    )

                    # Per-sample feature normalization configuration
                    # This normalizes each sample independently for better workspace generalization
                    normalization_config = feature_extraction_config.get(
                        "normalization", {}
                    )
                    use_per_sample_norm = normalization_config.get("enabled", False)
                    norm_method = normalization_config.get("method", "zscore")

                    if use_per_sample_norm:
                        self.logger.info(
                            f"✓ Per-sample normalization ENABLED (method={norm_method})"
                        )
                        per_sample_normalizer = PerSampleNormalizer(method=norm_method)
                    else:
                        self.logger.info("✓ Per-sample normalization DISABLED")
                        per_sample_normalizer = None

                    # Check if multiple modes are specified
                    extraction_modes = feature_extraction_config.get("modes", None)
                    if extraction_modes is None:
                        # Single mode (backward compatibility)
                        extraction_modes = [
                            feature_extraction_config.get("mode", "features")
                        ]

                    spectrogram_params = feature_extraction_config.get(
                        "spectrogram", {}
                    )

                    # Helper function to filter parameters for each extraction mode
                    def filter_params_for_mode(mode: str, all_params: dict) -> dict:
                        """Filter spectrogram parameters to only include those accepted by each method."""
                        if mode == "spectrogram":
                            # extract_spectrogram accepts: n_fft, hop_length, n_mels, fmin, fmax, time_bins, use_log_scale
                            return {
                                k: v
                                for k, v in all_params.items()
                                if k
                                in [
                                    "n_fft",
                                    "hop_length",
                                    "n_mels",
                                    "fmin",
                                    "fmax",
                                    "time_bins",
                                    "use_log_scale",
                                ]
                            }
                        elif mode == "mfcc":
                            # extract_mfcc accepts: n_mfcc, n_fft, hop_length, n_mels, fmin, fmax
                            return {
                                k: v
                                for k, v in all_params.items()
                                if k
                                in [
                                    "n_mfcc",
                                    "n_fft",
                                    "hop_length",
                                    "n_mels",
                                    "fmin",
                                    "fmax",
                                ]
                            }
                        elif mode in ["magnitude_spectrum", "power_spectrum"]:
                            # extract_magnitude_spectrum and extract_power_spectrum accept: n_fft, hop_length, normalize
                            return {
                                k: v
                                for k, v in all_params.items()
                                if k in ["n_fft", "hop_length", "normalize"]
                            }
                        elif mode == "chroma":
                            # extract_chroma_features accepts: n_chroma, n_fft, hop_length
                            return {
                                k: v
                                for k, v in all_params.items()
                                if k in ["n_chroma", "n_fft", "hop_length"]
                            }
                        else:
                            # For other modes, return empty dict (they don't use spectrogram_params)
                            return {}

                    # Process each mode
                    for extraction_mode in extraction_modes:
                        self.logger.info(
                            f"\n🔄 Processing feature extraction mode: {extraction_mode}"
                        )

                        # Log extraction mode details
                        if extraction_mode == "features":
                            self.logger.info(
                                "✓ Using hand-crafted features (65 dimensions)"
                            )
                        elif extraction_mode == "spectrogram":
                            n_mels = spectrogram_params.get("n_mels", 64)
                            time_bins = spectrogram_params.get("time_bins", 128)
                            self.logger.info(
                                f"✓ Using mel spectrograms ({n_mels}×{time_bins} = {n_mels*time_bins} dimensions)"
                            )
                        elif extraction_mode == "both":
                            n_mels = spectrogram_params.get("n_mels", 64)
                            time_bins = spectrogram_params.get("time_bins", 128)
                            self.logger.info(
                                f"✓ Using BOTH features + FULL spectrograms (65 + {n_mels*time_bins} = {65 + n_mels*time_bins} dimensions)"
                            )
                        elif extraction_mode == "mfcc":
                            n_mfcc = spectrogram_params.get("n_mfcc", 13)
                            time_bins = spectrogram_params.get("time_bins", 128)
                            self.logger.info(
                                f"✓ Using MFCC features ({n_mfcc}×{time_bins} = {n_mfcc*time_bins} dimensions)"
                            )
                        elif extraction_mode == "magnitude_spectrum":
                            n_fft = spectrogram_params.get("n_fft", 2048)
                            time_bins = spectrogram_params.get("time_bins", 128)
                            freq_bins = n_fft // 2 + 1
                            self.logger.info(
                                f"✓ Using magnitude spectrum ({freq_bins}×{time_bins} = {freq_bins*time_bins} dimensions)"
                            )
                        elif extraction_mode == "power_spectrum":
                            n_fft = spectrogram_params.get("n_fft", 2048)
                            time_bins = spectrogram_params.get("time_bins", 128)
                            freq_bins = n_fft // 2 + 1
                            self.logger.info(
                                f"✓ Using power spectrum ({freq_bins}×{time_bins} = {freq_bins*time_bins} dimensions)"
                            )
                        elif extraction_mode == "chroma":
                            n_chroma = spectrogram_params.get("n_chroma", 12)
                            time_bins = spectrogram_params.get("time_bins", 128)
                            self.logger.info(
                                f"✓ Using chroma features ({n_chroma}×{time_bins} = {n_chroma*time_bins} dimensions)"
                            )

                        # Create mode-specific key for storing results
                        mode_key = (
                            f"{batch_name}_{extraction_mode}"
                            if len(extraction_modes) > 1
                            else batch_name
                        )

                        feature_extractor = GeometricFeatureExtractor(
                            sr=48000,
                            use_workspace_invariant=use_workspace_invariant,
                            use_impulse_features=use_impulse_features,
                        )

                        X_feat = []
                        y_labels = []  # Track labels for augmented samples
                        failed_count = 0

                        for i, audio in enumerate(audio_data):
                            try:
                                # Apply smoothing if enabled
                                if apply_smoothing:
                                    audio = self._apply_high_pass_filter(
                                        audio, sr=48000, cutoff=cutoff_freq
                                    )

                                # Extract features based on mode
                                if extraction_mode == "features":
                                    # Hand-crafted features (current default)
                                    features = feature_extractor.extract_features(
                                        audio, method="comprehensive"
                                    )
                                elif extraction_mode == "spectrogram":
                                    # Spectrogram representation
                                    filtered_params = filter_params_for_mode(
                                        extraction_mode, spectrogram_params
                                    )
                                    # Use GPU acceleration if available
                                    if (
                                        hasattr(
                                            feature_extractor, "extract_spectrogram_gpu"
                                        )
                                        and HAS_GPU_SUPPORT
                                    ):
                                        spectrogram = (
                                            feature_extractor.extract_spectrogram_gpu(
                                                audio, **filtered_params
                                            )
                                        )
                                    else:
                                        spectrogram = (
                                            feature_extractor.extract_spectrogram(
                                                audio, **filtered_params
                                            )
                                        )
                                    # Flatten to 1D for compatibility with sklearn models
                                    features = spectrogram.flatten()
                                elif extraction_mode == "mfcc":
                                    # MFCC features
                                    filtered_params = filter_params_for_mode(
                                        extraction_mode, spectrogram_params
                                    )
                                    # Use GPU acceleration if available
                                    if (
                                        hasattr(feature_extractor, "extract_mfcc_gpu")
                                        and HAS_GPU_SUPPORT
                                    ):
                                        mfcc = feature_extractor.extract_mfcc_gpu(
                                            audio, **filtered_params
                                        )
                                    else:
                                        mfcc = feature_extractor.extract_mfcc(
                                            audio, **filtered_params
                                        )
                                    # Flatten to 1D for compatibility with sklearn models
                                    features = mfcc.flatten()
                                elif extraction_mode == "magnitude_spectrum":
                                    # Magnitude spectrum
                                    filtered_params = filter_params_for_mode(
                                        extraction_mode, spectrogram_params
                                    )
                                    # Use GPU acceleration if available
                                    if (
                                        hasattr(
                                            feature_extractor,
                                            "extract_magnitude_spectrum_gpu",
                                        )
                                        and HAS_GPU_SUPPORT
                                    ):
                                        mag_spec = feature_extractor.extract_magnitude_spectrum_gpu(
                                            audio, **filtered_params
                                        )
                                    else:
                                        mag_spec = feature_extractor.extract_magnitude_spectrum(
                                            audio, **filtered_params
                                        )
                                    # Flatten to 1D for compatibility with sklearn models
                                    features = mag_spec.flatten()
                                elif extraction_mode == "power_spectrum":
                                    # Power spectrum
                                    filtered_params = filter_params_for_mode(
                                        extraction_mode, spectrogram_params
                                    )
                                    # Use GPU acceleration if available
                                    if (
                                        hasattr(
                                            feature_extractor,
                                            "extract_power_spectrum_gpu",
                                        )
                                        and HAS_GPU_SUPPORT
                                    ):
                                        power_spec = feature_extractor.extract_power_spectrum_gpu(
                                            audio, **filtered_params
                                        )
                                    else:
                                        power_spec = (
                                            feature_extractor.extract_power_spectrum(
                                                audio, **filtered_params
                                            )
                                        )
                                    # Flatten to 1D for compatibility with sklearn models
                                    features = power_spec.flatten()
                                elif extraction_mode == "chroma":
                                    # Chroma features
                                    filtered_params = filter_params_for_mode(
                                        extraction_mode, spectrogram_params
                                    )
                                    chroma = feature_extractor.extract_chroma_features(
                                        audio, **filtered_params
                                    )
                                    # Flatten to 1D for compatibility with sklearn models
                                    features = chroma.flatten()
                                elif extraction_mode == "both":
                                    # Both features and FULL spectrogram
                                    hand_features = feature_extractor.extract_features(
                                        audio, method="comprehensive"
                                    )
                                    filtered_params = filter_params_for_mode(
                                        "spectrogram", spectrogram_params
                                    )
                                    # Use GPU acceleration if available
                                    if (
                                        hasattr(
                                            feature_extractor, "extract_spectrogram_gpu"
                                        )
                                        and HAS_GPU_SUPPORT
                                    ):
                                        spectrogram = (
                                            feature_extractor.extract_spectrogram_gpu(
                                                audio, **filtered_params
                                            )
                                        )
                                    else:
                                        spectrogram = (
                                            feature_extractor.extract_spectrogram(
                                                audio, **filtered_params
                                            )
                                        )
                                    # Use FULL spectrogram (no dimensionality reduction)
                                    spectrogram_full = spectrogram.flatten()
                                    # Concatenate hand-crafted features + full spectrogram
                                    features = np.concatenate(
                                        [hand_features, spectrogram_full]
                                    )
                                else:
                                    raise ValueError(
                                        f"Unknown extraction mode: {extraction_mode}"
                                    )

                                # Apply per-sample normalization if enabled
                                if per_sample_normalizer is not None:
                                    features = per_sample_normalizer.normalize(features)

                                X_feat.append(features)
                                y_labels.append(labels[i])

                                # Apply augmentation if enabled (training data only)
                                if augmenter is not None:
                                    for aug_idx in range(augmentation_factor):
                                        try:
                                            augmented_audio = augmenter.augment(
                                                audio, augment_type="all"
                                            )

                                            # Extract features from augmented audio (same mode)
                                            if extraction_mode == "features":
                                                aug_features = (
                                                    feature_extractor.extract_features(
                                                        augmented_audio,
                                                        method="comprehensive",
                                                    )
                                                )
                                            elif extraction_mode == "spectrogram":
                                                # Use GPU acceleration if available
                                                if (
                                                    hasattr(
                                                        feature_extractor,
                                                        "extract_spectrogram_gpu",
                                                    )
                                                    and HAS_GPU_SUPPORT
                                                ):
                                                    aug_spectrogram = feature_extractor.extract_spectrogram_gpu(
                                                        augmented_audio,
                                                        **spectrogram_params,
                                                    )
                                                else:
                                                    aug_spectrogram = feature_extractor.extract_spectrogram(
                                                        augmented_audio,
                                                        **spectrogram_params,
                                                    )
                                                aug_features = aug_spectrogram.flatten()
                                            elif extraction_mode == "mfcc":
                                                # Use GPU acceleration if available
                                                if (
                                                    hasattr(
                                                        feature_extractor,
                                                        "extract_mfcc_gpu",
                                                    )
                                                    and HAS_GPU_SUPPORT
                                                ):
                                                    aug_mfcc = feature_extractor.extract_mfcc_gpu(
                                                        augmented_audio,
                                                        **spectrogram_params,
                                                    )
                                                else:
                                                    aug_mfcc = (
                                                        feature_extractor.extract_mfcc(
                                                            augmented_audio,
                                                            **spectrogram_params,
                                                        )
                                                    )
                                                aug_features = aug_mfcc.flatten()
                                            elif (
                                                extraction_mode == "magnitude_spectrum"
                                            ):
                                                # Use GPU acceleration if available
                                                if (
                                                    hasattr(
                                                        feature_extractor,
                                                        "extract_magnitude_spectrum_gpu",
                                                    )
                                                    and HAS_GPU_SUPPORT
                                                ):
                                                    aug_mag_spec = feature_extractor.extract_magnitude_spectrum_gpu(
                                                        augmented_audio,
                                                        **spectrogram_params,
                                                    )
                                                else:
                                                    aug_mag_spec = feature_extractor.extract_magnitude_spectrum(
                                                        augmented_audio,
                                                        **spectrogram_params,
                                                    )
                                                aug_features = aug_mag_spec.flatten()
                                            elif extraction_mode == "power_spectrum":
                                                # Use GPU acceleration if available
                                                if (
                                                    hasattr(
                                                        feature_extractor,
                                                        "extract_power_spectrum_gpu",
                                                    )
                                                    and HAS_GPU_SUPPORT
                                                ):
                                                    aug_power_spec = feature_extractor.extract_power_spectrum_gpu(
                                                        augmented_audio,
                                                        **spectrogram_params,
                                                    )
                                                else:
                                                    aug_power_spec = feature_extractor.extract_power_spectrum(
                                                        augmented_audio,
                                                        **spectrogram_params,
                                                    )
                                                aug_features = aug_power_spec.flatten()
                                            elif extraction_mode == "chroma":
                                                aug_chroma = feature_extractor.extract_chroma_features(
                                                    augmented_audio,
                                                    **spectrogram_params,
                                                )
                                                aug_features = aug_chroma.flatten()
                                            elif extraction_mode == "both":
                                                aug_hand_features = (
                                                    feature_extractor.extract_features(
                                                        augmented_audio,
                                                        method="comprehensive",
                                                    )
                                                )
                                                aug_spectrogram = feature_extractor.extract_spectrogram(
                                                    augmented_audio,
                                                    **spectrogram_params,
                                                )
                                                # Use FULL spectrogram (same as main features)
                                                aug_spectrogram_full = (
                                                    aug_spectrogram.flatten()
                                                )
                                                aug_features = np.concatenate(
                                                    [
                                                        aug_hand_features,
                                                        aug_spectrogram_full,
                                                    ]
                                                )

                                            # Apply per-sample normalization to augmented features
                                            if per_sample_normalizer is not None:
                                                aug_features = (
                                                    per_sample_normalizer.normalize(
                                                        aug_features
                                                    )
                                                )

                                            X_feat.append(aug_features)
                                            y_labels.append(
                                                labels[i]
                                            )  # Same label as original
                                        except Exception as aug_e:
                                            # Skip failed augmentation
                                            pass

                                if (i + 1) % 50 == 0:
                                    self.logger.info(
                                        f"  Processed {i + 1}/{len(audio_data)} samples"
                                    )

                            except Exception as e:
                                failed_count += 1
                                if failed_count <= 5:  # Only log first few failures
                                    self.logger.warning(
                                        f"Failed to process sample {i}: {str(e)}"
                                    )

                        # Log processing results
                        self.logger.info(
                            f"✓ Processed {len(X_feat)} samples ({failed_count} failed)"
                        )

                        # Convert to numpy arrays
                        X_feat = np.array(X_feat)
                        y_labels = np.array(y_labels)

                        # Use y_labels (which includes augmented sample labels) instead of original labels
                        labels = np.array(y_labels)

                        # Map labels to grouped classes
                        labels = self._map_labels_to_groups(labels)
                        labels = np.array(labels)

                        # Update actual_classes to reflect grouped labels
                        actual_classes = sorted(list(set(labels)))

                        # ================================================================
                        # APPLY CLASS FILTERING AT DATA LOADING STAGE
                        # ================================================================
                        # Filter out specified classes IMMEDIATELY after loading
                        # This ensures filtered classes never enter the pipeline
                        class_filtering_config = self.config.get("class_filtering", {})

                        if class_filtering_config.get("enabled", False):
                            classes_to_exclude = class_filtering_config.get(
                                "classes_to_exclude_train", []
                            )
                            if classes_to_exclude:
                                self.logger.info("=" * 80)
                                self.logger.info(
                                    f"🔍 CLASS FILTERING AT DATA LOAD: {mode_key}"
                                )
                                self.logger.info(
                                    f"  Excluding classes: {classes_to_exclude}"
                                )
                                self.logger.info(f"  BEFORE filtering:")
                                self.logger.info(f"    Total samples: {len(X_feat)}")
                                self.logger.info(
                                    f"    Classes present: {actual_classes}"
                                )
                                self.logger.info(
                                    f"    Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
                                )

                                # Keep only samples NOT in excluded classes
                                mask = ~np.isin(labels, classes_to_exclude)
                                X_feat = X_feat[mask]
                                labels = labels[mask]

                                # Update metadata if needed
                                if isinstance(metadata, list):
                                    metadata = [
                                        m for i, m in enumerate(metadata) if mask[i]
                                    ]

                                # Update actual_classes after filtering
                                actual_classes = sorted(list(set(labels)))

                                self.logger.info(f"  AFTER filtering:")
                                self.logger.info(f"    Total samples: {len(X_feat)}")
                                self.logger.info(
                                    f"    Classes present: {actual_classes}"
                                )
                                self.logger.info(
                                    f"    Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
                                )
                                self.logger.info(
                                    f"  ✅ Filtered out {mask.sum() - len(X_feat)} samples"
                                )
                                self.logger.info("=" * 80)

                        # Store results for this batch and mode
                        batch_results[mode_key] = {
                            "features": X_feat,
                            "labels": labels,
                            "metadata": metadata,
                            "num_samples": len(X_feat),
                            "num_features": len(X_feat[0]) if len(X_feat) > 0 else 0,
                            "classes": actual_classes,
                            "class_distribution": dict(
                                zip(*np.unique(labels, return_counts=True))
                            ),
                            "extraction_mode": extraction_mode,  # Store which mode was used
                        }

                        self.logger.info(
                            f"✓ Stored results for {mode_key}: {len(X_feat)} samples, {len(X_feat[0]) if len(X_feat) > 0 else 0} features"
                        )
                        self.logger.info(
                            f"  Final classes in batch_results: {actual_classes}"
                        )

                    # Skip the rest of the processing since we've handled it in the loop
                    continue
                else:
                    self.logger.warning(f"No audio data loaded from {batch_name}")

            except Exception as e:
                self.logger.error(f"Error processing {batch_name}: {str(e)}")
                continue

        if not batch_results:
            raise ValueError("No data could be loaded from any batch")

        # Create summary statistics
        total_samples = sum(batch["num_samples"] for batch in batch_results.values())
        all_features = (
            batch_results[list(batch_results.keys())[0]]["num_features"]
            if batch_results
            else 0
        )
        all_classes = set()
        for batch in batch_results.values():
            all_classes.update(batch["classes"])

        # Log summary
        self.logger.info(
            f"Processed {len(batch_results)} batches with {total_samples} total samples"
        )
        for batch_name, batch_data in batch_results.items():
            self.logger.info(
                f"{batch_name}: {batch_data['num_samples']} samples, {len(batch_data['classes'])} classes"
            )

        # Prepare results
        results = {
            "batch_results": batch_results,
            "total_batches": len(batch_results),
            "total_samples": total_samples,
            "num_features": all_features,
            "num_classes": len(all_classes),
            "class_names": sorted(list(all_classes)),
            "batch_names": list(batch_results.keys()),
            "validation_datasets": validation_datasets_config,  # Store validation dataset info
            "training_datasets": [
                b for b in batch_results.keys() if b not in validation_datasets_config
            ],
            "multi_dataset_mode": multi_dataset_enabled,
        }

        # Add multi-dataset metadata if enabled
        if multi_dataset_enabled:
            results["multi_dataset_config"] = {
                "training_datasets": multi_dataset_config.get("training_datasets", []),
                "validation_dataset": multi_dataset_config.get(
                    "validation_dataset", None
                ),
                "train_test_split": multi_dataset_config.get("train_test_split", 0.8),
                "random_seed": multi_dataset_config.get("random_seed", 42),
                "stratify": multi_dataset_config.get("stratify", True),
            }

        # Save preprocessing summary
        summary = {
            "total_samples": total_samples,
            "total_features": all_features,
            "total_batches": len(batch_results),
            "batch_names": list(batch_results.keys()),
            "per_batch_info": {
                batch_name: {
                    "num_samples": batch_data["num_samples"],
                    "num_classes": len(batch_data["classes"]),
                    "classes": batch_data["classes"],
                    "class_distribution": batch_data["class_distribution"],
                }
                for batch_name, batch_data in batch_results.items()
            },
        }

        self.save_results(summary, "data_processing_summary.json")

        self.logger.info("Data processing experiment completed successfully")
        return results

    def _map_labels_to_groups(self, labels):
        """Map raw folder names to grouped classes: surface_* -> contact, no_surface_* -> no_contact, edge_* -> edge."""
        mapped_labels = []
        for label in labels:
            if isinstance(label, str):
                if label.startswith("surface"):
                    mapped_labels.append("contact")
                elif label.startswith("no_surface"):
                    mapped_labels.append("no_contact")
                elif label.startswith("edge"):
                    mapped_labels.append("edge")
                else:
                    mapped_labels.append(label)  # Fallback for unknown labels
            else:
                mapped_labels.append(str(label))  # Handle non-string labels
        return mapped_labels

    def _apply_high_pass_filter(
        self, audio: np.ndarray, sr: int, cutoff: float
    ) -> np.ndarray:
        """Apply a high-pass Butterworth filter to remove low-frequency noise from audio."""
        # Design a 4th-order Butterworth high-pass filter
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype="high")

        # Apply the filter
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio

    def _save_batch_data_processing_results(self, batch_data: dict, batch_name: str):
        """Save detailed data processing results for a specific batch."""
        import json
        import os

        # Ensure labels and features are numpy arrays
        batch_data["features"] = (
            np.array(batch_data["features"])
            if not isinstance(batch_data["features"], np.ndarray)
            else batch_data["features"]
        )
        batch_data["labels"] = (
            np.array(batch_data["labels"])
            if not isinstance(batch_data["labels"], np.ndarray)
            else batch_data["labels"]
        )

        # Create batch-specific output directory
        batch_output_dir = os.path.join(self.experiment_output_dir, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        # Create a serializable version of the results (without large numpy arrays)
        serializable_results = {
            "batch_name": batch_name,
            "num_samples": batch_data["num_samples"],
            "num_features": batch_data["num_features"],
            "classes": batch_data["classes"],
            "class_distribution": batch_data["class_distribution"],
            "feature_statistics": {
                "features_shape": batch_data["features"].shape,
                "labels_shape": batch_data["labels"].shape,
                "feature_mean": batch_data["features"].mean(axis=0).tolist(),
                "feature_std": batch_data["features"].std(axis=0).tolist(),
                "feature_min": batch_data["features"].min(axis=0).tolist(),
                "feature_max": batch_data["features"].max(axis=0).tolist(),
            },
        }

        # Save feature statistics and metadata
        results_path = os.path.join(
            batch_output_dir, f"{batch_name}_data_processing_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Save features and labels as separate numpy files for efficient loading
        features_path = os.path.join(batch_output_dir, f"{batch_name}_features.npy")
        labels_path = os.path.join(batch_output_dir, f"{batch_name}_labels.npy")
        metadata_path = os.path.join(batch_output_dir, f"{batch_name}_metadata.json")

        np.save(features_path, batch_data["features"])
        np.save(labels_path, batch_data["labels"])

        with open(metadata_path, "w") as f:
            json.dump(batch_data["metadata"], f, indent=2, default=str)

        # Create batch summary
        batch_summary = {
            "batch_name": batch_name,
            "num_samples": batch_data["num_samples"],
            "num_features": batch_data["num_features"],
            "num_classes": len(batch_data["classes"]),
            "classes": batch_data["classes"],
            "class_distribution": batch_data["class_distribution"],
            "files_saved": [
                f"{batch_name}_data_processing_results.json",
                f"{batch_name}_features.npy",
                f"{batch_name}_labels.npy",
                f"{batch_name}_metadata.json",
            ],
        }

        # Save batch summary
        summary_path = os.path.join(batch_output_dir, f"{batch_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2, default=str)

        self.logger.info(
            f"Batch {batch_name} data processing results saved to: {batch_output_dir}"
        )

    def _create_batch_plots(self, batch_data: dict, batch_name: str):
        """Create visualization plots for a specific batch."""
        try:
            # Create batch-specific output directory
            batch_output_dir = os.path.join(self.experiment_output_dir, batch_name)
            os.makedirs(batch_output_dir, exist_ok=True)

            # Create class distribution plot
            self._create_class_distribution_plot(
                batch_data, batch_name, batch_output_dir
            )

            # Create feature distribution plots
            self._create_feature_distribution_plot(
                batch_data, batch_name, batch_output_dir
            )

            # Create feature correlation heatmap
            self._create_feature_correlation_plot(
                batch_data, batch_name, batch_output_dir
            )

            # Create comprehensive data overview plot
            self._create_data_overview_plot(batch_data, batch_name, batch_output_dir)

            # Create spectrogram visualization for validation
            self._create_spectrogram_plot(batch_name, batch_output_dir)

        except Exception as e:
            self.logger.warning(f"Failed to create plots for batch {batch_name}: {e}")

    def _create_class_distribution_plot(
        self, batch_data: dict, batch_name: str, output_dir: str
    ):
        """Create class distribution visualization."""
        class_dist = batch_data["class_distribution"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))

        bars = ax1.bar(classes, counts, color=colors, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Classes")
        ax1.set_ylabel("Number of Samples")
        ax1.set_title(f"Class Distribution - {batch_name}")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{count}",
                ha="center",
                va="bottom",
            )

        # Pie chart
        wedges, texts, autotexts = ax2.pie(
            counts, labels=classes, autopct="%1.1f%%", colors=colors, startangle=90
        )
        ax2.set_title(f"Class Distribution Percentage - {batch_name}")

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_class_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_feature_distribution_plot(
        self, batch_data: dict, batch_name: str, output_dir: str
    ):
        """Create feature distribution and statistics plots."""
        features = batch_data["features"]
        feature_stats = batch_data.get("feature_statistics", {})

        # Create subplot grid
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Feature mean distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if "feature_mean" in feature_stats:
            ax1.hist(
                feature_stats["feature_mean"],
                bins=20,
                alpha=0.7,
                color="skyblue",
                edgecolor="black",
            )
            ax1.set_xlabel("Feature Mean Values")
            ax1.set_ylabel("Number of Features")
            ax1.set_title("Distribution of Feature Means")
            ax1.grid(alpha=0.3)

        # 2. Feature standard deviation
        ax2 = fig.add_subplot(gs[0, 1])
        if "feature_std" in feature_stats:
            ax2.hist(
                feature_stats["feature_std"],
                bins=20,
                alpha=0.7,
                color="lightgreen",
                edgecolor="black",
            )
            ax2.set_xlabel("Feature Standard Deviation")
            ax2.set_ylabel("Number of Features")
            ax2.set_title("Distribution of Feature Std Dev")
            ax2.grid(alpha=0.3)

        # 3. Feature range (max - min)
        ax3 = fig.add_subplot(gs[0, 2])
        if "feature_min" in feature_stats and "feature_max" in feature_stats:
            feature_ranges = np.array(feature_stats["feature_max"]) - np.array(
                feature_stats["feature_min"]
            )
            ax3.hist(
                feature_ranges,
                bins=20,
                alpha=0.7,
                color="lightcoral",
                edgecolor="black",
            )
            ax3.set_xlabel("Feature Range (Max - Min)")
            ax3.set_ylabel("Number of Features")
            ax3.set_title("Distribution of Feature Ranges")
            ax3.grid(alpha=0.3)

        # 4. Feature vs Index scatter plot (first 10 samples)
        ax4 = fig.add_subplot(gs[1, :])
        if features.shape[0] > 0:
            # Show first 10 samples across all features
            n_samples_to_show = min(10, features.shape[0])
            for i in range(n_samples_to_show):
                ax4.plot(
                    features[i],
                    alpha=0.7,
                    label=f"Sample {i+1}" if n_samples_to_show <= 5 else None,
                )

            ax4.set_xlabel("Feature Index")
            ax4.set_ylabel("Feature Value")
            ax4.set_title(
                f"Feature Values Across Indices (First {n_samples_to_show} Samples)"
            )
            ax4.grid(alpha=0.3)
            if n_samples_to_show <= 5:
                ax4.legend()

        plt.suptitle(f"Feature Statistics Overview - {batch_name}", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_feature_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_feature_correlation_plot(
        self, batch_data: dict, batch_name: str, output_dir: str
    ):
        """Create feature correlation heatmap."""
        features = batch_data["features"]

        # Sample features if there are too many (for readability)
        if features.shape[1] > 50:
            # Sample every nth feature to get around 20-30 features
            step = features.shape[1] // 25
            sampled_features = features[:, ::step]
            feature_indices = list(range(0, features.shape[1], step))
        else:
            sampled_features = features
            feature_indices = list(range(features.shape[1]))

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(sampled_features.T)

        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        im = ax.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

        # Set ticks and labels
        n_features = len(feature_indices)
        tick_spacing = max(1, n_features // 10)  # Show at most 10 tick labels
        tick_positions = list(range(0, n_features, tick_spacing))
        tick_labels = [f"F{feature_indices[i]}" for i in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_yticklabels(tick_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Correlation Coefficient", rotation=270, labelpad=15)

        ax.set_title(f"Feature Correlation Matrix - {batch_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_feature_correlation.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_data_overview_plot(
        self, batch_data: dict, batch_name: str, output_dir: str
    ):
        """Create comprehensive data overview plot."""
        features = batch_data["features"]
        labels = batch_data["labels"]
        class_dist = batch_data["class_distribution"]

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Dataset summary statistics
        ax1 = fig.add_subplot(gs[0, 0])
        stats_text = f"""Dataset Summary:
• Samples: {batch_data['num_samples']:,}
• Features: {batch_data['num_features']}
• Classes: {len(batch_data['classes'])}
• Class Names: {', '.join(batch_data['classes'])}

Feature Statistics:
• Mean Range: [{np.mean(features):.3f}]
• Std Range: [{np.std(features):.3f}]
• Min Value: {np.min(features):.3f}
• Max Value: {np.max(features):.3f}"""

        ax1.text(
            0.1,
            0.9,
            stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")
        ax1.set_title("Dataset Overview")

        # 2. Class distribution
        ax2 = fig.add_subplot(gs[0, 1])
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))

        bars = ax2.bar(classes, counts, color=colors, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Classes")
        ax2.set_ylabel("Sample Count")
        ax2.set_title("Class Distribution")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(counts) * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 3. Feature value distribution by class
        ax3 = fig.add_subplot(gs[1, :])
        unique_labels = np.unique(labels)
        n_features_to_show = min(10, features.shape[1])

        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_features = features[mask]

            # Calculate mean feature values for this class
            if len(class_features) > 0:
                mean_features = np.mean(class_features[:, :n_features_to_show], axis=0)
                ax3.plot(
                    range(n_features_to_show),
                    mean_features,
                    marker="o",
                    label=f"Class: {label}",
                    alpha=0.7,
                )

        ax3.set_xlabel("Feature Index")
        ax3.set_ylabel("Mean Feature Value")
        ax3.set_title(
            f"Mean Feature Values by Class (First {n_features_to_show} Features)"
        )
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Sample distribution visualization (if feasible)
        ax4 = fig.add_subplot(gs[2, :])

        # Create a simple 2D projection using first 2 features
        if features.shape[1] >= 2:
            for i, label in enumerate(unique_labels):
                mask = labels == label
                class_features = features[mask]

                if len(class_features) > 0:
                    ax4.scatter(
                        class_features[:, 0],
                        class_features[:, 1],
                        label=f"Class: {label}",
                        alpha=0.6,
                        s=30,
                    )

            ax4.set_xlabel("Feature 0")
            ax4.set_ylabel("Feature 1")
            ax4.set_title("Sample Distribution (First 2 Features)")
            ax4.legend()
            ax4.grid(alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "Insufficient features for 2D visualization",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Sample Distribution")

        plt.suptitle(f"Comprehensive Data Overview - {batch_name}", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_data_overview.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_spectrogram_plot(self, batch_name: str, output_dir: str):
        """Create spectrogram visualization for a sample recording to validate sweep presence."""
        try:
            import librosa
            import librosa.display

            # Get data directory path
            base_data_dir = self.config.get("base_data_dir", "data")
            batch_data_dir = f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}/{batch_name}/data"

            if not os.path.exists(batch_data_dir):
                self.logger.warning(f"Data directory not found: {batch_data_dir}")
                return

            # Find first WAV file
            wav_files = [f for f in os.listdir(batch_data_dir) if f.endswith(".wav")]
            if not wav_files:
                self.logger.warning(f"No WAV files found in {batch_data_dir}")
                return

            sample_file = os.path.join(batch_data_dir, wav_files[0])
            self.logger.info(f"Creating spectrogram from sample file: {sample_file}")

            # Load audio
            audio, sr = librosa.load(sample_file, sr=48000)

            # Apply same smoothing if enabled
            apply_smoothing = self.config.get("apply_audio_smoothing", False)
            if apply_smoothing:
                cutoff_freq = self.config.get("smoothing_cutoff_freq", 500)
                audio = self._apply_high_pass_filter(audio, sr=sr, cutoff=cutoff_freq)

            # Create spectrogram
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Compute STFT
            D = librosa.stft(audio, n_fft=2048, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # Display spectrogram
            img = librosa.display.specshow(
                S_db,
                sr=sr,
                hop_length=512,
                x_axis="time",
                y_axis="log",
                ax=ax,
                cmap="viridis",
            )

            ax.set_title(
                f"Spectrogram - {batch_name} (Sample: {os.path.basename(sample_file)})"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

            # Add colorbar
            fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{batch_name}_spectrogram.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            self.logger.info(f"Spectrogram saved to {output_dir}")

        except Exception as e:
            self.logger.warning(f"Failed to create spectrogram for {batch_name}: {e}")

    def _apply_motion_artifact_removal_to_single_audio(
        self, audio: np.ndarray, template: np.ndarray
    ) -> np.ndarray:
        """Apply motion artifact removal to a single audio signal (helper for validation)."""
        from acoustic_sensing.core.motion_artifact_removal import (
            _static_subtraction,
            _lms_adaptive_filter,
        )

        # Apply the same 2-stage pipeline used in the main processing
        clean1 = _static_subtraction(audio, template)
        clean_final = _lms_adaptive_filter(clean1, template)

        # Pad back to original length if necessary
        if len(clean_final) < len(audio):
            clean_final = np.pad(
                clean_final, (0, len(audio) - len(clean_final)), "constant"
            )

        return clean_final.astype(np.float32)

    def _calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        # Ensure same length
        min_len = min(len(signal), len(noise))
        signal = signal[:min_len]
        noise = noise[:min_len]

        # Calculate signal power
        signal_power = np.mean(signal**2)

        # Calculate noise power (using the static reference as noise estimate)
        noise_power = np.mean(noise**2)

        # Avoid division by zero
        if noise_power < 1e-10:
            return float("inf")

        # Calculate SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def _create_motion_artifact_comparison_plots(
        self,
        audio_data: np.ndarray,
        labels: np.ndarray,
        static_dir: Path,
        batch_name: str,
        output_dir: str,
    ):
        """Create before/after comparison spectrograms for motion artifact removal with quantitative validation."""
        try:
            # Import required libraries
            import librosa.display
            from acoustic_sensing.core.motion_artifact_removal import (
                compute_static_templates,
                remove_motion_artifacts,
            )

            # Create batch-specific output directory
            batch_output_dir = os.path.join(output_dir, batch_name)
            os.makedirs(batch_output_dir, exist_ok=True)

            # Get unique classes
            unique_labels = np.unique(labels)

            # Compute static templates for motion artifact removal
            templates = compute_static_templates(static_dir, sr=48000)

            # Store validation metrics
            validation_results = {}

            for class_name in unique_labels:
                if class_name not in ["contact", "edge", "no_contact"]:
                    continue

                # Find recordings of this class
                class_mask = labels == class_name
                class_indices = np.where(class_mask)[0]

                if len(class_indices) == 0:
                    continue

                # Use the first recording of this class as example
                example_idx = class_indices[0]
                original_audio = audio_data[example_idx]  # This is already cleaned!

                # We need the original uncleaned audio for comparison
                # Load it from the raw data before motion artifact removal
                base_data_dir = self.config.get("base_data_dir", "data")
                batch_data_dir = f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}/{batch_name}/data"

                # Find the corresponding WAV file
                wav_files = [
                    f for f in os.listdir(batch_data_dir) if f.endswith(".wav")
                ]
                # Match by index - this is approximate but works for validation
                if example_idx < len(wav_files):
                    raw_audio_file = os.path.join(
                        batch_data_dir, wav_files[example_idx]
                    )
                    raw_audio, _ = librosa.load(raw_audio_file, sr=48000)
                else:
                    self.logger.warning(
                        f"Could not find corresponding raw audio file for index {example_idx}"
                    )
                    continue

                # Load corresponding static reference
                static_files = list(static_dir.glob(f"*_{class_name}.wav"))
                if not static_files:
                    self.logger.warning(f"No static reference found for {class_name}")
                    continue

                # Use first static reference as example
                static_audio, _ = librosa.load(static_files[0], sr=48000)

                # Apply motion artifact removal to get the cleaned version
                if class_name in templates:
                    template = templates[class_name]
                    # Apply the same processing that was done during data loading
                    clean_audio = self._apply_motion_artifact_removal_to_single_audio(
                        raw_audio, template
                    )
                else:
                    clean_audio = raw_audio.copy()

                # Calculate quantitative metrics
                snr_before = self._calculate_snr(raw_audio, static_audio)
                snr_after = self._calculate_snr(clean_audio, static_audio)
                noise_reduction_db = snr_after - snr_before

                # Store validation results
                validation_results[class_name] = {
                    "snr_before": snr_before,
                    "snr_after": snr_after,
                    "noise_reduction_db": noise_reduction_db,
                    "rms_noise_removed": np.sqrt(
                        np.mean((raw_audio - clean_audio) ** 2)
                    ),
                }

                # Create comparison plot
                fig, axes = plt.subplots(4, 1, figsize=(12, 16))

                # 1. Original recording spectrogram (raw, uncleaned)
                D_orig = librosa.stft(raw_audio, n_fft=2048, hop_length=512)
                S_db_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
                img1 = librosa.display.specshow(
                    S_db_orig,
                    sr=48000,
                    hop_length=512,
                    x_axis="time",
                    y_axis="log",
                    ax=axes[0],
                    cmap="viridis",
                )
                axes[0].set_title(f"Original Recording (Raw) - {class_name}")
                axes[0].set_xlabel("")

                # 2. Static reference spectrogram
                D_static = librosa.stft(static_audio, n_fft=2048, hop_length=512)
                S_db_static = librosa.amplitude_to_db(np.abs(D_static), ref=np.max)
                img2 = librosa.display.specshow(
                    S_db_static,
                    sr=48000,
                    hop_length=512,
                    x_axis="time",
                    y_axis="log",
                    ax=axes[1],
                    cmap="viridis",
                )
                axes[1].set_title(f"Static Reference (Noise Template) - {class_name}")
                axes[1].set_xlabel("")

                # 3. Cleaned recording spectrogram
                D_clean = librosa.stft(clean_audio, n_fft=2048, hop_length=512)
                S_db_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
                img3 = librosa.display.specshow(
                    S_db_clean,
                    sr=48000,
                    hop_length=512,
                    x_axis="time",
                    y_axis="log",
                    ax=axes[2],
                    cmap="viridis",
                )
                axes[2].set_title(f"After Motion Artifact Removal - {class_name}")
                axes[2].set_xlabel("")

                # 4. What was removed - show the difference
                noise_removed = raw_audio - clean_audio
                D_removed = librosa.stft(noise_removed, n_fft=2048, hop_length=512)
                S_db_removed = librosa.amplitude_to_db(np.abs(D_removed), ref=np.max)
                img4 = librosa.display.specshow(
                    S_db_removed,
                    sr=48000,
                    hop_length=512,
                    x_axis="time",
                    y_axis="log",
                    ax=axes[3],
                    cmap="plasma",
                )
                axes[3].set_title(f"Removed Robot Noise (Raw - Clean) - {class_name}")

                # Add colorbar to the last subplot
                fig.colorbar(
                    img4,
                    ax=axes,
                    format="%+2.0f dB",
                    label="Amplitude (dB)",
                    shrink=0.8,
                )

                # Add quantitative metrics as text
                metrics_text = f"""
SNR Before: {snr_before:.1f} dB
SNR After: {snr_after:.1f} dB
Noise Reduction: {noise_reduction_db:.1f} dB
RMS Noise Removed: {validation_results[class_name]['rms_noise_removed']:.4f}
"""

                fig.text(
                    0.02,
                    0.98,
                    metrics_text,
                    transform=fig.transFigure,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

                plt.suptitle(
                    f"Motion Artifact Removal Validation - {batch_name} ({class_name})",
                    fontsize=14,
                    y=0.98,
                )
                plt.tight_layout()

                # Save the comparison plot
                output_path = os.path.join(
                    batch_output_dir,
                    f"{batch_name}_motion_artifact_validation_{class_name}.png",
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                self.logger.info(
                    f"Motion artifact validation plot saved: {output_path}"
                )

            # Save validation metrics
            if validation_results:
                validation_path = os.path.join(
                    batch_output_dir, f"{batch_name}_motion_artifact_metrics.json"
                )
                import json

                with open(validation_path, "w") as f:
                    json.dump(validation_results, f, indent=2)
                self.logger.info(f"Validation metrics saved: {validation_path}")

        except Exception as e:
            self.logger.warning(
                f"Failed to create motion artifact comparison plots: {e}"
            )
