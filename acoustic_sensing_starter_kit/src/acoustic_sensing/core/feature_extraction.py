"""
Enhanced Feature Extraction for Acoustic Geometric Discrimination
================================================================

This module provides comprehensive feature extraction optimized for geometric
reconstruction and discrimination tasks. Features are designed to capture:

1. Spectral characteristics (resonance patterns, frequency damping)
2. Temporal dynamics (contact burst, decay patterns)
3. Contact-specific signatures (chamber deformation, air gaps)
4. Multi-scale analysis (local and global patterns)

Based on: Zöller et al. (2020) - Contact deforms finger chamber, damping
high frequencies (>1kHz) and shifting resonances (500-800Hz).

Author: Enhanced for geometric discrimination
"""

import numpy as np
import pandas as pd
import librosa
import scipy.signal
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Optional, Union

# GPU acceleration imports (optional)
try:
    import torch
    import torchaudio
    import torchaudio.transforms as T

    HAS_GPU_SUPPORT = torch.cuda.is_available()
    print(f"GPU acceleration available: {HAS_GPU_SUPPORT}")
except ImportError:
    HAS_GPU_SUPPORT = False
    print("GPU acceleration not available - using CPU-only processing")


class GeometricFeatureExtractor:
    """
    Enhanced feature extractor optimized for geometric discrimination.

    Features include:
    - Resonance analysis (500-800Hz chamber resonance)
    - High-frequency damping patterns (>1kHz)
    - Contact burst characteristics
    - Spectral shape descriptors
    - Temporal envelope features
    - Multi-band energy analysis
    """

    def __init__(
        self,
        sr: int = 48000,
        n_fft: int = 4096,
        use_workspace_invariant: bool = True,
        use_impulse_features: bool = False,
        use_contact_physics_features: bool = True,
    ):
        """
        Initialize the feature extractor.

        Args:
            sr: Sample rate for audio processing
            n_fft: FFT size for spectral analysis
            use_workspace_invariant: Whether to include workspace-invariant features
                                    for better cross-workspace generalization
            use_impulse_features: Whether to include impulse response / transfer function
                                 features for better material discrimination
            use_contact_physics_features: Whether to include contact physics features
                                          (damping, energy transfer, etc.) for better
                                          contact detection accuracy
        """
        self.sr = sr
        self.n_fft = n_fft
        self.use_workspace_invariant = use_workspace_invariant
        self.use_impulse_features = use_impulse_features
        self.use_contact_physics_features = use_contact_physics_features
        self.freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Define frequency bands based on acoustic sensing research
        self.freq_bands = {
            "low": (50, 200),  # Low frequency structural resonances
            "resonance": (450, 850),  # Finger chamber resonance (500-800Hz)
            "mid": (850, 2000),  # Mid-frequency contact signatures
            "high": (2000, 8000),  # High-frequency damping region
            "ultra_high": (8000, 20000),  # Ultra-high frequencies (material texture)
        }

    def extract_features(
        self, audio: np.ndarray, method: str = "comprehensive"
    ) -> Union[np.ndarray, pd.Series]:
        """
        Extract features from audio signal.

        Args:
            audio: Input audio waveform
            method: Feature extraction method
                   - 'comprehensive': All geometric discrimination features
                   - 'resonance': Focus on chamber resonance features
                   - 'damping': Focus on high-frequency damping
                   - 'contact': Contact burst and temporal features
                   - 'legacy': Original STFT features for compatibility

        Returns:
            Feature vector as numpy array or pandas Series
        """
        # Ensure audio is floating-point for librosa compatibility
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        if method == "comprehensive":
            return self._extract_comprehensive_features(audio)
        elif method == "resonance":
            return self._extract_resonance_features(audio)
        elif method == "damping":
            return self._extract_damping_features(audio)
        elif method == "contact":
            return self._extract_contact_features(audio)
        elif method == "legacy":
            return self._extract_legacy_features(audio)
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")

    def _extract_comprehensive_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract all features for geometric discrimination."""
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        features = []

        # 1. Spectral features
        spectral_feats = self._extract_spectral_features(audio)
        features.extend(spectral_feats)

        # 2. Resonance analysis features
        resonance_feats = self._extract_resonance_features(audio)
        features.extend(resonance_feats)

        # 3. Damping pattern features
        damping_feats = self._extract_damping_features(audio)
        features.extend(damping_feats)

        # 4. Contact burst features
        contact_feats = self._extract_contact_features(audio)
        features.extend(contact_feats)

        # 5. Temporal envelope features
        envelope_feats = self._extract_envelope_features(audio)
        features.extend(envelope_feats)

        # 6. Multi-band energy features
        energy_feats = self._extract_energy_features(audio)
        features.extend(energy_feats)

        # 7. Workspace-invariant features (optional)
        if self.use_workspace_invariant:
            workspace_invariant_feats = self._extract_workspace_invariant_features(
                audio
            )
            features.extend(workspace_invariant_feats)

        # 8. Impulse response / Transfer function features (optional)
        if self.use_impulse_features:
            impulse_feats = self._extract_impulse_response_features(audio)
            features.extend(impulse_feats)

        # 9. Contact physics features (optional) - NEW!
        if self.use_contact_physics_features:
            contact_physics_feats = self._extract_contact_physics_features(audio)
            features.extend(contact_physics_feats)

        return np.array(features)

    def _extract_spectral_features(self, audio: np.ndarray) -> List[float]:
        """Extract frequency domain features."""
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Compute power spectrum
        stft = librosa.stft(audio, n_fft=self.n_fft)
        power_spectrum = np.abs(stft) ** 2
        power_sum = np.sum(power_spectrum, axis=1)

        features = []

        # Spectral centroid (frequency center of mass)
        spectral_centroid = np.sum(self.freq_bins * power_sum) / np.sum(power_sum)
        features.append(spectral_centroid)

        # Spectral bandwidth (frequency spread)
        spectral_bandwidth = np.sqrt(
            np.sum(((self.freq_bins - spectral_centroid) ** 2) * power_sum)
            / np.sum(power_sum)
        )
        features.append(spectral_bandwidth)

        # Spectral rolloff (95% energy point)
        cumulative_energy = np.cumsum(power_sum)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
        spectral_rolloff = (
            self.freq_bins[rolloff_idx[0]]
            if len(rolloff_idx) > 0
            else self.freq_bins[-1]
        )
        features.append(spectral_rolloff)

        # Spectral flatness (measure of noise-like character)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geo_mean = stats.gmean(power_sum[power_sum > 0])
            arith_mean = np.mean(power_sum)
            spectral_flatness = geo_mean / arith_mean if arith_mean > 0 else 0
        features.append(spectral_flatness)

        # Spectral contrast (energy difference between peaks and valleys)
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)[0]
            features.extend(contrast[:5])  # First 5 sub-bands
        except:
            features.extend([0] * 5)

        return features

    def _extract_resonance_features(self, audio: np.ndarray) -> List[float]:
        """Extract chamber resonance features (500-800Hz)."""
        features = []

        # Compute power spectrum
        stft = librosa.stft(audio, n_fft=self.n_fft)
        power_spectrum = np.abs(stft) ** 2
        power_sum = np.sum(power_spectrum, axis=1)

        # Focus on resonance band (450-850Hz)
        resonance_mask = (self.freq_bins >= self.freq_bands["resonance"][0]) & (
            self.freq_bins <= self.freq_bands["resonance"][1]
        )

        if np.any(resonance_mask):
            resonance_power = power_sum[resonance_mask]
            resonance_freqs = self.freq_bins[resonance_mask]

            # Peak amplitude in resonance band
            peak_amplitude = np.max(resonance_power)
            features.append(peak_amplitude)

            # Peak frequency
            peak_idx = np.argmax(resonance_power)
            peak_frequency = resonance_freqs[peak_idx]
            features.append(peak_frequency)

            # Total energy in resonance band
            resonance_energy = np.sum(resonance_power)
            features.append(resonance_energy)

            # Resonance sharpness (Q-factor approximation)
            # Find half-power points around peak
            half_power = peak_amplitude / 2
            above_half = resonance_power > half_power
            if np.any(above_half):
                half_power_indices = np.where(above_half)[0]
                bandwidth = (
                    resonance_freqs[half_power_indices[-1]]
                    - resonance_freqs[half_power_indices[0]]
                )
                q_factor = peak_frequency / bandwidth if bandwidth > 0 else 0
            else:
                q_factor = 0
            features.append(q_factor)

            # Resonance asymmetry (skewness of power distribution)
            resonance_skew = stats.skew(resonance_power)
            features.append(resonance_skew)

        else:
            features.extend([0] * 5)

        return features

    def _extract_damping_features(self, audio: np.ndarray) -> List[float]:
        """Extract high-frequency damping features (>1kHz)."""
        features = []

        # Compute power spectrum
        stft = librosa.stft(audio, n_fft=self.n_fft)
        power_spectrum = np.abs(stft) ** 2
        power_sum = np.sum(power_spectrum, axis=1)

        # High frequency analysis (>1kHz)
        high_freq_mask = self.freq_bins > 1000
        mid_freq_mask = (self.freq_bins >= 200) & (self.freq_bins <= 1000)

        if np.any(high_freq_mask) and np.any(mid_freq_mask):
            high_energy = np.sum(power_sum[high_freq_mask])
            mid_energy = np.sum(power_sum[mid_freq_mask])

            # High-to-mid frequency ratio (damping indicator)
            damping_ratio = high_energy / mid_energy if mid_energy > 0 else 0
            features.append(damping_ratio)

            # High frequency rolloff slope
            high_freqs = self.freq_bins[high_freq_mask]
            high_power = power_sum[high_freq_mask]

            if len(high_freqs) > 1:
                # Fit linear regression to log power vs frequency
                log_power = np.log(high_power + 1e-10)
                slope, _, _, _, _ = stats.linregress(high_freqs, log_power)
                features.append(slope)

                # High frequency decay rate
                decay_rate = -slope  # More negative slope = faster decay
                features.append(decay_rate)
            else:
                features.extend([0, 0])

        else:
            features.extend([0, 0, 0])

        # Ultra-high frequency analysis (8-20kHz)
        ultra_high_mask = (self.freq_bins >= 8000) & (self.freq_bins <= 20000)
        if np.any(ultra_high_mask):
            ultra_high_energy = np.sum(power_sum[ultra_high_mask])
            total_energy = np.sum(power_sum)
            ultra_high_ratio = (
                ultra_high_energy / total_energy if total_energy > 0 else 0
            )
            features.append(ultra_high_ratio)
        else:
            features.append(0)

        return features

    def _extract_contact_features(self, audio: np.ndarray) -> List[float]:
        """Extract contact burst and temporal features."""
        features = []

        # Contact burst analysis (first 0.5 seconds)
        burst_duration = 0.5  # seconds
        burst_samples = int(burst_duration * self.sr)
        burst_audio = audio[: min(burst_samples, len(audio))]

        if len(burst_audio) > 0:
            # RMS energy of contact burst
            rms_burst = np.sqrt(np.mean(burst_audio**2))
            features.append(rms_burst)

            # Peak amplitude of burst
            peak_burst = np.max(np.abs(burst_audio))
            features.append(peak_burst)

            # Crest factor (peak-to-RMS ratio)
            crest_factor = peak_burst / rms_burst if rms_burst > 0 else 0
            features.append(crest_factor)

            # Time to peak
            peak_idx = np.argmax(np.abs(burst_audio))
            time_to_peak = peak_idx / self.sr
            features.append(time_to_peak)

        else:
            features.extend([0, 0, 0, 0])

        # Attack and decay characteristics
        if len(audio) > 1000:  # Ensure minimum length
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            features.append(zcr_mean)

            # Temporal centroid (center of mass in time)
            envelope = np.abs(audio)
            time_indices = np.arange(len(envelope))
            temporal_centroid = (
                np.sum(time_indices * envelope) / np.sum(envelope)
                if np.sum(envelope) > 0
                else 0
            )
            temporal_centroid_normalized = temporal_centroid / len(audio)
            features.append(temporal_centroid_normalized)

        else:
            features.extend([0, 0])

        return features

    def _extract_envelope_features(self, audio: np.ndarray) -> List[float]:
        """Extract temporal envelope features."""
        features = []

        # Hilbert envelope
        analytic_signal = scipy.signal.hilbert(audio)
        envelope = np.abs(analytic_signal)

        if len(envelope) > 0:
            # Envelope statistics
            env_mean = np.mean(envelope)
            env_std = np.std(envelope)
            env_max = np.max(envelope)
            env_min = np.min(envelope)

            features.extend([env_mean, env_std, env_max, env_min])

            # Envelope shape descriptors
            env_skew = stats.skew(envelope)
            env_kurtosis = stats.kurtosis(envelope)
            features.extend([env_skew, env_kurtosis])

            # Envelope decay analysis
            # Find envelope peak and analyze decay
            peak_idx = np.argmax(envelope)
            if peak_idx < len(envelope) - 100:  # Ensure enough samples after peak
                decay_envelope = envelope[peak_idx:]
                if len(decay_envelope) > 10:
                    # Fit exponential decay
                    time_points = np.arange(len(decay_envelope)) / self.sr
                    try:
                        # Use log-linear fit for exponential decay
                        log_env = np.log(decay_envelope + 1e-10)
                        slope, _, _, _, _ = stats.linregress(time_points, log_env)
                        decay_rate = -slope  # Positive decay rate
                        features.append(decay_rate)
                    except:
                        features.append(0)
                else:
                    features.append(0)
            else:
                features.append(0)

        else:
            features.extend([0] * 7)

        return features

    def _extract_energy_features(self, audio: np.ndarray) -> List[float]:
        """Extract multi-band energy features."""
        features = []

        # Compute power spectrum
        stft = librosa.stft(audio, n_fft=self.n_fft)
        power_spectrum = np.abs(stft) ** 2
        power_sum = np.sum(power_spectrum, axis=1)
        total_energy = np.sum(power_sum)

        # Energy in each frequency band
        for band_name, (f_low, f_high) in self.freq_bands.items():
            band_mask = (self.freq_bins >= f_low) & (self.freq_bins <= f_high)
            if np.any(band_mask):
                band_energy = np.sum(power_sum[band_mask])
                band_ratio = band_energy / total_energy if total_energy > 0 else 0
                features.append(band_ratio)
            else:
                features.append(0)

        # Inter-band ratios (key for geometric discrimination)
        if len(features) >= 5:  # Ensure we have all bands
            # Resonance to high-frequency ratio (key geometric indicator)
            resonance_high_ratio = features[1] / features[3] if features[3] > 0 else 0
            features.append(resonance_high_ratio)

            # Low to mid ratio
            low_mid_ratio = features[0] / features[2] if features[2] > 0 else 0
            features.append(low_mid_ratio)

        else:
            features.extend([0, 0])

        return features

    def _extract_workspace_invariant_features(self, audio: np.ndarray) -> List[float]:
        """
        Extract workspace-invariant features for better cross-workspace generalization.

        These features use normalization and ratios to be less sensitive to:
        - Room acoustics (reverberation, ambient noise)
        - Microphone placement and sensitivity
        - Surface material variations
        - Recording conditions

        Returns:
            List of workspace-invariant features
        """
        features = []

        # Compute power spectrum
        stft = librosa.stft(audio, n_fft=self.n_fft)
        power_spectrum = np.abs(stft) ** 2
        power_sum = np.sum(power_spectrum, axis=1)
        total_energy = np.sum(power_sum)

        # 1. SPECTRAL BAND RATIOS (invariant to absolute amplitude)
        # Define broader bands for robust ratios
        bands_for_ratios = {
            "very_low": (50, 300),
            "low_mid": (300, 800),
            "mid": (800, 2000),
            "mid_high": (2000, 5000),
            "high": (5000, 12000),
        }

        band_energies = {}
        for band_name, (f_low, f_high) in bands_for_ratios.items():
            band_mask = (self.freq_bins >= f_low) & (self.freq_bins <= f_high)
            if np.any(band_mask):
                band_energies[band_name] = np.sum(power_sum[band_mask])
            else:
                band_energies[band_name] = (
                    1e-10  # Small value to avoid division by zero
                )

        # Critical ratios that capture contact vs no-contact
        # Ratio 1: Mid to High (contact dampens high frequencies)
        features.append(band_energies["mid"] / (band_energies["high"] + 1e-10))

        # Ratio 2: Low-mid to Mid-high (resonance shift indicator)
        features.append(band_energies["low_mid"] / (band_energies["mid_high"] + 1e-10))

        # Ratio 3: Very low to High (overall spectral tilt)
        features.append(band_energies["very_low"] / (band_energies["high"] + 1e-10))

        # Ratio 4: (Low-mid + Mid) / (Mid-high + High) - contact indicator
        numerator = band_energies["low_mid"] + band_energies["mid"]
        denominator = band_energies["mid_high"] + band_energies["high"]
        features.append(numerator / (denominator + 1e-10))

        # 2. NORMALIZED SPECTRAL SHAPE (invariant to overall gain)
        # Normalize power spectrum to unit energy
        if total_energy > 0:
            normalized_spectrum = power_sum / total_energy
        else:
            normalized_spectrum = power_sum

        # Spectral moments on normalized spectrum
        freqs = self.freq_bins

        # Normalized centroid
        norm_centroid = np.sum(freqs * normalized_spectrum)
        features.append(norm_centroid / 1000)  # Scale to ~0-10 range

        # Normalized spread (relative to centroid)
        norm_spread = np.sqrt(
            np.sum(((freqs - norm_centroid) ** 2) * normalized_spectrum)
        )
        features.append(norm_spread / (norm_centroid + 1e-10))

        # Normalized skewness (shape asymmetry)
        if norm_spread > 0:
            norm_skewness = np.sum(
                ((freqs - norm_centroid) ** 3) * normalized_spectrum
            ) / (norm_spread**3)
        else:
            norm_skewness = 0
        features.append(norm_skewness)

        # 3. TEMPORAL DYNAMICS RATIOS (invariant to signal amplitude)
        # Normalize audio to unit RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            normalized_audio = audio / rms
        else:
            normalized_audio = audio

        # Envelope extraction
        analytic_signal = scipy.signal.hilbert(normalized_audio)
        envelope = np.abs(analytic_signal)

        # Attack vs Decay ratio (contact has sharp attack, gradual decay)
        if len(envelope) > 10:
            peak_idx = np.argmax(envelope)
            attack_len = peak_idx if peak_idx > 0 else 1
            decay_len = len(envelope) - peak_idx if peak_idx < len(envelope) else 1
            attack_decay_ratio = attack_len / (decay_len + 1)
            features.append(attack_decay_ratio)

            # Attack slope (steepness of onset)
            if attack_len > 1:
                attack_slope = envelope[peak_idx] / attack_len
            else:
                attack_slope = 0
            features.append(attack_slope)

            # Decay rate (exponential decay characteristic)
            if decay_len > 1:
                decay_envelope = envelope[peak_idx:]
                # Fit exponential decay: avoid log(0)
                safe_decay = np.maximum(decay_envelope, 1e-10)
                log_decay = np.log(safe_decay)
                if len(log_decay) > 1:
                    decay_rate = -(log_decay[-1] - log_decay[0]) / len(log_decay)
                else:
                    decay_rate = 0
            else:
                decay_rate = 0
            features.append(decay_rate)
        else:
            features.extend([0, 0, 0])

        # 4. SPECTRAL FLUX (rate of spectral change - normalized)
        if stft.shape[1] > 1:
            spectral_flux_values = []
            for i in range(1, stft.shape[1]):
                prev_frame = np.abs(stft[:, i - 1])
                curr_frame = np.abs(stft[:, i])
                # Normalize each frame
                prev_norm = prev_frame / (np.sum(prev_frame) + 1e-10)
                curr_norm = curr_frame / (np.sum(curr_frame) + 1e-10)
                flux = np.sum((curr_norm - prev_norm) ** 2)
                spectral_flux_values.append(flux)

            avg_flux = np.mean(spectral_flux_values)
            std_flux = np.std(spectral_flux_values)
            features.append(avg_flux)
            features.append(std_flux)
        else:
            features.extend([0, 0])

        # 5. ZERO CROSSING RATE RATIO (temporal texture, normalized)
        # Compare different portions of signal
        if len(normalized_audio) > 100:
            zcr_full = np.sum(librosa.zero_crossings(normalized_audio)) / len(
                normalized_audio
            )

            # ZCR in first half vs second half
            mid_point = len(normalized_audio) // 2
            zcr_first = (
                np.sum(librosa.zero_crossings(normalized_audio[:mid_point])) / mid_point
            )
            zcr_second = np.sum(
                librosa.zero_crossings(normalized_audio[mid_point:])
            ) / (len(normalized_audio) - mid_point)
            zcr_ratio = zcr_first / (zcr_second + 1e-10)

            features.append(zcr_full)
            features.append(zcr_ratio)
        else:
            features.extend([0, 0])

        # 6. SPECTRAL CREST FACTOR (peakiness, invariant to scale)
        # For each band, compute crest factor
        for band_name in ["low_mid", "mid", "mid_high"]:
            f_low, f_high = bands_for_ratios[band_name]
            band_mask = (self.freq_bins >= f_low) & (self.freq_bins <= f_high)
            if np.any(band_mask):
                band_power = power_sum[band_mask]
                if len(band_power) > 0 and np.sum(band_power) > 0:
                    crest = np.max(band_power) / np.mean(band_power)
                else:
                    crest = 0
            else:
                crest = 0
            features.append(crest)

        return features

    def _extract_impulse_response_features(self, audio: np.ndarray) -> List[float]:
        """
        Extract impulse response / transfer function features.

        These features approximate the system's transfer function by analyzing
        the response to the chirp sweep excitation, extracting:
        - Resonance characteristics (peaks in frequency response)
        - Decay properties (how quickly energy dissipates)
        - Phase characteristics (timing relationships)

        This provides workspace-invariant features by characterizing the
        acoustic transfer function of contact vs no-contact states.
        """
        features = []

        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # 1. FREQUENCY RESPONSE ANALYSIS
        # Use larger FFT for better frequency resolution
        n_fft_ir = min(8192, len(audio))

        # Compute frequency response magnitude
        freq_response = np.fft.rfft(audio, n=n_fft_ir)
        freqs = np.fft.rfftfreq(n_fft_ir, 1 / self.sr)
        magnitude = np.abs(freq_response)

        # Normalize magnitude for workspace invariance
        if np.max(magnitude) > 0:
            magnitude_norm = magnitude / np.max(magnitude)
        else:
            magnitude_norm = magnitude

        # 2. RESONANCE PEAK DETECTION
        from scipy.signal import find_peaks

        # Find peaks in magnitude response
        try:
            peaks, properties = find_peaks(
                magnitude_norm, height=0.1, distance=20, prominence=0.05
            )

            if len(peaks) > 0:
                # Sort peaks by magnitude
                sorted_peak_indices = np.argsort(properties["peak_heights"])[::-1]
                sorted_peaks = peaks[sorted_peak_indices]

                # Primary resonance
                primary_freq = freqs[sorted_peaks[0]]
                primary_mag = magnitude_norm[sorted_peaks[0]]
                features.append(primary_freq / 1000)  # Normalize to kHz
                features.append(primary_mag)

                # Q-factor estimation for primary resonance
                peak_idx = sorted_peaks[0]
                half_power = primary_mag / np.sqrt(2)

                # Find -3dB bandwidth
                left_idx = peak_idx
                while left_idx > 0 and magnitude_norm[left_idx] > half_power:
                    left_idx -= 1
                right_idx = peak_idx
                while (
                    right_idx < len(magnitude_norm) - 1
                    and magnitude_norm[right_idx] > half_power
                ):
                    right_idx += 1

                if right_idx > left_idx:
                    bandwidth = freqs[right_idx] - freqs[left_idx]
                    q_factor = primary_freq / (bandwidth + 1e-10)
                else:
                    q_factor = 0
                features.append(min(q_factor, 100))  # Cap Q-factor

                # Number of significant resonances
                features.append(min(len(peaks), 10))

                # Secondary resonance if exists
                if len(sorted_peaks) > 1:
                    secondary_freq = freqs[sorted_peaks[1]]
                    secondary_mag = magnitude_norm[sorted_peaks[1]]
                    features.append(secondary_freq / 1000)
                    features.append(secondary_mag)
                    # Ratio between primary and secondary
                    features.append(primary_freq / (secondary_freq + 1e-10))
                else:
                    features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
        except Exception:
            features.extend([0, 0, 0, 0, 0, 0, 0])

        # 3. FREQUENCY RESPONSE SHAPE (Transfer function characteristics)
        # Centroid of frequency response
        if np.sum(magnitude) > 0:
            fr_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            features.append(fr_centroid / 1000)

            # Spread around centroid
            fr_spread = np.sqrt(
                np.sum((freqs - fr_centroid) ** 2 * magnitude) / np.sum(magnitude)
            )
            features.append(fr_spread / 1000)
        else:
            features.extend([0, 0])

        # 4. BAND ENERGY RATIOS FROM TRANSFER FUNCTION
        # These are more stable than raw audio band energies
        band_ranges = [
            (100, 500),  # Low
            (500, 1500),  # Mid-low (resonance region)
            (1500, 4000),  # Mid
            (4000, 10000),  # High
        ]

        band_mags = []
        for f_low, f_high in band_ranges:
            mask = (freqs >= f_low) & (freqs <= f_high)
            if np.any(mask):
                band_mags.append(np.mean(magnitude_norm[mask]))
            else:
                band_mags.append(0)

        # Band ratios (workspace-invariant)
        if band_mags[3] > 0:
            features.append(
                band_mags[1] / (band_mags[3] + 1e-10)
            )  # Mid-low to High ratio
        else:
            features.append(0)
        if band_mags[2] > 0:
            features.append(band_mags[0] / (band_mags[2] + 1e-10))  # Low to Mid ratio
        else:
            features.append(0)

        # 5. DECAY CHARACTERISTICS (Impulse response decay)
        # Analyze envelope decay
        analytic = scipy.signal.hilbert(audio)
        envelope = np.abs(analytic)

        if len(envelope) > 100:
            peak_idx = np.argmax(envelope)

            if peak_idx < len(envelope) - 50:
                decay_portion = envelope[peak_idx:]

                # Fit exponential decay
                decay_len = min(len(decay_portion), 1000)
                t = np.arange(decay_len) / self.sr
                y = decay_portion[:decay_len]

                # Log-linear fit for decay rate
                safe_y = np.maximum(y, 1e-10)
                log_y = np.log(safe_y)

                try:
                    slope, intercept, _, _, _ = stats.linregress(t, log_y)
                    decay_rate = -slope  # Positive = faster decay
                    features.append(min(decay_rate, 1000))  # Cap at reasonable value
                except Exception:
                    features.append(0)

                # T60-like measure (time to decay by 60dB)
                try:
                    initial_level = np.mean(envelope[peak_idx : peak_idx + 10])
                    target_level = initial_level * 0.001  # -60dB
                    decay_indices = np.where(decay_portion < target_level)[0]
                    if len(decay_indices) > 0:
                        t60 = decay_indices[0] / self.sr
                    else:
                        t60 = len(decay_portion) / self.sr
                    features.append(min(t60, 5))  # Cap at 5 seconds
                except Exception:
                    features.append(0)
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])

        # 6. PHASE CHARACTERISTICS (can indicate material properties)
        phase = np.angle(freq_response)

        # Phase linearity (group delay flatness)
        # Unwrap phase to avoid discontinuities
        phase_unwrapped = np.unwrap(phase)

        # Group delay is derivative of phase
        if len(phase_unwrapped) > 2:
            group_delay = -np.diff(phase_unwrapped) / (
                2 * np.pi * (freqs[1] - freqs[0]) + 1e-10
            )

            # Focus on meaningful frequency range (200-5000 Hz)
            freq_mask = (freqs[:-1] >= 200) & (freqs[:-1] <= 5000)
            if np.any(freq_mask):
                gd_subset = group_delay[freq_mask]
                gd_mean = np.mean(gd_subset)
                gd_std = np.std(gd_subset)
                features.append(gd_mean * 1000)  # Convert to ms
                features.append(gd_std * 1000)
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])

        return features

    def _extract_contact_physics_features(self, audio: np.ndarray) -> List[float]:
        """
        Extract physics-informed features that capture contact mechanics.

        These features model the actual physical phenomena that occur during
        acoustic tactile contact:
        - Impact transients (sharp onset when contact occurs)
        - Damping changes (contact damps vibrations faster)
        - Energy transfer (contact dissipates energy into surface)
        - Nonlinear effects (harmonic distortion from contact mechanics)
        - Resonance shifts (contact changes system resonances)

        Expected to improve cross-workspace generalization by focusing on
        universal physics rather than workspace-specific acoustics.

        Returns:
            List of 10 contact physics features
        """
        features = []

        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # ==================================================================
        # Feature 1-2: Impact Transient Detection
        # Contact creates sharp onset in amplitude envelope
        # ==================================================================
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)

        # Maximum onset strength (sharpness of impact)
        impact_sharpness = np.max(onset_env) if len(onset_env) > 0 else 0.0
        features.append(impact_sharpness)

        # Onset slope (how quickly amplitude rises)
        if len(onset_env) > 1:
            onset_diff = np.diff(onset_env)
            max_slope = np.max(onset_diff) if len(onset_diff) > 0 else 0.0
        else:
            max_slope = 0.0
        features.append(max_slope)

        # ==================================================================
        # Feature 3-4: Damping Coefficient Analysis
        # Contact increases damping → faster decay of vibrations
        # ==================================================================
        # Compute envelope of signal
        analytic_signal = scipy.signal.hilbert(audio)
        amplitude_envelope = np.abs(analytic_signal)

        # Fit exponential decay to envelope: A(t) = A0 * exp(-α*t)
        # Higher α = more damping = likely contact
        time_vector = np.arange(len(amplitude_envelope)) / self.sr

        # Only fit to decaying portion (after initial peak)
        peak_idx = np.argmax(amplitude_envelope)
        if peak_idx < len(amplitude_envelope) - 10:
            decay_envelope = amplitude_envelope[peak_idx:]
            decay_time = time_vector[peak_idx:] - time_vector[peak_idx]

            # Avoid log of zero
            decay_envelope = np.maximum(decay_envelope, 1e-10)

            # Linear fit in log space: log(A) = log(A0) - α*t
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coeffs = np.polyfit(
                        decay_time[: len(decay_envelope)], np.log(decay_envelope), deg=1
                    )
                    damping_coefficient = -coeffs[0]  # α (negative slope)
            except:
                damping_coefficient = 0.0
        else:
            damping_coefficient = 0.0

        features.append(damping_coefficient)

        # Decay rate (ratio of end to peak amplitude)
        decay_ratio = amplitude_envelope[-1] / (amplitude_envelope[peak_idx] + 1e-10)
        features.append(decay_ratio)

        # ==================================================================
        # Feature 5: Energy Transfer Rate
        # Contact dissipates energy into surface → energy drops faster
        # ==================================================================
        # Divide signal into 4 quarters, measure energy in each
        quarter_len = len(audio) // 4
        if quarter_len > 0:
            energies = []
            for i in range(4):
                start = i * quarter_len
                end = start + quarter_len if i < 3 else len(audio)
                quarter_energy = np.sum(audio[start:end] ** 2)
                energies.append(quarter_energy)

            # Energy transfer rate: how much energy is lost from start to end
            # Ratio of last quarter to first quarter (lower = more dissipation)
            energy_transfer_rate = 1.0 - (energies[-1] / (energies[0] + 1e-10))
            energy_transfer_rate = np.clip(energy_transfer_rate, 0, 1)
        else:
            energy_transfer_rate = 0.0

        features.append(energy_transfer_rate)

        # ==================================================================
        # Feature 6-7: Harmonic Distortion
        # Nonlinear contact mechanics introduce harmonics
        # ==================================================================
        # Compute power spectrum
        fft_vals = np.fft.rfft(audio)
        power_spectrum = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1 / self.sr)

        # Find fundamental frequency (first major peak)
        # Look in reasonable range (100-2000 Hz for contact sounds)
        fund_mask = (freqs >= 100) & (freqs <= 2000)
        if np.any(fund_mask):
            fund_spectrum = power_spectrum[fund_mask]
            fund_freqs = freqs[fund_mask]
            fund_idx = np.argmax(fund_spectrum)
            fundamental_freq = fund_freqs[fund_idx]
            fundamental_power = fund_spectrum[fund_idx]

            # Measure power at harmonics (2f, 3f, 4f, 5f)
            harmonic_power = 0.0
            for n in [2, 3, 4, 5]:
                harmonic_freq = n * fundamental_freq
                # Find closest frequency bin
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                if harmonic_idx < len(power_spectrum):
                    harmonic_power += power_spectrum[harmonic_idx]

            # Total Harmonic Distortion (THD) = harmonics / fundamental
            thd = harmonic_power / (fundamental_power + 1e-10)
        else:
            thd = 0.0
            fundamental_freq = 0.0

        features.append(thd)
        features.append(fundamental_freq)

        # ==================================================================
        # Feature 8: Resonance Frequency Shift
        # Contact changes system resonances (chamber deformation)
        # ==================================================================
        # Compare resonance band energy to total energy
        resonance_mask = (freqs >= 450) & (freqs <= 850)
        if np.any(resonance_mask):
            resonance_power = np.sum(power_spectrum[resonance_mask])
            total_power = np.sum(power_spectrum) + 1e-10
            resonance_ratio = resonance_power / total_power
        else:
            resonance_ratio = 0.0

        features.append(resonance_ratio)

        # ==================================================================
        # Feature 9: Contact Duration Estimation
        # How long is the signal above a threshold?
        # ==================================================================
        # Threshold at 20% of peak amplitude
        threshold = 0.2 * np.max(amplitude_envelope)
        above_threshold = amplitude_envelope > threshold
        contact_duration_samples = np.sum(above_threshold)
        contact_duration_sec = contact_duration_samples / self.sr

        features.append(contact_duration_sec)

        # ==================================================================
        # Feature 10: High-Frequency Damping Ratio
        # Contact specifically damps high frequencies (>2kHz)
        # ==================================================================
        # Compare high-frequency energy to low-frequency energy
        low_mask = (freqs >= 100) & (freqs <= 1000)
        high_mask = (freqs >= 2000) & (freqs <= 8000)

        if np.any(low_mask) and np.any(high_mask):
            low_energy = np.sum(power_spectrum[low_mask])
            high_energy = np.sum(power_spectrum[high_mask])
            # Ratio: lower means more HF damping (typical for contact)
            hf_damping_ratio = high_energy / (low_energy + 1e-10)
        else:
            hf_damping_ratio = 0.0

        features.append(hf_damping_ratio)

        return features

    def _extract_legacy_features(self, audio: np.ndarray) -> pd.Series:
        """Extract legacy STFT features for compatibility."""
        stft = librosa.stft(audio, n_fft=self.n_fft)
        features = np.abs(stft).sum(axis=1)
        return pd.Series(features, index=self.freq_bins)

    def get_feature_names(self, method: str = "comprehensive") -> List[str]:
        """Get feature names for the specified extraction method."""
        if method == "comprehensive":
            names = []
            # Spectral features
            names.extend(
                [
                    "spectral_centroid",
                    "spectral_bandwidth",
                    "spectral_rolloff",
                    "spectral_flatness",
                ]
                + [f"spectral_contrast_{i}" for i in range(5)]
            )
            # Resonance features
            names.extend(
                [
                    "resonance_peak_amp",
                    "resonance_peak_freq",
                    "resonance_energy",
                    "resonance_q_factor",
                    "resonance_skewness",
                ]
            )
            # Damping features
            names.extend(
                [
                    "damping_ratio",
                    "high_freq_slope",
                    "high_freq_decay_rate",
                    "ultra_high_ratio",
                ]
            )
            # Contact features
            names.extend(
                [
                    "burst_rms",
                    "burst_peak",
                    "burst_crest_factor",
                    "time_to_peak",
                    "zero_crossing_rate",
                    "temporal_centroid",
                ]
            )
            # Envelope features
            names.extend(
                [
                    "env_mean",
                    "env_std",
                    "env_max",
                    "env_min",
                    "env_skew",
                    "env_kurtosis",
                    "env_decay_rate",
                ]
            )
            # Energy features
            band_names = [f"{band}_energy_ratio" for band in self.freq_bands.keys()]
            names.extend(band_names + ["resonance_high_ratio", "low_mid_ratio"])

            # Workspace-invariant features (optional)
            if self.use_workspace_invariant:
                names.extend(
                    [
                        "wi_mid_to_high_ratio",
                        "wi_lowmid_to_midhigh_ratio",
                        "wi_verylow_to_high_ratio",
                        "wi_combined_contact_ratio",
                        "wi_norm_centroid",
                        "wi_norm_spread_ratio",
                        "wi_norm_skewness",
                        "wi_attack_decay_ratio",
                        "wi_attack_slope",
                        "wi_decay_rate",
                        "wi_avg_spectral_flux",
                        "wi_std_spectral_flux",
                        "wi_zcr_full",
                        "wi_zcr_ratio",
                        "wi_crest_lowmid",
                        "wi_crest_mid",
                        "wi_crest_midhigh",
                    ]
                )

            # Impulse response / Transfer function features (optional)
            if self.use_impulse_features:
                names.extend(
                    [
                        "ir_primary_resonance_freq",
                        "ir_primary_resonance_mag",
                        "ir_q_factor",
                        "ir_num_resonances",
                        "ir_secondary_resonance_freq",
                        "ir_secondary_resonance_mag",
                        "ir_primary_secondary_ratio",
                        "ir_freq_response_centroid",
                        "ir_freq_response_spread",
                        "ir_midlow_high_ratio",
                        "ir_low_mid_ratio",
                        "ir_decay_rate",
                        "ir_t60",
                        "ir_group_delay_mean",
                        "ir_group_delay_std",
                    ]
                )

            return names
        elif method == "legacy":
            return [f"freq_{f:.1f}Hz" for f in self.freq_bins]
        else:
            return [f"feature_{i}" for i in range(50)]  # Generic fallback

    def extract_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 64,
        fmin: float = 0,
        fmax: float = 8000,
        time_bins: int = 128,
        use_log_scale: bool = True,
    ) -> np.ndarray:
        """
        Extract mel spectrogram representation of audio signal.

        This provides a time-frequency representation that can be used as an
        alternative to hand-crafted features, especially for deep learning models.

        Args:
            audio: Input audio waveform
            n_fft: FFT window size (default: 512)
            hop_length: Hop length for STFT (default: 128)
            n_mels: Number of mel frequency bins (default: 64)
            fmin: Minimum frequency in Hz (default: 0)
            fmax: Maximum frequency in Hz (default: 8000)
            time_bins: Target number of time bins for consistent shape (default: 128)
            use_log_scale: Apply log scaling (dB scale) for better ML performance

        Returns:
            Mel spectrogram of shape (n_mels, time_bins)

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>> spectrogram = extractor.extract_spectrogram(audio)
            >>> print(spectrogram.shape)  # (64, 128)
        """
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Use instance sample rate if fmax is default
        if fmax == 8000 and hasattr(self, "sr"):
            fmax = min(8000, self.sr // 2)  # Nyquist limit

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        # Apply log scaling (convert to decibel scale)
        if use_log_scale:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize to fixed temporal dimension for consistent input shape
        if mel_spec.shape[1] != time_bins:
            # Pad or truncate to match time_bins
            if mel_spec.shape[1] < time_bins:
                # Pad with zeros
                pad_width = time_bins - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")
            else:
                # Truncate
                mel_spec = mel_spec[:, :time_bins]

        return mel_spec  # Shape: (n_mels, time_bins)

    def extract_features_or_spectrogram(
        self,
        audio: np.ndarray,
        mode: str = "features",
        spectrogram_params: Optional[Dict] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Unified interface for feature extraction with mode selection.

        This method allows easy switching between hand-crafted features,
        spectrograms, or both, making it simple to experiment with different
        input representations for ML models.

        Args:
            audio: Input audio waveform
            mode: Extraction mode - one of:
                  - "features": Hand-crafted feature vector (default)
                  - "spectrogram": Mel spectrogram (2D time-frequency)
                  - "both": Both features and spectrogram
            spectrogram_params: Optional dict of parameters for spectrogram extraction
                               (only used when mode="spectrogram" or mode="both")

        Returns:
            If mode="features": 1D feature vector (80 dims)
            If mode="spectrogram": 2D spectrogram (n_mels × time_bins)
            If mode="both": dict with keys "features" and "spectrogram"

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>>
            >>> # Extract features (default)
            >>> features = extractor.extract_features_or_spectrogram(audio, mode="features")
            >>>
            >>> # Extract spectrogram
            >>> spec = extractor.extract_features_or_spectrogram(audio, mode="spectrogram")
            >>>
            >>> # Extract both
            >>> both = extractor.extract_features_or_spectrogram(audio, mode="both")
            >>> features = both["features"]
            >>> spec = both["spectrogram"]
        """
        if mode == "features":
            return self.extract_features(audio, method="comprehensive")

        elif mode == "spectrogram":
            params = spectrogram_params or {}
            return self.extract_spectrogram(audio, **params)

        elif mode == "both":
            features = self.extract_features(audio, method="comprehensive")
            params = spectrogram_params or {}
            spectrogram = self.extract_spectrogram(audio, **params)
            return {"features": features, "spectrogram": spectrogram}

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'features', 'spectrogram', or 'both'"
            )

    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract Mel-frequency cepstral coefficients (MFCCs).

        MFCCs provide a compact representation of the spectral envelope,
        focusing on perceptually relevant frequency bands.

        Args:
            audio: Input audio waveform
            n_mfcc: Number of MFCC coefficients to return
            n_fft: FFT window size
            hop_length: Hop length between windows
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency (default: sr/2)

        Returns:
            MFCC matrix of shape (n_mfcc, time_frames)

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>> mfcc = extractor.extract_mfcc(audio, n_mfcc=13)
            >>> print(mfcc.shape)  # (13, time_frames)
        """
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        return mfcc  # Shape: (n_mfcc, time_frames)

    def extract_magnitude_spectrum(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract magnitude spectrum using STFT.

        This provides the raw frequency domain representation,
        showing magnitude at each frequency bin over time.

        Args:
            audio: Input audio waveform
            n_fft: FFT window size
            hop_length: Hop length between windows
            normalize: Whether to normalize by maximum value

        Returns:
            Magnitude spectrum of shape (n_fft//2 + 1, time_frames)

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>> mag_spec = extractor.extract_magnitude_spectrum(audio)
            >>> print(mag_spec.shape)  # (1025, time_frames)
        """
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Compute STFT
        stft = librosa.stft(
            y=audio,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Get magnitude spectrum
        mag_spec = np.abs(stft)

        # Normalize if requested
        if normalize and np.max(mag_spec) > 0:
            mag_spec = mag_spec / np.max(mag_spec)

        return mag_spec  # Shape: (n_fft//2 + 1, time_frames)

    def extract_power_spectrum(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract power spectrum using STFT.

        Power spectrum emphasizes energy distribution across frequencies,
        which may be more relevant for contact detection.

        Args:
            audio: Input audio waveform
            n_fft: FFT window size
            hop_length: Hop length between windows
            normalize: Whether to normalize by maximum value

        Returns:
            Power spectrum of shape (n_fft//2 + 1, time_frames)

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>> power_spec = extractor.extract_power_spectrum(audio)
            >>> print(power_spec.shape)  # (1025, time_frames)
        """
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Compute STFT
        stft = librosa.stft(
            y=audio,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # Get power spectrum (magnitude squared)
        power_spec = np.abs(stft) ** 2

        # Normalize if requested
        if normalize and np.max(power_spec) > 0:
            power_spec = power_spec / np.max(power_spec)

        return power_spec  # Shape: (n_fft//2 + 1, time_frames)

    def extract_chroma_features(
        self,
        audio: np.ndarray,
        n_chroma: int = 12,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Extract chroma features representing pitch classes.

        Chroma features fold the spectrum into 12 pitch classes,
        useful if there are harmonic patterns in the contact sounds.

        Args:
            audio: Input audio waveform
            n_chroma: Number of chroma bins (12 for full octave)
            n_fft: FFT window size
            hop_length: Hop length between windows

        Returns:
            Chroma features of shape (n_chroma, time_frames)

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>> chroma = extractor.extract_chroma_features(audio)
            >>> print(chroma.shape)  # (12, time_frames)
        """
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_chroma=n_chroma,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        return chroma  # Shape: (n_chroma, time_frames)

    # GPU-Accelerated Methods
    def extract_mfcc_gpu(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: Optional[int] = None,
    ) -> np.ndarray:
        """
        GPU-accelerated MFCC extraction using TorchAudio.

        Significantly faster than CPU-based librosa for large datasets.

        Args:
            audio: Input audio waveform
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length between windows
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency

        Returns:
            MFCC matrix of shape (n_mfcc, time_frames)
        """
        if not HAS_GPU_SUPPORT:
            # Fallback to CPU method
            return self.extract_mfcc(
                audio, n_mfcc, n_fft, hop_length, n_mels, fmin, fmax
            )

        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Convert to torch tensor and move to GPU
        audio_tensor = torch.from_numpy(audio).float().cuda()

        # Create MFCC transform
        mfcc_transform = T.MFCC(
            sample_rate=self.sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "f_min": fmin,
                "f_max": fmax if fmax else self.sr // 2,
            },
        ).cuda()

        # Extract MFCCs
        mfcc = mfcc_transform(audio_tensor.unsqueeze(0))  # Add batch dimension

        # Remove batch dimension and convert back to numpy
        return mfcc.squeeze(0).cpu().numpy()

    def extract_magnitude_spectrum_gpu(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        GPU-accelerated magnitude spectrum extraction using TorchAudio.

        Args:
            audio: Input audio waveform
            n_fft: FFT window size
            hop_length: Hop length between windows
            normalize: Whether to normalize by maximum value

        Returns:
            Magnitude spectrum of shape (n_fft//2 + 1, time_frames)
        """
        if not HAS_GPU_SUPPORT:
            return self.extract_magnitude_spectrum(audio, n_fft, hop_length, normalize)

        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Convert to torch tensor and move to GPU
        audio_tensor = torch.from_numpy(audio).float().cuda()

        # Compute STFT
        stft = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).cuda(),
            return_complex=True,
        )

        # Get magnitude spectrum
        mag_spec = torch.abs(stft)

        # Normalize if requested
        if normalize and torch.max(mag_spec) > 0:
            mag_spec = mag_spec / torch.max(mag_spec)

        # Convert back to numpy
        return mag_spec.cpu().numpy()

    def extract_power_spectrum_gpu(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        GPU-accelerated power spectrum extraction using TorchAudio.

        Args:
            audio: Input audio waveform
            n_fft: FFT window size
            hop_length: Hop length between windows
            normalize: Whether to normalize by maximum value

        Returns:
            Power spectrum of shape (n_fft//2 + 1, time_frames)
        """
        if not HAS_GPU_SUPPORT:
            return self.extract_power_spectrum(audio, n_fft, hop_length, normalize)

        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Convert to torch tensor and move to GPU
        audio_tensor = torch.from_numpy(audio).float().cuda()

        # Compute STFT
        stft = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).cuda(),
            return_complex=True,
        )

        # Get power spectrum (magnitude squared)
        power_spec = torch.abs(stft) ** 2

        # Normalize if requested
        if normalize and torch.max(power_spec) > 0:
            power_spec = power_spec / torch.max(power_spec)

        # Convert back to numpy
        return power_spec.cpu().numpy()

    def extract_spectrogram_gpu(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: int = 0,
        fmax: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        GPU-accelerated mel spectrogram extraction using TorchAudio.

        Args:
            audio: Input audio waveform
            n_fft: FFT window size
            hop_length: Hop length between windows
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency
            normalize: Whether to normalize by maximum value

        Returns:
            Mel spectrogram of shape (n_mels, time_frames)
        """
        if not HAS_GPU_SUPPORT:
            return self.extract_spectrogram(
                audio, n_fft, hop_length, n_mels, fmin, fmax, normalize
            )

        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Convert to torch tensor and move to GPU
        audio_tensor = torch.from_numpy(audio).float().cuda()

        # Create mel spectrogram transform
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax if fmax else self.sr // 2,
        ).cuda()

        # Extract mel spectrogram
        mel_spec = mel_transform(audio_tensor.unsqueeze(0))  # Add batch dimension

        # Convert to decibels (log scale)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # Normalize if requested
        if normalize:
            mel_spec_db = (mel_spec_db - torch.mean(mel_spec_db)) / (
                torch.std(mel_spec_db) + 1e-8
            )

        # Remove batch dimension and convert back to numpy
        return mel_spec_db.squeeze(0).cpu().numpy()


def extract_features(
    audio: np.ndarray, method: str = "comprehensive", sr: int = 48000, **kwargs
) -> Union[np.ndarray, pd.Series]:
    """
    Convenience function for feature extraction.

    Args:
        audio: Input audio waveform
        method: Feature extraction method
        sr: Sample rate
        **kwargs: Additional arguments for feature extractor

    Returns:
        Extracted features
    """
    # Ensure audio is floating-point
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32)

    extractor = GeometricFeatureExtractor(sr=sr, **kwargs)
    return extractor.extract_features(audio, method=method)

    def extract_spectrogram(
        self,
        audio: np.ndarray,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 64,
        fmin: float = 0,
        fmax: float = 8000,
        time_bins: int = 128,
        use_log_scale: bool = True,
    ) -> np.ndarray:
        """
        Extract mel spectrogram representation of audio signal.

        This provides a time-frequency representation that can be used as an
        alternative to hand-crafted features, especially for deep learning models.

        Args:
            audio: Input audio waveform
            n_fft: FFT window size (default: 512)
            hop_length: Hop length for STFT (default: 128)
            n_mels: Number of mel frequency bins (default: 64)
            fmin: Minimum frequency in Hz (default: 0)
            fmax: Maximum frequency in Hz (default: 8000)
            time_bins: Target number of time bins for consistent shape (default: 128)
            use_log_scale: Apply log scaling (dB scale) for better ML performance

        Returns:
            Mel spectrogram of shape (n_mels, time_bins)

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>> spectrogram = extractor.extract_spectrogram(audio)
            >>> print(spectrogram.shape)  # (64, 128)
        """
        # Ensure audio is floating-point
        if audio.dtype != np.float32 and audio.dtype != np.float64:
            audio = audio.astype(np.float32)

        # Use instance sample rate if fmax is default
        if fmax == 8000 and hasattr(self, "sr"):
            fmax = min(8000, self.sr // 2)  # Nyquist limit

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        # Apply log scaling (convert to decibel scale)
        if use_log_scale:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize to fixed temporal dimension for consistent input shape
        if mel_spec.shape[1] != time_bins:
            # Pad or truncate to match time_bins
            if mel_spec.shape[1] < time_bins:
                # Pad with zeros
                pad_width = time_bins - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")
            else:
                # Truncate
                mel_spec = mel_spec[:, :time_bins]

        return mel_spec  # Shape: (n_mels, time_bins)

    def extract_features_or_spectrogram(
        self,
        audio: np.ndarray,
        mode: str = "features",
        spectrogram_params: Optional[Dict] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Unified interface for feature extraction with mode selection.

        This method allows easy switching between hand-crafted features,
        spectrograms, or both, making it simple to experiment with different
        input representations for ML models.

        Args:
            audio: Input audio waveform
            mode: Extraction mode - one of:
                  - "features": Hand-crafted feature vector (default)
                  - "spectrogram": Mel spectrogram (2D time-frequency)
                  - "both": Both features and spectrogram
            spectrogram_params: Optional dict of parameters for spectrogram extraction
                               (only used when mode="spectrogram" or mode="both")

        Returns:
            If mode="features": 1D feature vector (80 dims)
            If mode="spectrogram": 2D spectrogram (n_mels × time_bins)
            If mode="both": dict with keys "features" and "spectrogram"

        Example:
            >>> extractor = GeometricFeatureExtractor(sr=48000)
            >>> audio = load_audio("contact.wav")
            >>>
            >>> # Extract features (default)
            >>> features = extractor.extract_features_or_spectrogram(audio, mode="features")
            >>>
            >>> # Extract spectrogram
            >>> spec = extractor.extract_features_or_spectrogram(audio, mode="spectrogram")
            >>>
            >>> # Extract both
            >>> both = extractor.extract_features_or_spectrogram(audio, mode="both")
            >>> features = both["features"]
            >>> spec = both["spectrogram"]
        """
        if mode == "features":
            return self.extract_features(audio, method="comprehensive")

        elif mode == "spectrogram":
            params = spectrogram_params or {}
            return self.extract_spectrogram(audio, **params)

        elif mode == "both":
            features = self.extract_features(audio, method="comprehensive")
            params = spectrogram_params or {}
            spectrogram = self.extract_spectrogram(audio, **params)
            return {"features": features, "spectrogram": spectrogram}

        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'features', 'spectrogram', or 'both'"
            )


# Backward compatibility functions
def audio_to_features(
    audio: np.ndarray,
    method: str = "stft",
    n_fft: int = 4096,
    in_dB: bool = False,
    sr: int = 48000,
) -> Union[pd.Series, np.ndarray]:
    """
    Legacy function for backward compatibility with existing code.

    Args:
        audio: Audio waveform
        method: Feature extraction method
        n_fft: FFT size for STFT
        in_dB: Convert to dB scale
        sr: Sample rate

    Returns:
        Feature vector or Series
    """
    extractor = GeometricFeatureExtractor(sr=sr, n_fft=n_fft)

    if method == "stft":
        features = extractor._extract_legacy_features(audio)
        if in_dB:
            features = pd.Series(
                librosa.amplitude_to_db(features.values, ref=np.max(features.values)),
                index=features.index,
            )
        return features
    elif method == "comprehensive":
        return extractor.extract_features(audio, method="comprehensive")
    elif method == "combined":
        # Legacy combined method
        stft_features = extractor._extract_legacy_features(audio).values
        resonance_features = extractor._extract_resonance_features(audio)
        contact_features = extractor._extract_contact_features(audio)
        combined = np.concatenate([stft_features, resonance_features, contact_features])
        return combined
    else:
        return extractor.extract_features(audio, method=method)
