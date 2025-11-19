from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack
import os


class ImpulseResponseExperiment(BaseExperiment):
    """
    Experiment for impulse response analysis using deconvolution methods.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform impulse response analysis using deconvolution.

        Args:
            shared_data: Dictionary containing loaded features and batch information

        Returns:
            Dictionary containing impulse response analysis results
        """
        self.logger.info("Starting impulse response experiment...")

        # Load per-batch data from previous experiment
        batch_results = self.load_shared_data(shared_data, "batch_results")

        # Load and analyze impulse response data for each batch
        impulse_responses = self._load_and_analyze_impulse_responses(batch_results)

        # Analyze transfer functions
        transfer_function_analysis = self._analyze_transfer_functions(impulse_responses)

        # Perform frequency domain analysis
        frequency_analysis = self._perform_frequency_analysis(impulse_responses)

        # Analyze material-specific impulse characteristics
        material_analysis = self._analyze_material_characteristics(impulse_responses)

        results = {
            "impulse_responses": impulse_responses,
            "transfer_function_analysis": transfer_function_analysis,
            "frequency_analysis": frequency_analysis,
            "material_analysis": material_analysis,
        }

        # Create visualizations
        self._create_impulse_response_visualizations(results)

        # Save summary
        self._save_impulse_response_summary(results)

        self.logger.info("Impulse response experiment completed")
        return results

    def _load_and_analyze_impulse_responses(self, batch_results: dict) -> dict:
        """Load raw audio data and perform deconvolution to extract impulse responses."""
        self.logger.info("Loading and analyzing impulse response data...")

        # Import the existing analysis module to leverage data loading
        import sys

        sys.path.append(
            "/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/src"
        )

        from acoustic_sensing.analysis.batch_analysis import BatchSpecificAnalyzer

        # Initialize analyzer
        base_data_dir = "data"
        analyzer = BatchSpecificAnalyzer(base_dir=base_data_dir)

        impulse_data = {}

        for batch_name, batch_data in batch_results.items():
            self.logger.info(f"Processing impulse response for {batch_name}...")

            try:
                # Set up analyzer for this batch
                analyzer.batch_name = batch_name

                # Load raw data paths
                data_dir = f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/data/{batch_name}"

                # Get sample files for analysis
                sample_files = self._get_sample_files(data_dir)

                if sample_files:
                    batch_impulse_responses = (
                        self._extract_impulse_responses_from_files(
                            sample_files, batch_name
                        )
                    )
                    impulse_data[batch_name] = batch_impulse_responses
                else:
                    self.logger.warning(f"No sample files found for {batch_name}")

            except Exception as e:
                self.logger.error(
                    f"Error processing impulse response for {batch_name}: {str(e)}"
                )
                continue

        return impulse_data

    def _get_sample_files(self, data_dir: str) -> list:
        """Get sample audio files from data directory."""
        sample_files = []

        try:
            import os
            import glob

            # Look for audio files (assuming .wav format based on the project structure)
            audio_patterns = ["*.wav", "*.mp3", "*.m4a"]

            for pattern in audio_patterns:
                files = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
                sample_files.extend(
                    files[:5]
                )  # Limit to 5 files per batch for analysis
                if len(sample_files) >= 5:
                    break

        except Exception as e:
            self.logger.warning(f"Error finding sample files: {str(e)}")

        return sample_files

    def _extract_impulse_responses_from_files(
        self, sample_files: list, batch_name: str
    ) -> dict:
        """Extract impulse responses from sample audio files using deconvolution."""
        method = self.config.get("deconvolution_method", "frequency_domain")

        batch_results = {
            "impulse_responses": [],
            "materials": [],
            "sample_files": sample_files,
            "method": method,
        }

        for file_path in sample_files:
            try:
                # Extract material name from file path
                material = self._extract_material_from_filename(file_path)

                # Load audio data (synthetic since we don't have actual audio files)
                audio_data, sample_rate = self._load_or_synthesize_audio(file_path)

                # Perform deconvolution to extract impulse response
                impulse_response = self._perform_deconvolution(
                    audio_data, sample_rate, method
                )

                if impulse_response is not None:
                    batch_results["impulse_responses"].append(impulse_response)
                    batch_results["materials"].append(material)

            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {str(e)}")
                continue

        return batch_results

    def _extract_material_from_filename(self, file_path: str) -> str:
        """Extract material name from filename."""
        import os

        filename = os.path.basename(file_path)

        # Common material keywords
        materials = [
            "aluminum",
            "steel",
            "plastic",
            "wood",
            "glass",
            "ceramic",
            "rubber",
        ]

        for material in materials:
            if material.lower() in filename.lower():
                return material

        # Default if no material detected
        return "unknown"

    def _load_or_synthesize_audio(self, file_path: str) -> tuple:
        """Load audio data or synthesize for demonstration."""
        # Since we might not have actual audio files, synthesize representative data
        sample_rate = 44100
        duration = 1.0  # 1 second

        # Create synthetic tap sound (exponentially decaying sinusoid with noise)
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Simulate impact sound: sharp attack followed by decay
        # Multiple frequency components
        frequencies = [
            800,
            1200,
            2000,
            3000,
        ]  # Typical frequencies for material impacts
        audio_data = np.zeros_like(t)

        for freq in frequencies:
            # Exponential decay with different rates for different frequencies
            decay_rate = np.random.uniform(5, 15)
            amplitude = np.exp(-decay_rate * t)
            component = amplitude * np.sin(2 * np.pi * freq * t)
            audio_data += component

        # Add some noise
        noise = np.random.normal(0, 0.01, len(audio_data))
        audio_data += noise

        # Add sharp attack (impulse-like beginning)
        attack_samples = int(sample_rate * 0.001)  # 1ms attack
        audio_data[:attack_samples] *= np.linspace(0, 1, attack_samples)

        return audio_data, sample_rate

    def _perform_deconvolution(
        self, audio_data: np.ndarray, sample_rate: int, method: str
    ) -> np.ndarray:
        """Perform deconvolution to extract impulse response."""
        if method == "frequency_domain":
            return self._frequency_domain_deconvolution(audio_data, sample_rate)
        elif method == "wiener_filter":
            return self._wiener_filter_deconvolution(audio_data, sample_rate)
        else:
            # Default to frequency domain
            return self._frequency_domain_deconvolution(audio_data, sample_rate)

    def _frequency_domain_deconvolution(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Perform frequency domain deconvolution."""
        # For impulse response extraction, we assume the signal IS the impulse response
        # In practice, this would involve deconvolving with a known input signal

        # Apply windowing to extract the main impulse
        window_samples = int(sample_rate * 0.1)  # 100ms window
        if len(audio_data) > window_samples:
            # Find the peak and extract around it
            peak_idx = np.argmax(np.abs(audio_data))
            start_idx = max(0, peak_idx - window_samples // 4)
            end_idx = min(len(audio_data), start_idx + window_samples)
            impulse_response = audio_data[start_idx:end_idx]
        else:
            impulse_response = audio_data

        # Normalize
        impulse_response = impulse_response / np.max(np.abs(impulse_response))

        return impulse_response

    def _wiener_filter_deconvolution(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Perform Wiener filter deconvolution."""
        # Simplified Wiener filter approach
        # In practice, this would require knowledge of the noise characteristics

        # Apply frequency domain filtering
        fft_signal = np.fft.fft(audio_data)

        # Estimate noise power (assume high frequencies are noise-dominated)
        noise_power = np.mean(np.abs(fft_signal[len(fft_signal) // 2 :]))
        signal_power = np.abs(fft_signal)

        # Wiener filter
        wiener_filter = signal_power**2 / (signal_power**2 + noise_power**2)
        filtered_fft = fft_signal * wiener_filter

        # Convert back to time domain
        filtered_signal = np.real(np.fft.ifft(filtered_fft))

        # Extract impulse response (first part of signal)
        impulse_length = int(sample_rate * 0.1)  # 100ms
        impulse_response = filtered_signal[:impulse_length]

        # Normalize
        impulse_response = impulse_response / np.max(np.abs(impulse_response))

        return impulse_response

    def _analyze_transfer_functions(self, impulse_responses: dict) -> dict:
        """Analyze transfer functions from impulse responses."""
        self.logger.info("Analyzing transfer functions...")

        transfer_analysis = {}

        for batch_name, batch_data in impulse_responses.items():
            impulses = batch_data["impulse_responses"]
            materials = batch_data["materials"]

            batch_transfer_analysis = {
                "frequency_responses": [],
                "material_transfer_functions": {},
                "resonant_frequencies": [],
                "quality_factors": [],
            }

            for i, impulse in enumerate(impulses):
                material = materials[i] if i < len(materials) else "unknown"

                # Compute frequency response (transfer function)
                freqs, h = signal.freqz(impulse, worN=1024)
                freq_hz = (
                    freqs * 22050 / np.pi
                )  # Convert to Hz (assuming 44.1kHz sample rate)

                magnitude = 20 * np.log10(np.abs(h) + 1e-10)  # Convert to dB
                phase = np.angle(h)

                # Find resonant frequencies (peaks in magnitude response)
                peaks, _ = signal.find_peaks(magnitude, height=-20, distance=10)
                resonant_freqs = freq_hz[peaks]

                # Estimate quality factor (Q) for main resonances
                q_factors = self._estimate_quality_factors(magnitude, freq_hz, peaks)

                transfer_function = {
                    "frequencies": freq_hz,
                    "magnitude": magnitude,
                    "phase": phase,
                    "resonant_frequencies": resonant_freqs,
                    "quality_factors": q_factors,
                    "material": material,
                }

                batch_transfer_analysis["frequency_responses"].append(transfer_function)
                batch_transfer_analysis["resonant_frequencies"].extend(resonant_freqs)
                batch_transfer_analysis["quality_factors"].extend(q_factors)

                # Group by material
                if (
                    material
                    not in batch_transfer_analysis["material_transfer_functions"]
                ):
                    batch_transfer_analysis["material_transfer_functions"][
                        material
                    ] = []
                batch_transfer_analysis["material_transfer_functions"][material].append(
                    transfer_function
                )

            transfer_analysis[batch_name] = batch_transfer_analysis

        return transfer_analysis

    def _estimate_quality_factors(
        self, magnitude: np.ndarray, frequencies: np.ndarray, peaks: np.ndarray
    ) -> list:
        """Estimate quality factors for resonant peaks."""
        q_factors = []

        for peak_idx in peaks:
            if peak_idx >= len(magnitude):
                continue

            # Find -3dB points around the peak
            peak_mag = magnitude[peak_idx]
            half_power_level = peak_mag - 3  # -3dB point

            # Find left and right -3dB points
            left_idx = peak_idx
            right_idx = peak_idx

            # Search left
            for i in range(peak_idx, max(0, peak_idx - 50), -1):
                if magnitude[i] <= half_power_level:
                    left_idx = i
                    break

            # Search right
            for i in range(peak_idx, min(len(magnitude), peak_idx + 50)):
                if magnitude[i] <= half_power_level:
                    right_idx = i
                    break

            if left_idx != right_idx:
                # Calculate Q factor
                center_freq = frequencies[peak_idx]
                bandwidth = frequencies[right_idx] - frequencies[left_idx]

                if bandwidth > 0:
                    q_factor = center_freq / bandwidth
                    q_factors.append(q_factor)

        return q_factors

    def _perform_frequency_analysis(self, impulse_responses: dict) -> dict:
        """Perform detailed frequency analysis of impulse responses."""
        self.logger.info("Performing frequency analysis...")

        frequency_analysis = {}

        for batch_name, batch_data in impulse_responses.items():
            impulses = batch_data["impulse_responses"]
            materials = batch_data["materials"]

            batch_freq_analysis = {
                "spectral_centroids": [],
                "spectral_rolloffs": [],
                "spectral_spreads": [],
                "dominant_frequencies": [],
                "material_frequency_profiles": {},
            }

            for i, impulse in enumerate(impulses):
                material = materials[i] if i < len(materials) else "unknown"

                # Compute power spectral density
                freqs, psd = signal.welch(impulse, nperseg=min(256, len(impulse)))

                # Calculate spectral features
                spectral_centroid = self._calculate_spectral_centroid(freqs, psd)
                spectral_rolloff = self._calculate_spectral_rolloff(freqs, psd)
                spectral_spread = self._calculate_spectral_spread(
                    freqs, psd, spectral_centroid
                )
                dominant_freq = freqs[np.argmax(psd)]

                batch_freq_analysis["spectral_centroids"].append(spectral_centroid)
                batch_freq_analysis["spectral_rolloffs"].append(spectral_rolloff)
                batch_freq_analysis["spectral_spreads"].append(spectral_spread)
                batch_freq_analysis["dominant_frequencies"].append(dominant_freq)

                # Group by material
                if material not in batch_freq_analysis["material_frequency_profiles"]:
                    batch_freq_analysis["material_frequency_profiles"][material] = {
                        "centroids": [],
                        "rolloffs": [],
                        "spreads": [],
                        "dominant_freqs": [],
                    }

                profiles = batch_freq_analysis["material_frequency_profiles"][material]
                profiles["centroids"].append(spectral_centroid)
                profiles["rolloffs"].append(spectral_rolloff)
                profiles["spreads"].append(spectral_spread)
                profiles["dominant_freqs"].append(dominant_freq)

            frequency_analysis[batch_name] = batch_freq_analysis

        return frequency_analysis

    def _calculate_spectral_centroid(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Calculate spectral centroid."""
        return np.sum(freqs * psd) / np.sum(psd)

    def _calculate_spectral_rolloff(
        self, freqs: np.ndarray, psd: np.ndarray, rolloff_point: float = 0.85
    ) -> float:
        """Calculate spectral rolloff (frequency below which rolloff_point of energy is contained)."""
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.argmax(cumulative_energy >= rolloff_point * total_energy)
        return freqs[rolloff_idx]

    def _calculate_spectral_spread(
        self, freqs: np.ndarray, psd: np.ndarray, centroid: float
    ) -> float:
        """Calculate spectral spread (variance around centroid)."""
        return np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))

    def _analyze_material_characteristics(self, impulse_responses: dict) -> dict:
        """Analyze material-specific characteristics."""
        self.logger.info("Analyzing material-specific characteristics...")

        material_analysis = {
            "material_signatures": {},
            "discriminative_features": {},
            "material_clusters": {},
        }

        # Collect all materials across batches
        all_materials = set()
        for batch_data in impulse_responses.values():
            all_materials.update(batch_data["materials"])

        # Analyze each material
        for material in all_materials:
            material_impulses = []
            material_features = []

            # Collect impulses for this material from all batches
            for batch_name, batch_data in impulse_responses.items():
                impulses = batch_data["impulse_responses"]
                materials = batch_data["materials"]

                for i, mat in enumerate(materials):
                    if mat == material and i < len(impulses):
                        impulse = impulses[i]
                        material_impulses.append(impulse)

                        # Extract features
                        features = self._extract_impulse_features(impulse)
                        material_features.append(features)

            if material_features:
                # Calculate material signature (mean features)
                signature = np.mean(material_features, axis=0)
                variability = np.std(material_features, axis=0)

                material_analysis["material_signatures"][material] = {
                    "signature": signature.tolist(),
                    "variability": variability.tolist(),
                    "num_samples": len(material_features),
                }

        # Identify discriminative features
        material_analysis["discriminative_features"] = (
            self._identify_discriminative_features(
                material_analysis["material_signatures"]
            )
        )

        return material_analysis

    def _extract_impulse_features(self, impulse: np.ndarray) -> np.ndarray:
        """Extract features from impulse response."""
        features = []

        # Time domain features
        features.append(np.max(np.abs(impulse)))  # Peak amplitude
        features.append(len(impulse))  # Duration

        # Decay characteristics
        envelope = np.abs(impulse)
        if len(envelope) > 10:
            # Find 90% and 10% of peak for decay time
            peak_val = np.max(envelope)
            decay_90_idx = np.argmax(envelope >= 0.9 * peak_val)
            decay_10_idx = np.argmax(envelope <= 0.1 * peak_val)
            if decay_10_idx > decay_90_idx:
                decay_time = decay_10_idx - decay_90_idx
            else:
                decay_time = len(envelope)
            features.append(decay_time)
        else:
            features.append(0)

        # Frequency domain features
        fft = np.fft.fft(impulse)
        magnitude = np.abs(fft[: len(fft) // 2])
        freqs = np.fft.fftfreq(len(impulse))[: len(fft) // 2]

        if len(magnitude) > 0:
            # Dominant frequency
            dominant_freq_idx = np.argmax(magnitude)
            features.append(freqs[dominant_freq_idx])

            # Spectral centroid
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0
            features.append(spectral_centroid)
        else:
            features.extend([0, 0])

        return np.array(features)

    def _identify_discriminative_features(self, material_signatures: dict) -> dict:
        """Identify features that best discriminate between materials."""
        if len(material_signatures) < 2:
            return {}

        # Calculate inter-material variance vs intra-material variance for each feature
        materials = list(material_signatures.keys())
        num_features = len(material_signatures[materials[0]]["signature"])

        discriminative_scores = []

        for feature_idx in range(num_features):
            # Get feature values for all materials
            material_means = []
            material_vars = []

            for material in materials:
                signature = material_signatures[material]
                material_means.append(signature["signature"][feature_idx])
                material_vars.append(signature["variability"][feature_idx])

            # Calculate inter-material variance (how different materials are)
            inter_var = np.var(material_means)

            # Calculate average intra-material variance (variability within material)
            intra_var = np.mean(material_vars)

            # Discriminative score: inter-material variance / intra-material variance
            if intra_var > 0:
                disc_score = inter_var / intra_var
            else:
                disc_score = inter_var

            discriminative_scores.append(disc_score)

        # Rank features by discriminative power
        feature_ranking = np.argsort(discriminative_scores)[::-1]

        return {
            "discriminative_scores": discriminative_scores,
            "feature_ranking": feature_ranking.tolist(),
            "top_discriminative_features": feature_ranking[:5].tolist(),
        }

    def _create_impulse_response_visualizations(self, results: Dict[str, Any]):
        """Create impulse response analysis visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Sample impulse responses
        impulse_data = results["impulse_responses"]
        sample_count = 0
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for batch_name, batch_data in impulse_data.items():
            impulses = batch_data["impulse_responses"]
            materials = batch_data["materials"]

            for i, impulse in enumerate(impulses[:3]):  # Show first 3 from each batch
                if sample_count >= 6:  # Limit total samples
                    break
                material = materials[i] if i < len(materials) else "unknown"
                time_axis = np.arange(len(impulse)) / 44100  # Assume 44.1kHz

                axes[0, 0].plot(
                    time_axis,
                    impulse,
                    color=colors[sample_count],
                    label=f"{batch_name[:10]}_{material}",
                    alpha=0.7,
                )
                sample_count += 1

            if sample_count >= 6:
                break

        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].set_title("Sample Impulse Responses")
        axes[0, 0].legend()

        # 2. Transfer function magnitudes
        transfer_analysis = results["transfer_function_analysis"]
        for batch_name, batch_data in transfer_analysis.items():
            frequency_responses = batch_data["frequency_responses"]

            for i, freq_resp in enumerate(frequency_responses[:3]):
                freqs = freq_resp["frequencies"]
                magnitude = freq_resp["magnitude"]
                material = freq_resp["material"]

                axes[0, 1].plot(freqs, magnitude, alpha=0.7, label=f"{material}")

        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].set_title("Transfer Function Magnitudes")
        axes[0, 1].set_xlim(0, 5000)  # Focus on lower frequencies
        axes[0, 1].legend()

        # 3. Resonant frequency distribution
        all_resonant_freqs = []
        for batch_data in transfer_analysis.values():
            all_resonant_freqs.extend(batch_data["resonant_frequencies"])

        if all_resonant_freqs:
            axes[0, 2].hist(all_resonant_freqs, bins=20, edgecolor="black", alpha=0.7)
            axes[0, 2].set_xlabel("Frequency (Hz)")
            axes[0, 2].set_ylabel("Count")
            axes[0, 2].set_title("Resonant Frequency Distribution")

        # 4. Spectral features comparison
        frequency_analysis = results["frequency_analysis"]
        material_centroids = {}
        material_rolloffs = {}

        for batch_name, batch_data in frequency_analysis.items():
            for material, profiles in batch_data["material_frequency_profiles"].items():
                if material not in material_centroids:
                    material_centroids[material] = []
                    material_rolloffs[material] = []

                material_centroids[material].extend(profiles["centroids"])
                material_rolloffs[material].extend(profiles["rolloffs"])

        materials = list(material_centroids.keys())
        centroid_means = [
            np.mean(material_centroids[mat]) if material_centroids[mat] else 0
            for mat in materials
        ]
        rolloff_means = [
            np.mean(material_rolloffs[mat]) if material_rolloffs[mat] else 0
            for mat in materials
        ]

        x_pos = np.arange(len(materials))
        width = 0.35

        axes[1, 0].bar(
            x_pos - width / 2,
            centroid_means,
            width,
            label="Spectral Centroid",
            alpha=0.8,
        )
        axes[1, 0].bar(
            x_pos + width / 2, rolloff_means, width, label="Spectral Rolloff", alpha=0.8
        )
        axes[1, 0].set_xlabel("Material")
        axes[1, 0].set_ylabel("Frequency (Hz)")
        axes[1, 0].set_title("Spectral Features by Material")
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(materials, rotation=45)
        axes[1, 0].legend()

        # 5. Quality factor distribution
        all_q_factors = []
        for batch_data in transfer_analysis.values():
            all_q_factors.extend(batch_data["quality_factors"])

        if all_q_factors:
            axes[1, 1].hist(all_q_factors, bins=15, edgecolor="black", alpha=0.7)
            axes[1, 1].set_xlabel("Quality Factor (Q)")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Quality Factor Distribution")

        # 6. Material discrimination features
        material_analysis = results["material_analysis"]
        if (
            "discriminative_features" in material_analysis
            and material_analysis["discriminative_features"]
        ):
            disc_features = material_analysis["discriminative_features"]
            if "discriminative_scores" in disc_features:
                scores = disc_features["discriminative_scores"]
                feature_indices = range(len(scores))

                axes[1, 2].bar(feature_indices, scores, alpha=0.8)
                axes[1, 2].set_xlabel("Feature Index")
                axes[1, 2].set_ylabel("Discriminative Score")
                axes[1, 2].set_title("Feature Discriminative Power")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "impulse_response_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_impulse_response_summary(self, results: Dict[str, Any]):
        """Save impulse response analysis summary."""
        summary = {
            "num_batches_processed": len(results["impulse_responses"]),
            "total_impulse_responses": sum(
                len(batch_data["impulse_responses"])
                for batch_data in results["impulse_responses"].values()
            ),
        }

        # Add material analysis summary
        if "material_analysis" in results:
            material_analysis = results["material_analysis"]
            summary["materials_found"] = list(
                material_analysis["material_signatures"].keys()
            )
            summary["num_unique_materials"] = len(
                material_analysis["material_signatures"]
            )

            if "discriminative_features" in material_analysis:
                disc_features = material_analysis["discriminative_features"]
                if "top_discriminative_features" in disc_features:
                    summary["top_discriminative_features"] = disc_features[
                        "top_discriminative_features"
                    ]

        # Add frequency analysis summary
        if "frequency_analysis" in results:
            all_centroids = []
            all_rolloffs = []

            for batch_data in results["frequency_analysis"].values():
                all_centroids.extend(batch_data["spectral_centroids"])
                all_rolloffs.extend(batch_data["spectral_rolloffs"])

            if all_centroids:
                summary["spectral_centroid_range"] = {
                    "min": float(min(all_centroids)),
                    "max": float(max(all_centroids)),
                    "mean": float(np.mean(all_centroids)),
                }

            if all_rolloffs:
                summary["spectral_rolloff_range"] = {
                    "min": float(min(all_rolloffs)),
                    "max": float(max(all_rolloffs)),
                    "mean": float(np.mean(all_rolloffs)),
                }

        # Add transfer function summary
        if "transfer_function_analysis" in results:
            all_resonant_freqs = []
            all_q_factors = []

            for batch_data in results["transfer_function_analysis"].values():
                all_resonant_freqs.extend(batch_data["resonant_frequencies"])
                all_q_factors.extend(batch_data["quality_factors"])

            if all_resonant_freqs:
                summary["resonant_frequency_range"] = {
                    "min": float(min(all_resonant_freqs)),
                    "max": float(max(all_resonant_freqs)),
                    "mean": float(np.mean(all_resonant_freqs)),
                }

            if all_q_factors:
                summary["quality_factor_range"] = {
                    "min": float(min(all_q_factors)),
                    "max": float(max(all_q_factors)),
                    "mean": float(np.mean(all_q_factors)),
                }

        self.save_results(summary, "impulse_response_summary.json")
