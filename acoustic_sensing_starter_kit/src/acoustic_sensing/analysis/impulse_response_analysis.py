"""
Impulse Response Analysis for Acoustic Sensing

This module implements deconvolution-based analysis to extract the system's
impulse response from sweep excitation signals. This provides cleaner features
for material property discrimination by removing the excitation signal characteristics.

Key advantages:
- Removes sweep signal artifacts from analysis
- Provides true system transfer function
- Better feature extraction for material properties
- Enables frequency response analysis
"""

import numpy as np
import librosa
import scipy.signal
from scipy.fft import fft, rfft, irfft
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


class ImpulseResponseAnalyzer:
    """
    Analyzes impulse responses by deconvolving excitation sweeps from measured responses.

    This provides the true system characteristics for better material discrimination.
    """

    def __init__(self, sr: int = 48000, n_fft: int = 8192):
        """
        Initialize the impulse response analyzer.

        Args:
            sr: Sample rate
            n_fft: FFT size for analysis
        """
        self.sr = sr
        self.n_fft = n_fft

    def load_sweep_signal(self, sweep_path: Path) -> np.ndarray:
        """
        Load the excitation sweep signal.

        Args:
            sweep_path: Path to sweep WAV file

        Returns:
            Sweep signal as numpy array
        """
        sweep, _ = librosa.load(sweep_path, sr=self.sr, mono=True)
        return sweep.astype(np.float32)

    def load_response_signal(self, response_path: Path) -> np.ndarray:
        """
        Load the measured response signal.

        Args:
            response_path: Path to response WAV file

        Returns:
            Response signal as numpy array
        """
        response, _ = librosa.load(response_path, sr=self.sr, mono=True)
        return response.astype(np.float32)

    def generate_inverse_filter(
        self, sweep: np.ndarray, fade_in: float = 0.1, fade_out: float = 0.01
    ) -> np.ndarray:
        """
        Generate the inverse filter for deconvolution.

        For a logarithmic sweep s(t), the inverse filter is s(-t) reversed in time.

        Args:
            sweep: Original sweep signal
            fade_in: Fade in time (seconds)
            fade_out: Fade out time (seconds)

        Returns:
            Inverse filter for deconvolution
        """
        # Create time-reversed sweep
        inv_filter = np.flip(sweep.copy())

        # Apply windowing to avoid artifacts
        n_samples = len(inv_filter)
        fade_in_samples = int(fade_in * self.sr)
        fade_out_samples = int(fade_out * self.sr)

        # Apply fade in at the beginning (which is the end of the original sweep)
        if fade_in_samples > 0:
            fade_in_window = np.linspace(0, 1, fade_in_samples)
            inv_filter[:fade_in_samples] *= fade_in_window

        # Apply fade out at the end
        if fade_out_samples > 0:
            fade_out_window = np.linspace(1, 0, fade_out_samples)
            inv_filter[-fade_out_samples:] *= fade_out_window

        return inv_filter.astype(np.float32)

    def deconvolve_response(
        self, response: np.ndarray, inv_filter: np.ndarray, impulse_length: int = 16384
    ) -> np.ndarray:
        """
        Deconvolve the response to get the impulse response.

        Args:
            response: Measured response signal
            inv_filter: Inverse filter
            impulse_length: Length of impulse response to extract

        Returns:
            Impulse response of the system
        """
        # Convolve response with inverse filter
        conv_result = scipy.signal.convolve(response, inv_filter, mode="full")

        # Extract the central portion (where the impulse response appears)
        # For sweep deconvolution, the impulse appears at a specific lag
        sweep_length = len(inv_filter)
        response_length = len(response)

        # The impulse response appears at lag = sweep_length
        start_idx = sweep_length
        end_idx = start_idx + impulse_length

        if end_idx > len(conv_result):
            end_idx = len(conv_result)

        impulse_response = conv_result[start_idx:end_idx]

        # Normalize
        if np.max(np.abs(impulse_response)) > 0:
            impulse_response = impulse_response / np.max(np.abs(impulse_response))

        return impulse_response.astype(np.float32)

    def extract_impulse_features(
        self, impulse_response: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract features from the impulse response.

        Args:
            impulse_response: Impulse response signal

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Basic temporal features
        features["impulse_peak"] = np.max(np.abs(impulse_response))
        features["impulse_peak_time"] = np.argmax(np.abs(impulse_response)) / self.sr
        features["impulse_rms"] = np.sqrt(np.mean(impulse_response**2))

        # Decay characteristics
        peak_idx = np.argmax(np.abs(impulse_response))
        decay_portion = impulse_response[peak_idx:]

        if len(decay_portion) > 10:
            # Exponential decay fit
            t = np.arange(len(decay_portion)) / self.sr
            y = np.abs(decay_portion)

            try:
                # Fit exponential decay: y = a * exp(-b * t)
                from scipy.optimize import curve_fit

                def exp_decay(t, a, b):
                    return a * np.exp(-b * t)

                popt, _ = curve_fit(
                    exp_decay,
                    t[: min(100, len(t))],
                    y[: min(100, len(y))],
                    p0=[y[0], 1000],
                    bounds=([0, 0], [np.inf, 10000]),
                )
                features["decay_amplitude"] = popt[0]
                features["decay_rate"] = popt[1]
            except:
                features["decay_amplitude"] = 0.0
                features["decay_rate"] = 0.0

        # Frequency domain features
        if len(impulse_response) >= self.n_fft:
            # Compute frequency response
            freq_response = rfft(impulse_response, n=self.n_fft)
            freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sr)

            magnitude = np.abs(freq_response)
            phase = np.angle(freq_response)

            # Resonance analysis
            # Find peaks in magnitude response
            peaks, properties = scipy.signal.find_peaks(
                magnitude, height=0.1, distance=10
            )

            if len(peaks) > 0:
                # Primary resonance
                primary_peak_idx = peaks[np.argmax(properties["peak_heights"])]
                features["primary_resonance_freq"] = freqs[primary_peak_idx]
                features["primary_resonance_magnitude"] = magnitude[primary_peak_idx]

                # Resonance Q factor (quality factor)
                peak_freq = freqs[primary_peak_idx]
                peak_mag = magnitude[primary_peak_idx]

                # Find -3dB points
                half_power = peak_mag / np.sqrt(2)
                left_idx = np.where((freqs < peak_freq) & (magnitude >= half_power))[0]
                right_idx = np.where((freqs > peak_freq) & (magnitude >= half_power))[0]

                if len(left_idx) > 0 and len(right_idx) > 0:
                    f_left = freqs[left_idx[-1]]
                    f_right = freqs[right_idx[0]]
                    features["resonance_q_factor"] = peak_freq / (f_right - f_left)
                else:
                    features["resonance_q_factor"] = 0.0

                # Multiple resonances
                features["num_resonances"] = len(peaks)
                if len(peaks) > 1:
                    sorted_peaks = peaks[np.argsort(properties["peak_heights"])[::-1]]
                    features["secondary_resonance_freq"] = freqs[sorted_peaks[1]]
                    features["secondary_resonance_magnitude"] = magnitude[
                        sorted_peaks[1]
                    ]
                else:
                    features["secondary_resonance_freq"] = 0.0
                    features["secondary_resonance_magnitude"] = 0.0
            else:
                features["primary_resonance_freq"] = 0.0
                features["primary_resonance_magnitude"] = 0.0
                features["resonance_q_factor"] = 0.0
                features["num_resonances"] = 0

            # Frequency response shape
            features["freq_response_centroid"] = np.sum(freqs * magnitude) / np.sum(
                magnitude
            )
            features["freq_response_spread"] = np.sqrt(
                np.sum((freqs - features["freq_response_centroid"]) ** 2 * magnitude)
                / np.sum(magnitude)
            )

            # High frequency damping
            high_freq_mask = freqs > 1000  # Above 1kHz
            if np.any(high_freq_mask):
                features["high_freq_energy"] = np.mean(magnitude[high_freq_mask] ** 2)
                features["high_freq_damping"] = np.mean(
                    magnitude[high_freq_mask]
                ) / np.mean(magnitude)
            else:
                features["high_freq_energy"] = 0.0
                features["high_freq_damping"] = 0.0

        return features

    def analyze_measurement(
        self, sweep_path: Path, response_path: Path
    ) -> Dict[str, float]:
        """
        Complete analysis pipeline: load signals, deconvolve, extract features.

        Args:
            sweep_path: Path to sweep signal
            response_path: Path to response signal

        Returns:
            Dictionary of impulse response features
        """
        try:
            # Load signals
            sweep = self.load_sweep_signal(sweep_path)
            response = self.load_response_signal(response_path)

            # Generate inverse filter
            inv_filter = self.generate_inverse_filter(sweep)

            # Deconvolve to get impulse response
            impulse_response = self.deconvolve_response(response, inv_filter)

            # Extract features
            features = self.extract_impulse_features(impulse_response)

            # Add metadata
            features["_impulse_length"] = len(impulse_response)
            features["_sweep_length"] = len(sweep)
            features["_response_length"] = len(response)

            return features

        except Exception as e:
            warnings.warn(f"Failed to analyze measurement: {e}")
            return {}

    def visualize_analysis(
        self,
        sweep: np.ndarray,
        response: np.ndarray,
        impulse_response: np.ndarray,
        save_path: Optional[Path] = None,
    ):
        """
        Visualize the deconvolution analysis.

        Args:
            sweep: Original sweep signal
            response: Measured response
            impulse_response: Computed impulse response
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle("Impulse Response Analysis", fontsize=14)

        # Time domain plots
        t_sweep = np.arange(len(sweep)) / self.sr
        t_response = np.arange(len(response)) / self.sr
        t_impulse = np.arange(len(impulse_response)) / self.sr

        # Sweep signal
        axes[0, 0].plot(t_sweep, sweep, "b-", alpha=0.7)
        axes[0, 0].set_title("Excitation Sweep Signal")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].grid(True, alpha=0.3)

        # Response signal
        axes[1, 0].plot(t_response, response, "r-", alpha=0.7)
        axes[1, 0].set_title("Measured Response Signal")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].grid(True, alpha=0.3)

        # Impulse response
        axes[2, 0].plot(t_impulse, impulse_response, "g-", alpha=0.7)
        axes[2, 0].set_title("System Impulse Response")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Amplitude")
        axes[2, 0].set_xlim(0, min(0.01, t_impulse[-1]))  # Show first 10ms
        axes[2, 0].grid(True, alpha=0.3)

        # Frequency domain plots
        sweep_fft = np.abs(rfft(sweep, n=self.n_fft))
        response_fft = np.abs(rfft(response, n=self.n_fft))
        impulse_fft = np.abs(rfft(impulse_response, n=self.n_fft))

        freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sr)

        # Sweep spectrum
        axes[0, 1].semilogx(freqs, 20 * np.log10(sweep_fft + 1e-10), "b-", alpha=0.7)
        axes[0, 1].set_title("Sweep Spectrum")
        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("Magnitude (dB)")
        axes[0, 1].grid(True, alpha=0.3)

        # Response spectrum
        axes[1, 1].semilogx(freqs, 20 * np.log10(response_fft + 1e-10), "r-", alpha=0.7)
        axes[1, 1].set_title("Response Spectrum")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("Magnitude (dB)")
        axes[1, 1].grid(True, alpha=0.3)

        # Impulse response spectrum (system transfer function)
        axes[2, 1].semilogx(freqs, 20 * np.log10(impulse_fft + 1e-10), "g-", alpha=0.7)
        axes[2, 1].set_title("System Transfer Function")
        axes[2, 1].set_xlabel("Frequency (Hz)")
        axes[2, 1].set_ylabel("Magnitude (dB)")
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_class_transfer_functions(
        self,
        sweep: np.ndarray,
        class_responses: Dict[str, List[np.ndarray]],
        save_path: Optional[Path] = None,
    ):
        """
        Visualize average transfer functions for each class to show acoustic differences.

        Args:
            sweep: Original sweep signal
            class_responses: Dictionary mapping class names to lists of response signals
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Per-Class Transfer Function Analysis", fontsize=14)

        # Generate inverse filter once
        inv_filter = self.generate_inverse_filter(sweep)

        # Colors for different classes
        class_names = list(class_responses.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

        # Compute transfer functions for each class
        class_transfer_funcs = {}
        class_impulse_responses = {}

        for class_name, responses in class_responses.items():
            impulse_responses = []
            transfer_funcs = []

            for response in responses:
                try:
                    # Deconvolve to get impulse response
                    impulse_response = self.deconvolve_response(response, inv_filter)
                    impulse_responses.append(impulse_response)

                    # Compute transfer function (frequency response)
                    transfer_func = np.abs(rfft(impulse_response, n=self.n_fft))
                    transfer_funcs.append(transfer_func)
                except Exception as e:
                    print(
                        f"Warning: Failed to process response for class {class_name}: {e}"
                    )
                    continue

            if transfer_funcs:
                # Average transfer functions across samples in this class
                avg_transfer_func = np.mean(transfer_funcs, axis=0)
                class_transfer_funcs[class_name] = avg_transfer_func

                # Average impulse responses
                avg_impulse_response = np.mean(impulse_responses, axis=0)
                class_impulse_responses[class_name] = avg_impulse_response

        freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sr)

        # Plot 1: Average transfer functions (magnitude)
        ax1 = axes[0, 0]
        for i, (class_name, transfer_func) in enumerate(class_transfer_funcs.items()):
            ax1.semilogx(
                freqs,
                20 * np.log10(transfer_func + 1e-10),
                color=colors[i],
                label=class_name,
                linewidth=2,
                alpha=0.8,
            )

        ax1.set_title("Average Transfer Functions by Class")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Transfer function differences (relative to first class)
        ax2 = axes[0, 1]
        if len(class_transfer_funcs) > 1:
            baseline_class = list(class_transfer_funcs.keys())[0]
            baseline_tf = class_transfer_funcs[baseline_class]

            for i, (class_name, transfer_func) in enumerate(
                class_transfer_funcs.items()
            ):
                if class_name != baseline_class:
                    # Compute difference in dB
                    diff_db = 20 * np.log10(transfer_func + 1e-10) - 20 * np.log10(
                        baseline_tf + 1e-10
                    )
                    ax2.semilogx(
                        freqs,
                        diff_db,
                        color=colors[i],
                        label=f"{class_name} vs {baseline_class}",
                        linewidth=2,
                        alpha=0.8,
                    )

        ax2.set_title("Transfer Function Differences")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude Difference (dB)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average impulse responses (time domain)
        ax3 = axes[1, 0]
        for i, (class_name, impulse_response) in enumerate(
            class_impulse_responses.items()
        ):
            t_impulse = np.arange(len(impulse_response)) / self.sr
            # Show first 10ms
            mask = t_impulse <= 0.01
            ax3.plot(
                t_impulse[mask] * 1000,
                impulse_response[mask],
                color=colors[i],
                label=class_name,
                linewidth=2,
                alpha=0.8,
            )

        ax3.set_title("Average Impulse Responses by Class")
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Amplitude")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Resonance analysis summary
        ax4 = axes[1, 1]
        resonance_data = []

        for class_name, transfer_func in class_transfer_funcs.items():
            # Find resonances (peaks in transfer function)
            peaks, properties = scipy.signal.find_peaks(
                transfer_func, height=0.1, distance=10
            )

            if len(peaks) > 0:
                # Get top 3 resonances
                sorted_peaks = peaks[np.argsort(properties["peak_heights"])[::-1]]
                top_freqs = freqs[sorted_peaks[:3]]
                top_mags = transfer_func[sorted_peaks[:3]]

                resonance_data.append((class_name, top_freqs, top_mags))

        # Plot resonance frequencies as scatter
        for i, (class_name, freqs_res, mags_res) in enumerate(resonance_data):
            ax4.scatter(
                freqs_res,
                20 * np.log10(mags_res + 1e-10),
                color=colors[i],
                label=class_name,
                s=100,
                alpha=0.8,
                marker=["o", "s", "^"][i % 3],
            )

        ax4.set_title("Resonance Patterns by Class")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Magnitude (dB)")
        ax4.set_xscale("log")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def demo_impulse_response_analysis():
    """
    Demonstration of impulse response analysis on sample data.
    """
    analyzer = ImpulseResponseAnalyzer()

    # Example usage with batch 1 data
    data_dir = Path(
        "/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/data/soft_finger_batch_1/data"
    )

    sweep_path = data_dir / "0_sweep.wav"
    response_path = data_dir / "1_finger tip.wav"  # Example response

    if sweep_path.exists() and response_path.exists():
        print("Analyzing impulse response...")

        # Load signals
        sweep = analyzer.load_sweep_signal(sweep_path)
        response = analyzer.load_response_signal(response_path)

        # Generate inverse filter
        inv_filter = analyzer.generate_inverse_filter(sweep)

        # Deconvolve
        impulse_response = analyzer.deconvolve_response(response, inv_filter)

        # Extract features
        features = analyzer.extract_impulse_features(impulse_response)

        print(f"Extracted {len(features)} impulse response features")
        print("Key features:")
        for key, value in list(features.items())[:10]:
            print(f"  {key}: {value:.4f}")

        # Visualize
        analyzer.visualize_analysis(
            sweep,
            response,
            impulse_response,
            save_path=data_dir / "impulse_response_analysis.png",
        )

        print(f"Visualization saved to: {data_dir / 'impulse_response_analysis.png'}")

        return features
    else:
        print("Sample data files not found")
        return {}


if __name__ == "__main__":
    demo_impulse_response_analysis()
