"""
Motion Artifact Removal Validation Experiment

This experiment provides comprehensive validation of motion artifact removal effectiveness
through quantitative metrics, A/B testing, and statistical analysis.
"""

import numpy as np
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from .base_experiment import BaseExperiment
from acoustic_sensing.core.motion_artifact_removal import apply_motion_artifact_removal


class MotionArtifactValidationExperiment(BaseExperiment):
    """
    Comprehensive validation experiment for motion artifact removal.

    This experiment:
    1. Compares model performance with/without motion artifact removal
    2. Calculates quantitative noise reduction metrics
    3. Performs statistical validation of improvements
    4. Generates validation reports and visualizations
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing experiment."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive motion artifact removal validation.

        Args:
            shared_data: Results from data processing experiment

        Returns:
            Validation results and metrics
        """
        self.logger.info("Starting motion artifact removal validation...")

        # Get processed data from data processing experiment
        if "batch_results" not in shared_data:
            raise ValueError(
                "No processed data found. Run data_processing experiment first."
            )

        # Configuration
        base_data_dir = self.config.get("base_data_dir", "data")
        static_dir = Path(
            f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}/collected_data_runs_validate_2025_12_16_v3_static_reference/data"
        )

        # Run A/B testing: with vs without motion artifact removal
        ab_results = self._run_ab_testing(shared_data, static_dir)

        # Calculate quantitative validation metrics
        quant_metrics = self._calculate_quantitative_metrics(shared_data, static_dir)

        # Generate validation report
        validation_report = self._generate_validation_report(ab_results, quant_metrics)

        # Save results
        self.save_results(validation_report, "motion_artifact_validation_report.json")

        # Create validation visualizations
        self._create_validation_plots(ab_results, quant_metrics)

        self.logger.info("Motion artifact removal validation completed")
        return validation_report

    def _run_ab_testing(
        self, shared_data: Dict[str, Any], static_dir: Path
    ) -> Dict[str, Any]:
        """Run A/B testing comparing model performance with/without motion artifact removal."""
        self.logger.info("Running A/B testing...")

        ab_results = {}

        for batch_name, batch_data in shared_data["batch_results"].items():
            self.logger.info(f"A/B testing for batch: {batch_name}")

            features = batch_data["features"]
            labels = batch_data["labels"]

            # Test B: With motion artifact removal (current processed data)
            accuracy_with_removal = self._evaluate_model_performance(features, labels)

            # Test A: Without motion artifact removal (reload raw data)
            features_without_removal = self._load_raw_features(batch_name)
            if features_without_removal is not None:
                accuracy_without_removal = self._evaluate_model_performance(
                    features_without_removal, labels
                )
            else:
                self.logger.warning(f"Could not load raw features for {batch_name}")
                accuracy_without_removal = None

            ab_results[batch_name] = {
                "accuracy_with_removal": accuracy_with_removal,
                "accuracy_without_removal": accuracy_without_removal,
                "improvement": (
                    accuracy_with_removal - accuracy_without_removal
                    if accuracy_without_removal is not None
                    else None
                ),
            }

        return ab_results

    def _load_raw_features(self, batch_name: str) -> np.ndarray:
        """Load raw features without motion artifact removal for A/B testing."""
        try:
            # Import required modules
            sys.path.append(
                "/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/src"
            )
            from acoustic_sensing.analysis.batch_analysis import BatchSpecificAnalyzer
            from acoustic_sensing.models.geometric_data_loader import (
                GeometricDataLoader,
            )
            from acoustic_sensing.core.feature_extraction import (
                GeometricFeatureExtractor,
            )

            # Initialize components
            base_data_dir = self.config.get("base_data_dir", "data")
            analyzer = BatchSpecificAnalyzer(base_dir=base_data_dir)
            data_loader = GeometricDataLoader(base_dir=base_data_dir, sr=48000)

            # Load raw data (same as data processing but without motion artifact removal)
            actual_classes = analyzer.detect_actual_classes(batch_name)
            audio_data, labels, metadata = data_loader.load_batch_data(
                batch_name,
                contact_positions=actual_classes,
                max_samples_per_class=None,
                verbose=False,
            )

            if len(audio_data) == 0:
                return None

            # Extract features from raw audio (no motion artifact removal)
            feature_extractor = GeometricFeatureExtractor(sr=48000)
            X_feat = []

            for audio in audio_data:
                try:
                    features = feature_extractor.extract_features(
                        audio, method="comprehensive"
                    )
                    X_feat.append(features)
                except Exception as e:
                    self.logger.warning(f"Failed to extract features: {e}")
                    continue

            return np.array(X_feat)

        except Exception as e:
            self.logger.error(f"Failed to load raw features for {batch_name}: {e}")
            return None

    def _evaluate_model_performance(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Evaluate model performance using cross-validation."""
        try:
            # Simple Random Forest classifier for evaluation
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, features, labels, cv=cv, scoring="accuracy")

            return np.mean(scores)

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return 0.0

    def _calculate_quantitative_metrics(
        self, shared_data: Dict[str, Any], static_dir: Path
    ) -> Dict[str, Any]:
        """Calculate quantitative metrics for motion artifact removal effectiveness."""
        self.logger.info("Calculating quantitative validation metrics...")

        metrics = {}

        for batch_name, batch_data in shared_data["batch_results"].items():
            batch_metrics = {}

            # Load sample raw audio for metrics calculation
            base_data_dir = self.config.get("base_data_dir", "data")
            batch_data_dir = f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}/{batch_name}/data"

            if not os.path.exists(batch_data_dir):
                continue

            wav_files = [f for f in os.listdir(batch_data_dir) if f.endswith(".wav")]
            if not wav_files:
                continue

            # Sample a few files for metrics
            sample_files = wav_files[: min(5, len(wav_files))]

            for wav_file in sample_files:
                try:
                    # Load raw audio
                    raw_audio, sr = librosa.load(
                        os.path.join(batch_data_dir, wav_file), sr=48000
                    )

                    # Determine class from filename
                    if "surface" in wav_file:
                        class_name = "contact"
                    elif "no_surface" in wav_file:
                        class_name = "no_contact"
                    elif "edge" in wav_file:
                        class_name = "edge"
                    else:
                        continue

                    # Load static reference
                    static_files = list(static_dir.glob(f"*_{class_name}.wav"))
                    if not static_files:
                        continue

                    static_audio, _ = librosa.load(static_files[0], sr=48000)

                    # Apply motion artifact removal
                    clean_audio = self._apply_motion_artifact_removal_to_single_audio(
                        raw_audio, static_audio
                    )

                    # === REALISTIC SNR CALCULATION ===
                    signal_power = np.mean(
                        raw_audio**2
                    )  # total power (contact + motion)
                    template_power = np.mean(static_audio**2)  # motion noise estimate

                    # Baseline SNR: total / estimated motion noise
                    snr_before = 10 * np.log10(signal_power / (template_power + 1e-10))

                    # Residual after removal (what was subtracted)
                    removed_noise = raw_audio - clean_audio
                    residual_power = np.mean(removed_noise**2)

                    # Estimated SNR after: total signal power / residual noise power
                    estimated_snr_after = 10 * np.log10(
                        signal_power / (residual_power + 1e-10)
                    )

                    # SNR improvement in dB
                    snr_improvement = estimated_snr_after - snr_before

                    # Direct noise reduction: how much motion power was reduced
                    noise_reduction_db = 10 * np.log10(
                        (template_power + 1e-10) / (residual_power + 1e-10)
                    )

                    # Spectral analysis
                    freq_before, psd_before = self._calculate_power_spectral_density(
                        raw_audio, sr
                    )
                    freq_after, psd_after = self._calculate_power_spectral_density(
                        clean_audio, sr
                    )

                    # Store metrics
                    if class_name not in batch_metrics:
                        batch_metrics[class_name] = []

                    batch_metrics[class_name].append(
                        {
                            "snr_before": snr_before,
                            "estimated_snr_after": estimated_snr_after,
                            "snr_improvement_db": snr_improvement,
                            "noise_reduction_db": noise_reduction_db,
                            "rms_amplitude_before": np.sqrt(np.mean(raw_audio**2)),
                            "rms_amplitude_after": np.sqrt(np.mean(clean_audio**2)),
                            "freq_range": [freq_before, freq_after],
                            "psd_range": [psd_before, psd_after],
                        }
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to calculate metrics for {wav_file}: {e}"
                    )
                    continue

            # Aggregate metrics per class
            aggregated_metrics = {}
            for class_name, class_samples in batch_metrics.items():
                aggregated_metrics[class_name] = {
                    "mean_snr_before": np.mean(
                        [s["snr_before"] for s in class_samples]
                    ),
                    "mean_estimated_snr_after": np.mean(
                        [s["estimated_snr_after"] for s in class_samples]
                    ),
                    "mean_snr_improvement": np.mean(
                        [s["snr_improvement_db"] for s in class_samples]
                    ),
                    "mean_noise_reduction": np.mean(
                        [s["noise_reduction_db"] for s in class_samples]
                    ),
                    "std_noise_reduction": np.std(
                        [s["noise_reduction_db"] for s in class_samples]
                    ),
                    "n_samples": len(class_samples),
                }

            metrics[batch_name] = aggregated_metrics

        return metrics

    def _calculate_power_spectral_density(
        self, audio: np.ndarray, sr: int, n_fft: int = 2048
    ):
        """Calculate power spectral density."""
        import scipy.signal

        freqs, psd = scipy.signal.welch(audio, fs=sr, nperseg=n_fft)
        return freqs, psd

    def _generate_validation_report(
        self, ab_results: Dict, quant_metrics: Dict
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "validation_summary": {
                "total_batches_tested": len(ab_results),
                "average_improvement": self._calculate_average_improvement(ab_results),
                "validation_confidence": self._assess_validation_confidence(
                    ab_results, quant_metrics
                ),
            },
            "ab_testing_results": ab_results,
            "quantitative_metrics": quant_metrics,
            "recommendations": self._generate_recommendations(
                ab_results, quant_metrics
            ),
        }

        return report

    def _calculate_average_improvement(self, ab_results: Dict) -> float:
        """Calculate average accuracy improvement across all batches."""
        improvements = []
        for batch_result in ab_results.values():
            if batch_result["improvement"] is not None:
                improvements.append(batch_result["improvement"])

        return np.mean(improvements) if improvements else 0.0

    def _assess_validation_confidence(
        self, ab_results: Dict, quant_metrics: Dict
    ) -> str:
        """Assess overall confidence in motion artifact removal effectiveness."""
        improvements = [
            r["improvement"]
            for r in ab_results.values()
            if r["improvement"] is not None
        ]

        if not improvements:
            return "insufficient_data"

        mean_improvement = np.mean(improvements)

        # Use realistic SNR improvement for confidence
        avg_snr_improvement = 0
        count = 0
        for batch_metrics in quant_metrics.values():
            for class_metrics in batch_metrics.values():
                avg_snr_improvement += class_metrics["mean_snr_improvement"]
                count += 1

        avg_snr_improvement = avg_snr_improvement / count if count > 0 else 0

        if mean_improvement > 0.05 and avg_snr_improvement > 8:
            return "high_confidence"
        elif mean_improvement > 0.02 and avg_snr_improvement > 3:
            return "moderate_confidence"
        elif mean_improvement > 0 and avg_snr_improvement > 0:
            return "low_confidence"
        else:
            return "insufficient_evidence"

    def _generate_recommendations(
        self, ab_results: Dict, quant_metrics: Dict
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        avg_improvement = self._calculate_average_improvement(ab_results)
        confidence = self._assess_validation_confidence(ab_results, quant_metrics)

        if confidence == "high_confidence":
            recommendations.append(
                "Motion artifact removal is highly effective. Enable by default for all experiments."
            )
        elif confidence == "moderate_confidence":
            recommendations.append(
                "Motion artifact removal shows moderate benefits. Consider enabling for critical applications."
            )
        elif confidence == "low_confidence":
            recommendations.append(
                "Motion artifact removal shows minimal benefits. Consider A/B testing for specific use cases."
            )
        else:
            recommendations.append(
                "Insufficient evidence of motion artifact removal effectiveness. Further validation needed."
            )

        # Class-specific insights
        for batch_name, batch_metrics in quant_metrics.items():
            for class_name, class_data in batch_metrics.items():
                noise_reduction = class_data["mean_noise_reduction"]
                if noise_reduction < 3:
                    recommendations.append(
                        f"Low noise reduction for {class_name} in {batch_name}. Consider improving static references or adding more templates."
                    )

        return recommendations

    def _create_validation_plots(self, ab_results: Dict, quant_metrics: Dict):
        """Create validation visualization plots."""
        try:
            os.makedirs(self.experiment_output_dir, exist_ok=True)

            self._create_ab_testing_plot(ab_results)
            self._create_quantitative_metrics_plot(quant_metrics)
            self._create_noise_reduction_plot(quant_metrics)

        except Exception as e:
            self.logger.warning(f"Failed to create validation plots: {e}")

    def _create_ab_testing_plot(self, ab_results: Dict):
        """Create A/B testing comparison plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        batches = list(ab_results.keys())
        with_removal = [r["accuracy_with_removal"] for r in ab_results.values()]
        without_removal = [r["accuracy_without_removal"] for r in ab_results.values()]

        x = np.arange(len(batches))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            without_removal,
            width,
            label="Without Motion Artifact Removal",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            with_removal,
            width,
            label="With Motion Artifact Removal",
            alpha=0.8,
        )

        ax.set_xlabel("Batch")
        ax.set_ylabel("Cross-Validation Accuracy")
        ax.set_title("A/B Testing: Motion Artifact Removal Impact on Model Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(batches, rotation=45)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "ab_testing_results.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_quantitative_metrics_plot(self, quant_metrics: Dict):
        """Create quantitative metrics summary plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        snr_before = []
        snr_after = []
        snr_improvement = []
        classes = []

        for batch_name, batch_data in quant_metrics.items():
            for class_name, class_metrics in batch_data.items():
                snr_before.append(class_metrics["mean_snr_before"])
                snr_after.append(class_metrics["mean_estimated_snr_after"])
                snr_improvement.append(class_metrics["mean_snr_improvement"])
                classes.append(f"{batch_name}\n{class_name}")

        x = np.arange(len(classes))
        axes[0, 0].bar(x, snr_before, alpha=0.6, label="Before", color="red")
        axes[0, 0].bar(x, snr_after, alpha=0.6, label="After", color="green")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(classes, rotation=45)
        axes[0, 0].set_ylabel("SNR (dB)")
        axes[0, 0].set_title("Estimated SNR Before/After Removal")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].hist(snr_improvement, bins=10, alpha=0.7, edgecolor="black")
        axes[0, 1].set_xlabel("SNR Improvement (dB)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of SNR Improvement")
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].scatter(snr_before, snr_improvement, alpha=0.6)
        axes[1, 0].set_xlabel("SNR Before (dB)")
        axes[1, 0].set_ylabel("SNR Improvement (dB)")
        axes[1, 0].set_title("Baseline SNR vs Improvement")
        axes[1, 0].grid(alpha=0.3)

        stats_text = f"""
Validation Summary:
• Average SNR Before: {np.mean(snr_before):.1f} dB
• Average SNR After: {np.mean(snr_after):.1f} dB
• Average SNR Improvement: {np.mean(snr_improvement):.1f} dB
• Classes Tested: {len(classes)}
"""
        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Validation Summary")

        plt.suptitle("Motion Artifact Removal Quantitative Validation", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "quantitative_metrics.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_noise_reduction_plot(self, quant_metrics: Dict):
        """Create noise reduction analysis plot."""
        fig, ax = plt.subplots(figsize=(12, 8))

        data = []
        labels = []

        for batch_name, batch_data in quant_metrics.items():
            for class_name, class_metrics in batch_data.items():
                data.append([class_metrics["mean_snr_improvement"]])
                labels.append(f"{batch_name}\n{class_name}")

        if data:
            ax.boxplot(data, labels=labels)
            ax.set_ylabel("SNR Improvement (dB)")
            ax.set_title("SNR Improvement by Batch and Class")
            ax.grid(alpha=0.3)
            ax.axhline(
                y=0, color="red", linestyle="--", alpha=0.7, label="No Improvement"
            )
            ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "noise_reduction_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _apply_motion_artifact_removal_to_single_audio(
        self, audio: np.ndarray, template: np.ndarray
    ) -> np.ndarray:
        """Apply motion artifact removal to a single audio signal (helper for validation)."""
        from acoustic_sensing.core.motion_artifact_removal import (
            _static_subtraction,
            _lms_adaptive_filter,
        )

        clean1 = _static_subtraction(audio, template)
        clean_final = _lms_adaptive_filter(clean1, template)

        if len(clean_final) < len(audio):
            clean_final = np.pad(
                clean_final, (0, len(audio) - len(clean_final)), "constant"
            )

        return clean_final.astype(np.float32)
