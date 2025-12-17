from .base_experiment import BaseExperiment
from typing import Dict, Any, List
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

        # Get available batches
        data_dir_path = f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}"
        available_batches = [
            d
            for d in os.listdir(data_dir_path)
            if (d == "balanced_collected_data_2025_12_16_v3_undersample")
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
                    # Apply motion artifact removal if enabled
                    motion_removal_config = self.config.get(
                        "motion_artifact_removal", {}
                    )
                    if motion_removal_config.get("enabled", False):
                        static_dir = Path(
                            f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}/collected_data_runs_validate_2025_12_16_v3_static_reference/data"
                        )
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
                        self.logger.info("Motion artifact removal disabled")

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

                    # Feature extraction (same as working code)
                    feature_extractor = GeometricFeatureExtractor(sr=48000)

                    X_feat = []
                    failed_count = 0

                    for i, audio in enumerate(audio_data):
                        try:
                            # Apply smoothing if enabled
                            if apply_smoothing:
                                audio = self._apply_high_pass_filter(
                                    audio, sr=48000, cutoff=cutoff_freq
                                )

                            features = feature_extractor.extract_features(
                                audio, method="comprehensive"
                            )
                            X_feat.append(features)

                            if (i + 1) % 50 == 0:
                                self.logger.info(
                                    f"  Processed {i + 1}/{len(audio_data)} samples"
                                )

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to extract features from sample {i}: {e}"
                            )
                            if X_feat:
                                X_feat.append(np.zeros_like(X_feat[0]))
                            failed_count += 1

                    if failed_count > 0:
                        self.logger.warning(
                            f"Failed to extract features from {failed_count} samples"
                        )

                    # Convert to numpy array
                    X_feat = np.array(X_feat)
                    labels = np.array(labels)

                    # NEW: Map labels to grouped classes
                    labels = self._map_labels_to_groups(labels)
                    labels = np.array(labels)

                    # Update actual_classes to reflect grouped labels
                    actual_classes = sorted(list(set(labels)))

                    # Store results for this batch
                    batch_results[batch_name] = {
                        "features": X_feat,
                        "labels": labels,
                        "metadata": metadata,
                        "num_samples": len(X_feat),
                        "num_features": len(X_feat[0]) if len(X_feat) > 0 else 0,
                        "classes": actual_classes,
                        "class_distribution": dict(
                            zip(*np.unique(labels, return_counts=True))
                        ),
                    }

                    self.logger.info(
                        f"Loaded {len(X_feat)} feature vectors from {batch_name} with {len(actual_classes)} classes"
                    )

                    # Save batch-specific results
                    self._save_batch_data_processing_results(
                        batch_results[batch_name], batch_name
                    )

                    # Create batch-specific plots
                    self._create_batch_plots(batch_results[batch_name], batch_name)
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
