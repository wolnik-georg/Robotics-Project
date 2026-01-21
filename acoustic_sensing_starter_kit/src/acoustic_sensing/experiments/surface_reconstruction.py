"""
Surface Reconstruction Experiment

Reconstructs 2D surface maps from acoustic sweep data using trained models.
This experiment:
1. Loads a trained model from multi-dataset training
2. Processes sweep data with spatial coordinates
3. Generates predictions for each spatial position
4. Creates 2D visualizations comparing predictions vs ground truth
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from acoustic_sensing.experiments.base_experiment import BaseExperiment


class SurfaceReconstructionExperiment(BaseExperiment):
    """Reconstruct surface maps from acoustic sweep data using trained models."""

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """Initialize surface reconstruction experiment.

        Args:
            config: Experiment configuration
            output_dir: Directory for saving results
        """
        super().__init__(config, output_dir)
        self.logger = logging.getLogger(__name__)

    def get_dependencies(self) -> List[str]:
        """Return list of required experiments.

        Surface reconstruction depends on multi_dataset_training.

        Returns:
            List of experiment names this experiment depends on
        """
        return ["multi_dataset_training"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute surface reconstruction experiment.

        Args:
            shared_data: Shared data from previous experiments

        Returns:
            Dictionary containing reconstruction results and metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("🗺️  Starting Surface Reconstruction Experiment")
        self.logger.info("=" * 80)

        # Get trained models from multi-dataset training
        if "trained_models" not in shared_data:
            raise ValueError(
                "Trained models not found in shared_data. "
                "Please run multi_dataset_training experiment first."
            )

        trained_models = shared_data["trained_models"]
        scaler = shared_data.get("scaler")
        feature_names = shared_data.get("feature_names")
        classes = shared_data.get("classes")

        if scaler is None or feature_names is None or classes is None:
            raise ValueError(
                "Missing required data (scaler, feature_names, or classes) in shared_data"
            )

        # Get sweep dataset path from config
        sweep_dataset = self.config.get("sweep_dataset")
        if not sweep_dataset:
            raise ValueError(
                "sweep_dataset not specified in config. "
                "Please specify the dataset folder containing sweep.csv"
            )

        base_data_dir = Path(self.config.get("base_data_dir", "data"))
        sweep_path = base_data_dir / sweep_dataset

        if not sweep_path.exists():
            raise ValueError(f"Sweep dataset not found: {sweep_path}")

        self.logger.info(f"📂 Loading sweep data from: {sweep_dataset}")

        # Load sweep CSV
        sweep_csv_path = sweep_path / "sweep.csv"
        if not sweep_csv_path.exists():
            raise ValueError(f"sweep.csv not found in {sweep_path}")

        sweep_df = pd.read_csv(sweep_csv_path)
        self.logger.info(f"✓ Loaded {len(sweep_df)} sweep points")
        self.logger.info(f"  Columns: {', '.join(sweep_df.columns.tolist())}")

        # Extract features from sweep audio files
        self.logger.info("🎵 Extracting features from sweep audio files...")
        sweep_features, sweep_labels, sweep_coords = self._process_sweep_data(
            sweep_df, sweep_path, feature_names
        )

        self.logger.info(f"✓ Processed {len(sweep_features)} samples with features")

        # Scale features using the same scaler from training
        self.logger.info("📊 Scaling features...")
        sweep_features_scaled = scaler.transform(sweep_features)

        # Make predictions with all trained models
        self.logger.info("🤖 Generating predictions from trained models...")
        predictions = {}
        prediction_probabilities = {}

        for model_name, model_info in trained_models.items():
            model = model_info["model"]
            y_pred = model.predict(sweep_features_scaled)
            y_proba = model.predict_proba(sweep_features_scaled)

            predictions[model_name] = y_pred
            prediction_probabilities[model_name] = y_proba

            accuracy = accuracy_score(sweep_labels, y_pred)
            self.logger.info(f"  {model_name}: {accuracy:.2%} accuracy")

        # Select best model for reconstruction
        best_model_name = max(
            trained_models.keys(),
            key=lambda name: trained_models[name]["validation_accuracy"],
        )
        best_predictions = predictions[best_model_name]
        best_probabilities = prediction_probabilities[best_model_name]

        self.logger.info(f"\n🏆 Using best model for reconstruction: {best_model_name}")
        self.logger.info(
            f"  Validation accuracy: {trained_models[best_model_name]['validation_accuracy']:.2%}"
        )

        # Create reconstruction visualizations
        self.logger.info("\n📊 Creating surface reconstruction visualizations...")

        reconstruction_results = {
            "sweep_dataset": sweep_dataset,
            "num_points": len(sweep_features),
            "best_model": best_model_name,
            "predictions": predictions,
            "sweep_coordinates": sweep_coords,
            "sweep_labels": sweep_labels,
        }

        # 1. Ground truth surface map
        self._create_surface_map(
            sweep_coords,
            sweep_labels,
            classes,
            "Ground Truth Surface Map",
            "ground_truth_surface.png",
        )

        # 2. Predicted surface map (best model)
        self._create_surface_map(
            sweep_coords,
            best_predictions,
            classes,
            f"Predicted Surface Map ({best_model_name})",
            "predicted_surface.png",
        )

        # 3. Side-by-side comparison
        self._create_comparison_maps(
            sweep_coords, sweep_labels, best_predictions, classes, best_model_name
        )

        # 4. Prediction confidence map
        self._create_confidence_map(
            sweep_coords, best_probabilities, classes, best_model_name
        )

        # 5. Error map (misclassifications)
        self._create_error_map(
            sweep_coords, sweep_labels, best_predictions, classes, best_model_name
        )

        # 6. Per-model comparison
        self._create_model_comparison_maps(
            sweep_coords, sweep_labels, predictions, classes
        )

        # Calculate detailed metrics
        metrics = self._calculate_reconstruction_metrics(
            sweep_labels, predictions, classes
        )
        reconstruction_results["metrics"] = metrics

        # Save results
        self.logger.info("\n💾 Saving reconstruction results...")
        self.save_results(reconstruction_results, "surface_reconstruction_results.json")

        self.logger.info("\n✅ Surface Reconstruction Experiment Complete!")
        self.logger.info("=" * 80)

        return reconstruction_results

    def _process_sweep_data(
        self, sweep_df: pd.DataFrame, sweep_path: Path, feature_names: List[str]
    ) -> tuple:
        """Process sweep data: extract features and coordinates.

        Args:
            sweep_df: Sweep dataframe with coordinates and filenames
            sweep_path: Path to sweep dataset folder
            feature_names: List of feature names to extract

        Returns:
            Tuple of (features, labels, coordinates)
        """
        import librosa
        from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor

        features_list = []
        labels_list = []
        coords_list = []

        # Create feature extractor
        feature_extractor = GeometricFeatureExtractor(sr=48000)

        for idx, row in sweep_df.iterrows():
            # Get audio filename
            audio_filename = row["acoustic_filename"]
            # Handle relative paths like "./data/1_edge.wav"
            if audio_filename.startswith("./"):
                audio_filename = audio_filename[2:]  # Remove "./"
            audio_path = sweep_path / audio_filename

            if not audio_path.exists():
                self.logger.warning(f"Audio file not found: {audio_path}, skipping")
                continue

            # Load and extract features
            try:
                # Load audio using librosa
                audio, sr = librosa.load(str(audio_path), sr=48000, mono=True)

                # Extract features using the feature extractor
                features_dict = feature_extractor.extract_features(
                    audio, method="comprehensive"
                )

                # Convert dict/Series to feature vector in correct order
                if isinstance(features_dict, pd.Series):
                    feature_vector = features_dict.values
                elif isinstance(features_dict, dict):
                    feature_vector = np.array(
                        [features_dict.get(name, 0.0) for name in feature_names]
                    )
                else:
                    feature_vector = features_dict

                features_list.append(feature_vector)
                labels_list.append(row["relabeled_label"])
                coords_list.append((row["normalized_x"], row["normalized_y"]))

            except Exception as e:
                self.logger.warning(
                    f"Failed to extract features from {audio_path}: {e}"
                )
                continue

        features = np.array(features_list)
        labels = np.array(labels_list)
        coords = np.array(coords_list)

        return features, labels, coords

    def _create_surface_map(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        classes: List[str],
        title: str,
        filename: str,
    ):
        """Create a 2D surface map visualization.

        Args:
            coords: Nx2 array of (x, y) coordinates
            labels: N array of class labels
            classes: List of class names
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create color map for classes
        class_colors = {
            "contact": "#1f77b4",  # Blue
            "edge": "#ff7f0e",  # Orange
            "no_contact": "#2ca02c",  # Green
        }

        # Plot each class with different color
        for cls in classes:
            mask = labels == cls
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[class_colors.get(cls, "gray")],
                    label=cls,
                    s=50,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                )

        ax.set_xlabel("Normalized X Position", fontsize=12)
        ax.set_ylabel("Normalized Y Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        self.save_plot(fig, filename)
        plt.close()

    def _create_comparison_maps(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
        model_name: str,
    ):
        """Create side-by-side comparison of ground truth vs predictions.

        Args:
            coords: Nx2 array of (x, y) coordinates
            true_labels: N array of true class labels
            pred_labels: N array of predicted class labels
            classes: List of class names
            model_name: Name of the model used for predictions
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        class_colors = {
            "contact": "#1f77b4",
            "edge": "#ff7f0e",
            "no_contact": "#2ca02c",
        }

        # Ground truth
        for cls in classes:
            mask = true_labels == cls
            if np.any(mask):
                axes[0].scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[class_colors.get(cls, "gray")],
                    label=cls,
                    s=50,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                )

        axes[0].set_xlabel("Normalized X Position", fontsize=12)
        axes[0].set_ylabel("Normalized Y Position", fontsize=12)
        axes[0].set_title("Ground Truth", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect("equal")

        # Predictions
        for cls in classes:
            mask = pred_labels == cls
            if np.any(mask):
                axes[1].scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[class_colors.get(cls, "gray")],
                    label=cls,
                    s=50,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=0.5,
                )

        axes[1].set_xlabel("Normalized X Position", fontsize=12)
        axes[1].set_ylabel("Normalized Y Position", fontsize=12)
        axes[1].set_title(f"Predictions ({model_name})", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect("equal")

        accuracy = accuracy_score(true_labels, pred_labels)
        plt.suptitle(
            f"Surface Reconstruction Comparison (Accuracy: {accuracy:.2%})",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        self.save_plot(fig, "comparison_surface_maps.png")
        plt.close()

    def _create_confidence_map(
        self,
        coords: np.ndarray,
        probabilities: np.ndarray,
        classes: List[str],
        model_name: str,
    ):
        """Create a heatmap showing prediction confidence.

        Args:
            coords: Nx2 array of (x, y) coordinates
            probabilities: Nx3 array of class probabilities
            classes: List of class names
            model_name: Name of the model used for predictions
        """
        # Get maximum probability (confidence) for each prediction
        max_confidence = np.max(probabilities, axis=1)

        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=max_confidence,
            cmap="RdYlGn",
            s=50,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            vmin=0,
            vmax=1,
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Prediction Confidence", fontsize=12)

        ax.set_xlabel("Normalized X Position", fontsize=12)
        ax.set_ylabel("Normalized Y Position", fontsize=12)
        ax.set_title(
            f"Prediction Confidence Map ({model_name})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        self.save_plot(fig, "confidence_map.png")
        plt.close()

    def _create_error_map(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        classes: List[str],
        model_name: str,
    ):
        """Create a map highlighting prediction errors.

        Args:
            coords: Nx2 array of (x, y) coordinates
            true_labels: N array of true class labels
            pred_labels: N array of predicted class labels
            classes: List of class names
            model_name: Name of the model used for predictions
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Identify correct and incorrect predictions
        correct_mask = true_labels == pred_labels
        incorrect_mask = ~correct_mask

        # Plot correct predictions in green
        ax.scatter(
            coords[correct_mask, 0],
            coords[correct_mask, 1],
            c="green",
            label="Correct",
            s=50,
            alpha=0.5,
            edgecolors="black",
            linewidths=0.5,
        )

        # Plot incorrect predictions in red
        ax.scatter(
            coords[incorrect_mask, 0],
            coords[incorrect_mask, 1],
            c="red",
            label="Incorrect",
            s=80,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            marker="x",
        )

        ax.set_xlabel("Normalized X Position", fontsize=12)
        ax.set_ylabel("Normalized Y Position", fontsize=12)
        ax.set_title(
            f"Prediction Error Map ({model_name})", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        accuracy = accuracy_score(true_labels, pred_labels)
        error_rate = 1 - accuracy
        ax.text(
            0.02,
            0.98,
            f"Accuracy: {accuracy:.2%}\nError Rate: {error_rate:.2%}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )

        plt.tight_layout()
        self.save_plot(fig, "error_map.png")
        plt.close()

    def _create_model_comparison_maps(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        predictions: Dict[str, np.ndarray],
        classes: List[str],
    ):
        """Create comparison maps for all models.

        Args:
            coords: Nx2 array of (x, y) coordinates
            true_labels: N array of true class labels
            predictions: Dictionary of model_name -> predicted labels
            classes: List of class names
        """
        num_models = len(predictions)
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()

        class_colors = {
            "contact": "#1f77b4",
            "edge": "#ff7f0e",
            "no_contact": "#2ca02c",
        }

        for idx, (model_name, pred_labels) in enumerate(predictions.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            for cls in classes:
                mask = pred_labels == cls
                if np.any(mask):
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        c=[class_colors.get(cls, "gray")],
                        label=cls,
                        s=40,
                        alpha=0.7,
                        edgecolors="black",
                        linewidths=0.5,
                    )

            accuracy = accuracy_score(true_labels, pred_labels)

            ax.set_xlabel("Normalized X", fontsize=10)
            ax.set_ylabel("Normalized Y", fontsize=10)
            ax.set_title(f"{model_name}\n(Acc: {accuracy:.2%})", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        # Hide unused subplots
        for idx in range(num_models, len(axes)):
            axes[idx].axis("off")

        plt.suptitle(
            "Model Comparison: Surface Reconstructions", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        self.save_plot(fig, "model_comparison_maps.png")
        plt.close()

    def _calculate_reconstruction_metrics(
        self,
        true_labels: np.ndarray,
        predictions: Dict[str, np.ndarray],
        classes: List[str],
    ) -> Dict[str, Any]:
        """Calculate detailed metrics for each model's reconstruction.

        Args:
            true_labels: N array of true class labels
            predictions: Dictionary of model_name -> predicted labels
            classes: List of class names

        Returns:
            Dictionary of metrics for each model
        """
        metrics = {}

        for model_name, pred_labels in predictions.items():
            model_metrics = {
                "accuracy": float(accuracy_score(true_labels, pred_labels)),
                "classification_report": classification_report(
                    true_labels, pred_labels, target_names=classes, output_dict=True
                ),
                "confusion_matrix": confusion_matrix(
                    true_labels, pred_labels, labels=classes
                ).tolist(),
            }

            metrics[model_name] = model_metrics

            self.logger.info(f"\n📊 Metrics for {model_name}:")
            self.logger.info(f"  Accuracy: {model_metrics['accuracy']:.4f}")
            self.logger.info("\n  Classification Report:")
            for cls in classes:
                cls_metrics = model_metrics["classification_report"][cls]
                self.logger.info(
                    f"    {cls:15s} - Precision: {cls_metrics['precision']:.3f}, "
                    f"Recall: {cls_metrics['recall']:.3f}, "
                    f"F1-Score: {cls_metrics['f1-score']:.3f}"
                )

        return metrics
