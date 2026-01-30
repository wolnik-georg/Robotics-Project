#!/usr/bin/env python3
"""
Surface Reconstruction Runner

Standalone script to run surface reconstruction using pre-trained models
on collected datasets with sweep.csv (spatial coordinates).

Usage:
    python run_surface_reconstruction.py --model path/to/model.pkl --dataset collected_data_folder

    # To replicate validation accuracy, filter to balanced dataset subset:
    python run_surface_reconstruction.py --model path/to/model.pkl \\
        --dataset collected_data_runs_2026_01_15_workspace_1_squares_cutout \\
        --use-balanced-subset balanced_collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled_undersample

Author: Georg Wolnik
Date: January 2026
"""

import argparse
import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SurfaceReconstructor:
    """Surface reconstruction using pre-trained acoustic sensing models."""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "reconstruction_results",
        base_data_dir: str = "data",
        use_workspace_invariant: bool = True,
        use_impulse_features: bool = True,
    ):
        """
        Initialize the surface reconstructor.

        Args:
            model_path: Path to the saved model .pkl file
            output_dir: Directory for saving reconstruction results
            base_data_dir: Base directory for dataset folders
            use_workspace_invariant: Use workspace-invariant features (must match training)
            use_impulse_features: Use impulse response features (must match training)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.base_data_dir = Path(base_data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Feature extraction settings (must match training!)
        self.use_workspace_invariant = use_workspace_invariant
        self.use_impulse_features = use_impulse_features

        # Load the trained model
        self._load_model()

        # Feature extractor (configured to match training)
        self.feature_extractor = GeometricFeatureExtractor(
            sr=48000,
            use_workspace_invariant=use_workspace_invariant,
            use_impulse_features=use_impulse_features,
        )
        logger.info(
            f"  ✓ Feature config: workspace_invariant={use_workspace_invariant}, impulse={use_impulse_features}"
        )

        # Color scheme (colorblind-friendly)
        self.class_colors = {
            "contact": "#2166ac",  # Blue
            "edge": "#f4a582",  # Orange/Salmon
            "no_contact": "#4dac26",  # Green
        }
        self.class_to_num = {"contact": 0, "edge": 1, "no_contact": 2}

    def _load_model(self):
        """Load the trained model, scaler, and metadata."""
        logger.info(f"📦 Loading trained model from: {self.model_path}")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.classes = model_data.get("classes", ["contact", "no_contact"])
        self.classifier_name = model_data.get("classifier_name", "Unknown")
        self.metrics = model_data.get("metrics", {})

        logger.info(f"  ✓ Model: {self.classifier_name}")
        logger.info(f"  ✓ Classes: {self.classes}")
        if "validation_accuracy" in self.metrics:
            logger.info(
                f"  ✓ Training Val Accuracy: {self.metrics['validation_accuracy']:.2%}"
            )

    def process_dataset(
        self,
        dataset_name: str,
        label_column: str = "relabeled_label",
        exclude_classes: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        confidence_mode: str = "reject",
        default_class: str = "no_contact",
        balanced_subset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a collected dataset and run surface reconstruction.

        Args:
            dataset_name: Name of the dataset folder (e.g., collected_data_runs_*)
            label_column: Column name for labels in sweep.csv
            exclude_classes: List of class labels to exclude (e.g., ["edge"])
                            Use this to match training classes when model was
                            trained without certain classes.
            confidence_threshold: Min confidence to accept predictions (0.0-1.0).
                                 If None, no filtering. Training used 0.9.
            confidence_mode: "reject" (exclude low-confidence from metrics) or
                            "default" (assign default_class to low-confidence).
            default_class: Class to assign when mode="default".
            balanced_subset: If provided, filter to only include files from this
                           balanced dataset. This allows replicating validation
                           accuracy by using the exact same subset of files.

        Returns:
            Dictionary with reconstruction results
        """
        dataset_path = self.base_data_dir / dataset_name

        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_path}")

        sweep_csv_path = dataset_path / "sweep.csv"
        if not sweep_csv_path.exists():
            raise ValueError(f"sweep.csv not found in {dataset_path}")

        logger.info(f"\n📂 Processing dataset: {dataset_name}")

        # Load sweep CSV
        sweep_df = pd.read_csv(sweep_csv_path)
        original_count = len(sweep_df)
        logger.info(f"  ✓ Loaded {original_count} sweep points")
        logger.info(f"  Columns: {', '.join(sweep_df.columns.tolist())}")

        # Filter to balanced subset if specified (for validation accuracy replication)
        if balanced_subset:
            matched_files = self._get_balanced_subset_files(
                dataset_path, balanced_subset
            )
            if matched_files:
                before_count = len(sweep_df)
                sweep_df = sweep_df[sweep_df["acoustic_filename"].isin(matched_files)]
                logger.info(
                    f"  🎯 Filtered to balanced subset: {len(sweep_df)}/{before_count} files"
                )
                logger.info(f"     (Using files from: {balanced_subset})")
            else:
                logger.warning(
                    f"  ⚠ Could not find matching files in balanced dataset: {balanced_subset}"
                )

        # Determine label column
        if label_column not in sweep_df.columns:
            if "label" in sweep_df.columns:
                label_column = "label"
            else:
                raise ValueError(
                    f"Label column '{label_column}' not found in sweep.csv"
                )

        logger.info(f"  Using label column: {label_column}")

        # Show class distribution before filtering
        class_counts = sweep_df[label_column].value_counts()
        logger.info(f"  Class distribution: {dict(class_counts)}")

        # Filter out excluded classes (e.g., "edge" if training was binary)
        if exclude_classes:
            exclude_set = set(exclude_classes)
            before_count = len(sweep_df)
            sweep_df = sweep_df[~sweep_df[label_column].isin(exclude_set)]
            filtered_count = before_count - len(sweep_df)
            logger.info(
                f"  ⚠ Excluded classes {exclude_classes}: removed {filtered_count} samples"
            )
            logger.info(f"    Remaining: {len(sweep_df)} samples")

        # Extract features from audio files
        features, labels, coords = self._extract_features(
            sweep_df, dataset_path, label_column
        )

        logger.info(f"  ✓ Extracted features for {len(features)} samples")
        logger.info(f"    Feature dimension: {features.shape[1]}")

        # Scale features using the trained scaler
        features_scaled = self.scaler.transform(features)

        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)

        # Get confidence scores
        confidences = np.max(probabilities, axis=1)

        # Calculate raw accuracy (no filtering)
        raw_accuracy = accuracy_score(labels, predictions)
        logger.info(f"  🎯 Raw Reconstruction Accuracy: {raw_accuracy:.2%}")
        logger.info(f"     Mean confidence: {np.mean(confidences):.2%}")
        logger.info(f"     Median confidence: {np.median(confidences):.2%}")

        # Apply confidence filtering if specified
        filtered_accuracy = None
        confidence_stats = None
        if confidence_threshold is not None:
            high_conf_mask = confidences >= confidence_threshold
            high_conf_count = np.sum(high_conf_mask)
            low_conf_count = len(predictions) - high_conf_count

            confidence_stats = {
                "threshold": confidence_threshold,
                "mode": confidence_mode,
                "high_confidence_count": int(high_conf_count),
                "low_confidence_count": int(low_conf_count),
                "high_confidence_pct": float(100 * high_conf_count / len(predictions)),
            }

            if confidence_mode == "reject":
                # Only evaluate on high-confidence predictions
                if high_conf_count > 0:
                    filtered_labels = labels[high_conf_mask]
                    filtered_preds = predictions[high_conf_mask]
                    filtered_accuracy = accuracy_score(filtered_labels, filtered_preds)
                    logger.info(
                        f"\n  📊 Confidence Filtering (threshold={confidence_threshold}, mode=reject):"
                    )
                    logger.info(
                        f"     Kept: {high_conf_count}/{len(predictions)} ({confidence_stats['high_confidence_pct']:.1f}%)"
                    )
                    logger.info(f"     🎯 Filtered Accuracy: {filtered_accuracy:.2%}")
                else:
                    logger.warning(
                        f"  ⚠ No predictions above threshold {confidence_threshold}"
                    )
            elif confidence_mode == "default":
                # Assign default class to low-confidence predictions
                adjusted_predictions = predictions.copy()
                adjusted_predictions[~high_conf_mask] = default_class
                filtered_accuracy = accuracy_score(labels, adjusted_predictions)
                logger.info(
                    f"\n  📊 Confidence Filtering (threshold={confidence_threshold}, mode=default):"
                )
                logger.info(
                    f"     Defaulted to '{default_class}': {low_conf_count}/{len(predictions)}"
                )
                logger.info(f"     🎯 Adjusted Accuracy: {filtered_accuracy:.2%}")
                # Use adjusted predictions for visualization
                predictions = adjusted_predictions

        # Use filtered accuracy if available, otherwise raw
        accuracy = filtered_accuracy if filtered_accuracy is not None else raw_accuracy

        # Generate visualizations
        logger.info("\n🎨 Generating surface reconstruction visualizations...")
        self._create_all_visualizations(
            dataset_name, coords, labels, predictions, probabilities, accuracy
        )

        # Build results dictionary
        results = {
            "dataset": dataset_name,
            "num_samples": len(features),
            "model_used": self.classifier_name,
            "raw_accuracy": float(raw_accuracy),
            "accuracy": float(accuracy),
            "confidence_filtering": confidence_stats,
            "coordinates": coords.tolist(),
            "true_labels": labels.tolist(),
            "predictions": predictions.tolist(),
        }

        # Save results
        results_file = self.output_dir / f"{dataset_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  💾 Saved results to: {results_file}")

        return results

    def _get_balanced_subset_files(
        self, original_dataset_path: Path, balanced_dataset_name: str
    ) -> Set[str]:
        """
        Get the set of original audio filenames that correspond to the balanced dataset.

        The balanced datasets contain copies of audio files from the original dataset
        with renumbered filenames. We use MD5 hashing to match balanced files back
        to their original counterparts.

        Args:
            original_dataset_path: Path to the original collected dataset
            balanced_dataset_name: Name of the balanced dataset folder

        Returns:
            Set of original filenames (in sweep.csv format, e.g., "./data/119_contact.wav")
        """
        balanced_path = self.base_data_dir / balanced_dataset_name / "data"
        original_data_path = original_dataset_path / "data"

        if not balanced_path.exists():
            logger.warning(f"  Balanced dataset not found: {balanced_path}")
            return set()

        if not original_data_path.exists():
            logger.warning(f"  Original data folder not found: {original_data_path}")
            return set()

        # Build hash map for original files
        logger.info("  📊 Building file hash mapping...")
        original_hashes = {}
        for f in original_data_path.iterdir():
            if f.suffix == ".wav":
                hash_val = hashlib.md5(f.read_bytes()).hexdigest()
                original_hashes[hash_val] = f.name

        # Map balanced files to original filenames
        matched_files = set()
        for f in balanced_path.iterdir():
            if f.suffix == ".wav":
                hash_val = hashlib.md5(f.read_bytes()).hexdigest()
                original_name = original_hashes.get(hash_val)
                if original_name:
                    # Format to match sweep.csv acoustic_filename column
                    matched_files.add(f"./data/{original_name}")

        logger.info(f"     Matched {len(matched_files)} files via MD5 hash")
        return matched_files

    def _extract_features(
        self, sweep_df: pd.DataFrame, dataset_path: Path, label_column: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from sweep audio files."""
        features_list = []
        labels_list = []
        coords_list = []

        logger.info("  🎵 Extracting features from audio files...")

        for idx, row in sweep_df.iterrows():
            audio_filename = row["acoustic_filename"]

            # Handle relative paths like "./data/1_edge.wav"
            if audio_filename.startswith("./"):
                audio_filename = audio_filename[2:]

            audio_path = dataset_path / audio_filename

            if not audio_path.exists():
                logger.warning(f"    ⚠ Audio file not found: {audio_path}")
                continue

            try:
                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=48000, mono=True)

                # Extract features
                features_dict = self.feature_extractor.extract_features(
                    audio, method="comprehensive"
                )

                # Convert to feature vector
                if isinstance(features_dict, pd.Series):
                    feature_vector = features_dict.values
                elif isinstance(features_dict, dict):
                    feature_vector = np.array(list(features_dict.values()))
                else:
                    feature_vector = features_dict

                features_list.append(feature_vector)
                labels_list.append(row[label_column])
                coords_list.append((row["normalized_x"], row["normalized_y"]))

            except Exception as e:
                logger.warning(f"    ⚠ Failed to process {audio_path}: {e}")
                continue

            # Progress indicator
            if (idx + 1) % 50 == 0:
                logger.info(f"    Processed {idx + 1}/{len(sweep_df)} samples...")

        return (
            np.array(features_list),
            np.array(labels_list),
            np.array(coords_list),
        )

    def _create_all_visualizations(
        self,
        dataset_name: str,
        coords: np.ndarray,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        accuracy: float,
    ):
        """Generate all surface reconstruction visualizations."""
        output_subdir = self.output_dir / dataset_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Get unique classes from actual data
        all_classes = sorted(set(true_labels) | set(predictions))

        # 1. Ground truth grid
        self._create_grid_heatmap(
            coords,
            true_labels,
            all_classes,
            "Ground Truth Surface",
            output_subdir / "01_ground_truth_grid.png",
        )

        # 2. Predicted grid
        self._create_grid_heatmap(
            coords,
            predictions,
            all_classes,
            f"Predicted Surface ({self.classifier_name})",
            output_subdir / "02_predicted_grid.png",
        )

        # 3. Side-by-side comparison
        self._create_side_by_side(
            coords,
            true_labels,
            predictions,
            all_classes,
            accuracy,
            output_subdir / "03_comparison.png",
        )

        # 4. Error map
        self._create_error_map(
            coords,
            true_labels,
            predictions,
            all_classes,
            output_subdir / "04_error_map.png",
        )

        # 5. Confidence map
        self._create_confidence_map(
            coords,
            true_labels,
            probabilities,
            all_classes,
            output_subdir / "05_confidence_map.png",
        )

        # 6. Presentation summary
        self._create_presentation_summary(
            coords,
            true_labels,
            predictions,
            probabilities,
            all_classes,
            accuracy,
            dataset_name,
            output_subdir / "06_presentation_summary.png",
        )

        logger.info(f"  ✓ Saved 6 visualizations to: {output_subdir}")

    def _calculate_cell_size(self, coords: np.ndarray) -> Tuple[float, float]:
        """
        Calculate cell size for grid visualization.

        Uses robust method to handle robot positioning noise -
        rounds coordinates to 3 decimal places before computing grid spacing.
        """
        # Round to reduce floating point noise from robot positioning
        rounded_x = np.round(coords[:, 0], 3)
        rounded_y = np.round(coords[:, 1], 3)

        unique_x = np.sort(np.unique(rounded_x))
        unique_y = np.sort(np.unique(rounded_y))

        # Calculate cell size from rounded unique values
        if len(unique_x) > 1:
            diffs_x = np.diff(unique_x)
            # Use median to be robust to outliers
            dx = (
                np.median(diffs_x[diffs_x > 0.001]) if np.any(diffs_x > 0.001) else 0.05
            )
        else:
            dx = 0.1

        if len(unique_y) > 1:
            diffs_y = np.diff(unique_y)
            dy = (
                np.median(diffs_y[diffs_y > 0.001]) if np.any(diffs_y > 0.001) else 0.05
            )
        else:
            dy = 0.1

        # Ensure reasonable minimum size
        dx = max(dx, 0.02)
        dy = max(dy, 0.02)

        return dx, dy

    def _create_grid_heatmap(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        classes: List[str],
        title: str,
        save_path: Path,
    ):
        """Create a grid-based heatmap with filled cells."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        # Create color mapping
        color_list = [self.class_colors.get(c, "gray") for c in classes]
        cmap = ListedColormap(color_list)

        # Plot each point as a filled rectangle
        for i, (coord, label) in enumerate(zip(coords, labels)):
            x, y = coord
            if label in classes:
                color = self.class_colors.get(label, "gray")
            else:
                color = "gray"

            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)

        # Set limits with padding
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)

        # Create legend
        legend_patches = [
            mpatches.Patch(
                color=self.class_colors.get(c, "gray"),
                label=c.replace("_", " ").title(),
            )
            for c in classes
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=11)

        ax.set_xlabel("Normalized X Position", fontsize=12)
        ax.set_ylabel("Normalized Y Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linestyle="--")

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_side_by_side(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        classes: List[str],
        accuracy: float,
        save_path: Path,
    ):
        """Create side-by-side comparison of ground truth vs predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        for ax, labels, title in [
            (axes[0], true_labels, "Ground Truth"),
            (axes[1], predictions, f"Predicted (Accuracy: {accuracy:.1%})"),
        ]:
            for coord, label in zip(coords, labels):
                x, y = coord
                color = self.class_colors.get(label, "gray")
                rect = Rectangle(
                    (x - dx / 2, y - dy / 2),
                    dx,
                    dy,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.3,
                )
                ax.add_patch(rect)

            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            ax.set_xlim(x_min - dx, x_max + dx)
            ax.set_ylim(y_min - dy, y_max + dy)
            ax.set_xlabel("Normalized X", fontsize=11)
            ax.set_ylabel("Normalized Y", fontsize=11)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

        # Shared legend
        legend_patches = [
            mpatches.Patch(
                color=self.class_colors.get(c, "gray"),
                label=c.replace("_", " ").title(),
            )
            for c in classes
        ]
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(classes),
            fontsize=11,
            bbox_to_anchor=(0.5, 1.02),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_error_map(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        classes: List[str],
        save_path: Path,
    ):
        """Create error map highlighting misclassifications."""
        fig, ax = plt.subplots(figsize=(10, 10))

        errors = true_labels != predictions
        correct = ~errors

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        # Plot correct predictions (muted colors)
        for coord, label in zip(coords[correct], predictions[correct]):
            x, y = coord
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor="lightgray",
                edgecolor="white",
                linewidth=0.3,
                alpha=0.5,
            )
            ax.add_patch(rect)

        # Plot errors (bright red with edge showing true class)
        for coord, true_label in zip(coords[errors], true_labels[errors]):
            x, y = coord
            true_color = self.class_colors.get(true_label, "blue")
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor="#d73027",  # Red for error
                edgecolor=true_color,
                linewidth=2,
            )
            ax.add_patch(rect)

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)

        error_rate = errors.sum() / len(errors)
        ax.set_title(
            f"Error Map: {errors.sum()} Misclassifications ({error_rate:.1%})",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Normalized X", fontsize=11)
        ax.set_ylabel("Normalized Y", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        # Legend
        legend_patches = [
            mpatches.Patch(color="lightgray", label="Correct", alpha=0.5),
            mpatches.Patch(color="#d73027", label="Error"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=11)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_confidence_map(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
        classes: List[str],
        save_path: Path,
    ):
        """Create confidence map showing prediction certainty."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Get max probability (confidence) for each sample
        max_probs = np.max(probabilities, axis=1)

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        # Plot cells with confidence-based coloring
        for coord, confidence in zip(coords, max_probs):
            x, y = coord
            # Use confidence as alpha (0.3 to 1.0)
            alpha = 0.3 + 0.7 * confidence
            # Color based on confidence level
            if confidence > 0.8:
                color = "#2166ac"  # High confidence - blue
            elif confidence > 0.6:
                color = "#92c5de"  # Medium confidence - light blue
            else:
                color = "#f4a582"  # Low confidence - orange

            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                alpha=alpha,
            )
            ax.add_patch(rect)

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)

        mean_conf = np.mean(max_probs)
        ax.set_title(
            f"Prediction Confidence (Mean: {mean_conf:.1%})",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Normalized X", fontsize=11)
        ax.set_ylabel("Normalized Y", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

        # Legend
        legend_patches = [
            mpatches.Patch(color="#2166ac", label="High (>80%)"),
            mpatches.Patch(color="#92c5de", label="Medium (60-80%)"),
            mpatches.Patch(color="#f4a582", label="Low (<60%)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=11)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_presentation_summary(
        self,
        coords: np.ndarray,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        classes: List[str],
        accuracy: float,
        dataset_name: str,
        save_path: Path,
    ):
        """Create a presentation-ready summary figure (2x2 grid)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        def setup_ax(ax, title):
            ax.set_xlim(x_min - dx, x_max + dx)
            ax.set_ylim(y_min - dy, y_max + dy)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Normalized X", fontsize=10)
            ax.set_ylabel("Normalized Y", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

        # 1. Ground truth (top-left)
        ax = axes[0, 0]
        for coord, label in zip(coords, true_labels):
            x, y = coord
            color = self.class_colors.get(label, "gray")
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
            )
            ax.add_patch(rect)
        setup_ax(ax, "Ground Truth")

        # 2. Predictions (top-right)
        ax = axes[0, 1]
        for coord, label in zip(coords, predictions):
            x, y = coord
            color = self.class_colors.get(label, "gray")
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
            )
            ax.add_patch(rect)
        setup_ax(ax, f"Predicted ({self.classifier_name})")

        # 3. Error map (bottom-left)
        ax = axes[1, 0]
        errors = true_labels != predictions
        for coord, is_error, true_label in zip(coords, errors, true_labels):
            x, y = coord
            if is_error:
                color = "#d73027"
                alpha = 1.0
            else:
                color = "lightgray"
                alpha = 0.5
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                alpha=alpha,
            )
            ax.add_patch(rect)
        setup_ax(
            ax,
            f"Errors: {errors.sum()}/{len(errors)} ({errors.sum()/len(errors)*100:.1f}%)",
        )

        # 4. Confusion matrix (bottom-right)
        ax = axes[1, 1]
        cm = confusion_matrix(true_labels, predictions, labels=classes)

        # Normalize for display
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm_normalized[i, j]:.0%})",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(
            [c.replace("_", " ").title() for c in classes], rotation=45, ha="right"
        )
        ax.set_yticklabels([c.replace("_", " ").title() for c in classes])
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Proportion", fontsize=10)

        # Add overall title
        fig.suptitle(
            f"Surface Reconstruction: {dataset_name}\n"
            f"Model: {self.classifier_name} | Accuracy: {accuracy:.1%}",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )

        # Legend for class colors
        legend_patches = [
            mpatches.Patch(
                color=self.class_colors.get(c, "gray"),
                label=c.replace("_", " ").title(),
            )
            for c in classes
        ]
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(classes),
            fontsize=11,
            bbox_to_anchor=(0.5, 0.99),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run surface reconstruction using trained models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model .pkl file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder name (e.g., collected_data_runs_2026_01_15_workspace_1_pure_contact)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="surface_reconstruction_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="relabeled_label",
        help="Column name for labels in sweep.csv",
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        nargs="+",
        default=None,
        help="Classes to exclude from reconstruction (e.g., --exclude-classes edge). "
        "Use when the model was trained without certain classes.",
    )
    parser.add_argument(
        "--exclude-edge",
        action="store_true",
        help="Shorthand to exclude 'edge' class (equivalent to --exclude-classes edge)",
    )
    parser.add_argument(
        "--no-workspace-invariant",
        action="store_true",
        help="Disable workspace-invariant features (default: enabled to match training)",
    )
    parser.add_argument(
        "--no-impulse-features",
        action="store_true",
        help="Disable impulse response features (default: enabled to match training)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for filtering predictions (0.0-1.0). "
        "Training used 0.9. If not set, no filtering is applied.",
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        choices=["reject", "default"],
        default="reject",
        help="Confidence filtering mode: 'reject' excludes low-confidence from metrics, "
        "'default' assigns default class to low-confidence predictions.",
    )
    parser.add_argument(
        "--default-class",
        type=str,
        default="no_contact",
        help="Default class for low-confidence predictions when mode='default'.",
    )
    parser.add_argument(
        "--use-balanced-subset",
        type=str,
        default=None,
        help="Filter to only include files from this balanced dataset. "
        "Use to replicate validation accuracy by using the exact same file subset. "
        "Example: --use-balanced-subset balanced_collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled_undersample",
    )
    args = parser.parse_args()

    # Handle exclude-edge shorthand
    exclude_classes = args.exclude_classes
    if args.exclude_edge:
        exclude_classes = exclude_classes or []
        if "edge" not in exclude_classes:
            exclude_classes.append("edge")

    # Run reconstruction
    reconstructor = SurfaceReconstructor(
        model_path=args.model,
        output_dir=args.output,
        base_data_dir=args.data_dir,
        use_workspace_invariant=not args.no_workspace_invariant,
        use_impulse_features=not args.no_impulse_features,
    )

    results = reconstructor.process_dataset(
        dataset_name=args.dataset,
        label_column=args.label_column,
        exclude_classes=exclude_classes,
        confidence_threshold=args.confidence_threshold,
        confidence_mode=args.confidence_mode,
        default_class=args.default_class,
        balanced_subset=args.use_balanced_subset,
    )

    logger.info("\n" + "=" * 60)
    logger.info("✅ Surface Reconstruction Complete!")
    logger.info(f"   Dataset: {results['dataset']}")
    logger.info(f"   Samples: {results['num_samples']}")
    if results.get("confidence_filtering"):
        logger.info(f"   Raw Accuracy: {results['raw_accuracy']:.2%}")
        logger.info(f"   Filtered Accuracy: {results['accuracy']:.2%}")
    else:
        logger.info(f"   Accuracy: {results['accuracy']:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
