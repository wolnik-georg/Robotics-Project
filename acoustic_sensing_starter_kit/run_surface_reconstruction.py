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

        # Color scheme (more distinct, colorblind-friendly)
        self.class_colors = {
            "contact": "#1b7837",  # Dark Green (was blue)
            "edge": "#ff7f00",  # Bright Orange (more distinct)
            "no_contact": "#d73027",  # Red (was green)
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

        # Check for original_label column to identify originally-excluded classes
        # This is important when edges were relabeled (e.g., edge -> contact)
        has_original_label = "original_label" in sweep_df.columns
        if has_original_label:
            original_class_counts = sweep_df["original_label"].value_counts()
            logger.info(f"  Original label distribution: {dict(original_class_counts)}")

        # Identify excluded classes (for visualization) but DON'T filter them out yet
        # We'll include them in visualization but exclude from accuracy calculation
        # Check BOTH label_column and original_label for excluded classes
        if exclude_classes:
            exclude_set = set(exclude_classes)
            # First check the label column
            excluded_by_label = sweep_df[label_column].isin(exclude_set)
            # Also check original_label if it exists (for relabeled edges)
            if has_original_label:
                excluded_by_original = sweep_df["original_label"].isin(exclude_set)
                excluded_in_sweep = excluded_by_label | excluded_by_original
            else:
                excluded_in_sweep = excluded_by_label
            excluded_count = excluded_in_sweep.sum()
            logger.info(
                f"  ℹ️  Classes {exclude_classes} will be shown but excluded from accuracy ({excluded_count} samples)"
            )
            # Store the exclusion info in the dataframe for later use
            sweep_df["_excluded"] = excluded_in_sweep
        else:
            sweep_df["_excluded"] = False

        # Extract features from ALL audio files (including excluded classes)
        features, labels, coords, excluded_mask, original_labels = (
            self._extract_features(sweep_df, dataset_path, label_column)
        )

        logger.info(f"  ✓ Extracted features for {len(features)} samples")
        logger.info(f"    Feature dimension: {features.shape[1]}")

        # Scale features using the trained scaler
        features_scaled = self.scaler.transform(features)

        # Make predictions for ALL samples
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)

        # Get confidence scores
        confidences = np.max(probabilities, axis=1)

        # Use the excluded_mask from feature extraction (based on original_label)
        included_mask = ~excluded_mask

        # For excluded samples, set prediction to their original label for visualization
        # (e.g., edge samples show as "edge" not what the model predicted)
        predictions_for_viz = predictions.copy()
        if np.any(excluded_mask):
            predictions_for_viz[excluded_mask] = original_labels[excluded_mask]
            logger.info(f"  ℹ️  Excluded from accuracy: {np.sum(excluded_mask)} samples")
        # Calculate raw accuracy (only on NON-excluded samples)
        if np.sum(included_mask) > 0:
            raw_accuracy = accuracy_score(
                labels[included_mask], predictions[included_mask]
            )
            logger.info(
                f"  🎯 Raw Reconstruction Accuracy: {raw_accuracy:.2%} (on {np.sum(included_mask)} samples)"
            )
        else:
            raw_accuracy = 0.0
            logger.warning("  ⚠ No samples left after exclusion!")
        logger.info(f"     Mean confidence: {np.mean(confidences[included_mask]):.2%}")
        logger.info(
            f"     Median confidence: {np.median(confidences[included_mask]):.2%}"
        )

        # Apply confidence filtering if specified (only on included samples)
        filtered_accuracy = None
        confidence_stats = None
        low_confidence_mask = np.zeros(
            len(features), dtype=bool
        )  # Track low-confidence samples

        if confidence_threshold is not None and np.sum(included_mask) > 0:
            # Only consider included samples for confidence filtering
            high_conf_mask = (confidences >= confidence_threshold) & included_mask
            high_conf_count = np.sum(high_conf_mask)
            low_conf_count = np.sum(included_mask) - high_conf_count

            confidence_stats = {
                "threshold": confidence_threshold,
                "mode": confidence_mode,
                "high_confidence_count": int(high_conf_count),
                "low_confidence_count": int(low_conf_count),
                "high_confidence_pct": (
                    float(100 * high_conf_count / np.sum(included_mask))
                    if np.sum(included_mask) > 0
                    else 0
                ),
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
                        f"     Kept: {high_conf_count}/{np.sum(included_mask)} ({confidence_stats['high_confidence_pct']:.1f}%)"
                    )
                    logger.info(f"     🎯 Filtered Accuracy: {filtered_accuracy:.2%}")

                    # Mark low-confidence predictions to be skipped in visualization
                    low_confidence_mask = ~high_conf_mask & included_mask
                    logger.info(
                        f"     ℹ️  Visualization: {low_conf_count} low-confidence predictions will be skipped (not shown)"
                    )
                else:
                    logger.warning(
                        f"  ⚠ No predictions above threshold {confidence_threshold}"
                    )
            elif confidence_mode == "default":
                # Assign default class to low-confidence predictions (only for included)
                adjusted_predictions = predictions.copy()
                low_conf_included = (
                    ~(confidences >= confidence_threshold)
                ) & included_mask
                adjusted_predictions[low_conf_included] = default_class
                filtered_accuracy = accuracy_score(
                    labels[included_mask], adjusted_predictions[included_mask]
                )
                logger.info(
                    f"\n  📊 Confidence Filtering (threshold={confidence_threshold}, mode=default):"
                )
                logger.info(
                    f"     Defaulted to '{default_class}': {low_conf_count}/{np.sum(included_mask)}"
                )
                logger.info(f"     🎯 Adjusted Accuracy: {filtered_accuracy:.2%}")
                # Use adjusted predictions for visualization
                predictions_for_viz = adjusted_predictions.copy()
                predictions_for_viz[excluded_mask] = original_labels[excluded_mask]

        # Use filtered accuracy if available, otherwise raw
        accuracy = filtered_accuracy if filtered_accuracy is not None else raw_accuracy

        # For visualization, use original_labels for ground truth of excluded samples
        # This shows "edge" on the plot instead of the relabeled class
        true_labels_for_viz = labels.copy()
        if np.any(excluded_mask):
            true_labels_for_viz[excluded_mask] = original_labels[excluded_mask]

        # Generate visualizations (use predictions_for_viz which keeps excluded classes as-is)
        logger.info("\n🎨 Generating surface reconstruction visualizations...")
        self._create_all_visualizations(
            dataset_name,
            coords,
            true_labels_for_viz,
            predictions_for_viz,
            probabilities,
            accuracy,
            excluded_mask=excluded_mask,
            low_confidence_mask=low_confidence_mask,  # Skip these in visualization
        )

        # Build results dictionary
        results = {
            "dataset": dataset_name,
            "num_samples": len(features),
            "num_included": int(np.sum(included_mask)),
            "num_excluded": int(np.sum(excluded_mask)),
            "model_used": self.classifier_name,
            "raw_accuracy": float(raw_accuracy),
            "accuracy": float(accuracy),
            "confidence_filtering": confidence_stats,
            "coordinates": coords.tolist(),
            "true_labels": labels.tolist(),
            "original_labels": original_labels.tolist(),
            "predictions": predictions_for_viz.tolist(),
            "excluded_classes": exclude_classes or [],
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from sweep audio files.

        Returns:
            Tuple of (features, labels, coords, excluded_mask)
        """
        features_list = []
        labels_list = []
        coords_list = []
        excluded_list = []
        original_labels_list = []

        logger.info("  🎵 Extracting features from audio files...")

        has_original_label = "original_label" in sweep_df.columns
        has_excluded = "_excluded" in sweep_df.columns

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
                excluded_list.append(row["_excluded"] if has_excluded else False)
                original_labels_list.append(
                    row["original_label"] if has_original_label else row[label_column]
                )

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
            np.array(excluded_list, dtype=bool),
            np.array(original_labels_list),
        )

    def _create_all_visualizations(
        self,
        dataset_name: str,
        coords: np.ndarray,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        accuracy: float,
        excluded_mask: Optional[np.ndarray] = None,
        low_confidence_mask: Optional[np.ndarray] = None,
    ):
        """Generate all surface reconstruction visualizations.

        Args:
            excluded_mask: Boolean array indicating which samples are from excluded
                          classes (e.g., edge). These will be shown with special
                          styling but not counted in accuracy.
            low_confidence_mask: Boolean array indicating which samples have low
                          confidence and should be skipped (not visualized at all).
        """
        output_subdir = self.output_dir / dataset_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Get unique classes from actual data
        all_classes = sorted(set(true_labels) | set(predictions))

        # If no excluded_mask provided, nothing is excluded
        if excluded_mask is None:
            excluded_mask = np.zeros(len(true_labels), dtype=bool)

        # If no low_confidence_mask provided, show all predictions
        if low_confidence_mask is None:
            low_confidence_mask = np.zeros(len(true_labels), dtype=bool)

        # 1. Ground truth grid (always show all, no low-confidence filtering)
        self._create_grid_heatmap(
            coords,
            true_labels,
            all_classes,
            "Ground Truth Surface",
            output_subdir / "01_ground_truth_grid.png",
            excluded_mask=excluded_mask,
            low_confidence_mask=np.zeros(
                len(true_labels), dtype=bool
            ),  # Always show ground truth
        )

        # 2. Predicted grid (skip low-confidence predictions)
        self._create_grid_heatmap(
            coords,
            predictions,
            all_classes,
            f"Predicted Surface ({self.classifier_name})",
            output_subdir / "02_predicted_grid.png",
            excluded_mask=excluded_mask,
            low_confidence_mask=low_confidence_mask,  # Skip low-confidence
        )

        # 3. Side-by-side comparison (skip low-confidence in predictions only)
        self._create_side_by_side(
            coords,
            true_labels,
            predictions,
            all_classes,
            accuracy,
            output_subdir / "03_comparison.png",
            excluded_mask=excluded_mask,
            low_confidence_mask=low_confidence_mask,  # Skip low-confidence
        )

        # 4. Error map (skip low-confidence)
        self._create_error_map(
            coords,
            true_labels,
            predictions,
            all_classes,
            output_subdir / "04_error_map.png",
            excluded_mask=excluded_mask,
            low_confidence_mask=low_confidence_mask,
        )

        # 5. Confidence map (always show all to see confidence distribution)
        self._create_confidence_map(
            coords,
            true_labels,
            probabilities,
            all_classes,
            output_subdir / "05_confidence_map.png",
            excluded_mask=excluded_mask,
            low_confidence_mask=np.zeros(
                len(true_labels), dtype=bool
            ),  # Show all for confidence viz
        )

        # 6. Presentation summary (skip low-confidence in predictions)
        self._create_presentation_summary(
            coords,
            true_labels,
            predictions,
            probabilities,
            all_classes,
            accuracy,
            dataset_name,
            output_subdir / "06_presentation_summary.png",
            excluded_mask=excluded_mask,
            low_confidence_mask=low_confidence_mask,
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
        excluded_mask: Optional[np.ndarray] = None,
        low_confidence_mask: Optional[np.ndarray] = None,
    ):
        """Create a grid-based heatmap with filled cells.

        Args:
            excluded_mask: Boolean array marking excluded samples (e.g., edge class).
                          These will be shown with a hatched pattern.
            low_confidence_mask: Boolean array marking low-confidence samples.
                          These will be skipped (not visualized at all).
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        if excluded_mask is None:
            excluded_mask = np.zeros(len(labels), dtype=bool)

        if low_confidence_mask is None:
            low_confidence_mask = np.zeros(len(labels), dtype=bool)

        # Plot each point as a filled rectangle
        for i, (coord, label) in enumerate(zip(coords, labels)):
            # Skip low-confidence predictions entirely
            if low_confidence_mask[i]:
                continue

            x, y = coord
            x, y = coord
            is_excluded = excluded_mask[i]

            if label in self.class_colors:
                color = self.class_colors[label]
            else:
                color = "gray"

            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white" if not is_excluded else "black",
                linewidth=0.5 if not is_excluded else 1.0,
                hatch="//" if is_excluded else None,  # Hatching for excluded
                alpha=0.7 if is_excluded else 1.0,
            )
            ax.add_patch(rect)

        # Set limits to FULL surface (0 to 1) with small padding
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        # Create legend (include excluded class info if present)
        legend_patches = [
            mpatches.Patch(
                color=self.class_colors.get(c, "gray"),
                label=c.replace("_", " ").title(),
            )
            for c in classes
            if c not in ["edge"]  # Regular classes (skip edge for special handling)
        ]
        # Add edge legend if present
        if "edge" in classes:
            legend_patches.append(
                mpatches.Patch(
                    facecolor=self.class_colors.get("edge", "#ff7f00"),
                    edgecolor="black",
                    hatch="//",
                    label="Edge",
                    alpha=0.7,
                )
            )
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
        excluded_mask: Optional[np.ndarray] = None,
        low_confidence_mask: Optional[np.ndarray] = None,
    ):
        """Create side-by-side comparison of ground truth vs predictions."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        if excluded_mask is None:
            excluded_mask = np.zeros(len(true_labels), dtype=bool)

        if low_confidence_mask is None:
            low_confidence_mask = np.zeros(len(true_labels), dtype=bool)

        for ax_idx, (ax, labels, title) in enumerate(
            [
                (axes[0], true_labels, "Ground Truth"),
                (axes[1], predictions, f"Predicted (Accuracy: {accuracy:.1%})"),
            ]
        ):
            for i, (coord, label) in enumerate(zip(coords, labels)):
                # Skip low-confidence predictions in the prediction panel (ax_idx==1)
                if ax_idx == 1 and low_confidence_mask[i]:
                    continue

                x, y = coord
                is_excluded = excluded_mask[i]
                color = self.class_colors.get(label, "gray")
                rect = Rectangle(
                    (x - dx / 2, y - dy / 2),
                    dx,
                    dy,
                    facecolor=color,
                    edgecolor="white" if not is_excluded else "black",
                    linewidth=0.3 if not is_excluded else 0.8,
                    hatch="//" if is_excluded else None,
                    alpha=0.7 if is_excluded else 1.0,
                )
                ax.add_patch(rect)

            # Set limits to FULL surface (0 to 1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
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
            if c not in ["edge"]
        ]
        if "edge" in classes:
            legend_patches.append(
                mpatches.Patch(
                    facecolor=self.class_colors.get("edge", "#ff7f00"),
                    edgecolor="black",
                    hatch="//",
                    label="Edge",
                    alpha=0.7,
                )
            )
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(legend_patches),
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
        excluded_mask: Optional[np.ndarray] = None,
        low_confidence_mask: Optional[np.ndarray] = None,
    ):
        """Create error map highlighting misclassifications."""
        fig, ax = plt.subplots(figsize=(10, 10))

        if excluded_mask is None:
            excluded_mask = np.zeros(len(true_labels), dtype=bool)

        if low_confidence_mask is None:
            low_confidence_mask = np.zeros(len(true_labels), dtype=bool)

        # Only count errors on non-excluded, high-confidence samples
        included_mask = ~excluded_mask & ~low_confidence_mask
        errors = (true_labels != predictions) & included_mask
        correct = (true_labels == predictions) & included_mask

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        # Plot excluded samples first (background)
        for i in np.where(excluded_mask)[0]:
            # Skip if also low confidence
            if low_confidence_mask[i]:
                continue
            x, y = coords[i]
            color = self.class_colors.get(true_labels[i], "gray")
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                hatch="//",
                alpha=0.4,
            )
            ax.add_patch(rect)

        # Plot correct predictions (muted colors)
        for i in np.where(correct)[0]:
            x, y = coords[i]
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
        for i in np.where(errors)[0]:
            x, y = coords[i]
            true_color = self.class_colors.get(true_labels[i], "blue")
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor="#d73027",  # Red for error
                edgecolor=true_color,
                linewidth=2,
            )
            ax.add_patch(rect)

        # Set limits to FULL surface (0 to 1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        # Error rate on included samples only
        num_included = np.sum(included_mask)
        error_count = np.sum(errors)
        error_rate = error_count / num_included if num_included > 0 else 0
        ax.set_title(
            f"Error Map: {error_count} Misclassifications ({error_rate:.1%})",
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
        if np.any(excluded_mask):
            legend_patches.append(
                mpatches.Patch(
                    facecolor="gray",
                    edgecolor="black",
                    hatch="//",
                    label="Excluded",
                    alpha=0.4,
                )
            )
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
        excluded_mask: Optional[np.ndarray] = None,
        low_confidence_mask: Optional[np.ndarray] = None,
    ):
        """Create confidence map showing prediction certainty."""
        fig, ax = plt.subplots(figsize=(10, 10))

        if excluded_mask is None:
            excluded_mask = np.zeros(len(true_labels), dtype=bool)

        if low_confidence_mask is None:
            low_confidence_mask = np.zeros(len(true_labels), dtype=bool)

        # Get max probability (confidence) for each sample
        max_probs = np.max(probabilities, axis=1)

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        # Plot cells with confidence-based coloring
        for i, (coord, confidence) in enumerate(zip(coords, max_probs)):
            # For confidence map, we can show all (including low confidence) to visualize the distribution
            # The low_confidence_mask is informational only here
            x, y = coord
            is_excluded = excluded_mask[i]

            if is_excluded:
                # Excluded samples shown with hatching
                color = self.class_colors.get(true_labels[i], "gray")
                rect = Rectangle(
                    (x - dx / 2, y - dy / 2),
                    dx,
                    dy,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch="//",
                    alpha=0.4,
                )
            else:
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

        # Set limits to FULL surface (0 to 1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        # Mean confidence on included samples only
        included_mask = ~excluded_mask
        mean_conf = np.mean(max_probs[included_mask]) if np.any(included_mask) else 0
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
        if np.any(excluded_mask):
            legend_patches.append(
                mpatches.Patch(
                    facecolor="gray",
                    edgecolor="black",
                    hatch="//",
                    label="Excluded",
                    alpha=0.4,
                )
            )
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
        excluded_mask: Optional[np.ndarray] = None,
        low_confidence_mask: Optional[np.ndarray] = None,
    ):
        """Create a presentation-ready summary figure (2x2 grid)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        if excluded_mask is None:
            excluded_mask = np.zeros(len(true_labels), dtype=bool)

        if low_confidence_mask is None:
            low_confidence_mask = np.zeros(len(true_labels), dtype=bool)

        included_mask = ~excluded_mask

        # Calculate cell size using robust method
        dx, dy = self._calculate_cell_size(coords)

        def setup_ax(ax, title):
            # Set limits to FULL surface (0 to 1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Normalized X", fontsize=10)
            ax.set_ylabel("Normalized Y", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

        # 1. Ground truth (top-left) - always show all
        ax = axes[0, 0]
        for i, (coord, label) in enumerate(zip(coords, true_labels)):
            x, y = coord
            is_excluded = excluded_mask[i]
            color = self.class_colors.get(label, "gray")
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white" if not is_excluded else "black",
                linewidth=0.3 if not is_excluded else 0.8,
                hatch="//" if is_excluded else None,
                alpha=0.7 if is_excluded else 1.0,
            )
            ax.add_patch(rect)
        setup_ax(ax, "Ground Truth")

        # 2. Predictions (top-right) - skip low-confidence predictions
        ax = axes[0, 1]
        for i, (coord, label) in enumerate(zip(coords, predictions)):
            # Skip low-confidence predictions
            if low_confidence_mask[i]:
                continue

            x, y = coord
            is_excluded = excluded_mask[i]
            color = self.class_colors.get(label, "gray")
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white" if not is_excluded else "black",
                linewidth=0.3 if not is_excluded else 0.8,
                hatch="//" if is_excluded else None,
                alpha=0.7 if is_excluded else 1.0,
            )
            ax.add_patch(rect)
        setup_ax(ax, f"Predicted ({self.classifier_name})")

        # 3. Error map (bottom-left) - only count errors on included, high-confidence samples
        ax = axes[1, 0]
        errors = (true_labels != predictions) & included_mask & ~low_confidence_mask
        num_included = np.sum(included_mask & ~low_confidence_mask)
        for i, (coord, true_label) in enumerate(zip(coords, true_labels)):
            # Skip low-confidence predictions
            if low_confidence_mask[i]:
                continue
            x, y = coord
            is_excluded = excluded_mask[i]
            is_error = errors[i]

            if is_excluded:
                color = self.class_colors.get(true_label, "gray")
                alpha = 0.4
                hatch = "//"
            elif is_error:
                color = "#d73027"
                alpha = 1.0
                hatch = None
            else:
                color = "lightgray"
                alpha = 0.5
                hatch = None
            rect = Rectangle(
                (x - dx / 2, y - dy / 2),
                dx,
                dy,
                facecolor=color,
                edgecolor="white" if not is_excluded else "black",
                linewidth=0.3,
                alpha=alpha,
                hatch=hatch,
            )
            ax.add_patch(rect)
        error_rate = errors.sum() / num_included if num_included > 0 else 0
        setup_ax(
            ax,
            f"Errors: {errors.sum()}/{num_included} ({error_rate*100:.1f}%)",
        )

        # 4. Confusion matrix (bottom-right) - only on included samples
        ax = axes[1, 1]
        # Filter to only included samples for confusion matrix
        included_true = true_labels[included_mask]
        included_pred = predictions[included_mask]
        # Use ALL classes from the model to show complete confusion matrix
        # even if some classes have zero samples in this specific object
        cm_classes = self.model.classes_
        cm = confusion_matrix(included_true, included_pred, labels=cm_classes)

        # Normalize for display
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

        # Add text annotations
        for i in range(len(cm_classes)):
            for j in range(len(cm_classes)):
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

        ax.set_xticks(range(len(cm_classes)))
        ax.set_yticks(range(len(cm_classes)))
        ax.set_xticklabels(
            [c.replace("_", " ").title() for c in cm_classes], rotation=45, ha="right"
        )
        ax.set_yticklabels([c.replace("_", " ").title() for c in cm_classes])
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

        # Legend for class colors (include edge with hatching if present)
        legend_patches = [
            mpatches.Patch(
                color=self.class_colors.get(c, "gray"),
                label=c.replace("_", " ").title(),
            )
            for c in classes
            if c not in ["edge"]
        ]
        if "edge" in classes:
            legend_patches.append(
                mpatches.Patch(
                    facecolor=self.class_colors.get("edge", "#ff7f00"),
                    edgecolor="black",
                    hatch="//",
                    label="Edge",
                    alpha=0.7,
                )
            )
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(legend_patches),
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
