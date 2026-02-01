#!/usr/bin/env python3
"""
Simple Surface Reconstruction using Balanced Datasets
=====================================================

This module performs surface reconstruction using:
1. A trained model (from discrimination_analysis)
2. Balanced datasets with sweep.csv (containing position info)

The sweep.csv in balanced datasets provides (x, y) positions for each audio file,
enabling reconstruction without complex file matching.

Usage:
    from acoustic_sensing.experiments.surface_reconstruction_simple import SurfaceReconstructor

    reconstructor = SurfaceReconstructor(model_path, feature_extractor)
    results = reconstructor.reconstruct_dataset(dataset_path, output_dir)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import librosa
import logging
from sklearn.preprocessing import StandardScaler

# Import the feature extractor used during training
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor


class SurfaceReconstructor:
    """
    Surface reconstruction using balanced datasets with position info.
    """

    def __init__(
        self,
        model_path: str,
        sr: int = 48000,
        feature_config: Optional[Dict] = None,
        confidence_config: Optional[Dict] = None,
        position_aggregation: str = "highest_confidence",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the surface reconstructor.

        Args:
            model_path: Path to trained model .pkl file
            sr: Sample rate for audio
            feature_config: Feature extraction configuration
            confidence_config: Confidence filtering config from config.yml
                - enabled: bool (default False)
                - threshold: float (default 0.9)
                - mode: str "reject" or "default"
            position_aggregation: How to aggregate multiple samples at same position
                - "none": No aggregation, use all samples (may have duplicates)
                - "highest_confidence": Take prediction with highest confidence per position
                - "majority_vote": Take most common prediction per position
                - "confidence_weighted": Weight votes by confidence, take highest
            logger: Optional logger
        """
        self.sr = sr
        self.logger = logger or logging.getLogger(__name__)
        self.feature_config = feature_config or {}
        self.position_aggregation = position_aggregation

        # Confidence filtering config (from config.yml)
        self.confidence_config = confidence_config or {}
        self.confidence_enabled = self.confidence_config.get("enabled", False)
        self.confidence_threshold = self.confidence_config.get("threshold", 0.9)
        self.confidence_mode = self.confidence_config.get("mode", "reject")

        if self.confidence_enabled:
            self.logger.info(
                f"Confidence filtering enabled: threshold={self.confidence_threshold}, mode={self.confidence_mode}"
            )

        if self.position_aggregation != "none":
            self.logger.info(f"Position aggregation: {self.position_aggregation}")

        # Load model
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load trained model and scaler."""
        self.logger.info(f"Loading model from: {self.model_path}")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data.get("scaler")
        self.classes = model_data.get("classes", ["contact", "no_contact"])
        self.feature_names = model_data.get("feature_names", [])

        # Detect expected number of features from the scaler
        if self.scaler is not None and hasattr(self.scaler, "n_features_in_"):
            self.expected_features = self.scaler.n_features_in_
            self.logger.info(f"Model expects {self.expected_features} features")
        else:
            self.expected_features = None

        self.logger.info(f"Model loaded: {type(self.model).__name__}")
        self.logger.info(f"Classes: {self.classes}")

    def _aggregate_by_position(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Aggregate multiple predictions at the same position.

        Args:
            positions: Array of (x, y) positions
            labels: Ground truth labels
            predictions: Model predictions
            confidence: Prediction confidence scores

        Returns:
            Tuple of aggregated (positions, labels, predictions, confidence)
        """
        if self.position_aggregation == "none":
            return positions, labels, predictions, confidence

        # Create position keys for grouping
        pos_keys = [f"{p[0]:.6f},{p[1]:.6f}" for p in positions]
        unique_keys = list(dict.fromkeys(pos_keys))  # Preserve order

        n_original = len(positions)
        n_unique = len(unique_keys)

        if n_original == n_unique:
            self.logger.info("No duplicate positions found, skipping aggregation")
            return positions, labels, predictions, confidence

        self.logger.info(
            f"Aggregating {n_original} samples → {n_unique} unique positions "
            f"(method: {self.position_aggregation})"
        )

        # Group samples by position
        from collections import defaultdict

        groups = defaultdict(list)
        for i, key in enumerate(pos_keys):
            groups[key].append(i)

        # Aggregate each group
        agg_positions = []
        agg_labels = []
        agg_predictions = []
        agg_confidence = []

        for key in unique_keys:
            indices = groups[key]

            # Position and label are the same for all samples at this position
            agg_positions.append(positions[indices[0]])
            agg_labels.append(labels[indices[0]])  # Ground truth is same

            if len(indices) == 1:
                # Only one sample, no aggregation needed
                agg_predictions.append(predictions[indices[0]])
                agg_confidence.append(confidence[indices[0]])
            else:
                # Multiple samples - aggregate
                group_preds = predictions[indices]
                group_conf = confidence[indices]

                if self.position_aggregation == "highest_confidence":
                    # Take prediction with highest confidence
                    best_idx = np.argmax(group_conf)
                    agg_predictions.append(group_preds[best_idx])
                    agg_confidence.append(group_conf[best_idx])

                elif self.position_aggregation == "majority_vote":
                    # Take most common prediction (tie: use first)
                    unique_preds, counts = np.unique(group_preds, return_counts=True)
                    winner_idx = np.argmax(counts)
                    winner_pred = unique_preds[winner_idx]
                    # Use mean confidence of winning class
                    winner_mask = group_preds == winner_pred
                    agg_predictions.append(winner_pred)
                    agg_confidence.append(np.mean(group_conf[winner_mask]))

                elif self.position_aggregation == "confidence_weighted":
                    # Sum confidences per class, take class with highest total
                    unique_preds = np.unique(group_preds)
                    class_scores = {}
                    for pred in unique_preds:
                        mask = group_preds == pred
                        class_scores[pred] = np.sum(group_conf[mask])

                    winner_pred = max(class_scores, key=class_scores.get)
                    # Normalize confidence
                    total_conf = sum(class_scores.values())
                    agg_predictions.append(winner_pred)
                    agg_confidence.append(class_scores[winner_pred] / total_conf)
                else:
                    raise ValueError(
                        f"Unknown aggregation method: {self.position_aggregation}"
                    )

        return (
            np.array(agg_positions),
            np.array(agg_labels),
            np.array(agg_predictions),
            np.array(agg_confidence),
        )

    def _aggregate_for_visualization(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Always aggregate by position for visualization using highest confidence.

        This ensures visualizations show ONE prediction per position - the best guess.
        This is separate from the metrics aggregation and always uses highest_confidence.
        """
        from collections import defaultdict

        # Create position keys for grouping
        pos_keys = [f"{p[0]:.6f},{p[1]:.6f}" for p in positions]
        unique_keys = list(dict.fromkeys(pos_keys))

        if len(positions) == len(unique_keys):
            # No duplicates, return as-is
            return positions, labels, predictions, confidence

        # Group samples by position
        groups = defaultdict(list)
        for i, key in enumerate(pos_keys):
            groups[key].append(i)

        # Aggregate each group using highest confidence
        agg_positions = []
        agg_labels = []
        agg_predictions = []
        agg_confidence = []

        for key in unique_keys:
            indices = groups[key]

            agg_positions.append(positions[indices[0]])
            agg_labels.append(labels[indices[0]])

            if len(indices) == 1:
                agg_predictions.append(predictions[indices[0]])
                agg_confidence.append(confidence[indices[0]])
            else:
                # Take prediction with highest confidence
                group_conf = confidence[indices]
                best_idx = np.argmax(group_conf)
                agg_predictions.append(predictions[indices[best_idx]])
                agg_confidence.append(group_conf[best_idx])

        return (
            np.array(agg_positions),
            np.array(agg_labels),
            np.array(agg_predictions),
            np.array(agg_confidence),
        )

    def reconstruct_dataset(
        self, dataset_path: str, output_dir: str, feature_extractor=None
    ) -> Dict[str, Any]:
        """
        Reconstruct surface from a balanced dataset.

        Args:
            dataset_path: Path to balanced dataset folder (containing data/ and sweep.csv)
            output_dir: Output directory for visualizations
            feature_extractor: Feature extractor instance (from pipeline)

        Returns:
            Dictionary with reconstruction results
        """
        dataset_path = Path(dataset_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = dataset_path.name
        self.logger.info(f"Reconstructing: {dataset_name}")

        # Load sweep.csv
        sweep_path = dataset_path / "sweep.csv"
        if not sweep_path.exists():
            raise FileNotFoundError(f"sweep.csv not found in {dataset_path}")

        sweep_df = pd.read_csv(sweep_path)
        self.logger.info(f"Loaded {len(sweep_df)} samples from sweep.csv")

        # Extract features for all samples
        features, labels, positions = self._extract_all_features(
            dataset_path, sweep_df, feature_extractor
        )

        if len(features) == 0:
            raise ValueError("No features extracted!")

        # Scale features
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # Predict
        predictions = self.model.predict(features_scaled)

        # Get prediction probabilities if available
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features_scaled)
            confidence = np.max(probabilities, axis=1)
        else:
            confidence = np.ones(len(predictions))

        # Store raw sample count before aggregation
        n_raw_samples = len(predictions)
        raw_accuracy = np.mean(predictions == labels)
        self.logger.info(
            f"Raw accuracy (all {n_raw_samples} samples): {raw_accuracy:.2%}"
        )

        # Aggregate by position if enabled
        positions, labels, predictions, confidence = self._aggregate_by_position(
            positions, labels, predictions, confidence
        )

        # Calculate accuracy after aggregation
        accuracy = np.mean(predictions == labels)
        n_total = len(predictions)
        if n_total != n_raw_samples:
            self.logger.info(
                f"Aggregated accuracy ({n_total} positions): {accuracy:.2%}"
            )

        # Apply confidence filtering (from config)
        high_conf_accuracy = 0.0
        n_high_conf = 0

        if self.confidence_enabled:
            high_conf_mask = confidence >= self.confidence_threshold
            n_high_conf = np.sum(high_conf_mask)

            if n_high_conf > 0:
                high_conf_accuracy = np.mean(
                    predictions[high_conf_mask] == labels[high_conf_mask]
                )
                self.logger.info(
                    f"High-confidence accuracy (≥{self.confidence_threshold:.0%}): {high_conf_accuracy:.2%} "
                    f"({n_high_conf}/{n_total} samples = {n_high_conf/n_total:.1%})"
                )
            else:
                self.logger.warning(
                    f"No samples with confidence ≥ {self.confidence_threshold:.0%}"
                )

        # Create visualizations
        results = self._create_visualizations(
            positions,
            labels,
            predictions,
            confidence,
            output_dir,
            dataset_name,
            accuracy,
            high_conf_accuracy=high_conf_accuracy,
            confidence_threshold=(
                self.confidence_threshold if self.confidence_enabled else None
            ),
            n_high_conf=n_high_conf,
        )

        results.update(
            {
                "dataset": dataset_name,
                "n_samples": len(labels),
                "accuracy": accuracy,
                "confidence_filtering_enabled": self.confidence_enabled,
                "high_conf_accuracy": (
                    high_conf_accuracy if self.confidence_enabled else None
                ),
                "confidence_threshold": (
                    self.confidence_threshold if self.confidence_enabled else None
                ),
                "n_high_confidence": n_high_conf if self.confidence_enabled else None,
                "pct_high_confidence": (
                    n_high_conf / n_total
                    if self.confidence_enabled and n_total > 0
                    else None
                ),
                "mean_confidence": float(np.mean(confidence)),
                "position_aggregation": self.position_aggregation,
                "n_raw_samples": n_raw_samples,
                "n_positions": n_total,
                "raw_accuracy": raw_accuracy,
            }
        )

        return results

    def _extract_all_features(
        self, dataset_path: Path, sweep_df: pd.DataFrame, feature_extractor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features for all samples in the dataset.

        Returns:
            Tuple of (features, labels, positions)
        """
        features_list = []
        labels_list = []
        positions_list = []

        # Get label column
        label_col = (
            "relabeled_label"
            if "relabeled_label" in sweep_df.columns
            else "original_label"
        )

        data_dir = dataset_path / "data"

        # If no feature extractor provided but we know expected features,
        # create the GeometricFeatureExtractor that matches training
        if feature_extractor is None and self.expected_features is not None:
            # Create default GeometricFeatureExtractor matching training config
            # The training used workspace_invariant=True and impulse_features=True
            self.logger.info(
                f"Creating GeometricFeatureExtractor (expected {self.expected_features} features)"
            )
            feature_extractor = GeometricFeatureExtractor(
                sr=self.sr,
                use_workspace_invariant=True,
                use_impulse_features=True,  # Matches multi_dataset_config.yml
                use_contact_physics_features=True,
            )

        for idx, row in sweep_df.iterrows():
            filename = row["filename"]
            audio_path = data_dir / filename

            if not audio_path.exists():
                self.logger.warning(f"Audio file not found: {audio_path}")
                continue

            try:
                # Load audio
                audio, _ = librosa.load(audio_path, sr=self.sr)

                # Extract features
                if feature_extractor is not None:
                    feat = feature_extractor.extract_features(audio)
                else:
                    # Fallback: use simple features (shouldn't reach here normally)
                    feat = self._extract_simple_features(audio)

                features_list.append(feat)
                labels_list.append(row[label_col])

                # Get position (normalized x, y)
                x = row.get("normalized_x", 0.5)
                y = row.get("normalized_y", 0.5)
                positions_list.append([x, y])

            except Exception as e:
                self.logger.warning(f"Failed to process {filename}: {e}")
                continue

            if (idx + 1) % 100 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(sweep_df)} samples...")

        return (
            np.array(features_list),
            np.array(labels_list),
            np.array(positions_list),
        )

    def _extract_simple_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract simple features as fallback.
        """
        # Basic spectral features
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        )
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        )
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        )
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        rms = np.mean(librosa.feature.rms(y=audio))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        return np.concatenate(
            [
                [
                    spectral_centroid,
                    spectral_bandwidth,
                    spectral_rolloff,
                    zero_crossing_rate,
                    rms,
                ],
                mfcc_means,
            ]
        )

    def _infer_edge_positions(
        self, positions: np.ndarray, grid_spacing: float = 0.0125
    ) -> np.ndarray:
        """
        Infer edge positions by finding gaps in contact + no_contact coverage.

        The robot moves in 1cm increments, which corresponds to ~0.0125 in normalized coordinates.
        We infer the actual grid from the data positions, then mark uncovered grid points as "edge".

        Args:
            positions: Actual (x, y) positions from contact/no-contact data
            grid_spacing: Expected spacing between grid points (default: 0.0125 ≈ 1cm)

        Returns:
            Array of inferred edge positions (x, y)
        """
        if len(positions) == 0:
            return np.array([]).reshape(0, 2)

        # Extract unique X and Y positions from the actual data
        unique_x = np.unique(positions[:, 0])
        unique_y = np.unique(positions[:, 1])

        # Infer the full grid from the data's X and Y ranges
        # Use the actual unique positions as the grid axes
        x_grid = unique_x
        y_grid = unique_y

        # Create all possible grid combinations
        xx, yy = np.meshgrid(x_grid, y_grid)
        full_grid = np.column_stack([xx.ravel(), yy.ravel()])

        # Create a set of actual positions for fast lookup (round to avoid floating point issues)
        tolerance = grid_spacing * 0.3  # 30% of grid spacing for matching
        positions_set = set()
        for pos in positions:
            # Round to 4 decimal places for consistent comparison
            pos_key = (round(pos[0], 4), round(pos[1], 4))
            positions_set.add(pos_key)

        # Find grid points NOT in the contact/no-contact data
        edge_positions = []
        for grid_point in full_grid:
            grid_key = (round(grid_point[0], 4), round(grid_point[1], 4))

            # Check if this grid point is in the data
            found = False
            for pos_key in positions_set:
                dist = np.sqrt(
                    (pos_key[0] - grid_key[0]) ** 2 + (pos_key[1] - grid_key[1]) ** 2
                )
                if dist < tolerance:
                    found = True
                    break

            if not found:
                edge_positions.append(grid_point)

        if len(edge_positions) == 0:
            self.logger.info(
                f"No edge positions inferred (all grid points covered by data)"
            )
            return np.array([]).reshape(0, 2)

        edge_array = np.array(edge_positions)
        self.logger.info(
            f"Inferred {len(edge_array)} edge positions from grid gaps (grid: {len(x_grid)}×{len(y_grid)} = {len(full_grid)} total)"
        )
        return edge_array

    def _create_visualizations(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        confidence: np.ndarray,
        output_dir: Path,
        dataset_name: str,
        accuracy: float,
        high_conf_accuracy: float = 0.0,
        confidence_threshold: float = 0.9,
        n_high_conf: int = 0,
    ) -> Dict[str, str]:
        """
        Create surface reconstruction visualizations with inferred edge positions.

        Always aggregates by position for visualization (using highest confidence)
        to show one prediction per position - the "best guess" for each location.

        Edge positions are inferred from missing grid coverage and shown in BLACK.
        Edge positions are NOT used in training/testing/validation or metrics calculation.
        """
        # Aggregate by position for visualization (always use highest confidence)
        # This ensures we show the best prediction per position
        viz_positions, viz_labels, viz_predictions, viz_confidence = (
            self._aggregate_for_visualization(
                positions, labels, predictions, confidence
            )
        )

        # DISABLED: Edge visualization removed per user request
        # edge_positions = self._infer_edge_positions(viz_positions)
        edge_positions = np.array([]).reshape(0, 2)  # Empty array - no edges

        x = viz_positions[:, 0]
        y = viz_positions[:, 1]

        # Calculate per-position accuracy for display (contact/no-contact ONLY)
        viz_accuracy = np.mean(viz_predictions == viz_labels)

        # Get unique classes
        unique_labels = np.unique(np.concatenate([viz_labels, viz_predictions]))

        # Create color map (edge is BLACK for inferred positions)
        colors = {"contact": "green", "no_contact": "red", "edge": "black"}

        # 1. Ground Truth vs Predicted (side by side) with inferred edges
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Ground truth (aggregated - one point per position)
        for label in unique_labels:
            mask = viz_labels == label
            axes[0].scatter(
                x[mask],
                y[mask],
                c=colors.get(label, "gray"),
                label=label,
                alpha=0.7,
                s=50,
            )

        # Add inferred edge positions to ground truth (black)
        if len(edge_positions) > 0:
            axes[0].scatter(
                edge_positions[:, 0],
                edge_positions[:, 1],
                c="black",
                label="edge (inferred)",
                alpha=0.5,
                s=30,
                marker="s",
            )

        axes[0].set_xlabel("Normalized X")
        axes[0].set_ylabel("Normalized Y")
        total_positions = len(viz_labels) + len(edge_positions)
        axes[0].set_title(
            f"Ground Truth ({len(viz_labels)} data + {len(edge_positions)} edge = {total_positions} total)"
        )
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].set_aspect("equal")

        # Predicted (aggregated - best prediction per position)
        for label in unique_labels:
            mask = viz_predictions == label
            axes[1].scatter(
                x[mask],
                y[mask],
                c=colors.get(label, "gray"),
                label=label,
                alpha=0.7,
                s=50,
            )

        # Add inferred edge positions to predictions (black, no model prediction)
        if len(edge_positions) > 0:
            axes[1].scatter(
                edge_positions[:, 0],
                edge_positions[:, 1],
                c="black",
                label="edge (no prediction)",
                alpha=0.5,
                s=30,
                marker="s",
            )

        axes[1].set_xlabel("Normalized X")
        axes[1].set_ylabel("Normalized Y")
        # Show both raw sample accuracy and per-position accuracy (contact/no-contact ONLY)
        title = (
            f"Predicted (Acc: {accuracy:.1%} on {len(viz_labels)} contact/no-contact)"
        )
        axes[1].set_title(title)
        axes[1].legend()
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_aspect("equal")

        plt.suptitle(f"Surface Reconstruction: {dataset_name}", fontsize=14)
        plt.tight_layout()

        comparison_path = output_dir / f"{dataset_name}_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Error Map (per-position, contact/no-contact only) with inferred edges
        fig, ax = plt.subplots(figsize=(8, 7))

        correct_mask = viz_predictions == viz_labels

        ax.scatter(
            x[correct_mask],
            y[correct_mask],
            c="green",
            label="Correct",
            alpha=0.6,
            s=50,
        )
        ax.scatter(
            x[~correct_mask],
            y[~correct_mask],
            c="red",
            label="Error",
            alpha=0.8,
            s=80,
            marker="x",
        )

        # Add inferred edge positions (no error calculation for edge)
        if len(edge_positions) > 0:
            ax.scatter(
                edge_positions[:, 0],
                edge_positions[:, 1],
                c="black",
                label="edge (no prediction)",
                alpha=0.5,
                s=30,
                marker="s",
            )

        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.set_title(
            f"Error Map: {dataset_name}\nAccuracy: {viz_accuracy:.1%} ({len(viz_labels)} contact/no-contact)"
        )
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        error_path = output_dir / f"{dataset_name}_error_map.png"
        plt.savefig(error_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 3. Confidence Map (per-position, contact/no-contact only) with inferred edges
        fig, ax = plt.subplots(figsize=(8, 7))

        scatter = ax.scatter(
            x, y, c=viz_confidence, cmap="RdYlGn", vmin=0.5, vmax=1.0, alpha=0.7, s=50
        )
        plt.colorbar(scatter, label="Confidence")

        # Add inferred edge positions (no confidence, shown as black)
        if len(edge_positions) > 0:
            ax.scatter(
                edge_positions[:, 0],
                edge_positions[:, 1],
                c="black",
                label="edge (no prediction)",
                alpha=0.5,
                s=30,
                marker="s",
            )
            ax.legend()

        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.set_title(
            f"Confidence Map: {dataset_name}\nMean: {np.mean(viz_confidence):.1%} ({len(viz_labels)} predictions)"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        confidence_path = output_dir / f"{dataset_name}_confidence.png"
        plt.savefig(confidence_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved visualizations to: {output_dir}")
        self.logger.info(f"  Contact/No-contact: {len(viz_labels)} positions")
        self.logger.info(f"  Inferred edges: {len(edge_positions)} positions")
        self.logger.info(
            f"  Total coverage: {len(viz_labels) + len(edge_positions)} positions"
        )

        return {
            "comparison_plot": str(comparison_path),
            "error_map": str(error_path),
            "confidence_map": str(confidence_path),
        }


def run_reconstruction_on_validation_datasets(
    model_path: str,
    validation_datasets: List[str],
    base_data_dir: str,
    output_dir: str,
    feature_extractor=None,
    sr: int = 48000,
) -> Dict[str, Any]:
    """
    Run surface reconstruction on all validation datasets.

    Args:
        model_path: Path to trained model
        validation_datasets: List of validation dataset names
        base_data_dir: Base directory containing datasets
        output_dir: Output directory for results
        feature_extractor: Feature extractor from pipeline
        sr: Sample rate

    Returns:
        Dictionary with reconstruction results for each dataset
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running reconstruction on {len(validation_datasets)} datasets")

    # Create reconstructor
    reconstructor = SurfaceReconstructor(model_path, sr=sr, logger=logger)

    all_results = {}

    for dataset_name in validation_datasets:
        dataset_path = Path(base_data_dir) / dataset_name

        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}")
            continue

        # Check for sweep.csv
        if not (dataset_path / "sweep.csv").exists():
            logger.warning(f"No sweep.csv in {dataset_name}, skipping")
            continue

        try:
            dataset_output = Path(output_dir) / dataset_name
            results = reconstructor.reconstruct_dataset(
                str(dataset_path), str(dataset_output), feature_extractor
            )
            all_results[dataset_name] = results

        except Exception as e:
            logger.error(f"Failed to reconstruct {dataset_name}: {e}")
            all_results[dataset_name] = {"error": str(e)}

    # Summary
    successful = [k for k, v in all_results.items() if "error" not in v]
    if successful:
        accuracies = [all_results[k]["accuracy"] for k in successful]
        logger.info(f"\n{'='*50}")
        logger.info(f"RECONSTRUCTION SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Datasets: {len(successful)}/{len(validation_datasets)}")
        logger.info(f"Mean Accuracy: {np.mean(accuracies):.1%}")
        for name in successful:
            logger.info(f"  {name}: {all_results[name]['accuracy']:.1%}")

    return all_results
