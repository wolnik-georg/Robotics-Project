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


class SurfaceReconstructor:
    """
    Surface reconstruction using balanced datasets with position info.
    """

    def __init__(
        self,
        model_path: str,
        sr: int = 48000,
        feature_config: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the surface reconstructor.

        Args:
            model_path: Path to trained model .pkl file
            sr: Sample rate for audio
            feature_config: Feature extraction configuration
            logger: Optional logger
        """
        self.sr = sr
        self.logger = logger or logging.getLogger(__name__)
        self.feature_config = feature_config or {}

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

        self.logger.info(f"Model loaded: {type(self.model).__name__}")
        self.logger.info(f"Classes: {self.classes}")

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

        # Calculate accuracy
        accuracy = np.mean(predictions == labels)
        self.logger.info(f"Reconstruction accuracy: {accuracy:.2%}")

        # Create visualizations
        results = self._create_visualizations(
            positions,
            labels,
            predictions,
            confidence,
            output_dir,
            dataset_name,
            accuracy,
        )

        results.update(
            {
                "dataset": dataset_name,
                "n_samples": len(labels),
                "accuracy": accuracy,
                "mean_confidence": float(np.mean(confidence)),
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
                    # Fallback: use simple features
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

    def _create_visualizations(
        self,
        positions: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        confidence: np.ndarray,
        output_dir: Path,
        dataset_name: str,
        accuracy: float,
    ) -> Dict[str, str]:
        """
        Create surface reconstruction visualizations.
        """
        x = positions[:, 0]
        y = positions[:, 1]

        # Get unique classes
        unique_labels = np.unique(np.concatenate([labels, predictions]))

        # Create color map
        colors = {"contact": "green", "no_contact": "red", "edge": "orange"}

        # 1. Ground Truth vs Predicted (side by side)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Ground truth
        for label in unique_labels:
            mask = labels == label
            axes[0].scatter(
                x[mask],
                y[mask],
                c=colors.get(label, "gray"),
                label=label,
                alpha=0.7,
                s=50,
            )
        axes[0].set_xlabel("Normalized X")
        axes[0].set_ylabel("Normalized Y")
        axes[0].set_title("Ground Truth")
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].set_aspect("equal")

        # Predicted
        for label in unique_labels:
            mask = predictions == label
            axes[1].scatter(
                x[mask],
                y[mask],
                c=colors.get(label, "gray"),
                label=label,
                alpha=0.7,
                s=50,
            )
        axes[1].set_xlabel("Normalized X")
        axes[1].set_ylabel("Normalized Y")
        axes[1].set_title(f"Predicted (Accuracy: {accuracy:.1%})")
        axes[1].legend()
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_aspect("equal")

        plt.suptitle(f"Surface Reconstruction: {dataset_name}", fontsize=14)
        plt.tight_layout()

        comparison_path = output_dir / f"{dataset_name}_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Error Map
        fig, ax = plt.subplots(figsize=(8, 7))

        correct_mask = predictions == labels

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

        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.set_title(f"Error Map: {dataset_name}\nAccuracy: {accuracy:.1%}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        error_path = output_dir / f"{dataset_name}_error_map.png"
        plt.savefig(error_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 3. Confidence Map
        fig, ax = plt.subplots(figsize=(8, 7))

        scatter = ax.scatter(
            x, y, c=confidence, cmap="RdYlGn", vmin=0.5, vmax=1.0, alpha=0.7, s=50
        )
        plt.colorbar(scatter, label="Confidence")

        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.set_title(f"Confidence Map: {dataset_name}\nMean: {np.mean(confidence):.1%}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        confidence_path = output_dir / f"{dataset_name}_confidence.png"
        plt.savefig(confidence_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved visualizations to: {output_dir}")

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
