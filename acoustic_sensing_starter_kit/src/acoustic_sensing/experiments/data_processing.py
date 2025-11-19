from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import os


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
            if d.startswith("soft_finger_batch_")
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
                    self.logger.info(
                        f"Extracting features from {len(audio_data)} audio samples..."
                    )

                    # Feature extraction (same as working code)
                    feature_extractor = GeometricFeatureExtractor(sr=48000)

                    X_feat = []
                    failed_count = 0

                    for i, audio in enumerate(audio_data):
                        try:
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
