from .base_experiment import BaseExperiment
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import json


class MultiDatasetTrainingExperiment(BaseExperiment):
    """
    Multi-Dataset Training Experiment

    Trains models on combined data from multiple datasets (with 80/20 train/test split)
    and validates on a separate holdout dataset to test generalization.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data_processing experiment."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run multi-dataset training and validation for all feature extraction modes.

        Args:
            shared_data: Contains processed data from data_processing experiment

        Returns:
            Dictionary containing training results and validation metrics for all modes
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Multi-Dataset Training & Validation Experiment")
        self.logger.info("=" * 80)

        # Check if multi-dataset mode is enabled
        multi_dataset_config = self.config.get("multi_dataset_training", {})
        if not multi_dataset_config.get("enabled", False):
            self.logger.warning(
                "Multi-dataset training is DISABLED. Enable it in config to use this experiment."
            )
            return {
                "status": "skipped",
                "reason": "multi_dataset_training.enabled = false",
            }

        # Extract configuration
        training_dataset_names = multi_dataset_config.get("training_datasets", [])
        validation_dataset_name = multi_dataset_config.get("validation_dataset", None)
        train_split = multi_dataset_config.get("train_test_split", 0.8)
        random_seed = multi_dataset_config.get("random_seed", 42)
        stratify = multi_dataset_config.get("stratify", True)

        if not training_dataset_names or not validation_dataset_name:
            raise ValueError(
                "Must specify 'training_datasets' (list) and 'validation_dataset' (string) in config"
            )

        # Get batch results from data_processing
        batch_results = shared_data.get("batch_results", {})
        if not batch_results:
            raise ValueError(
                "No batch_results found in shared_data. Run data_processing first."
            )

        # Detect feature extraction modes
        all_keys = list(batch_results.keys())

        # Check if we have mode-suffixed keys (e.g., "dataset_features", "dataset_spectrogram")
        mode_suffixed_keys = [
            k
            for k in all_keys
            if "_" in k
            and k.split("_", 1)[1]
            in [
                "features",
                "spectrogram",
                "mfcc",
                "magnitude_spectrum",
                "power_spectrum",
                "chroma",
                "both",
            ]
        ]

        if mode_suffixed_keys:
            # Multiple modes detected - extract unique modes
            modes = set()
            base_datasets = set()
            for key in mode_suffixed_keys:
                base, mode = key.rsplit("_", 1)
                modes.add(mode)
                base_datasets.add(base)

            modes = sorted(list(modes))
            base_datasets = sorted(list(base_datasets))

            self.logger.info(f"🔍 Detected multiple feature extraction modes: {modes}")
            self.logger.info(f"📊 Base datasets: {base_datasets}")

            # Run training for each mode
            all_results = {}
            for mode in modes:
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"🎯 Processing mode: {mode}")
                self.logger.info(f"{'='*80}")

                mode_results = self._run_single_mode(
                    mode,
                    training_dataset_names,
                    validation_dataset_name,
                    batch_results,
                    train_split,
                    random_seed,
                    stratify,
                )
                all_results[mode] = mode_results

            # Generate summary across all modes
            summary = self._generate_multi_mode_summary(all_results)
            all_results["_multi_mode_summary"] = summary

            return all_results

        else:
            # Single mode (backward compatibility)
            self.logger.info("🔍 Single feature extraction mode detected")
            results = self._run_single_mode(
                None,
                training_dataset_names,
                validation_dataset_name,
                batch_results,
                train_split,
                random_seed,
                stratify,
            )
            return results

        # Combine training datasets
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Step 1: Combining Training Datasets")
        self.logger.info("=" * 80)

        X_train_combined = []
        y_train_combined = []

        for dataset_name in training_dataset_names:
            if dataset_name not in batch_results:
                raise ValueError(
                    f"Training dataset '{dataset_name}' not found in processed data"
                )

            batch_data = batch_results[dataset_name]
            X = batch_data["features"]
            y = batch_data["labels"]

            self.logger.info(f"  {dataset_name}: {len(X)} samples")
            X_train_combined.append(X)
            y_train_combined.append(y)

        # Concatenate all training data
        X_train_combined = np.vstack(X_train_combined)
        y_train_combined = np.concatenate(y_train_combined)

        self.logger.info(f"✅ Combined Training Data: {len(X_train_combined)} samples")

        # Get feature names from first batch (consistent across all batches)
        first_batch_name = list(batch_results.keys())[0]
        feature_names = batch_results[first_batch_name].get("feature_names", None)
        if feature_names is None:
            # If not in batch results, create generic names
            num_features = X_train_combined.shape[1]
            feature_names = [f"feature_{i}" for i in range(num_features)]
            self.logger.warning(
                f"Feature names not found in batch results. Using generic names."
            )

        # Show class distribution in combined training data
        unique, counts = np.unique(y_train_combined, return_counts=True)
        self.logger.info("Class distribution in combined training data:")
        for cls, count in zip(unique, counts):
            self.logger.info(
                f"  {cls}: {count} samples ({count/len(y_train_combined)*100:.1f}%)"
            )

        # Split combined training data into train/test
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Step 2: Splitting Combined Data into Train/Test")
        self.logger.info("=" * 80)

        if stratify:
            stratify_labels = y_train_combined
        else:
            stratify_labels = None

        X_train, X_test, y_train, y_test = train_test_split(
            X_train_combined,
            y_train_combined,
            train_size=train_split,
            random_state=random_seed,
            stratify=stratify_labels,
        )

        self.logger.info(f"✅ Training Set: {len(X_train)} samples")
        self.logger.info(f"✅ Test Set: {len(X_test)} samples")

        # Load validation dataset
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Step 3: Loading Validation Dataset (Holdout)")
        self.logger.info("=" * 80)

        if validation_dataset_name not in batch_results:
            raise ValueError(
                f"Validation dataset '{validation_dataset_name}' not found in processed data"
            )

        validation_data = batch_results[validation_dataset_name]
        X_validation = validation_data["features"]
        y_validation = validation_data["labels"]

        self.logger.info(
            f"✅ Validation Set: {len(X_validation)} samples (never seen during training)"
        )

        # Show class distribution in validation data
        unique_val, counts_val = np.unique(y_validation, return_counts=True)
        self.logger.info("Class distribution in validation data:")
        for cls, count in zip(unique_val, counts_val):
            self.logger.info(
                f"  {cls}: {count} samples ({count/len(y_validation)*100:.1f}%)"
            )

        # Feature Scaling
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Step 4: Feature Scaling")
        self.logger.info("=" * 80)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_validation_scaled = scaler.transform(X_validation)

        self.logger.info(
            "✅ Features scaled using StandardScaler (fit on training data only)"
        )

        # Train multiple classifiers
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Step 5: Training Multiple Classifiers")
        self.logger.info("=" * 80)

        classifiers = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=random_seed, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=random_seed
            ),
            "SVM (RBF)": SVC(kernel="rbf", random_state=random_seed, probability=True),
            "SVM (Linear)": SVC(
                kernel="linear", random_state=random_seed, probability=True
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=random_seed, n_jobs=-1
            ),
        }

        results = {
            "config": {
                "training_datasets": training_dataset_names,
                "validation_dataset": validation_dataset_name,
                "train_split": train_split,
                "random_seed": random_seed,
                "stratify": stratify,
                "num_train_samples": len(X_train),
                "num_test_samples": len(X_test),
                "num_validation_samples": len(X_validation),
                "num_features": X_train.shape[1],
                "classes": list(unique),
            },
            "models": {},
            "trained_models": {},  # Store actual model objects for reuse
            "scaler": scaler,  # Store scaler for feature transformation
            "feature_names": feature_names,  # Store feature names
            "classes": list(unique),  # Store class names at top level too
        }

        for clf_name, clf in classifiers.items():
            self.logger.info(f"\n🤖 Training: {clf_name}")

            # Train
            clf.fit(X_train_scaled, y_train)

            # Predict on test set
            y_test_pred = clf.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")

            # Predict on validation set (HOLDOUT)
            y_val_pred = clf.predict(X_validation_scaled)
            val_accuracy = accuracy_score(y_validation, y_val_pred)
            val_f1 = f1_score(y_validation, y_val_pred, average="weighted")

            self.logger.info(
                f"  Test Accuracy: {test_accuracy:.4f} | F1: {test_f1:.4f}"
            )
            self.logger.info(
                f"  Validation Accuracy: {val_accuracy:.4f} | F1: {val_f1:.4f}"
            )

            # Store results (metrics)
            results["models"][clf_name] = {
                "test_accuracy": float(test_accuracy),
                "test_f1": float(test_f1),
                "validation_accuracy": float(val_accuracy),
                "validation_f1": float(val_f1),
                "test_predictions": y_test_pred.tolist(),
                "validation_predictions": y_val_pred.tolist(),
                "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
                "validation_confusion_matrix": confusion_matrix(
                    y_validation, y_val_pred
                ).tolist(),
                "test_classification_report": classification_report(
                    y_test, y_test_pred, output_dict=True
                ),
                "validation_classification_report": classification_report(
                    y_validation, y_val_pred, output_dict=True
                ),
            }

            # Store trained model object for reuse
            results["trained_models"][clf_name] = {
                "model": clf,
                "test_accuracy": float(test_accuracy),
                "validation_accuracy": float(val_accuracy),
            }

        # Save results
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Step 6: Saving Results")
        self.logger.info("=" * 80)

        self.save_results(results, "multi_dataset_training_results.json")

        # Create visualizations
        self._create_performance_comparison_plot(results)
        self._create_confusion_matrices(results, y_test, y_validation, unique)
        self._create_generalization_analysis_plot(results)

        # NEW: Additional visualizations showing dataset-level clustering
        # Get original dataset data for visualization
        dataset_features = {}
        dataset_labels = {}
        for dataset_name in training_dataset_names + [validation_dataset_name]:
            batch_data = batch_results[dataset_name]
            dataset_features[dataset_name] = batch_data["features"]
            dataset_labels[dataset_name] = batch_data["labels"]

        self._create_pca_visualization_by_dataset(
            dataset_features,
            dataset_labels,
            training_dataset_names,
            validation_dataset_name,
            unique,
            scaler,
        )
        self._create_tsne_visualization_by_dataset(
            dataset_features,
            dataset_labels,
            training_dataset_names,
            validation_dataset_name,
            unique,
            scaler,
        )
        self._create_top3_confusion_matrices(results, y_test, y_validation, unique)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ Multi-Dataset Training & Validation Completed")
        self.logger.info("=" * 80)

        return results

    def _create_performance_comparison_plot(self, results: Dict[str, Any]):
        """Create bar plot comparing test vs validation performance."""
        models = list(results["models"].keys())
        test_acc = [results["models"][m]["test_accuracy"] for m in models]
        val_acc = [results["models"][m]["validation_accuracy"] for m in models]
        test_f1 = [results["models"][m]["test_f1"] for m in models]
        val_f1 = [results["models"][m]["validation_f1"] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy comparison
        x = np.arange(len(models))
        width = 0.35

        ax1.bar(
            x - width / 2, test_acc, width, label="Test Set", alpha=0.8, color="skyblue"
        )
        ax1.bar(
            x + width / 2,
            val_acc,
            width,
            label="Validation Set (Holdout)",
            alpha=0.8,
            color="coral",
        )
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Model Accuracy: Test vs Validation")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        ax1.set_ylim([0, 1.0])

        # F1 Score comparison
        ax2.bar(
            x - width / 2,
            test_f1,
            width,
            label="Test Set",
            alpha=0.8,
            color="lightgreen",
        )
        ax2.bar(
            x + width / 2,
            val_f1,
            width,
            label="Validation Set (Holdout)",
            alpha=0.8,
            color="salmon",
        )
        ax2.set_xlabel("Model")
        ax2.set_ylabel("F1 Score")
        ax2.set_title("Model F1 Score: Test vs Validation")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_ylim([0, 1.0])

        plt.tight_layout()
        self.save_plot(fig, "performance_comparison.png")
        plt.close()

    def _create_confusion_matrices(
        self, results: Dict[str, Any], y_test, y_validation, classes
    ):
        """Create confusion matrices for best performing model."""
        # Find best model based on validation accuracy
        best_model = max(
            results["models"].items(), key=lambda x: x[1]["validation_accuracy"]
        )[0]

        self.logger.info(f"📊 Creating confusion matrices for best model: {best_model}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Test confusion matrix
        cm_test = np.array(results["models"][best_model]["test_confusion_matrix"])
        sns.heatmap(
            cm_test,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            xticklabels=classes,
            yticklabels=classes,
        )
        ax1.set_title(f"Confusion Matrix - Test Set\n{best_model}")
        ax1.set_ylabel("True Label")
        ax1.set_xlabel("Predicted Label")

        # Validation confusion matrix
        cm_val = np.array(results["models"][best_model]["validation_confusion_matrix"])
        sns.heatmap(
            cm_val,
            annot=True,
            fmt="d",
            cmap="Oranges",
            ax=ax2,
            xticklabels=classes,
            yticklabels=classes,
        )
        ax2.set_title(f"Confusion Matrix - Validation Set (Holdout)\n{best_model}")
        ax2.set_ylabel("True Label")
        ax2.set_xlabel("Predicted Label")

        plt.tight_layout()
        self.save_plot(fig, "confusion_matrices_best_model.png")
        plt.close()

    def _create_generalization_analysis_plot(self, results: Dict[str, Any]):
        """Create plot showing generalization gap (test acc - validation acc)."""
        models = list(results["models"].keys())
        test_acc = np.array([results["models"][m]["test_accuracy"] for m in models])
        val_acc = np.array(
            [results["models"][m]["validation_accuracy"] for m in models]
        )

        generalization_gap = test_acc - val_acc

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        colors = ["green" if gap < 0 else "red" for gap in generalization_gap]
        bars = ax.barh(models, generalization_gap, color=colors, alpha=0.7)

        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Generalization Gap (Test Acc - Validation Acc)")
        ax.set_title(
            "Model Generalization Analysis\n(Negative = Better generalization to unseen data)"
        )
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (model, gap) in enumerate(zip(models, generalization_gap)):
            ax.text(
                gap + 0.005 if gap > 0 else gap - 0.005,
                i,
                f"{gap:.3f}",
                va="center",
                ha="left" if gap > 0 else "right",
                fontsize=9,
            )

        plt.tight_layout()
        self.save_plot(fig, "generalization_analysis.png")
        plt.close()

    def _create_pca_visualization_by_dataset(
        self,
        dataset_features,
        dataset_labels,
        training_dataset_names,
        validation_dataset_name,
        classes,
        scaler,
    ):
        """Create PCA visualization showing original dataset distributions."""
        from sklearn.decomposition import PCA

        self.logger.info("📊 Creating PCA visualization by dataset...")

        # Combine all data and scale together
        all_features = []
        all_labels = []
        dataset_indices = {}
        current_idx = 0

        for dataset_name in training_dataset_names + [validation_dataset_name]:
            features = dataset_features[dataset_name]
            labels = dataset_labels[dataset_name]
            all_features.append(features)
            all_labels.append(labels)
            dataset_indices[dataset_name] = (current_idx, current_idx + len(features))
            current_idx += len(features)

        X_all = np.vstack(all_features)
        y_all = np.concatenate(all_labels)

        # Scale all data together
        X_all_scaled = scaler.transform(X_all)

        # Fit PCA on combined scaled data
        pca = PCA(n_components=2)
        X_all_pca = pca.fit_transform(X_all_scaled)

        # Create dataset labels for each sample
        dataset_labels_array = []
        for dataset_name in training_dataset_names + [validation_dataset_name]:
            start_idx, end_idx = dataset_indices[dataset_name]
            dataset_labels_array.extend([dataset_name] * (end_idx - start_idx))
        dataset_labels_array = np.array(dataset_labels_array)

        # Create 2 plots: one with all data combined, one with separate panels
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        # Use different colors for each class (dynamic based on actual classes)
        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        class_colors = {
            cls: default_colors[i % len(default_colors)]
            for i, cls in enumerate(classes)
        }

        # Dataset markers (dynamic based on number of datasets)
        marker_styles = ["o", "s", "^", "v", "D", "p", "h"]
        all_dataset_names = training_dataset_names + [validation_dataset_name]
        dataset_markers = {
            name: marker_styles[i % len(marker_styles)]
            for i, name in enumerate(all_dataset_names)
        }

        # Plot 0: All datasets combined - only show class labels
        dataset_names = training_dataset_names + [validation_dataset_name]

        # Plot by class only - ignore dataset origin
        for cls in classes:
            class_mask = y_all == cls

            axes[0].scatter(
                X_all_pca[class_mask, 0],
                X_all_pca[class_mask, 1],
                label=cls,
                alpha=0.6,
                s=40,
                c=[class_colors[cls]],
                edgecolors="black",
                linewidths=0.5,
            )

        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        axes[0].set_title("All Datasets Combined")
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Plot 1-3: Each dataset separately
        for idx, dataset_name in enumerate(dataset_names, start=1):
            start_idx, end_idx = dataset_indices[dataset_name]
            X_dataset_pca = X_all_pca[start_idx:end_idx]
            y_dataset = y_all[start_idx:end_idx]

            for cls in classes:
                mask = y_dataset == cls
                axes[idx].scatter(
                    X_dataset_pca[mask, 0],
                    X_dataset_pca[mask, 1],
                    label=cls,
                    alpha=0.6,
                    s=30,
                    c=[class_colors[cls]],
                )

            axes[idx].set_xlabel(
                f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
            )
            axes[idx].set_ylabel(
                f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
            )

            # Label training datasets vs validation
            if dataset_name in training_dataset_names:
                axes[idx].set_title(
                    f"{dataset_name} (Training)\n(n={end_idx - start_idx})"
                )
            else:
                axes[idx].set_title(
                    f"{dataset_name} (Validation - Holdout)\n(n={end_idx - start_idx})"
                )

            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.suptitle(
            "PCA Analysis: Dataset-Level Clustering (Combined PCA Fit)",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        self.save_plot(fig, "pca_analysis.png")
        plt.close()

    def _create_tsne_visualization_by_dataset(
        self,
        dataset_features,
        dataset_labels,
        training_dataset_names,
        validation_dataset_name,
        classes,
        scaler,
    ):
        """Create t-SNE visualization showing original dataset distributions."""
        from sklearn.manifold import TSNE

        self.logger.info("📊 Creating t-SNE visualization by dataset...")

        # Combine all data and scale together
        all_features = []
        all_labels = []
        dataset_indices = {}
        current_idx = 0

        for dataset_name in training_dataset_names + [validation_dataset_name]:
            features = dataset_features[dataset_name]
            labels = dataset_labels[dataset_name]
            all_features.append(features)
            all_labels.append(labels)
            dataset_indices[dataset_name] = (current_idx, current_idx + len(features))
            current_idx += len(features)

        X_all = np.vstack(all_features)
        y_all = np.concatenate(all_labels)

        # Scale all data together
        X_all_scaled = scaler.transform(X_all)

        # Create dataset labels for each sample BEFORE sampling
        dataset_labels_full = []
        for dataset_name in training_dataset_names + [validation_dataset_name]:
            start_idx, end_idx = dataset_indices[dataset_name]
            dataset_labels_full.extend([dataset_name] * (end_idx - start_idx))
        dataset_labels_full = np.array(dataset_labels_full)

        # Subsample if too large (t-SNE is slow)
        max_samples = 2000
        if len(X_all_scaled) > max_samples:
            self.logger.info(
                f"  Subsampling from {len(X_all_scaled)} to {max_samples} samples for t-SNE"
            )
            indices = np.random.choice(len(X_all_scaled), max_samples, replace=False)
            X_sampled = X_all_scaled[indices]
            y_sampled = y_all[indices]
            dataset_labels_sampled = dataset_labels_full[indices]
        else:
            X_sampled = X_all_scaled
            y_sampled = y_all
            dataset_labels_sampled = dataset_labels_full

        # Fit t-SNE on sampled data
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sampled)

        # Create 2 plots: one with all data combined, one with separate panels
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        # Use different colors for each class (dynamic based on actual classes)
        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        class_colors = {
            cls: default_colors[i % len(default_colors)]
            for i, cls in enumerate(classes)
        }

        # Dataset markers (dynamic based on number of datasets)
        marker_styles = ["o", "s", "^", "v", "D", "p", "h"]
        all_dataset_names = training_dataset_names + [validation_dataset_name]
        dataset_markers = {
            name: marker_styles[i % len(marker_styles)]
            for i, name in enumerate(all_dataset_names)
        }

        # Plot 0: All datasets combined - only show class labels
        dataset_names = training_dataset_names + [validation_dataset_name]

        # Plot by class only - ignore dataset origin
        for cls in classes:
            class_mask = y_sampled == cls

            axes[0].scatter(
                X_tsne[class_mask, 0],
                X_tsne[class_mask, 1],
                label=cls,
                alpha=0.6,
                s=40,
                c=[class_colors[cls]],
                edgecolors="black",
                linewidths=0.5,
            )

        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")
        axes[0].set_title("All Datasets Combined")
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)

        # Plot 1-3: Each dataset separately
        for idx, dataset_name in enumerate(dataset_names, start=1):
            dataset_mask = dataset_labels_sampled == dataset_name
            X_dataset_tsne = X_tsne[dataset_mask]
            y_dataset = y_sampled[dataset_mask]

            for cls in classes:
                mask = y_dataset == cls
                axes[idx].scatter(
                    X_dataset_tsne[mask, 0],
                    X_dataset_tsne[mask, 1],
                    label=cls,
                    alpha=0.6,
                    s=30,
                    c=[class_colors[cls]],
                )

            axes[idx].set_xlabel("t-SNE 1")
            axes[idx].set_ylabel("t-SNE 2")

            # Label training datasets vs validation
            if dataset_name in training_dataset_names:
                axes[idx].set_title(
                    f"{dataset_name} (Training)\n(n={np.sum(dataset_mask)})"
                )
            else:
                axes[idx].set_title(
                    f"{dataset_name} (Validation - Holdout)\n(n={np.sum(dataset_mask)})"
                )

            axes[idx].legend()
            axes[idx].grid(alpha=0.3)

        plt.suptitle(
            "t-SNE Analysis: Dataset-Level Clustering (Combined t-SNE Fit)",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        self.save_plot(fig, "tsne_analysis.png")
        plt.close()

    def _create_top3_confusion_matrices(
        self, results: Dict[str, Any], y_test, y_validation, classes
    ):
        """Create confusion matrices for top 3 performing models."""
        self.logger.info("📊 Creating confusion matrices for top 3 models...")

        # Rank models by validation accuracy
        model_rankings = sorted(
            results["models"].items(),
            key=lambda x: x[1]["validation_accuracy"],
            reverse=True,
        )

        # Get top 3
        top3_models = model_rankings[:3]

        # Create figure with 3 rows x 2 columns (test & validation for each model)
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))

        for i, (model_name, model_results) in enumerate(top3_models):
            # Test confusion matrix
            cm_test = np.array(model_results["test_confusion_matrix"])
            sns.heatmap(
                cm_test,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[i, 0],
                xticklabels=classes,
                yticklabels=classes,
                cbar=True,
            )
            axes[i, 0].set_title(
                f"#{i+1}: {model_name}\nTest Set (Acc: {model_results['test_accuracy']:.3f})"
            )
            axes[i, 0].set_ylabel("True Label")
            axes[i, 0].set_xlabel("Predicted Label")

            # Validation confusion matrix
            cm_val = np.array(model_results["validation_confusion_matrix"])
            sns.heatmap(
                cm_val,
                annot=True,
                fmt="d",
                cmap="Oranges",
                ax=axes[i, 1],
                xticklabels=classes,
                yticklabels=classes,
                cbar=True,
            )
            axes[i, 1].set_title(
                f"#{i+1}: {model_name}\nValidation Set (Acc: {model_results['validation_accuracy']:.3f})"
            )
            axes[i, 1].set_ylabel("True Label")
            axes[i, 1].set_xlabel("Predicted Label")

        plt.suptitle(
            "Top 3 Models: Confusion Matrices (Test vs Validation)",
            fontsize=16,
            y=0.995,
        )
        plt.tight_layout()
        self.save_plot(fig, "top3_confusion_matrices.png")
        plt.close()

        plt.tight_layout()
        self.save_plot(fig, "generalization_analysis.png")
        plt.close()

    def _run_single_mode(
        self,
        mode: Optional[str],
        training_dataset_names: List[str],
        validation_dataset_name: str,
        batch_results: Dict[str, Any],
        train_split: float,
        random_seed: int,
        stratify: bool,
    ) -> Dict[str, Any]:
        """
        Run training and validation for a single feature extraction mode.

        Args:
            mode: Feature extraction mode (None for single mode backward compatibility)
            training_dataset_names: List of training dataset names
            validation_dataset_name: Validation dataset name
            batch_results: Processed batch data
            train_split: Train/test split ratio
            random_seed: Random seed
            stratify: Whether to stratify splits

        Returns:
            Results dictionary for this mode
        """
        # Determine dataset key format
        if mode is not None:
            # Multi-mode: keys are "dataset_mode"
            training_keys = [f"{name}_{mode}" for name in training_dataset_names]
            validation_key = f"{validation_dataset_name}_{mode}"
        else:
            # Single mode: keys are just dataset names
            training_keys = training_dataset_names
            validation_key = validation_dataset_name

        self.logger.info(f"📚 Training Datasets: {training_keys}")
        self.logger.info(f"🎯 Validation Dataset: {validation_key}")
        self.logger.info(
            f"📊 Train/Test Split: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%"
        )
        self.logger.info(f"🔀 Random Seed: {random_seed}")
        self.logger.info(f"⚖️  Stratified Split: {stratify}")

        # Combine training datasets
        X_train_combined = []
        y_train_combined = []

        for dataset_key in training_keys:
            if dataset_key not in batch_results:
                raise ValueError(
                    f"Training dataset '{dataset_key}' not found in processed data"
                )

            batch_data = batch_results[dataset_key]
            X = batch_data["features"]
            y = batch_data["labels"]

            self.logger.info(f"  {dataset_key}: {len(X)} samples")
            X_train_combined.append(X)
            y_train_combined.append(y)

        # Concatenate all training data
        X_train_combined = np.vstack(X_train_combined)
        y_train_combined = np.concatenate(y_train_combined)

        self.logger.info(f"✅ Combined Training Data: {len(X_train_combined)} samples")

        # Continue with the rest of the training logic...
        # Get validation data
        if validation_key not in batch_results:
            raise ValueError(
                f"Validation dataset '{validation_key}' not found in processed data"
            )

        validation_data = batch_results[validation_key]
        X_validation = validation_data["features"]
        y_validation = validation_data["labels"]

        self.logger.info(f"✅ Validation Data: {len(X_validation)} samples")

        # Split training data into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_combined,
            y_train_combined,
            test_size=1 - train_split,
            random_state=random_seed,
            stratify=y_train_combined if stratify else None,
        )

        self.logger.info(f"✅ Training split: {len(X_train)} train, {len(X_test)} test")

        # Get feature names from first batch (consistent across all batches)
        first_batch_key = training_keys[0]
        feature_names = self._get_feature_names(batch_results[first_batch_key])

        # Get unique classes
        classes = sorted(list(set(y_train)))
        self.logger.info(f"🏷️  Classes: {classes}")

        # Run model training and evaluation
        results = self._run_model_training(
            X_train,
            X_test,
            y_train,
            y_test,
            X_validation,
            y_validation,
            classes,
            feature_names,
        )

        return results

    def _generate_multi_mode_summary(
        self, all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary comparing performance across all modes.

        Args:
            all_results: Results for each mode

        Returns:
            Summary dictionary
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 MULTI-MODE COMPARISON SUMMARY")
        self.logger.info("=" * 80)

        summary = {"modes_compared": list(all_results.keys()), "mode_performance": {}}

        # Extract key metrics for each mode
        for mode, results in all_results.items():
            if mode.startswith("_"):
                continue  # Skip summary keys

            # Get top model performance
            if "models" in results and results["models"]:
                top_model = max(
                    results["models"].values(),
                    key=lambda x: x.get("validation_accuracy", 0),
                )

                summary["mode_performance"][mode] = {
                    "validation_accuracy": top_model.get("validation_accuracy", 0),
                    "test_accuracy": top_model.get("test_accuracy", 0),
                    "top_model": top_model.get("model_name", "unknown"),
                    "num_features": results.get("num_features", 0),
                    "num_samples": results.get("num_samples", 0),
                }

                self.logger.info(f"🎯 {mode}:")
                self.logger.info(
                    f"   Validation: {top_model.get('validation_accuracy', 0):.3f}"
                )
                self.logger.info(f"   Test: {top_model.get('test_accuracy', 0):.3f}")
                self.logger.info(f"   Features: {results.get('num_features', 0)}")
                self.logger.info(
                    f"   Top Model: {top_model.get('model_name', 'unknown')}"
                )
            else:
                self.logger.warning(f"⚠️  No model results found for mode {mode}")

        # Rank modes by validation accuracy
        if summary["mode_performance"]:
            ranked_modes = sorted(
                summary["mode_performance"].items(),
                key=lambda x: x[1]["validation_accuracy"],
                reverse=True,
            )

            summary["ranking"] = ranked_modes
            summary["best_mode"] = ranked_modes[0][0]
            summary["best_validation_accuracy"] = ranked_modes[0][1][
                "validation_accuracy"
            ]

            self.logger.info(
                f"\n🏆 BEST MODE: {summary['best_mode']} "
                f"({summary['best_validation_accuracy']:.3f} validation accuracy)"
            )

        return summary
