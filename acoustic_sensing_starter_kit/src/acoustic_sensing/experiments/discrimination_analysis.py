from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import os


class DiscriminationAnalysisExperiment(BaseExperiment):
    """
    Experiment for material discrimination analysis using multiple classifiers.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform material discrimination analysis with multiple classifiers.

        Args:
            shared_data: Dictionary containing loaded features and labels

        Returns:
            Dictionary containing classification results and model performance
        """
        self.logger.info("Starting discrimination analysis experiment...")

        # Load per-batch data from previous experiment
        batch_results = self.load_shared_data(shared_data, "batch_results")

        # Define classifiers to test
        classifiers = self._get_classifiers()

        # Perform analysis for each batch separately
        all_batch_results = {}

        for batch_name, batch_data in batch_results.items():
            self.logger.info(f"Analyzing batch: {batch_name}")

            X = batch_data["features"]
            y = batch_data["labels"]

            if len(X) == 0:
                self.logger.warning(f"No data available for batch {batch_name}")
                continue

            # Standardize features for this batch
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform cross-validation analysis for this batch
            batch_cv_results = self._perform_cross_validation(
                X_scaled, y, classifiers, batch_name
            )

            # Store results for this batch
            all_batch_results[batch_name] = {
                "cv_results": batch_cv_results,
                "scaler": scaler,
                "num_samples": len(X),
                "num_features": X.shape[1],
                "classes": batch_data["classes"],
                "class_distribution": batch_data["class_distribution"],
            }

            # Save batch-specific results
            self._save_batch_discrimination_results(
                all_batch_results[batch_name], batch_name
            )

            # Create batch-specific plots
            self._create_batch_plots(
                all_batch_results[batch_name], batch_name, batch_results[batch_name]
            )

        # Perform cross-batch analysis (train on one batch, test on another)
        cross_batch_results = self._perform_cross_batch_analysis(
            batch_results, classifiers
        )

        # Find best performing batch and classifier
        best_batch_info = self._find_best_batch_performance(all_batch_results)

        results = {
            "batch_performance_results": all_batch_results,
            "cross_batch_results": cross_batch_results,
            "best_batch_info": best_batch_info,
            "total_batches": len(all_batch_results),
        }

        # Create visualizations
        try:
            self._create_performance_visualizations(
                all_batch_results, cross_batch_results
            )
            self.logger.info("Performance visualizations created successfully")
        except Exception as e:
            self.logger.warning(f"Could not create visualizations: {str(e)}")
            # Continue without visualizations

        # Save summary
        self._save_discrimination_summary(results)

        self.logger.info("Discrimination analysis experiment completed")
        return results

    def _get_classifiers(self) -> dict:
        """Define classifiers to evaluate."""
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM (RBF)": SVC(kernel="rbf", random_state=42),
            "SVM (Linear)": SVC(kernel="linear", random_state=42),
            "K-NN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

        if self.config.get("include_lda", True):
            classifiers["Linear Discriminant Analysis"] = LinearDiscriminantAnalysis()

        return classifiers

    def _save_batch_discrimination_results(self, batch_results: dict, batch_name: str):
        """Save detailed discrimination results for a specific batch."""
        import json
        import os

        # Create batch-specific output directory
        batch_output_dir = os.path.join(self.experiment_output_dir, batch_name)
        os.makedirs(batch_output_dir, exist_ok=True)

        # Create a serializable version of the results
        serializable_results = {
            "batch_name": batch_name,
            "num_samples": batch_results["num_samples"],
            "num_features": batch_results["num_features"],
            "classes": batch_results["classes"],
            "class_distribution": batch_results["class_distribution"],
            "cv_results": batch_results["cv_results"],
        }

        # Save full batch results
        results_path = os.path.join(
            batch_output_dir, f"{batch_name}_discrimination_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Create batch summary with best performing classifiers
        cv_results = batch_results["cv_results"]

        # Filter out failed classifiers (those without mean_accuracy)
        successful_results = {
            name: result
            for name, result in cv_results.items()
            if "mean_accuracy" in result and "error" not in result
        }

        batch_summary = {
            "batch_name": batch_name,
            "num_samples": batch_results["num_samples"],
            "num_classes": len(batch_results["classes"]),
            "classes": batch_results["classes"],
        }

        if successful_results:
            best_classifier = max(
                successful_results.keys(),
                key=lambda k: successful_results[k]["mean_accuracy"],
            )
            worst_classifier = min(
                successful_results.keys(),
                key=lambda k: successful_results[k]["mean_accuracy"],
            )

            batch_summary.update(
                {
                    "best_classifier": {
                        "name": best_classifier,
                        "mean_accuracy": successful_results[best_classifier][
                            "mean_accuracy"
                        ],
                        "std_accuracy": successful_results[best_classifier][
                            "std_accuracy"
                        ],
                    },
                    "worst_classifier": {
                        "name": worst_classifier,
                        "mean_accuracy": successful_results[worst_classifier][
                            "mean_accuracy"
                        ],
                        "std_accuracy": successful_results[worst_classifier][
                            "std_accuracy"
                        ],
                    },
                    "all_classifiers_performance": {
                        name: {
                            "mean_accuracy": result["mean_accuracy"],
                            "std_accuracy": result["std_accuracy"],
                        }
                        for name, result in successful_results.items()
                    },
                }
            )
        else:
            batch_summary.update(
                {
                    "note": "No classifiers succeeded (likely due to single class in batch)",
                    "all_classifiers_performance": cv_results,
                }
            )

        # Save batch summary
        summary_path = os.path.join(
            batch_output_dir, f"{batch_name}_discrimination_summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2, default=str)

        self.logger.info(
            f"Batch {batch_name} discrimination results saved to: {batch_output_dir}"
        )

    def _create_batch_plots(
        self, batch_results: dict, batch_name: str, batch_data: dict
    ):
        """Create visualization plots for a specific batch."""
        try:
            # Create batch-specific output directory
            batch_output_dir = os.path.join(self.experiment_output_dir, batch_name)
            os.makedirs(batch_output_dir, exist_ok=True)

            # Create classifier performance comparison plot
            self._create_batch_performance_plot(
                batch_results, batch_name, batch_output_dir
            )

            # Create confusion matrices for successful classifiers
            X = batch_data["features"]
            y = batch_data["labels"]
            self._create_batch_confusion_matrices(
                batch_results, batch_name, batch_output_dir, X, y
            )

        except Exception as e:
            self.logger.warning(f"Failed to create plots for batch {batch_name}: {e}")

    def _create_batch_performance_plot(
        self, batch_results: dict, batch_name: str, output_dir: str
    ):
        """Create performance comparison plot for a single batch."""
        import matplotlib.pyplot as plt

        cv_results = batch_results["cv_results"]

        # Filter successful classifiers
        successful_results = {
            name: result
            for name, result in cv_results.items()
            if "mean_accuracy" in result and "error" not in result
        }

        if not successful_results:
            self.logger.warning(f"No successful classifiers for batch {batch_name}")
            return

        # Prepare data for plotting
        classifiers = list(successful_results.keys())
        means = [successful_results[clf]["mean_accuracy"] for clf in classifiers]
        stds = [successful_results[clf]["std_accuracy"] for clf in classifiers]

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Bar plot with error bars
        bars = ax.bar(
            range(len(classifiers)),
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )

        # Customize plot
        ax.set_xlabel("Classifiers")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Classifier Performance Comparison - {batch_name}")
        ax.set_xticks(range(len(classifiers)))
        ax.set_xticklabels(classifiers, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                f"{mean:.3f}±{std:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Color-code bars by performance
        for i, (bar, mean) in enumerate(zip(bars, means)):
            if mean == max(means):
                bar.set_color("gold")  # Best performer
            elif mean == min(means):
                bar.set_color("lightcoral")  # Worst performer

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_classifier_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_batch_confusion_matrices(
        self,
        batch_results: dict,
        batch_name: str,
        output_dir: str,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Create confusion matrices for successful classifiers."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import StratifiedKFold
        import seaborn as sns

        cv_results = batch_results["cv_results"]

        # Filter successful classifiers and get top 3
        successful_results = {
            name: result
            for name, result in cv_results.items()
            if "mean_accuracy" in result and "error" not in result
        }

        if not successful_results:
            return

        # Get top 3 classifiers
        top_classifiers = sorted(
            successful_results.keys(),
            key=lambda k: successful_results[k]["mean_accuracy"],
            reverse=True,
        )[:3]

        if len(top_classifiers) == 0:
            return

        # Create confusion matrices
        fig, axes = plt.subplots(
            1, len(top_classifiers), figsize=(6 * len(top_classifiers), 5)
        )
        if len(top_classifiers) == 1:
            axes = [axes]

        # Get classifiers
        classifiers = self._get_classifiers()
        cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=42
        )  # Use fewer folds

        for i, clf_name in enumerate(top_classifiers):
            try:
                clf = classifiers[clf_name]

                # Aggregate confusion matrix across CV folds
                y_true_all = []
                y_pred_all = []

                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                    y_true_all.extend(y_test)
                    y_pred_all.extend(y_pred)

                # Create confusion matrix
                cm = confusion_matrix(y_true_all, y_pred_all, labels=np.unique(y))

                # Normalize confusion matrix
                cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

                # Clean up labels by removing "finger" prefix for better readability
                unique_labels = np.unique(y)
                cleaned_labels = [
                    label.replace("finger_", "").replace("finger", "")
                    for label in unique_labels
                ]

                # Plot
                im = sns.heatmap(
                    cm_norm,
                    annot=True,
                    fmt=".2f",
                    cmap="Blues",
                    xticklabels=cleaned_labels,
                    yticklabels=cleaned_labels,
                    ax=axes[i],
                    cbar=True,
                )

                axes[i].set_title(
                    f'{clf_name}\nAcc: {successful_results[clf_name]["mean_accuracy"]:.3f}'
                )
                axes[i].set_xlabel("Predicted Label")
                axes[i].set_ylabel("True Label")

            except Exception as e:
                self.logger.warning(
                    f"Failed to create confusion matrix for {clf_name}: {e}"
                )
                axes[i].text(
                    0.5,
                    0.5,
                    f"Failed to create\nconfusion matrix\nfor {clf_name}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )

        plt.suptitle(f"Confusion Matrices - {batch_name}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{batch_name}_confusion_matrices.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _perform_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifiers: dict,
        batch_name: str = "unknown",
    ) -> dict:
        """Perform cross-validation for all classifiers."""
        self.logger.info(f"Performing cross-validation analysis for {batch_name}...")

        cv_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, clf in classifiers.items():
            self.logger.info(f"Evaluating {name}...")

            try:
                # Perform cross-validation
                scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

                cv_results[name] = {
                    "scores": scores.tolist(),
                    "mean_accuracy": float(scores.mean()),
                    "std_accuracy": float(scores.std()),
                    "best_score": float(scores.max()),
                    "worst_score": float(scores.min()),
                }

                self.logger.info(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {str(e)}")
                cv_results[name] = {"error": str(e)}

        return cv_results

    def _perform_batch_analysis(
        self, X: np.ndarray, y: np.ndarray, batch_info: dict, classifiers: dict
    ) -> dict:
        """Analyze performance across different batches."""
        self.logger.info("Performing batch-specific analysis...")

        batch_results = {}

        # Test each classifier on each batch
        for name, clf in classifiers.items():
            batch_results[name] = {}

            for batch_name, info in batch_info.items():
                indices = info["indices"]
                X_batch = X[indices]
                y_batch = y[indices]

                if len(np.unique(y_batch)) < 2:
                    # Skip if batch doesn't have multiple classes
                    continue

                try:
                    # Train on all data except this batch, test on this batch
                    train_mask = np.ones(len(X), dtype=bool)
                    train_mask[indices] = False

                    X_train, X_test = X[train_mask], X_batch
                    y_train, y_test = y[train_mask], y_batch

                    # Train and test
                    clf_copy = type(clf)(**clf.get_params())
                    clf_copy.fit(X_train, y_train)
                    predictions = clf_copy.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)

                    batch_results[name][batch_name] = {
                        "accuracy": float(accuracy),
                        "num_samples": len(y_test),
                        "num_classes": len(np.unique(y_test)),
                    }

                except Exception as e:
                    self.logger.error(
                        f"Error in batch analysis for {name} on {batch_name}: {str(e)}"
                    )
                    batch_results[name][batch_name] = {"error": str(e)}

        return batch_results

    def _generate_detailed_reports(
        self, X: np.ndarray, y: np.ndarray, classifiers: dict
    ) -> dict:
        """Generate detailed classification reports."""
        self.logger.info("Generating detailed classification reports...")

        detailed_results = {}

        # Use train-test split for detailed analysis
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        for name, clf in classifiers.items():
            try:
                # Train and predict
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)

                # Generate classification report
                class_report = classification_report(
                    y_test, predictions, output_dict=True
                )

                # Generate confusion matrix
                conf_matrix = confusion_matrix(y_test, predictions)

                # Calculate per-class metrics
                unique_classes = np.unique(y)
                per_class_metrics = {}
                for cls in unique_classes:
                    if cls in class_report:
                        per_class_metrics[cls] = class_report[cls]

                detailed_results[name] = {
                    "accuracy": float(accuracy_score(y_test, predictions)),
                    "classification_report": class_report,
                    "confusion_matrix": conf_matrix.tolist(),
                    "per_class_metrics": per_class_metrics,
                }

                # Create confusion matrix visualization
                self._create_confusion_matrix_plot(conf_matrix, unique_classes, name)

            except Exception as e:
                self.logger.error(f"Error generating report for {name}: {str(e)}")
                detailed_results[name] = {"error": str(e)}

        return detailed_results

    def _perform_lda_analysis(
        self, X: np.ndarray, y: np.ndarray, batch_info: dict
    ) -> dict:
        """Perform Linear Discriminant Analysis."""
        self.logger.info("Performing LDA analysis...")

        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            # Fit LDA
            lda = LinearDiscriminantAnalysis()
            X_lda = lda.fit_transform(X, y)

            # Analyze discriminant loadings
            discriminant_loadings = lda.scalings_

            # Create LDA visualization
            self._create_lda_visualization(X_lda, y, batch_info)

            return {
                "lda_components": X_lda,
                "discriminant_loadings": discriminant_loadings.tolist(),
                "explained_variance_ratio": (
                    lda.explained_variance_ratio_.tolist()
                    if hasattr(lda, "explained_variance_ratio_")
                    else None
                ),
                "n_components": X_lda.shape[1],
            }

        except Exception as e:
            self.logger.error(f"Error in LDA analysis: {str(e)}")
            return {"error": str(e)}

    def _create_performance_visualizations(self, cv_results: dict, batch_results: dict):
        """Create performance visualization plots."""
        # Cross-validation results plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Cross-validation accuracy comparison
        classifier_names = []
        mean_scores = []
        std_scores = []

        for name, results in cv_results.items():
            if "error" not in results:
                classifier_names.append(name)
                mean_scores.append(results["mean_accuracy"])
                std_scores.append(results["std_accuracy"])

        y_pos = np.arange(len(classifier_names))
        axes[0, 0].barh(y_pos, mean_scores, xerr=std_scores, capsize=5)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(classifier_names)
        axes[0, 0].set_xlabel("Accuracy")
        axes[0, 0].set_title("Cross-Validation Performance")

        # 2. Score distribution
        all_scores = []
        all_labels = []
        for name, results in cv_results.items():
            if "error" not in results:
                all_scores.extend(results["scores"])
                all_labels.extend([name] * len(results["scores"]))

        df_scores = pd.DataFrame({"Classifier": all_labels, "Accuracy": all_scores})
        sns.boxplot(data=df_scores, x="Accuracy", y="Classifier", ax=axes[0, 1])
        axes[0, 1].set_title("Accuracy Distribution (Cross-Validation)")

        # 3. Batch performance heatmap (if available)
        if batch_results:
            batch_performance = []
            batch_names = []
            classifier_list = []

            for clf_name, batch_data in batch_results.items():
                for batch_name, results in batch_data.items():
                    if "error" not in results:
                        batch_performance.append(results["accuracy"])
                        batch_names.append(batch_name)
                        classifier_list.append(clf_name)

            if batch_performance:
                df_batch = pd.DataFrame(
                    {
                        "Classifier": classifier_list,
                        "Batch": batch_names,
                        "Accuracy": batch_performance,
                    }
                )

                pivot_batch = df_batch.pivot(
                    index="Classifier", columns="Batch", values="Accuracy"
                )
                sns.heatmap(
                    pivot_batch, annot=True, fmt=".3f", ax=axes[1, 0], cmap="viridis"
                )
                axes[1, 0].set_title("Batch-Specific Performance")

        # 4. Performance ranking
        sorted_results = sorted(
            [
                (name, results["mean_accuracy"])
                for name, results in cv_results.items()
                if "error" not in results
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        names, scores = zip(*sorted_results) if sorted_results else ([], [])
        axes[1, 1].bar(range(len(names)), scores)
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45)
        axes[1, 1].set_ylabel("Mean Accuracy")
        axes[1, 1].set_title("Classifier Ranking")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "discrimination_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_confusion_matrix_plot(
        self, conf_matrix: np.ndarray, class_labels: list, classifier_name: str
    ):
        """Create confusion matrix visualization."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.title(f"Confusion Matrix - {classifier_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.experiment_output_dir,
                f'confusion_matrix_{classifier_name.replace(" ", "_")}.png',
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_lda_visualization(
        self, X_lda: np.ndarray, y: np.ndarray, batch_info: dict
    ):
        """Create LDA visualization."""
        if X_lda.shape[1] >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot by class
            unique_classes = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

            for i, cls in enumerate(unique_classes):
                mask = y == cls
                axes[0].scatter(
                    X_lda[mask, 0], X_lda[mask, 1], c=[colors[i]], label=cls, alpha=0.6
                )
            axes[0].set_xlabel("First Discriminant")
            axes[0].set_ylabel("Second Discriminant")
            axes[0].set_title("LDA Projection - By Material")
            axes[0].legend()

            # Plot by batch
            batch_colors = plt.cm.Set3(np.linspace(0, 1, len(batch_info)))
            for i, (batch_name, info) in enumerate(batch_info.items()):
                indices = info["indices"]
                axes[1].scatter(
                    X_lda[indices, 0],
                    X_lda[indices, 1],
                    c=[batch_colors[i]],
                    label=batch_name,
                    alpha=0.6,
                )
            axes[1].set_xlabel("First Discriminant")
            axes[1].set_ylabel("Second Discriminant")
            axes[1].set_title("LDA Projection - By Batch")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.experiment_output_dir, "lda_projection.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _find_best_classifier(self, cv_results: dict) -> dict:
        """Find the best performing classifier."""
        best_name = None
        best_score = -1

        for name, results in cv_results.items():
            if "error" not in results and results["mean_accuracy"] > best_score:
                best_score = results["mean_accuracy"]
                best_name = name

        return {"name": best_name, "accuracy": best_score} if best_name else {}

    def _save_discrimination_summary(self, results: dict):
        """Save discrimination analysis summary."""

        # Extract overall performance metrics from batch results
        all_batch_performance = {}
        best_overall = {"accuracy": 0, "batch": "", "classifier": ""}

        for batch_name, batch_data in results["batch_performance_results"].items():
            cv_results = batch_data["cv_results"]

            # Find best classifier for this batch
            best_for_batch = self._find_best_classifier(cv_results)
            if best_for_batch.get("accuracy", 0) > best_overall["accuracy"]:
                best_overall.update(
                    {
                        "accuracy": best_for_batch["accuracy"],
                        "batch": batch_name,
                        "classifier": best_for_batch["name"],
                    }
                )

            # Store performance for this batch
            all_batch_performance[batch_name] = {
                "best_classifier": best_for_batch,
                "cv_performance": {
                    name: {
                        "mean_accuracy": res["mean_accuracy"],
                        "std_accuracy": res["std_accuracy"],
                    }
                    for name, res in cv_results.items()
                    if "error" not in res
                },
                "num_classifiers_tested": len(cv_results),
            }

        summary = {
            "best_overall": best_overall,
            "per_batch_performance": all_batch_performance,
            "total_batches": results["total_batches"],
            "cross_batch_results": results.get("cross_batch_results", {}),
        }

        self.save_results(summary, "discrimination_summary.json")

    def _perform_cross_batch_analysis(
        self, batch_results: dict, classifiers: dict
    ) -> dict:
        """Perform cross-batch analysis (train on one batch, test on others)."""
        self.logger.info("Performing cross-batch analysis...")

        cross_batch_results = {}
        batch_names = list(batch_results.keys())

        for train_batch in batch_names:
            for test_batch in batch_names:
                if train_batch == test_batch:
                    continue

                train_data = batch_results[train_batch]
                test_data = batch_results[test_batch]

                X_train = train_data["features"]
                y_train = train_data["labels"]
                X_test = test_data["features"]
                y_test = test_data["labels"]

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                pair_key = f"{train_batch}_to_{test_batch}"
                cross_batch_results[pair_key] = {}

                for clf_name, clf in classifiers.items():
                    try:
                        clf_copy = clone(clf)
                        clf_copy.fit(X_train_scaled, y_train)
                        accuracy = clf_copy.score(X_test_scaled, y_test)
                        cross_batch_results[pair_key][clf_name] = {
                            "accuracy": accuracy,
                            "train_samples": len(X_train),
                            "test_samples": len(X_test),
                        }
                    except Exception as e:
                        cross_batch_results[pair_key][clf_name] = {"error": str(e)}

        return cross_batch_results

    def _find_best_batch_performance(self, batch_results: dict) -> dict:
        """Find the best performing batch and classifier."""
        best_performance = 0
        best_info = {}

        for batch_name, batch_data in batch_results.items():
            for clf_name, clf_result in batch_data["cv_results"].items():
                if "error" not in clf_result:
                    accuracy = clf_result["mean_accuracy"]
                    if accuracy > best_performance:
                        best_performance = accuracy
                        best_info = {
                            "batch_name": batch_name,
                            "classifier": clf_name,
                            "accuracy": accuracy,
                            "std": clf_result["std_accuracy"],
                        }

        return best_info

    def _create_performance_visualizations(
        self, batch_results: dict, cross_batch_results: dict
    ):
        """Create performance visualization plots."""
        # This is a placeholder - visualization implementation would go here
        self.logger.info("Creating performance visualizations...")
