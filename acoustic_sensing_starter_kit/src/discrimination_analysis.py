"""
Statistical Discrimination Analysis for Acoustic Geometric Sensing
================================================================

This module provides comprehensive statistical analysis tools to quantify
geometric discrimination capability in acoustic sensing data. It includes:

1. Separability metrics and statistical tests
2. Classification performance analysis
3. Feature importance and selection
4. Cross-validation and robustness testing
5. Signal-to-noise analysis
6. Publication-ready statistical reports

Author: Enhanced for geometric discrimination analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class GeometricDiscriminationAnalyzer:
    """
    Comprehensive analyzer for geometric discrimination capability.

    Provides statistical tests, classification analysis, and feature
    importance analysis to prove geometric discrimination in acoustic sensing.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the analyzer.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def analyze_class_separability(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Comprehensive analysis of class separability using multiple metrics.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Class labels
            feature_names: Feature names for reporting

        Returns:
            Dictionary with separability analysis results
        """
        # Encode string labels to numeric if needed
        y_encoded = self.label_encoder.fit_transform(y)
        unique_labels = self.label_encoder.classes_

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        results = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": len(unique_labels),
            "class_names": unique_labels.tolist(),
            "class_distribution": {
                str(label): int(np.sum(y == label)) for label in unique_labels
            },
        }

        # 1. Statistical significance tests (ANOVA for each feature)
        feature_anova = []
        for i in range(X.shape[1]):
            groups = [X_scaled[y_encoded == j, i] for j in range(len(unique_labels))]
            if all(len(group) > 0 for group in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                feature_name = feature_names[i] if feature_names else f"Feature_{i}"
                feature_anova.append(
                    {
                        "feature": feature_name,
                        "f_statistic": f_stat,
                        "p_value": p_val,
                        "significant": p_val < 0.05,
                    }
                )

        results["feature_anova"] = feature_anova
        results["significant_features"] = sum(
            1 for f in feature_anova if f["significant"]
        )
        results["significant_feature_ratio"] = results["significant_features"] / len(
            feature_anova
        )

        # 2. Linear Discriminant Analysis
        try:
            lda = LinearDiscriminantAnalysis()
            X_lda = lda.fit_transform(X_scaled, y_encoded)

            # LDA separability metrics
            eigenvalues = lda.explained_variance_ratio_
            results["lda"] = {
                "explained_variance_ratio": eigenvalues.tolist(),
                "total_explained_variance": np.sum(eigenvalues),
                "n_discriminant_functions": len(eigenvalues),
                "discriminant_scores": X_lda,
            }
        except Exception as e:
            warnings.warn(f"LDA analysis failed: {e}")
            results["lda"] = None

        # 3. Inter-class vs Intra-class distances
        class_distances = self._analyze_class_distances(
            X_scaled, y_encoded, unique_labels
        )
        results.update(class_distances)

        # 4. Feature importance via Random Forest
        rf_importance = self._analyze_feature_importance(
            X_scaled, y_encoded, feature_names
        )
        results["feature_importance"] = rf_importance

        # 5. Multivariate normality tests (per class)
        normality_results = self._test_multivariate_normality(
            X_scaled, y_encoded, unique_labels
        )
        results["normality_tests"] = normality_results

        self.results["separability"] = results
        return results

    def _analyze_class_distances(
        self, X: np.ndarray, y: np.ndarray, class_names: np.ndarray
    ) -> Dict:
        """Analyze inter-class and intra-class distances."""
        n_classes = len(class_names)

        # Calculate class centroids
        centroids = []
        for i in range(n_classes):
            class_mask = y == i
            if np.any(class_mask):
                centroid = np.mean(X[class_mask], axis=0)
                centroids.append(centroid)
            else:
                centroids.append(np.zeros(X.shape[1]))

        centroids = np.array(centroids)

        # Inter-class distances (between centroids)
        inter_distances = cdist(centroids, centroids)

        # Intra-class distances (within each class)
        intra_distances = {}
        intra_variances = {}

        for i, class_name in enumerate(class_names):
            class_mask = y == i
            class_data = X[class_mask]

            if len(class_data) > 1:
                # Average distance from centroid
                centroid = centroids[i]
                distances = np.linalg.norm(class_data - centroid, axis=1)
                intra_distances[class_name] = {
                    "mean_distance_to_centroid": np.mean(distances),
                    "std_distance_to_centroid": np.std(distances),
                    "max_distance_to_centroid": np.max(distances),
                }

                # Class variance (trace of covariance matrix)
                intra_variances[class_name] = np.trace(np.cov(class_data.T))
            else:
                intra_distances[class_name] = {
                    "mean_distance_to_centroid": 0,
                    "std_distance_to_centroid": 0,
                    "max_distance_to_centroid": 0,
                }
                intra_variances[class_name] = 0

        # Fisher's discriminant ratio (simplified)
        mean_intra_variance = np.mean(list(intra_variances.values()))
        mean_inter_distance = np.mean(inter_distances[inter_distances > 0])
        fisher_ratio = (
            mean_inter_distance / mean_intra_variance
            if mean_intra_variance > 0
            else np.inf
        )

        return {
            "inter_class_distances": inter_distances.tolist(),
            "mean_inter_class_distance": mean_inter_distance,
            "intra_class_distances": intra_distances,
            "intra_class_variances": intra_variances,
            "fisher_discriminant_ratio": fisher_ratio,
            "class_centroids": centroids.tolist(),
        }

    def _analyze_feature_importance(
        self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]]
    ) -> Dict:
        """Analyze feature importance using Random Forest."""
        try:
            rf = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            )
            rf.fit(X, y)

            importances = rf.feature_importances_

            if feature_names:
                importance_df = pd.DataFrame(
                    {"feature": feature_names, "importance": importances}
                ).sort_values("importance", ascending=False)

                return {
                    "importances": importances.tolist(),
                    "feature_ranking": importance_df.to_dict("records"),
                    "top_10_features": importance_df.head(10)["feature"].tolist(),
                }
            else:
                return {
                    "importances": importances.tolist(),
                    "top_10_indices": np.argsort(importances)[-10:].tolist(),
                }

        except Exception as e:
            warnings.warn(f"Feature importance analysis failed: {e}")
            return {"importances": [], "error": str(e)}

    def _test_multivariate_normality(
        self, X: np.ndarray, y: np.ndarray, class_names: np.ndarray
    ) -> Dict:
        """Test multivariate normality for each class."""
        normality_results = {}

        for i, class_name in enumerate(class_names):
            class_mask = y == i
            class_data = X[class_mask]

            if len(class_data) < 3:  # Need at least 3 samples
                normality_results[class_name] = {
                    "sample_size": len(class_data),
                    "test": "insufficient_samples",
                }
                continue

            # Shapiro-Wilk test for each dimension (if reasonable number of features)
            if X.shape[1] <= 10 and len(class_data) <= 5000:
                try:
                    shapiro_results = []
                    for j in range(X.shape[1]):
                        stat, p_val = stats.shapiro(class_data[:, j])
                        shapiro_results.append(
                            {
                                "dimension": j,
                                "statistic": stat,
                                "p_value": p_val,
                                "is_normal": p_val > 0.05,
                            }
                        )

                    normality_results[class_name] = {
                        "sample_size": len(class_data),
                        "shapiro_wilk": shapiro_results,
                        "fraction_normal_dimensions": np.mean(
                            [r["is_normal"] for r in shapiro_results]
                        ),
                    }

                except Exception as e:
                    normality_results[class_name] = {
                        "sample_size": len(class_data),
                        "test": "shapiro_failed",
                        "error": str(e),
                    }
            else:
                # For high-dimensional data, just record basic statistics
                normality_results[class_name] = {
                    "sample_size": len(class_data),
                    "test": "high_dimensional_skip",
                    "mean_skewness": np.mean(
                        [
                            stats.skew(class_data[:, j])
                            for j in range(min(5, X.shape[1]))
                        ]
                    ),
                    "mean_kurtosis": np.mean(
                        [
                            stats.kurtosis(class_data[:, j])
                            for j in range(min(5, X.shape[1]))
                        ]
                    ),
                }

        return normality_results

    def evaluate_classification_performance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_multiple_classifiers: bool = True,
        cv_folds: int = 5,
    ) -> Dict:
        """
        Evaluate classification performance using multiple algorithms.

        Args:
            X: Feature matrix
            y: Class labels
            test_multiple_classifiers: Whether to test multiple classifier types
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with classification performance results
        """
        # Encode string labels to numeric if needed
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        results = {
            "dataset_info": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y_encoded)),
            }
        }

        # Define classifiers to test
        if test_multiple_classifiers:
            classifiers = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                "SVM (RBF)": SVC(kernel="rbf", random_state=self.random_state),
                "SVM (Linear)": SVC(kernel="linear", random_state=self.random_state),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                "Logistic Regression": LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
            }
        else:
            classifiers = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                )
            }

        # Cross-validation strategy
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        classifier_results = {}

        for name, clf in classifiers.items():
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(
                    clf, X_scaled, y_encoded, cv=cv, scoring="accuracy"
                )
                f1_scores = cross_val_score(
                    clf, X_scaled, y_encoded, cv=cv, scoring="f1_macro"
                )

                # Fit on full dataset for additional metrics
                clf.fit(X_scaled, y_encoded)
                y_pred = clf.predict(X_scaled)

                # Detailed metrics
                accuracy = accuracy_score(y_encoded, y_pred)
                f1 = f1_score(y_encoded, y_pred, average="macro")
                precision, recall, f1_per_class, _ = precision_recall_fscore_support(
                    y_encoded, y_pred, average=None
                )

                classifier_results[name] = {
                    "cv_accuracy_mean": np.mean(cv_scores),
                    "cv_accuracy_std": np.std(cv_scores),
                    "cv_f1_mean": np.mean(f1_scores),
                    "cv_f1_std": np.std(f1_scores),
                    "train_accuracy": accuracy,
                    "train_f1_macro": f1,
                    "per_class_precision": precision.tolist(),
                    "per_class_recall": recall.tolist(),
                    "per_class_f1": f1_per_class.tolist(),
                    "confusion_matrix": confusion_matrix(y_encoded, y_pred).tolist(),
                }

                # Feature importance if available
                if hasattr(clf, "feature_importances_"):
                    classifier_results[name][
                        "feature_importances"
                    ] = clf.feature_importances_.tolist()
                elif hasattr(clf, "coef_"):
                    classifier_results[name]["feature_coefficients"] = (
                        np.abs(clf.coef_).mean(axis=0).tolist()
                    )

            except Exception as e:
                warnings.warn(f"Classification with {name} failed: {e}")
                classifier_results[name] = {"error": str(e)}

        results["classifiers"] = classifier_results

        # Identify best performing classifier
        valid_classifiers = {
            name: res for name, res in classifier_results.items() if "error" not in res
        }

        if valid_classifiers:
            best_classifier = max(
                valid_classifiers.keys(),
                key=lambda x: valid_classifiers[x]["cv_accuracy_mean"],
            )
            results["best_classifier"] = {
                "name": best_classifier,
                "cv_accuracy": valid_classifiers[best_classifier]["cv_accuracy_mean"],
                "cv_f1": valid_classifiers[best_classifier]["cv_f1_mean"],
            }

        self.results["classification"] = results
        return results

    def analyze_feature_stability(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 100,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Analyze feature importance stability across bootstrap samples.

        Args:
            X: Feature matrix
            y: Class labels
            n_bootstrap: Number of bootstrap iterations
            feature_names: Feature names for reporting

        Returns:
            Dictionary with feature stability analysis
        """
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        # Bootstrap feature importance analysis
        importance_distributions = []

        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_scaled[bootstrap_indices]
            y_boot = y_encoded[bootstrap_indices]

            # Ensure all classes are represented
            unique_classes = np.unique(y_boot)
            if len(unique_classes) < len(np.unique(y_encoded)):
                continue  # Skip this bootstrap sample

            try:
                rf = RandomForestClassifier(
                    n_estimators=50, random_state=self.random_state + i
                )
                rf.fit(X_boot, y_boot)
                importance_distributions.append(rf.feature_importances_)
            except:
                continue

        if not importance_distributions:
            return {"error": "No valid bootstrap samples obtained"}

        importance_distributions = np.array(importance_distributions)

        # Calculate stability metrics
        mean_importance = np.mean(importance_distributions, axis=0)
        std_importance = np.std(importance_distributions, axis=0)
        cv_importance = std_importance / (
            mean_importance + 1e-8
        )  # Coefficient of variation

        # Rank stability (how often each feature appears in top-k)
        top_k = min(10, X.shape[1])
        top_k_counts = np.zeros(X.shape[1])

        for importance_vec in importance_distributions:
            top_indices = np.argsort(importance_vec)[-top_k:]
            top_k_counts[top_indices] += 1

        top_k_stability = top_k_counts / len(importance_distributions)

        results = {
            "n_bootstrap_samples": len(importance_distributions),
            "mean_feature_importance": mean_importance.tolist(),
            "std_feature_importance": std_importance.tolist(),
            "cv_feature_importance": cv_importance.tolist(),
            "top_k_stability": top_k_stability.tolist(),
            "stable_features": np.where(cv_importance < 0.3)[
                0
            ].tolist(),  # Low CV = stable
            "unstable_features": np.where(cv_importance > 1.0)[
                0
            ].tolist(),  # High CV = unstable
        }

        if feature_names:
            # Create feature stability report
            stability_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_importance": mean_importance,
                    "std_importance": std_importance,
                    "cv_importance": cv_importance,
                    "top_k_stability": top_k_stability,
                }
            ).sort_values("mean_importance", ascending=False)

            results["feature_stability_report"] = stability_df.to_dict("records")
            results["most_stable_features"] = stability_df[
                stability_df["cv_importance"] < 0.3
            ]["feature"].tolist()

        self.results["feature_stability"] = results
        return results

    def generate_discrimination_report(self, save_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive text report of discrimination analysis.

        Args:
            save_path: Path to save the report (optional)

        Returns:
            Report as string
        """
        if not self.results:
            return "No analysis results available. Run analysis methods first."

        report_lines = [
            "=" * 80,
            "ACOUSTIC GEOMETRIC DISCRIMINATION ANALYSIS REPORT",
            "=" * 80,
            "",
        ]

        # Separability analysis
        if "separability" in self.results:
            sep = self.results["separability"]
            report_lines.extend(
                [
                    "SEPARABILITY ANALYSIS",
                    "-" * 40,
                    f"Dataset: {sep['n_samples']} samples, {sep['n_features']} features, {sep['n_classes']} classes",
                    f"Classes: {', '.join(sep['class_names'])}",
                    "",
                    "Class Distribution:",
                ]
            )

            for class_name, count in sep["class_distribution"].items():
                percentage = count / sep["n_samples"] * 100
                report_lines.append(
                    f"  {class_name}: {count} samples ({percentage:.1f}%)"
                )

            report_lines.extend(
                [
                    "",
                    f"Statistically significant features: {sep['significant_features']}/{len(sep['feature_anova'])} ({sep['significant_feature_ratio']*100:.1f}%)",
                    f"Fisher discriminant ratio: {sep['fisher_discriminant_ratio']:.3f}",
                    "",
                ]
            )

            if sep["lda"]:
                lda = sep["lda"]
                report_lines.extend(
                    [
                        f"LDA explained variance: {lda['total_explained_variance']:.3f}",
                        f"Number of discriminant functions: {lda['n_discriminant_functions']}",
                        "",
                    ]
                )

        # Classification performance
        if "classification" in self.results:
            cls = self.results["classification"]
            report_lines.extend(
                [
                    "CLASSIFICATION PERFORMANCE",
                    "-" * 40,
                ]
            )

            if "best_classifier" in cls:
                best = cls["best_classifier"]
                report_lines.extend(
                    [
                        f"Best classifier: {best['name']}",
                        f"Cross-validation accuracy: {best['cv_accuracy']:.3f}",
                        f"Cross-validation F1-score: {best['cv_f1']:.3f}",
                        "",
                    ]
                )

            report_lines.append("All Classifiers:")
            for name, results in cls["classifiers"].items():
                if "error" not in results:
                    report_lines.append(
                        f"  {name}: Accuracy={results['cv_accuracy_mean']:.3f}±{results['cv_accuracy_std']:.3f}, F1={results['cv_f1_mean']:.3f}±{results['cv_f1_std']:.3f}"
                    )
                else:
                    report_lines.append(f"  {name}: Failed ({results['error']})")

            report_lines.append("")

        # Feature stability
        if "feature_stability" in self.results:
            stab = self.results["feature_stability"]
            report_lines.extend(
                [
                    "FEATURE STABILITY ANALYSIS",
                    "-" * 40,
                    f"Bootstrap samples: {stab['n_bootstrap_samples']}",
                    f"Stable features (CV < 0.3): {len(stab['stable_features'])}",
                    f"Unstable features (CV > 1.0): {len(stab['unstable_features'])}",
                    "",
                ]
            )

        # Conclusions
        report_lines.extend(
            [
                "CONCLUSIONS",
                "-" * 40,
            ]
        )

        # Determine if geometric discrimination is proven
        discrimination_proven = False
        evidence = []

        if "separability" in self.results:
            sep = self.results["separability"]
            if sep["significant_feature_ratio"] > 0.2:
                evidence.append(
                    f"Strong statistical evidence: {sep['significant_feature_ratio']*100:.1f}% of features show significant class differences"
                )
                discrimination_proven = True

            if sep["fisher_discriminant_ratio"] > 2.0:
                evidence.append(
                    f"Good class separability: Fisher ratio = {sep['fisher_discriminant_ratio']:.3f}"
                )
                discrimination_proven = True

        if (
            "classification" in self.results
            and "best_classifier" in self.results["classification"]
        ):
            best = self.results["classification"]["best_classifier"]
            if best["cv_accuracy"] > 0.8:
                evidence.append(
                    f"High classification accuracy: {best['cv_accuracy']:.3f}"
                )
                discrimination_proven = True
            elif best["cv_accuracy"] > 0.6:
                evidence.append(
                    f"Moderate classification accuracy: {best['cv_accuracy']:.3f}"
                )

        if discrimination_proven:
            report_lines.extend(
                ["✓ GEOMETRIC DISCRIMINATION CAPABILITY CONFIRMED", "", "Evidence:"]
            )
            for e in evidence:
                report_lines.append(f"  • {e}")
        else:
            report_lines.extend(
                [
                    "⚠ GEOMETRIC DISCRIMINATION CAPABILITY UNCERTAIN",
                    "",
                    "Limited evidence for reliable geometric discrimination.",
                    "Consider: more data, feature engineering, or different sensor setup.",
                ]
            )

        report_lines.extend(["", "=" * 80])

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)

        return report


def quick_discrimination_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Quick comprehensive discrimination analysis.

    Args:
        X: Feature matrix
        y: Class labels
        feature_names: Feature names
        verbose: Print progress

    Returns:
        Complete analysis results
    """
    analyzer = GeometricDiscriminationAnalyzer()

    if verbose:
        print("Running separability analysis...")
    sep_results = analyzer.analyze_class_separability(X, y, feature_names)

    if verbose:
        print("Running classification analysis...")
    cls_results = analyzer.evaluate_classification_performance(X, y)

    if verbose:
        print("Running feature stability analysis...")
    stab_results = analyzer.analyze_feature_stability(
        X, y, n_bootstrap=50, feature_names=feature_names
    )

    if verbose:
        print("Generating report...")
    report = analyzer.generate_discrimination_report()

    if verbose:
        print("\n" + report)

    return {
        "separability": sep_results,
        "classification": cls_results,
        "feature_stability": stab_results,
        "report": report,
    }
