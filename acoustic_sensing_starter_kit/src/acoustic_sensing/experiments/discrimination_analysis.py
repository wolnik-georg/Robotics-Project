from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
from sklearn.base import BaseEstimator, ClassifierMixin

# GPU-accelerated classifiers and normalization wrappers
try:
    from .gpu_classifiers import (
        GPUMLPClassifier,
        NormalizedClassifierWrapper,
        RelativeFeatureClassifier,
        SpectrogramCNNClassifier,
        SpectrogramCNN_MLPClassifier,
        SpectrogramCNN_AdvancedClassifier,
        SpectrogramResNetClassifier,
        get_device,
    )

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """XGBoost wrapper that handles string labels."""

    def __init__(self, n_estimators=100, random_state=42, eval_metric="mlogloss"):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.label_encoder = LabelEncoder()
        self.model = None

    @property
    def _estimator_type(self):
        """Declare this as a classifier for sklearn compatibility."""
        return "classifier"

    def fit(self, X, y):
        # Encode string labels to numeric
        y_encoded = self.label_encoder.fit_transform(y)
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            eval_metric=self.eval_metric,
        )
        self.model.fit(X, y_encoded)
        return self

    def predict(self, X):
        # Predict numeric labels and convert back to original labels
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBMWrapper(BaseEstimator, ClassifierMixin):
    """LightGBM wrapper that handles string labels."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.verbose = verbose
        self.label_encoder = LabelEncoder()
        self.model = None

    def fit(self, X, y):
        # Encode string labels to numeric
        y_encoded = self.label_encoder.fit_transform(y)
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.model.fit(X, y_encoded)
        else:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )
        return self

    def predict(self, X):
        # Predict numeric labels and convert back to original labels
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MLPWrapper(BaseEstimator, ClassifierMixin):
    """MLP wrapper that handles string labels."""

    _estimator_type = "classifier"  # Required for VotingClassifier compatibility

    def __sklearn_tags__(self):
        """Return sklearn tags for this estimator (required for sklearn 1.6+)."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=None,
        verbose=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.verbose = verbose
        self.label_encoder = LabelEncoder()
        self.model = None

    def fit(self, X, y):
        # Encode string labels to numeric
        y_encoded = self.label_encoder.fit_transform(y)
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model.fit(X, y_encoded)
        self.classes_ = self.label_encoder.classes_  # Required for VotingClassifier
        return self

    def predict(self, X):
        # Predict numeric labels and convert back to original labels
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import os


class PCAClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper that applies PCA dimensionality reduction before classification.

    This addresses the curse of dimensionality by:
    1. Reducing 55 features → ~19 PCA components (95% variance)
    2. Removing correlated/redundant features
    3. Improving generalization on validation set
    """

    def __init__(self, base_classifier, n_components=0.95, pca_whiten=False):
        """
        Args:
            base_classifier: The classifier to use after PCA
            n_components: Number of components or variance to retain (default: 0.95 = 95%)
            pca_whiten: Whether to whiten PCA components (normalize variance)
        """
        self.base_classifier = base_classifier
        self.n_components = n_components
        self.pca_whiten = pca_whiten
        self.pipeline = None

    def fit(self, X, y):
        """Fit PCA + classifier pipeline."""
        # Create pipeline: PCA -> Classifier
        self.pipeline = Pipeline(
            [
                ("pca", PCA(n_components=self.n_components, whiten=self.pca_whiten)),
                ("classifier", clone(self.base_classifier)),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Predict using PCA-transformed features."""
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using PCA-transformed features."""
        return self.pipeline.predict_proba(X)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            "base_classifier": self.base_classifier,
            "n_components": self.n_components,
            "pca_whiten": self.pca_whiten,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class DiscriminationAnalysisExperiment(BaseExperiment):
    """
    Experiment for material discrimination analysis using multiple classifiers.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing."""
        return ["data_processing"]

    def apply_confidence_filtering(
        self,
        y_true,
        y_pred,
        probabilities,
        threshold=0.7,
        mode="reject",
        default_class=None,
    ):
        """
        Filter predictions based on confidence threshold.

        Args:
            y_true: True labels (numpy array)
            y_pred: Predicted labels (numpy array)
            probabilities: Prediction probabilities from predict_proba (shape: [n_samples, n_classes])
            threshold: Minimum confidence to accept (0.0-1.0)
            mode: "reject" (exclude low-confidence) or "default" (assign default class)
            default_class: Default class to use if mode="default"

        Returns:
            Tuple of (filtered_y_true, filtered_y_pred, confidence_stats)
        """
        import numpy as np

        # Get maximum probability for each prediction (confidence)
        confidences = np.max(probabilities, axis=1)

        # Identify high-confidence predictions
        high_confidence_mask = confidences >= threshold

        # Statistics
        total_samples = len(y_pred)
        high_conf_count = np.sum(high_confidence_mask)
        low_conf_count = total_samples - high_conf_count

        stats = {
            "total_samples": total_samples,
            "high_confidence": high_conf_count,
            "low_confidence": low_conf_count,
            "high_confidence_pct": (
                100 * high_conf_count / total_samples if total_samples > 0 else 0
            ),
            "low_confidence_pct": (
                100 * low_conf_count / total_samples if total_samples > 0 else 0
            ),
            "mean_confidence": np.mean(confidences),
            "median_confidence": np.median(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
        }

        # Apply filtering based on mode
        if mode == "reject":
            # Reject low-confidence predictions (exclude from evaluation)
            filtered_y_true = y_true[high_confidence_mask]
            filtered_y_pred = y_pred[high_confidence_mask]

            self.logger.info(f"  📊 Confidence Filtering (threshold={threshold}):")
            self.logger.info(
                f"    Kept: {high_conf_count}/{total_samples} ({stats['high_confidence_pct']:.1f}%)"
            )
            self.logger.info(
                f"    Rejected: {low_conf_count}/{total_samples} ({stats['low_confidence_pct']:.1f}%)"
            )
            self.logger.info(f"    Mean confidence: {stats['mean_confidence']:.3f}")
            self.logger.info(f"    Median confidence: {stats['median_confidence']:.3f}")

        elif mode == "default":
            # Assign default class to low-confidence predictions
            filtered_y_true = y_true.copy()
            filtered_y_pred = y_pred.copy()
            filtered_y_pred[~high_confidence_mask] = default_class

            self.logger.info(f"  📊 Confidence Filtering (threshold={threshold}):")
            self.logger.info(
                f"    High confidence: {high_conf_count}/{total_samples} ({stats['high_confidence_pct']:.1f}%)"
            )
            self.logger.info(
                f"    Defaulted to '{default_class}': {low_conf_count}/{total_samples} ({stats['low_confidence_pct']:.1f}%)"
            )
            self.logger.info(f"    Mean confidence: {stats['mean_confidence']:.3f}")
            self.logger.info(f"    Median confidence: {stats['median_confidence']:.3f}")

        else:
            # No filtering
            filtered_y_true = y_true
            filtered_y_pred = y_pred

        return filtered_y_true, filtered_y_pred, stats

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

        # Check if validation datasets are specified
        validation_datasets = shared_data.get("validation_datasets", [])
        tuning_datasets = shared_data.get("hyperparameter_tuning_datasets", [])
        final_test_datasets = shared_data.get("final_test_datasets", [])
        training_datasets = shared_data.get(
            "training_datasets", list(batch_results.keys())
        )

        # Determine split mode: 3-way, 2-way, or standard
        if tuning_datasets and final_test_datasets:
            # 3-WAY SPLIT MODE
            self.logger.info("=" * 80)
            self.logger.info("🎯 3-WAY SPLIT MODE ENABLED")
            self.logger.info("=" * 80)
            self.logger.info(f"📋 DATASET CONFIGURATION:")
            self.logger.info(f"  1️⃣  TRAINING datasets ({len(training_datasets)}):")
            for ds in training_datasets:
                self.logger.info(f"     - {ds}")
            self.logger.info(f"  2️⃣  TUNING datasets ({len(tuning_datasets)}):")
            for ds in tuning_datasets:
                self.logger.info(f"     - {ds}")
            self.logger.info(f"  3️⃣  TEST datasets ({len(final_test_datasets)}):")
            for ds in final_test_datasets:
                self.logger.info(f"     - {ds}")
            self.logger.info("=" * 80)

            # Combine training datasets
            X_train_list = []
            y_train_list = []
            for dataset_name in training_datasets:
                if dataset_name in batch_results:
                    X_train_list.append(batch_results[dataset_name]["features"])
                    y_train_list.append(batch_results[dataset_name]["labels"])

            X_train_combined = np.vstack(X_train_list) if X_train_list else np.array([])
            y_train_combined = (
                np.concatenate(y_train_list) if y_train_list else np.array([])
            )

            # Combine tuning datasets
            X_tuning_list = []
            y_tuning_list = []
            for dataset_name in tuning_datasets:
                if dataset_name in batch_results:
                    X_tuning_list.append(batch_results[dataset_name]["features"])
                    y_tuning_list.append(batch_results[dataset_name]["labels"])

            X_tuning_combined = (
                np.vstack(X_tuning_list) if X_tuning_list else np.array([])
            )
            y_tuning_combined = (
                np.concatenate(y_tuning_list) if y_tuning_list else np.array([])
            )

            # Combine final test datasets
            X_test_list = []
            y_test_list = []
            for dataset_name in final_test_datasets:
                if dataset_name in batch_results:
                    X_test_list.append(batch_results[dataset_name]["features"])
                    y_test_list.append(batch_results[dataset_name]["labels"])

            X_test_combined = np.vstack(X_test_list) if X_test_list else np.array([])
            y_test_combined = (
                np.concatenate(y_test_list) if y_test_list else np.array([])
            )

            self.logger.info(
                f"✓ Combined training data: {len(X_train_combined)} samples from {len(X_train_list)} datasets"
            )
            self.logger.info(
                f"✓ Combined tuning data: {len(X_tuning_combined)} samples from {len(X_tuning_list)} datasets"
            )
            self.logger.info(
                f"✓ Combined final test data: {len(X_test_combined)} samples from {len(X_test_list)} datasets"
            )
            self.logger.info("=" * 80)

            # Verify no dataset overlap
            train_set = set(training_datasets)
            tune_set = set(tuning_datasets)
            test_set = set(final_test_datasets)
            if train_set & tune_set:
                self.logger.warning(
                    f"⚠️  WARNING: Overlap between training and tuning: {train_set & tune_set}"
                )
            if train_set & test_set:
                self.logger.warning(
                    f"⚠️  WARNING: Overlap between training and test: {train_set & test_set}"
                )
            if tune_set & test_set:
                self.logger.warning(
                    f"⚠️  WARNING: Overlap between tuning and test: {tune_set & test_set}"
                )

            if (
                not (train_set & tune_set)
                and not (train_set & test_set)
                and not (tune_set & test_set)
            ):
                self.logger.info(
                    "✅ VERIFIED: No dataset overlap - proper 3-way split maintained"
                )

            # Run 3-way split analysis
            results = self._run_with_3way_split(
                X_train_combined,
                y_train_combined,
                X_tuning_combined,
                y_tuning_combined,
                X_test_combined,
                y_test_combined,
                training_datasets,
                tuning_datasets,
                final_test_datasets,
                batch_results,
            )

            return results

        elif validation_datasets:
            self.logger.info(f"✓ Validation mode enabled")
            self.logger.info(f"  Training datasets: {training_datasets}")
            self.logger.info(f"  Validation datasets: {validation_datasets}")

            # Domain Adaptation: Mix hold-out data into training (optional)
            domain_adaptation_config = self.config.get("domain_adaptation", {})
            use_domain_adaptation = domain_adaptation_config.get("enabled", False)
            holdout_train_split = domain_adaptation_config.get(
                "holdout_train_split", 0.3
            )

            if use_domain_adaptation and validation_datasets:
                self.logger.info("=" * 80)
                self.logger.info("🔄 DOMAIN ADAPTATION ENABLED")
                self.logger.info("=" * 80)
                self.logger.info(
                    f"Strategy: Mix {holdout_train_split*100:.0f}% of hold-out data into training"
                )
                self.logger.info(
                    f"Purpose: Test if models memorized surface patterns vs learned general contact"
                )
                self.logger.info("=" * 80)

            # Combine training datasets
            X_train_list = []
            y_train_list = []
            for dataset_name in training_datasets:
                if dataset_name in batch_results:
                    X_train_list.append(batch_results[dataset_name]["features"])
                    y_train_list.append(batch_results[dataset_name]["labels"])
                    self.logger.info(f"📊 DEBUG - Training dataset '{dataset_name}':")
                    self.logger.info(
                        f"    Classes: {batch_results[dataset_name]['classes']}"
                    )
                    self.logger.info(
                        f"    Distribution: {batch_results[dataset_name]['class_distribution']}"
                    )

            X_train_combined = np.vstack(X_train_list) if X_train_list else np.array([])
            y_train_combined = (
                np.concatenate(y_train_list) if y_train_list else np.array([])
            )

            self.logger.info("=" * 80)
            self.logger.info("📊 DEBUG - COMBINED TRAINING DATA:")
            self.logger.info(f"    Total samples: {len(X_train_combined)}")
            self.logger.info(
                f"    Unique classes: {sorted(list(set(y_train_combined)))}"
            )
            self.logger.info(
                f"    Class distribution: {dict(zip(*np.unique(y_train_combined, return_counts=True)))}"
            )
            self.logger.info("=" * 80)

            # Combine validation datasets
            X_val_list = []
            y_val_list = []
            for dataset_name in validation_datasets:
                if dataset_name in batch_results:
                    X_val_list.append(batch_results[dataset_name]["features"])
                    y_val_list.append(batch_results[dataset_name]["labels"])
                    self.logger.info(f"📊 DEBUG - Validation dataset '{dataset_name}':")
                    self.logger.info(
                        f"    Classes: {batch_results[dataset_name]['classes']}"
                    )
                    self.logger.info(
                        f"    Distribution: {batch_results[dataset_name]['class_distribution']}"
                    )

            X_val_combined = np.vstack(X_val_list) if X_val_list else np.array([])
            y_val_combined = np.concatenate(y_val_list) if y_val_list else np.array([])

            self.logger.info("=" * 80)
            self.logger.info("📊 DEBUG - COMBINED VALIDATION DATA:")
            self.logger.info(f"    Total samples: {len(X_val_combined)}")
            self.logger.info(f"    Unique classes: {sorted(list(set(y_val_combined)))}")
            self.logger.info(
                f"    Class distribution: {dict(zip(*np.unique(y_val_combined, return_counts=True)))}"
            )
            self.logger.info("=" * 80)

            # Apply domain adaptation split if enabled
            if use_domain_adaptation and len(X_val_combined) > 0:
                from sklearn.model_selection import train_test_split

                # Split validation data: some for training, some for validation
                X_val_train, X_val_val, y_val_train, y_val_val = train_test_split(
                    X_val_combined,
                    y_val_combined,
                    train_size=holdout_train_split,
                    random_state=42,
                    stratify=y_val_combined,  # Maintain class balance
                )

                # Add hold-out training portion to main training data
                X_train_combined = np.vstack([X_train_combined, X_val_train])
                y_train_combined = np.concatenate([y_train_combined, y_val_train])

                # Update validation to be only the held-out portion
                X_val_combined = X_val_val
                y_val_combined = y_val_val

                self.logger.info(f"🔄 Domain Adaptation Applied:")
                self.logger.info(
                    f"  Added {len(X_val_train)} hold-out samples to training"
                )
                self.logger.info(
                    f"  Kept {len(X_val_val)} hold-out samples for validation"
                )
                self.logger.info(
                    f"  New training total: {len(X_train_combined)} samples"
                )
                self.logger.info("=" * 80)

            # Apply class filtering if enabled
            class_filtering_config = self.config.get("class_filtering", {})
            if class_filtering_config.get("enabled", False):
                self.logger.info("=" * 80)
                self.logger.info("🎯 CLASS FILTERING ENABLED")
                self.logger.info("=" * 80)

                # Filter training data
                classes_to_exclude_train = class_filtering_config.get(
                    "classes_to_exclude_train", []
                )
                if classes_to_exclude_train:
                    self.logger.info(
                        f"Filtering training data to exclude: {classes_to_exclude_train}"
                    )
                    self.logger.info(
                        f"  Before filtering: {len(X_train_combined)} samples"
                    )
                    self.logger.info(
                        f"  Class distribution before: {dict(zip(*np.unique(y_train_combined, return_counts=True)))}"
                    )

                    # Keep only samples NOT in excluded classes
                    train_mask = ~np.isin(y_train_combined, classes_to_exclude_train)
                    X_train_combined = X_train_combined[train_mask]
                    y_train_combined = y_train_combined[train_mask]

                    self.logger.info(
                        f"  After filtering: {len(X_train_combined)} samples"
                    )
                    self.logger.info(
                        f"  Class distribution after: {dict(zip(*np.unique(y_train_combined, return_counts=True)))}"
                    )

                # Filter validation data
                classes_to_exclude_validation = class_filtering_config.get(
                    "classes_to_exclude_validation", []
                )
                if classes_to_exclude_validation:
                    self.logger.info(
                        f"Filtering validation data to exclude: {classes_to_exclude_validation}"
                    )
                    self.logger.info(
                        f"  Before filtering: {len(X_val_combined)} samples"
                    )
                    self.logger.info(
                        f"  Class distribution before: {dict(zip(*np.unique(y_val_combined, return_counts=True)))}"
                    )

                    # Keep only samples NOT in excluded classes
                    val_mask = ~np.isin(y_val_combined, classes_to_exclude_validation)
                    X_val_combined = X_val_combined[val_mask]
                    y_val_combined = y_val_combined[val_mask]

                    self.logger.info(
                        f"  After filtering: {len(X_val_combined)} samples"
                    )
                    self.logger.info(
                        f"  Class distribution after: {dict(zip(*np.unique(y_val_combined, return_counts=True)))}"
                    )

                self.logger.info("=" * 80)

            self.logger.info(
                f"✓ Combined training data: {len(X_train_combined)} samples"
            )
            self.logger.info(
                f"✓ Combined validation data: {len(X_val_combined)} samples"
            )

            # Run validation-aware analysis
            results = self._run_with_validation(
                X_train_combined,
                y_train_combined,
                X_val_combined,
                y_val_combined,
                training_datasets,
                validation_datasets,
                batch_results,
            )

            return results
        else:
            self.logger.info(f"✓ Standard mode - no validation datasets specified")
            self.logger.info(f"  Will combine all datasets and use cross-validation")

            # COMBINE ALL DATASETS when no validation is specified
            X_combined_list = []
            y_combined_list = []
            dataset_names = []

            for dataset_name, batch_data in batch_results.items():
                X_combined_list.append(batch_data["features"])
                y_combined_list.append(batch_data["labels"])
                dataset_names.append(dataset_name)

            X_combined = np.vstack(X_combined_list) if X_combined_list else np.array([])
            y_combined = (
                np.concatenate(y_combined_list) if y_combined_list else np.array([])
            )

            self.logger.info(
                f"✓ Combined {len(dataset_names)} datasets: {dataset_names}"
            )
            self.logger.info(f"✓ Total samples: {len(X_combined)}")
            self.logger.info(
                f"✓ Class distribution: {dict(zip(*np.unique(y_combined, return_counts=True)))}"
            )

            # Apply class filtering if enabled (for standard mode)
            class_filtering_config = self.config.get("class_filtering", {})
            if class_filtering_config.get("enabled", False):
                classes_to_exclude = class_filtering_config.get(
                    "classes_to_exclude_train", []
                )
                if classes_to_exclude:
                    self.logger.info("=" * 80)
                    self.logger.info("🎯 CLASS FILTERING ENABLED (Standard Mode)")
                    self.logger.info("=" * 80)
                    self.logger.info(
                        f"Filtering combined data to exclude: {classes_to_exclude}"
                    )
                    self.logger.info(f"  Before filtering: {len(X_combined)} samples")
                    self.logger.info(
                        f"  Class distribution before: {dict(zip(*np.unique(y_combined, return_counts=True)))}"
                    )

                    # Keep only samples NOT in excluded classes
                    combined_mask = ~np.isin(y_combined, classes_to_exclude)
                    X_combined = X_combined[combined_mask]
                    y_combined = y_combined[combined_mask]

                    self.logger.info(f"  After filtering: {len(X_combined)} samples")
                    self.logger.info(
                        f"  Class distribution after: {dict(zip(*np.unique(y_combined, return_counts=True)))}"
                    )
                    self.logger.info("=" * 80)

        # Define classifiers to test
        classifiers = self._get_classifiers()

        # Run analysis on COMBINED data
        self.logger.info("Analyzing combined dataset...")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        # Perform cross-validation analysis on combined data
        combined_cv_results = self._perform_cross_validation(
            X_scaled, y_combined, classifiers, "combined_datasets"
        )

        # Store results
        combined_results = {
            "cv_results": combined_cv_results,
            "scaler": scaler,
            "num_samples": len(X_combined),
            "num_features": X_combined.shape[1],
            "classes": sorted(list(set(y_combined))),
            "class_distribution": dict(zip(*np.unique(y_combined, return_counts=True))),
            "dataset_names": dataset_names,
        }

        # Save combined results
        self._save_batch_discrimination_results(combined_results, "combined_datasets")

        # Create plots for combined data
        combined_batch_data = {
            "features": X_combined,
            "labels": y_combined,
            "classes": sorted(list(set(y_combined))),
        }
        self._create_batch_plots(
            combined_results, "combined_datasets", combined_batch_data
        )

        # Find best performing classifier
        best_clf_name = max(
            combined_cv_results.items(), key=lambda x: x[1]["mean_accuracy"]
        )[0]
        best_accuracy = combined_cv_results[best_clf_name]["mean_accuracy"]

        best_batch_info = {
            "batch_name": "combined_datasets",
            "classifier": best_clf_name,
            "accuracy": best_accuracy,
        }

        # Save the best model
        classifiers_dict = self._get_classifiers()
        best_clf = classifiers_dict[best_clf_name]
        best_clf.fit(X_scaled, y_combined)  # Train on all combined data

        model_data = {
            "model": best_clf,
            "scaler": scaler,
            "classes": sorted(list(set(y_combined))),
            "batch_name": "combined_datasets",
            "accuracy": best_accuracy,
            "feature_names": [f"feature_{i}" for i in range(X_combined.shape[1])],
        }
        import pickle

        model_path = os.path.join(
            self.experiment_output_dir, "best_discrimination_model.pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        self.logger.info(
            f"Saved best model ({best_clf_name}, acc: {best_accuracy:.3f}) to {model_path}"
        )

        results = {
            "batch_performance_results": {"combined_datasets": combined_results},
            "cross_batch_results": {},  # No cross-batch analysis when combining
            "best_batch_info": best_batch_info,
            "best_classifier": {
                "name": best_clf_name,
                "mean_accuracy": best_accuracy,
                "validation_accuracy": best_accuracy,  # For orch compatibility
            },
            "total_batches": 1,  # One combined dataset
        }

        # Create performance visualizations
        self.logger.info("Creating performance visualizations...")
        self._create_performance_visualizations(
            {"combined_datasets": combined_results}, {}  # Empty cross-batch results
        )
        self.logger.info("Performance visualizations created successfully")

        # Save overall results summary
        summary_path = os.path.join(
            self.experiment_output_dir, "discrimination_summary.json"
        )
        self.save_results(
            {
                "combined_datasets_performance": {
                    name: {
                        "mean_accuracy": metrics["mean_accuracy"],
                        "std_accuracy": metrics["std_accuracy"],
                    }
                    for name, metrics in combined_cv_results.items()
                },
                "best_classifier": best_clf_name,
                "best_accuracy": best_accuracy,
                "total_samples": len(X_combined),
                "num_features": X_combined.shape[1],
                "datasets_combined": dataset_names,
            },
            "discrimination_summary.json",
        )
        self.logger.info(f"Results saved to: {summary_path}")

        self.logger.info("Discrimination analysis experiment completed")
        return results

    def _OLD_run_per_batch(self, batch_results, classifiers):
        """OLD METHOD - runs analysis on each batch separately (DEPRECATED)"""
        # Perform analysis for each batch separately
        all_batch_results = {}

        for batch_name, batch_data in batch_results.items():
            self.logger.info(f"Analyzing batch: {batch_name}")

            X = batch_data["features"]
            y = batch_data["labels"]

            # Ensure X and y are numpy arrays
            X = np.array(X) if not isinstance(X, np.ndarray) else X
            y = np.array(y) if not isinstance(y, np.ndarray) else y

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

        # NEW: Save the best model for reuse
        if best_batch_info:
            best_clf_name = best_batch_info["classifier"]
            best_batch_name = best_batch_info["batch_name"]
            best_batch_data = batch_results[best_batch_name]

            # Get the classifier and fit it on the best batch's data
            classifiers_dict = self._get_classifiers()
            best_clf = classifiers_dict[best_clf_name]
            X_best = best_batch_data["features"]
            y_best = best_batch_data["labels"]
            scaler = StandardScaler()
            X_best_scaled = scaler.fit_transform(X_best)
            best_clf.fit(X_best_scaled, y_best)  # Train the model

            # Save model, scaler, and metadata
            model_data = {
                "model": best_clf,
                "scaler": scaler,
                "classes": best_batch_data["classes"],
                "batch_name": best_batch_name,
                "accuracy": best_batch_info["accuracy"],
                "feature_names": [
                    f"feature_{i}" for i in range(X_best.shape[1])
                ],  # Optional
            }
            import pickle

            model_path = os.path.join(
                self.experiment_output_dir, "best_discrimination_model.pkl"
            )
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
            self.logger.info(
                f"Saved best model ({best_clf_name}, acc: {best_batch_info['accuracy']:.3f}) to {model_path}"
            )

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

    def _run_with_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        training_datasets: list,
        validation_datasets: list,
        batch_results: dict,
    ) -> Dict[str, Any]:
        """
        Run discrimination analysis with separate validation set.

        Args:
            X_train: Combined training features
            y_train: Combined training labels
            X_val: Combined validation features
            y_val: Combined validation labels
            training_datasets: List of training dataset names
            validation_datasets: List of validation dataset names
            batch_results: Original batch results

        Returns:
            Analysis results with both test and validation metrics
        """
        from sklearn.model_selection import train_test_split

        self.logger.info("=" * 80)
        self.logger.info("Running Discrimination Analysis with Validation Set")
        self.logger.info("=" * 80)

        # DEBUG: Log data before split
        self.logger.info("📊 DEBUG - DATA BEFORE TRAIN/TEST SPLIT:")
        self.logger.info(f"    Training data: {len(X_train)} samples")
        self.logger.info(f"    Training classes: {sorted(list(set(y_train)))}")
        self.logger.info(
            f"    Training distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
        )
        self.logger.info(f"    Validation data: {len(X_val)} samples")
        self.logger.info(f"    Validation classes: {sorted(list(set(y_val)))}")
        self.logger.info(
            f"    Validation distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}"
        )

        # Use 5-fold stratified cross-validation on training data
        self.logger.info("=" * 80)
        self.logger.info("🔄 Using 5-Fold Stratified Cross-Validation on Training Data")
        self.logger.info("=" * 80)
        self.logger.info(f"Training set: {len(X_train)} samples (all data used in CV)")
        self.logger.info(
            f"Validation set: {len(X_val)} samples (held-out for final eval)"
        )

        # Scale features - fit scaler on ALL training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Get classifiers
        classifiers = self._get_classifiers()

        # Setup cross-validation
        from sklearn.model_selection import cross_validate, cross_val_predict

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Train and evaluate each classifier with cross-validation
        results_dict = {}
        trained_classifiers = {}  # Store trained classifier objects for saving

        for clf_name, clf in classifiers.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Evaluating: {clf_name}")
            self.logger.info(f"{'='*60}")

            try:
                # Perform 5-fold cross-validation on training data
                self.logger.info(f"  Running 5-fold cross-validation...")

                cv_results = cross_validate(
                    clf,
                    X_train_scaled,
                    y_train,
                    cv=cv,
                    scoring=["accuracy", "f1_weighted"],
                    return_train_score=True,
                    n_jobs=-1,  # Use all CPU cores
                )

                # Calculate CV statistics
                train_accuracy_cv = cv_results["train_accuracy"].mean()
                train_accuracy_std = cv_results["train_accuracy"].std()
                test_accuracy_cv = cv_results[
                    "test_accuracy"
                ].mean()  # This is the CV test score
                test_accuracy_std = cv_results["test_accuracy"].std()
                train_f1_cv = cv_results["train_f1_weighted"].mean()
                test_f1_cv = cv_results["test_f1_weighted"].mean()
                test_f1_std = cv_results["test_f1_weighted"].std()

                self.logger.info(f"  ✓ Cross-Validation Results (5 folds):")
                self.logger.info(
                    f"    Train Accuracy: {train_accuracy_cv:.4f} ± {train_accuracy_std:.4f}"
                )
                self.logger.info(
                    f"    CV Test Accuracy: {test_accuracy_cv:.4f} ± {test_accuracy_std:.4f}"
                )
                self.logger.info(f"    Train F1: {train_f1_cv:.4f}")
                self.logger.info(
                    f"    CV Test F1: {test_f1_cv:.4f} ± {test_f1_std:.4f}"
                )

                # Get CV predictions for confusion matrix
                self.logger.info(f"  Collecting CV predictions for confusion matrix...")
                y_train_cv_pred = cross_val_predict(
                    clf,
                    X_train_scaled,
                    y_train,
                    cv=cv,
                    n_jobs=-1,
                )
                self.logger.info(f"  ✓ Collected {len(y_train_cv_pred)} CV predictions")

                # Now train final model on ALL training data for validation set evaluation
                self.logger.info(f"  Training final model on all training data...")
                clf.fit(X_train_scaled, y_train)

                # Store the trained classifier
                trained_classifiers[clf_name] = clf

                # Get confidence filtering config
                conf_config = self.config.get("confidence_filtering", {})
                conf_enabled = conf_config.get("enabled", False)
                conf_threshold = conf_config.get("threshold", 0.7)
                conf_mode = conf_config.get("mode", "reject")
                conf_default_class = conf_config.get("default_class", "no_contact")

                # Predict on FULL training set (for overfitting analysis)
                y_train_pred = clf.predict(X_train_scaled)
                y_train_proba = clf.predict_proba(X_train_scaled)

                if conf_enabled:
                    self.logger.info(
                        f"  🔍 Applying confidence filtering to TRAIN set:"
                    )
                    y_train_filtered, y_train_pred_filtered, train_conf_stats = (
                        self.apply_confidence_filtering(
                            y_train,  # Full training labels
                            y_train_pred,
                            y_train_proba,
                            threshold=conf_threshold,
                            mode=conf_mode,
                            default_class=conf_default_class,
                        )
                    )
                else:
                    y_train_filtered = y_train  # Full training labels
                    y_train_pred_filtered = y_train_pred
                    train_conf_stats = None

                train_accuracy = accuracy_score(y_train_filtered, y_train_pred_filtered)
                train_f1 = f1_score(
                    y_train_filtered, y_train_pred_filtered, average="weighted"
                )

                # Predict on validation set
                y_val_pred = clf.predict(X_val_scaled)
                y_val_proba = clf.predict_proba(X_val_scaled)

                if conf_enabled:
                    self.logger.info(
                        f"  🔍 Applying confidence filtering to VALIDATION set:"
                    )
                    y_val_filtered, y_val_pred_filtered, val_conf_stats = (
                        self.apply_confidence_filtering(
                            y_val,
                            y_val_pred,
                            y_val_proba,
                            threshold=conf_threshold,
                            mode=conf_mode,
                            default_class=conf_default_class,
                        )
                    )
                else:
                    y_val_filtered = y_val
                    y_val_pred_filtered = y_val_pred
                    val_conf_stats = None

                val_accuracy = accuracy_score(y_val_filtered, y_val_pred_filtered)
                val_f1 = f1_score(
                    y_val_filtered, y_val_pred_filtered, average="weighted"
                )

                self.logger.info(f"  ✓ Final Model Performance:")
                self.logger.info(
                    f"    Train Accuracy (all data): {train_accuracy:.4f} | F1: {train_f1:.4f}"
                )
                self.logger.info(
                    f"    Validation Accuracy: {val_accuracy:.4f} | F1: {val_f1:.4f}"
                )

                # Store results with CV metrics
                results_dict[clf_name] = {
                    "train_accuracy": train_accuracy,
                    "train_f1": train_f1,
                    "cv_test_accuracy": test_accuracy_cv,  # Cross-validation test accuracy
                    "cv_test_accuracy_std": test_accuracy_std,  # Standard deviation across folds
                    "cv_test_f1": test_f1_cv,
                    "cv_test_f1_std": test_f1_std,
                    "cv_train_accuracy": train_accuracy_cv,
                    "cv_train_accuracy_std": train_accuracy_std,
                    "validation_accuracy": val_accuracy,  # Held-out validation set
                    "validation_f1": val_f1,
                    "validation_predictions": y_val_pred,
                    "cv_predictions": y_train_cv_pred,  # NEW: Store CV predictions
                    "train_confidence_stats": train_conf_stats,
                    "validation_confidence_stats": val_conf_stats,
                }
            except Exception as e:
                self.logger.warning(f"  ⚠ Failed to train {clf_name}: {e}")
                import traceback

                self.logger.warning(traceback.format_exc())
                continue

        # Create summary
        results = {
            "validation_mode": True,
            "cross_validation_enabled": True,  # NEW: Flag that we used CV
            "cv_folds": 5,  # NEW: Number of CV folds
            "training_datasets": training_datasets,
            "validation_datasets": validation_datasets,
            "classifier_results": results_dict,
            "num_train_samples": len(X_train),  # Full training set
            "num_val_samples": len(X_val),
            "batch_performance_results": {},  # For compatibility
            "cross_batch_results": {},  # For compatibility
        }

        # Find best classifier based on CV test accuracy
        best_clf_name = max(
            results_dict.keys(), key=lambda k: results_dict[k]["cv_test_accuracy"]
        )
        results["best_classifier"] = {
            "name": best_clf_name,
            "train_accuracy": results_dict[best_clf_name]["train_accuracy"],
            "cv_test_accuracy": results_dict[best_clf_name]["cv_test_accuracy"],
            "cv_test_accuracy_std": results_dict[best_clf_name]["cv_test_accuracy_std"],
            "validation_accuracy": results_dict[best_clf_name]["validation_accuracy"],
        }

        self.logger.info(f"\n🏆 Best Classifier (by CV accuracy): {best_clf_name}")
        self.logger.info(
            f"  Train Accuracy: {results_dict[best_clf_name]['train_accuracy']:.4f}"
        )
        self.logger.info(
            f"  CV Test Accuracy: {results_dict[best_clf_name]['cv_test_accuracy']:.4f} ± {results_dict[best_clf_name]['cv_test_accuracy_std']:.4f}"
        )
        self.logger.info(
            f"  Validation Accuracy: {results_dict[best_clf_name]['validation_accuracy']:.4f}"
        )

        # Save validation results
        self._save_validation_results(
            results,
            X_train,  # Full training set
            y_train,
            X_val,
            y_val,
            scaler,
        )

        # Save top 3 trained models for reconstruction
        self._save_top_models(
            trained_classifiers=trained_classifiers,
            results_dict=results_dict,
            scaler=scaler,
            classes=sorted(list(set(y_train))),  # Use full training labels
            metric_key="cv_test_accuracy",  # Use CV test accuracy for ranking
            top_n=3,
        )

        return results

    def _run_with_3way_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_tuning: np.ndarray,
        y_tuning: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        training_datasets: list,
        tuning_datasets: list,
        final_test_datasets: list,
        batch_results: dict,
    ) -> Dict[str, Any]:
        """
        Run discrimination analysis with 3-way split (train/tuning/test).

        This implements a rigorous ML evaluation pipeline:
        1. Train models on training datasets
        2. Tune hyperparameters/select models on tuning datasets
        3. Report final unbiased performance on test datasets

        Args:
            X_train: Training features
            y_train: Training labels
            X_tuning: Tuning features (for hyperparameter optimization)
            y_tuning: Tuning labels
            X_test: Final test features (for unbiased evaluation)
            y_test: Final test labels
            training_datasets: List of training dataset names
            tuning_datasets: List of tuning dataset names
            final_test_datasets: List of final test dataset names
            batch_results: Original batch results

        Returns:
            Analysis results with train/tuning/test metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("Running Discrimination Analysis with 3-Way Split")
        self.logger.info("=" * 80)
        self.logger.info("Step 1: Train models on training datasets")
        self.logger.info("Step 2: Tune hyperparameters on tuning datasets")
        self.logger.info(
            "Step 3: Final evaluation on test datasets (unbiased performance)"
        )
        self.logger.info("=" * 80)

        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Tuning set: {len(X_tuning)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")

        # FEATURE SELECTION (if enabled)
        feature_selection_config = self.config.get("feature_selection", {})
        use_feature_selection = feature_selection_config.get("enabled", False)

        if use_feature_selection:
            top_k = feature_selection_config.get("top_k_features", 50)
            selection_method = feature_selection_config.get("method", "ensemble")

            self.logger.info("=" * 80)
            self.logger.info(f"🔍 FEATURE SELECTION ENABLED")
            self.logger.info(f"  Method: {selection_method}")
            self.logger.info(f"  Selecting top {top_k} of {X_train.shape[1]} features")
            self.logger.info("=" * 80)

            # Analyze feature importance and select best features
            selected_indices, avg_ranks, feature_analysis = (
                self._analyze_feature_importance(
                    X_train,
                    y_train,
                    X_tuning,
                    y_tuning,
                    top_k=top_k,
                    method=selection_method,
                )
            )

            # Apply feature selection to all datasets
            X_train = X_train[:, selected_indices]
            X_tuning = X_tuning[:, selected_indices]
            X_test = X_test[:, selected_indices]

            self.logger.info(
                f"✅ Feature selection complete: {X_train.shape[1]} features selected"
            )
            self.logger.info(f"   Performance impact:")
            self.logger.info(
                f"   - All features:      {feature_analysis.get('all_features_score', 0.0):.4f}"
            )
            self.logger.info(
                f"   - Selected features: {feature_analysis.get('selected_features_score', 0.0):.4f}"
            )
            self.logger.info(
                f"   - Difference:        {feature_analysis.get('score_diff', 0.0):+.4f}"
            )
            self.logger.info("=" * 80)
        else:
            self.logger.info(
                f"Feature selection DISABLED - using all {X_train.shape[1]} features"
            )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_tuning_scaled = scaler.transform(X_tuning)
        X_test_scaled = scaler.transform(X_test)

        # Get classifiers
        classifiers = self._get_classifiers()

        # Train and evaluate each classifier
        results_dict = {}
        for clf_name, clf in classifiers.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training: {clf_name}")
            self.logger.info(f"{'='*60}")

            try:
                # STEP 1: Hyperparameter tuning on training + tuning data
                # For GPU-MLP classifiers, use external validation
                if hasattr(clf, "fit") and "GPU-MLP" in clf_name:
                    self.logger.info(
                        f"  [GPU-MLP] Using dedicated tuning data for validation ({len(X_tuning_scaled)} samples)"
                    )
                    # GPU models handle validation internally during training
                    clf.fit(
                        X_train_scaled, y_train, X_val=X_tuning_scaled, y_val=y_tuning
                    )
                    best_clf = clf
                    search_results = None
                else:
                    # Sklearn models: tune hyperparameters using RandomizedSearchCV
                    best_clf, search_results = self._tune_hyperparameters(
                        clf_name,
                        clf,
                        X_train_scaled,
                        y_train,
                        X_tuning_scaled,
                        y_tuning,
                        n_iter=20,  # Number of random parameter combinations to try
                    )

                # STEP 2: Evaluate on all three datasets
                # Predict on train set (for overfitting analysis)
                y_train_pred = best_clf.predict(X_train_scaled)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred, average="weighted")

                # Predict on tuning set (validation performance)
                y_tuning_pred = best_clf.predict(X_tuning_scaled)
                tuning_accuracy = accuracy_score(y_tuning, y_tuning_pred)
                tuning_f1 = f1_score(y_tuning, y_tuning_pred, average="weighted")

                # Predict on test set (final unbiased evaluation)
                y_test_pred = best_clf.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average="weighted")

                self.logger.info(f"\n  Results:")
                self.logger.info(
                    f"    Train Accuracy: {train_accuracy:.4f} | F1: {train_f1:.4f}"
                )
                self.logger.info(
                    f"    Tuning Accuracy: {tuning_accuracy:.4f} | F1: {tuning_f1:.4f}"
                )
                self.logger.info(
                    f"    Test Accuracy: {test_accuracy:.4f} | F1: {test_f1:.4f}"
                )

                # Store results
                result_entry = {
                    "train_accuracy": train_accuracy,
                    "train_f1": train_f1,
                    "tuning_accuracy": tuning_accuracy,
                    "tuning_f1": tuning_f1,
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1,
                    "tuning_predictions": y_tuning_pred,
                    "test_predictions": y_test_pred,
                    "best_estimator": best_clf,
                }

                # Add hyperparameter search info if available
                if search_results is not None:
                    result_entry["best_params"] = search_results.best_params_
                    result_entry["cv_score"] = search_results.best_score_

                results_dict[clf_name] = result_entry
            except Exception as e:
                self.logger.warning(f"  ⚠ Failed to train {clf_name}: {e}")
                continue

        # Create summary
        results = {
            "split_mode": "3-way",
            "training_datasets": training_datasets,
            "tuning_datasets": tuning_datasets,
            "final_test_datasets": final_test_datasets,
            "classifier_results": results_dict,
            "num_train_samples": len(X_train),
            "num_tuning_samples": len(X_tuning),
            "num_test_samples": len(X_test),
            "batch_performance_results": {},  # For compatibility
            "cross_batch_results": {},  # For compatibility
        }

        # Find best classifier based on tuning accuracy
        best_clf_name = max(
            results_dict.keys(), key=lambda k: results_dict[k]["tuning_accuracy"]
        )
        results["best_classifier"] = {
            "name": best_clf_name,
            "train_accuracy": results_dict[best_clf_name]["train_accuracy"],
            "tuning_accuracy": results_dict[best_clf_name]["tuning_accuracy"],
            "test_accuracy": results_dict[best_clf_name]["test_accuracy"],
        }

        self.logger.info("\n" + "=" * 80)
        self.logger.info("BEST CLASSIFIER (selected on tuning set)")
        self.logger.info("=" * 80)
        self.logger.info(f"Model: {best_clf_name}")
        self.logger.info(
            f"  Train Accuracy: {results_dict[best_clf_name]['train_accuracy']:.4f}"
        )
        self.logger.info(
            f"  Tuning Accuracy: {results_dict[best_clf_name]['tuning_accuracy']:.4f}"
        )
        self.logger.info(
            f"  Test Accuracy: {results_dict[best_clf_name]['test_accuracy']:.4f}"
        )

        # Save 3-way split results
        self._save_3way_split_results(
            results,
            X_train,
            y_train,
            X_tuning,
            y_tuning,
            X_test,
            y_test,
            scaler,
        )

        # Save top 3 trained models for reconstruction
        # Extract trained classifiers from results_dict
        trained_classifiers = {
            clf_name: res["best_estimator"]
            for clf_name, res in results_dict.items()
            if "best_estimator" in res
        }
        self._save_top_models(
            trained_classifiers=trained_classifiers,
            results_dict=results_dict,
            scaler=scaler,
            classes=sorted(list(set(y_train))),
            metric_key="tuning_accuracy",  # Use tuning accuracy for 3-way split
            top_n=3,
        )

        return results

    def _get_hyperparameter_search_spaces(self) -> Dict[str, Dict]:
        """
        Define hyperparameter search spaces for each classifier type.

        Returns:
            Dictionary mapping classifier names to their hyperparameter grids
        """
        return {
            "Random Forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
            "SVM (RBF)": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "kernel": ["rbf"],
            },
            "SVM (Linear)": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear"],
            },
            "K-NN": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
            "Logistic Regression": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
                "solver": ["lbfgs", "saga"],
                "max_iter": [1000],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
                "subsample": [0.8, 0.9, 1.0],
            },
            "Extra Trees": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 9],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
        }

    def _tune_hyperparameters(
        self,
        clf_name: str,
        base_clf,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_tuning: np.ndarray,
        y_tuning: np.ndarray,
        n_iter: int = 20,
    ):
        """
        Tune hyperparameters using the tuning dataset.

        Args:
            clf_name: Name of the classifier
            base_clf: Base classifier instance
            X_train: Training features
            y_train: Training labels
            X_tuning: Tuning features (used as validation)
            y_tuning: Tuning labels
            n_iter: Number of random parameter settings to try

        Returns:
            Best estimator and search results
        """
        search_spaces = self._get_hyperparameter_search_spaces()

        # Check if this classifier has a defined search space
        if clf_name not in search_spaces:
            self.logger.info(
                f"  No hyperparameter search space defined for {clf_name}, using default parameters"
            )
            base_clf.fit(X_train, y_train)
            return base_clf, None

        self.logger.info(
            f"  Tuning hyperparameters with RandomizedSearchCV ({n_iter} iterations)..."
        )

        param_grid = search_spaces[clf_name]

        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            base_clf,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,  # 3-fold CV on training data
            scoring="accuracy",
            n_jobs=-1,  # Use all CPU cores
            random_state=42,
            verbose=0,
        )

        # Fit on training data (CV happens internally)
        search.fit(X_train, y_train)

        # Evaluate best model on tuning set
        tuning_score = search.best_estimator_.score(X_tuning, y_tuning)

        self.logger.info(f"  Best params: {search.best_params_}")
        self.logger.info(f"  CV score (train): {search.best_score_:.4f}")
        self.logger.info(f"  Tuning score: {tuning_score:.4f}")

        return search.best_estimator_, search

    def _analyze_feature_importance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None,
        top_k: int = 50,
    ) -> tuple:
        """
        Analyze feature importance using ensemble methods and select top features.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Optional list of feature names
            top_k: Number of top features to select

        Returns:
            tuple: (selected_feature_indices, importance_scores, feature_analysis_dict)
        """
        self.logger.info(
            f"\n🔍 Analyzing feature importance (selecting top {top_k} features)..."
        )

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Use multiple methods to get robust feature importance
        importance_scores = {}

        # 1. Random Forest importance
        self.logger.info("  Computing Random Forest importance...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importance_scores["random_forest"] = rf.feature_importances_

        # 2. Gradient Boosting importance
        self.logger.info("  Computing Gradient Boosting importance...")
        gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
        gb.fit(X_train, y_train)
        importance_scores["gradient_boosting"] = gb.feature_importances_

        # 3. XGBoost importance
        self.logger.info("  Computing XGBoost importance...")
        xgb_model = XGBoostWrapper(n_estimators=200, random_state=42)
        xgb_model.fit(X_train, y_train)
        importance_scores["xgboost"] = xgb_model.model.feature_importances_

        # Aggregate importance scores (average rank across methods)
        self.logger.info("  Aggregating importance scores...")
        ranks = np.zeros((len(feature_names), len(importance_scores)))

        for i, (method, scores) in enumerate(importance_scores.items()):
            # Convert to ranks (higher importance = lower rank number)
            ranks[:, i] = len(scores) - np.argsort(np.argsort(scores))

        # Average ranks across methods (lower = more important)
        avg_ranks = np.mean(ranks, axis=1)

        # Get top-k features
        selected_indices = np.argsort(avg_ranks)[:top_k]

        # Create analysis dictionary
        feature_analysis = {
            "total_features": len(feature_names),
            "selected_features": top_k,
            "selected_indices": selected_indices.tolist(),
            "selected_names": [feature_names[i] for i in selected_indices],
            "importance_by_method": {
                method: scores.tolist() for method, scores in importance_scores.items()
            },
            "average_ranks": avg_ranks.tolist(),
            "top_features_details": [
                {
                    "rank": int(i + 1),
                    "index": int(idx),
                    "name": feature_names[idx],
                    "avg_rank": float(avg_ranks[idx]),
                    "rf_importance": float(importance_scores["random_forest"][idx]),
                    "gb_importance": float(importance_scores["gradient_boosting"][idx]),
                    "xgb_importance": float(importance_scores["xgboost"][idx]),
                }
                for i, idx in enumerate(selected_indices)
            ],
        }

        # Test performance with selected features
        self.logger.info(f"\n  Evaluating feature selection impact...")
        self.logger.info(f"  Original features: {X_train.shape[1]}")
        self.logger.info(f"  Selected features: {top_k}")

        # Train models with all features
        rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_all.fit(X_train, y_train)
        score_all = rf_all.score(X_val, y_val)

        # Train models with selected features
        X_train_selected = X_train[:, selected_indices]
        X_val_selected = X_val[:, selected_indices]
        rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selected.fit(X_train_selected, y_train)
        score_selected = rf_selected.score(X_val_selected, y_val)

        feature_analysis["performance_comparison"] = {
            "all_features_val_acc": float(score_all),
            "selected_features_val_acc": float(score_selected),
            "improvement": float(score_selected - score_all),
        }

        self.logger.info(f"  All features validation accuracy: {score_all:.4f}")
        self.logger.info(
            f"  Selected features validation accuracy: {score_selected:.4f}"
        )
        self.logger.info(f"  Improvement: {score_selected - score_all:+.4f}")

        # Save feature importance plot
        self._plot_feature_importance(
            selected_indices[:20],  # Top 20 for visualization
            feature_names,
            importance_scores,
        )

        return selected_indices, avg_ranks, feature_analysis

    def _plot_feature_importance(
        self,
        top_indices: np.ndarray,
        feature_names: List[str],
        importance_scores: Dict[str, np.ndarray],
    ):
        """Plot feature importance for top features."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        methods = ["random_forest", "gradient_boosting", "xgboost"]
        titles = ["Random Forest", "Gradient Boosting", "XGBoost"]

        for ax, method, title in zip(axes, methods, titles):
            scores = importance_scores[method][top_indices]
            names = [feature_names[i] for i in top_indices]

            ax.barh(range(len(scores)), scores)
            ax.set_yticks(range(len(scores)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel("Importance Score")
            ax.set_title(f"{title}\nTop {len(scores)} Features")
            ax.invert_yaxis()

        plt.tight_layout()
        self.save_plot(fig, "feature_importance_analysis.png")
        plt.close()

    def _get_classifiers(self) -> dict:
        """Define classifiers to evaluate."""
        classifiers = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            # "SVM (RBF)": SVC(kernel="rbf", random_state=42),
            # "SVM (Linear)": SVC(kernel="linear", random_state=42),
            "K-NN": KNeighborsClassifier(n_neighbors=5),
            # "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            # "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            # "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
            # "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
            # "XGBoost": XGBoostWrapper(
            #     n_estimators=100, random_state=42, eval_metric="mlogloss"
            # ),
            # "Naive Bayes": GaussianNB(),
            # "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
            # MLP with regularization for better generalization
            # "MLP (Small)": MLPWrapper(
            #     hidden_layer_sizes=(64, 32),
            #     activation="relu",
            #     solver="adam",
            #     alpha=0.01,  # L2 regularization
            #     batch_size=32,
            #     learning_rate="adaptive",
            #     learning_rate_init=0.001,
            #     max_iter=500,
            #     early_stopping=True,
            #     validation_fraction=0.15,
            #     n_iter_no_change=20,
            #     random_state=42,
            #     verbose=False,
            # ),
            "MLP (Medium)": MLPWrapper(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,  # L2 regularization
                batch_size=32,
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
                verbose=False,
            ),
            # "MLP (Large)": MLPWrapper(
            #     hidden_layer_sizes=(256, 128, 64, 32),
            #     activation="relu",
            #     solver="adam",
            #     alpha=0.005,  # Less regularization for larger network
            #     batch_size=32,
            #     learning_rate="adaptive",
            #     learning_rate_init=0.001,
            #     max_iter=1000,  # Train longer
            #     early_stopping=True,
            #     validation_fraction=0.15,
            #     n_iter_no_change=30,
            #     random_state=42,
            #     verbose=False,
            # ),
            # Optimized variants around Medium (which performed best)
            # "MLP (HighReg)": MLPWrapper(
            #     hidden_layer_sizes=(128, 64, 32),
            #     activation="relu",
            #     solver="adam",
            #     alpha=0.1,  # Much higher regularization to prevent overfitting
            #     batch_size=32,
            #     learning_rate="adaptive",
            #     learning_rate_init=0.001,
            #     max_iter=1000,
            #     early_stopping=True,
            #     validation_fraction=0.2,  # More validation data for early stopping
            #     n_iter_no_change=30,
            #     random_state=42,
            #     verbose=False,
            # ),
            # "MLP (VeryHighReg)": MLPWrapper(
            #     hidden_layer_sizes=(64, 32),  # Smaller network
            #     activation="relu",
            #     solver="adam",
            #     alpha=0.5,  # Very high regularization
            #     batch_size=64,  # Larger batch size for stability
            #     learning_rate="adaptive",
            #     learning_rate_init=0.0005,  # Lower learning rate
            #     max_iter=1000,
            #     early_stopping=True,
            #     validation_fraction=0.2,
            #     n_iter_no_change=40,  # More patience
            #     random_state=42,
            #     verbose=False,
            # ),
            # # NEW: PCA-based variants to reduce overfitting through dimensionality reduction
            # "PCA+MLP (HighReg)": PCAClassifierWrapper(
            #     base_classifier=MLPWrapper(
            #         hidden_layer_sizes=(128, 64, 32),
            #         activation="relu",
            #         solver="adam",
            #         alpha=0.1,
            #         batch_size=32,
            #         learning_rate="adaptive",
            #         learning_rate_init=0.001,
            #         max_iter=1000,
            #         early_stopping=True,
            #         validation_fraction=0.2,
            #         n_iter_no_change=30,
            #         random_state=42,
            #         verbose=False,
            #     ),
            #     n_components=0.95,  # Keep 95% variance (~109 components based on analysis)
            #     pca_whiten=False,
            # ),
            # "PCA+MLP (VeryHighReg)": PCAClassifierWrapper(
            #     base_classifier=MLPWrapper(
            #         hidden_layer_sizes=(64, 32),
            #         activation="relu",
            #         solver="adam",
            #         alpha=0.5,
            #         batch_size=64,
            #         learning_rate="adaptive",
            #         learning_rate_init=0.0005,
            #         max_iter=1000,
            #         early_stopping=True,
            #         validation_fraction=0.2,
            #         n_iter_no_change=40,
            #         random_state=42,
            #         verbose=False,
            #     ),
            #     n_components=0.95,
            #     pca_whiten=False,
            # ),
            # # NEW: Tree-based models that often generalize better than neural networks
            # "Random Forest (Tuned)": RandomForestClassifier(
            #     n_estimators=200,  # More trees for better generalization
            #     max_depth=20,  # Limit depth to prevent overfitting
            #     min_samples_split=10,  # Require more samples to split
            #     min_samples_leaf=5,  # Require more samples per leaf
            #     max_features="sqrt",  # Use sqrt features for each split
            #     random_state=42,
            #     n_jobs=-1,  # Use all CPU cores
            # ),
            # "Gradient Boosting (Tuned)": GradientBoostingClassifier(
            #     n_estimators=200,
            #     learning_rate=0.1,
            #     max_depth=6,  # Moderate depth
            #     min_samples_split=20,
            #     min_samples_leaf=10,
            #     subsample=0.8,  # Use 80% of data for each tree (stochastic boosting)
            #     random_state=42,
            # ),
            #     "MLP (Wide)": MLPWrapper(
            #         hidden_layer_sizes=(256, 128),  # Fewer layers, wider
            #         activation="relu",
            #         solver="adam",
            #         alpha=0.015,
            #         batch_size=32,
            #         learning_rate="adaptive",
            #         learning_rate_init=0.001,
            #         max_iter=600,
            #         early_stopping=True,
            #         validation_fraction=0.15,
            #         n_iter_no_change=20,
            #         random_state=42,
            #         verbose=False,
            #     ),
            #     "MLP (Deep-Narrow)": MLPWrapper(
            #         hidden_layer_sizes=(64, 64, 64, 32),  # Deeper, narrower
            #         activation="relu",
            #         solver="adam",
            #         alpha=0.012,
            #         batch_size=32,
            #         learning_rate="adaptive",
            #         learning_rate_init=0.001,
            #         max_iter=600,
            #         early_stopping=True,
            #         validation_fraction=0.15,
            #         n_iter_no_change=20,
            #         random_state=42,
            #         verbose=False,
            #     ),
        }

        # if self.config.get("include_lda", True):
        #     classifiers["Linear Discriminant Analysis"] = LinearDiscriminantAnalysis()

        # Add ensemble classifier combining Random Forest and SVM
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm_clf = SVC(
            kernel="rbf", probability=True, random_state=42
        )  # probability=True for soft voting
        # classifiers["Ensemble (RF + SVM)"] = VotingClassifier(
        #     estimators=[("rf", rf_clf), ("svm", svm_clf)], voting="soft"
        # )

        # ========================================================================
        # NEW: PCA-BASED VARIANTS (Dimensionality Reduction to Combat Overfitting)
        # ========================================================================
        # Problem: 55 features → curse of dimensionality, overfitting
        # Solution: PCA reduces to ~19 components (95% variance)
        # Expected: Better generalization on validation set

        # PCA + XGBoost (best traditional ML)
        # classifiers["PCA+XGBoost"] = PCAClassifierWrapper(
        #     base_classifier=XGBoostWrapper(
        #         n_estimators=100, random_state=42, eval_metric="mlogloss"
        #     ),
        #     n_components=0.95,  # Keep 95% variance (~19 components)
        #     pca_whiten=False,
        # )

        # # PCA + Random Forest
        # classifiers["PCA+RandomForest"] = PCAClassifierWrapper(
        #     base_classifier=RandomForestClassifier(n_estimators=100, random_state=42),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # PCA + Gradient Boosting
        # classifiers["PCA+GradientBoosting"] = PCAClassifierWrapper(
        #     base_classifier=GradientBoostingClassifier(random_state=42),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # PCA + MLP (Medium) - previously best on validation
        # classifiers["PCA+MLP(Medium)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # PCA + MLP (Large)
        # classifiers["PCA+MLP(Large)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(256, 128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.005,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=1000,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=30,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # PCA + Extra Trees
        # classifiers["PCA+ExtraTrees"] = PCAClassifierWrapper(
        #     base_classifier=ExtraTreesClassifier(n_estimators=100, random_state=42),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # ========================================================================
        # NEW: PCA VARIANTS WITH DIFFERENT COMPONENT COUNTS (Option 1)
        # ========================================================================
        # Testing if 95% variance is optimal or if we can do better with:
        # - Less components (90%) = more aggressive noise removal
        # - More components (99%) = keep more information
        # - Fixed counts (15, 20, 25) = explicit control

        # # PCA (90% variance) + MLP - More aggressive dimensionality reduction
        # classifiers["PCA90+MLP(Medium)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.90,  # Keep only 90% variance
        #     pca_whiten=False,
        # )

        # # PCA (99% variance) + MLP - Keep almost all information
        # classifiers["PCA99+MLP(Medium)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.99,  # Keep 99% variance
        #     pca_whiten=False,
        # )

        # # Fixed 15 components + MLP
        # classifiers["PCA15+MLP(Medium)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=15,  # Fixed 15 components
        #     pca_whiten=False,
        # )

        # # Fixed 20 components + MLP
        # classifiers["PCA20+MLP(Medium)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=20,  # Fixed 20 components
        #     pca_whiten=False,
        # )

        # ========================================================================
        # NEW: OPTIMIZED MLP ARCHITECTURES FOR PCA FEATURES (Option 2)
        # ========================================================================
        # Since PCA+MLP(Medium) works best, let's tune the MLP architecture

        # # Higher regularization - prevent overfitting on reduced features
        # classifiers["PCA+MLP(HighReg)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.05,  # Much higher regularization
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=800,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=25,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # Smaller network - less capacity = less overfitting
        # classifiers["PCA+MLP(Compact)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(64, 32),  # Smaller
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.02,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # Wider network - more neurons per layer
        # classifiers["PCA+MLP(Wide)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(256, 128),  # Wider, fewer layers
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.02,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=600,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # Deeper narrow network
        # classifiers["PCA+MLP(DeepNarrow)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(64, 64, 64, 32),  # Deeper, narrower
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.015,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=600,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.95,
        #     pca_whiten=False,
        # )

        # # PCA with whitening + MLP - normalize component variances
        # classifiers["PCA+MLP(Whitened)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.95,
        #     pca_whiten=True,  # Whitening enabled
        # )

        # # Best combo: 90% PCA + High Regularization
        # classifiers["PCA90+MLP(HighReg)"] = PCAClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.03,  # Higher regularization
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=600,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=25,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     n_components=0.90,
        #     pca_whiten=False,
        # )

        # ========================================================================
        # NEW: GPU-ACCELERATED CLASSIFIERS (PyTorch + CUDA)
        # ========================================================================
        # Use GPU for faster training (5-10x speedup over sklearn MLP)

        if GPU_AVAILABLE:
            self.logger.info(f"✓ GPU classifiers enabled (device: {get_device()})")

            # GPU MLP (Medium) - GPU-accelerated version of best performer
            # classifiers["GPU-MLP (Medium)"] = GPUMLPClassifier(
            #     hidden_layer_sizes=(128, 64, 32),
            #     dropout=0.3,
            #     learning_rate=0.001,
            #     weight_decay=0.01,
            #     batch_size=64,
            #     max_epochs=500,
            #     early_stopping=True,
            #     patience=25,
            #     validation_fraction=0.15,
            #     use_batch_norm=True,
            #     random_state=42,
            #     verbose=False,
            # )

            # GPU MLP (Medium-HighReg) - Higher regularization for generalization
            classifiers["GPU-MLP (Medium-HighReg)"] = GPUMLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                dropout=0.4,  # Higher dropout
                learning_rate=0.001,
                weight_decay=0.02,  # Higher L2
                batch_size=64,
                max_epochs=500,
                early_stopping=True,
                patience=30,
                validation_fraction=0.15,
                use_batch_norm=True,
                random_state=42,
                verbose=False,
            )

            # # GPU MLP (Large) - Larger network for complex patterns
            # classifiers["GPU-MLP (Large)"] = GPUMLPClassifier(
            #     hidden_layer_sizes=(256, 128, 64, 32),
            #     dropout=0.35,
            #     learning_rate=0.001,
            #     weight_decay=0.01,
            #     batch_size=64,
            #     max_epochs=800,
            #     early_stopping=True,
            #     patience=30,
            #     validation_fraction=0.15,
            #     use_batch_norm=True,
            #     random_state=42,
            #     verbose=False,
            # )

            # # GPU MLP (Deep) - Deeper network
            # classifiers["GPU-MLP (Deep)"] = GPUMLPClassifier(
            #     hidden_layer_sizes=(128, 128, 64, 64, 32),
            #     dropout=0.35,
            #     learning_rate=0.001,
            #     weight_decay=0.015,
            #     batch_size=64,
            #     max_epochs=600,
            #     early_stopping=True,
            #     patience=25,
            #     validation_fraction=0.15,
            #     use_batch_norm=True,
            #     random_state=42,
            #     verbose=False,
            # )
        else:
            self.logger.info("⚠ GPU classifiers not available (PyTorch/CUDA not found)")

        # ========================================================================
        # NEW: NORMALIZED CLASSIFIERS (Robust normalization for workspace invariance)
        # ========================================================================
        # These address the overfitting issue where tree models learn absolute
        # feature values that differ between workspaces.

        # if GPU_AVAILABLE:
        #     # Robust-normalized GPU MLP
        #     classifiers["Robust-GPU-MLP (Medium)"] = NormalizedClassifierWrapper(
        #         base_classifier=GPUMLPClassifier(
        #             hidden_layer_sizes=(128, 64, 32),
        #             dropout=0.3,
        #             learning_rate=0.001,
        #             weight_decay=0.01,
        #             batch_size=64,
        #             max_epochs=500,
        #             early_stopping=True,
        #             patience=25,
        #             use_batch_norm=True,
        #             random_state=42,
        #         ),
        #         normalization="robust",
        #         clip_outliers=True,
        #     )

        #     # Quantile-normalized GPU MLP
        #     classifiers["Quantile-GPU-MLP (Medium)"] = NormalizedClassifierWrapper(
        #         base_classifier=GPUMLPClassifier(
        #             hidden_layer_sizes=(128, 64, 32),
        #             dropout=0.3,
        #             learning_rate=0.001,
        #             weight_decay=0.01,
        #             batch_size=64,
        #             max_epochs=500,
        #             early_stopping=True,
        #             patience=25,
        #             use_batch_norm=True,
        #             random_state=42,
        #         ),
        #         normalization="quantile",
        #         clip_outliers=False,
        #     )

        # # Robust-normalized sklearn MLP (fallback if no GPU)
        # classifiers["Robust-MLP (Medium)"] = NormalizedClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     normalization="robust",
        #     clip_outliers=True,
        # )

        # # Quantile-normalized sklearn MLP
        # classifiers["Quantile-MLP (Medium)"] = NormalizedClassifierWrapper(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     normalization="quantile",
        #     clip_outliers=False,
        # )

        # Robust-normalized Random Forest
        # classifiers["Robust-RandomForest"] = NormalizedClassifierWrapper(
        #     base_classifier=RandomForestClassifier(n_estimators=100, random_state=42),
        #     normalization="robust",
        #     clip_outliers=True,
        # )

        # # Robust-normalized XGBoost
        # classifiers["Robust-XGBoost"] = NormalizedClassifierWrapper(
        #     base_classifier=XGBoostWrapper(
        #         n_estimators=100, random_state=42, eval_metric="mlogloss"
        #     ),
        #     normalization="robust",
        #     clip_outliers=True,
        # )

        # # Robust-normalized Gradient Boosting
        # classifiers["Robust-GradientBoosting"] = NormalizedClassifierWrapper(
        #     base_classifier=GradientBoostingClassifier(random_state=42),
        #     normalization="robust",
        #     clip_outliers=True,
        # )

        # ========================================================================
        # NEW: RELATIVE FEATURE CLASSIFIERS (Ratio-based features)
        # ========================================================================
        # These compute ratios between feature groups to focus on relative
        # relationships rather than absolute values

        # if GPU_AVAILABLE:
        #     classifiers["Relative-GPU-MLP (Medium)"] = RelativeFeatureClassifier(
        #         base_classifier=GPUMLPClassifier(
        #             hidden_layer_sizes=(128, 64, 32),
        #             dropout=0.3,
        #             learning_rate=0.001,
        #             weight_decay=0.01,
        #             batch_size=64,
        #             max_epochs=500,
        #             early_stopping=True,
        #             patience=25,
        #             use_batch_norm=True,
        #             random_state=42,
        #         ),
        #         add_original=True,
        #     )

        # classifiers["Relative-MLP (Medium)"] = RelativeFeatureClassifier(
        #     base_classifier=MLPWrapper(
        #         hidden_layer_sizes=(128, 64, 32),
        #         activation="relu",
        #         solver="adam",
        #         alpha=0.01,
        #         batch_size=32,
        #         learning_rate="adaptive",
        #         learning_rate_init=0.001,
        #         max_iter=500,
        #         early_stopping=True,
        #         validation_fraction=0.15,
        #         n_iter_no_change=20,
        #         random_state=42,
        #         verbose=False,
        #     ),
        #     add_original=True,
        # )

        # ========================================================================
        # OPTIMIZED MLP FROM HYPERPARAMETER TUNING
        # ========================================================================
        # Found via Optuna Bayesian optimization (50 trials)
        # Optimized for validation accuracy (cross-workspace generalization)

        # if GPU_AVAILABLE:
        # GPU-accelerated optimized MLP
        # classifiers["GPU-MLP (Tuned)"] = GPUMLPClassifier(
        #     hidden_layer_sizes=(256, 192, 160),
        #     dropout=0.162,
        #     learning_rate=0.000131,
        #     weight_decay=0.000294,
        #     batch_size=32,
        #     max_epochs=500,
        #     early_stopping=True,
        #     patience=25,
        #     use_batch_norm=False,
        #     random_state=42,
        # )

        # # Variant with slightly higher regularization
        # classifiers["GPU-MLP (Tuned-HighReg)"] = GPUMLPClassifier(
        #     hidden_layer_sizes=(256, 192, 160),
        #     dropout=0.25,
        #     learning_rate=0.000131,
        #     weight_decay=0.001,
        #     batch_size=32,
        #     max_epochs=500,
        #     early_stopping=True,
        #     patience=30,
        #     use_batch_norm=False,
        #     random_state=42,
        # )

        # ========================================================================
        # CNN CLASSIFIERS FOR SPECTROGRAM DATA
        # ========================================================================
        # These classifiers expect 2D spectrogram input (n_mels × time_bins)
        # They work with flattened spectrograms and automatically reshape them

        # SIMPLIFIED CNN: Standard Conv2D for small dataset generalization
        # WHY SIMPLIFIED: Depthwise separable + attention was TOO COMPLEX for 968 samples
        # New approach: Fewer parameters, more regularization, can actually learn
        # classifiers["CNN-Spectrogram"] = SpectrogramCNNClassifier(
        #     input_shape=(80, 128),  # Match spectrogram size
        #     dropout=0.5,  # INCREASED: Strong dropout in simple architecture
        #     learning_rate=0.001,  # INCREASED: Can learn faster with simpler model
        #     weight_decay=0.01,  # INCREASED: Need strong L2 with small dataset
        #     batch_size=16,
        #     max_epochs=200,  # REDUCED: Simpler model converges faster
        #     early_stopping=True,
        #     patience=30,  # REDUCED: Expect faster convergence
        #     random_state=42,
        #     verbose=True,
        # )

        # ADVANCED CNN: Depthwise Separable + Attention (previously achieved 65% training)
        # This was removed but user wants it back to compare
        # MORE COMPLEX: Needs larger datasets (3K+ samples) to shine
        # Use case: When simple CNN plateaus and you have enough data
        # classifiers["CNN-Advanced-Spectrogram"] = SpectrogramCNN_AdvancedClassifier(
        #     input_shape=(80, 128),
        #     dropout=0.4,  # Moderate dropout (complex architecture provides regularization)
        #     learning_rate=0.0003,  # Moderate LR for complex model
        #     weight_decay=0.005,  # Light L2 (attention mechanism helps prevent overfitting)
        #     batch_size=16,
        #     max_epochs=200,
        #     early_stopping=True,
        #     patience=40,  # More patience for complex architecture
        #     random_state=42,
        #     verbose=True,
        # )

        # # CNN + MLP Hybrid: RESTORED to v1 settings that achieved 93.6% train accuracy
        # # Original settings from successful run - keeping capacity for learning
        # classifiers["CNN-MLP-Spectrogram"] = SpectrogramCNN_MLPClassifier(
        #     input_shape=(80, 128),
        #     cnn_channels=[32, 64, 128],
        #     mlp_hidden_dims=[96, 48],  # RESTORED: Original capacity (not [64,32])
        #     dropout=0.6,  # RESTORED: Original dropout (not 0.7)
        #     learning_rate=0.00003,  # RESTORED: Original LR (not 0.00002)
        #     weight_decay=0.02,  # RESTORED: Original L2 (not 0.03)
        #     batch_size=8,
        #     max_epochs=400,  # RESTORED: Original max epochs (not 200)
        #     early_stopping=True,
        #     patience=50,  # RESTORED: Original patience (not 30)
        #     random_state=42,
        #     verbose=True,
        # )

        # RESIDUAL CNN (ResNet-style): DISABLED - Performed poorly (50% random guessing)
        # Issue: Too deep for small dataset, massive validation loss spikes
        # Skip connections didn't help - model couldn't learn at all
        # classifiers["CNN-ResNet-Spectrogram"] = SpectrogramResNetClassifier(
        #     input_shape=(80, 128),
        #     dropout=0.5,
        #     learning_rate=0.0005,
        #     weight_decay=0.01,
        #     batch_size=16,
        #     max_epochs=250,
        #     early_stopping=True,
        #     patience=40,
        #     random_state=42,
        #     verbose=True,
        # )

        # ========================================================================
        # TREE-BASED GRADIENT BOOSTING MODELS
        # ========================================================================
        # Tree-based models often capture different patterns than neural networks
        # Good for non-linear feature interactions and robust to feature scaling

        # # XGBoost - Extreme Gradient Boosting
        # classifiers["XGBoost"] = XGBoostWrapper(
        #     n_estimators=200,
        #     random_state=42,
        # )

        # # LightGBM - Fast gradient boosting (if available)
        # if LIGHTGBM_AVAILABLE:
        #     classifiers["LightGBM"] = LightGBMWrapper(
        #         n_estimators=200,
        #         learning_rate=0.1,
        #         max_depth=-1,  # No depth limit
        #         num_leaves=31,
        #         random_state=42,
        #         verbose=-1,  # Suppress output
        #     )

        # ========================================================================
        # ENSEMBLE OF TOP GENERALIZING MODELS
        # ========================================================================
        # Combine predictions from models with best validation accuracy
        # Expanded to 5 models for better generalization through diversity

        # Soft voting ensemble of top sklearn MLPs
        # classifiers["Ensemble (Top5-MLP)"] = VotingClassifier(
        #     estimators=[
        #         # Model 1: High regularization, medium depth
        #         (
        #             "mlp_highreg",
        #             MLPWrapper(
        #                 hidden_layer_sizes=(128, 64, 32),
        #                 activation="relu",
        #                 solver="adam",
        #                 alpha=0.02,  # High regularization
        #                 batch_size=32,
        #                 learning_rate="adaptive",
        #                 learning_rate_init=0.001,
        #                 max_iter=800,
        #                 early_stopping=True,
        #                 validation_fraction=0.15,
        #                 n_iter_no_change=25,
        #                 random_state=42,
        #             ),
        #         ),
        #         # Model 2: Deeper architecture, moderate regularization
        #         (
        #             "mlp_deep",
        #             MLPWrapper(
        #                 hidden_layer_sizes=(256, 128, 64, 32),
        #                 activation="relu",
        #                 solver="adam",
        #                 alpha=0.005,
        #                 batch_size=32,
        #                 learning_rate="adaptive",
        #                 learning_rate_init=0.001,
        #                 max_iter=1000,
        #                 early_stopping=True,
        #                 validation_fraction=0.15,
        #                 n_iter_no_change=30,
        #                 random_state=43,
        #             ),
        #         ),
        #         # Model 3: Wide shallow architecture (different inductive bias)
        #         (
        #             "mlp_wide_shallow",
        #             MLPWrapper(
        #                 hidden_layer_sizes=(256, 128),
        #                 activation="relu",
        #                 solver="adam",
        #                 alpha=0.015,  # Moderate-high regularization
        #                 batch_size=64,
        #                 learning_rate="adaptive",
        #                 learning_rate_init=0.0005,  # Lower LR for stability
        #                 max_iter=600,
        #                 early_stopping=True,
        #                 validation_fraction=0.15,
        #                 n_iter_no_change=25,
        #                 random_state=44,
        #             ),
        #         ),
        #         # Model 4: Very high regularization (underfits less to training data)
        #         (
        #             "mlp_veryhighreg",
        #             MLPWrapper(
        #                 hidden_layer_sizes=(96, 48, 24),
        #                 activation="relu",
        #                 solver="adam",
        #                 alpha=0.05,  # Very high regularization for generalization
        #                 batch_size=32,
        #                 learning_rate="adaptive",
        #                 learning_rate_init=0.001,
        #                 max_iter=500,
        #                 early_stopping=True,
        #                 validation_fraction=0.15,
        #                 n_iter_no_change=20,
        #                 random_state=45,
        #             ),
        #         ),
        #         # Model 5: Tanh activation (different non-linearity for diversity)
        #         (
        #             "mlp_tanh",
        #             MLPWrapper(
        #                 hidden_layer_sizes=(128, 64, 32),
        #                 activation="tanh",  # Different activation function
        #                 solver="adam",
        #                 alpha=0.01,
        #                 batch_size=32,
        #                 learning_rate="adaptive",
        #                 learning_rate_init=0.001,
        #                 max_iter=700,
        #                 early_stopping=True,
        #                 validation_fraction=0.15,
        #                 n_iter_no_change=25,
        #                 random_state=46,
        #             ),
        #         ),
        #     ],
        #     voting="soft",
        # )

        # # Original Top3 ensemble (best performer in v3 experiments)
        classifiers["Ensemble (Top3-MLP)"] = VotingClassifier(
            estimators=[
                (
                    "mlp_med_highreg",
                    MLPWrapper(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.02,
                        batch_size=32,
                        learning_rate="adaptive",
                        learning_rate_init=0.001,
                        max_iter=800,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                        random_state=42,
                    ),
                ),
                (
                    "mlp_large",
                    MLPWrapper(
                        hidden_layer_sizes=(256, 128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.005,
                        batch_size=32,
                        learning_rate="adaptive",
                        learning_rate_init=0.001,
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=43,
                    ),
                ),
                (
                    "mlp_medium",
                    MLPWrapper(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.01,
                        batch_size=32,
                        learning_rate="adaptive",
                        learning_rate_init=0.001,
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        random_state=44,
                    ),
                ),
            ],
            voting="soft",
        )

        # ========================================================================
        # MEGA-ENSEMBLE: Diverse architectures for better generalization
        # ========================================================================
        # Combines top performers from different model families:
        # - GPU-MLP variants (different regularization/depth) - GPU auto-detected
        # - Tree-based models (capture non-linear patterns)
        # Note: GPUMLPClassifier automatically falls back to CPU if GPU unavailable

        # if GPU_AVAILABLE:
        #     classifiers["MegaEnsemble-Diverse"] = VotingClassifier(
        #         estimators=[
        #             # Best individual model: GPU-MLP with high regularization
        #             (
        #                 "gpu_mlp_highreg_best",
        #                 GPUMLPClassifier(
        #                     hidden_layer_sizes=(128, 64),
        #                     dropout=0.1,
        #                     learning_rate=0.001,
        #                     weight_decay=0.01,
        #                     batch_size=32,
        #                     max_epochs=800,
        #                     early_stopping=True,
        #                     patience=25,
        #                     use_batch_norm=False,
        #                     random_state=42,
        #                 ),
        #             ),
        #             # GPU-MLP with different architecture
        #             (
        #                 "gpu_mlp_tuned",
        #                 GPUMLPClassifier(
        #                     hidden_layer_sizes=(128, 96, 64),
        #                     dropout=0.15,
        #                     learning_rate=0.001,
        #                     weight_decay=0.01,
        #                     batch_size=32,
        #                     max_epochs=500,
        #                     early_stopping=True,
        #                     patience=25,
        #                     use_batch_norm=False,
        #                     random_state=42,
        #                 ),
        #             ),
        #             # Gradient Boosting for tree-based perspective
        #             (
        #                 "gradient_boost",
        #                 GradientBoostingClassifier(
        #                     n_estimators=150,
        #                     max_depth=5,
        #                     learning_rate=0.1,
        #                     subsample=0.8,
        #                     min_samples_split=10,
        #                     random_state=42,
        #                 ),
        #             ),
        #             # XGBoost for different boosting approach
        #             (
        #                 "xgboost",
        #                 XGBoostWrapper(
        #                     n_estimators=200,
        #                     random_state=42,
        #                 ),
        #             ),
        #             # Large GPU-MLP with lower regularization
        #             (
        #                 "gpu_mlp_large_lowreg",
        #                 GPUMLPClassifier(
        #                     hidden_layer_sizes=(256, 128, 64),
        #                     dropout=0.05,
        #                     learning_rate=0.001,
        #                     weight_decay=0.001,
        #                     batch_size=32,
        #                     max_epochs=1000,
        #                     early_stopping=True,
        #                     patience=30,
        #                     validation_fraction=0.15,
        #                     use_batch_norm=False,
        #                     random_state=43,
        #                 ),
        #             ),
        #             # Robust GPU-MLP (different feature preprocessing)
        #             (
        #                 "gpu_mlp_robust",
        #                 NormalizedClassifierWrapper(
        #                     base_classifier=GPUMLPClassifier(
        #                         hidden_layer_sizes=(128, 64),
        #                         dropout=0.1,
        #                         learning_rate=0.001,
        #                         weight_decay=0.005,
        #                         batch_size=32,
        #                         max_epochs=500,
        #                         early_stopping=True,
        #                         patience=20,
        #                         use_batch_norm=False,
        #                         random_state=44,
        #                     ),
        #                     normalization="robust",
        #                 ),
        #             ),
        #         ],
        #         voting="soft",
        #     )

        #     # Compact mega-ensemble with just the top 3 diverse models
        #     classifiers["MegaEnsemble-Top3"] = VotingClassifier(
        #         estimators=[
        #             # Best GPU-MLP (high regularization)
        #             (
        #                 "gpu_mlp_best",
        #                 GPUMLPClassifier(
        #                     hidden_layer_sizes=(128, 64),
        #                     dropout=0.1,
        #                     learning_rate=0.001,
        #                     weight_decay=0.01,
        #                     batch_size=32,
        #                     max_epochs=800,
        #                     early_stopping=True,
        #                     patience=25,
        #                     use_batch_norm=False,
        #                     random_state=42,
        #                 ),
        #             ),
        #             # GPU-MLP tuned architecture
        #             (
        #                 "gpu_mlp_tuned_best",
        #                 GPUMLPClassifier(
        #                     hidden_layer_sizes=(128, 96, 64),
        #                     dropout=0.15,
        #                     learning_rate=0.001,
        #                     weight_decay=0.01,
        #                     batch_size=32,
        #                     max_epochs=500,
        #                     early_stopping=True,
        #                     patience=25,
        #                     use_batch_norm=False,
        #                     random_state=43,
        #                 ),
        #             ),
        #             # Best tree model for diversity
        #             (
        #                 "gradient_boost_best",
        #                 GradientBoostingClassifier(
        #                     n_estimators=150,
        #                     max_depth=5,
        #                     learning_rate=0.1,
        #                     subsample=0.8,
        #                     min_samples_split=10,
        #                     random_state=42,
        #                 ),
        #             ),
        #         ],
        #         voting="soft",
        #     )

        return classifiers

    def _save_top_models(
        self,
        trained_classifiers: dict,
        results_dict: dict,
        scaler,
        classes: list,
        metric_key: str = "validation_accuracy",
        top_n: int = 3,
    ):
        """
        Save top N performing trained models for use in reconstruction.

        Args:
            trained_classifiers: Dict of classifier_name -> trained classifier object
            results_dict: Dict of classifier_name -> performance metrics
            scaler: The fitted StandardScaler used for training
            classes: List of class labels
            metric_key: Which metric to use for ranking (e.g., "validation_accuracy")
            top_n: Number of top models to save (default 3)
        """
        import pickle

        # Create models output directory
        models_dir = os.path.join(self.experiment_output_dir, "trained_models")
        os.makedirs(models_dir, exist_ok=True)

        # Filter to only classifiers that were successfully trained
        valid_classifiers = {
            name: results_dict[name]
            for name in trained_classifiers.keys()
            if name in results_dict and metric_key in results_dict[name]
        }

        if not valid_classifiers:
            self.logger.warning("No valid classifiers to save")
            return

        # Sort by metric (descending) and take top N
        sorted_classifiers = sorted(
            valid_classifiers.items(), key=lambda x: x[1][metric_key], reverse=True
        )

        # Take top N (or all if less than N)
        top_classifiers = sorted_classifiers[: min(top_n, len(sorted_classifiers))]

        self.logger.info(
            f"\n💾 Saving top {len(top_classifiers)} models for reconstruction..."
        )

        saved_models_info = []

        for rank, (clf_name, metrics) in enumerate(top_classifiers, 1):
            clf = trained_classifiers[clf_name]
            accuracy = metrics[metric_key]

            # Create model data bundle
            model_data = {
                "model": clf,
                "scaler": scaler,
                "classes": classes,
                "classifier_name": clf_name,
                "rank": rank,
                "metrics": {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in metrics.items()
                    if not k.endswith("_predictions") and not k.endswith("_stats")
                },
            }

            # Save model file
            safe_name = clf_name.replace(" ", "_").replace("/", "_").lower()
            model_filename = f"model_rank{rank}_{safe_name}.pkl"
            model_path = os.path.join(models_dir, model_filename)

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            saved_models_info.append(
                {
                    "rank": rank,
                    "classifier": clf_name,
                    "accuracy": float(accuracy),
                    "filename": model_filename,
                }
            )

            self.logger.info(
                f"  #{rank}: {clf_name} ({metric_key}: {accuracy:.4f}) → {model_filename}"
            )

        # Save summary of saved models
        summary = {
            "top_n": top_n,
            "metric_used": metric_key,
            "models_saved": len(saved_models_info),
            "models": saved_models_info,
        }

        summary_path = os.path.join(models_dir, "saved_models_summary.json")
        import json

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"  📋 Summary saved to: {summary_path}")
        self.logger.info(f"✅ Models saved to: {models_dir}/")

    def _save_validation_results(
        self,
        results: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scaler,
    ):
        """Save validation mode results to disk (with cross-validation)."""
        import json
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # Create output directory
        output_dir = os.path.join(self.experiment_output_dir, "validation_results")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare serializable results
        classifier_results = results["classifier_results"]
        serializable_results = {
            "validation_mode": True,
            "cross_validation_enabled": results.get("cross_validation_enabled", False),
            "cv_folds": results.get("cv_folds", 5),
            "training_datasets": results["training_datasets"],
            "validation_datasets": results["validation_datasets"],
            "num_train_samples": results["num_train_samples"],
            "num_val_samples": results["num_val_samples"],
            "best_classifier": results["best_classifier"],
            "classifier_performance": {},
        }

        # Store performance for each classifier
        for clf_name, clf_results in classifier_results.items():
            perf_dict = {
                "train_accuracy": float(clf_results["train_accuracy"]),
                "train_f1": float(clf_results["train_f1"]),
                "validation_accuracy": float(clf_results["validation_accuracy"]),
                "validation_f1": float(clf_results["validation_f1"]),
            }
            # Add CV metrics if available
            if "cv_test_accuracy" in clf_results:
                perf_dict.update(
                    {
                        "cv_test_accuracy": float(clf_results["cv_test_accuracy"]),
                        "cv_test_accuracy_std": float(
                            clf_results["cv_test_accuracy_std"]
                        ),
                        "cv_test_f1": float(clf_results["cv_test_f1"]),
                        "cv_test_f1_std": float(clf_results["cv_test_f1_std"]),
                    }
                )
            serializable_results["classifier_performance"][clf_name] = perf_dict

        # Save JSON results
        results_path = os.path.join(output_dir, "discrimination_summary.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to: {results_path}")

        # Create performance comparison plot
        self._create_validation_performance_plot(classifier_results, output_dir)

        # Create confusion matrices for best classifier
        best_clf_name = results["best_classifier"]["name"]
        self._create_validation_confusion_matrices(
            classifier_results[best_clf_name],
            y_train,  # Use full training set
            y_val,
            best_clf_name,
            output_dir,
        )

    def _save_3way_split_results(
        self,
        results: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_tuning: np.ndarray,
        y_tuning: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler,
    ):
        """Save 3-way split mode results to disk."""
        import json
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # Create output directory
        output_dir = os.path.join(self.experiment_output_dir, "3way_split_results")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare serializable results
        classifier_results = results["classifier_results"]
        serializable_results = {
            "split_mode": "3-way",
            "training_datasets": results["training_datasets"],
            "tuning_datasets": results["tuning_datasets"],
            "final_test_datasets": results["final_test_datasets"],
            "num_train_samples": results["num_train_samples"],
            "num_tuning_samples": results["num_tuning_samples"],
            "num_test_samples": results["num_test_samples"],
            "best_classifier": results["best_classifier"],
            "classifier_performance": {},
        }

        # Store performance for each classifier
        for clf_name, clf_results in classifier_results.items():
            serializable_results["classifier_performance"][clf_name] = {
                "train_accuracy": float(clf_results["train_accuracy"]),
                "train_f1": float(clf_results["train_f1"]),
                "tuning_accuracy": float(clf_results["tuning_accuracy"]),
                "tuning_f1": float(clf_results["tuning_f1"]),
                "test_accuracy": float(clf_results["test_accuracy"]),
                "test_f1": float(clf_results["test_f1"]),
            }

        # Save JSON results
        results_path = os.path.join(output_dir, "discrimination_summary.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"3-way split results saved to: {results_path}")

        # Create performance comparison plot
        self._create_3way_performance_plot(classifier_results, output_dir)

        # Create confusion matrices for best classifier
        best_clf_name = results["best_classifier"]["name"]
        self._create_3way_confusion_matrices(
            classifier_results[best_clf_name],
            y_tuning,
            y_test,
            best_clf_name,
            output_dir,
        )

    def _create_validation_performance_plot(
        self, classifier_results: dict, output_dir: str
    ):
        """Create bar plot comparing CV test and validation performance."""
        import matplotlib.pyplot as plt

        clf_names = list(classifier_results.keys())
        train_accs = [r["train_accuracy"] for r in classifier_results.values()]
        # Use CV test accuracy instead of single test split
        cv_test_accs = [r["cv_test_accuracy"] for r in classifier_results.values()]
        cv_test_stds = [
            r.get("cv_test_accuracy_std", 0) for r in classifier_results.values()
        ]
        val_accs = [r["validation_accuracy"] for r in classifier_results.values()]

        fig, ax = plt.subplots(figsize=(16, 8))
        x = np.arange(len(clf_names))
        width = 0.25

        ax.bar(
            x - width,
            train_accs,
            width,
            label="Train Accuracy",
            alpha=0.7,
            color="lightblue",
        )
        # Plot CV test accuracy with error bars
        ax.bar(
            x,
            cv_test_accs,
            width,
            label="CV Test Accuracy (5-fold)",
            alpha=0.8,
            color="orange",
            yerr=cv_test_stds,
            capsize=5,
        )
        ax.bar(
            x + width,
            val_accs,
            width,
            label="Validation Accuracy",
            alpha=0.8,
            color="green",
        )

        ax.set_ylabel("Accuracy")
        ax.set_title("Classifier Performance: Train vs CV Test vs Validation")
        ax.set_xticks(x)
        ax.set_xticklabels(clf_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "classifier_performance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Performance plot saved to: {plot_path}")

    def _create_3way_performance_plot(self, classifier_results: dict, output_dir: str):
        """Create bar plot comparing train, tuning, and test performance."""
        import matplotlib.pyplot as plt

        clf_names = list(classifier_results.keys())
        train_accs = [r["train_accuracy"] for r in classifier_results.values()]
        tuning_accs = [r["tuning_accuracy"] for r in classifier_results.values()]
        test_accs = [r["test_accuracy"] for r in classifier_results.values()]

        fig, ax = plt.subplots(figsize=(18, 8))
        x = np.arange(len(clf_names))
        width = 0.25

        ax.bar(
            x - width,
            train_accs,
            width,
            label="Train Accuracy",
            alpha=0.7,
            color="lightblue",
        )
        ax.bar(
            x,
            tuning_accs,
            width,
            label="Tuning Accuracy (Hyperparameter Selection)",
            alpha=0.8,
            color="orange",
        )
        ax.bar(
            x + width,
            test_accs,
            width,
            label="Test Accuracy (Final Evaluation)",
            alpha=0.8,
            color="green",
        )

        ax.set_ylabel("Accuracy")
        ax.set_title("Classifier Performance: 3-Way Split (Train / Tuning / Test)")
        ax.set_xticks(x)
        ax.set_xticklabels(clf_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add horizontal line at best tuning accuracy
        best_tuning_acc = max(tuning_accs)
        ax.axhline(
            y=best_tuning_acc,
            color="r",
            linestyle="--",
            alpha=0.3,
            label=f"Best Tuning Acc: {best_tuning_acc:.3f}",
        )

        plt.tight_layout()

        plot_path = os.path.join(output_dir, "classifier_performance_3way.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"3-way performance plot saved to: {plot_path}")

    def _create_3way_confusion_matrices(
        self,
        clf_results: dict,
        y_tuning: np.ndarray,
        y_test: np.ndarray,
        clf_name: str,
        output_dir: str,
    ):
        """Create confusion matrices for tuning and test sets."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Tuning set confusion matrix
        cm_tuning = confusion_matrix(y_tuning, clf_results["tuning_predictions"])
        sns.heatmap(
            cm_tuning,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            cbar=True,
        )
        axes[0].set_title(f"{clf_name}\nTuning Set Confusion Matrix")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        # Test set confusion matrix
        cm_test = confusion_matrix(y_test, clf_results["test_predictions"])
        sns.heatmap(
            cm_test,
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=axes[1],
            cbar=True,
        )
        axes[1].set_title(f"{clf_name}\nTest Set Confusion Matrix")
        axes[1].set_ylabel("True Label")
        axes[1].set_xlabel("Predicted Label")

        plt.tight_layout()

        cm_path = os.path.join(output_dir, f"confusion_matrices_3way_{clf_name}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"3-way confusion matrices saved to: {cm_path}")

    def _create_validation_confusion_matrices(
        self,
        clf_results: dict,
        y_train: np.ndarray,  # Training labels for CV confusion matrix
        y_val: np.ndarray,  # Validation labels
        clf_name: str,
        output_dir: str,
    ):
        """Create confusion matrices for CV test folds and validation set (side-by-side)."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # DEBUG: Log label information
        self.logger.info("=" * 80)
        self.logger.info(f"📊 DEBUG - CONFUSION MATRIX DATA for {clf_name}:")
        self.logger.info(f"    Training set (for CV):")
        self.logger.info(
            f"      Unique classes in y_train: {sorted(list(set(y_train)))}"
        )
        self.logger.info(
            f"      Distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
        )
        self.logger.info(f"    Validation set:")
        self.logger.info(f"      Unique classes in y_val: {sorted(list(set(y_val)))}")
        self.logger.info(
            f"      Distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}"
        )

        # Check if we have CV predictions
        has_cv_preds = "cv_predictions" in clf_results
        self.logger.info(f"    Has CV predictions: {has_cv_preds}")
        if has_cv_preds:
            self.logger.info(
                f"      CV predictions shape: {len(clf_results['cv_predictions'])}"
            )
        self.logger.info("=" * 80)

        # Get unique labels from actual data
        train_labels = sorted(list(set(y_train)))
        val_labels = sorted(list(set(y_val)))

        # Create side-by-side confusion matrices if we have both CV and validation predictions
        if has_cv_preds and "validation_predictions" in clf_results:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # LEFT: CV Test Confusion Matrix (aggregated across folds)
            y_train_cv_pred = clf_results["cv_predictions"]
            cm_cv = confusion_matrix(y_train, y_train_cv_pred, labels=train_labels)

            self.logger.info(f"📊 CV test confusion matrix shape: {cm_cv.shape}")
            self.logger.info(f"    Labels used: {train_labels}")

            sns.heatmap(
                cm_cv,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[0],
                xticklabels=train_labels,
                yticklabels=train_labels,
                cbar=True,
            )
            axes[0].set_title(
                f"{clf_name}\nCross-Validation Test (5-Fold Aggregated)",
                fontsize=12,
                fontweight="bold",
            )
            axes[0].set_ylabel("True Label", fontsize=11)
            axes[0].set_xlabel("Predicted Label", fontsize=11)

            # RIGHT: Validation Confusion Matrix (hold-out set)
            y_val_pred = clf_results["validation_predictions"]
            cm_val = confusion_matrix(y_val, y_val_pred, labels=val_labels)

            self.logger.info(f"📊 Validation confusion matrix shape: {cm_val.shape}")
            self.logger.info(f"    Labels used: {val_labels}")

            sns.heatmap(
                cm_val,
                annot=True,
                fmt="d",
                cmap="Greens",
                ax=axes[1],
                xticklabels=val_labels,
                yticklabels=val_labels,
                cbar=True,
            )
            axes[1].set_title(
                f"{clf_name}\nValidation Set (Hold-out)", fontsize=12, fontweight="bold"
            )
            axes[1].set_ylabel("True Label", fontsize=11)
            axes[1].set_xlabel("Predicted Label", fontsize=11)

            plt.suptitle(
                "Confusion Matrix Comparison: CV Test vs Validation",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()
            plot_path = os.path.join(
                output_dir, "confusion_matrix_cv_vs_validation.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(
                f"CV + Validation confusion matrices saved to: {plot_path}"
            )

        # Also save individual validation confusion matrix (for backwards compatibility)
        if "validation_predictions" in clf_results:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            # Validation confusion matrix
            y_val_pred = clf_results["validation_predictions"]
            cm_val = confusion_matrix(y_val, y_val_pred, labels=val_labels)

            sns.heatmap(
                cm_val,
                annot=True,
                fmt="d",
                cmap="Greens",
                ax=ax,
                xticklabels=val_labels,
                yticklabels=val_labels,
            )
            ax.set_title(f"{clf_name} - Validation Set (Hold-out)")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")

            plt.tight_layout()
            plot_path = os.path.join(output_dir, "confusion_matrix_validation.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Validation confusion matrix saved to: {plot_path}")

        self.logger.info("=" * 80)

    def _save_batch_discrimination_results(self, batch_results: dict, batch_name: str):
        import json

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

        # DEBUG: Log batch data
        self.logger.info("=" * 80)
        self.logger.info(f"📊 DEBUG - BATCH CONFUSION MATRIX DATA for '{batch_name}':")
        self.logger.info(f"    Unique classes in y: {sorted(list(set(y)))}")
        self.logger.info(
            f"    Distribution: {dict(zip(*np.unique(y, return_counts=True)))}"
        )
        self.logger.info("=" * 80)

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

        # Get unique labels from data
        unique_labels = sorted(list(set(y)))
        self.logger.info(f"📊 Creating confusion matrices with labels: {unique_labels}")

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

                # Convert to numpy arrays to ensure proper handling
                y_true_all = np.array(y_true_all)
                y_pred_all = np.array(y_pred_all)

                # Create confusion matrix with explicit labels
                cm = confusion_matrix(y_true_all, y_pred_all, labels=unique_labels)

                # Normalize confusion matrix
                cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

                # Clean up labels by removing "finger" prefix for better readability
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
