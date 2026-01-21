"""
GPU-accelerated PyTorch classifiers for acoustic sensing.

This module provides PyTorch-based neural network classifiers that leverage
GPU acceleration for faster training compared to sklearn's MLPClassifier.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional, List
import warnings


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


class AcousticMLP(nn.Module):
    """
    Multi-layer perceptron for acoustic classification.

    Designed to work well with acoustic features and provide
    better generalization through:
    - Batch normalization
    - Dropout regularization
    - Residual connections (optional)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


class GPUMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    GPU-accelerated MLP classifier using PyTorch.

    Provides sklearn-compatible interface for easy integration with
    existing pipeline while leveraging GPU for 5-10x faster training.

    Features:
    - Automatic GPU detection and usage
    - Early stopping to prevent overfitting
    - Batch normalization and dropout for regularization
    - Label encoding for string labels
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,  # L2 regularization
        batch_size: int = 64,
        max_epochs: int = 500,
        early_stopping: bool = True,
        patience: int = 20,
        validation_fraction: float = 0.15,
        use_batch_norm: bool = True,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.use_batch_norm = use_batch_norm
        self.random_state = random_state
        self.verbose = verbose

        self.device = get_device()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPUMLPClassifier":
        """Fit the model to training data."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        num_classes = len(self.classes_)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y_encoded)

        # Split for validation if early stopping
        if self.early_stopping:
            n_val = int(len(X) * self.validation_fraction)
            indices = np.random.permutation(len(X))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train = X_tensor[train_indices]
            y_train = y_tensor[train_indices]
            X_val = X_tensor[val_indices]
            y_val = y_tensor[val_indices]
        else:
            X_train, y_train = X_tensor, y_tensor
            X_val, y_val = None, None

        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Initialize model
        input_dim = X.shape[1]
        self.model = AcousticMLP(
            input_dim=input_dim,
            hidden_dims=list(self.hidden_layer_sizes),
            num_classes=num_classes,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            if self.early_stopping and X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_device = X_val.to(self.device)
                    y_val_device = y_val.to(self.device)
                    val_outputs = self.model(X_val_device)
                    val_loss = criterion(val_outputs, y_val_device).item()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(
                        f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        y_pred = self.label_encoder.inverse_transform(predicted.cpu().numpy())
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()


class NormalizedClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Classifier wrapper that applies per-feature normalization for better
    workspace generalization.

    This addresses the issue where tree models overfit to absolute feature
    values that differ between workspaces. By normalizing using robust
    statistics (median, IQR) instead of mean/std, outliers have less impact.

    Additional normalization strategies:
    - 'standard': Standard z-score normalization (mean=0, std=1)
    - 'robust': Robust scaling using median and IQR (default)
    - 'relative': Compute relative features (ratios between feature groups)
    - 'quantile': Transform to uniform distribution
    """

    def __init__(
        self,
        base_classifier,
        normalization: str = "robust",
        clip_outliers: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        self.base_classifier = base_classifier
        self.normalization = normalization
        self.clip_outliers = clip_outliers
        self.clip_range = clip_range

        self.median_ = None
        self.iqr_ = None
        self.mean_ = None
        self.std_ = None
        self.quantile_transformer = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NormalizedClassifierWrapper":
        """Fit the normalizer and base classifier."""
        X_normalized = self._fit_normalize(X)
        self.base_classifier.fit(X_normalized, y)
        return self

    def _fit_normalize(self, X: np.ndarray) -> np.ndarray:
        """Fit normalization parameters and transform data."""
        if self.normalization == "robust":
            self.median_ = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self.iqr_ = q75 - q25
            self.iqr_[self.iqr_ == 0] = 1.0  # Avoid division by zero

            X_normalized = (X - self.median_) / self.iqr_

        elif self.normalization == "standard":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1.0

            X_normalized = (X - self.mean_) / self.std_

        elif self.normalization == "quantile":
            from sklearn.preprocessing import QuantileTransformer

            self.quantile_transformer = QuantileTransformer(
                output_distribution="normal", random_state=42
            )
            X_normalized = self.quantile_transformer.fit_transform(X)

        else:
            X_normalized = X.copy()

        # Clip outliers
        if self.clip_outliers:
            X_normalized = np.clip(X_normalized, self.clip_range[0], self.clip_range[1])

        return X_normalized

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.normalization == "robust":
            X_normalized = (X - self.median_) / self.iqr_

        elif self.normalization == "standard":
            X_normalized = (X - self.mean_) / self.std_

        elif self.normalization == "quantile":
            X_normalized = self.quantile_transformer.transform(X)

        else:
            X_normalized = X.copy()

        if self.clip_outliers:
            X_normalized = np.clip(X_normalized, self.clip_range[0], self.clip_range[1])

        return X_normalized

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using normalized features."""
        X_normalized = self._transform(X)
        return self.base_classifier.predict(X_normalized)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using normalized features."""
        X_normalized = self._transform(X)
        return self.base_classifier.predict_proba(X_normalized)

    def get_params(self, deep: bool = True):
        """Get parameters for sklearn compatibility."""
        return {
            "base_classifier": self.base_classifier,
            "normalization": self.normalization,
            "clip_outliers": self.clip_outliers,
            "clip_range": self.clip_range,
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RelativeFeatureClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that uses relative features (ratios) instead of absolute values.

    This can help with workspace invariance by focusing on relationships
    between features rather than their absolute magnitudes.

    For example:
    - Ratio of high-freq to low-freq energy
    - Ratio of envelope features to spectral features
    - Normalized resonance magnitudes
    """

    def __init__(
        self,
        base_classifier,
        feature_groups: Optional[dict] = None,
        add_original: bool = True,
    ):
        """
        Args:
            base_classifier: The underlying classifier
            feature_groups: Dict mapping group names to feature indices
                e.g., {'spectral': [0,1,2], 'temporal': [3,4,5]}
            add_original: Whether to include original features
        """
        self.base_classifier = base_classifier
        self.feature_groups = feature_groups
        self.add_original = add_original
        self.scaler = StandardScaler()

    def _compute_relative_features(self, X: np.ndarray) -> np.ndarray:
        """Compute relative features from input."""
        relative_features = []

        # If no feature groups specified, create automatic ratios
        if self.feature_groups is None:
            n_features = X.shape[1]
            # Split features into thirds and compute ratios
            third = n_features // 3

            # Avoid division by zero
            eps = 1e-8

            # Ratio of first third to second third (mean of each)
            first_mean = np.mean(X[:, :third], axis=1, keepdims=True) + eps
            second_mean = np.mean(X[:, third : 2 * third], axis=1, keepdims=True) + eps
            third_mean = np.mean(X[:, 2 * third :], axis=1, keepdims=True) + eps

            relative_features.append(first_mean / second_mean)
            relative_features.append(second_mean / third_mean)
            relative_features.append(first_mean / third_mean)

            # Feature-to-mean ratios (normalized features)
            overall_mean = np.mean(X, axis=1, keepdims=True) + eps
            relative_features.append(X / overall_mean)

            # Standard deviation normalized
            overall_std = np.std(X, axis=1, keepdims=True) + eps
            relative_features.append(
                (X - np.mean(X, axis=1, keepdims=True)) / overall_std
            )

        if self.add_original:
            relative_features.insert(0, X)

        return np.hstack(relative_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RelativeFeatureClassifier":
        """Fit on relative features."""
        X_relative = self._compute_relative_features(X)
        X_scaled = self.scaler.fit_transform(X_relative)
        self.base_classifier.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using relative features."""
        X_relative = self._compute_relative_features(X)
        X_scaled = self.scaler.transform(X_relative)
        return self.base_classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_relative = self._compute_relative_features(X)
        X_scaled = self.scaler.transform(X_relative)
        return self.base_classifier.predict_proba(X_scaled)


# Convenience function to check GPU availability
def print_gpu_info():
    """Print GPU information for debugging."""
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
