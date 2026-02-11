"""
GPU-accelerated PyTorch classifiers for acoustic sensing.

This module provides PyTorch-based neural network classifiers that leverage
GPU acceleration for faster training compared to sklearn's MLPClassifier.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List
import warnings

# Suppress CUDA initialization warnings
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass  # Silently fallback to CPU if CUDA fails

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
    except Exception:
        pass

    return torch.device("cpu")


class NumpyDataset(Dataset):
    """Memory-efficient dataset that keeps data in numpy arrays until needed.

    This avoids creating large PyTorch tensors in CPU/GPU memory upfront.
    Instead, data is converted to tensors batch-by-batch during training.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        add_channel_dim: bool = True,
        augment: bool = False,
    ):
        self.X = X
        self.y = y
        self.add_channel_dim = add_channel_dim
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment: time and frequency masking to prevent overfitting.

        This forces the model to learn robust features instead of memorizing
        specific surface patterns.
        """
        if not self.augment:
            return spec

        # spec shape: (n_mels, time_bins) or (1, n_mels, time_bins)
        if spec.ndim == 3:
            spec = spec.squeeze(0)

        n_mels, time_bins = spec.shape

        # Frequency masking: mask random frequency bands
        if np.random.random() < 0.5:
            f_mask_size = np.random.randint(1, max(2, n_mels // 8))
            f_start = np.random.randint(0, n_mels - f_mask_size)
            spec[f_start : f_start + f_mask_size, :] = 0

        # Time masking: mask random time windows
        if np.random.random() < 0.5:
            t_mask_size = np.random.randint(1, max(2, time_bins // 8))
            t_start = np.random.randint(0, time_bins - t_mask_size)
            spec[:, t_start : t_start + t_mask_size] = 0

        # Add noise: slight Gaussian noise to prevent exact memorization
        if np.random.random() < 0.3:
            noise = torch.randn_like(spec) * 0.01
            spec = spec + noise

        if self.add_channel_dim:
            spec = spec.unsqueeze(0)

        return spec

    def __getitem__(self, idx):
        # Convert to tensor only when requested (batch-by-batch)
        x = torch.FloatTensor(self.X[idx])

        # Apply augmentation during training
        if self.augment:
            x = self._augment_spectrogram(x)
        elif self.add_channel_dim and x.ndim == 2:  # (H, W) -> (1, H, W)
            x = x.unsqueeze(0)

        y = torch.LongTensor([self.y[idx]])[0]
        return x, y


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

    _estimator_type = "classifier"  # Required for VotingClassifier compatibility

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

    def __sklearn_tags__(self):
        """Return sklearn tags for this estimator (required for sklearn 1.6+)."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    @property
    def _estimator_type(self):
        """Return estimator type for sklearn compatibility."""
        return "classifier"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "GPUMLPClassifier":
        """
        Fit the model to training data.

        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation features (preferred over internal split)
            y_val: Optional validation labels

        Returns:
            Fitted classifier

        Note:
            If X_val and y_val are provided, they will be used for validation
            instead of splitting the training data. This is preferred for
            3-way split pipelines where dedicated tuning data exists.
        """
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

        # Use external validation data if provided (3-way split mode)
        if X_val is not None and y_val is not None:
            if self.verbose:
                print(
                    f"Using external validation data: {len(X_val)} samples (3-way split)"
                )
            X_train = X_tensor
            y_train = y_tensor
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val_encoded)

        # Split for validation if early stopping and no external validation
        elif self.early_stopping:
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


# ============================================================================
# CNN Classifiers for Spectrogram Data
# ============================================================================


class SpectrogramCNN(nn.Module):
    """
    Simplified CNN for acoustic contact detection on small datasets.

    DESIGN PHILOSOPHY:
    - SIMPLE standard convolutions (not depthwise separable - too complex for 968 samples)
    - SMALL capacity (avoid overfitting on limited data)
    - STRONG regularization (dropout + weight decay)

    Architecture optimized for:
    - Small dataset generalization (< 1000 samples)
    - Contact vs no-contact binary classification
    - Cross-surface generalization

    Architecture:
        Conv(16) → Pool → Conv(32) → Pool → Conv(64) → GlobalPool → Dense(64) → Output

    Rationale:
    - 3 conv layers: Shallow enough to not overfit
    - Small channels (16→32→64): Fewer parameters = better generalization
    - Single FC layer (64-dim): Minimal capacity in classifier
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),  # (n_mels, time_bins)
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        n_mels, time_bins = input_shape

        # Conv Block 1: Low-level features (simple edges, onsets)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)  # Light dropout early

        # Conv Block 2: Mid-level features (frequency patterns)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)  # Moderate dropout

        # Conv Block 3: High-level features (contact signatures)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.4)  # Heavy dropout

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_fc = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)  # DIRECT to output (no hidden layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, n_mels, time_bins)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # → (batch, 64)
        x = self.dropout_fc(x)
        x = self.fc(x)

        return x


class SpectrogramCNN_MLP(nn.Module):
    """
    CNN + MLP Hybrid architecture for spectrogram classification.

    Uses CNN layers to extract spatial features from spectrograms,
    then feeds them through MLP layers for final classification.

    This architecture combines:
    - CNN: Learns local time-frequency patterns
    - MLP: Learns global decision boundaries

    Architecture:
        Conv2D(32) → Pool → Conv2D(64) → Pool → Conv2D(128) → Pool
        → Flatten → Dense(512) → Dropout → Dense(256) → Dropout → Output
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),  # (n_mels, time_bins)
        num_classes: int = 2,
        cnn_channels: List[int] = [32, 64, 128],
        mlp_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5,
    ):
        super().__init__()

        n_mels, time_bins = input_shape

        # CNN Feature Extractor
        cnn_layers = []
        in_channels = 1

        for out_channels in cnn_channels:
            cnn_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(0.25),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate flattened size after CNN
        # Each pool divides by 2, so len(cnn_channels) pools divide by 2^len
        num_pools = len(cnn_channels)
        pooled_mels = n_mels // (2**num_pools)
        pooled_time = time_bins // (2**num_pools)
        flattened_size = cnn_channels[-1] * pooled_mels * pooled_time

        # MLP Classifier
        mlp_layers = []
        prev_dim = flattened_size

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        mlp_layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, n_mels, time_bins)

        # CNN feature extraction
        x = self.cnn(x)

        # Flatten for MLP
        x = x.view(x.size(0), -1)

        # MLP classification
        x = self.mlp(x)

        return x


class SpectrogramCNN_Advanced(nn.Module):
    """
    Advanced CNN with Depthwise Separable Convolutions + Channel Attention.

    DESIGN PHILOSOPHY:
    - DEPTHWISE SEPARABLE convolutions for efficient parameter usage
    - CHANNEL ATTENTION to focus on important frequency bands
    - HIGHER CAPACITY for larger datasets (3K+ samples)

    This architecture achieved 65% training accuracy on cutout-only datasets.
    More complex than simple CNN - needs more data to shine.

    Architecture optimized for:
    - High-frequency transients (contact impact)
    - Learning which frequency bands are important
    - Larger datasets (3K+ samples)

    Architecture:
        DepthConv(32) → SepConv(64) → SepConv(128)
        → Channel Attention → GlobalPool → Dense(128) → Output

    Use when:
    - You have 3K+ training samples
    - Simple CNN plateaus and you need more capacity
    - You want model to learn frequency importance
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),  # (n_mels, time_bins)
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        n_mels, time_bins = input_shape

        # Depthwise Separable Conv Block 1: Learn frequency patterns
        self.depthwise1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1)
        self.pointwise1 = nn.Conv2d(1, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Depthwise Separable Conv Block 2: Learn time-frequency interactions
        self.depthwise2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise2 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Depthwise Separable Conv Block 3: High-level features
        self.depthwise3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.pointwise3 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Channel Attention: Focus on important frequency bands
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128 // 4, 128, kernel_size=1),
            nn.Sigmoid(),
        )

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, n_mels, time_bins)

        # Block 1: Initial feature extraction
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)  # → (batch, 32, n_mels/2, time/2)

        # Block 2: Mid-level features
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)  # → (batch, 64, n_mels/4, time/4)

        # Block 3: High-level features
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)  # → (batch, 128, n_mels/8, time/8)

        # Apply channel attention (focus on important frequencies)
        attention = self.channel_attention(x)
        x = x * attention

        # Global pooling and classification
        x = self.global_pool(x)  # → (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # → (batch, 128)

        # Dense layers with batch norm
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ------------------------------------------------------------------------------
# Residual CNN (ResNet-style) for Spectrograms
# ------------------------------------------------------------------------------
class SpectrogramResNet(nn.Module):
    """
    Residual CNN for spectrogram classification (ResNet-style).
    Designed for audio: learns hierarchical time-frequency patterns, avoids vanishing gradients.
    Architecture: Conv2d(32) → Conv2d(32) + skip → Conv2d(64) → Conv2d(64) + skip → Conv2d(128) → GlobalPool → Dense(64) → Output
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        n_mels, time_bins = input_shape
        # Initial conv
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Residual block 1
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # Downsample
        self.pool1 = nn.MaxPool2d(2, 2)
        # Residual block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Residual block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Initial conv
        x = torch.relu(self.bn1(self.conv1(x)))
        # Residual block 1
        identity = x
        out = torch.relu(self.bn2(self.conv2(x)))
        x = out + identity
        x = self.pool1(x)
        # Residual block 2
        identity = torch.relu(self.bn3(self.conv3(x)))
        out = torch.relu(self.bn4(self.conv4(identity)))
        x = out + identity
        x = self.pool2(x)
        # Residual block 3
        identity = torch.relu(self.bn5(self.conv5(x)))
        out = torch.relu(self.bn6(self.conv6(identity)))
        x = out + identity
        x = self.pool3(x)
        # Global pool and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Sklearn-compatible wrapper for Residual CNN
class SpectrogramResNetClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        max_epochs: int = 200,
        early_stopping: bool = True,
        patience: int = 30,
        validation_fraction: float = 0.15,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.input_shape = input_shape
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.device = get_device()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    @property
    def _estimator_type(self):
        return "classifier"

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        expected_size = self.input_shape[0] * self.input_shape[1]
        if X.ndim == 2 and X.shape[1] == expected_size:
            return X.reshape(-1, self.input_shape[0], self.input_shape[1])
        elif X.ndim == 3:
            return X
        else:
            raise ValueError(
                f"Input shape {X.shape} doesn't match expected flattened {expected_size} or 2D {self.input_shape}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        X_reshaped = self._reshape_input(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped,
            y_encoded,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y_encoded,
        )
        num_classes = len(self.classes_)
        self.model = SpectrogramResNet(
            input_shape=self.input_shape, num_classes=num_classes, dropout=self.dropout
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_dataset = NumpyDataset(
            X_train, y_train, add_channel_dim=True, augment=True
        )
        val_dataset = NumpyDataset(X_val, y_val, add_channel_dim=True, augment=False)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.max_epochs):
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
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            val_loss /= len(val_loader)
            val_acc = correct / total
            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            if self.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_reshaped = self._reshape_input(X)
        dataset = NumpyDataset(
            X_reshaped, np.zeros(len(X_reshaped)), add_channel_dim=True, augment=False
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_proba = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_proba.append(proba)
        return np.vstack(all_proba)


class SpectrogramCNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for Pure CNN spectrogram classifier.

    This classifier expects 2D spectrogram inputs (flattened or reshaped from storage).
    Automatically handles:
    - Reshaping flattened spectrograms to 2D
    - GPU acceleration
    - Label encoding
    - Early stopping

    Usage:
        clf = SpectrogramCNNClassifier(input_shape=(128, 256))
        clf.fit(X_train_flat, y_train)  # X can be flattened
        predictions = clf.predict(X_test_flat)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 300,
        early_stopping: bool = True,
        patience: int = 20,
        validation_fraction: float = 0.15,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.input_shape = input_shape
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

        self.device = get_device()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    @property
    def _estimator_type(self):
        return "classifier"

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape flattened spectrogram to 2D if needed."""
        expected_size = self.input_shape[0] * self.input_shape[1]

        if X.ndim == 2 and X.shape[1] == expected_size:
            # Flattened: reshape to (batch, n_mels, time_bins)
            return X.reshape(-1, self.input_shape[0], self.input_shape[1])
        elif X.ndim == 3:
            # Already 2D: (batch, n_mels, time_bins)
            return X
        else:
            raise ValueError(
                f"Input shape {X.shape} doesn't match expected "
                f"flattened {expected_size} or 2D {self.input_shape}"
            )

    def _train_with_gradient_accumulation(
        self, train_loader, val_loader, criterion, optimizer, num_classes: int
    ):
        """Training loop with gradient accumulation for memory efficiency."""
        # Calculate gradient accumulation steps
        # Aim for effective batch size of 16
        accumulation_steps = max(1, 16 // self.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y) / accumulation_steps

                # Backward pass (accumulate gradients)
                loss.backward()

                # Update weights every accumulation_steps batches
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()  # Clear cache to prevent fragmentation

                train_loss += loss.item() * accumulation_steps

            # Update if there are remaining gradients
            if (batch_idx + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss /= len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / total

                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Acc: {val_acc:.4f}"
                    )

                # Early stopping
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}"
                    )

        # Final cache clear
        torch.cuda.empty_cache()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectrogramCNNClassifier":
        """Fit the CNN model."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        num_classes = len(self.classes_)

        # Reshape input
        X_reshaped = self._reshape_input(X)

        # Split into train/validation
        if self.early_stopping and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X_reshaped,
                y_encoded,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_encoded,
            )
        else:
            X_train, y_train = X_reshaped, y_encoded
            X_val, y_val = None, None

        # Create model
        self.model = SpectrogramCNN(
            input_shape=self.input_shape, num_classes=num_classes, dropout=self.dropout
        ).to(self.device)

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create data loaders (memory-efficient - keeps data in numpy)
        # Use drop_last=True to avoid batch_size=1 which breaks batch normalization
        train_dataset = NumpyDataset(
            X_train, y_train, add_channel_dim=True, augment=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,  # Avoid batch_size=1 at end
        )

        val_loader = None
        if X_val is not None:
            val_dataset = NumpyDataset(
                X_val, y_val, add_channel_dim=True, augment=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,  # Keep all validation samples
            )

        # Use gradient accumulation training loop
        self._train_with_gradient_accumulation(
            train_loader, val_loader, criterion, optimizer, num_classes
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities in batches to avoid OOM."""
        self.model.eval()

        X_reshaped = self._reshape_input(X)

        # Create dataset and dataloader for batch prediction
        dataset = NumpyDataset(
            X_reshaped, np.zeros(len(X_reshaped)), add_channel_dim=True
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_proba = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_proba.append(proba)

        return np.vstack(all_proba)


class SpectrogramCNN_MLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for CNN + MLP Hybrid classifier.

    Similar to SpectrogramCNNClassifier but uses a hybrid architecture
    with deeper MLP layers after CNN feature extraction.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),
        cnn_channels: List[int] = [32, 64, 128],
        mlp_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 300,
        early_stopping: bool = True,
        patience: int = 20,
        validation_fraction: float = 0.15,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.input_shape = input_shape
        self.cnn_channels = cnn_channels
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

        self.device = get_device()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    @property
    def _estimator_type(self):
        return "classifier"

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape flattened spectrogram to 2D if needed."""
        expected_size = self.input_shape[0] * self.input_shape[1]

        if X.ndim == 2 and X.shape[1] == expected_size:
            return X.reshape(-1, self.input_shape[0], self.input_shape[1])
        elif X.ndim == 3:
            return X
        else:
            raise ValueError(
                f"Input shape {X.shape} doesn't match expected "
                f"flattened {expected_size} or 2D {self.input_shape}"
            )

    def _train_with_gradient_accumulation(
        self, train_loader, val_loader, criterion, optimizer, num_classes: int
    ):
        """Training loop with gradient accumulation for memory efficiency."""
        # Calculate gradient accumulation steps
        # Aim for effective batch size of 16
        accumulation_steps = max(1, 16 // self.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y) / accumulation_steps

                # Backward pass (accumulate gradients)
                loss.backward()

                # Update weights every accumulation_steps batches
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()  # Clear cache to prevent fragmentation

                train_loss += loss.item() * accumulation_steps

            # Update if there are remaining gradients
            if (batch_idx + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss /= len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / total

                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Acc: {val_acc:.4f}"
                    )

                # Early stopping
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}"
                    )

        # Final cache clear
        torch.cuda.empty_cache()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectrogramCNN_MLPClassifier":
        """Fit the CNN+MLP model."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        num_classes = len(self.classes_)

        # Reshape input
        X_reshaped = self._reshape_input(X)

        # Split into train/validation
        if self.early_stopping and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split

            X_train, X_val, y_train, y_val = train_test_split(
                X_reshaped,
                y_encoded,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_encoded,
            )
        else:
            X_train, y_train = X_reshaped, y_encoded
            X_val, y_val = None, None

        # Create model
        self.model = SpectrogramCNN_MLP(
            input_shape=self.input_shape,
            num_classes=num_classes,
            cnn_channels=self.cnn_channels,
            mlp_hidden_dims=self.mlp_hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Setup training (same as SpectrogramCNNClassifier)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create data loaders (memory-efficient - keeps data in numpy)
        # Use drop_last=True to avoid batch_size=1 which breaks batch normalization
        train_dataset = NumpyDataset(
            X_train, y_train, add_channel_dim=True, augment=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,  # Avoid batch_size=1 at end
        )

        val_loader = None
        if X_val is not None:
            val_dataset = NumpyDataset(
                X_val, y_val, add_channel_dim=True, augment=False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,  # Keep all validation samples
            )

        # Use gradient accumulation training loop
        self._train_with_gradient_accumulation(
            train_loader, val_loader, criterion, optimizer, num_classes
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities in batches to avoid OOM."""
        self.model.eval()

        X_reshaped = self._reshape_input(X)

        # Create dataset and dataloader for batch prediction
        dataset = NumpyDataset(
            X_reshaped, np.zeros(len(X_reshaped)), add_channel_dim=True
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_proba = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_proba.append(proba)

        return np.vstack(all_proba)


class SpectrogramCNN_AdvancedClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for Advanced CNN (Depthwise Separable + Attention).

    This classifier uses a more sophisticated architecture designed for:
    - Larger datasets (3K+ samples)
    - Learning frequency band importance via attention
    - Efficient parameter usage via depthwise separable convolutions

    Use this when Simple CNN plateaus and you have sufficient training data.

    This architecture previously achieved 65% training accuracy on cutout-only datasets.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        input_shape: Tuple[int, int] = (128, 256),
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 300,
        early_stopping: bool = True,
        patience: int = 20,
        validation_fraction: float = 0.15,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ):
        self.input_shape = input_shape
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

        self.device = get_device()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    @property
    def _estimator_type(self):
        return "classifier"

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """Reshape flattened spectrogram to 2D if needed."""
        expected_size = self.input_shape[0] * self.input_shape[1]

        if X.ndim == 2 and X.shape[1] == expected_size:
            # Flattened: reshape to (batch, n_mels, time_bins)
            return X.reshape(-1, self.input_shape[0], self.input_shape[1])
        elif X.ndim == 3:
            # Already 2D: (batch, n_mels, time_bins)
            return X
        else:
            raise ValueError(
                f"Input shape {X.shape} doesn't match expected "
                f"flattened {expected_size} or 2D {self.input_shape}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the advanced CNN model."""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_

        # Reshape input
        X_reshaped = self._reshape_input(X)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped,
            y_encoded,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        # Create model
        num_classes = len(self.classes_)
        self.model = SpectrogramCNN_Advanced(
            input_shape=self.input_shape, num_classes=num_classes, dropout=self.dropout
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Data loaders with augmentation
        train_dataset = NumpyDataset(
            X_train, y_train, add_channel_dim=True, augment=True
        )
        val_dataset = NumpyDataset(X_val, y_val, add_channel_dim=True, augment=False)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

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
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            val_loss /= len(val_loader)
            val_acc = correct / total

            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )

            # Early stopping
            if self.early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities in batches to avoid OOM."""
        self.model.eval()

        X_reshaped = self._reshape_input(X)

        # Create dataset and dataloader for batch prediction
        dataset = NumpyDataset(
            X_reshaped, np.zeros(len(X_reshaped)), add_channel_dim=True, augment=False
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_proba = []
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_proba.append(proba)

        return np.vstack(all_proba)
