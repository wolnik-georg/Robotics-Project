"""
Saliency Analysis for Acoustic Geometric Discrimination
======================================================

This module provides comprehensive saliency analysis to understand which parts
of the acoustic signal are most important for geometric discrimination tasks.

Features:
- Multiple saliency methods (gradient-based, integrated gradients, LIME)
- Support for both raw audio and engineered features
- Batch-specific analysis for different experimental conditions
- Advanced CNN architectures for temporal and spectral analysis
- Comprehensive visualization and interpretation tools
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Add the src directory to the path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import librosa
from scipy import signal

# Try to import LIME for model-agnostic explanations
try:
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print(
        "Warning: LIME not available. Install with 'pip install lime' for additional explanations."
    )


class AdvancedAcousticCNN(nn.Module):
    """
    Advanced CNN for acoustic signal analysis with multiple architectures.
    """

    def __init__(
        self, input_size: int, n_classes: int, architecture: str = "spectro_temporal"
    ):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.architecture = architecture

        if architecture == "spectro_temporal":
            self._build_spectro_temporal()
        elif architecture == "temporal_only":
            self._build_temporal_only()
        elif architecture == "feature_based":
            self._build_feature_based()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_spectro_temporal(self):
        """CNN for spectrogram-like data (frequency x time)."""
        # Assuming input is reshaped to (batch, 1, freq_bins, time_frames)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size dynamically
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, self.n_classes)

    def _build_temporal_only(self):
        """1D CNN for raw temporal audio data."""
        self.conv1 = nn.Conv1d(1, 64, 64, stride=16)
        self.conv2 = nn.Conv1d(64, 128, 32, stride=8)
        self.conv3 = nn.Conv1d(128, 256, 16, stride=4)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        self.fc1 = nn.Linear(256 * 32, 512)
        self.fc2 = nn.Linear(512, self.n_classes)

    def _build_feature_based(self):
        """Dense network for engineered features."""
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, self.n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if self.architecture == "spectro_temporal":
            return self._forward_spectro_temporal(x)
        elif self.architecture == "temporal_only":
            return self._forward_temporal_only(x)
        elif self.architecture == "feature_based":
            return self._forward_feature_based(x)

    def _forward_spectro_temporal(self, x):
        # Reshape to (batch, 1, freq, time) if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def _forward_temporal_only(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def _forward_feature_based(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.fc4(x)


class AcousticSaliencyAnalyzer:
    """
    Comprehensive saliency analysis for acoustic geometric discrimination.
    """

    def __init__(self, batch_configs: Dict, base_data_dir: Path):
        self.batch_configs = batch_configs
        self.base_data_dir = base_data_dir
        self.models = {}
        self.label_encoders = {}
        self.scalers = {}
        self.saliency_results = {}

    def load_batch_data(
        self, batch_name: str, data_type: str = "features"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a specific batch.

        Args:
            batch_name: Name of the batch
            data_type: 'features', 'raw_audio', or 'spectrograms'
        """
        batch_dir = self.base_data_dir / batch_name

        if data_type == "features":
            # Load engineered features
            results_dir = Path(f"batch_analysis_results/{batch_name}")
            features_file = results_dir / f"{batch_name}_features.csv"

            if features_file.exists():
                df = pd.read_csv(features_file)
                X = df.drop(["simplified_label", "original_label"], axis=1).values
                y = df["simplified_label"].values
                return X, y
            else:
                raise FileNotFoundError(f"Features file not found: {features_file}")

        elif data_type == "raw_audio":
            # Load raw audio files
            return self._load_raw_audio_batch(batch_name)

        elif data_type == "spectrograms":
            # Load and compute spectrograms
            return self._load_spectrogram_batch(batch_name)

        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def _load_raw_audio_batch(self, batch_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw audio files for a batch."""
        from acoustic_sensing.models.geometric_data_loader import GeometricDataLoader

        config = self.batch_configs[batch_name]
        loader = GeometricDataLoader(self.base_data_dir / batch_name)
        audio_data, labels = loader.load_batch_data(
            expected_classes=config["expected_classes"],
            class_mapping=config["class_mapping"],
            max_samples_per_class=50,  # Limit for memory
        )

        # Pad/truncate to consistent length
        target_length = 55200  # ~1.15 seconds at 48kHz
        X = []
        for audio in audio_data:
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            X.append(audio)

        return np.array(X), np.array(labels)

    def _load_spectrogram_batch(self, batch_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load audio files and compute spectrograms."""
        X_audio, y = self._load_raw_audio_batch(batch_name)

        # Compute spectrograms
        X_spectro = []
        for audio in X_audio:
            # Compute STFT
            stft = librosa.stft(audio, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)

            # Convert to mel-spectrogram for better representation
            mel_spec = librosa.feature.melspectrogram(
                S=magnitude**2, n_mels=128, sr=48000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            X_spectro.append(mel_spec_db)

        return np.array(X_spectro), y

    def train_model(
        self,
        batch_name: str,
        data_type: str = "features",
        architecture: str = None,
        epochs: int = 100,
    ) -> float:
        """
        Train a neural network model for a specific batch and data type.

        Returns:
            Test accuracy
        """
        print(f"\n🧠 TRAINING MODEL FOR {batch_name} ({data_type})")
        print("-" * 50)

        # Load data
        X, y = self.load_batch_data(batch_name, data_type)
        print(f"Data shape: {X.shape}, Classes: {np.unique(y)}")

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders[batch_name] = le

        # Auto-select architecture if not specified
        if architecture is None:
            if data_type == "features":
                architecture = "feature_based"
            elif data_type == "raw_audio":
                architecture = "temporal_only"
            elif data_type == "spectrograms":
                architecture = "spectro_temporal"

        # Prepare data
        if data_type == "features":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[batch_name] = scaler
            X_tensor = torch.FloatTensor(X_scaled)
        else:
            X_tensor = torch.FloatTensor(X)

        y_tensor = torch.LongTensor(y_encoded)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Create model
        input_size = X.shape[1] if data_type == "features" else X.shape[1:]
        n_classes = len(np.unique(y_encoded))
        model = AdvancedAcousticCNN(input_size, n_classes, architecture)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}"
                )

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        print(f"✅ Test Accuracy: {accuracy:.3f}")

        # Store model
        self.models[f"{batch_name}_{data_type}"] = {
            "model": model,
            "data_type": data_type,
            "architecture": architecture,
            "accuracy": accuracy,
            "test_data": (X_test, y_test),
        }

        return accuracy

    def compute_gradient_saliency(
        self, batch_name: str, data_type: str = "features", sample_idx: int = 0
    ) -> np.ndarray:
        """
        Compute gradient-based saliency map for a specific sample.
        """
        model_key = f"{batch_name}_{data_type}"
        if model_key not in self.models:
            raise ValueError(f"Model not trained for {model_key}")

        model_info = self.models[model_key]
        model = model_info["model"]
        X_test, y_test = model_info["test_data"]

        # Get sample
        x = X_test[sample_idx : sample_idx + 1].clone()
        x.requires_grad_(True)

        model.eval()

        # Forward pass
        output = model(x)
        predicted_class = torch.argmax(output, dim=1)

        # Backward pass on predicted class
        model.zero_grad()
        output[0, predicted_class].backward()

        # Get gradients
        saliency = torch.abs(x.grad).squeeze().detach().numpy()

        return saliency

    def compute_integrated_gradients(
        self,
        batch_name: str,
        data_type: str = "features",
        sample_idx: int = 0,
        steps: int = 50,
    ) -> np.ndarray:
        """
        Compute integrated gradients for more robust saliency.
        """
        model_key = f"{batch_name}_{data_type}"
        model_info = self.models[model_key]
        model = model_info["model"]
        X_test, y_test = model_info["test_data"]

        # Get sample
        x = X_test[sample_idx : sample_idx + 1]
        baseline = torch.zeros_like(x)

        # Create interpolated samples
        alphas = torch.linspace(0, 1, steps)
        integrated_grads = torch.zeros_like(x)

        model.eval()

        for alpha in alphas:
            # Interpolate between baseline and input
            x_interp = baseline + alpha * (x - baseline)
            x_interp.requires_grad_(True)

            # Forward pass
            output = model(x_interp)
            predicted_class = torch.argmax(output, dim=1)

            # Backward pass
            model.zero_grad()
            output[0, predicted_class].backward()

            # Accumulate gradients
            integrated_grads += x_interp.grad / steps

        # Scale by input difference
        integrated_grads *= x - baseline

        return torch.abs(integrated_grads).squeeze().detach().numpy()

    def compute_lime_explanation(self, batch_name: str, sample_idx: int = 0) -> Dict:
        """
        Compute LIME explanation for feature-based models.
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available. Install with 'pip install lime'")

        model_key = f"{batch_name}_features"
        if model_key not in self.models:
            raise ValueError(f"Feature-based model not trained for {batch_name}")

        model_info = self.models[model_key]
        model = model_info["model"]
        X_test, y_test = model_info["test_data"]

        # Get training data for LIME
        X_train, _ = self.load_batch_data(batch_name, "features")
        scaler = self.scalers[batch_name]
        X_train_scaled = scaler.transform(X_train)

        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X_train_scaled,
            mode="classification",
            feature_names=[f"feature_{i}" for i in range(X_train_scaled.shape[1])],
        )

        # Define prediction function
        def predict_fn(x):
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                outputs = model(x_tensor)
                probs = F.softmax(outputs, dim=1)
                return probs.numpy()

        # Get explanation
        instance = X_test[sample_idx].numpy()
        explanation = explainer.explain_instance(
            instance, predict_fn, num_features=len(instance)
        )

        return {
            "explanation": explanation,
            "feature_importance": dict(explanation.as_list()),
        }

    def analyze_batch_saliency(self, batch_name: str, num_samples: int = 10) -> Dict:
        """
        Comprehensive saliency analysis for a batch.
        """
        print(f"\n📊 SALIENCY ANALYSIS FOR {batch_name}")
        print("=" * 60)

        results = {
            "batch_name": batch_name,
            "config": self.batch_configs[batch_name],
            "saliency_maps": {},
            "feature_importance": {},
            "model_performance": {},
        }

        # Train models for different data types
        data_types = [
            "features"
        ]  # Start with features, can extend to raw_audio, spectrograms

        for data_type in data_types:
            try:
                accuracy = self.train_model(batch_name, data_type)
                results["model_performance"][data_type] = accuracy

                # Compute saliency for multiple samples
                gradient_saliencies = []
                integrated_grad_saliencies = []

                for i in range(
                    min(
                        num_samples,
                        len(self.models[f"{batch_name}_{data_type}"]["test_data"][0]),
                    )
                ):
                    # Gradient-based saliency
                    grad_sal = self.compute_gradient_saliency(batch_name, data_type, i)
                    gradient_saliencies.append(grad_sal)

                    # Integrated gradients
                    int_grad_sal = self.compute_integrated_gradients(
                        batch_name, data_type, i
                    )
                    integrated_grad_saliencies.append(int_grad_sal)

                # Average saliency maps
                results["saliency_maps"][data_type] = {
                    "gradient": np.mean(gradient_saliencies, axis=0),
                    "integrated_gradients": np.mean(integrated_grad_saliencies, axis=0),
                    "gradient_std": np.std(gradient_saliencies, axis=0),
                    "integrated_gradients_std": np.std(
                        integrated_grad_saliencies, axis=0
                    ),
                }

                # LIME explanation for features
                if data_type == "features" and LIME_AVAILABLE:
                    try:
                        lime_result = self.compute_lime_explanation(batch_name, 0)
                        results["feature_importance"]["lime"] = lime_result[
                            "feature_importance"
                        ]
                    except Exception as e:
                        print(f"Warning: LIME analysis failed: {e}")

            except Exception as e:
                print(f"Error analyzing {data_type} for {batch_name}: {e}")

        self.saliency_results[batch_name] = results
        return results

    def visualize_saliency_maps(self, batch_name: str, save_dir: Path = None):
        """
        Create comprehensive saliency visualizations.
        """
        if batch_name not in self.saliency_results:
            print(f"No saliency results for {batch_name}")
            return

        results = self.saliency_results[batch_name]

        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f'Saliency Analysis: {batch_name}\n{results["config"]["description"]}',
            fontsize=16,
            fontweight="bold",
        )

        # Get feature names (assuming we have features data)
        if "features" in results["saliency_maps"]:
            # Load feature names
            try:
                features_file = Path(
                    f"batch_analysis_results/{batch_name}/{batch_name}_features.csv"
                )
                if features_file.exists():
                    df = pd.read_csv(features_file)
                    feature_names = [
                        col
                        for col in df.columns
                        if col not in ["simplified_label", "original_label"]
                    ]
                else:
                    feature_names = [
                        f"Feature_{i}"
                        for i in range(
                            len(results["saliency_maps"]["features"]["gradient"])
                        )
                    ]
            except:
                feature_names = [
                    f"Feature_{i}"
                    for i in range(
                        len(results["saliency_maps"]["features"]["gradient"])
                    )
                ]

            # Gradient saliency
            ax1 = axes[0, 0]
            grad_sal = results["saliency_maps"]["features"]["gradient"]
            bars1 = ax1.bar(range(len(grad_sal)), grad_sal, color="skyblue", alpha=0.7)
            ax1.set_title("Gradient-based Saliency")
            ax1.set_xlabel("Feature Index")
            ax1.set_ylabel("Importance")

            # Highlight top features
            top_indices = np.argsort(grad_sal)[-5:]
            for idx in top_indices:
                bars1[idx].set_color("orange")

            # Integrated gradients
            ax2 = axes[0, 1]
            int_grad_sal = results["saliency_maps"]["features"]["integrated_gradients"]
            bars2 = ax2.bar(
                range(len(int_grad_sal)), int_grad_sal, color="lightgreen", alpha=0.7
            )
            ax2.set_title("Integrated Gradients Saliency")
            ax2.set_xlabel("Feature Index")
            ax2.set_ylabel("Importance")

            # Highlight top features
            top_indices = np.argsort(int_grad_sal)[-5:]
            for idx in top_indices:
                bars2[idx].set_color("red")

            # Feature importance comparison
            ax3 = axes[1, 0]
            top_n = 10
            top_grad_indices = np.argsort(grad_sal)[-top_n:]
            top_int_grad_indices = np.argsort(int_grad_sal)[-top_n:]

            comparison_data = {
                "Gradient": grad_sal[top_grad_indices],
                "Integrated Gradients": int_grad_sal[top_int_grad_indices],
            }

            x_pos = np.arange(top_n)
            width = 0.35
            ax3.bar(
                x_pos - width / 2,
                comparison_data["Gradient"],
                width,
                label="Gradient",
                alpha=0.7,
                color="skyblue",
            )
            ax3.bar(
                x_pos + width / 2,
                comparison_data["Integrated Gradients"],
                width,
                label="Integrated Gradients",
                alpha=0.7,
                color="lightgreen",
            )

            ax3.set_title(f"Top {top_n} Feature Importance Comparison")
            ax3.set_xlabel("Feature Rank")
            ax3.set_ylabel("Importance")
            ax3.legend()

            # Feature correlation heatmap
            ax4 = axes[1, 1]
            try:
                # Create correlation between different saliency methods
                correlation_matrix = np.corrcoef([grad_sal, int_grad_sal])
                im = ax4.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                ax4.set_title("Saliency Method Correlation")
                ax4.set_xticks([0, 1])
                ax4.set_yticks([0, 1])
                ax4.set_xticklabels(["Gradient", "Int. Grad"])
                ax4.set_yticklabels(["Gradient", "Int. Grad"])

                # Add correlation values
                for i in range(2):
                    for j in range(2):
                        ax4.text(
                            j,
                            i,
                            f"{correlation_matrix[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color=(
                                "white"
                                if abs(correlation_matrix[i, j]) > 0.5
                                else "black"
                            ),
                        )

                plt.colorbar(im, ax=ax4)
            except:
                ax4.text(
                    0.5,
                    0.5,
                    "Correlation analysis\nnot available",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.set_title("Saliency Method Correlation")

        plt.tight_layout()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            plt.savefig(
                save_dir / f"{batch_name}_saliency_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Save detailed feature importance
            self._save_detailed_results(batch_name, save_dir)

        plt.show()

    def _save_detailed_results(self, batch_name: str, save_dir: Path):
        """Save detailed saliency results to files."""
        results = self.saliency_results[batch_name]

        # Save saliency maps as CSV
        if "features" in results["saliency_maps"]:
            saliency_df = pd.DataFrame(
                {
                    "feature_index": range(
                        len(results["saliency_maps"]["features"]["gradient"])
                    ),
                    "gradient_saliency": results["saliency_maps"]["features"][
                        "gradient"
                    ],
                    "integrated_gradients": results["saliency_maps"]["features"][
                        "integrated_gradients"
                    ],
                    "gradient_std": results["saliency_maps"]["features"][
                        "gradient_std"
                    ],
                    "integrated_gradients_std": results["saliency_maps"]["features"][
                        "integrated_gradients_std"
                    ],
                }
            )

            # Add feature names if available
            try:
                features_file = Path(
                    f"batch_analysis_results/{batch_name}/{batch_name}_features.csv"
                )
                if features_file.exists():
                    df = pd.read_csv(features_file)
                    feature_names = [
                        col
                        for col in df.columns
                        if col not in ["simplified_label", "original_label"]
                    ]
                    saliency_df["feature_name"] = feature_names
            except:
                saliency_df["feature_name"] = [
                    f"feature_{i}" for i in range(len(saliency_df))
                ]

            saliency_df.to_csv(
                save_dir / f"{batch_name}_saliency_maps.csv", index=False
            )

        # Save summary results as JSON
        summary = {
            "batch_name": batch_name,
            "config": results["config"],
            "model_performance": results["model_performance"],
            "top_features": {},
        }

        if "features" in results["saliency_maps"]:
            grad_sal = results["saliency_maps"]["features"]["gradient"]
            int_grad_sal = results["saliency_maps"]["features"]["integrated_gradients"]

            summary["top_features"] = {
                "gradient_top_10": np.argsort(grad_sal)[-10:].tolist(),
                "integrated_gradients_top_10": np.argsort(int_grad_sal)[-10:].tolist(),
                "gradient_values_top_10": grad_sal[np.argsort(grad_sal)[-10:]].tolist(),
                "integrated_gradients_values_top_10": int_grad_sal[
                    np.argsort(int_grad_sal)[-10:]
                ].tolist(),
            }

        with open(save_dir / f"{batch_name}_saliency_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def analyze_all_batches(self, save_results: bool = True) -> Dict:
        """
        Run comprehensive saliency analysis for all available batches.
        """
        print("🧪 COMPREHENSIVE SALIENCY ANALYSIS")
        print("=" * 80)

        all_results = {}

        for batch_name in self.batch_configs.keys():
            batch_dir = self.base_data_dir / batch_name
            if batch_dir.exists():
                try:
                    results = self.analyze_batch_saliency(batch_name)
                    all_results[batch_name] = results

                    if save_results:
                        save_dir = Path(f"batch_analysis_results/{batch_name}")
                        self.visualize_saliency_maps(batch_name, save_dir)

                    print(f"✅ {batch_name} saliency analysis completed")
                except Exception as e:
                    print(f"❌ {batch_name} saliency analysis failed: {e}")
                    import traceback

                    traceback.print_exc()

        # Generate combined summary
        if save_results and all_results:
            self._generate_combined_saliency_summary(all_results)

        return all_results

    def _generate_combined_saliency_summary(self, all_results: Dict):
        """Generate a combined summary of all saliency analyses."""
        summary_path = Path("batch_analysis_results/combined_saliency_summary.txt")

        with open(summary_path, "w") as f:
            f.write("COMBINED SALIENCY ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for batch_name, result in all_results.items():
                config = result["config"]
                f.write(f"BATCH: {batch_name}\n")
                f.write(f"Experiment: {config['description']}\n")
                f.write(f"Research Question: {config['research_question']}\n")

                if "features" in result["model_performance"]:
                    accuracy = result["model_performance"]["features"]
                    f.write(f"Model Accuracy: {accuracy:.3f}\n")

                if "features" in result["saliency_maps"]:
                    grad_sal = result["saliency_maps"]["features"]["gradient"]
                    int_grad_sal = result["saliency_maps"]["features"][
                        "integrated_gradients"
                    ]

                    f.write(
                        f"Top Gradient Features: {np.argsort(grad_sal)[-5:].tolist()}\n"
                    )
                    f.write(
                        f"Top Int. Grad Features: {np.argsort(int_grad_sal)[-5:].tolist()}\n"
                    )
                    f.write(
                        f"Saliency Consistency (corr): {np.corrcoef(grad_sal, int_grad_sal)[0,1]:.3f}\n"
                    )

                f.write("\n")

        print(f"📋 Combined saliency summary saved to: {summary_path}")


if __name__ == "__main__":
    # Example usage
    from acoustic_sensing.analysis.batch_analysis import BatchSpecificAnalyzer

    # Get batch configurations
    analyzer = BatchSpecificAnalyzer()
    batch_configs = analyzer.batch_configs
    base_data_dir = analyzer.base_dir

    # Create saliency analyzer
    saliency_analyzer = AcousticSaliencyAnalyzer(batch_configs, base_data_dir)

    # Run analysis for all batches
    results = saliency_analyzer.analyze_all_batches(save_results=True)

    print("\n🎉 Saliency analysis completed!")
    print("Check the batch_analysis_results directories for detailed results.")
