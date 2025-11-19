from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for saliency analysis."""

    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class SaliencyAnalysisExperiment(BaseExperiment):
    """
    Experiment for neural network saliency analysis to understand feature importance.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform saliency analysis using neural networks.

        Args:
            shared_data: Dictionary containing loaded features and labels

        Returns:
            Dictionary containing saliency maps and feature importance analysis
        """
        self.logger.info("Starting saliency analysis experiment...")

        # Load per-batch data from previous experiment
        batch_results = self.load_shared_data(shared_data, "batch_results")

        # For saliency analysis, combine all batch data for comprehensive analysis
        all_X = []
        all_y = []

        for batch_name, batch_data in batch_results.items():
            X_batch = batch_data["features"]
            y_batch = batch_data["labels"]

            all_X.append(X_batch)
            all_y.append(y_batch)

        # Combine all batches
        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        self.logger.info(
            f"Combined data: {len(X)} samples, {X.shape[1]} features across {len(np.unique(y))} classes"
        )

        # Prepare data for neural network
        X_processed, y_processed, scaler, label_encoder = self._prepare_data(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y_processed,
            test_size=0.2,
            random_state=42,
            stratify=y_processed,
        )

        # Create and train neural network
        model = self._create_and_train_model(X_train, y_train, X_test, y_test)

        # Perform saliency analysis
        saliency_results = self._perform_saliency_analysis(
            model, X_test, y_test, scaler
        )

        # Analyze feature importance patterns
        feature_analysis = self._analyze_feature_patterns(saliency_results, X.shape[1])

        results = {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "saliency_maps": saliency_results,
            "feature_analysis": feature_analysis,
            "model_performance": self._evaluate_model(model, X_test, y_test),
        }

        # Create visualizations
        self._create_saliency_visualizations(saliency_results, feature_analysis)

        # Save summary
        self._save_saliency_summary(results)

        self.logger.info("Saliency analysis experiment completed")
        return results

    def _prepare_data(self, X: np.ndarray, y: np.ndarray):
        """Prepare data for neural network training."""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Encode labels
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return X_scaled, y_encoded, scaler, label_encoder

    def _create_and_train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> nn.Module:
        """Create and train neural network model."""
        self.logger.info("Creating and training neural network...")

        # Model parameters
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        hidden_size = min(128, input_size * 2)  # Adaptive hidden size

        # Create model
        model = SimpleClassifier(input_size, hidden_size, num_classes)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = self.config.get("neural_network_epochs", 100)

        # Training loop
        model.train()
        train_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor).item()
                    _, predicted = torch.max(test_outputs, 1)
                    accuracy = (predicted == y_test_tensor).float().mean().item()

                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
                )
                model.train()

        model.eval()
        return model

    def _perform_saliency_analysis(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler: StandardScaler,
    ) -> dict:
        """Perform saliency analysis using different gradient methods."""
        self.logger.info("Performing saliency analysis...")

        saliency_results = {}
        gradient_methods = self.config.get("gradient_methods", ["basic", "integrated"])
        num_samples = min(self.config.get("num_samples", 10), len(X_test))

        # Select random samples for analysis
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_samples = X_test[sample_indices]
        y_samples = y_test[sample_indices]

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_samples)
        X_tensor.requires_grad_(True)

        if "basic" in gradient_methods:
            saliency_results["basic_gradients"] = self._compute_basic_gradients(
                model, X_tensor, y_samples
            )

        if "integrated" in gradient_methods:
            saliency_results["integrated_gradients"] = (
                self._compute_integrated_gradients(model, X_tensor, y_samples)
            )

        # Compute feature importance statistics
        saliency_results["feature_importance_stats"] = (
            self._compute_feature_importance_stats(saliency_results)
        )

        return saliency_results

    def _compute_basic_gradients(
        self, model: nn.Module, X_tensor: torch.Tensor, y_samples: np.ndarray
    ) -> dict:
        """Compute basic gradients for saliency."""
        model.eval()
        gradients = []

        for i in range(len(X_tensor)):
            X_sample = X_tensor[i : i + 1]
            X_sample.retain_grad()

            # Forward pass
            output = model(X_sample)
            target_class = torch.LongTensor([y_samples[i]])

            # Compute loss and gradients
            loss = nn.CrossEntropyLoss()(output, target_class)
            loss.backward()

            # Store gradients
            grad = X_sample.grad.detach().numpy()[0]
            gradients.append(grad)

            # Clear gradients
            model.zero_grad()

        return {
            "gradients": np.array(gradients),
            "mean_absolute_gradient": np.mean(np.abs(gradients), axis=0),
            "std_gradient": np.std(gradients, axis=0),
        }

    def _compute_integrated_gradients(
        self,
        model: nn.Module,
        X_tensor: torch.Tensor,
        y_samples: np.ndarray,
        steps: int = 50,
    ) -> dict:
        """Compute integrated gradients for saliency."""
        model.eval()
        integrated_grads = []

        # Create baseline (zeros)
        baseline = torch.zeros_like(X_tensor)

        for i in range(len(X_tensor)):
            X_sample = X_tensor[i : i + 1]
            baseline_sample = baseline[i : i + 1]
            target_class = y_samples[i]

            # Compute integrated gradients
            integrated_grad = self._compute_integrated_gradient_sample(
                model, baseline_sample, X_sample, target_class, steps
            )
            integrated_grads.append(integrated_grad)

        return {
            "integrated_gradients": np.array(integrated_grads),
            "mean_integrated_gradient": np.mean(np.abs(integrated_grads), axis=0),
            "std_integrated_gradient": np.std(integrated_grads, axis=0),
        }

    def _compute_integrated_gradient_sample(
        self,
        model: nn.Module,
        baseline: torch.Tensor,
        input_tensor: torch.Tensor,
        target_class: int,
        steps: int,
    ) -> np.ndarray:
        """Compute integrated gradients for a single sample."""
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).view(-1, 1)
        interpolated = baseline + alphas * (input_tensor - baseline)
        interpolated.requires_grad_(True)

        # Compute gradients for interpolated inputs
        gradients = torch.zeros_like(interpolated)

        for j in range(steps):
            interp_input = interpolated[j : j + 1]
            output = model(interp_input)
            target_output = output[0, target_class]

            grad = torch.autograd.grad(target_output, interp_input, retain_graph=True)[
                0
            ]
            gradients[j] = grad

        # Integrate gradients
        avg_gradients = torch.mean(gradients, dim=0)
        integrated_gradients = (input_tensor - baseline) * avg_gradients

        return integrated_gradients.detach().numpy()[0]

    def _compute_feature_importance_stats(self, saliency_results: dict) -> dict:
        """Compute feature importance statistics across all methods."""
        importance_stats = {}

        for method, results in saliency_results.items():
            if "mean_" in method:  # Skip aggregated results
                continue

            if "gradients" in results:
                gradients = results["gradients"]
                importance_stats[method] = {
                    "feature_ranking": np.argsort(np.mean(np.abs(gradients), axis=0))[
                        ::-1
                    ].tolist(),
                    "top_10_features": np.argsort(np.mean(np.abs(gradients), axis=0))[
                        -10:
                    ][::-1].tolist(),
                    "importance_scores": np.mean(np.abs(gradients), axis=0).tolist(),
                }

        return importance_stats

    def _analyze_feature_patterns(
        self, saliency_results: dict, num_features: int
    ) -> dict:
        """Analyze patterns in feature importance."""
        self.logger.info("Analyzing feature importance patterns...")

        analysis = {
            "consistently_important": [],
            "method_specific": {},
            "feature_groups": {},
        }

        # Find consistently important features across methods
        if len(saliency_results.get("feature_importance_stats", {})) > 1:
            all_rankings = []
            for method, stats in saliency_results["feature_importance_stats"].items():
                if "top_10_features" in stats:
                    all_rankings.append(set(stats["top_10_features"]))

            if len(all_rankings) > 1:
                # Find intersection of top features across methods
                consistent_features = set.intersection(*all_rankings)
                analysis["consistently_important"] = list(consistent_features)

        # Analyze feature groups (acoustic vs impulse response features)
        # Assume first 38 features are acoustic, rest are impulse response
        acoustic_features = list(range(38))
        impulse_features = list(range(38, num_features))

        for method, stats in saliency_results.get(
            "feature_importance_stats", {}
        ).items():
            if "importance_scores" in stats:
                scores = np.array(stats["importance_scores"])

                acoustic_importance = (
                    np.mean(scores[acoustic_features]) if acoustic_features else 0
                )
                impulse_importance = (
                    np.mean(scores[impulse_features]) if impulse_features else 0
                )

                analysis["feature_groups"][method] = {
                    "acoustic_importance": float(acoustic_importance),
                    "impulse_importance": float(impulse_importance),
                    "acoustic_vs_impulse_ratio": (
                        float(acoustic_importance / impulse_importance)
                        if impulse_importance > 0
                        else float("inf")
                    ),
                }

        return analysis

    def _evaluate_model(
        self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Evaluate model performance."""
        model.eval()

        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.LongTensor(y_test)

        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).float().mean().item()

            # Compute per-class accuracy
            unique_classes = np.unique(y_test)
            per_class_acc = {}
            for cls in unique_classes:
                mask = y_test == cls
                if mask.any():
                    cls_acc = (predicted[mask] == y_tensor[mask]).float().mean().item()
                    per_class_acc[int(cls)] = cls_acc

        return {"overall_accuracy": accuracy, "per_class_accuracy": per_class_acc}

    def _create_saliency_visualizations(
        self, saliency_results: dict, feature_analysis: dict
    ):
        """Create saliency analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Feature importance heatmap
        if "feature_importance_stats" in saliency_results:
            importance_data = []
            method_names = []

            for method, stats in saliency_results["feature_importance_stats"].items():
                if "importance_scores" in stats:
                    importance_data.append(stats["importance_scores"])
                    method_names.append(method)

            if importance_data:
                importance_matrix = np.array(importance_data)
                sns.heatmap(
                    importance_matrix,
                    yticklabels=method_names,
                    cmap="viridis",
                    ax=axes[0, 0],
                )
                axes[0, 0].set_title("Feature Importance Heatmap")
                axes[0, 0].set_xlabel("Feature Index")

        # 2. Top features comparison
        if "feature_importance_stats" in saliency_results:
            for i, (method, stats) in enumerate(
                saliency_results["feature_importance_stats"].items()
            ):
                if "importance_scores" in stats:
                    top_indices = np.argsort(stats["importance_scores"])[-20:]
                    top_scores = np.array(stats["importance_scores"])[top_indices]

                    axes[0, 1].barh(
                        range(len(top_indices)), top_scores, alpha=0.7, label=method
                    )
                    axes[0, 1].set_yticks(range(len(top_indices)))
                    axes[0, 1].set_yticklabels(
                        [f"Feature {idx}" for idx in top_indices]
                    )
                    break  # Show only first method to avoid overcrowding

            axes[0, 1].set_xlabel("Importance Score")
            axes[0, 1].set_title("Top 20 Most Important Features")

        # 3. Acoustic vs Impulse Response importance
        if "feature_groups" in feature_analysis:
            methods = list(feature_analysis["feature_groups"].keys())
            acoustic_scores = [
                feature_analysis["feature_groups"][m]["acoustic_importance"]
                for m in methods
            ]
            impulse_scores = [
                feature_analysis["feature_groups"][m]["impulse_importance"]
                for m in methods
            ]

            x = np.arange(len(methods))
            width = 0.35

            axes[1, 0].bar(
                x - width / 2, acoustic_scores, width, label="Acoustic Features"
            )
            axes[1, 0].bar(
                x + width / 2, impulse_scores, width, label="Impulse Response Features"
            )
            axes[1, 0].set_xlabel("Method")
            axes[1, 0].set_ylabel("Average Importance")
            axes[1, 0].set_title("Feature Group Importance Comparison")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(methods, rotation=45)
            axes[1, 0].legend()

        # 4. Consistency analysis
        if (
            "consistently_important" in feature_analysis
            and feature_analysis["consistently_important"]
        ):
            consistent_features = feature_analysis["consistently_important"]
            axes[1, 1].bar(
                range(len(consistent_features)), [1] * len(consistent_features)
            )
            axes[1, 1].set_xlabel("Feature Index")
            axes[1, 1].set_ylabel("Consistency")
            axes[1, 1].set_title(
                f"Consistently Important Features (n={len(consistent_features)})"
            )
            axes[1, 1].set_xticks(range(len(consistent_features)))
            axes[1, 1].set_xticklabels(
                [f"F{f}" for f in consistent_features], rotation=45
            )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No consistently\nimportant features\nfound across methods",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Feature Consistency Analysis")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "saliency_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_saliency_summary(self, results: dict):
        """Save saliency analysis summary."""
        summary = {
            "model_performance": results["model_performance"],
            "consistently_important_features": results["feature_analysis"].get(
                "consistently_important", []
            ),
            "num_consistently_important": len(
                results["feature_analysis"].get("consistently_important", [])
            ),
            "feature_group_analysis": results["feature_analysis"].get(
                "feature_groups", {}
            ),
            "methods_used": list(
                results["saliency_maps"].get("feature_importance_stats", {}).keys()
            ),
        }

        # Add top features for each method
        for method, stats in (
            results["saliency_maps"].get("feature_importance_stats", {}).items()
        ):
            if "top_10_features" in stats:
                summary[f"{method}_top_features"] = stats["top_10_features"]

        self.save_results(summary, "saliency_summary.json")
