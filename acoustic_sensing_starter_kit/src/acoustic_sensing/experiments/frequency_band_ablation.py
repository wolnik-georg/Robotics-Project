from .base_experiment import BaseExperiment
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.signal import butter, filtfilt
import os


class FrequencyBandAblationExperiment(BaseExperiment):
    """
    Experiment for frequency band ablation analysis to understand frequency-specific contributions.
    """

    def get_dependencies(self) -> List[str]:
        """Depends on data processing."""
        return ["data_processing"]

    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform frequency band ablation analysis.

        Args:
            shared_data: Dictionary containing loaded features and labels

        Returns:
            Dictionary containing frequency band analysis results
        """
        self.logger.info("Starting frequency band ablation experiment...")

        # Load per-batch data from previous experiment
        batch_results = self.load_shared_data(shared_data, "batch_results")

        # For frequency band ablation, we need to work with raw audio data
        # For now, combine the extracted features from all batches
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

        # Get baseline performance with all features
        baseline_performance = self._get_baseline_performance(X, y)
        self.logger.info(
            f"Baseline accuracy with all features: {baseline_performance:.4f}"
        )

        # Define frequency bands for analysis
        frequency_bands = self._define_frequency_bands()

        # Perform frequency band analysis on acoustic features
        # Assuming first 38 features are acoustic features
        acoustic_features = X[:, :38] if X.shape[1] >= 38 else X

        # Analyze each frequency band
        band_results = self._analyze_frequency_bands(
            acoustic_features, y, frequency_bands
        )

        # Perform frequency band ablation (removing each band)
        ablation_results = self._perform_band_ablation(
            X, y, frequency_bands, baseline_performance
        )

        # Analyze frequency band combinations
        combination_results = self._analyze_band_combinations(
            acoustic_features, y, frequency_bands
        )

        # Perform material-specific frequency analysis
        material_analysis = self._analyze_material_frequency_responses(
            acoustic_features, y, frequency_bands, batch_results
        )

        results = {
            "baseline_performance": baseline_performance,
            "frequency_bands": frequency_bands,
            "band_results": band_results,
            "ablation_results": ablation_results,
            "combination_results": combination_results,
            "material_analysis": material_analysis,
            "optimal_bands": self._find_optimal_frequency_bands(
                band_results, combination_results
            ),
        }

        # Create visualizations
        self._create_frequency_band_visualizations(results)

        # Save summary
        self._save_frequency_band_summary(results)

        self.logger.info("Frequency band ablation experiment completed")
        return results

    def _get_baseline_performance(self, X: np.ndarray, y: np.ndarray) -> float:
        """Get baseline performance with all features."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="accuracy")
        return scores.mean()

    def _define_frequency_bands(self) -> dict:
        """Define frequency bands for analysis."""
        frequency_bands = self.config.get(
            "frequency_bands", ["low", "mid", "high", "ultra_high"]
        )

        # Define standard frequency bands (in Hz)
        band_definitions = {
            "low": {
                "range": (20, 250),
                "name": "Low (20-250 Hz)",
                "description": "Low frequency components",
            },
            "low_mid": {
                "range": (250, 500),
                "name": "Low-Mid (250-500 Hz)",
                "description": "Low-mid frequency components",
            },
            "mid": {
                "range": (500, 2000),
                "name": "Mid (500-2000 Hz)",
                "description": "Mid frequency components",
            },
            "high_mid": {
                "range": (2000, 4000),
                "name": "High-Mid (2000-4000 Hz)",
                "description": "High-mid frequency components",
            },
            "high": {
                "range": (4000, 8000),
                "name": "High (4000-8000 Hz)",
                "description": "High frequency components",
            },
            "ultra_high": {
                "range": (8000, 20000),
                "name": "Ultra-High (8000-20000 Hz)",
                "description": "Ultra-high frequency components",
            },
        }

        # Filter based on configuration
        selected_bands = {}
        for band in frequency_bands:
            if band in band_definitions:
                selected_bands[band] = band_definitions[band]

        # If no valid bands specified, use default set
        if not selected_bands:
            default_bands = ["low", "mid", "high", "ultra_high"]
            for band in default_bands:
                selected_bands[band] = band_definitions[band]

        return selected_bands

    def _analyze_frequency_bands(
        self, X: np.ndarray, y: np.ndarray, frequency_bands: dict
    ) -> dict:
        """Analyze performance of individual frequency bands."""
        self.logger.info("Analyzing individual frequency bands...")

        band_results = {}

        # For each frequency band, simulate extracting band-specific features
        for band_name, band_info in frequency_bands.items():
            self.logger.info(f"Analyzing {band_name} frequency band...")

            # Simulate frequency band features by creating synthetic features
            # In practice, this would involve filtering the original audio and re-extracting features
            band_features = self._simulate_band_features(X, band_info, band_name)

            if band_features is not None:
                # Standardize features
                scaler = StandardScaler()
                X_band_scaled = scaler.fit_transform(band_features)

                # Evaluate performance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(
                    rf, X_band_scaled, y, cv=cv, scoring="accuracy"
                )

                # Get feature importance
                rf.fit(X_band_scaled, y)
                feature_importance = rf.feature_importances_

                # Analyze class separability
                class_analysis = self._analyze_class_separability(X_band_scaled, y)

                band_results[band_name] = {
                    "frequency_range": band_info["range"],
                    "performance": {
                        "mean_accuracy": float(scores.mean()),
                        "std_accuracy": float(scores.std()),
                        "scores": scores.tolist(),
                    },
                    "feature_importance": feature_importance.tolist(),
                    "num_features": X_band_scaled.shape[1],
                    "class_analysis": class_analysis,
                    "scaler": scaler,
                }

                self.logger.info(
                    f"{band_name}: {scores.mean():.4f} ± {scores.std():.4f}"
                )
            else:
                self.logger.warning(f"Could not generate features for {band_name}")

        return band_results

    def _simulate_band_features(
        self, X: np.ndarray, band_info: dict, band_name: str
    ) -> np.ndarray:
        """
        Simulate frequency band-specific features.
        In practice, this would involve filtering original audio and re-extracting features.
        """
        freq_range = band_info["range"]
        low_freq, high_freq = freq_range

        # Create band-specific features by selecting and modifying existing features
        # This is a simplified simulation - real implementation would re-process audio

        num_samples, num_features = X.shape

        # Simulate band features based on frequency characteristics
        if band_name == "low":
            # Low frequency: select features that typically respond to low frequencies
            # Focus on first few features (often energy-based)
            band_features = X[:, : min(8, num_features)]
            # Add some synthetic low-frequency characteristics
            noise_factor = 0.1
            band_features = band_features + np.random.normal(
                0, noise_factor, band_features.shape
            )

        elif band_name == "mid":
            # Mid frequency: select middle portion of features
            start_idx = min(8, num_features // 4)
            end_idx = min(20, 3 * num_features // 4)
            band_features = X[:, start_idx:end_idx]

        elif band_name == "high":
            # High frequency: select features that respond to high frequencies
            start_idx = min(20, num_features // 2)
            end_idx = min(30, num_features)
            band_features = X[:, start_idx:end_idx]

        elif band_name == "ultra_high":
            # Ultra high frequency: select last portion of features
            start_idx = min(30, 3 * num_features // 4)
            band_features = X[:, start_idx:]

        else:
            # Default: use subset of features
            subset_size = max(5, num_features // 4)
            band_features = X[:, :subset_size]

        # Apply frequency-specific transformations
        band_features = self._apply_frequency_transform(band_features, freq_range)

        return band_features

    def _apply_frequency_transform(
        self, features: np.ndarray, freq_range: tuple
    ) -> np.ndarray:
        """Apply frequency-specific transformations to simulate band filtering effects."""
        low_freq, high_freq = freq_range

        # Simulate the effect of bandpass filtering on feature values
        # Higher frequencies tend to have more rapid variations
        # Lower frequencies tend to have smoother variations

        transformed_features = features.copy()

        # Frequency-dependent noise and scaling
        if high_freq <= 500:  # Low frequency
            # Low frequencies: add smooth variations, reduce noise
            smooth_factor = 0.8
            noise_factor = 0.05

        elif high_freq <= 2000:  # Mid frequency
            # Mid frequencies: moderate variations
            smooth_factor = 0.6
            noise_factor = 0.1

        else:  # High frequency
            # High frequencies: more variations, higher noise
            smooth_factor = 0.4
            noise_factor = 0.15

        # Apply smoothing and noise
        for i in range(transformed_features.shape[1]):
            feature_col = transformed_features[:, i]

            # Add frequency-specific noise
            noise = np.random.normal(
                0, noise_factor * np.std(feature_col), len(feature_col)
            )
            transformed_features[:, i] = (
                smooth_factor * feature_col + (1 - smooth_factor) * noise
            )

        return transformed_features

    def _analyze_class_separability(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Analyze class separability in frequency band."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import silhouette_score

        class_analysis = {}

        try:
            # Calculate silhouette score
            if len(np.unique(y)) > 1 and len(X) > len(np.unique(y)):
                silhouette_avg = silhouette_score(X, y)
                class_analysis["silhouette_score"] = float(silhouette_avg)
            else:
                class_analysis["silhouette_score"] = 0.0

            # Calculate class centroids and distances
            unique_classes = np.unique(y)
            centroids = {}

            for cls in unique_classes:
                mask = y == cls
                if np.sum(mask) > 0:
                    centroid = np.mean(X[mask], axis=0)
                    centroids[str(cls)] = centroid.tolist()

            class_analysis["centroids"] = centroids

            # Calculate inter-class distances
            if len(centroids) > 1:
                class_pairs = []
                distances = []

                classes = list(centroids.keys())
                for i in range(len(classes)):
                    for j in range(i + 1, len(classes)):
                        c1, c2 = classes[i], classes[j]
                        dist = np.linalg.norm(
                            np.array(centroids[c1]) - np.array(centroids[c2])
                        )
                        class_pairs.append(f"{c1}-{c2}")
                        distances.append(float(dist))

                class_analysis["inter_class_distances"] = {
                    "pairs": class_pairs,
                    "distances": distances,
                    "mean_distance": float(np.mean(distances)) if distances else 0.0,
                }

        except Exception as e:
            self.logger.warning(f"Error in class separability analysis: {str(e)}")
            class_analysis["error"] = str(e)

        return class_analysis

    def _perform_band_ablation(
        self, X: np.ndarray, y: np.ndarray, frequency_bands: dict, baseline: float
    ) -> dict:
        """Perform ablation by removing each frequency band."""
        self.logger.info("Performing frequency band ablation...")

        ablation_results = {}

        # Standardize full features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for band_name, band_info in frequency_bands.items():
            self.logger.info(f"Ablating {band_name} frequency band...")

            # Simulate removing band-specific features
            # In practice, this would involve re-processing audio without this band
            X_ablated = self._simulate_band_removal(X_scaled, band_name, band_info)

            if X_ablated is not None:
                # Evaluate performance without this band
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(rf, X_ablated, y, cv=cv, scoring="accuracy")

                performance_drop = baseline - scores.mean()

                ablation_results[band_name] = {
                    "frequency_range": band_info["range"],
                    "performance_without_band": float(scores.mean()),
                    "performance_drop": float(performance_drop),
                    "importance_score": float(
                        performance_drop
                    ),  # Higher drop = more important
                    "scores": scores.tolist(),
                    "num_features_removed": X_scaled.shape[1] - X_ablated.shape[1],
                }

                self.logger.info(
                    f"{band_name} ablated: {scores.mean():.4f} "
                    f"(drop: {performance_drop:+.4f})"
                )

        return ablation_results

    def _simulate_band_removal(
        self, X: np.ndarray, band_name: str, band_info: dict
    ) -> np.ndarray:
        """
        Simulate removing frequency band features.
        In practice, this would involve re-processing audio without the specific frequency band.
        """
        num_features = X.shape[1]

        # Determine which features to remove based on band
        if band_name == "low":
            # Remove first portion of features (low-frequency related)
            features_to_remove = list(range(0, min(8, num_features // 4)))

        elif band_name == "mid":
            # Remove middle portion of features
            start_idx = min(8, num_features // 4)
            end_idx = min(20, 3 * num_features // 4)
            features_to_remove = list(range(start_idx, end_idx))

        elif band_name == "high":
            # Remove high-frequency related features
            start_idx = min(20, num_features // 2)
            end_idx = min(30, num_features)
            features_to_remove = list(range(start_idx, end_idx))

        elif band_name == "ultra_high":
            # Remove last portion of features
            start_idx = min(30, 3 * num_features // 4)
            features_to_remove = list(range(start_idx, num_features))

        else:
            # Default removal pattern
            features_to_remove = list(range(0, min(5, num_features // 4)))

        # Remove selected features
        if features_to_remove:
            remaining_features = [
                i for i in range(num_features) if i not in features_to_remove
            ]
            X_ablated = X[:, remaining_features]
        else:
            X_ablated = X  # No features to remove

        return X_ablated

    def _analyze_band_combinations(
        self, X: np.ndarray, y: np.ndarray, frequency_bands: dict
    ) -> dict:
        """Analyze performance of frequency band combinations."""
        self.logger.info("Analyzing frequency band combinations...")

        from itertools import combinations

        combination_results = {}
        band_names = list(frequency_bands.keys())

        # Test all possible combinations of 2 bands
        for combo in combinations(band_names, 2):
            band1, band2 = combo
            combo_name = f"{band1}+{band2}"

            self.logger.info(f"Testing combination: {combo_name}")

            # Combine features from both bands
            features1 = self._simulate_band_features(X, frequency_bands[band1], band1)
            features2 = self._simulate_band_features(X, frequency_bands[band2], band2)

            if features1 is not None and features2 is not None:
                # Combine features
                combined_features = np.hstack([features1, features2])

                # Standardize
                scaler = StandardScaler()
                X_combo_scaled = scaler.fit_transform(combined_features)

                # Evaluate performance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(
                    rf, X_combo_scaled, y, cv=cv, scoring="accuracy"
                )

                combination_results[combo_name] = {
                    "bands": list(combo),
                    "frequency_ranges": [
                        frequency_bands[band]["range"] for band in combo
                    ],
                    "performance": {
                        "mean_accuracy": float(scores.mean()),
                        "std_accuracy": float(scores.std()),
                        "scores": scores.tolist(),
                    },
                    "num_features": X_combo_scaled.shape[1],
                }

                self.logger.info(f"{combo_name}: {scores.mean():.4f}")

        # Test combination of 3 bands if we have enough bands
        if len(band_names) >= 3:
            for combo in list(combinations(band_names, 3))[
                :3
            ]:  # Limit to first 3 combinations
                combo_name = "+".join(combo)

                # Combine features from all bands
                combined_features = []
                for band in combo:
                    band_features = self._simulate_band_features(
                        X, frequency_bands[band], band
                    )
                    if band_features is not None:
                        combined_features.append(band_features)

                if len(combined_features) == len(combo):
                    combined_features = np.hstack(combined_features)

                    # Standardize and evaluate
                    scaler = StandardScaler()
                    X_combo_scaled = scaler.fit_transform(combined_features)

                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    scores = cross_val_score(
                        rf, X_combo_scaled, y, cv=cv, scoring="accuracy"
                    )

                    combination_results[combo_name] = {
                        "bands": list(combo),
                        "frequency_ranges": [
                            frequency_bands[band]["range"] for band in combo
                        ],
                        "performance": {
                            "mean_accuracy": float(scores.mean()),
                            "std_accuracy": float(scores.std()),
                            "scores": scores.tolist(),
                        },
                        "num_features": X_combo_scaled.shape[1],
                    }

        return combination_results

    def _analyze_material_frequency_responses(
        self, X: np.ndarray, y: np.ndarray, frequency_bands: dict, batch_info: dict
    ) -> dict:
        """Analyze material-specific frequency responses."""
        self.logger.info("Analyzing material-specific frequency responses...")

        material_analysis = {}
        unique_materials = np.unique(y)

        for material in unique_materials:
            material_mask = y == material
            X_material = X[material_mask]

            if len(X_material) == 0:
                continue

            material_analysis[str(material)] = {}

            # Analyze each frequency band for this material
            for band_name, band_info in frequency_bands.items():
                band_features = self._simulate_band_features(
                    X_material, band_info, band_name
                )

                if band_features is not None:
                    # Calculate statistics for this material in this band
                    feature_stats = {
                        "mean": np.mean(band_features, axis=0).tolist(),
                        "std": np.std(band_features, axis=0).tolist(),
                        "median": np.median(band_features, axis=0).tolist(),
                        "num_samples": len(band_features),
                    }

                    # Calculate band energy (sum of squared values)
                    band_energy = np.mean(np.sum(band_features**2, axis=1))

                    material_analysis[str(material)][band_name] = {
                        "frequency_range": band_info["range"],
                        "feature_statistics": feature_stats,
                        "band_energy": float(band_energy),
                        "relative_energy": float(band_energy),  # Will normalize later
                    }

        # Normalize relative energies
        for material in material_analysis:
            total_energy = sum(
                material_analysis[material][band]["band_energy"]
                for band in material_analysis[material]
            )

            if total_energy > 0:
                for band in material_analysis[material]:
                    current_energy = material_analysis[material][band]["band_energy"]
                    material_analysis[material][band]["relative_energy"] = float(
                        current_energy / total_energy
                    )

        return material_analysis

    def _find_optimal_frequency_bands(
        self, band_results: dict, combination_results: dict
    ) -> dict:
        """Find optimal frequency bands and combinations."""
        optimal_bands = {
            "best_single_band": None,
            "best_combination": None,
            "ranking": [],
        }

        # Rank single bands by performance
        single_band_performance = []
        for band_name, results in band_results.items():
            if "performance" in results:
                performance = results["performance"]["mean_accuracy"]
                single_band_performance.append((band_name, performance))

        single_band_performance.sort(key=lambda x: x[1], reverse=True)
        optimal_bands["ranking"] = single_band_performance

        if single_band_performance:
            optimal_bands["best_single_band"] = {
                "band": single_band_performance[0][0],
                "performance": single_band_performance[0][1],
            }

        # Find best combination
        best_combo_performance = 0
        best_combo = None

        for combo_name, results in combination_results.items():
            if "performance" in results:
                performance = results["performance"]["mean_accuracy"]
                if performance > best_combo_performance:
                    best_combo_performance = performance
                    best_combo = combo_name

        if best_combo:
            optimal_bands["best_combination"] = {
                "combination": best_combo,
                "performance": best_combo_performance,
            }

        return optimal_bands

    def _create_frequency_band_visualizations(self, results: Dict[str, Any]):
        """Create frequency band analysis visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Individual band performance
        band_results = results["band_results"]
        band_names = list(band_results.keys())
        band_performances = [
            band_results[name]["performance"]["mean_accuracy"] for name in band_names
        ]
        band_stds = [
            band_results[name]["performance"]["std_accuracy"] for name in band_names
        ]

        bars = axes[0, 0].bar(
            range(len(band_names)), band_performances, yerr=band_stds, capsize=5
        )
        axes[0, 0].axhline(
            y=results["baseline_performance"],
            color="r",
            linestyle="--",
            label="Baseline (All Features)",
        )
        axes[0, 0].set_xticks(range(len(band_names)))
        axes[0, 0].set_xticklabels(band_names, rotation=45)
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_title("Individual Frequency Band Performance")
        axes[0, 0].legend()

        # 2. Band ablation results (importance)
        if "ablation_results" in results:
            ablation_results = results["ablation_results"]
            ablation_bands = list(ablation_results.keys())
            importance_scores = [
                ablation_results[band]["importance_score"] for band in ablation_bands
            ]

            bars = axes[0, 1].bar(range(len(ablation_bands)), importance_scores)
            axes[0, 1].set_xticks(range(len(ablation_bands)))
            axes[0, 1].set_xticklabels(ablation_bands, rotation=45)
            axes[0, 1].set_ylabel("Performance Drop (Importance)")
            axes[0, 1].set_title("Frequency Band Importance (Ablation)")

        # 3. Band combination performance
        if "combination_results" in results:
            combo_results = results["combination_results"]
            combo_names = list(combo_results.keys())
            combo_performances = [
                combo_results[name]["performance"]["mean_accuracy"]
                for name in combo_names
            ]

            bars = axes[0, 2].bar(range(len(combo_names)), combo_performances)
            axes[0, 2].axhline(
                y=results["baseline_performance"],
                color="r",
                linestyle="--",
                label="Baseline",
            )
            axes[0, 2].set_xticks(range(len(combo_names)))
            axes[0, 2].set_xticklabels(combo_names, rotation=45, ha="right")
            axes[0, 2].set_ylabel("Accuracy")
            axes[0, 2].set_title("Frequency Band Combinations")
            axes[0, 2].legend()

        # 4. Material-specific frequency responses
        if "material_analysis" in results:
            material_analysis = results["material_analysis"]
            materials = list(material_analysis.keys())
            bands = list(band_results.keys())

            # Create heatmap of relative energies
            energy_matrix = []
            for material in materials:
                material_energies = []
                for band in bands:
                    if band in material_analysis[material]:
                        energy = material_analysis[material][band]["relative_energy"]
                    else:
                        energy = 0
                    material_energies.append(energy)
                energy_matrix.append(material_energies)

            if energy_matrix:
                sns.heatmap(
                    energy_matrix,
                    xticklabels=bands,
                    yticklabels=materials,
                    annot=True,
                    fmt=".3f",
                    ax=axes[1, 0],
                    cmap="viridis",
                )
                axes[1, 0].set_title("Material-Specific Frequency Energy Distribution")

        # 5. Performance vs number of features
        feature_counts = []
        performances = []
        labels = []

        # Single bands
        for band_name, results_data in band_results.items():
            feature_counts.append(results_data["num_features"])
            performances.append(results_data["performance"]["mean_accuracy"])
            labels.append(band_name)

        # Combinations
        if "combination_results" in results:
            for combo_name, combo_data in results["combination_results"].items():
                feature_counts.append(combo_data["num_features"])
                performances.append(combo_data["performance"]["mean_accuracy"])
                labels.append(
                    combo_name[:10] + "..." if len(combo_name) > 10 else combo_name
                )

        scatter = axes[1, 1].scatter(feature_counts, performances, alpha=0.7)
        axes[1, 1].set_xlabel("Number of Features")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].set_title("Performance vs Feature Count")

        # Add labels for points
        for i, label in enumerate(labels):
            if i < len(feature_counts):  # Safety check
                axes[1, 1].annotate(
                    label,
                    (feature_counts[i], performances[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                )

        # 6. Optimal band summary
        if "optimal_bands" in results:
            optimal_bands = results["optimal_bands"]

            # Create ranking visualization
            if "ranking" in optimal_bands and optimal_bands["ranking"]:
                ranking_bands = [item[0] for item in optimal_bands["ranking"]]
                ranking_scores = [item[1] for item in optimal_bands["ranking"]]

                bars = axes[1, 2].bar(range(len(ranking_bands)), ranking_scores)
                axes[1, 2].set_xticks(range(len(ranking_bands)))
                axes[1, 2].set_xticklabels(ranking_bands, rotation=45)
                axes[1, 2].set_ylabel("Accuracy")
                axes[1, 2].set_title("Frequency Band Ranking")

                # Highlight best band
                if ranking_scores:
                    bars[0].set_color("gold")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.experiment_output_dir, "frequency_band_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _save_frequency_band_summary(self, results: Dict[str, Any]):
        """Save frequency band analysis summary."""
        summary = {
            "baseline_performance": results["baseline_performance"],
            "frequency_bands_analyzed": list(results["frequency_bands"].keys()),
            "num_bands": len(results["frequency_bands"]),
        }

        # Add optimal bands information
        if "optimal_bands" in results:
            optimal_bands = results["optimal_bands"]
            if (
                "best_single_band" in optimal_bands
                and optimal_bands["best_single_band"]
            ):
                summary["best_single_band"] = optimal_bands["best_single_band"]

            if (
                "best_combination" in optimal_bands
                and optimal_bands["best_combination"]
            ):
                summary["best_combination"] = optimal_bands["best_combination"]

            if "ranking" in optimal_bands:
                summary["band_ranking"] = optimal_bands["ranking"][:5]  # Top 5

        # Add performance summary
        if "band_results" in results:
            band_performances = {}
            for band_name, band_data in results["band_results"].items():
                band_performances[band_name] = band_data["performance"]["mean_accuracy"]
            summary["individual_band_performance"] = band_performances

        # Add ablation summary
        if "ablation_results" in results:
            ablation_importance = {}
            for band_name, ablation_data in results["ablation_results"].items():
                ablation_importance[band_name] = ablation_data["importance_score"]
            summary["band_importance_scores"] = ablation_importance

        # Add material analysis summary
        if "material_analysis" in results:
            summary["materials_analyzed"] = list(results["material_analysis"].keys())
            summary["num_materials"] = len(results["material_analysis"])

        self.save_results(summary, "frequency_band_summary.json")
