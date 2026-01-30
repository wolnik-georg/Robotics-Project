from .base_experiment import BaseExperiment
from .data_processing import DataProcessingExperiment
from .dimensionality_reduction import DimensionalityReductionExperiment
from .discrimination_analysis import DiscriminationAnalysisExperiment
from .saliency_analysis import SaliencyAnalysisExperiment
from .feature_ablation import FeatureAblationExperiment
from .impulse_response import ImpulseResponseExperiment
from .frequency_band_ablation import FrequencyBandAblationExperiment
from .multi_dataset_training import MultiDatasetTrainingExperiment
from .surface_reconstruction import SurfaceReconstructionExperiment

from typing import Dict, Any, List, Type
import yaml
import logging
from pathlib import Path
import os


class ExperimentOrchestrator:
    """
    Orchestrates the execution of modular acoustic sensing experiments.
    Handles dependencies, configuration, and result aggregation.
    """

    def __init__(self, config_path: str, output_dir: str):
        """
        Initialize the experiment orchestrator.

        Args:
            config_path: Path to YAML configuration file
            output_dir: Base directory for experiment outputs
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.config = self._load_config()
        self.logger = self._setup_logger()

        # Experiment registry
        self.experiment_classes = {
            "data_processing": DataProcessingExperiment,
            "dimensionality_reduction": DimensionalityReductionExperiment,
            "discrimination_analysis": DiscriminationAnalysisExperiment,
            "saliency_analysis": SaliencyAnalysisExperiment,
            "feature_ablation": FeatureAblationExperiment,
            "impulse_response": ImpulseResponseExperiment,
            "frequency_band_ablation": FrequencyBandAblationExperiment,
            "multi_dataset_training": MultiDatasetTrainingExperiment,
            "surface_reconstruction": SurfaceReconstructionExperiment,
        }

        # Execution state
        self.executed_experiments = {}
        self.shared_data = {}

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {self.config_path}: {str(e)}"
            )

    def _setup_logger(self) -> logging.Logger:
        """Set up orchestrator logger."""
        logger = logging.getLogger("experiment_orchestrator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - ORCHESTRATOR - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def run_experiments(self) -> Dict[str, Any]:
        """
        Run all enabled experiments in dependency order.

        Returns:
            Dictionary containing all experiment results and summary
        """
        self.logger.info("Starting experiment orchestration...")

        # Get enabled experiments
        enabled_experiments = self._get_enabled_experiments()

        if not enabled_experiments:
            self.logger.warning("No experiments are enabled in configuration")
            return {}

        # Resolve execution order based on dependencies
        execution_order = self._resolve_execution_order(enabled_experiments)

        self.logger.info(f"Execution order: {execution_order}")

        # Execute experiments in order
        results = {}
        for experiment_name in execution_order:
            try:
                result = self._execute_experiment(experiment_name)
                results[experiment_name] = result
                self.executed_experiments[experiment_name] = result
                self.logger.info(f"✓ Completed: {experiment_name}")

            except Exception as e:
                self.logger.error(f"✗ Failed: {experiment_name} - {str(e)}")
                results[experiment_name] = {"error": str(e)}

                # Check if this failure blocks other experiments
                if self._is_critical_experiment(experiment_name, execution_order):
                    self.logger.error(
                        f"Critical experiment {experiment_name} failed. Stopping execution."
                    )
                    break

        # Generate overall summary
        summary = self._generate_execution_summary(results)
        results["_execution_summary"] = summary

        # Save orchestrator results
        self._save_orchestrator_results(results)

        self.logger.info("Experiment orchestration completed")
        return results

    def _get_enabled_experiments(self) -> List[str]:
        """Get list of experiments enabled in configuration."""
        enabled = []

        # Always include data processing if any experiments are enabled
        experiments_config = self.config.get("experiments", {})
        if any(
            exp_config.get("enabled", False)
            for exp_config in experiments_config.values()
        ):
            enabled.append("data_processing")

        # Add other enabled experiments
        for exp_name, exp_config in experiments_config.items():
            if exp_config.get("enabled", False):
                enabled.append(exp_name)

        return list(set(enabled))  # Remove duplicates

    def _resolve_execution_order(self, enabled_experiments: List[str]) -> List[str]:
        """
        Resolve experiment execution order based on dependencies.

        Args:
            enabled_experiments: List of enabled experiment names

        Returns:
            List of experiments in execution order
        """
        # Build dependency graph
        dependency_graph = {}

        for exp_name in enabled_experiments:
            if exp_name in self.experiment_classes:
                # Create temporary instance to get dependencies
                temp_exp = self.experiment_classes[exp_name]({}, "")
                dependencies = temp_exp.get_dependencies()

                # Filter dependencies to only include enabled experiments
                dependencies = [
                    dep for dep in dependencies if dep in enabled_experiments
                ]
                dependency_graph[exp_name] = dependencies

        # Topological sort to resolve execution order
        execution_order = []
        remaining = set(enabled_experiments)

        while remaining:
            # Find experiments with no unresolved dependencies
            ready = []
            for exp_name in remaining:
                deps = dependency_graph.get(exp_name, [])
                if all(dep in execution_order for dep in deps):
                    ready.append(exp_name)

            if not ready:
                # Circular dependency or missing dependency
                missing_deps = []
                for exp_name in remaining:
                    deps = dependency_graph.get(exp_name, [])
                    unresolved = [dep for dep in deps if dep not in execution_order]
                    if unresolved:
                        missing_deps.extend(unresolved)

                if missing_deps:
                    self.logger.warning(
                        f"Missing dependencies: {missing_deps}. "
                        f"Adding them to execution order."
                    )
                    for dep in missing_deps:
                        if dep not in execution_order:
                            execution_order.append(dep)
                else:
                    # Add remaining experiments (might have circular dependencies)
                    self.logger.warning(
                        f"Possible circular dependencies. "
                        f"Adding remaining experiments: {list(remaining)}"
                    )
                    execution_order.extend(list(remaining))
                    break

            # Sort ready experiments for deterministic order
            ready.sort()

            # Add ready experiments to execution order
            for exp_name in ready:
                if exp_name not in execution_order:
                    execution_order.append(exp_name)
                remaining.discard(exp_name)

        return execution_order

    def _execute_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """
        Execute a single experiment.

        Args:
            experiment_name: Name of experiment to execute

        Returns:
            Experiment results
        """
        self.logger.info(f"Executing experiment: {experiment_name}")

        # Get experiment class
        if experiment_name not in self.experiment_classes:
            raise ValueError(f"Unknown experiment: {experiment_name}")

        experiment_class = self.experiment_classes[experiment_name]

        # Get experiment configuration
        if experiment_name == "data_processing":
            # Data processing uses global config plus data_processing specific config
            exp_config = {
                "base_data_dir": self.config.get("base_data_dir", "data"),
                "enabled": True,
                "multi_dataset_training": self.config.get("multi_dataset_training", {}),
                "datasets": self.config.get("datasets", []),  # Add datasets field
                "validation_datasets": self.config.get(
                    "validation_datasets", []
                ),  # Add validation datasets
                "hyperparameter_tuning_datasets": self.config.get(
                    "hyperparameter_tuning_datasets", []
                ),  # Add tuning datasets for 3-way split
                "final_test_datasets": self.config.get(
                    "final_test_datasets", []
                ),  # Add final test datasets for 3-way split
                "feature_extraction": self.config.get(
                    "feature_extraction", {}
                ),  # Add feature extraction config (features/spectrogram/both)
                "class_filtering": self.config.get(
                    "class_filtering", {}
                ),  # Add class filtering config (filter edge samples)
                "domain_adaptation": self.config.get(
                    "domain_adaptation", {}
                ),  # Add domain adaptation config (mix hold-out into training)
            }
            # Add data_processing specific config
            data_processing_config = self.config.get("experiments", {}).get(
                "data_processing", {}
            )
            exp_config.update(data_processing_config)
        elif experiment_name == "multi_dataset_training":
            # Multi-dataset training needs top-level multi_dataset_training config
            exp_config = {
                "enabled": self.config.get("experiments", {})
                .get("multi_dataset_training", {})
                .get("enabled", False),
                "multi_dataset_training": self.config.get("multi_dataset_training", {}),
            }
        else:
            exp_config = self.config.get("experiments", {}).get(experiment_name, {})

        # Create experiment instance
        experiment = experiment_class(exp_config, self.output_dir)

        # Check if experiment is enabled
        if not experiment.is_enabled() and experiment_name not in [
            "data_processing",
            "multi_dataset_training",
        ]:
            self.logger.info(f"Experiment {experiment_name} is disabled. Skipping.")
            return {"status": "skipped", "reason": "disabled"}

        # Execute experiment
        try:
            results = experiment.run(self.shared_data)

            # Update shared data with results
            experiment.update_shared_data(self.shared_data, results)

            return {"status": "completed", "results": results}

        except Exception as e:
            self.logger.error(f"Error executing {experiment_name}: {str(e)}")
            raise

    def _is_critical_experiment(
        self, experiment_name: str, execution_order: List[str]
    ) -> bool:
        """
        Check if an experiment is critical (other experiments depend on it).

        Args:
            experiment_name: Name of the failed experiment
            execution_order: Current execution order

        Returns:
            True if experiment is critical
        """
        # Data processing is always critical
        if experiment_name == "data_processing":
            return True

        # Check if any remaining experiments depend on this one
        remaining_experiments = execution_order[
            execution_order.index(experiment_name) + 1 :
        ]

        for remaining_exp in remaining_experiments:
            if remaining_exp in self.experiment_classes:
                temp_exp = self.experiment_classes[remaining_exp]({}, "")
                dependencies = temp_exp.get_dependencies()
                if experiment_name in dependencies:
                    return True

        return False

    def _generate_execution_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate execution summary."""
        summary = {
            "total_experiments": len(results),
            "successful_experiments": 0,
            "failed_experiments": 0,
            "skipped_experiments": 0,
            "experiment_status": {},
            "execution_time": {},  # Could add timing if needed
            "key_findings": [],
        }

        # Count experiment statuses
        for exp_name, exp_result in results.items():
            if exp_name.startswith("_"):  # Skip meta results
                continue

            if isinstance(exp_result, dict):
                status = exp_result.get("status", "unknown")
                summary["experiment_status"][exp_name] = status

                if status == "completed":
                    summary["successful_experiments"] += 1
                elif status == "skipped":
                    summary["skipped_experiments"] += 1
                elif "error" in exp_result:
                    summary["failed_experiments"] += 1

        # Extract key findings
        summary["key_findings"] = self._extract_key_findings(results)

        return summary

    def _extract_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Extract key findings from experiment results."""
        findings = []

        # Data processing findings
        if (
            "data_processing" in results
            and results["data_processing"].get("status") == "completed"
        ):
            data_results = results["data_processing"]["results"]
            findings.append(
                f"Processed {data_results.get('num_samples', 0)} samples with "
                f"{data_results.get('num_features', 0)} features across "
                f"{data_results.get('num_classes', 0)} classes"
            )

        # Discrimination analysis findings
        if (
            "discrimination_analysis" in results
            and results["discrimination_analysis"].get("status") == "completed"
        ):
            disc_results = results["discrimination_analysis"]["results"]
            best_classifier = disc_results.get("best_classifier", {})
            if best_classifier:
                findings.append(
                    f"Best classifier: {best_classifier.get('name', 'Unknown')} "
                    f"with {best_classifier.get('validation_accuracy', 0):.4f} validation accuracy"
                )

        # Feature ablation findings
        if (
            "feature_ablation" in results
            and results["feature_ablation"].get("status") == "completed"
        ):
            ablation_results = results["feature_ablation"]["results"]
            synthesis = ablation_results.get("synthesis", {})
            key_insights = synthesis.get("key_insights", [])
            findings.extend(key_insights[:3])  # Add top 3 insights

        # Saliency analysis findings
        if (
            "saliency_analysis" in results
            and results["saliency_analysis"].get("status") == "completed"
        ):
            saliency_results = results["saliency_analysis"]["results"]
            feature_analysis = saliency_results.get("feature_analysis", {})
            consistent_features = feature_analysis.get("consistently_important", [])
            if consistent_features:
                findings.append(
                    f"Found {len(consistent_features)} consistently important features "
                    f"across saliency methods"
                )

        # Frequency band findings
        if (
            "frequency_band_ablation" in results
            and results["frequency_band_ablation"].get("status") == "completed"
        ):
            freq_results = results["frequency_band_ablation"]["results"]
            optimal_bands = freq_results.get("optimal_bands", {})
            if "best_single_band" in optimal_bands:
                best_band = optimal_bands["best_single_band"]
                findings.append(
                    f"Most informative frequency band: {best_band.get('band', 'Unknown')} "
                    f"({best_band.get('performance', 0):.4f} accuracy)"
                )

        return findings

    def _save_orchestrator_results(self, results: Dict[str, Any]):
        """Save orchestrator results and configuration."""
        import json
        import pickle
        from datetime import datetime

        # Save configuration used
        config_save_path = os.path.join(self.output_dir, "experiment_config_used.yml")
        with open(config_save_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Save execution summary as JSON
        summary_path = os.path.join(self.output_dir, "execution_summary.json")
        summary_data = {
            "execution_timestamp": datetime.now().isoformat(),
            "config_path": self.config_path,
            "output_directory": self.output_dir,
            "summary": results.get("_execution_summary", {}),
            "experiment_list": list(results.keys()),
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        # Save full results as pickle (for complex objects)
        results_path = os.path.join(self.output_dir, "full_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

        self.logger.info(f"Results saved to {self.output_dir}")
        self.logger.info(f"Summary available at: {summary_path}")

    def get_experiment_result(self, experiment_name: str) -> Any:
        """Get results from a specific experiment."""
        if experiment_name in self.executed_experiments:
            return self.executed_experiments[experiment_name]
        else:
            raise ValueError(
                f"Experiment '{experiment_name}' has not been executed or failed"
            )

    def list_available_experiments(self) -> List[str]:
        """List all available experiments."""
        return list(self.experiment_classes.keys())

    def validate_config(self) -> Dict[str, Any]:
        """Validate the configuration file."""
        validation_results = {"valid": True, "errors": [], "warnings": []}

        # Check required sections
        required_sections = ["experiments", "output", "base_data_dir"]
        for section in required_sections:
            if section not in self.config:
                validation_results["errors"].append(
                    f"Missing required section: {section}"
                )

        # Check experiment configurations
        experiments_config = self.config.get("experiments", {})
        for exp_name, exp_config in experiments_config.items():
            if exp_name not in self.experiment_classes:
                validation_results["warnings"].append(f"Unknown experiment: {exp_name}")

            if not isinstance(exp_config, dict):
                validation_results["errors"].append(
                    f"Invalid config for {exp_name}: must be dict"
                )

        # Check data directory
        base_data_dir = self.config.get("base_data_dir", "data")
        full_data_path = f"/home/georg/Desktop/Robotics-Project/acoustic_sensing_starter_kit/{base_data_dir}"
        if not os.path.exists(full_data_path):
            validation_results["warnings"].append(
                f"Data directory does not exist: {full_data_path}"
            )

        validation_results["valid"] = len(validation_results["errors"]) == 0

        return validation_results
