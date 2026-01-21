from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import os
from pathlib import Path


class BaseExperiment(ABC):
    """
    Abstract base class for all acoustic sensing experiments.
    Provides common functionality and interface for modular experiments.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialize base experiment.

        Args:
            config: Configuration dictionary for this experiment
            output_dir: Base output directory for results
        """
        self.config = config
        self.output_dir = output_dir
        self.experiment_name = self.__class__.__name__.replace("Experiment", "").lower()
        self.logger = self._setup_logger()

        # Create experiment-specific output directory
        self.experiment_output_dir = os.path.join(output_dir, self.experiment_name)
        Path(self.experiment_output_dir).mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Set up experiment-specific logger."""
        logger = logging.getLogger(f"acoustic_sensing.{self.experiment_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"%(asctime)s - {self.experiment_name} - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """
        Return list of experiment names this experiment depends on.
        Dependencies will be executed first.

        Returns:
            List of experiment class names (without 'Experiment' suffix)
        """
        pass

    @abstractmethod
    def run(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the experiment.

        Args:
            shared_data: Dictionary containing results from previous experiments
                        and shared resources like loaded data, features, etc.

        Returns:
            Dictionary containing this experiment's results to be shared
            with subsequent experiments
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this experiment is enabled in configuration."""
        return self.config.get("enabled", False)

    def save_results(self, results: Dict[str, Any], filename: str) -> str:
        """
        Save experiment results to file.

        Args:
            results: Results dictionary to save
            filename: Name of the output file

        Returns:
            Path to saved file
        """
        import pickle
        import json

        filepath = os.path.join(self.experiment_output_dir, filename)

        # Try to save as JSON first (for readability), fallback to pickle
        try:
            if filename.endswith(".json"):
                with open(filepath, "w") as f:
                    json.dump(results, f, indent=2, default=str)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(results, f)
        except (TypeError, ValueError):
            # Fallback to pickle for complex objects
            pickle_filepath = filepath.replace(".json", ".pkl")
            with open(pickle_filepath, "wb") as f:
                pickle.dump(results, f)
            filepath = pickle_filepath

        self.logger.info(f"Results saved to: {filepath}")
        return filepath

    def save_plot(self, fig, filename: str) -> str:
        """
        Save matplotlib figure to file.

        Args:
            fig: Matplotlib figure object
            filename: Name of the output file (e.g., 'plot.png')

        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.experiment_output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        self.logger.info(f"Plot saved to: {filepath}")
        return filepath

    def load_shared_data(
        self, shared_data: Dict[str, Any], key: str, required: bool = True
    ) -> Any:
        """
        Load data from shared data dictionary with error handling.

        Args:
            shared_data: Shared data dictionary
            key: Key to load
            required: Whether this data is required

        Returns:
            Data from shared dictionary

        Raises:
            ValueError: If required data is missing
        """
        if key not in shared_data:
            if required:
                raise ValueError(
                    f"Required data '{key}' not found in shared data. "
                    f"Available keys: {list(shared_data.keys())}"
                )
            else:
                self.logger.warning(f"Optional data '{key}' not found in shared data")
                return None
        return shared_data[key]

    def update_shared_data(
        self, shared_data: Dict[str, Any], updates: Dict[str, Any]
    ) -> None:
        """
        Update shared data dictionary with this experiment's results.

        Args:
            shared_data: Shared data dictionary to update
            updates: Dictionary of updates to apply
        """
        shared_data.update(updates)
        self.logger.info(f"Updated shared data with keys: {list(updates.keys())}")
