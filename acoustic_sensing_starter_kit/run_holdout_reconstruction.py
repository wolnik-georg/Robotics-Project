#!/usr/bin/env python3
"""
Run surface reconstruction on the holdout dataset using the trained model.
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.experiments.surface_reconstruction_simple import (
    SurfaceReconstructor,
    run_reconstruction_on_validation_datasets,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        base_dir,
        "training_all_workspaces_holdout_val/discriminationanalysis/trained_models/model_rank1_random_forest.pkl",
    )

    validation_datasets = ["balanced_holdout_undersample"]
    base_data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(
        base_dir, "training_all_workspaces_holdout_val/reconstruction_results"
    )

    logger.info("=" * 60)
    logger.info("SURFACE RECONSTRUCTION ON HOLDOUT DATASET")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Validation datasets: {validation_datasets}")
    logger.info(f"Output: {output_dir}")

    # Run reconstruction
    results = run_reconstruction_on_validation_datasets(
        model_path=model_path,
        validation_datasets=validation_datasets,
        base_data_dir=base_data_dir,
        output_dir=output_dir,
        feature_extractor=None,  # Will be created automatically from model's expected features
        sr=48000,
    )

    logger.info("\n" + "=" * 60)
    logger.info("RECONSTRUCTION COMPLETE")
    logger.info("=" * 60)
    for dataset, result in results.items():
        if "error" in result:
            logger.error(f"  {dataset}: FAILED - {result['error']}")
        else:
            logger.info(f"  {dataset}: Accuracy = {result['accuracy']:.2%}")
            logger.info(f"    Position accuracy: {result.get('accuracy', 'N/A')}")
            logger.info(
                f"    Mean confidence: {result.get('mean_confidence', 'N/A'):.2%}"
            )

    return results


if __name__ == "__main__":
    main()
