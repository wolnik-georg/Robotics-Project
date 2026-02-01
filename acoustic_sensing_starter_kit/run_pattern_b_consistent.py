"""
Pattern B Reconstruction with CONSISTENT confidence filtering.

Uses the SAME confidence filtering as training:
  - enabled: True
  - threshold: 0.9
  - mode: reject

Pattern B: WS1+WS2+WS3 train → WS4 holdout validation
"""

import sys
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path.cwd() / "src"))
from acoustic_sensing.experiments.surface_reconstruction_simple import (
    SurfaceReconstructor,
)
from acoustic_sensing.core.feature_extraction import GeometricFeatureExtractor

# CONSISTENT confidence filtering (same as training config)
CONFIDENCE_CONFIG = {"enabled": True, "threshold": 0.9, "mode": "reject"}

# Pattern B model
MODEL_PATH = "training_all_workspaces_holdout_val/discriminationanalysis/trained_models/model_rank1_random_forest.pkl"

# TEST datasets (WS1 + WS2 + WS3)
TEST_DATASETS = [
    ("TEST_WS1_squares_cutout", "data/balanced_workspace_1_squares_cutout_oversample"),
    ("TEST_WS1_pure_contact", "data/balanced_workspace_1_pure_contact_oversample"),
    (
        "TEST_WS1_pure_no_contact",
        "data/balanced_workspace_1_pure_no_contact_oversample",
    ),
    ("TEST_WS2_squares_cutout", "data/balanced_workspace_2_squares_cutout"),
    ("TEST_WS2_pure_contact", "data/balanced_workspace_2_pure_contact"),
    ("TEST_WS2_pure_no_contact", "data/balanced_workspace_2_pure_no_contact"),
    ("TEST_WS3_squares_cutout", "data/balanced_workspace_3_squares_cutout_v1"),
    ("TEST_WS3_pure_contact", "data/balanced_workspace_3_pure_contact"),
    ("TEST_WS3_pure_no_contact", "data/balanced_workspace_3_pure_no_contact"),
]

# HOLDOUT dataset (WS4)
HOLDOUT_DATASETS = [
    ("HOLDOUT_WS4", "data/balanced_holdout_undersample"),
]

fe = GeometricFeatureExtractor(
    use_workspace_invariant=True, use_impulse_features=True, sr=48000
)
output_base = Path("pattern_b_consistent_reconstruction")
output_base.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PATTERN B RECONSTRUCTION - CONSISTENT CONFIDENCE FILTERING")
print(
    f"  Confidence: enabled={CONFIDENCE_CONFIG['enabled']}, threshold={CONFIDENCE_CONFIG['threshold']}, mode={CONFIDENCE_CONFIG['mode']}"
)
print("=" * 80)

all_results = []

# Process TEST datasets
print("\n📊 TEST DATASETS (WS1 + WS2 + WS3):")
for ds_name, ds_path in TEST_DATASETS:
    if not Path(ds_path).exists():
        print(f"  ⚠️ {ds_name}: Dataset not found")
        continue

    output_dir = output_base / ds_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rec = SurfaceReconstructor(
        model_path=MODEL_PATH,
        sr=48000,
        confidence_config=CONFIDENCE_CONFIG,
        position_aggregation="none",
        logger=logger,
    )

    results = rec.reconstruct_dataset(ds_path, str(output_dir), feature_extractor=fe)

    all_results.append(
        {
            "type": "TEST",
            "name": ds_name,
            "overall_acc": results["accuracy"] * 100,
            "high_conf_acc": results.get("high_conf_accuracy", 0) * 100,
            "n_samples": results["n_samples"],
            "n_high_conf": results.get("n_high_confidence", 0),
            "pct_high_conf": (
                results.get("pct_high_confidence", 0) * 100
                if results.get("pct_high_confidence")
                else 0
            ),
        }
    )

    print(f"\n  {ds_name}:")
    print(
        f"    Overall: {results['accuracy']*100:.1f}% ({results['n_samples']} samples)"
    )
    if results.get("high_conf_accuracy"):
        print(
            f"    High-conf (≥90%): {results['high_conf_accuracy']*100:.1f}% ({results.get('n_high_confidence')}/{results['n_samples']} = {results.get('pct_high_confidence',0)*100:.1f}%)"
        )

# Process HOLDOUT datasets
print("\n\n📊 HOLDOUT DATASET (WS4):")
for ds_name, ds_path in HOLDOUT_DATASETS:
    if not Path(ds_path).exists():
        print(f"  ⚠️ {ds_name}: Dataset not found")
        continue

    output_dir = output_base / ds_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rec = SurfaceReconstructor(
        model_path=MODEL_PATH,
        sr=48000,
        confidence_config=CONFIDENCE_CONFIG,
        position_aggregation="none",
        logger=logger,
    )

    results = rec.reconstruct_dataset(ds_path, str(output_dir), feature_extractor=fe)

    all_results.append(
        {
            "type": "HOLDOUT",
            "name": ds_name,
            "overall_acc": results["accuracy"] * 100,
            "high_conf_acc": results.get("high_conf_accuracy", 0) * 100,
            "n_samples": results["n_samples"],
            "n_high_conf": results.get("n_high_confidence", 0),
            "pct_high_conf": (
                results.get("pct_high_confidence", 0) * 100
                if results.get("pct_high_confidence")
                else 0
            ),
        }
    )

    print(f"\n  {ds_name}:")
    print(
        f"    Overall: {results['accuracy']*100:.1f}% ({results['n_samples']} samples)"
    )
    if results.get("high_conf_accuracy"):
        print(
            f"    High-conf (≥90%): {results['high_conf_accuracy']*100:.1f}% ({results.get('n_high_confidence')}/{results['n_samples']} = {results.get('pct_high_confidence',0)*100:.1f}%)"
        )

# Summary
print("\n" + "=" * 80)
print("📋 PATTERN B SUMMARY (CONSISTENT WITH TRAINING)")
print("=" * 80)

test_results = [r for r in all_results if r["type"] == "TEST"]
holdout_results = [r for r in all_results if r["type"] == "HOLDOUT"]

if test_results:
    test_overall_avg = np.mean([r["overall_acc"] for r in test_results])
    test_highconf_avg = np.mean(
        [r["high_conf_acc"] for r in test_results if r["high_conf_acc"] > 0]
    )
    test_coverage = np.mean([r["pct_high_conf"] for r in test_results])

    print(f"\nTEST (WS1+WS2+WS3):")
    print(f"  Overall accuracy (all samples):  {test_overall_avg:.1f}%")
    print(f"  High-conf accuracy (≥90% conf):  {test_highconf_avg:.1f}%")
    print(f"  High-conf coverage:              {test_coverage:.1f}%")

if holdout_results:
    holdout_overall_avg = np.mean([r["overall_acc"] for r in holdout_results])
    holdout_highconf_avg = np.mean(
        [r["high_conf_acc"] for r in holdout_results if r["high_conf_acc"] > 0]
    )
    holdout_coverage = np.mean([r["pct_high_conf"] for r in holdout_results])

    print(f"\nHOLDOUT (WS4):")
    print(f"  Overall accuracy (all samples):  {holdout_overall_avg:.1f}%")
    print(f"  High-conf accuracy (≥90% conf):  {holdout_highconf_avg:.1f}%")
    print(f"  High-conf coverage:              {holdout_coverage:.1f}%")

print(f"\nGENERALIZATION GAP (using high-conf accuracy for consistency):")
print(f"  TEST high-conf:    {test_highconf_avg:.1f}%")
print(f"  HOLDOUT high-conf: {holdout_highconf_avg:.1f}%")
print(f"  Gap:               {test_highconf_avg - holdout_highconf_avg:.1f}%")

print(f"\n✅ Output saved to: {output_base}/")
