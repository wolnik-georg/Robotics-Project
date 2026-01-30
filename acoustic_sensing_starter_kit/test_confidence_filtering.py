#!/usr/bin/env python3
"""
Quick test script to verify confidence filtering works.

This script:
1. Checks the config has confidence filtering enabled
2. Shows how to enable it
3. Explains what the output will look like
"""

import yaml
from pathlib import Path

# Load config
config_path = Path("configs/multi_dataset_config.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("CONFIDENCE FILTERING CONFIGURATION CHECK")
print("=" * 70)

# Check current configuration
conf_config = config.get("discrimination_analysis", {}).get("confidence_filtering", {})

print(f"\nCurrent Configuration:")
print(f"  Enabled: {conf_config.get('enabled', False)}")
print(f"  Threshold: {conf_config.get('threshold', 0.7)}")
print(f"  Mode: {conf_config.get('mode', 'reject')}")
print(f"  Default Class: {conf_config.get('default_class', 'no_contact')}")

print(f"\n" + "=" * 70)
print("HOW TO ENABLE CONFIDENCE FILTERING")
print("=" * 70)

print(
    """
1. Edit configs/multi_dataset_config.yml

2. Under 'discrimination_analysis:', change:
   
   confidence_filtering:
     enabled: true  # <- Change this from false to true
     threshold: 0.7  # Adjust if needed (0.0-1.0)
     mode: "reject"  # or "default"
     default_class: "no_contact"  # Safe default for robotics

3. Run your experiment:
   python3 run_modular_experiments.py configs/multi_dataset_config.yml
"""
)

print("=" * 70)
print("WHAT TO EXPECT IN THE OUTPUT")
print("=" * 70)

print(
    """
With confidence filtering enabled, you'll see in the logs:

  🔍 Applying confidence filtering to TRAIN set:
    📊 Confidence Filtering (threshold=0.7):
      Kept: 1200/1356 (88.5%)
      Rejected: 156/1356 (11.5%)
      Mean confidence: 0.823
      Median confidence: 0.851
  
  🔍 Applying confidence filtering to VALIDATION set:
    📊 Confidence Filtering (threshold=0.7):
      Kept: 1300/1520 (85.5%)
      Rejected: 220/1520 (14.5%)
      Mean confidence: 0.798
      Median confidence: 0.812

The accuracies reported will be for HIGH-CONFIDENCE predictions only.

TWO MODES:

1. "reject" mode (current):
   - Removes low-confidence predictions from evaluation
   - Shows accuracy on only high-confidence samples
   - Use this to see: "If I only trust confident predictions, how accurate are they?"
   
2. "default" mode:
   - Assigns safe default class (e.g., "no_contact") to low-confidence predictions
   - Use this for robotics: "When uncertain, assume no contact for safety"
   - All samples kept in evaluation, but uncertain → default class
"""
)

print("=" * 70)
print("CONFIDENCE STATISTICS SAVED")
print("=" * 70)

print(
    """
The confidence statistics for each classifier are saved in results:

results_dict[clf_name]["validation_confidence_stats"] = {
    "total_samples": 1520,
    "high_confidence": 1300,
    "low_confidence": 220,
    "high_confidence_pct": 85.5,
    "low_confidence_pct": 14.5,
    "mean_confidence": 0.798,
    "median_confidence": 0.812,
    "min_confidence": 0.501,
    "max_confidence": 0.999,
}

You can analyze these to:
- See which models are overconfident (high confidence but wrong)
- Find optimal threshold value
- Compare confidence distributions across models
"""
)

print("=" * 70)
print("EXAMPLE USE CASES")
print("=" * 70)

print(
    """
For Robotic Control:

1. Set mode="default" and default_class="no_contact"
   → When robot is uncertain, assume no contact (safer)
   
2. Set threshold=0.8 (high confidence required)
   → Only act on very confident predictions
   
3. Monitor low_confidence_pct:
   → If >30% predictions are uncertain, model needs improvement

For Model Analysis:

1. Set mode="reject" and threshold=0.7
   → See accuracy on confident predictions only
   
2. Compare validation_accuracy vs validation_confidence_stats
   → If accuracy high but many rejected, model is uncertain
   
3. Try different thresholds (0.5, 0.6, 0.7, 0.8, 0.9)
   → Find trade-off between coverage and accuracy
"""
)

print("=" * 70)

if __name__ == "__main__":
    print("\n✅ Configuration check complete!")
    print(
        f"   To enable: Edit {config_path} and set confidence_filtering.enabled: true"
    )
