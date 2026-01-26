#!/usr/bin/env python3
"""
Verify that the 3-way split is correctly separating datasets.
"""
import sys
import yaml
from pathlib import Path

config_file = "configs/3way_split_config.yml"

print("=" * 70)
print("VERIFYING 3-WAY SPLIT DATASET SEPARATION")
print("=" * 70)

# Load config
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Extract dataset lists
training_datasets = config.get("datasets", [])
tuning_datasets = config.get("hyperparameter_tuning_datasets", [])
test_datasets = config.get("final_test_datasets", [])

print("\n📋 CONFIG FILE DATASETS:")
print(f"\n1️⃣  Training Datasets ({len(training_datasets)}):")
for ds in training_datasets:
    print(f"   - {ds}")

print(f"\n2️⃣  Tuning Datasets ({len(tuning_datasets)}):")
for ds in tuning_datasets:
    print(f"   - {ds}")

print(f"\n3️⃣  Test Datasets ({len(test_datasets)}):")
for ds in test_datasets:
    print(f"   - {ds}")

# Check for overlaps
print("\n🔍 CHECKING FOR DATASET OVERLAP:")
train_set = set(training_datasets)
tune_set = set(tuning_datasets)
test_set = set(test_datasets)

train_tune_overlap = train_set & tune_set
train_test_overlap = train_set & test_set
tune_test_overlap = tune_set & test_set

if train_tune_overlap:
    print(f"❌ OVERLAP between training and tuning: {train_tune_overlap}")
else:
    print(f"✅ No overlap between training and tuning")

if train_test_overlap:
    print(f"❌ OVERLAP between training and test: {train_test_overlap}")
else:
    print(f"✅ No overlap between training and test")

if tune_test_overlap:
    print(f"❌ OVERLAP between tuning and test: {tune_test_overlap}")
else:
    print(f"✅ No overlap between tuning and test")

# Check workspace separation
print("\n🏢 WORKSPACE VERIFICATION:")


def get_workspace(dataset_name):
    if "workspace_1" in dataset_name or "workspace1" in dataset_name:
        return 1
    elif "workspace_2" in dataset_name or "workspace2" in dataset_name:
        return 2
    elif (
        "workspace_3" in dataset_name
        or "workspace3" in dataset_name
        or "workspace_3" in dataset_name.lower()
    ):
        return 3
    return None


train_workspaces = set(get_workspace(ds) for ds in training_datasets)
tune_workspaces = set(get_workspace(ds) for ds in tuning_datasets)
test_workspaces = set(get_workspace(ds) for ds in test_datasets)

print(
    f"Training datasets from workspaces: {sorted([w for w in train_workspaces if w])}"
)
print(f"Tuning datasets from workspaces: {sorted([w for w in tune_workspaces if w])}")
print(f"Test datasets from workspaces: {sorted([w for w in test_workspaces if w])}")

# Expected: Training=2, Tuning=3, Test=1
if train_workspaces == {2} and tune_workspaces == {3} and test_workspaces == {1}:
    print("✅ PERFECT: Training=WS2, Tuning=WS3, Test=WS1")
else:
    print("⚠️  Warning: Workspace separation may not be ideal")

print("\n" + "=" * 70)
print("✅ CONFIG VERIFICATION COMPLETE")
print("=" * 70)
print("\nNow checking if the CODE is using these correctly...")
print("Run the pipeline and check logs for:")
print("  - 'Combined training data: X samples' ")
print("  - 'Combined tuning data: Y samples'")
print("  - 'Combined final test data: Z samples'")
print("\nThe samples should match ONLY their respective datasets!")
