#!/bin/bash
# Create all 3-class balanced datasets (undersample only for speed)

echo "Creating TRAINING datasets..."

# Workspace 3
python3 balance_dataset.py --input data/collected_data_runs_2025_12_17_v2_workspace_3_squares_cutout_relabeled --output balanced_workspace_3_3class_squares_cutout_v1 --method undersample
python3 balance_dataset.py --input data/collected_data_runs_2025_12_17_v2_workspace_3_squares_cutout_relabeled --output balanced_workspace_3_3class_squares_cutout_v2 --method undersample
python3 balance_dataset.py --input data/collected_data_runs_2026_01_14_workspace_3_pure_contact_relabeled --output balanced_workspace_3_3class_pure_contact --method undersample
python3 balance_dataset.py --input data/collected_data_runs_2026_01_14_workspace_3_pure_no_contact --output balanced_workspace_3_3class_pure_no_contact --method undersample

# Workspace 1
python3 balance_dataset.py --input data/collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled --output balanced_workspace_1_3class_squares_cutout --method undersample
python3 balance_dataset.py --input data/collected_data_runs_2026_01_15_workspace_1_pure_no_contact --output balanced_workspace_1_3class_pure_no_contact --method undersample
python3 balance_dataset.py --input data/collected_data_runs_2026_01_15_workspace_1_pure_contact_relabeled --output balanced_workspace_1_3class_pure_contact --method undersample

echo "Creating VALIDATION datasets..."

# Workspace 2
python3 balance_dataset.py --input data/collected_data_runs_2026_01_15_workspace_2_squares_cutout_relabeled --output balanced_workspace_2_3class_squares_cutout --method undersample
python3 balance_dataset.py --input data/collected_data_runs_2026_01_15_workspace_2_pure_no_contact --output balanced_workspace_2_3class_pure_no_contact --method undersample
python3 balance_dataset.py --input data/collected_data_runs_2026_01_15_workspace_2_pure_contact_relabeled --output balanced_workspace_2_3class_pure_contact --method undersample

echo "✅ All 3-class datasets created!"