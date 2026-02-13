#!/bin/bash
###############################################################################
# Dataset Balancing Shell Wrapper
###############################################################################
#
# USAGE:
#   bash run_balance_datasets.sh
#
# WHAT IT DOES:
#   Wrapper script that calls create_fully_balanced_datasets.py to create
#   perfectly balanced 3-class datasets for all workspace rotations.
#
# OUTPUTS:
#   data/fully_balanced_datasets/
#       rotation1_train/  (WS1+WS3 combined, balanced)
#       rotation1_val/    (WS2, balanced)
#       rotation2_train/  (WS2+WS3 combined, balanced)
#       rotation2_val/    (WS1, balanced)
#       rotation3_train/  (WS1+WS2 combined, balanced)
#       rotation3_val/    (WS3, balanced)
#       holdout/          (WS4 Object D, balanced)
#
# BALANCE CRITERIA:
#   ✓ Classes: 33/33/33 (contact/no_contact/edge)
#   ✓ Objects: 33/33/33 (A_cutout/B_empty/C_full) in training sets
#   ✓ Workspaces: 50/50 in combined training sets
#
# See README.md Section "Quick Start → 1. Dataset Balancing"
###############################################################################

echo "=============================================================================="
echo "CREATING FULLY BALANCED DATASETS"
echo "=============================================================================="
echo ""
echo "This will create:"
echo "  - Individual balanced workspaces (WS1, WS2, WS3, WS4)"
echo "  - Rotation 1: Train WS1+WS3, Val WS2"
echo "  - Rotation 2: Train WS2+WS3, Val WS1"
echo "  - Rotation 3: Train WS1+WS2, Val WS3"
echo ""
echo "All datasets will be balanced:"
echo "  ✓ Classes: 33/33/33 (contact/no_contact/edge)"
echo "  ✓ Objects: 33/33/33 (A_cutout/B_empty/C_full)"
echo "  ✓ Workspaces: 50/50 (in training sets)"
echo ""
echo "=============================================================================="
echo ""

# Run the balancing script
python create_fully_balanced_datasets.py \
  --ws1-cutout "data/collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled" \
  --ws1-contact "data/collected_data_runs_2026_01_15_workspace_1_pure_contact_relabeled" \
  --ws1-no-contact "data/collected_data_runs_2026_01_15_workspace_1_pure_no_contact" \
  --ws2-cutout "data/collected_data_runs_2026_01_15_workspace_2_squares_cutout_relabeled" \
  --ws2-contact "data/collected_data_runs_2026_01_15_workspace_2_pure_contact_relabeled" \
  --ws2-no-contact "data/collected_data_runs_2026_01_15_workspace_2_pure_no_contact" \
  --ws3-cutout "data/collected_data_runs_2025_12_17_v2_workspace_3_squares_cutout_relabeled" \
  --ws3-contact "data/collected_data_runs_2026_01_14_workspace_3_pure_contact_relabeled" \
  --ws3-no-contact "data/collected_data_runs_2026_01_14_workspace_3_pure_no_contact" \
  --ws4-holdout "data/collected_data_runs_2026_01_27_hold_out_dataset_relabeled" \
  --output-dir "data/fully_balanced_datasets" \
  --seed 42

echo ""
echo "=============================================================================="
echo "DONE!"
echo "=============================================================================="
echo ""
echo "Next steps:"
echo "  1. Check: data/fully_balanced_datasets/balance_summary_report.txt"
echo "  2. Update your config to use rotation datasets"
echo "  3. Run experiments!"
echo ""
