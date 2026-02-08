#!/bin/bash
# Run complete 3-class experimental suite with dataset rotations
# This script runs 3 different train/validation splits to test generalization

set -e  # Exit on error

echo "=========================================="
echo "3-CLASS EXPERIMENTAL SUITE"
echo "=========================================="
echo ""
echo "This will run 3 dataset rotations:"
echo "  1. Train: WS1+WS3 → Validate: WS2"
echo "  2. Train: WS2+WS3 → Validate: WS1"
echo "  3. Train: WS1+WS2 → Validate: WS3"
echo ""
echo "Each rotation includes:"
echo "  - 5-fold stratified cross-validation on training data"
echo "  - Final model evaluation on held-out validation workspace"
echo "  - Confusion matrices for both CV test and validation"
echo "  - Full 3-class problem (contact, no_contact, edge)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Rotation 1: Train on WS1+WS3, Validate on WS2
echo ""
echo "=========================================="
echo "ROTATION 1: Train WS1+WS3, Validate WS2"
echo "=========================================="
echo "Starting at: $(date)"
python3 run_modular_experiments.py configs/multi_dataset_config.yml
echo "Completed at: $(date)"

# Rotation 2: Train on WS2+WS3, Validate on WS1
echo ""
echo "=========================================="
echo "ROTATION 2: Train WS2+WS3, Validate WS1"
echo "=========================================="
echo "Starting at: $(date)"
python3 run_modular_experiments.py configs/rotation_ws2_ws3_train_ws1_val.yml
echo "Completed at: $(date)"

# Rotation 3: Train on WS1+WS2, Validate on WS3
echo ""
echo "=========================================="
echo "ROTATION 3: Train WS1+WS2, Validate WS3"
echo "=========================================="
echo "Starting at: $(date)"
python3 run_modular_experiments.py configs/rotation_ws1_ws2_train_ws3_val.yml
echo "Completed at: $(date)"

# Summary
echo ""
echo "=========================================="
echo "ALL ROTATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  Rotation 1: test_pipeline_3class_v1/"
echo "  Rotation 2: test_pipeline_3class_rotation2_ws2ws3_train_ws1_val/"
echo "  Rotation 3: test_pipeline_3class_rotation3_ws1ws2_train_ws3_val/"
echo ""
echo "Next steps:"
echo "  1. Compare CV accuracy across rotations (should be consistent)"
echo "  2. Compare validation accuracy (tests workspace generalization)"
echo "  3. Examine confusion matrices (CV vs validation patterns)"
echo "  4. Analyze edge class performance across all rotations"
echo ""
echo "Completed at: $(date)"
