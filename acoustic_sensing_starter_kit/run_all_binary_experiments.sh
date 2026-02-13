#!/bin/bash
###############################################################################
# Run All 3 Binary Classification Experiments (for 3-Class vs Binary Comparison)
###############################################################################
#
# USAGE:
#   bash run_all_binary_experiments.sh
#
# WHAT IT DOES:
#   Runs all 3 workspace rotations in BINARY mode (contact vs no_contact, 
#   excluding edge samples) for comparison against 3-class results.
#
#   Uses binary-specific configs:
#       - configs/rotation1_binary.yml
#       - configs/rotation2_binary.yml
#       - configs/rotation3_binary.yml
#
# OUTPUTS:
#   fully_balanced_rotation1_binary/
#   fully_balanced_rotation2_binary/
#   fully_balanced_rotation3_binary/
#
# KEY FINDING:
#   Binary performs WORSE than random guessing!
#       - 3-class: 34.5% validation (1.04× over 33.3% random)
#       - Binary: 45.1% validation (0.90× over 50.0% random) ⚠️
#   
#   Proves edge samples contain essential discriminative information.
#
# See README.md Section "Performance Summary → 3-Class vs Binary"
###############################################################################

echo "=================================================="
echo "Running Binary Classification Experiments"
echo "=================================================="
echo "Mode: Binary (contact vs no_contact, edge excluded)"
echo "Random baseline: 50% (vs 33.3% for 3-class)"
echo "Expected time: ~3 hours (1 hour per rotation)"
echo "=================================================="
echo ""

# Rotation 1: Train WS1+WS3, Validate WS2
echo "🔄 ROTATION 1: Training on WS1+WS3, validating on WS2..."
python3 run_modular_experiments.py configs/rotation1_binary.yml
if [ $? -eq 0 ]; then
    echo "✅ Rotation 1 complete!"
else
    echo "❌ Rotation 1 failed!"
    exit 1
fi
echo ""

# Rotation 2: Train WS2+WS3, Validate WS1
echo "🔄 ROTATION 2: Training on WS2+WS3, validating on WS1..."
python3 run_modular_experiments.py configs/rotation2_binary.yml
if [ $? -eq 0 ]; then
    echo "✅ Rotation 2 complete!"
else
    echo "❌ Rotation 2 failed!"
    exit 1
fi
echo ""

# Rotation 3: Train WS1+WS2, Validate WS3
echo "🔄 ROTATION 3: Training on WS1+WS2, validating on WS3..."
python3 run_modular_experiments.py configs/rotation3_binary.yml
if [ $? -eq 0 ]; then
    echo "✅ Rotation 3 complete!"
else
    echo "❌ Rotation 3 failed!"
    exit 1
fi
echo ""

echo "=================================================="
echo "✅ ALL BINARY EXPERIMENTS COMPLETE!"
echo "=================================================="
echo "Results saved to:"
echo "  - fully_balanced_rotation1_binary/"
echo "  - fully_balanced_rotation2_binary/"
echo "  - fully_balanced_rotation3_binary/"
echo ""
echo "Next steps:"
echo "  1. Compare binary vs 3-class validation accuracy"
echo "  2. Calculate normalized performance: (val_acc - 50%) / 50%"
echo "  3. Compare to 3-class normalized: (val_acc - 33.3%) / 33.3%"
echo "  4. Decision: Keep Section IV.D if 3-class wins!"
echo "=================================================="
