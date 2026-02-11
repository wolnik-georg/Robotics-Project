#!/bin/bash

# Run Object Generalization Experiments
# Train on WS1+2+3 (Objects A, B, C), Validate on WS4 (Object D)
# Runs both 3-class and binary experiments sequentially

set -e  # Exit on error

echo "================================================================================"
echo "OBJECT GENERALIZATION EXPERIMENTS - FULLY BALANCED WS4 HOLDOUT"
echo "================================================================================"
echo ""
echo "Training: WS1 + WS2 + WS3 (Objects A, B, C)"
echo "Validation: WS4 (Object D - never seen during training)"
echo ""
echo "Experiments:"
echo "  1. 3-class classification (contact, no_contact, edge)"
echo "  2. Binary classification (contact vs no_contact)"
echo ""
echo "================================================================================"
echo ""

# Experiment 1: 3-class
echo "📊 EXPERIMENT 1/2: 3-Class Object Generalization"
echo "================================================================================"
python3 run_modular_experiments.py configs/object_generalization_3class.yml
echo ""
echo "✅ 3-class experiment complete!"
echo ""

# Experiment 2: Binary
echo "📊 EXPERIMENT 2/2: Binary Object Generalization"
echo "================================================================================"
python3 run_modular_experiments.py configs/object_generalization_binary.yml
echo ""
echo "✅ Binary experiment complete!"
echo ""

# Summary
echo "================================================================================"
echo "✅ ALL OBJECT GENERALIZATION EXPERIMENTS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - object_generalization_ws4_holdout_3class_fully_balanced/"
echo "  - object_generalization_ws4_holdout_binary_fully_balanced/"
echo ""
echo "================================================================================"
