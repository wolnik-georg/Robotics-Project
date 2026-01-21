#!/bin/bash
# Quick Commands for Multi-Dataset Training

echo "🚀 Multi-Dataset Training & Validation - Quick Commands"
echo "========================================================"
echo ""

# Command 1: Run with multi-dataset config
echo "1️⃣  Run Multi-Dataset Training:"
echo "   python3 run_modular_experiments.py configs/multi_dataset_config.yml"
echo ""

# Command 2: Run with standard config (single dataset)
echo "2️⃣  Run Standard Single-Dataset Mode:"
echo "   python3 run_modular_experiments.py configs/experiment_config.yml"
echo ""

# Command 3: View results
echo "3️⃣  View Results:"
echo "   ls modular_analysis_results_multi_dataset/multi_dataset_training/"
echo "   cat modular_analysis_results_multi_dataset/multi_dataset_training/multi_dataset_training_results.json"
echo ""

# Command 4: Open visualizations
echo "4️⃣  Open Visualizations:"
echo "   xdg-open modular_analysis_results_multi_dataset/multi_dataset_training/performance_comparison.png"
echo "   xdg-open modular_analysis_results_multi_dataset/multi_dataset_training/confusion_matrices_best_model.png"
echo "   xdg-open modular_analysis_results_multi_dataset/multi_dataset_training/generalization_analysis.png"
echo ""

echo "📚 Documentation:"
echo "   cat docs/MULTI_DATASET_TRAINING.md"
echo ""

echo "🔧 Configuration:"
echo "   Edit: configs/multi_dataset_config.yml"
echo "   Key settings:"
echo "   - multi_dataset_training.enabled: true/false"
echo "   - multi_dataset_training.training_datasets: [dataset1, dataset2]"
echo "   - multi_dataset_training.validation_dataset: dataset3"
echo ""
