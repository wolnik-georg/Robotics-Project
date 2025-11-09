#!/usr/bin/env python3
"""
Comprehensive Advanced Saliency Analysis Test Suite
==================================================

This script tests all advanced saliency analysis components to ensure
they work correctly with your installed dependencies.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("🔬 Testing Advanced Imports...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        import shap
        print(f"✅ SHAP {shap.__version__} imported successfully")
        
        from lime.lime_tabular import LimeTabularExplainer
        print("✅ LIME imported successfully")
        
        import plotly
        print(f"✅ Plotly {plotly.__version__} imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pytorch_models():
    """Test PyTorch model architectures."""
    print("\n🧠 Testing PyTorch CNN Architectures...")
    
    try:
        from saliency_analysis import AdvancedAcousticCNN
        import torch
        
        # Test all architectures
        architectures = [
            ('feature_based', 38, torch.randn(5, 38)),
            ('temporal_only', 55200, torch.randn(5, 55200)),
            ('spectro_temporal', (128, 100), torch.randn(5, 128, 100))
        ]
        
        for arch_name, input_size, test_input in architectures:
            try:
                model = AdvancedAcousticCNN(input_size, 4, arch_name)
                output = model(test_input)
                print(f"✅ {arch_name}: {test_input.shape} -> {output.shape}")
            except Exception as e:
                print(f"❌ {arch_name} failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ PyTorch model test failed: {e}")
        return False

def test_gradient_saliency():
    """Test gradient-based saliency methods."""
    print("\n🎯 Testing Gradient-Based Saliency...")
    
    try:
        from saliency_analysis import AcousticSaliencyAnalyzer
        from batch_specific_analysis import BatchSpecificAnalyzer
        
        # Setup
        batch_analyzer = BatchSpecificAnalyzer()
        saliency_analyzer = AcousticSaliencyAnalyzer(
            batch_analyzer.batch_configs, 
            batch_analyzer.base_dir
        )
        
        # Train a small model
        print("  Training test model...")
        accuracy = saliency_analyzer.train_model(
            'soft_finger_batch_1', 
            data_type='features', 
            epochs=5
        )
        print(f"  Model accuracy: {accuracy:.3f}")
        
        # Test gradient saliency
        print("  Computing gradient saliency...")
        grad_saliency = saliency_analyzer.compute_gradient_saliency(
            'soft_finger_batch_1', 'features', 0
        )
        print(f"  ✅ Gradient saliency shape: {grad_saliency.shape}")
        
        # Test integrated gradients
        print("  Computing integrated gradients...")
        int_grad = saliency_analyzer.compute_integrated_gradients(
            'soft_finger_batch_1', 'features', 0, steps=10
        )
        print(f"  ✅ Integrated gradients shape: {int_grad.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Gradient saliency test failed: {e}")
        traceback.print_exc()
        return False

def test_interpretability_methods():
    """Test SHAP and LIME interpretability."""
    print("\n🔍 Testing Interpretability Methods...")
    
    try:
        from feature_saliency_analysis import SklearnFeatureSaliencyAnalyzer
        from batch_specific_analysis import BatchSpecificAnalyzer
        
        # Setup
        batch_analyzer = BatchSpecificAnalyzer()
        saliency_analyzer = SklearnFeatureSaliencyAnalyzer(
            batch_analyzer.batch_configs, 
            batch_analyzer.base_dir
        )
        
        # Train models
        print("  Training interpretable models...")
        model_results = saliency_analyzer.train_interpretable_models('soft_finger_batch_1')
        print(f"  ✅ Trained {len(model_results)} models")
        
        # Test SHAP
        print("  Computing SHAP values...")
        shap_results = saliency_analyzer.compute_shap_values('soft_finger_batch_1')
        print(f"  ✅ SHAP computed for {len(shap_results['feature_importance'])} features")
        
        # Test LIME
        print("  Computing LIME explanations...")
        lime_results = saliency_analyzer.compute_lime_explanations('soft_finger_batch_1', num_samples=3)
        print(f"  ✅ LIME computed for {len(lime_results['avg_feature_importance'])} features")
        
        return True
    except Exception as e:
        print(f"❌ Interpretability test failed: {e}")
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the complete advanced saliency pipeline."""
    print("\n🚀 Testing Complete Advanced Pipeline...")
    
    try:
        from saliency_analysis import AcousticSaliencyAnalyzer
        from batch_specific_analysis import BatchSpecificAnalyzer
        
        # Setup
        batch_analyzer = BatchSpecificAnalyzer()
        saliency_analyzer = AcousticSaliencyAnalyzer(
            batch_analyzer.batch_configs, 
            batch_analyzer.base_dir
        )
        
        # Run small-scale analysis
        print("  Running batch saliency analysis...")
        results = saliency_analyzer.analyze_batch_saliency('soft_finger_batch_1', num_samples=5)
        
        # Check results structure
        required_keys = ['batch_name', 'config', 'saliency_maps', 'model_performance']
        for key in required_keys:
            if key not in results:
                print(f"❌ Missing key in results: {key}")
                return False
        
        print(f"  ✅ Complete pipeline results generated")
        print(f"  ✅ Model performance: {results['model_performance']}")
        print(f"  ✅ Saliency maps computed for data types: {list(results['saliency_maps'].keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive test suite."""
    print("🔬 ADVANCED SALIENCY ANALYSIS - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print("Testing all advanced saliency analysis components...")
    print("")
    
    tests = [
        ("Import Test", test_imports),
        ("PyTorch Models", test_pytorch_models), 
        ("Gradient Saliency", test_gradient_saliency),
        ("Interpretability Methods", test_interpretability_methods),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"TEST: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print('='*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Advanced saliency analysis is ready!")
        print("\nYou can now use:")
        print("• PyTorch CNN-based saliency maps")
        print("• Gradient and integrated gradient analysis") 
        print("• SHAP value explanations")
        print("• LIME local interpretable explanations")
        print("• Complete advanced saliency pipeline")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())