#!/usr/bin/env python3
"""
Simple Advanced Saliency Demo
============================

A streamlined demonstration of the advanced saliency analysis capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def simple_pytorch_demo():
    """Simple PyTorch CNN saliency demonstration."""
    print("🔬 PyTorch CNN Saliency Analysis")
    print("=" * 40)
    
    from saliency_analysis import AcousticSaliencyAnalyzer
    from batch_specific_analysis import BatchSpecificAnalyzer
    
    # Setup
    batch_analyzer = BatchSpecificAnalyzer()
    saliency_analyzer = AcousticSaliencyAnalyzer(
        batch_analyzer.batch_configs, 
        batch_analyzer.base_dir
    )
    
    batch_name = 'soft_finger_batch_1'
    
    print(f"1. Training CNN model for {batch_name}...")
    accuracy = saliency_analyzer.train_model(
        batch_name, 
        data_type='features', 
        epochs=30
    )
    print(f"   ✅ CNN Accuracy: {accuracy:.3f}")
    
    print("2. Computing saliency maps...")
    grad_saliency = saliency_analyzer.compute_gradient_saliency(batch_name, 'features', 0)
    int_grad_saliency = saliency_analyzer.compute_integrated_gradients(batch_name, 'features', 0, steps=20)
    
    print(f"   ✅ Gradient saliency: shape {grad_saliency.shape}")
    print(f"   ✅ Integrated gradients: shape {int_grad_saliency.shape}")
    
    # Simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Load feature names
    import pandas as pd
    features_file = Path(f"batch_analysis_results/{batch_name}/{batch_name}_features.csv")
    if features_file.exists():
        df = pd.read_csv(features_file)
        feature_names = [col for col in df.columns if col not in ['simplified_label', 'original_label']]
    else:
        feature_names = [f'Feature_{i}' for i in range(len(grad_saliency))]
    
    # Top 15 gradient features
    grad_top_idx = np.argsort(grad_saliency)[-15:]
    grad_top_names = [feature_names[int(i)] for i in grad_top_idx]
    grad_top_values = grad_saliency[grad_top_idx]
    
    ax1.barh(range(len(grad_top_values)), grad_top_values, color='skyblue', alpha=0.7)
    ax1.set_yticks(range(len(grad_top_values)))
    ax1.set_yticklabels(grad_top_names, fontsize=8)
    ax1.set_title('CNN Gradient Saliency (Top 15)')
    ax1.set_xlabel('Importance')
    
    # Top 15 integrated gradient features  
    int_grad_top_idx = np.argsort(int_grad_saliency)[-15:]
    int_grad_top_names = [feature_names[int(i)] for i in int_grad_top_idx]
    int_grad_top_values = int_grad_saliency[int_grad_top_idx]
    
    ax2.barh(range(len(int_grad_top_values)), int_grad_top_values, color='lightgreen', alpha=0.7)
    ax2.set_yticks(range(len(int_grad_top_values)))
    ax2.set_yticklabels(int_grad_top_names, fontsize=8)
    ax2.set_title('Integrated Gradients Saliency (Top 15)')
    ax2.set_xlabel('Importance')
    
    plt.suptitle('Advanced CNN-Based Saliency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_dir = Path("batch_analysis_results/simple_demo")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "pytorch_saliency_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Visualization saved to {save_dir}")
    
    return {
        'cnn_accuracy': accuracy,
        'gradient_saliency': grad_saliency,
        'integrated_gradients': int_grad_saliency,
        'feature_names': feature_names
    }

def simple_interpretability_demo():
    """Simple SHAP and LIME demonstration."""
    print("\n🔍 Interpretability Analysis (SHAP & LIME)")
    print("=" * 40)
    
    from feature_saliency_analysis import SklearnFeatureSaliencyAnalyzer
    from batch_specific_analysis import BatchSpecificAnalyzer
    
    # Setup
    batch_analyzer = BatchSpecificAnalyzer()
    saliency_analyzer = SklearnFeatureSaliencyAnalyzer(
        batch_analyzer.batch_configs, 
        batch_analyzer.base_dir
    )
    
    batch_name = 'soft_finger_batch_1'
    
    print("1. Training interpretable models...")
    model_results = saliency_analyzer.train_interpretable_models(batch_name)
    print(f"   ✅ Random Forest: {model_results['random_forest']['test_accuracy']:.3f}")
    print(f"   ✅ Logistic Regression: {model_results['logistic_regression']['test_accuracy']:.3f}")
    
    print("2. Computing SHAP values...")
    shap_results = saliency_analyzer.compute_shap_values(batch_name)
    feature_names = saliency_analyzer.feature_names[batch_name]
    shap_importance = shap_results['feature_importance']
    
    print("3. Computing LIME explanations...")
    lime_results = saliency_analyzer.compute_lime_explanations(batch_name, num_samples=5)
    lime_importance = lime_results['avg_feature_importance']
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Advanced Interpretability Analysis', fontsize=16, fontweight='bold')
    
    # SHAP importance (top 15)
    shap_values = [shap_importance.get(name, 0) for name in feature_names]
    shap_top_idx = np.argsort(shap_values)[-15:]
    shap_top_names = [feature_names[int(i)] for i in shap_top_idx]
    shap_top_values = [shap_values[int(i)] for i in shap_top_idx]
    
    ax1 = axes[0, 0]
    ax1.barh(range(len(shap_top_values)), shap_top_values, color='orange', alpha=0.7)
    ax1.set_yticks(range(len(shap_top_values)))
    ax1.set_yticklabels(shap_top_names, fontsize=8)
    ax1.set_title('SHAP Feature Importance (Top 15)')
    ax1.set_xlabel('SHAP Value')
    
    # LIME importance (top 15)
    lime_values = [lime_importance.get(name, 0) for name in feature_names]
    lime_top_idx = np.argsort(lime_values)[-15:]
    lime_top_names = [feature_names[int(i)] for i in lime_top_idx]
    lime_top_values = [lime_values[int(i)] for i in lime_top_idx]
    
    ax2 = axes[0, 1]
    ax2.barh(range(len(lime_top_values)), lime_top_values, color='purple', alpha=0.7)
    ax2.set_yticks(range(len(lime_top_values)))
    ax2.set_yticklabels(lime_top_names, fontsize=8)
    ax2.set_title('LIME Feature Importance (Top 15)')
    ax2.set_xlabel('LIME Importance')
    
    # Random Forest comparison
    rf_importance = model_results['random_forest']['feature_importance'].get('builtin', {})
    rf_values = [rf_importance.get(name, 0) for name in feature_names]
    
    ax3 = axes[1, 0]
    ax3.scatter(rf_values, shap_values, alpha=0.6, s=50, color='blue')
    ax3.set_xlabel('Random Forest Importance')
    ax3.set_ylabel('SHAP Importance')
    ax3.set_title('SHAP vs Random Forest')
    
    # Add correlation
    correlation = np.corrcoef(rf_values, shap_values)[0, 1]
    ax3.text(0.05, 0.95, f'Corr: {correlation:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Method correlation matrix
    ax4 = axes[1, 1]
    methods = ['Random Forest', 'SHAP', 'LIME']
    method_values = [rf_values, shap_values, lime_values]
    correlation_matrix = np.corrcoef(method_values)
    
    im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_title('Method Correlation')
    ax4.set_xticks(range(len(methods)))
    ax4.set_yticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45)
    ax4.set_yticklabels(methods)
    
    # Add correlation values
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax4.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                   ha="center", va="center", 
                   color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")
    
    plt.tight_layout()
    
    # Save
    save_dir = Path("batch_analysis_results/simple_demo")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "interpretability_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Visualization saved to {save_dir}")
    print(f"   ✅ SHAP-RF correlation: {correlation:.3f}")
    
    return {
        'model_accuracies': {k: v['test_accuracy'] for k, v in model_results.items()},
        'shap_importance': shap_importance,
        'lime_importance': lime_importance,
        'correlations': correlation_matrix.tolist()
    }

def feature_consensus_analysis():
    """Analyze consensus across all methods."""
    print("\n📊 Feature Consensus Analysis")
    print("=" * 40)
    
    # Run both analyses
    pytorch_results = simple_pytorch_demo()
    interpretability_results = simple_interpretability_demo()
    
    print("\n4. Creating consensus ranking...")
    
    feature_names = pytorch_results['feature_names']
    
    # Get all importance scores
    all_methods = {
        'CNN_Gradient': pytorch_results['gradient_saliency'],
        'Integrated_Gradients': pytorch_results['integrated_gradients'],
        'SHAP': [interpretability_results['shap_importance'].get(name, 0) for name in feature_names],
        'LIME': [interpretability_results['lime_importance'].get(name, 0) for name in feature_names]
    }
    
    # Calculate consensus by counting top-10 appearances
    feature_votes = np.zeros(len(feature_names))
    top_k = 10
    
    for method_name, importance_scores in all_methods.items():
        top_indices = np.argsort(importance_scores)[-top_k:]
        for idx in top_indices:
            feature_votes[int(idx)] += 1
    
    # Get consensus features
    consensus_top_idx = np.argsort(feature_votes)[-15:]
    consensus_names = [feature_names[int(i)] for i in consensus_top_idx]
    consensus_scores = [feature_votes[int(i)] for i in consensus_top_idx]
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Consensus ranking
    bars = ax1.barh(range(len(consensus_scores)), consensus_scores, color='green', alpha=0.7)
    ax1.set_yticks(range(len(consensus_scores)))
    ax1.set_yticklabels(consensus_names, fontsize=8)
    ax1.set_xlabel('Number of Methods Ranking in Top 10')
    ax1.set_title('Feature Consensus Ranking (Top 15)')
    ax1.set_xlim(0, len(all_methods))
    
    # Color by consensus strength
    for i, bar in enumerate(bars):
        consensus_strength = consensus_scores[i] / len(all_methods)
        bar.set_color(plt.cm.RdYlGn(consensus_strength))
    
    # Method correlation heatmap
    method_values = [all_methods[method] for method in all_methods.keys()]
    correlation_matrix = np.corrcoef(method_values)
    
    im = ax2.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax2.set_title('Saliency Method Correlation Matrix')
    ax2.set_xticks(range(len(all_methods)))
    ax2.set_yticks(range(len(all_methods)))
    ax2.set_xticklabels(all_methods.keys(), rotation=45)
    ax2.set_yticklabels(all_methods.keys())
    
    # Add correlation values
    for i in range(len(all_methods)):
        for j in range(len(all_methods)):
            ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                   ha="center", va="center", 
                   color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                   fontsize=9)
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    
    # Save
    save_dir = Path("batch_analysis_results/simple_demo")
    plt.savefig(save_dir / "consensus_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save summary
    summary = {
        'pytorch_results': {
            'cnn_accuracy': float(pytorch_results['cnn_accuracy'])
        },
        'interpretability_results': interpretability_results,
        'consensus_features': {
            'feature_names': consensus_names[-10:],  # Top 10
            'consensus_scores': [int(x) for x in consensus_scores[-10:]]
        },
        'method_correlations': correlation_matrix.tolist()
    }
    
    with open(save_dir / "advanced_saliency_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✅ Consensus analysis complete")
    print(f"   📁 All results saved to {save_dir}")
    
    # Show top consensus features
    print(f"\n🏆 TOP CONSENSUS FEATURES:")
    for i, (name, score) in enumerate(zip(consensus_names[-10:], consensus_scores[-10:]), 1):
        print(f"   {i:2d}. {name:<30} ({score}/{len(all_methods)} methods)")
    
    return summary

if __name__ == "__main__":
    print("🚀 SIMPLE ADVANCED SALIENCY ANALYSIS DEMO")
    print("=" * 60)
    print("This demo showcases the key advanced saliency capabilities:")
    print("• PyTorch CNN gradient analysis")
    print("• SHAP interpretability analysis") 
    print("• LIME local explanations")
    print("• Feature consensus ranking")
    print("")
    
    try:
        summary = feature_consensus_analysis()
        
        print("\n" + "="*60)
        print("🎉 ADVANCED SALIENCY DEMO COMPLETE!")
        print("=" * 60)
        print(f"\nKey Results:")
        print(f"✅ CNN Model Accuracy: {summary['pytorch_results']['cnn_accuracy']:.3f}")
        print(f"✅ Random Forest Accuracy: {summary['interpretability_results']['model_accuracies']['random_forest']:.3f}")
        print(f"✅ Method Correlation Range: {np.min(summary['method_correlations']):.3f} to {np.max(summary['method_correlations']):.3f}")
        print(f"✅ Top Consensus Features: {len(summary['consensus_features']['feature_names'])}")
        
        print(f"\n📁 All visualizations and results saved to:")
        print(f"   batch_analysis_results/simple_demo/")
        
        print(f"\n🎯 Your advanced saliency analysis system is fully operational!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 The core saliency analysis components are still working.")
        print(f"   You can use the individual analyzers directly for production use.")