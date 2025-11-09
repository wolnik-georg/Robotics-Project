#!/usr/bin/env python3
"""
Advanced Saliency Analysis Demo
==============================

This script demonstrates the full capabilities of the advanced saliency analysis
system for acoustic geometric discrimination.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def demo_pytorch_saliency():
    """Demonstrate PyTorch-based saliency analysis."""
    print("🔬 PyTorch CNN Saliency Analysis Demo")
    print("=" * 50)
    
    from saliency_analysis import AcousticSaliencyAnalyzer
    from batch_specific_analysis import BatchSpecificAnalyzer
    
    # Setup
    batch_analyzer = BatchSpecificAnalyzer()
    saliency_analyzer = AcousticSaliencyAnalyzer(
        batch_analyzer.batch_configs, 
        batch_analyzer.base_dir
    )
    
    # Analyze one batch with full advanced features
    batch_name = 'soft_finger_batch_1'
    print(f"Analyzing {batch_name} with advanced methods...")
    
    # Train CNN model
    print("\n1. Training CNN model...")
    accuracy = saliency_analyzer.train_model(
        batch_name, 
        data_type='features', 
        architecture='feature_based',
        epochs=50
    )
    print(f"   CNN Model Accuracy: {accuracy:.3f}")
    
    # Compute different saliency methods
    print("\n2. Computing saliency maps...")
    methods = {}
    
    # Gradient-based saliency
    grad_saliency = saliency_analyzer.compute_gradient_saliency(batch_name, 'features', 0)
    methods['Gradient'] = grad_saliency
    
    # Integrated gradients (more robust)
    int_grad_saliency = saliency_analyzer.compute_integrated_gradients(batch_name, 'features', 0)
    methods['Integrated Gradients'] = int_grad_saliency
    
    # LIME explanation
    lime_result = saliency_analyzer.compute_lime_explanation(batch_name, 0)
    lime_importance = np.array([lime_result['feature_importance'].get(f'feature_{i}', 0) for i in range(38)])
    methods['LIME'] = np.abs(lime_importance)
    
    print("   ✅ Gradient saliency computed")
    print("   ✅ Integrated gradients computed") 
    print("   ✅ LIME explanations computed")
    
    # Create comparison plot
    print("\n3. Creating advanced saliency visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Advanced CNN-Based Saliency Analysis\nBatch: soft_finger_batch_1', fontsize=16)
    
    # Load feature names
    features_file = Path(f"batch_analysis_results/{batch_name}/{batch_name}_features.csv")
    if features_file.exists():
        import pandas as pd
        df = pd.read_csv(features_file)
        feature_names = [col for col in df.columns if col not in ['simplified_label', 'original_label']]
    else:
        feature_names = [f'Feature_{i}' for i in range(38)]
    
    # Plot each method
    for idx, (method_name, saliency) in enumerate(methods.items()):
        if idx < 3:  # First 3 plots
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Get top features
            top_indices = np.argsort(saliency)[-15:]
            top_values = saliency[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            bars = ax.barh(range(len(top_values)), top_values, 
                          color=plt.cm.viridis(idx/3), alpha=0.7)
            ax.set_yticks(range(len(top_values)))
            ax.set_yticklabels(top_names, fontsize=8)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{method_name} Saliency')
            
            # Highlight top 5
            for i in range(max(0, len(bars)-5), len(bars)):
                bars[i].set_alpha(1.0)
    
    # Method correlation in 4th subplot
    ax = axes[1, 1]
    method_values = list(methods.values())
    correlation_matrix = np.corrcoef(method_values)
    
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Saliency Method Correlation')
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods.keys(), rotation=45)
    ax.set_yticklabels(methods.keys())
    
    # Add correlation values
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                   ha="center", va="center", 
                   color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Save the plot
    save_dir = Path("batch_analysis_results/advanced_demo")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "advanced_pytorch_saliency_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Advanced visualization saved to {save_dir}")
    
    return methods, accuracy

def demo_shap_analysis():
    """Demonstrate SHAP analysis capabilities."""
    print("\n🔍 SHAP Analysis Demo")
    print("=" * 50)
    
    from feature_saliency_analysis import SklearnFeatureSaliencyAnalyzer
    from batch_specific_analysis import BatchSpecificAnalyzer
    import shap
    
    # Setup
    batch_analyzer = BatchSpecificAnalyzer()
    saliency_analyzer = SklearnFeatureSaliencyAnalyzer(
        batch_analyzer.batch_configs, 
        batch_analyzer.base_dir
    )
    
    batch_name = 'soft_finger_batch_1'
    
    # Train models
    print("1. Training interpretable models...")
    model_results = saliency_analyzer.train_interpretable_models(batch_name)
    print(f"   ✅ Trained models with accuracies:")
    for model_name, results in model_results.items():
        print(f"      {model_name}: {results['test_accuracy']:.3f}")
    
    # SHAP analysis
    print("\n2. Computing SHAP values...")
    shap_results = saliency_analyzer.compute_shap_values(batch_name)
    
    # Get feature names and SHAP importance
    feature_names = saliency_analyzer.feature_names[batch_name]
    shap_importance = shap_results['feature_importance']
    
    # Create SHAP summary plot
    print("\n3. Creating SHAP visualization...")
    
    # Get data for SHAP plotting
    X, y, _ = saliency_analyzer.load_batch_features(batch_name)
    scaler = saliency_analyzer.scalers[batch_name] 
    X_scaled = scaler.transform(X)
    
    # Create SHAP plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # SHAP feature importance
    ax1 = axes[0]
    importance_values = [shap_importance[name] for name in feature_names]
    top_indices = np.argsort(importance_values)[-15:]
    top_names = [feature_names[i] for i in top_indices]
    top_values = [importance_values[i] for i in top_indices]
    
    bars = ax1.barh(range(len(top_values)), top_values, color='lightblue', alpha=0.7)
    ax1.set_yticks(range(len(top_values)))
    ax1.set_yticklabels(top_names, fontsize=8)
    ax1.set_xlabel('SHAP Importance')
    ax1.set_title('SHAP Feature Importance (Top 15)')
    
    # Highlight top 5
    for i in range(max(0, len(bars)-5), len(bars)):
        bars[i].set_color('orange')
    
    # SHAP vs Random Forest comparison
    ax2 = axes[1]
    rf_importance = model_results['random_forest']['feature_importance'].get('builtin', {})
    rf_values = [rf_importance.get(name, 0) for name in feature_names]
    shap_values = [shap_importance.get(name, 0) for name in feature_names]
    
    ax2.scatter(rf_values, shap_values, alpha=0.6, s=50)
    ax2.set_xlabel('Random Forest Importance')
    ax2.set_ylabel('SHAP Importance')
    ax2.set_title('SHAP vs Random Forest Importance')
    
    # Add correlation coefficient
    correlation = np.corrcoef(rf_values, shap_values)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Add trend line
    z = np.polyfit(rf_values, shap_values, 1)
    p = np.poly1d(z)
    ax2.plot(rf_values, p(rf_values), "r--", alpha=0.8)
    
    plt.tight_layout()
    save_dir = Path("batch_analysis_results/advanced_demo")
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / "advanced_shap_analysis_demo.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ SHAP visualization saved to {save_dir}")
    print(f"   ✅ SHAP-RF correlation: {correlation:.3f}")
    
    return shap_results

def demo_comparison_analysis():
    """Compare all saliency methods."""
    print("\n🔬 Comprehensive Method Comparison")
    print("=" * 50)
    
    # Run both analyses
    print("Running PyTorch saliency analysis...")
    pytorch_methods, cnn_accuracy = demo_pytorch_saliency()
    
    print("Running SHAP analysis...")
    shap_results = demo_shap_analysis()
    
    # Create comprehensive comparison
    print("\n📊 Creating comprehensive comparison...")
    
    # Load feature names
    from feature_saliency_analysis import SklearnFeatureSaliencyAnalyzer
    from batch_specific_analysis import BatchSpecificAnalyzer
    
    batch_analyzer = BatchSpecificAnalyzer()
    saliency_analyzer = SklearnFeatureSaliencyAnalyzer(
        batch_analyzer.batch_configs, 
        batch_analyzer.base_dir
    )
    
    # Get Random Forest results for comparison
    model_results = saliency_analyzer.train_interpretable_models('soft_finger_batch_1')
    rf_importance = model_results['random_forest']['feature_importance'].get('builtin', {})
    
    # Combine all methods
    feature_names = saliency_analyzer.feature_names['soft_finger_batch_1']
    all_methods = {
        'Random Forest': [rf_importance.get(name, 0) for name in feature_names],
        'CNN Gradient': pytorch_methods['Gradient'],
        'Integrated Gradients': pytorch_methods['Integrated Gradients'], 
        'LIME': pytorch_methods['LIME'],
        'SHAP': [shap_results['feature_importance'].get(name, 0) for name in feature_names]
    }
    
    # Create comprehensive correlation matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Correlation heatmap
    method_values = list(all_methods.values())
    correlation_matrix = np.corrcoef(method_values)
    
    im = ax1.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax1.set_title('Complete Saliency Method Correlation Matrix')
    ax1.set_xticks(range(len(all_methods)))
    ax1.set_yticks(range(len(all_methods)))
    ax1.set_xticklabels(all_methods.keys(), rotation=45)
    ax1.set_yticklabels(all_methods.keys())
    
    # Add correlation values
    for i in range(len(all_methods)):
        for j in range(len(all_methods)):
            ax1.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                   ha="center", va="center", 
                   color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                   fontsize=9)
    
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    
    # Top feature consensus
    ax2.set_title('Feature Ranking Consensus (Top 10)')
    
    # Count how many methods rank each feature in top 10
    top_k = 10
    feature_votes = np.zeros(len(feature_names))
    
    for method_name, importance in all_methods.items():
        top_indices = np.argsort(importance)[-top_k:]
        feature_votes[top_indices] += 1
    
    # Get most consensus features
    consensus_indices = np.argsort(feature_votes)[-15:]
    consensus_names = [feature_names[i] for i in consensus_indices]
    consensus_scores = feature_votes[consensus_indices]
    
    bars = ax2.barh(range(len(consensus_scores)), consensus_scores, color='green', alpha=0.7)
    ax2.set_yticks(range(len(consensus_scores)))
    ax2.set_yticklabels(consensus_names, fontsize=8)
    ax2.set_xlabel('Number of Methods Ranking in Top 10')
    ax2.set_xlim(0, len(all_methods))
    
    # Color bars by consensus level
    for i, bar in enumerate(bars):
        consensus = consensus_scores[i] / len(all_methods)
        bar.set_color(plt.cm.RdYlGn(consensus))
    
    plt.tight_layout()
    save_dir = Path("batch_analysis_results/advanced_demo")
    plt.savefig(save_dir / "comprehensive_saliency_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save summary results
    summary = {
        'analysis_methods': list(all_methods.keys()),
        'model_accuracies': {
            'cnn_features': float(cnn_accuracy),
            'random_forest': float(model_results['random_forest']['test_accuracy']),
            'logistic_regression': float(model_results['logistic_regression']['test_accuracy']),
            'svm_linear': float(model_results['svm_linear']['test_accuracy'])
        },
        'method_correlations': correlation_matrix.tolist(),
        'top_consensus_features': {
            'feature_names': consensus_names[-10:],  # Top 10 consensus
            'consensus_scores': consensus_scores[-10:].tolist()
        }
    }
    
    with open(save_dir / "advanced_saliency_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Comprehensive analysis complete!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"🎯 Model accuracies range from {min(summary['model_accuracies'].values()):.3f} to {max(summary['model_accuracies'].values()):.3f}")
    print(f"🔗 Strongest method correlation: {np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]):.3f}")
    
    return summary

if __name__ == "__main__":
    print("🚀 ADVANCED SALIENCY ANALYSIS - COMPLETE DEMO")
    print("=" * 70)
    print("This demo showcases all advanced saliency capabilities:")
    print("• PyTorch CNN-based gradient saliency")
    print("• Integrated gradients for robust explanations")
    print("• SHAP values for feature interactions")
    print("• LIME for local interpretable explanations")
    print("• Comprehensive method comparison and consensus")
    print("")
    
    # Run complete demo
    summary = demo_comparison_analysis()
    
    print("\n" + "="*70)
    print("🎉 ADVANCED SALIENCY ANALYSIS DEMO COMPLETE!")
    print("=" * 70)
    print("\nKey Achievements:")
    print("✅ Multiple saliency methods successfully applied")
    print("✅ CNN and traditional ML models compared")
    print("✅ Feature importance consensus identified")
    print("✅ Comprehensive visualizations generated")
    print("✅ All results saved for further analysis")
    print("\nYour advanced saliency analysis system is fully operational!")