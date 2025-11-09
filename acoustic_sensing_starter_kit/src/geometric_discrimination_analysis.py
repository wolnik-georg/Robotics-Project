#!/usr/bin/env python3
"""
Comprehensive Acoustic Geometric Discrimination Analysis
======================================================

This script performs a complete analysis to prove geometric discrimination 
capability in acoustic sensing data. It includes:

1. Data loading from recorded WAV files
2. Comprehensive feature extraction
3. t-SNE and PCA analysis
4. Statistical discrimination analysis
5. Publication-ready visualizations
6. Detailed reporting

Usage:
    python geometric_discrimination_analysis.py [--config config.json] [--output output_dir]

Based on your original t-SNE analysis but significantly enhanced for
comprehensive geometric discrimination proof.

Author: Enhanced ML Pipeline for Acoustic Sensing
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geometric_data_loader import GeometricDataLoader, print_dataset_summary
from feature_extraction import GeometricFeatureExtractor
from dimensionality_analysis import GeometricDimensionalityAnalyzer
from discrimination_analysis import GeometricDiscriminationAnalyzer
import preprocessing


class GeometricDiscriminationPipeline:
    """
    Complete pipeline for acoustic geometric discrimination analysis.
    
    Combines data loading, feature extraction, dimensionality reduction,
    and statistical analysis to prove geometric discrimination capability.
    """
    
    def __init__(self, config: Optional[Dict] = None, output_dir: str = "results"):
        """
        Initialize the analysis pipeline.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save results
        """
        self.config = config or self._get_default_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_loader = GeometricDataLoader(
            base_dir=self.config['data']['base_dir'],
            sr=self.config['data']['sample_rate']
        )
        
        self.feature_extractor = GeometricFeatureExtractor(
            sr=self.config['data']['sample_rate'],
            n_fft=self.config['features']['n_fft']
        )
        
        self.dim_analyzer = GeometricDimensionalityAnalyzer(
            random_state=self.config['analysis']['random_state']
        )
        
        self.disc_analyzer = GeometricDiscriminationAnalyzer(
            random_state=self.config['analysis']['random_state']
        )
        
        # Results storage
        self.results = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'data': {
                'base_dir': '../data',
                'sample_rate': 48000,
                'batches': None,  # None = all available batches
                'contact_positions': ['finger tip', 'finger middle', 'finger bottom', 'finger blank'],
                'max_samples_per_class': None  # None = all samples
            },
            'features': {
                'method': 'comprehensive',
                'n_fft': 4096,
                'standardize_length': True,
                'length_method': 'pad_truncate'
            },
            'analysis': {
                'random_state': 42,
                'tsne_perplexity': [5, 10, 20, 30, 50],
                'pca_explained_variance_threshold': 0.95,
                'test_multiple_classifiers': True,
                'cv_folds': 5,
                'bootstrap_iterations': 100
            },
            'visualization': {
                'figsize_large': [15, 10],
                'figsize_medium': [12, 8],
                'figsize_comparison': [18, 6],
                'dpi': 300,
                'style': 'default'
            }
        }
    
    def load_and_prepare_data(self) -> None:
        """Load and prepare data for analysis."""
        print("=" * 60)
        print("LOADING AND PREPARING DATA")
        print("=" * 60)
        
        # Load data
        batch_names = self.config['data']['batches']
        if batch_names is None:
            batch_names = self.data_loader.get_available_batches()
            print(f"Auto-detected batches: {batch_names}")
        
        audio_data, labels, metadata = self.data_loader.load_multiple_batches(
            batch_names=batch_names,
            contact_positions=self.config['data']['contact_positions'],
            max_samples_per_class=self.config['data']['max_samples_per_class'],
            verbose=True
        )
        
        # Print dataset summary
        print_dataset_summary(audio_data, labels, metadata)
        
        # Standardize audio length if requested
        if self.config['features']['standardize_length']:
            print(f"\nStandardizing audio lengths using method: {self.config['features']['length_method']}")
            standardized_audio = self.data_loader.standardize_audio_length(
                audio_data, 
                method=self.config['features']['length_method']
            )
            audio_data = standardized_audio
        
        # Store data
        self.results['data'] = {
            'audio_data': audio_data,
            'labels': labels,
            'metadata': metadata,
            'simplified_labels': self.data_loader.simplify_labels(labels)
        }
        
        print(f"\nData prepared: {len(audio_data)} samples ready for feature extraction")\n    \n    def extract_features(self) -> None:\n        \"\"\"Extract features from all audio samples.\"\"\"\n        print(\"\\n\" + \"=\" * 60)\n        print(\"FEATURE EXTRACTION\")\n        print(\"=\" * 60)\n        \n        audio_data = self.results['data']['audio_data']\n        method = self.config['features']['method']\n        \n        print(f\"Extracting features using method: {method}\")\n        print(f\"Processing {len(audio_data)} audio samples...\")\n        \n        # Extract features\n        features_list = []\n        failed_extractions = 0\n        \n        for i, audio in enumerate(audio_data):\n            try:\n                features = self.feature_extractor.extract_features(audio, method=method)\n                features_list.append(features)\n                \n                if (i + 1) % 100 == 0:\n                    print(f\"  Processed {i + 1}/{len(audio_data)} samples\")\n                    \n            except Exception as e:\n                print(f\"  Warning: Failed to extract features from sample {i}: {e}\")\n                # Use zero features as fallback\n                if features_list:\n                    features_list.append(np.zeros_like(features_list[0]))\n                else:\n                    # If first sample failed, we need to determine feature size\n                    dummy_features = self.feature_extractor.extract_features(\n                        np.zeros(1000), method=method\n                    )\n                    features_list.append(np.zeros_like(dummy_features))\n                failed_extractions += 1\n        \n        if failed_extractions > 0:\n            print(f\"Warning: {failed_extractions} feature extractions failed\")\n        \n        # Convert to numpy array\n        features_matrix = np.array(features_list)\n        feature_names = self.feature_extractor.get_feature_names(method=method)\n        \n        print(f\"Feature extraction complete: {features_matrix.shape}\")\n        print(f\"Features: {len(feature_names)} dimensions\")\n        \n        # Store features\n        self.results['features'] = {\n            'features_matrix': features_matrix,\n            'feature_names': feature_names,\n            'extraction_method': method,\n            'failed_extractions': failed_extractions\n        }\n    \n    def perform_dimensionality_reduction(self) -> None:\n        \"\"\"Perform PCA and t-SNE analysis.\"\"\"\n        print(\"\\n\" + \"=\" * 60)\n        print(\"DIMENSIONALITY REDUCTION ANALYSIS\")\n        print(\"=\" * 60)\n        \n        features_matrix = self.results['features']['features_matrix']\n        \n        # PCA Analysis\n        print(\"Performing PCA analysis...\")\n        pca_embedding, pca_results = self.dim_analyzer.fit_transform_pca(\n            features_matrix,\n            explained_variance_threshold=self.config['analysis']['pca_explained_variance_threshold']\n        )\n        \n        print(f\"PCA: {pca_results['n_components']} components explain {pca_results['total_explained_variance']:.3f} variance\")\n        \n        # t-SNE Analysis with multiple perplexity values\n        print(\"Performing t-SNE analysis...\")\n        perplexity_values = self.config['analysis']['tsne_perplexity']\n        tsne_embedding, tsne_results = self.dim_analyzer.fit_transform_tsne(\n            features_matrix,\n            perplexity=perplexity_values\n        )\n        \n        print(f\"t-SNE: Optimal perplexity = {tsne_results['optimal_perplexity']}\")\n        \n        # UMAP if available\n        umap_embedding = None\n        umap_results = None\n        try:\n            print(\"Performing UMAP analysis...\")\n            umap_embedding, umap_results = self.dim_analyzer.fit_transform_umap(features_matrix)\n            print(\"UMAP analysis completed\")\n        except ImportError:\n            print(\"UMAP not available (install with 'pip install umap-learn')\")\n        except Exception as e:\n            print(f\"UMAP analysis failed: {e}\")\n        \n        # Store results\n        self.results['dimensionality_reduction'] = {\n            'pca': {\n                'embedding': pca_embedding,\n                'results': pca_results\n            },\n            'tsne': {\n                'embedding': tsne_embedding,\n                'results': tsne_results\n            }\n        }\n        \n        if umap_embedding is not None:\n            self.results['dimensionality_reduction']['umap'] = {\n                'embedding': umap_embedding,\n                'results': umap_results\n            }\n    \n    def analyze_separability(self) -> None:\n        \"\"\"Perform comprehensive separability analysis.\"\"\"\n        print(\"\\n\" + \"=\" * 60)\n        print(\"SEPARABILITY AND DISCRIMINATION ANALYSIS\")\n        print(\"=\" * 60)\n        \n        features_matrix = self.results['features']['features_matrix']\n        labels = self.results['data']['simplified_labels']  # Use simplified labels\n        feature_names = self.results['features']['feature_names']\n        \n        # 1. Separability analysis\n        print(\"Analyzing class separability...\")\n        sep_results = self.disc_analyzer.analyze_class_separability(\n            features_matrix, labels, feature_names\n        )\n        \n        # 2. Classification performance\n        print(\"Evaluating classification performance...\")\n        cls_results = self.disc_analyzer.evaluate_classification_performance(\n            features_matrix, labels,\n            test_multiple_classifiers=self.config['analysis']['test_multiple_classifiers'],\n            cv_folds=self.config['analysis']['cv_folds']\n        )\n        \n        # 3. Feature stability\n        print(\"Analyzing feature stability...\")\n        stab_results = self.disc_analyzer.analyze_feature_stability(\n            features_matrix, labels,\n            n_bootstrap=self.config['analysis']['bootstrap_iterations'],\n            feature_names=feature_names\n        )\n        \n        # 4. Embedding separability\n        embedding_separability = {}\n        for method_name, method_data in self.results['dimensionality_reduction'].items():\n            embedding = method_data['embedding']\n            if embedding is not None and embedding.shape[1] <= 10:  # Reasonable dimensionality\n                sep_analysis = self.dim_analyzer.analyze_separability(\n                    embedding, labels, method_name\n                )\n                embedding_separability[method_name] = sep_analysis\n        \n        # Store results\n        self.results['discrimination_analysis'] = {\n            'separability': sep_results,\n            'classification': cls_results,\n            'feature_stability': stab_results,\n            'embedding_separability': embedding_separability\n        }\n        \n        # Generate comprehensive report\n        print(\"Generating discrimination report...\")\n        report = self.disc_analyzer.generate_discrimination_report()\n        self.results['discrimination_analysis']['report'] = report\n        \n        print(\"\\nDISCRIMINATION ANALYSIS SUMMARY:\")\n        print(report)\n    \n    def create_visualizations(self) -> None:\n        \"\"\"Create comprehensive visualizations.\"\"\"\n        print(\"\\n\" + \"=\" * 60)\n        print(\"CREATING VISUALIZATIONS\")\n        print(\"=\" * 60)\n        \n        labels = self.results['data']['simplified_labels']\n        \n        # Set matplotlib style\n        plt.style.use(self.config['visualization']['style'])\n        \n        # 1. t-SNE visualization (main result - like your original example)\n        print(\"Creating t-SNE visualization...\")\n        tsne_embedding = self.results['dimensionality_reduction']['tsne']['embedding']\n        \n        fig_tsne = self.dim_analyzer.plot_2d_embedding(\n            tsne_embedding, labels,\n            title=\"t-SNE: Acoustic Geometric Discrimination\",\n            figsize=tuple(self.config['visualization']['figsize_medium']),\n            save_path=self.output_dir / \"tsne_geometric_discrimination.png\"\n        )\n        plt.close(fig_tsne)\n        \n        # 2. PCA analysis plots\n        print(\"Creating PCA analysis plots...\")\n        pca_results = self.results['dimensionality_reduction']['pca']['results']\n        feature_names = self.results['features']['feature_names']\n        \n        fig_pca = self.dim_analyzer.plot_pca_analysis(\n            pca_results, feature_names,\n            figsize=tuple(self.config['visualization']['figsize_large']),\n            save_path=self.output_dir / \"pca_analysis.png\"\n        )\n        plt.close(fig_pca)\n        \n        # 3. PCA embedding visualization\n        print(\"Creating PCA embedding visualization...\")\n        pca_embedding = self.results['dimensionality_reduction']['pca']['embedding']\n        \n        fig_pca_2d = self.dim_analyzer.plot_2d_embedding(\n            pca_embedding[:, :2], labels,  # First 2 components\n            title=\"PCA: Principal Components Analysis\",\n            figsize=tuple(self.config['visualization']['figsize_medium']),\n            save_path=self.output_dir / \"pca_embedding.png\"\n        )\n        plt.close(fig_pca_2d)\n        \n        # 4. Method comparison plot\n        print(\"Creating method comparison plot...\")\n        embeddings_dict = {\n            't-SNE': tsne_embedding,\n            'PCA': pca_embedding[:, :2]\n        }\n        \n        if 'umap' in self.results['dimensionality_reduction']:\n            umap_embedding = self.results['dimensionality_reduction']['umap']['embedding']\n            embeddings_dict['UMAP'] = umap_embedding\n        \n        fig_comparison = self.dim_analyzer.create_comparison_plot(\n            embeddings_dict, labels,\n            figsize=tuple(self.config['visualization']['figsize_comparison']),\n            save_path=self.output_dir / \"method_comparison.png\"\n        )\n        plt.close(fig_comparison)\n        \n        # 5. t-SNE perplexity comparison (if multiple tested)\n        tsne_results = self.results['dimensionality_reduction']['tsne']['results']\n        if len(tsne_results['perplexity_results']) > 1:\n            print(\"Creating t-SNE perplexity comparison...\")\n            \n            from dimensionality_analysis import compare_perplexity_values\n            features_matrix = self.results['features']['features_matrix']\n            perplexity_values = list(tsne_results['perplexity_results'].keys())\n            \n            fig_perp = compare_perplexity_values(\n                features_matrix, labels, perplexity_values,\n                figsize=tuple([20, 8])\n            )\n            fig_perp.savefig(self.output_dir / \"tsne_perplexity_comparison.png\", \n                           dpi=self.config['visualization']['dpi'], bbox_inches='tight')\n            plt.close(fig_perp)\n        \n        print(f\"Visualizations saved to: {self.output_dir}\")\n    \n    def save_results(self) -> None:\n        \"\"\"Save analysis results to files.\"\"\"\n        print(\"\\n\" + \"=\" * 60)\n        print(\"SAVING RESULTS\")\n        print(\"=\" * 60)\n        \n        # Save discrimination report\n        report_path = self.output_dir / \"discrimination_analysis_report.txt\"\n        with open(report_path, 'w') as f:\n            f.write(self.results['discrimination_analysis']['report'])\n        print(f\"Discrimination report saved: {report_path}\")\n        \n        # Save feature matrix and labels as CSV\n        features_df = pd.DataFrame(\n            self.results['features']['features_matrix'],\n            columns=self.results['features']['feature_names']\n        )\n        features_df['label'] = self.results['data']['simplified_labels']\n        features_df['original_label'] = self.results['data']['labels']\n        \n        features_path = self.output_dir / \"features_with_labels.csv\"\n        features_df.to_csv(features_path, index=False)\n        print(f\"Features and labels saved: {features_path}\")\n        \n        # Save t-SNE and PCA embeddings\n        tsne_embedding = self.results['dimensionality_reduction']['tsne']['embedding']\n        pca_embedding = self.results['dimensionality_reduction']['pca']['embedding']\n        \n        embeddings_df = pd.DataFrame({\n            'tsne_1': tsne_embedding[:, 0],\n            'tsne_2': tsne_embedding[:, 1],\n            'pca_1': pca_embedding[:, 0],\n            'pca_2': pca_embedding[:, 1],\n            'label': self.results['data']['simplified_labels'],\n            'original_label': self.results['data']['labels']\n        })\n        \n        embeddings_path = self.output_dir / \"embeddings.csv\"\n        embeddings_df.to_csv(embeddings_path, index=False)\n        print(f\"Embeddings saved: {embeddings_path}\")\n        \n        # Save configuration\n        config_path = self.output_dir / \"analysis_config.json\"\n        with open(config_path, 'w') as f:\n            json.dump(self.config, f, indent=2)\n        print(f\"Configuration saved: {config_path}\")\n        \n        # Save metadata and summary statistics\n        metadata = self.results['data']['metadata']\n        summary_stats = {\n            'dataset_summary': {\n                'total_samples': len(self.results['data']['audio_data']),\n                'n_features': self.results['features']['features_matrix'].shape[1],\n                'n_classes': len(np.unique(self.results['data']['simplified_labels'])),\n                'classes': np.unique(self.results['data']['simplified_labels']).tolist(),\n                'batches_used': list(metadata['batches'].keys()) if 'batches' in metadata else [],\n            },\n            'pca_summary': {\n                'n_components': self.results['dimensionality_reduction']['pca']['results']['n_components'],\n                'explained_variance': self.results['dimensionality_reduction']['pca']['results']['total_explained_variance']\n            },\n            'tsne_summary': {\n                'optimal_perplexity': self.results['dimensionality_reduction']['tsne']['results']['optimal_perplexity']\n            }\n        }\n        \n        summary_path = self.output_dir / \"analysis_summary.json\"\n        with open(summary_path, 'w') as f:\n            json.dump(summary_stats, f, indent=2)\n        print(f\"Analysis summary saved: {summary_path}\")\n    \n    def run_complete_analysis(self) -> None:\n        \"\"\"Run the complete analysis pipeline.\"\"\"\n        print(\"ACOUSTIC GEOMETRIC DISCRIMINATION ANALYSIS\")\n        print(f\"Output directory: {self.output_dir}\")\n        print(f\"Configuration: {len(self.config)} sections\")\n        \n        try:\n            # Run analysis steps\n            self.load_and_prepare_data()\n            self.extract_features()\n            self.perform_dimensionality_reduction()\n            self.analyze_separability()\n            self.create_visualizations()\n            self.save_results()\n            \n            print(\"\\n\" + \"=\" * 60)\n            print(\"ANALYSIS COMPLETE!\")\n            print(\"=\" * 60)\n            print(f\"Results saved to: {self.output_dir}\")\n            print(\"\\nKey files created:\")\n            print(f\"  • tsne_geometric_discrimination.png - Main t-SNE result\")\n            print(f\"  • discrimination_analysis_report.txt - Comprehensive report\")\n            print(f\"  • features_with_labels.csv - Extracted features\")\n            print(f\"  • embeddings.csv - t-SNE and PCA coordinates\")\n            print(\"\\nCheck the discrimination report for statistical evidence of geometric discrimination capability.\")\n            \n        except Exception as e:\n            print(f\"\\nERROR: Analysis failed: {e}\")\n            import traceback\n            traceback.print_exc()\n            raise\n\n\ndef main():\n    \"\"\"Main function with command line interface.\"\"\"\n    parser = argparse.ArgumentParser(\n        description=\"Comprehensive Acoustic Geometric Discrimination Analysis\"\n    )\n    parser.add_argument(\n        '--config', \n        type=str, \n        help='Path to configuration JSON file'\n    )\n    parser.add_argument(\n        '--output', \n        type=str, \n        default='results',\n        help='Output directory for results (default: results)'\n    )\n    parser.add_argument(\n        '--batches',\n        nargs='+',\n        help='Specific batch names to analyze (e.g., soft_finger_batch_1 soft_finger_batch_2)'\n    )\n    parser.add_argument(\n        '--max-samples',\n        type=int,\n        help='Maximum samples per class (for testing)'\n    )\n    \n    args = parser.parse_args()\n    \n    # Load configuration\n    config = None\n    if args.config and os.path.exists(args.config):\n        with open(args.config, 'r') as f:\n            config = json.load(f)\n    \n    # Override config with command line arguments\n    if config is None:\n        config = {}\n    \n    if args.batches:\n        if 'data' not in config:\n            config['data'] = {}\n        config['data']['batches'] = args.batches\n    \n    if args.max_samples:\n        if 'data' not in config:\n            config['data'] = {}\n        config['data']['max_samples_per_class'] = args.max_samples\n    \n    # Run analysis\n    pipeline = GeometricDiscriminationPipeline(config=config, output_dir=args.output)\n    pipeline.run_complete_analysis()\n\n\nif __name__ == \"__main__\":\n    main()"