#!/usr/bin/env python3
"""
Complete Acoustic Sensing Integration Example
Demonstrates the full optimized pipeline from data collection to real-time sensing

This script shows how to integrate all the components we've developed:
1. Optimized feature extraction
2. Configurable training pipeline  
3. Real-time sensing system
4. Performance validation
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
from pathlib import Path
import logging

# Import our optimized components
from optimized_feature_sets import OptimizedFeatureExtractor, FeatureSetConfig
from training_integration import ConfigurableTrainingPipeline
from real_time_optimized_sensor import OptimizedRealTimeSensor, SensorConfig
from geometric_reconstruction_example import GeometricReconstructionPipeline

class IntegratedAcousticSystem:
    """
    Complete integrated acoustic sensing system
    
    Provides end-to-end functionality from training to real-time operation
    with optimized performance and configurable feature sets.
    """
    
    def __init__(self, data_path: str, output_dir: str = "integrated_system_output"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_extractor = None
        self.training_pipeline = None
        self.real_time_sensor = None
        self.geometric_pipeline = None
        
        # System state
        self.trained_models = {}
        self.performance_metrics = {}
        self.current_mode = 'OPTIMAL'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_training_pipeline(self):
        """Initialize and configure the training pipeline"""
        self.logger.info("Setting up training pipeline...")
        
        self.training_pipeline = ConfigurableTrainingPipeline(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir)
        )
        
        self.logger.info("Training pipeline ready")
        
    def train_all_modes(self) -> dict:
        """
        Train models for all feature modes (MINIMAL, OPTIMAL, RESEARCH)
        
        Returns:
            dict: Training results for each mode
        """
        if not self.training_pipeline:
            self.setup_training_pipeline()
            
        self.logger.info("Starting training for all modes...")
        
        modes = ['MINIMAL', 'OPTIMAL', 'RESEARCH']
        training_results = {}
        
        for mode in modes:
            self.logger.info(f"Training {mode} mode...")
            
            # Train model for this mode
            results = self.training_pipeline.train_with_feature_set(mode)
            training_results[mode] = results
            
            # Save model
            model_path = self.output_dir / f"model_{mode.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(results['model'], f)
                
            # Store performance metrics
            self.performance_metrics[mode] = {
                'accuracy': results['test_accuracy'],
                'feature_count': len(results['feature_names']),
                'processing_time': results.get('avg_processing_time', 0)
            }
            
            self.logger.info(f"{mode} mode - Accuracy: {results['test_accuracy']:.3f}, Features: {len(results['feature_names'])}")
            
        # Generate comparison plots
        self._create_training_comparison_plots(training_results)
        
        self.trained_models = training_results
        return training_results
        
    def setup_real_time_system(self, mode: str = 'OPTIMAL'):
        """
        Setup real-time sensing system with specified feature mode
        
        Args:
            mode: Feature mode ('MINIMAL', 'OPTIMAL', 'RESEARCH')
        """
        model_path = self.output_dir / f"model_{mode.lower()}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Please train first.")
            
        self.logger.info(f"Setting up real-time system with {mode} mode...")
        
        # Configure sensor
        config = SensorConfig(
            feature_mode=mode,
            buffer_size=1024,
            processing_timeout=0.001 if mode == 'OPTIMAL' else 0.002
        )
        
        # Initialize real-time sensor
        self.real_time_sensor = OptimizedRealTimeSensor(str(model_path), config)
        self.current_mode = mode
        
        self.logger.info(f"Real-time system ready with {mode} mode")
        
    def run_performance_benchmark(self, test_audio_files: list = None) -> dict:
        """
        Run comprehensive performance benchmark across all modes
        
        Args:
            test_audio_files: Optional list of test audio files
            
        Returns:
            dict: Benchmark results
        """
        if not self.trained_models:
            self.logger.warning("No trained models found. Training first...")
            self.train_all_modes()
            
        self.logger.info("Running performance benchmark...")
        
        benchmark_results = {}
        
        # Test each mode
        for mode in ['MINIMAL', 'OPTIMAL', 'RESEARCH']:
            self.logger.info(f"Benchmarking {mode} mode...")
            
            # Setup real-time system for this mode
            self.setup_real_time_system(mode)
            
            # Run timing tests
            timing_results = self._run_timing_benchmark(num_samples=1000)
            
            # Combine with training metrics
            benchmark_results[mode] = {
                **self.performance_metrics[mode],
                **timing_results,
                'efficiency_score': self._calculate_efficiency_score(
                    timing_results['avg_processing_time_ms'],
                    self.performance_metrics[mode]['accuracy'],
                    self.performance_metrics[mode]['feature_count']
                )
            }
            
        # Create benchmark visualization
        self._create_benchmark_plots(benchmark_results)
        
        # Save results
        results_path = self.output_dir / "benchmark_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
            
        return benchmark_results
        
    def demonstrate_real_time_operation(self, duration_seconds: float = 30.0):
        """
        Demonstrate real-time operation with live performance monitoring
        
        Args:
            duration_seconds: How long to run the demonstration
        """
        if not self.real_time_sensor:
            self.setup_real_time_system()
            
        self.logger.info(f"Starting {duration_seconds}s real-time demonstration...")
        
        # Start real-time processing
        self.real_time_sensor.start_processing()
        
        start_time = time.time()
        results_log = []
        
        try:
            while time.time() - start_time < duration_seconds:
                # Simulate audio input (replace with actual microphone in real use)
                audio_chunk = np.random.randn(self.real_time_sensor.config.buffer_size) * 0.1
                
                # Add some simulated contact events
                if np.random.random() < 0.1:  # 10% chance of contact simulation
                    audio_chunk += np.random.randn(len(audio_chunk)) * 0.5  # Add contact-like signal
                    
                # Process audio
                success = self.real_time_sensor.add_audio_data(audio_chunk)
                
                if success:
                    # Check for results
                    result = self.real_time_sensor.get_latest_result()
                    if result and 'error' not in result:
                        results_log.append({
                            'timestamp': result['timestamp'],
                            'contact_detected': result['contact_detected'],
                            'confidence': result['confidence'],
                            'processing_time_ms': result['processing_time_ms']
                        })
                        
                        # Print periodic updates
                        if len(results_log) % 50 == 0:
                            stats = self.real_time_sensor.get_performance_stats()
                            self.logger.info(f"Processed {len(results_log)} samples, "
                                          f"Avg time: {stats.get('avg_processing_time_ms', 0):.2f}ms")
                            
                time.sleep(0.02)  # ~50 Hz simulation
                
        finally:
            # Stop processing
            self.real_time_sensor.stop_processing()
            
        # Analyze demonstration results
        if results_log:
            self._analyze_demo_results(results_log)
            
        self.logger.info("Real-time demonstration completed")
        
    def setup_geometric_reconstruction(self):
        """Setup geometric reconstruction pipeline"""
        if not self.real_time_sensor:
            self.setup_real_time_system()
            
        self.geometric_pipeline = GeometricReconstructionPipeline(
            feature_extractor=self.real_time_sensor.feature_extractor,
            model=self.real_time_sensor.model
        )
        
        self.logger.info("Geometric reconstruction pipeline ready")
        
    def run_complete_workflow_demo(self):
        """
        Run a complete end-to-end workflow demonstration
        """
        self.logger.info("Starting complete workflow demonstration...")
        
        # Step 1: Train all models
        self.logger.info("\n=== Step 1: Training Models ===")
        training_results = self.train_all_modes()
        
        # Step 2: Performance benchmark
        self.logger.info("\n=== Step 2: Performance Benchmark ===")
        benchmark_results = self.run_performance_benchmark()
        
        # Step 3: Real-time demonstration
        self.logger.info("\n=== Step 3: Real-time Demonstration ===")
        self.demonstrate_real_time_operation(duration_seconds=10.0)
        
        # Step 4: Results summary
        self.logger.info("\n=== Step 4: Results Summary ===")
        self._print_workflow_summary(benchmark_results)
        
        self.logger.info("Complete workflow demonstration finished")
        
    def _run_timing_benchmark(self, num_samples: int = 1000) -> dict:
        """Run timing benchmark for current mode"""
        processing_times = []
        
        for _ in range(num_samples):
            # Generate test audio
            audio_chunk = np.random.randn(self.real_time_sensor.config.buffer_size)
            
            # Time the processing
            start_time = time.perf_counter()
            features = self.real_time_sensor.feature_extractor.extract_features(
                audio_chunk, self.real_time_sensor.config.sample_rate
            )
            prediction = self.real_time_sensor.model.predict(features.reshape(1, -1))
            end_time = time.perf_counter()
            
            processing_times.append((end_time - start_time) * 1000)  # Convert to ms
            
        return {
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'std_processing_time_ms': np.std(processing_times),
            'samples_tested': num_samples
        }
        
    def _calculate_efficiency_score(self, processing_time_ms: float, accuracy: float, feature_count: int) -> float:
        """Calculate efficiency score combining speed, accuracy, and complexity"""
        # Normalize metrics (lower processing time and feature count are better)
        time_score = max(0, 1 - processing_time_ms / 10.0)  # Normalize to 10ms max
        accuracy_score = accuracy
        complexity_score = max(0, 1 - feature_count / 40.0)  # Normalize to 40 features max
        
        # Weighted combination
        efficiency_score = (0.4 * time_score + 0.4 * accuracy_score + 0.2 * complexity_score)
        return efficiency_score
        
    def _create_training_comparison_plots(self, training_results: dict):
        """Create plots comparing training results across modes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        modes = list(training_results.keys())
        accuracies = [training_results[mode]['test_accuracy'] for mode in modes]
        feature_counts = [len(training_results[mode]['feature_names']) for mode in modes]
        
        # Accuracy comparison
        axes[0, 0].bar(modes, accuracies)
        axes[0, 0].set_title('Test Accuracy by Mode')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Feature count comparison
        axes[0, 1].bar(modes, feature_counts)
        axes[0, 1].set_title('Feature Count by Mode')
        axes[0, 1].set_ylabel('Number of Features')
        
        # Confusion matrix for OPTIMAL mode
        if 'OPTIMAL' in training_results:
            cm = training_results['OPTIMAL']['confusion_matrix']
            im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[1, 0].set_title('OPTIMAL Mode - Confusion Matrix')
            plt.colorbar(im, ax=axes[1, 0])
            
        # Feature importance for OPTIMAL mode
        if 'OPTIMAL' in training_results:
            feature_names = training_results['OPTIMAL']['feature_names']
            importance = training_results['OPTIMAL'].get('feature_importance', np.ones(len(feature_names)))
            
            axes[1, 1].barh(range(len(feature_names)), importance)
            axes[1, 1].set_yticks(range(len(feature_names)))
            axes[1, 1].set_yticklabels(feature_names)
            axes[1, 1].set_title('OPTIMAL Mode - Feature Importance')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_benchmark_plots(self, benchmark_results: dict):
        """Create benchmark visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        modes = list(benchmark_results.keys())
        
        # Processing time comparison
        times = [benchmark_results[mode]['avg_processing_time_ms'] for mode in modes]
        axes[0, 0].bar(modes, times)
        axes[0, 0].set_title('Average Processing Time')
        axes[0, 0].set_ylabel('Time (ms)')
        
        # Accuracy vs Speed
        accuracies = [benchmark_results[mode]['accuracy'] for mode in modes]
        axes[0, 1].scatter(times, accuracies, s=100)
        for i, mode in enumerate(modes):
            axes[0, 1].annotate(mode, (times[i], accuracies[i]), xytext=(5, 5), 
                              textcoords='offset points')
        axes[0, 1].set_xlabel('Processing Time (ms)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Processing Speed')
        
        # Efficiency scores
        efficiency_scores = [benchmark_results[mode]['efficiency_score'] for mode in modes]
        axes[1, 0].bar(modes, efficiency_scores)
        axes[1, 0].set_title('Efficiency Score')
        axes[1, 0].set_ylabel('Score (higher is better)')
        axes[1, 0].set_ylim(0, 1)
        
        # Feature count vs accuracy
        feature_counts = [benchmark_results[mode]['feature_count'] for mode in modes]
        axes[1, 1].scatter(feature_counts, accuracies, s=100)
        for i, mode in enumerate(modes):
            axes[1, 1].annotate(mode, (feature_counts[i], accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Feature Complexity vs Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _analyze_demo_results(self, results_log: list):
        """Analyze and visualize demonstration results"""
        df = pd.DataFrame(results_log)
        
        if len(df) > 0:
            # Create analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Processing time over time
            axes[0, 0].plot(df['processing_time_ms'])
            axes[0, 0].set_title('Processing Time Over Time')
            axes[0, 0].set_xlabel('Sample')
            axes[0, 0].set_ylabel('Time (ms)')
            
            # Confidence distribution
            axes[0, 1].hist(df['confidence'], bins=20, alpha=0.7)
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Frequency')
            
            # Contact detection rate
            contact_rate = df['contact_detected'].rolling(window=50).mean()
            axes[1, 0].plot(contact_rate)
            axes[1, 0].set_title('Contact Detection Rate (50-sample window)')
            axes[1, 0].set_xlabel('Sample')
            axes[1, 0].set_ylabel('Contact Rate')
            
            # Processing time histogram
            axes[1, 1].hist(df['processing_time_ms'], bins=20, alpha=0.7)
            axes[1, 1].set_title('Processing Time Distribution')
            axes[1, 1].set_xlabel('Time (ms)')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'demo_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save statistics
            stats = {
                'total_samples': len(df),
                'avg_processing_time_ms': df['processing_time_ms'].mean(),
                'max_processing_time_ms': df['processing_time_ms'].max(),
                'contact_detection_rate': df['contact_detected'].mean(),
                'avg_confidence': df['confidence'].mean()
            }
            
            stats_path = self.output_dir / 'demo_statistics.json'
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
                
    def _print_workflow_summary(self, benchmark_results: dict):
        """Print a comprehensive workflow summary"""
        print("\n" + "="*60)
        print("ACOUSTIC SENSING SYSTEM - WORKFLOW SUMMARY")
        print("="*60)
        
        print("\nMODE COMPARISON:")
        print("-" * 40)
        for mode in ['MINIMAL', 'OPTIMAL', 'RESEARCH']:
            if mode in benchmark_results:
                result = benchmark_results[mode]
                print(f"{mode:>8}: {result['accuracy']:.3f} accuracy, "
                      f"{result['feature_count']:2d} features, "
                      f"{result['avg_processing_time_ms']:.2f}ms, "
                      f"efficiency: {result['efficiency_score']:.3f}")
                      
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        
        # Find best mode for each criteria
        best_accuracy = max(benchmark_results.keys(), key=lambda x: benchmark_results[x]['accuracy'])
        best_speed = min(benchmark_results.keys(), key=lambda x: benchmark_results[x]['avg_processing_time_ms'])
        best_efficiency = max(benchmark_results.keys(), key=lambda x: benchmark_results[x]['efficiency_score'])
        
        print(f"• Best accuracy: {best_accuracy} mode ({benchmark_results[best_accuracy]['accuracy']:.3f})")
        print(f"• Fastest processing: {best_speed} mode ({benchmark_results[best_speed]['avg_processing_time_ms']:.2f}ms)")
        print(f"• Best efficiency: {best_efficiency} mode (score: {benchmark_results[best_efficiency]['efficiency_score']:.3f})")
        
        print(f"\nGeneral recommendation: Use {best_efficiency} mode for optimal balance of speed and accuracy")
        print(f"For real-time applications: Use {best_speed} mode if speed is critical")
        print(f"For research applications: Use {best_accuracy} mode for maximum accuracy")
        
        print("\nOUTPUT FILES:")
        print("-" * 40)
        output_files = list(self.output_dir.glob('*'))
        for file in sorted(output_files):
            print(f"• {file.name}")
            
        print(f"\nAll results saved to: {self.output_dir}")


# Example usage and main execution
if __name__ == "__main__":
    # Initialize the integrated system
    # Replace with your actual data path
    data_path = "../data/soft_finger_batch_1"
    
    system = IntegratedAcousticSystem(data_path)
    
    print("Integrated Acoustic Sensing System")
    print("=================================")
    
    try:
        # Run the complete workflow demonstration
        system.run_complete_workflow_demo()
        
        print(f"\nWorkflow completed successfully!")
        print(f"Check the output directory: {system.output_dir}")
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        print("Please ensure you have the required data and dependencies installed")