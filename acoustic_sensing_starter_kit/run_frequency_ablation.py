#!/usr/bin/env python3
"""
Quick execution script for frequency band ablation analysis.

This script runs the comprehensive frequency band analysis to validate
the 200-2000Hz discriminative range claim.

Usage:
    python run_frequency_ablation.py [batch_names...]

Examples:
    python run_frequency_ablation.py                           # All batches
    python run_frequency_ablation.py soft_finger_batch_1       # Single batch
    python run_frequency_ablation.py soft_finger_batch_1 soft_finger_batch_2  # Multiple batches
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from acoustic_sensing.features.frequency_band_ablation import (
    FrequencyBandAblationAnalyzer,
)


def main():
    """Execute frequency band ablation analysis."""

    # Parse command line arguments
    if len(sys.argv) > 1:
        batch_names = sys.argv[1:]
        print(f"Running analysis on specified batches: {batch_names}")
    else:
        batch_names = None
        print("Running analysis on all available batches")

    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure you're in the acoustic_sensing_starter_kit directory")
        return 1

    # Initialize and run analyzer
    try:
        analyzer = FrequencyBandAblationAnalyzer(
            base_data_dir="data", output_dir="frequency_band_analysis_results"
        )

        print(f"\n🔬 Starting frequency band ablation analysis...")
        print(f"Output directory: {analyzer.output_dir}")

        results = analyzer.analyze_all_batches(
            batch_names=batch_names, save_results=True
        )

        if results:
            print(f"\n✅ Analysis completed successfully!")
            print(f"Check {analyzer.output_dir} for detailed results")
        else:
            print(f"\n❌ No results generated")
            return 1

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
