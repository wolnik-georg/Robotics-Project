#!/usr/bin/env python3
"""
Main entry point for running modular acoustic sensing experiments.
This script replaces the monolithic batch analysis with a clean, modular approach.

Usage:
    python run_modular_experiments.py [config_file] [output_dir]

Example:
    python run_modular_experiments.py configs/experiment_config.yml modular_results
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from acoustic_sensing.experiments.orchestrator import ExperimentOrchestrator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run modular acoustic sensing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_modular_experiments.py
  python run_modular_experiments.py configs/my_config.yml
  python run_modular_experiments.py configs/my_config.yml custom_output
  python run_modular_experiments.py --validate-only
        """,
    )

    parser.add_argument(
        "config_file",
        nargs="?",
        default="configs/experiment_config.yml",
        help="Path to experiment configuration file (default: configs/experiment_config.yml)",
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default="modular_analysis_results",
        help="Output directory for results (default: modular_analysis_results)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running experiments",
    )

    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from matplotlib and other libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def validate_paths(config_file: str, output_dir: str):
    """Validate input paths and create output directory."""
    # Check config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        print(f"💡 Make sure you're running from the project root directory")
        print(f"💡 Available config: configs/experiment_config.yml")
        sys.exit(1)

    # Create output directory
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"📂 Output directory: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"❌ Could not create output directory {output_dir}: {e}")
        sys.exit(1)

    return os.path.abspath(config_file), os.path.abspath(output_dir)


def print_experiment_list():
    """Print available experiments."""
    print("\n🧪 Available Experiments:")
    print("=" * 50)

    experiments = [
        ("data_processing", "Load and preprocess acoustic data"),
        ("dimensionality_reduction", "PCA and t-SNE analysis"),
        ("discrimination_analysis", "Multi-classifier material discrimination"),
        ("saliency_analysis", "Neural network feature importance analysis"),
        ("feature_ablation", "Systematic feature importance testing"),
        ("impulse_response", "Deconvolution and transfer function analysis"),
        ("frequency_band_ablation", "Frequency-specific contribution analysis"),
    ]

    for name, description in experiments:
        print(f"  • {name:<25} - {description}")

    print("\n💡 Configure experiments in configs/experiment_config.yml")
    print("   Set 'enabled: true' for experiments you want to run")


def main():
    """Main execution function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # List experiments if requested
    if args.list_experiments:
        print_experiment_list()
        return

    # Validate paths
    config_file, output_dir = validate_paths(args.config_file, args.output_dir)

    try:
        # Load config to check for custom output directory
        import yaml

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Use output directory from config if specified, otherwise use command line arg
        if "output" in config and "base_dir" in config["output"]:
            output_dir = os.path.abspath(config["output"]["base_dir"])
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            print(f"📂 Output directory (from config): {output_dir}")
        else:
            print(f"📂 Output directory: {output_dir}")

        # Initialize orchestrator
        print(f"🚀 Initializing Experiment Orchestrator...")
        print(f"📋 Config: {config_file}")

        orchestrator = ExperimentOrchestrator(config_file, output_dir)

        # Validate configuration
        print(f"✅ Validating configuration...")
        validation_result = orchestrator.validate_config()

        if not validation_result["valid"]:
            print("❌ Configuration validation failed:")
            for error in validation_result["errors"]:
                print(f"   🔸 ERROR: {error}")
            for warning in validation_result["warnings"]:
                print(f"   🔸 WARNING: {warning}")

            if validation_result["errors"]:
                sys.exit(1)

        if validation_result["warnings"]:
            print("⚠️  Configuration warnings:")
            for warning in validation_result["warnings"]:
                print(f"   🔸 {warning}")

        # If validate-only, exit here
        if args.validate_only:
            print("✅ Configuration is valid!")
            return

        # List enabled experiments
        enabled_experiments = orchestrator._get_enabled_experiments()
        print(f"\n🎯 Enabled experiments: {', '.join(enabled_experiments)}")

        if not enabled_experiments:
            print("❌ No experiments are enabled in the configuration!")
            print("💡 Edit your config file to enable experiments:")
            print("   experiments:")
            print("     discrimination_analysis:")
            print("       enabled: true")
            return

        # Run experiments
        print(f"\n🏃 Running experiments...")
        print("=" * 50)

        results = orchestrator.run_experiments()

        # Print summary
        print("\n" + "=" * 50)
        print("📊 EXECUTION SUMMARY")
        print("=" * 50)

        if "_execution_summary" in results:
            summary = results["_execution_summary"]
            print(f"✅ Successful: {summary.get('successful_experiments', 0)}")
            print(f"❌ Failed: {summary.get('failed_experiments', 0)}")
            print(f"⏭️  Skipped: {summary.get('skipped_experiments', 0)}")

            # Print key findings
            key_findings = summary.get("key_findings", [])
            if key_findings:
                print(f"\n🔍 Key Findings:")
                for finding in key_findings:
                    print(f"   • {finding}")

        # Print output locations
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"   📄 Summary: {os.path.join(output_dir, 'execution_summary.json')}")
        print(f"   📊 Individual experiment results in subdirectories")

        print(f"\n🎉 Experiment execution completed!")

    except KeyboardInterrupt:
        print(f"\n⏹️  Execution interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Execution failed: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
