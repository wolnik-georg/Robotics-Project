#!/usr/bin/env python3
"""
Cleanup script for acoustic sensing project restructuring

This script helps identify and optionally remove duplicate files after
the restructuring to the new acoustic_sensing package structure.

Run with --dry-run to see what would be removed without actually removing anything.
"""

import os
import sys
import argparse
from pathlib import Path


def get_migrated_files():
    """List of files that have been migrated to the new structure"""
    return {
        # Core files
        "feature_extraction.py": "acoustic_sensing/core/",
        "preprocessing.py": "acoustic_sensing/core/",
        "data_management.py": "acoustic_sensing/core/",
        # Feature files
        "optimized_feature_sets.py": "acoustic_sensing/features/optimized_sets.py",
        "feature_ablation_analysis.py": "acoustic_sensing/features/ablation_analysis.py",
        "feature_saliency_analysis.py": "acoustic_sensing/features/",
        "saliency_analysis.py": "acoustic_sensing/features/",
        # Model files
        "training_integration.py": "acoustic_sensing/models/training.py",
        "geometric_reconstruction_example.py": "acoustic_sensing/models/geometric_reconstruction.py",
        "geometric_data_loader.py": "acoustic_sensing/models/",
        # Sensor files
        "real_time_optimized_sensor.py": "acoustic_sensing/sensors/real_time_sensor.py",
        # Analysis files
        "batch_specific_analysis.py": "acoustic_sensing/analysis/batch_analysis.py",
        "discrimination_analysis.py": "acoustic_sensing/analysis/",
        "dimensionality_analysis.py": "acoustic_sensing/analysis/",
        # Visualization files
        "create_publication_plots.py": "acoustic_sensing/visualization/publication_plots.py",
        # Demo files
        "integrated_system_demo.py": "acoustic_sensing/demo/integrated_demo.py",
        # Legacy files
        "A_record.py": "acoustic_sensing/legacy/",
        "B_train.py": "acoustic_sensing/legacy/",
        "C_sense.py": "acoustic_sensing/legacy/",
        # Documentation files (moved to docs)
        "ADVANCED_SALIENCY_VERIFICATION.md": "acoustic_sensing/docs/",
        "FEATURE_SELECTION_GUIDE.md": "acoustic_sensing/docs/",
        "FINAL_PROJECT_SUMMARY.md": "acoustic_sensing/docs/",
        "MEASUREMENT_IMPROVEMENT_PLAN.md": "acoustic_sensing/docs/",
        "PROJECT_COMPLETE_SUMMARY.md": "acoustic_sensing/docs/",
        "README_Enhanced_Pipeline.md": "acoustic_sensing/docs/",
        "SALIENCY_ANALYSIS_SUMMARY.md": "acoustic_sensing/docs/",
    }


def check_file_exists_in_new_location(src_dir, old_file, new_location):
    """Check if file exists in new location"""
    if new_location.endswith(".py"):
        # Specific file mapping
        return (src_dir / new_location).exists()
    else:
        # Directory mapping - check if file exists in that directory
        return (src_dir / new_location / old_file).exists()


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup duplicate files after restructuring"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing",
    )
    parser.add_argument(
        "--force", action="store_true", help="Actually remove files (use with caution)"
    )
    args = parser.parse_args()

    src_dir = Path("src")
    if not src_dir.exists():
        print("Error: src directory not found. Run from project root.")
        sys.exit(1)

    migrated_files = get_migrated_files()
    files_to_remove = []
    missing_migrations = []

    print("🔍 Analyzing migrated files...")
    print("=" * 50)

    for old_file, new_location in migrated_files.items():
        old_path = src_dir / old_file

        if old_path.exists():
            # Check if it exists in new location
            if check_file_exists_in_new_location(src_dir, old_file, new_location):
                files_to_remove.append(old_path)
                print(f"✅ {old_file} → {new_location} (can be removed)")
            else:
                missing_migrations.append((old_file, new_location))
                print(f"⚠️  {old_file} → {new_location} (NEW FILE NOT FOUND)")
        else:
            print(f"ℹ️  {old_file} (already removed)")

    print("\n📊 SUMMARY")
    print("=" * 50)
    print(f"Files that can be safely removed: {len(files_to_remove)}")
    print(f"Files with missing migrations: {len(missing_migrations)}")

    if missing_migrations:
        print("\n❌ MISSING MIGRATIONS:")
        for old_file, new_location in missing_migrations:
            print(f"   {old_file} should be at {new_location}")
        print("\nPlease complete the migration before running cleanup.")
        return

    if files_to_remove:
        print("\n📝 FILES TO REMOVE:")
        for file_path in files_to_remove:
            print(f"   {file_path}")

        if args.dry_run:
            print("\n🧪 DRY RUN - No files were actually removed")
            print("   Run with --force to actually remove files")
        elif args.force:
            print("\n🗑️  REMOVING FILES...")
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    print(f"   ✅ Removed {file_path}")
                except Exception as e:
                    print(f"   ❌ Error removing {file_path}: {e}")
            print("\n✅ Cleanup completed!")
        else:
            print("\n⚠️  Use --dry-run to preview or --force to actually remove files")
    else:
        print("\n✅ No duplicate files found - cleanup already completed!")

    # Show new structure
    print("\n🏗️  NEW PACKAGE STRUCTURE:")
    acoustic_sensing_dir = src_dir / "acoustic_sensing"
    if acoustic_sensing_dir.exists():
        for item in sorted(acoustic_sensing_dir.rglob("*.py")):
            if "__pycache__" not in str(item):
                relative_path = item.relative_to(src_dir)
                print(f"   {relative_path}")


if __name__ == "__main__":
    main()
