#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Dataset Balancing Pipeline - One Script Does Everything

This script takes raw collected datasets and creates fully balanced datasets
with 3-level balancing:
    1. Class balance: 33/33/33 (contact/no_contact/edge)
    2. Object balance: 33/33/33 (A_cutout/B_empty/C_full)
    3. Workspace balance: 50/50 (for training set combinations)

Outputs rotation-ready datasets that can be directly used in config files.

USAGE:
------
    # Create all balanced datasets for all rotations
    python create_fully_balanced_datasets.py --config balance_config.yml

    # Or specify datasets manually
    python create_fully_balanced_datasets.py \
        --ws1-cutout data/collected_data_runs_2026_01_15_workspace_1_squares_cutout_relabeled \
        --ws1-contact data/collected_data_runs_2026_01_15_workspace_1_pure_contact_relabeled \
        --ws1-no-contact data/collected_data_runs_2026_01_15_workspace_1_pure_no_contact \
        --ws2-cutout ... --ws2-contact ... --ws2-no-contact ... \
        --ws3-cutout ... --ws3-contact ... --ws3-no-contact ... \
        --output-dir data/fully_balanced_datasets

OUTPUTS:
--------
    data/fully_balanced_datasets/
    ├─ workspace_1_balanced/          # Individual balanced workspaces
    ├─ workspace_2_balanced/
    ├─ workspace_3_balanced/
    ├─ workspace_4_balanced/          # Holdout (if provided)
    ├─ rotation1_train/               # WS1+WS3 balanced 50/50
    ├─ rotation1_val/                 # WS2
    ├─ rotation2_train/               # WS2+WS3 balanced 50/50
    ├─ rotation2_val/                 # WS1
    ├─ rotation3_train/               # WS1+WS2 balanced 50/50
    ├─ rotation3_val/                 # WS3
    └─ summary_report.txt             # Balance statistics

Then update your config to use these:
    datasets:
      - "fully_balanced_datasets/rotation1_train"
    validation_datasets:
      - "fully_balanced_datasets/rotation1_val"
"""

import numpy as np
import pandas as pd
import os
import shutil
import argparse
import yaml
from pathlib import Path
from collections import Counter
import random

RANDOM_SEED = 42
EXPECTED_CLASSES = ["contact", "no_contact", "edge"]


class DatasetBalancer:
    """Handles all dataset loading, combining, and balancing operations."""

    def __init__(self, seed=RANDOM_SEED):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def load_dataset(self, dataset_path, workspace_id, object_type):
        """
        Load a single dataset with metadata.

        Args:
            dataset_path: Path to dataset folder (containing data/ and sweep.csv)
            workspace_id: Workspace identifier (1, 2, 3, 4)
            object_type: Object type ('A_cutout', 'B_empty', 'C_full')

        Returns:
            DataFrame with sweep data including workspace and object metadata
        """
        sweep_path = Path(dataset_path) / "sweep.csv"

        if not sweep_path.exists():
            print(f"⚠️  Warning: {sweep_path} not found, skipping...")
            return None

        try:
            df = pd.read_csv(sweep_path)

            # Add metadata
            df["workspace_id"] = f"WS{workspace_id}"
            df["object_type"] = object_type
            df["source_dataset"] = str(dataset_path)

            # Extract filename if needed
            if "filename" not in df.columns:
                df["filename"] = df["acoustic_filename"].apply(
                    lambda x: os.path.basename(x) if pd.notna(x) else None
                )

            # Get label column
            if "relabeled_label" in df.columns:
                df["label"] = df["relabeled_label"]
            elif "original_label" in df.columns:
                df["label"] = df["original_label"]
            else:
                print(f"❌ Error: No label column found in {sweep_path}")
                return None

            print(f"✅ Loaded {len(df):,} samples from {dataset_path}")
            return df

        except Exception as e:
            print(f"❌ Error loading {sweep_path}: {e}")
            return None

    def combine_objects_for_workspace(self, object_dfs):
        """
        Combine multiple object datasets for a single workspace.

        Args:
            object_dfs: List of DataFrames, one per object type

        Returns:
            Combined DataFrame
        """
        valid_dfs = [df for df in object_dfs if df is not None]

        if not valid_dfs:
            return None

        combined_df = pd.concat(valid_dfs, ignore_index=True)
        return combined_df

    def balance_classes(self, df):
        """
        Balance classes to 33/33/33 using undersampling.

        Args:
            df: DataFrame with 'label' column

        Returns:
            Balanced DataFrame
        """
        class_counts = df["label"].value_counts()

        print(f"\n  Class distribution before balancing:")
        for cls in EXPECTED_CLASSES:
            count = class_counts.get(cls, 0)
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            print(f"    {cls:15s}: {count:6,} ({pct:5.1f}%)")

        # Find minimum count
        available_classes = [
            cls for cls in EXPECTED_CLASSES if cls in class_counts.index
        ]
        min_count = min([class_counts[cls] for cls in available_classes])

        print(f"\n  Undersampling to {min_count} samples per class...")

        # Sample each class
        balanced_dfs = []
        for cls in available_classes:
            class_df = df[df["label"] == cls]
            if len(class_df) > min_count:
                sampled_df = class_df.sample(n=min_count, random_state=self.seed)
            else:
                sampled_df = class_df
            balanced_dfs.append(sampled_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        print(f"\n  Class distribution after balancing:")
        for cls in EXPECTED_CLASSES:
            count = (balanced_df["label"] == cls).sum()
            pct = (count / len(balanced_df) * 100) if len(balanced_df) > 0 else 0
            print(f"    {cls:15s}: {count:6,} ({pct:5.1f}%)")

        return balanced_df

    def balance_objects(self, df):
        """
        Balance object types to 33/33/33 while maintaining class balance.

        Args:
            df: DataFrame with 'object_type' and 'label' columns

        Returns:
            Balanced DataFrame
        """
        object_counts = df["object_type"].value_counts()

        print(f"\n  Object distribution before balancing:")
        for obj in sorted(df["object_type"].unique()):
            count = object_counts.get(obj, 0)
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            print(f"    {obj:15s}: {count:6,} ({pct:5.1f}%)")

        # Find minimum object count
        min_object_count = object_counts.min()

        print(f"\n  Balancing objects to {min_object_count} samples each...")

        # For each object, undersample while maintaining class balance within that object
        balanced_object_dfs = []

        for obj in sorted(df["object_type"].unique()):
            obj_df = df[df["object_type"] == obj]

            # Calculate target samples per class for this object
            # We want min_object_count total samples with balanced classes
            obj_classes = obj_df["label"].unique()
            n_classes = len(obj_classes)
            samples_per_class = min_object_count // n_classes

            # Sample each class within this object
            obj_class_dfs = []
            for cls in obj_classes:
                obj_class_df = obj_df[obj_df["label"] == cls]
                if len(obj_class_df) >= samples_per_class:
                    sampled = obj_class_df.sample(
                        n=samples_per_class, random_state=self.seed
                    )
                else:
                    sampled = obj_class_df
                obj_class_dfs.append(sampled)

            balanced_obj_df = pd.concat(obj_class_dfs, ignore_index=True)
            balanced_object_dfs.append(balanced_obj_df)

        balanced_df = pd.concat(balanced_object_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        print(f"\n  Object distribution after balancing:")
        for obj in sorted(balanced_df["object_type"].unique()):
            count = (balanced_df["object_type"] == obj).sum()
            pct = (count / len(balanced_df) * 100) if len(balanced_df) > 0 else 0
            print(f"    {obj:15s}: {count:6,} ({pct:5.1f}%)")

        return balanced_df

    def balance_workspaces(self, workspace_dfs):
        """
        Balance multiple workspaces to equal representation (50/50 for 2 workspaces, etc.).
        THEN re-balance classes to 33/33/33.

        Args:
            workspace_dfs: List of (workspace_id, DataFrame) tuples

        Returns:
            Combined balanced DataFrame
        """
        if len(workspace_dfs) <= 1:
            return workspace_dfs[0][1] if workspace_dfs else None

        print(f"\n  Workspace distribution before balancing:")
        for ws_id, ws_df in workspace_dfs:
            print(f"    {ws_id}: {len(ws_df):6,} samples")

        # Find minimum workspace count
        min_ws_count = min(len(ws_df) for _, ws_df in workspace_dfs)

        print(f"\n  Balancing workspaces to {min_ws_count} samples each...")

        # Undersample each workspace
        balanced_ws_dfs = []
        for ws_id, ws_df in workspace_dfs:
            if len(ws_df) > min_ws_count:
                # Stratified sampling: maintain class balance while undersampling
                sampled_dfs = []
                for cls in ws_df["label"].unique():
                    cls_df = ws_df[ws_df["label"] == cls]
                    cls_proportion = len(cls_df) / len(ws_df)
                    cls_target = int(min_ws_count * cls_proportion)
                    if len(cls_df) >= cls_target and cls_target > 0:
                        sampled_cls = cls_df.sample(
                            n=cls_target, random_state=self.seed
                        )
                    else:
                        sampled_cls = cls_df
                    sampled_dfs.append(sampled_cls)
                sampled_ws = pd.concat(sampled_dfs, ignore_index=True)
            else:
                sampled_ws = ws_df

            balanced_ws_dfs.append((ws_id, sampled_ws))

        # Combine all workspaces
        combined_df = pd.concat(
            [ws_df for _, ws_df in balanced_ws_dfs], ignore_index=True
        )
        combined_df = combined_df.sample(frac=1, random_state=self.seed).reset_index(
            drop=True
        )

        print(f"\n  Workspace distribution after workspace balancing:")
        for ws_id, ws_df in balanced_ws_dfs:
            count = len(ws_df)
            pct = (count / len(combined_df) * 100) if len(combined_df) > 0 else 0
            print(f"    {ws_id}: {count:6,} samples ({pct:5.1f}%)")

        # CRITICAL: Re-balance classes after combining workspaces!
        # The combined dataset may have class imbalance even if workspaces are balanced
        print(f"\n  Re-balancing classes after workspace combination...")
        combined_df = self.balance_classes(combined_df)

        return combined_df

    def save_balanced_dataset(self, df, output_dir, dataset_name):
        """
        Save balanced dataset with sweep.csv and audio files.

        Args:
            df: Balanced DataFrame
            output_dir: Base output directory
            dataset_name: Name for this dataset (e.g., 'workspace_1_balanced')
        """
        output_path = Path(output_dir) / dataset_name
        output_data_dir = output_path / "data"

        # Create directories
        if output_path.exists():
            shutil.rmtree(output_path)
        output_data_dir.mkdir(parents=True, exist_ok=True)

        # Prepare new sweep data
        new_sweep_rows = []

        # Copy files and build new sweep CSV
        for idx, (_, row) in enumerate(df.iterrows()):
            # Source file
            src_filename = row["filename"]
            source_dataset = row["source_dataset"]
            src_path = Path(source_dataset) / "data" / src_filename

            # New filename with sequential counter
            label = row["label"]
            new_filename = f"{idx}_{label}.wav"
            dst_path = output_data_dir / new_filename

            # Copy file
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"⚠️  Warning: Source file not found: {src_path}")
                continue

            # Create new sweep row
            new_row = row.copy()
            new_row["acoustic_filename"] = f"./data/{new_filename}"
            new_row["filename"] = new_filename

            # Use 'label' as the canonical label column
            if "relabeled_label" in new_row:
                new_row["relabeled_label"] = label
            if "original_label" in new_row:
                new_row["original_label"] = label

            new_sweep_rows.append(new_row)

        # Save new sweep.csv
        new_sweep_df = pd.DataFrame(new_sweep_rows)
        sweep_output_path = output_path / "sweep.csv"
        new_sweep_df.to_csv(sweep_output_path, index=False)

        print(f"\n✅ Saved {dataset_name}:")
        print(f"   - {len(new_sweep_rows):,} audio files in {output_data_dir}")
        print(f"   - sweep.csv at {sweep_output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Complete dataset balancing pipeline - from raw data to rotation-ready datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Workspace 1 inputs
    parser.add_argument(
        "--ws1-cutout", type=str, help="Workspace 1 cutout dataset path"
    )
    parser.add_argument(
        "--ws1-contact", type=str, help="Workspace 1 pure contact dataset path"
    )
    parser.add_argument(
        "--ws1-no-contact", type=str, help="Workspace 1 pure no-contact dataset path"
    )

    # Workspace 2 inputs
    parser.add_argument(
        "--ws2-cutout", type=str, help="Workspace 2 cutout dataset path"
    )
    parser.add_argument(
        "--ws2-contact", type=str, help="Workspace 2 pure contact dataset path"
    )
    parser.add_argument(
        "--ws2-no-contact", type=str, help="Workspace 2 pure no-contact dataset path"
    )

    # Workspace 3 inputs
    parser.add_argument(
        "--ws3-cutout", type=str, help="Workspace 3 cutout dataset path"
    )
    parser.add_argument(
        "--ws3-contact", type=str, help="Workspace 3 pure contact dataset path"
    )
    parser.add_argument(
        "--ws3-no-contact", type=str, help="Workspace 3 pure no-contact dataset path"
    )

    # Workspace 4 (holdout) input
    parser.add_argument(
        "--ws4-holdout", type=str, help="Workspace 4 holdout dataset path (Object D)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/fully_balanced_datasets",
        help="Output directory for all balanced datasets",
    )

    # Options
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-rotations",
        action="store_true",
        help="Only create individual workspace datasets, skip rotation combinations",
    )

    args = parser.parse_args()

    # Initialize balancer
    balancer = DatasetBalancer(seed=args.seed)

    print("=" * 80)
    print("COMPLETE DATASET BALANCING PIPELINE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics for summary
    summary_stats = {}

    # ========================================================================
    # STEP 1: Process Individual Workspaces (3-Level Balance)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: CREATING BALANCED INDIVIDUAL WORKSPACES")
    print("=" * 80)

    workspace_paths = {}
    workspace_dfs = {}

    for ws_id in [1, 2, 3, 4]:
        ws_name = f"workspace_{ws_id}"

        # Get dataset paths for this workspace
        if ws_id == 1:
            cutout_path = args.ws1_cutout
            contact_path = args.ws1_contact
            no_contact_path = args.ws1_no_contact
        elif ws_id == 2:
            cutout_path = args.ws2_cutout
            contact_path = args.ws2_contact
            no_contact_path = args.ws2_no_contact
        elif ws_id == 3:
            cutout_path = args.ws3_cutout
            contact_path = args.ws3_contact
            no_contact_path = args.ws3_no_contact
        elif ws_id == 4:
            # Workspace 4 is special - single holdout dataset
            if not args.ws4_holdout:
                continue
            print(f"\n{'='*60}")
            print(f"Processing Workspace {ws_id} (Holdout - Object D)")
            print("=" * 60)

            ws4_df = balancer.load_dataset(args.ws4_holdout, ws_id, "D_holdout")
            if ws4_df is None:
                continue

            # Balance classes only (no object balancing for single object)
            balanced_ws4 = balancer.balance_classes(ws4_df)

            # Save
            ws4_output = balancer.save_balanced_dataset(
                balanced_ws4, args.output_dir, f"{ws_name}_balanced"
            )
            workspace_paths[ws_id] = ws4_output
            workspace_dfs[ws_id] = balanced_ws4

            summary_stats[ws_name] = {
                "total_samples": len(balanced_ws4),
                "class_balance": balanced_ws4["label"].value_counts().to_dict(),
                "object_balance": balanced_ws4["object_type"].value_counts().to_dict(),
            }
            continue

        # Skip if paths not provided
        if not cutout_path and not contact_path and not no_contact_path:
            continue

        print(f"\n{'='*60}")
        print(f"Processing Workspace {ws_id}")
        print("=" * 60)

        # Load object datasets
        object_dfs = []

        if cutout_path:
            cutout_df = balancer.load_dataset(cutout_path, ws_id, "A_cutout")
            if cutout_df is not None:
                object_dfs.append(cutout_df)

        if contact_path:
            contact_df = balancer.load_dataset(contact_path, ws_id, "C_full")
            if contact_df is not None:
                object_dfs.append(contact_df)

        if no_contact_path:
            no_contact_df = balancer.load_dataset(no_contact_path, ws_id, "B_empty")
            if no_contact_df is not None:
                object_dfs.append(no_contact_df)

        if not object_dfs:
            print(f"⚠️  No valid datasets found for Workspace {ws_id}, skipping...")
            continue

        # Combine objects
        print(f"\nCombining {len(object_dfs)} object dataset(s)...")
        combined_ws = balancer.combine_objects_for_workspace(object_dfs)

        if combined_ws is None:
            continue

        print(f"Combined: {len(combined_ws):,} total samples")

        # Apply 3-level balancing
        print(f"\n--- Level 1: Balancing Classes (Initial) ---")
        balanced_ws = balancer.balance_classes(combined_ws)

        if len(balanced_ws["object_type"].unique()) > 1:
            print(f"\n--- Level 2: Balancing Objects ---")
            balanced_ws = balancer.balance_objects(balanced_ws)

            # Re-balance classes after object balancing
            print(f"\n--- Level 1b: Re-balancing Classes (After Object Balance) ---")
            balanced_ws = balancer.balance_classes(balanced_ws)

        # Save balanced workspace
        ws_output = balancer.save_balanced_dataset(
            balanced_ws, args.output_dir, f"{ws_name}_balanced"
        )

        workspace_paths[ws_id] = ws_output
        workspace_dfs[ws_id] = balanced_ws

        # Track stats
        summary_stats[ws_name] = {
            "total_samples": len(balanced_ws),
            "class_balance": balanced_ws["label"].value_counts().to_dict(),
            "object_balance": balanced_ws["object_type"].value_counts().to_dict(),
        }

    # ========================================================================
    # STEP 2: Create Rotation Datasets (Workspace Balance)
    # ========================================================================
    if not args.skip_rotations and len(workspace_dfs) >= 3:
        print("\n" + "=" * 80)
        print("STEP 2: CREATING ROTATION DATASETS WITH WORKSPACE BALANCE")
        print("=" * 80)

        rotations = [
            {"name": "rotation1", "train": [1, 3], "val": [2]},
            {"name": "rotation2", "train": [2, 3], "val": [1]},
            {"name": "rotation3", "train": [1, 2], "val": [3]},
        ]

        for rotation in rotations:
            rot_name = rotation["name"]
            train_ws_ids = rotation["train"]
            val_ws_ids = rotation["val"]

            print(f"\n{'='*60}")
            print(f"Creating {rot_name.upper()}")
            print(f"Train: WS{'+WS'.join(map(str, train_ws_ids))}")
            print(f"Val:   WS{'+WS'.join(map(str, val_ws_ids))}")
            print("=" * 60)

            # Prepare training workspaces
            train_ws_dfs = [
                (f"WS{ws_id}", workspace_dfs[ws_id])
                for ws_id in train_ws_ids
                if ws_id in workspace_dfs
            ]

            if not train_ws_dfs:
                print(f"⚠️  No training workspaces available for {rot_name}")
                continue

            # Balance workspaces (Level 3)
            print(f"\n--- Level 3: Balancing Workspaces (50/50 split) ---")
            balanced_train = balancer.balance_workspaces(train_ws_dfs)

            # Save training set
            train_output = balancer.save_balanced_dataset(
                balanced_train, args.output_dir, f"{rot_name}_train"
            )

            # Prepare validation set (just copy individual workspace)
            val_ws_id = val_ws_ids[0]
            if val_ws_id in workspace_dfs:
                val_df = workspace_dfs[val_ws_id]
                val_output = balancer.save_balanced_dataset(
                    val_df, args.output_dir, f"{rot_name}_val"
                )

            # Track stats
            summary_stats[f"{rot_name}_train"] = {
                "total_samples": len(balanced_train),
                "class_balance": balanced_train["label"].value_counts().to_dict(),
                "workspace_balance": balanced_train["workspace_id"]
                .value_counts()
                .to_dict(),
            }
            summary_stats[f"{rot_name}_val"] = {
                "total_samples": len(val_df),
                "class_balance": val_df["label"].value_counts().to_dict(),
                "workspace_balance": val_df["workspace_id"].value_counts().to_dict(),
            }

    # ========================================================================
    # STEP 3: Generate Summary Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    summary_path = output_dir / "balance_summary_report.txt"

    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULLY BALANCED DATASETS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        for dataset_name, stats in summary_stats.items():
            f.write(f"\n{dataset_name.upper()}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Samples: {stats['total_samples']:,}\n\n")

            if "class_balance" in stats:
                f.write("Class Balance:\n")
                for cls in EXPECTED_CLASSES:
                    count = stats["class_balance"].get(cls, 0)
                    pct = (
                        (count / stats["total_samples"] * 100)
                        if stats["total_samples"] > 0
                        else 0
                    )
                    f.write(f"  {cls:15s}: {count:6,} ({pct:5.1f}%)\n")
                f.write("\n")

            if "object_balance" in stats:
                f.write("Object Balance:\n")
                for obj, count in sorted(stats["object_balance"].items()):
                    pct = (
                        (count / stats["total_samples"] * 100)
                        if stats["total_samples"] > 0
                        else 0
                    )
                    f.write(f"  {obj:15s}: {count:6,} ({pct:5.1f}%)\n")
                f.write("\n")

            if "workspace_balance" in stats:
                f.write("Workspace Balance:\n")
                for ws, count in sorted(stats["workspace_balance"].items()):
                    pct = (
                        (count / stats["total_samples"] * 100)
                        if stats["total_samples"] > 0
                        else 0
                    )
                    f.write(f"  {ws:15s}: {count:6,} ({pct:5.1f}%)\n")
                f.write("\n")

    print(f"\n✅ Summary report saved to: {summary_path}")

    # ========================================================================
    # STEP 4: Generate Config Template
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING CONFIG TEMPLATES")
    print("=" * 80)

    config_template = """# Generated Config for Balanced Datasets
# Use these datasets in your multi_dataset_config.yml

# ROTATION 1: Train WS1+WS3, Validate WS2
datasets:
  - "fully_balanced_datasets/rotation1_train"
validation_datasets:
  - "fully_balanced_datasets/rotation1_val"

# ROTATION 2: Train WS2+WS3, Validate WS1
# datasets:
#   - "fully_balanced_datasets/rotation2_train"
# validation_datasets:
#   - "fully_balanced_datasets/rotation2_val"

# ROTATION 3: Train WS1+WS2, Validate WS3
# datasets:
#   - "fully_balanced_datasets/rotation3_train"
# validation_datasets:
#   - "fully_balanced_datasets/rotation3_val"

# All datasets are balanced:
# - Classes: 33/33/33 (contact/no_contact/edge)
# - Objects: 33/33/33 (A_cutout/B_empty/C_full)
# - Workspaces: 50/50 (in training sets)
"""

    config_path = output_dir / "example_config.yml"
    with open(config_path, "w") as f:
        f.write(config_template)

    print(f"\n✅ Example config saved to: {config_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll balanced datasets saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review summary report: {summary_path}")
    print(f"  2. Update your config with rotation datasets")
    print(f"  3. Run experiments with fully balanced data!")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
