#!/usr/bin/env python3
"""
Dataset Balance Investigation Script

Analyzes class balance, workspace balance, and object balance across all 3-class datasets.
Investigates potential causes of performance variation (WS1: 85%, WS2: 60%, WS3: 35%).

Usage:
    python analyze_dataset_balance.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import yaml

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Base data directory
BASE_DATA_DIR = "data"

# Dataset definitions from config
TRAINING_DATASETS = [
    "balanced_workspace_1_3class_squares_cutout",
    "balanced_workspace_1_3class_pure_no_contact",
    "balanced_workspace_1_3class_pure_contact",
    "balanced_workspace_3_3class_squares_cutout_v1",
    "balanced_workspace_3_3class_squares_cutout_v2",
    "balanced_workspace_3_3class_pure_contact",
    "balanced_workspace_3_3class_pure_no_contact",
]

VALIDATION_DATASETS = [
    "balanced_workspace_2_3class_squares_cutout",
    "balanced_workspace_2_3class_pure_no_contact",
    "balanced_workspace_2_3class_pure_contact",
]

# Expected classes
EXPECTED_CLASSES = ["contact", "no_contact", "edge"]


def load_dataset_metadata(dataset_name):
    """Load sweep.csv metadata for a dataset."""
    sweep_path = Path(BASE_DATA_DIR) / dataset_name / "sweep.csv"

    if not sweep_path.exists():
        print(f"⚠️  Warning: {sweep_path} not found, skipping...")
        return None

    try:
        df = pd.read_csv(sweep_path)

        # Extract workspace and object info from dataset name
        # Format: balanced_workspace_X_3class_OBJECT
        parts = dataset_name.split("_")
        workspace_idx = parts.index("workspace") + 1
        workspace = f"WS{parts[workspace_idx]}"

        # Extract object type
        if "squares_cutout" in dataset_name:
            object_type = "A_cutout"
        elif "pure_contact" in dataset_name:
            object_type = "C_full"
        elif "pure_no_contact" in dataset_name:
            object_type = "B_empty"
        else:
            object_type = "unknown"

        # Add metadata columns
        df["workspace"] = workspace
        df["object"] = object_type
        df["dataset_name"] = dataset_name

        return df
    except Exception as e:
        print(f"❌ Error loading {sweep_path}: {e}")
        return None


def analyze_overall_balance(all_data):
    """Analyze overall class balance across all datasets."""
    print("\n" + "=" * 80)
    print("📊 PHASE 1: OVERALL CLASS BALANCE ANALYSIS")
    print("=" * 80)

    # Count overall classes
    class_counts = all_data["relabeled_label"].value_counts()
    total_samples = len(all_data)

    print(f"\nTotal Samples: {total_samples:,}")
    print("\nOverall Class Distribution:")
    print("-" * 50)

    for cls in EXPECTED_CLASSES:
        count = class_counts.get(cls, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {cls:15s}: {count:6,} ({percentage:5.1f}%)")

    # Check balance deviation from 33.33%
    print("\nBalance Check (deviation from 33.33%):")
    print("-" * 50)
    expected_pct = 100.0 / len(EXPECTED_CLASSES)

    for cls in EXPECTED_CLASSES:
        count = class_counts.get(cls, 0)
        actual_pct = (count / total_samples * 100) if total_samples > 0 else 0
        deviation = actual_pct - expected_pct
        status = (
            "✅" if abs(deviation) <= 2.0 else "⚠️" if abs(deviation) <= 5.0 else "❌"
        )
        print(f"  {cls:15s}: {deviation:+5.1f}% {status}")

    return class_counts


def analyze_workspace_balance(all_data):
    """Analyze class balance per workspace."""
    print("\n" + "=" * 80)
    print("🏗️  PHASE 2: WORKSPACE-SPECIFIC CLASS BALANCE")
    print("=" * 80)

    workspace_class_dist = {}

    for workspace in sorted(all_data["workspace"].unique()):
        ws_data = all_data[all_data["workspace"] == workspace]
        ws_total = len(ws_data)

        print(f"\n{workspace} Distribution (n={ws_total:,}):")
        print("-" * 50)

        class_counts = ws_data["relabeled_label"].value_counts()
        workspace_class_dist[workspace] = {}

        for cls in EXPECTED_CLASSES:
            count = class_counts.get(cls, 0)
            percentage = (count / ws_total * 100) if ws_total > 0 else 0
            workspace_class_dist[workspace][cls] = {
                "count": count,
                "percentage": percentage,
            }
            print(f"  {cls:15s}: {count:6,} ({percentage:5.1f}%)")

    # Cross-workspace comparison
    print("\n" + "-" * 80)
    print("Cross-Workspace Class Balance Comparison:")
    print("-" * 80)
    print(
        f"{'Workspace':<12} {'Contact %':>12} {'No-Contact %':>15} {'Edge %':>10} {'Total':>10}"
    )
    print("-" * 80)

    for workspace in sorted(workspace_class_dist.keys()):
        contact_pct = workspace_class_dist[workspace]["contact"]["percentage"]
        no_contact_pct = workspace_class_dist[workspace]["no_contact"]["percentage"]
        edge_pct = workspace_class_dist[workspace]["edge"]["percentage"]
        total = sum(
            workspace_class_dist[workspace][cls]["count"] for cls in EXPECTED_CLASSES
        )

        print(
            f"{workspace:<12} {contact_pct:11.1f}% {no_contact_pct:14.1f}% {edge_pct:9.1f}% {total:9,}"
        )

    return workspace_class_dist


def analyze_object_distribution(all_data):
    """Analyze object distribution across workspaces and classes."""
    print("\n" + "=" * 80)
    print("🎯 PHASE 3: OBJECT DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Overall object distribution
    print("\nOverall Object Distribution:")
    print("-" * 50)
    object_counts = all_data["object"].value_counts()
    total = len(all_data)

    for obj, count in object_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {obj:20s}: {count:6,} ({percentage:5.1f}%)")

    # Object-Class mapping
    print("\nObject-Class Mapping:")
    print("-" * 80)
    print(
        f"{'Object':<20} {'Contact %':>12} {'No-Contact %':>15} {'Edge %':>10} {'Total':>10}"
    )
    print("-" * 80)

    for obj in sorted(all_data["object"].unique()):
        obj_data = all_data[all_data["object"] == obj]
        obj_total = len(obj_data)
        class_counts = obj_data["relabeled_label"].value_counts()

        contact_pct = (
            (class_counts.get("contact", 0) / obj_total * 100) if obj_total > 0 else 0
        )
        no_contact_pct = (
            (class_counts.get("no_contact", 0) / obj_total * 100)
            if obj_total > 0
            else 0
        )
        edge_pct = (
            (class_counts.get("edge", 0) / obj_total * 100) if obj_total > 0 else 0
        )

        print(
            f"{obj:<20} {contact_pct:11.1f}% {no_contact_pct:14.1f}% {edge_pct:9.1f}% {obj_total:9,}"
        )

    # Workspace-Object combinations
    print("\nWorkspace × Object Distribution:")
    print("-" * 80)
    print(
        f"{'Workspace':<12} {'A_cutout':>12} {'B_empty':>12} {'C_full':>12} {'Total':>10}"
    )
    print("-" * 80)

    workspace_object_dist = {}
    for workspace in sorted(all_data["workspace"].unique()):
        ws_data = all_data[all_data["workspace"] == workspace]
        ws_total = len(ws_data)
        object_counts = ws_data["object"].value_counts()

        a_count = object_counts.get("A_cutout", 0)
        b_count = object_counts.get("B_empty", 0)
        c_count = object_counts.get("C_full", 0)

        workspace_object_dist[workspace] = {
            "A_cutout": a_count,
            "B_empty": b_count,
            "C_full": c_count,
            "total": ws_total,
        }

        print(
            f"{workspace:<12} {a_count:12,} {b_count:12,} {c_count:12,} {ws_total:9,}"
        )

    return workspace_object_dist


def analyze_spatial_coverage(all_data):
    """Analyze spatial sampling density per workspace."""
    print("\n" + "=" * 80)
    print("📍 PHASE 4: SPATIAL COVERAGE ANALYSIS")
    print("=" * 80)

    for workspace in sorted(all_data["workspace"].unique()):
        ws_data = all_data[all_data["workspace"] == workspace]

        # Count unique positions (using rounded x,y coordinates)
        ws_data["pos_x_rounded"] = ws_data["x"].round(3)
        ws_data["pos_y_rounded"] = ws_data["y"].round(3)
        unique_positions = ws_data.groupby(["pos_x_rounded", "pos_y_rounded"]).size()

        n_positions = len(unique_positions)
        total_samples = len(ws_data)
        avg_samples_per_pos = total_samples / n_positions if n_positions > 0 else 0

        print(f"\n{workspace} Spatial Coverage:")
        print("-" * 50)
        print(f"  Unique Positions:     {n_positions:6,}")
        print(f"  Total Samples:        {total_samples:6,}")
        print(f"  Avg Samples/Position: {avg_samples_per_pos:6.1f}")
        print(f"  Min Samples/Position: {unique_positions.min():6,}")
        print(f"  Max Samples/Position: {unique_positions.max():6,}")

        # Class distribution across positions
        print(f"\n  Position-Class Distribution:")
        position_classes = (
            ws_data.groupby(["pos_x_rounded", "pos_y_rounded", "relabeled_label"])
            .size()
            .unstack(fill_value=0)
        )

        contact_positions = (position_classes.get("contact", 0) > 0).sum()
        no_contact_positions = (position_classes.get("no_contact", 0) > 0).sum()
        edge_positions = (position_classes.get("edge", 0) > 0).sum()

        print(f"    Contact positions:    {contact_positions:6,}")
        print(f"    No-contact positions: {no_contact_positions:6,}")
        print(f"    Edge positions:       {edge_positions:6,}")


def analyze_rotation_splits(train_data, val_data):
    """Analyze train/validation split for rotation."""
    print("\n" + "=" * 80)
    print("🔄 ROTATION ANALYSIS: Train/Validation Split")
    print("=" * 80)

    train_workspaces = ", ".join(sorted(train_data["workspace"].unique()))
    val_workspaces = ", ".join(sorted(val_data["workspace"].unique()))

    print(f"\nTraining Workspaces:   {train_workspaces}")
    print(f"Validation Workspaces: {val_workspaces}")

    print(f"\nTraining Set (n={len(train_data):,}):")
    print("-" * 50)
    train_class_counts = train_data["relabeled_label"].value_counts()
    for cls in EXPECTED_CLASSES:
        count = train_class_counts.get(cls, 0)
        percentage = (count / len(train_data) * 100) if len(train_data) > 0 else 0
        print(f"  {cls:15s}: {count:6,} ({percentage:5.1f}%)")

    print(f"\nValidation Set (n={len(val_data):,}):")
    print("-" * 50)
    val_class_counts = val_data["relabeled_label"].value_counts()
    for cls in EXPECTED_CLASSES:
        count = val_class_counts.get(cls, 0)
        percentage = (count / len(val_data) * 100) if len(val_data) > 0 else 0
        print(f"  {cls:15s}: {count:6,} ({percentage:5.1f}%)")


def create_visualizations(all_data, train_data, val_data, workspace_class_dist):
    """Create visualization plots."""
    print("\n" + "=" * 80)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 80)

    output_dir = Path("data_balance_analysis")
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Overall class distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Balance Analysis", fontsize=16, fontweight="bold")

    # 1a: Overall class balance
    ax = axes[0, 0]
    class_counts = all_data["relabeled_label"].value_counts()
    colors = {"contact": "#2ecc71", "no_contact": "#e74c3c", "edge": "#f39c12"}
    class_colors = [colors.get(cls, "#95a5a6") for cls in class_counts.index]

    class_counts.plot(kind="bar", ax=ax, color=class_colors)
    ax.set_title("Overall Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.axhline(
        y=len(all_data) / 3, color="gray", linestyle="--", label="Expected (33.33%)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add percentages on bars
    for i, (cls, count) in enumerate(class_counts.items()):
        pct = count / len(all_data) * 100
        ax.text(i, count, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")

    # 1b: Workspace class distribution
    ax = axes[0, 1]
    ws_df = pd.DataFrame(
        {
            ws: {
                cls: workspace_class_dist[ws][cls]["count"] for cls in EXPECTED_CLASSES
            }
            for ws in sorted(workspace_class_dist.keys())
        }
    ).T

    ws_df.plot(kind="bar", ax=ax, color=[colors[cls] for cls in EXPECTED_CLASSES])
    ax.set_title("Class Distribution per Workspace")
    ax.set_xlabel("Workspace")
    ax.set_ylabel("Sample Count")
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 1c: Workspace class percentages (stacked)
    ax = axes[1, 0]
    ws_pct_df = pd.DataFrame(
        {
            ws: {
                cls: workspace_class_dist[ws][cls]["percentage"]
                for cls in EXPECTED_CLASSES
            }
            for ws in sorted(workspace_class_dist.keys())
        }
    ).T

    ws_pct_df.plot(
        kind="bar", stacked=True, ax=ax, color=[colors[cls] for cls in EXPECTED_CLASSES]
    )
    ax.set_title("Class Distribution per Workspace (%)")
    ax.set_xlabel("Workspace")
    ax.set_ylabel("Percentage (%)")
    ax.axhline(y=33.33, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=66.67, color="gray", linestyle="--", alpha=0.5)
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # 1d: Object distribution
    ax = axes[1, 1]
    object_counts = all_data["object"].value_counts()
    object_counts.plot(kind="bar", ax=ax, color="#3498db")
    ax.set_title("Object Distribution")
    ax.set_xlabel("Object Type")
    ax.set_ylabel("Sample Count")
    ax.grid(axis="y", alpha=0.3)

    # Add percentages on bars
    for i, (obj, count) in enumerate(object_counts.items()):
        pct = count / len(all_data) * 100
        ax.text(i, count, f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "1_overall_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()

    # Figure 2: Train/Val comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Training vs Validation Split Analysis", fontsize=16, fontweight="bold"
    )

    # 2a: Train vs Val class counts
    ax = axes[0]
    train_counts = train_data["relabeled_label"].value_counts()
    val_counts = val_data["relabeled_label"].value_counts()

    comparison_df = pd.DataFrame(
        {
            "Training": [train_counts.get(cls, 0) for cls in EXPECTED_CLASSES],
            "Validation": [val_counts.get(cls, 0) for cls in EXPECTED_CLASSES],
        },
        index=EXPECTED_CLASSES,
    )

    comparison_df.plot(kind="bar", ax=ax, color=["#3498db", "#e74c3c"])
    ax.set_title("Train vs Validation Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.legend(title="Split")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 2b: Train vs Val percentages
    ax = axes[1]
    train_pct = train_data["relabeled_label"].value_counts() / len(train_data) * 100
    val_pct = val_data["relabeled_label"].value_counts() / len(val_data) * 100

    comparison_pct_df = pd.DataFrame(
        {
            "Training": [train_pct.get(cls, 0) for cls in EXPECTED_CLASSES],
            "Validation": [val_pct.get(cls, 0) for cls in EXPECTED_CLASSES],
        },
        index=EXPECTED_CLASSES,
    )

    comparison_pct_df.plot(kind="bar", ax=ax, color=["#3498db", "#e74c3c"])
    ax.set_title("Train vs Validation Class Distribution (%)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage (%)")
    ax.axhline(y=33.33, color="gray", linestyle="--", label="Expected (33.33%)")
    ax.legend(title="Split")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    output_path = output_dir / "2_train_val_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()

    # Figure 3: Heatmap of workspace × class
    fig, ax = plt.subplots(figsize=(10, 6))

    heatmap_data = pd.DataFrame(
        {
            ws: {
                cls: workspace_class_dist[ws][cls]["percentage"]
                for cls in EXPECTED_CLASSES
            }
            for ws in sorted(workspace_class_dist.keys())
        }
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Percentage (%)"},
        vmin=0,
        vmax=50,
    )
    ax.set_title(
        "Workspace × Class Distribution Heatmap", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Workspace", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)

    plt.tight_layout()
    output_path = output_dir / "3_workspace_class_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {output_path}")
    plt.close()

    print(f"\n📁 All visualizations saved to: {output_dir}/")


def interpret_findings(all_data, workspace_class_dist, train_data, val_data):
    """Provide interpretation and insights."""
    print("\n" + "=" * 80)
    print("🔍 PHASE 5: INTERPRETATION & CRITICAL INSIGHTS")
    print("=" * 80)

    # Calculate key metrics
    overall_balance = all_data["relabeled_label"].value_counts()
    total = len(all_data)

    contact_pct = overall_balance.get("contact", 0) / total * 100
    no_contact_pct = overall_balance.get("no_contact", 0) / total * 100
    edge_pct = overall_balance.get("edge", 0) / total * 100

    print(f"\n1. OVERALL BALANCE ASSESSMENT:")
    print("-" * 50)
    print(f"   Claimed: 33/33/33 split")
    print(
        f"   Actual:  {contact_pct:.1f}/{no_contact_pct:.1f}/{edge_pct:.1f} (Contact/No-Contact/Edge)"
    )

    max_deviation = max(
        abs(contact_pct - 33.33), abs(no_contact_pct - 33.33), abs(edge_pct - 33.33)
    )

    if max_deviation <= 2.0:
        print(
            f"   ✅ BALANCED: Maximum deviation = {max_deviation:.1f}% (within ±2% tolerance)"
        )
    elif max_deviation <= 5.0:
        print(
            f"   ⚠️  SLIGHT IMBALANCE: Maximum deviation = {max_deviation:.1f}% (within ±5% tolerance)"
        )
    else:
        print(
            f"   ❌ SIGNIFICANT IMBALANCE: Maximum deviation = {max_deviation:.1f}% (exceeds ±5%)"
        )

    print(f"\n2. WORKSPACE-SPECIFIC INSIGHTS:")
    print("-" * 50)

    # Compare edge case percentages across workspaces
    ws_edge_pcts = {
        ws: workspace_class_dist[ws]["edge"]["percentage"]
        for ws in workspace_class_dist
    }

    ws1_edge = ws_edge_pcts.get("WS1", 0)
    ws2_edge = ws_edge_pcts.get("WS2", 0)
    ws3_edge = ws_edge_pcts.get("WS3", 0)

    print(f"   Edge case percentages:")
    print(f"     WS1 (val 85%): {ws1_edge:5.1f}%")
    print(f"     WS2 (val 60%): {ws2_edge:5.1f}%")
    print(f"     WS3 (val 35%): {ws3_edge:5.1f}%")

    if ws3_edge > ws1_edge * 1.2:
        print(
            f"   🎯 KEY FINDING: WS3 has {ws3_edge/ws1_edge:.1f}× MORE edge cases than WS1"
        )
        print(f"      → This likely explains why WS3 validation fails (35%)")
        print(
            f"      → Model trained on WS1+WS2 edge patterns can't generalize to WS3 edges"
        )

    if ws1_edge < 30:
        print(f"   🎯 KEY FINDING: WS1 has FEWER edge cases ({ws1_edge:.1f}%)")
        print(f"      → This likely explains why WS1 validation succeeds (85%)")
        print(f"      → Easier workspace with less challenging edge cases")

    print(f"\n3. EFFECTIVE BASELINE RECALCULATION:")
    print("-" * 50)

    # For WS3 validation, recalculate baseline based on actual class distribution
    val_workspaces = val_data["workspace"].unique()
    if "WS2" in val_workspaces:
        ws2_val_data = val_data[val_data["workspace"] == "WS2"]
        ws2_val_counts = ws2_val_data["relabeled_label"].value_counts()
        ws2_val_total = len(ws2_val_data)

        ws2_contact_pct = ws2_val_counts.get("contact", 0) / ws2_val_total * 100
        ws2_no_contact_pct = ws2_val_counts.get("no_contact", 0) / ws2_val_total * 100
        ws2_edge_pct = ws2_val_counts.get("edge", 0) / ws2_val_total * 100

        # Random baseline = largest class percentage
        ws2_baseline = max(ws2_contact_pct, ws2_no_contact_pct, ws2_edge_pct)

        print(f"   WS2 Validation Class Distribution:")
        print(
            f"     Contact: {ws2_contact_pct:.1f}%, No-Contact: {ws2_no_contact_pct:.1f}%, Edge: {ws2_edge_pct:.1f}%"
        )
        print(
            f"   Effective WS2 baseline (random majority): {ws2_baseline:.1f}% (not 33.3%)"
        )
        print(f"   Reported WS2 validation accuracy: 60.0%")
        print(
            f"   Normalized performance: 60.0% / {ws2_baseline:.1f}% = {60.0/ws2_baseline:.2f}× over baseline"
        )

    print(f"\n4. IMPACT ON RESULTS:")
    print("-" * 50)

    if max_deviation <= 5.0:
        print(f"   ✅ Overall results remain VALID")
        print(f"   ✅ 60% average validation is meaningful")
        print(f"   ✅ Class balance within acceptable tolerance")
        print(f"\n   Recommendation:")
        print(
            f"     - Add footnote: 'Class balance verified: {contact_pct:.1f}/{no_contact_pct:.1f}/{edge_pct:.1f}'"
        )
        print(f"     - Clarify workspace-specific distributions in methods")
        print(f"     - Emphasize WS3 difficulty due to edge case prevalence")
    else:
        print(f"   ⚠️  Results may need REVISION")
        print(f"   ⚠️  Significant class imbalance detected")
        print(f"   ⚠️  Baseline assumptions may be incorrect")
        print(f"\n   Recommendation:")
        print(f"     - Re-balance datasets to achieve 33/33/33 split")
        print(f"     - Re-run experiments with balanced data")
        print(f"     - Recalculate all reported metrics")

    print(f"\n5. SUMMARY:")
    print("-" * 50)
    print(f"   Total samples analyzed: {total:,}")
    print(f"   Training samples: {len(train_data):,}")
    print(f"   Validation samples: {len(val_data):,}")
    print(f"   Workspaces: {len(workspace_class_dist)}")
    print(f"   Objects: {len(all_data['object'].unique())}")
    print(f"   Overall balance: {contact_pct:.1f}/{no_contact_pct:.1f}/{edge_pct:.1f}")


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("🔬 DATASET BALANCE INVESTIGATION")
    print("=" * 80)
    print("\nAnalyzing datasets from multi_dataset_config.yml")
    print(f"Training datasets: {len(TRAINING_DATASETS)}")
    print(f"Validation datasets: {len(VALIDATION_DATASETS)}")

    # Load all datasets
    print("\n📂 Loading datasets...")
    all_datasets = []

    for dataset_name in TRAINING_DATASETS + VALIDATION_DATASETS:
        print(f"   Loading: {dataset_name}...", end=" ")
        df = load_dataset_metadata(dataset_name)
        if df is not None:
            all_datasets.append(df)
            print(f"✅ ({len(df):,} samples)")
        else:
            print("❌")

    if not all_datasets:
        print("\n❌ No datasets loaded successfully!")
        return

    # Combine all data
    all_data = pd.concat(all_datasets, ignore_index=True)
    print(
        f"\n✅ Loaded {len(all_data):,} total samples from {len(all_datasets)} datasets"
    )

    # Split into training and validation
    train_data = all_data[all_data["dataset_name"].isin(TRAINING_DATASETS)]
    val_data = all_data[all_data["dataset_name"].isin(VALIDATION_DATASETS)]

    # Run analyses
    class_counts = analyze_overall_balance(all_data)
    workspace_class_dist = analyze_workspace_balance(all_data)
    analyze_object_distribution(all_data)
    analyze_spatial_coverage(all_data)
    analyze_rotation_splits(train_data, val_data)

    # Create visualizations
    create_visualizations(all_data, train_data, val_data, workspace_class_dist)

    # Interpret findings
    interpret_findings(all_data, workspace_class_dist, train_data, val_data)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nCheck the 'data_balance_analysis/' directory for visualizations.")


if __name__ == "__main__":
    main()
