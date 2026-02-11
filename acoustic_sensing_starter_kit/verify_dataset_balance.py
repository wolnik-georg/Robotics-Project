#!/usr/bin/env python3
"""
Verify and Visualize Dataset Balance
====================================

Purpose: Verify that fully_balanced_datasets are actually 33/33/33 balanced
         across classes (contact/no-contact/edge) and 50/50 balanced across
         workspaces.

Generates:
- Console report with exact counts
- Visualization plots showing balance
- Summary table for report/supplementary materials

Author: Georg Wolnik
Date: February 9, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 11


def load_dataset_metadata(csv_path):
    """Load dataset and extract metadata."""
    df = pd.read_csv(csv_path)

    # The sweep.csv already has workspace_id, object_type, label columns
    # No need to extract anything

    return df


def count_class_distribution(df, dataset_name):
    """Count samples per class."""
    class_counts = df["label"].value_counts().sort_index()
    total = len(df)

    print(f"\n{'='*60}")
    print(f"📊 {dataset_name}")
    print(f"{'='*60}")
    print(f"Total Samples: {total:,}")
    print(f"\nClass Distribution:")

    for label, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"  {label:12s}: {count:5,} samples ({percentage:5.2f}%)")

    # Check balance (should be ~33.33% each)
    expected_per_class = total / 3
    print(f"\nBalance Check (expected ~{expected_per_class:.0f} per class):")

    for label, count in class_counts.items():
        deviation = count - expected_per_class
        deviation_pct = (deviation / expected_per_class) * 100
        status = "✅" if abs(deviation_pct) < 5 else "⚠️"
        print(
            f"  {label:12s}: {deviation:+6.1f} samples ({deviation_pct:+5.2f}%) {status}"
        )

    return class_counts


def count_workspace_distribution(df, dataset_name):
    """Count samples per workspace."""
    workspace_counts = df["workspace_id"].value_counts().sort_index()
    total = len(df)

    print(f"\nWorkspace Distribution:")

    for ws, count in workspace_counts.items():
        percentage = (count / total) * 100
        print(f"  {ws}: {count:5,} samples ({percentage:5.2f}%)")

    # Check balance (should be ~50% each for 2-workspace training sets)
    if len(workspace_counts) == 2:
        expected_per_ws = total / 2
        print(
            f"\nWorkspace Balance Check (expected ~{expected_per_ws:.0f} per workspace):"
        )

        for ws, count in workspace_counts.items():
            deviation = count - expected_per_ws
            deviation_pct = (deviation / expected_per_ws) * 100
            status = "✅" if abs(deviation_pct) < 5 else "⚠️"
            print(
                f"  {ws}: {deviation:+6.1f} samples ({deviation_pct:+5.2f}%) {status}"
            )

    return workspace_counts


def count_object_distribution(df, dataset_name):
    """Count samples per object type."""
    object_counts = df["object_type"].value_counts().sort_index()
    total = len(df)

    print(f"\nObject Distribution:")

    for obj, count in object_counts.items():
        percentage = (count / total) * 100
        print(f"  {obj}: {count:5,} samples ({percentage:5.2f}%)")

    return object_counts


def create_balance_visualization(all_stats, output_dir):
    """Create comprehensive visualization of dataset balance."""

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        "Dataset Balance Verification: Fully Balanced Datasets",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    rotations = ["Rotation 1", "Rotation 2", "Rotation 3"]
    dataset_types = ["Train", "Val"]

    # Color palette
    class_colors = {"contact": "#2ecc71", "no-contact": "#e74c3c", "edge": "#3498db"}

    # Plot class distributions
    for i, rotation in enumerate(rotations):
        for j, dtype in enumerate(dataset_types):
            ax = axes[i, j]

            key = f"{rotation} {dtype}"
            if key not in all_stats:
                ax.axis("off")
                continue

            stats = all_stats[key]
            class_counts = stats["class_counts"]
            total = stats["total"]

            # Create bar plot
            labels = list(class_counts.keys())
            counts = list(class_counts.values())
            colors = [class_colors.get(label, "#95a5a6") for label in labels]

            bars = ax.bar(
                labels,
                counts,
                color=colors,
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            )

            # Add percentage labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                percentage = (count / total) * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{percentage:.1f}%\n({count:,})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            # Add 33.33% reference line
            ax.axhline(
                y=total / 3,
                color="gray",
                linestyle="--",
                linewidth=2,
                alpha=0.5,
                label="Perfect Balance (33.33%)",
            )

            ax.set_title(
                f"{rotation} {dtype}\n({total:,} samples)",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_ylabel("Sample Count", fontsize=10)
            ax.set_ylim(0, max(counts) * 1.2)
            ax.grid(axis="y", alpha=0.3)

            if i == 2:
                ax.set_xlabel("Class", fontsize=10)

    # Summary statistics in third column
    for i in range(3):
        ax = axes[i, 2]
        ax.axis("off")

        rotation = rotations[i]

        # Gather statistics for this rotation
        train_key = f"{rotation} Train"
        val_key = f"{rotation} Val"

        if train_key in all_stats and val_key in all_stats:
            train_stats = all_stats[train_key]
            val_stats = all_stats[val_key]

            summary_text = f"{rotation} Summary\n" + "=" * 30 + "\n\n"

            # Train stats
            summary_text += f"Training Set:\n"
            summary_text += f"  Total: {train_stats['total']:,} samples\n"
            summary_text += (
                f"  Workspaces: {', '.join(train_stats['workspace_counts'].keys())}\n"
            )
            summary_text += f"  Balance: "

            train_class_counts = train_stats["class_counts"]
            train_total = train_stats["total"]
            deviations = []
            for label, count in train_class_counts.items():
                expected = train_total / 3
                dev_pct = abs((count - expected) / expected * 100)
                deviations.append(dev_pct)

            max_dev = max(deviations)
            if max_dev < 2:
                summary_text += "✅ Excellent (<2%)\n"
            elif max_dev < 5:
                summary_text += "✅ Good (<5%)\n"
            else:
                summary_text += "⚠️ Needs Review\n"

            summary_text += "\n"

            # Val stats
            summary_text += f"Validation Set:\n"
            summary_text += f"  Total: {val_stats['total']:,} samples\n"
            summary_text += (
                f"  Workspaces: {', '.join(val_stats['workspace_counts'].keys())}\n"
            )
            summary_text += f"  Balance: "

            val_class_counts = val_stats["class_counts"]
            val_total = val_stats["total"]
            deviations = []
            for label, count in val_class_counts.items():
                expected = val_total / 3
                dev_pct = abs((count - expected) / expected * 100)
                deviations.append(dev_pct)

            max_dev = max(deviations)
            if max_dev < 2:
                summary_text += "✅ Excellent (<2%)\n"
            elif max_dev < 5:
                summary_text += "✅ Good (<5%)\n"
            else:
                summary_text += "⚠️ Needs Review\n"

            ax.text(
                0.1,
                0.5,
                summary_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
            )

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "dataset_balance_verification.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved visualization: {output_path}")

    return output_path


def create_summary_table(all_stats, output_dir):
    """Create summary table for report."""

    rows = []

    for key in sorted(all_stats.keys()):
        stats = all_stats[key]

        # Parse key
        parts = key.split()
        rotation = parts[0] + " " + parts[1]  # "Rotation 1"
        dtype = parts[2]  # "Train" or "Val"

        # Class percentages
        total = stats["total"]
        class_counts = stats["class_counts"]

        contact_pct = (class_counts.get("contact", 0) / total) * 100
        no_contact_pct = (class_counts.get("no-contact", 0) / total) * 100
        edge_pct = (class_counts.get("edge", 0) / total) * 100

        # Workspace info
        workspaces = ", ".join(sorted(stats["workspace_counts"].keys()))

        rows.append(
            {
                "Rotation": rotation,
                "Type": dtype,
                "Total": total,
                "Workspaces": workspaces,
                "Contact %": f"{contact_pct:.1f}",
                "No-Contact %": f"{no_contact_pct:.1f}",
                "Edge %": f"{edge_pct:.1f}",
                "Max Deviation": f"{max(abs(contact_pct - 33.33), abs(no_contact_pct - 33.33), abs(edge_pct - 33.33)):.1f}",
            }
        )

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / "dataset_balance_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved summary table: {csv_path}")

    # Print formatted table
    print(f"\n{'='*100}")
    print("📋 DATASET BALANCE SUMMARY TABLE")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}\n")

    return csv_path


def verify_all_datasets(base_dir="data/fully_balanced_datasets"):
    """Main verification function."""

    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"❌ ERROR: Directory not found: {base_path}")
        return

    print("=" * 60)
    print("🔍 DATASET BALANCE VERIFICATION")
    print("=" * 60)
    print(f"Base Directory: {base_path}")
    print(f"Checking for balanced 33/33/33 class splits...")
    print("=" * 60)

    all_stats = {}

    # Check each rotation
    for rotation_num in [1, 2, 3]:
        for dtype in ["train", "val"]:
            # Datasets are directories with sweep.csv inside
            dataset_dir = base_path / f"rotation{rotation_num}_{dtype}"
            csv_file = dataset_dir / "sweep.csv"

            if not csv_file.exists():
                print(f"⚠️  WARNING: File not found: {csv_file}")
                continue

            # Load and analyze
            df = load_dataset_metadata(csv_file)
            dataset_name = (
                f"Rotation {rotation_num} {'Train' if dtype == 'train' else 'Val'}"
            )

            # Count distributions
            class_counts = count_class_distribution(df, dataset_name)
            workspace_counts = count_workspace_distribution(df, dataset_name)
            object_counts = count_object_distribution(df, dataset_name)

            # Store stats
            key = dataset_name
            all_stats[key] = {
                "total": len(df),
                "class_counts": class_counts.to_dict(),
                "workspace_counts": workspace_counts.to_dict(),
                "object_counts": object_counts.to_dict(),
            }

    # Create output directory
    output_dir = Path("analysis_results/balance_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations and tables
    create_balance_visualization(all_stats, output_dir)
    create_summary_table(all_stats, output_dir)

    # Save raw statistics as JSON
    json_path = output_dir / "dataset_statistics.json"
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"✅ Saved raw statistics: {json_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("\nFiles generated:")
    print(f"  1. dataset_balance_verification.png  (visualization)")
    print(f"  2. dataset_balance_summary.csv       (summary table)")
    print(f"  3. dataset_statistics.json           (raw statistics)")
    print("=" * 60)


if __name__ == "__main__":
    verify_all_datasets()
