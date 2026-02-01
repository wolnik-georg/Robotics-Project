#!/usr/bin/env python3
"""
Generate animated ground truth visualization for presentation.
Shows the surface being "painted" as the robot moves through positions.

Output: GIF animation showing progressive ground truth reveal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import imageio.v2 as imageio
import os


def create_animated_gt(sweep_csv_path: str, output_path: str, fps: int = 10):
    """
    Create animated GIF showing ground truth being painted progressively.

    Args:
        sweep_csv_path: Path to sweep.csv with position data
        output_path: Output GIF path
        fps: Frames per second
    """
    # Load data
    df = pd.read_csv(sweep_csv_path)

    # Get unique positions sorted by point_index (order of collection)
    df_unique = df.drop_duplicates(subset=["point_index"]).sort_values("point_index")

    # Extract coordinates and labels
    x = df_unique["normalized_x"].values
    y = df_unique["normalized_y"].values
    labels = df_unique["relabeled_label"].values

    # Color mapping
    color_map = {
        "contact": "#2ecc71",  # Green
        "no_contact": "#e74c3c",  # Red
        "edge": "#f39c12",  # Orange
    }

    # Create frames directory
    frames_dir = Path(output_path).parent / "frames_temp"
    frames_dir.mkdir(exist_ok=True)

    n_positions = len(x)
    frame_paths = []

    # Generate frames - show progressive reveal
    # Use fewer frames for speed (every 2nd or 3rd position)
    step = max(1, n_positions // 50)  # ~50 frames max
    frame_indices = list(range(0, n_positions, step)) + [n_positions - 1]

    print(f"Generating {len(frame_indices)} frames...")

    for frame_num, end_idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot all positions up to current point
        for i in range(end_idx + 1):
            color = color_map.get(labels[i], "#95a5a6")
            alpha = 1.0 if i == end_idx else 0.8  # Current point slightly highlighted
            size = 150 if i == end_idx else 100
            ax.scatter(
                x[i],
                y[i],
                c=color,
                s=size,
                alpha=alpha,
                edgecolors="white",
                linewidths=0.5,
            )

        # Show current robot position marker
        ax.scatter(
            x[end_idx],
            y[end_idx],
            c="blue",
            s=300,
            marker="x",
            linewidths=3,
            label="Robot Position",
        )

        # Styling
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("X Position (normalized)", fontsize=12)
        ax.set_ylabel("Y Position (normalized)", fontsize=12)
        ax.set_title(
            f"Ground Truth Surface Mapping\nPosition {end_idx + 1} / {n_positions}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Legend
        legend_elements = [
            mpatches.Patch(color="#2ecc71", label="Contact"),
            mpatches.Patch(color="#e74c3c", label="No Contact"),
            mpatches.Patch(color="#f39c12", label="Edge"),
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="blue",
                linestyle="None",
                markersize=10,
                label="Robot",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Progress bar at bottom
        progress = (end_idx + 1) / n_positions
        ax.axhline(y=-0.05, xmin=0, xmax=progress, color="#3498db", linewidth=8)
        ax.text(0.5, -0.08, f"{int(progress*100)}% Complete", ha="center", fontsize=10)

        # Save frame
        frame_path = frames_dir / f"frame_{frame_num:04d}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
        plt.close()
        frame_paths.append(str(frame_path))

        if frame_num % 10 == 0:
            print(f"  Frame {frame_num + 1}/{len(frame_indices)}")

    # Add pause frames at the end (show final result longer)
    for _ in range(fps * 2):  # 2 seconds pause
        frame_paths.append(frame_paths[-1])

    # Create GIF
    print("Creating GIF...")
    images = [imageio.imread(fp) for fp in frame_paths]
    imageio.mimsave(output_path, images, fps=fps, loop=0)

    # Cleanup temp frames
    for fp in set(frame_paths):
        os.remove(fp)
    frames_dir.rmdir()

    print(f"✅ Animation saved to: {output_path}")
    return output_path


def create_simple_static_comparison(sweep_csv_path: str, output_path: str):
    """
    Create a simple static side-by-side: empty grid → full ground truth.
    For use in presentation if animated GIF doesn't work.
    """
    df = pd.read_csv(sweep_csv_path)
    df_unique = df.drop_duplicates(subset=["point_index"]).sort_values("point_index")

    x = df_unique["normalized_x"].values
    y = df_unique["normalized_y"].values
    labels = df_unique["relabeled_label"].values

    color_map = {"contact": "#2ecc71", "no_contact": "#e74c3c", "edge": "#f39c12"}
    colors = [color_map.get(l, "#95a5a6") for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Empty grid with robot path
    ax1 = axes[0]
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title("Robot Sweep Path", fontsize=14, fontweight="bold")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")
    # Show path as line
    ax1.plot(x, y, "b-", alpha=0.3, linewidth=1)
    ax1.scatter(x, y, c="lightgray", s=50, alpha=0.5)
    ax1.scatter(x[0], y[0], c="green", s=200, marker="o", label="Start", zorder=5)
    ax1.scatter(x[-1], y[-1], c="red", s=200, marker="s", label="End", zorder=5)
    ax1.legend()

    # Right: Full ground truth
    ax2 = axes[1]
    ax2.scatter(x, y, c=colors, s=100, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title("Ground Truth Surface Map", fontsize=14, fontweight="bold")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="Contact"),
        mpatches.Patch(color="#e74c3c", label="No Contact"),
        mpatches.Patch(color="#f39c12", label="Edge"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ Static comparison saved to: {output_path}")


if __name__ == "__main__":
    # Find a good dataset with squares_cutout (has interesting pattern)
    data_paths = [
        "data/collected_data_runs_2026_01_15_workspace1/sweep.csv",
        "data/collected_data_runs_2025_12_15_v1/workspace_1_squares_cutout/sweep.csv",
    ]

    # Find first existing file
    sweep_path = None
    for p in data_paths:
        if Path(p).exists():
            sweep_path = p
            break

    if sweep_path is None:
        # Find any sweep.csv
        import glob

        files = glob.glob("data/**/sweep.csv", recursive=True)
        if files:
            # Prefer one with 'squares' or 'cutout' in path
            for f in files:
                if "squares" in f or "cutout" in f:
                    sweep_path = f
                    break
            if sweep_path is None:
                sweep_path = files[0]

    if sweep_path is None:
        print("❌ No sweep.csv found!")
        exit(1)

    print(f"Using: {sweep_path}")

    # Create output directory
    output_dir = Path("presentation_animations")
    output_dir.mkdir(exist_ok=True)

    # Generate static comparison first (fast)
    create_simple_static_comparison(
        sweep_path, str(output_dir / "ground_truth_static_comparison.png")
    )

    # Generate animated GIF
    create_animated_gt(
        sweep_path, str(output_dir / "ground_truth_animation.gif"), fps=10
    )

    print(f"\n✅ All outputs in: {output_dir}/")
