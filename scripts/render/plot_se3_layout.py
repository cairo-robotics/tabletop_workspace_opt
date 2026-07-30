#!/usr/bin/env python3
"""Matplotlib top-down visualization of an SE(3) ME-optimized scene layout.

Produces a PNG with:
  - table outline
  - per-object rotated rectangle footprint (from half_extents + yaw)
  - object label
  - per-object grasp marker: arrow from pregrasp to grasp in XY, color
    coded by approach type (red = top-down, blue = side)
  - start (EE home) marker for visual reference

Usage:
    python3 scripts/render/plot_se3_layout.py \
        --scene config/scenes/scene_breakfast_easy_se3_me_optimized.yaml \
        --output results/se3_visualize/breakfast_easy.png

    # Batch mode — render all SE(3) ME-optimized scenes into one directory:
    python3 scripts/render/plot_se3_layout.py --all --output-dir results/se3_visualize
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from envopt.grasp_library import GraspLibrary
from envopt.grasp_visualization import (
    SceneLayout,
    load_optimized_scene,
    resolve_all_grasps,
    find_missing_grasps,
)

EE_HOME_XY = (0.0, 0.0)  # Sawyer home; used only as a visual reference marker.

COLOR_TOP = "#d62728"        # red
COLOR_SIDE = "#1f77b4"       # blue
COLOR_OBJECT = "#aaaaaa"
COLOR_TABLE_EDGE = "#555555"


def plot_scene(
    scene: SceneLayout,
    library: GraspLibrary,
    output_path: str,
    title: str = None,
) -> None:
    """Render one scene + all grasps to a PNG file."""
    resolved = resolve_all_grasps(scene, library)
    missing = find_missing_grasps(scene, library)

    fig, ax = plt.subplots(figsize=(9, 9))

    tx, ty, _ = scene.table_pos
    thx, thy, _ = scene.table_half_extents
    table_rect = patches.Rectangle(
        (tx - thx, ty - thy), 2 * thx, 2 * thy,
        linewidth=1.5, edgecolor=COLOR_TABLE_EDGE, facecolor="#fafafa", zorder=0,
    )
    ax.add_patch(table_rect)

    for obj in scene.objects:
        x, y, _ = obj.position
        hx, hy, _ = obj.half_extents
        rect = patches.Rectangle(
            (x - hx, y - hy), 2 * hx, 2 * hy,
            linewidth=1.2, edgecolor="#333333", facecolor=COLOR_OBJECT,
            alpha=0.55, zorder=1,
        )
        t = (transforms.Affine2D()
             .rotate_around(x, y, obj.yaw)
             + ax.transData)
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.plot(x, y, marker="o", color="#222222", markersize=3, zorder=3)
        ax.annotate(
            obj.short_name,
            xy=(x, y),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9, color="#222222", zorder=4,
        )

    top_seen = False
    side_seen = False
    for rg in resolved:
        gx, gy, _ = rg.grasp_pos
        px, py, _ = rg.pregrasp_pos
        color = COLOR_TOP if rg.approach_type == "top" else COLOR_SIDE
        label = None
        if rg.approach_type == "top" and not top_seen:
            label = "top-down grasp"
            top_seen = True
        elif rg.approach_type == "side" and not side_seen:
            label = "side grasp"
            side_seen = True

        if rg.approach_type == "top":
            ax.plot(
                gx, gy, marker="x", color=color, markersize=10,
                markeredgewidth=2.2, zorder=5, label=label,
            )
        else:
            dx = gx - px
            dy = gy - py
            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                ax.plot(gx, gy, marker="s", color=color, markersize=7,
                        zorder=5, label=label)
            else:
                ax.annotate(
                    "",
                    xy=(gx, gy), xytext=(px, py),
                    arrowprops=dict(
                        arrowstyle="->", color=color, lw=2.0,
                        shrinkA=0, shrinkB=0,
                    ),
                    zorder=5,
                )
                if label is not None:
                    ax.plot([], [], color=color, lw=2.0, label=label)
            ax.plot(gx, gy, marker="o", color=color, markersize=4, zorder=6)

        ax.annotate(
            rg.grasp_name,
            xy=(gx, gy),
            xytext=(8, -10), textcoords="offset points",
            fontsize=7, color=color, zorder=6,
        )

    ax.plot(
        EE_HOME_XY[0], EE_HOME_XY[1],
        marker="*", color="#2ca02c", markersize=16,
        label="EE home (approx.)", zorder=5,
    )

    pad = 0.15
    ax.set_xlim(tx - thx - pad, tx + thx + pad)
    ax.set_ylim(ty - thy - pad, ty + thy + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(title or scene.name)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    stats = (
        f"objects: {len(scene.objects)}   "
        f"top: {sum(1 for r in resolved if r.approach_type == 'top')}   "
        f"side: {sum(1 for r in resolved if r.approach_type == 'side')}"
    )
    if missing:
        stats += f"\nmissing grasps: {', '.join(missing)}"
    ax.text(
        0.99, 0.01, stats,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, lw=0.5),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=str, default=None,
        help="Path to a single scene_*_se3_me_optimized.yaml file.",
    )
    parser.add_argument(
        "--grasp-library", type=str,
        default=os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml"),
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (single-scene mode).")
    parser.add_argument("--all", action="store_true",
                        help="Render every scene_*_se3_me_optimized.yaml under config/scenes/.")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(PROJECT_ROOT, "results", "se3_visualize"),
        help="Destination directory for --all mode.",
    )
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    library = GraspLibrary.load(args.grasp_library)

    if args.all:
        scenes_dir = os.path.join(PROJECT_ROOT, "config", "scenes")
        scene_files = sorted(
            os.path.join(scenes_dir, f)
            for f in os.listdir(scenes_dir)
            if f.endswith("_se3_me_optimized.yaml")
        )
        if not scene_files:
            print(f"No scene_*_se3_me_optimized.yaml files in {scenes_dir}", file=sys.stderr)
            return 1
        os.makedirs(args.output_dir, exist_ok=True)
        for scene_path in scene_files:
            scene = load_optimized_scene(scene_path)
            stem = os.path.splitext(os.path.basename(scene_path))[0]
            out = os.path.join(args.output_dir, f"{stem}.png")
            plot_scene(scene, library, out)
            print(f"[{scene.name}] {len(scene.objects)} objects -> {out}")
        return 0

    if not args.scene:
        parser.error("--scene is required (or pass --all).")
    scene = load_optimized_scene(args.scene)
    out = args.output or os.path.join(
        PROJECT_ROOT, "results", "se3_visualize",
        f"{os.path.splitext(os.path.basename(args.scene))[0]}.png",
    )
    plot_scene(scene, library, out, title=args.title)
    print(f"[{scene.name}] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
