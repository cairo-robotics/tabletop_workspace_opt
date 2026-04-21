"""Shared helpers for visualizing optimized SE(3) scene layouts.

Loads a `config/scenes/scene_*_se3_optimized.yaml` layout and the grasp
library `config/grasp_poses_3d.yaml`, resolves every object's grasp into
the world frame, and returns a flat list of dicts ready for plotting or
rendering. Used by `scripts/plot_se3_layout.py` (matplotlib top-down)
and reusable by any future MuJoCo / RViz grasp visualizer.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml

from envopt.grasp_library import GraspLibrary


@dataclass
class ObjectEntry:
    short_name: str
    mujoco_name: str
    position: np.ndarray          # (3,) world frame
    yaw: float                    # radians
    half_extents: np.ndarray      # (3,)


@dataclass
class SceneLayout:
    name: str
    objects: List[ObjectEntry]
    table_pos: np.ndarray
    table_half_extents: np.ndarray


@dataclass
class ResolvedGrasp:
    short_name: str
    obj_pos: np.ndarray
    obj_yaw: float
    obj_half_extents: np.ndarray
    grasp_pos: np.ndarray         # (3,) world frame
    grasp_R: np.ndarray           # (3,3) world frame
    pregrasp_pos: np.ndarray      # (3,) world frame
    approach_type: str            # "top" or "side"
    approach_world: np.ndarray    # (3,) unit vector, world-frame direction from pregrasp -> grasp
    grasp_name: str = ""          # human label from the library (e.g. "top_center", "side_x")


def load_optimized_scene(scene_yaml_path: str) -> SceneLayout:
    """Load an optimized scene YAML.

    Skips placeholder entries with `det_id: -1` or missing `position`/`yaw`
    — those are stubs for trays, bins, and held-object markers that do
    not participate in the layout optimization.
    """
    with open(scene_yaml_path) as f:
        raw = yaml.safe_load(f)
    scene_block = raw.get("scene", raw)
    name = scene_block.get("name", os.path.basename(scene_yaml_path))

    objects: List[ObjectEntry] = []
    for obj in scene_block.get("objects", []):
        if obj.get("det_id") == -1:
            continue
        if "position" not in obj or "yaw" not in obj:
            continue
        objects.append(ObjectEntry(
            short_name=obj["short_name"],
            mujoco_name=obj.get("mujoco_name", obj["short_name"]),
            position=np.asarray(obj["position"], dtype=float),
            yaw=float(obj["yaw"]),
            half_extents=np.asarray(obj["half_extents"], dtype=float),
        ))

    table = scene_block.get("table", {})
    return SceneLayout(
        name=name,
        objects=objects,
        table_pos=np.asarray(table.get("pos_world", [0.6, 0.0, 0.915]), dtype=float),
        table_half_extents=np.asarray(table.get("half_extents", [0.4, 0.7, 0.027]), dtype=float),
    )


def _classify_approach(grasp_pos: np.ndarray, pregrasp_pos: np.ndarray) -> (str, np.ndarray):
    """Return ('top'|'side', unit approach direction in world frame).

    Approach direction points FROM the pre-grasp TO the grasp. If its
    z-component dominates (|dz| > max(|dx|, |dy|)), we call it top.
    """
    approach = grasp_pos - pregrasp_pos
    n = np.linalg.norm(approach)
    if n < 1e-9:
        return "top", np.array([0.0, 0.0, -1.0])
    approach_unit = approach / n
    ax, ay, az = abs(approach_unit[0]), abs(approach_unit[1]), abs(approach_unit[2])
    kind = "top" if az > max(ax, ay) else "side"
    return kind, approach_unit


def resolve_all_grasps(
    scene: SceneLayout,
    library: GraspLibrary,
) -> List[ResolvedGrasp]:
    """Resolve the grasp library entry for every object in the scene.

    Objects whose short_name is not in the library are silently skipped
    and reported in the returned list's `missing` attribute via the
    caller's own check — this helper does not warn.
    """
    resolved: List[ResolvedGrasp] = []
    for obj in scene.objects:
        entry = library.get(obj.short_name)
        if entry is None:
            continue
        poses = entry.resolve(obj.position, obj.yaw)
        kind, approach_world = _classify_approach(
            poses["grasp_pos"], poses["pregrasp_pos"])
        resolved.append(ResolvedGrasp(
            short_name=obj.short_name,
            obj_pos=obj.position,
            obj_yaw=obj.yaw,
            obj_half_extents=obj.half_extents,
            grasp_pos=poses["grasp_pos"],
            grasp_R=poses["grasp_R"],
            pregrasp_pos=poses["pregrasp_pos"],
            approach_type=kind,
            approach_world=approach_world,
            grasp_name=entry.name,
        ))
    return resolved


def find_missing_grasps(
    scene: SceneLayout,
    library: GraspLibrary,
) -> List[str]:
    """Return object short_names that are in the scene but missing a library entry."""
    return [o.short_name for o in scene.objects if library.get(o.short_name) is None]
