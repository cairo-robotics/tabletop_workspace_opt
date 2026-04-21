"""Loader for the 3D grasp library used by the SE(3) shared-autonomy.

The 3D library has exactly one entry per object, specified in the
object's body frame. The entry fields are:

    name:           human-readable label.
    offset:         {x, y, z} grasp tip offset in the object frame.
    orientation:    {qw, qx, qy, qz} grasp tip orientation in object frame.
    approach:
        standoff:   pre-grasp standoff distance in meters.
        axis:       [x, y, z] unit vector in the grasp frame pointing
                    FROM the pre-grasp TO the grasp. Default [0, 0, -1]
                    (standard top-down).

See `config/grasp_poses_3d.yaml` for the canonical file.
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import yaml

from envopt.se3_utils import compose_grasp_pose, pregrasp_pose, quat_to_rot


@dataclass
class GraspEntry:
    name: str
    local_offset: np.ndarray      # (3,) in object frame
    local_quat: np.ndarray        # (w, x, y, z) in object frame
    approach_standoff: float      # meters
    approach_axis: np.ndarray     # (3,) in grasp frame

    def resolve(self, obj_pos, obj_yaw):
        """Compose the world-frame grasp and pre-grasp poses.

        Returns a dict with:
            grasp_pos, grasp_quat, grasp_R   — the grasp in world frame
            pregrasp_pos, pregrasp_quat      — pre-grasp standoff pose
        """
        grasp_pos, grasp_quat = compose_grasp_pose(
            obj_pos, obj_yaw, self.local_offset, self.local_quat)
        grasp_R = quat_to_rot(grasp_quat)
        pg_pos, pg_quat = pregrasp_pose(
            grasp_pos, grasp_quat,
            self.approach_standoff, self.approach_axis)
        return {
            "grasp_pos": grasp_pos,
            "grasp_quat": grasp_quat,
            "grasp_R": grasp_R,
            "pregrasp_pos": pg_pos,
            "pregrasp_quat": pg_quat,
        }


class GraspLibrary:
    """Dict-like wrapper around the grasp library YAML.

    Usage:
        lib = GraspLibrary.load('config/grasp_poses_3d.yaml')
        entry = lib['banana']
        poses = entry.resolve(obj_pos=[0.6, 0.1, 0.95], obj_yaw=0.0)
    """

    def __init__(self, entries: Dict[str, GraspEntry]):
        self.entries = entries

    def __getitem__(self, name: str) -> GraspEntry:
        return self.entries[name]

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def get(self, name: str, default=None) -> Optional[GraspEntry]:
        return self.entries.get(name, default)

    @classmethod
    def load(cls, yaml_path: str) -> "GraspLibrary":
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        entries = {}
        for obj_name, spec in data.get("grasps", {}).items():
            entries[obj_name] = cls._parse_entry(obj_name, spec)
        return cls(entries)

    @staticmethod
    def _parse_entry(obj_name: str, spec: dict) -> GraspEntry:
        offset = spec["offset"]
        local_offset = np.array([
            offset.get("x", 0.0),
            offset.get("y", 0.0),
            offset.get("z", 0.0),
        ], dtype=float)

        orient = spec["orientation"]
        # Accept either qw/qx/qy/qz or qx/qy/qz/qw naming
        w = orient.get("qw", orient.get("w", 1.0))
        x = orient.get("qx", orient.get("x", 0.0))
        y = orient.get("qy", orient.get("y", 0.0))
        z = orient.get("qz", orient.get("z", 0.0))
        local_quat = np.array([w, x, y, z], dtype=float)
        n = np.linalg.norm(local_quat)
        if n < 1e-9:
            raise ValueError(f"Zero quaternion in grasp '{obj_name}'")
        local_quat /= n

        approach = spec.get("approach", {})
        standoff = float(approach.get("standoff", 0.10))
        axis = approach.get("axis", [0.0, 0.0, -1.0])
        approach_axis = np.asarray(axis, dtype=float)
        approach_axis /= (np.linalg.norm(approach_axis) + 1e-12)

        name = spec.get("name", f"{obj_name}_grasp")
        return GraspEntry(
            name=name,
            local_offset=local_offset,
            local_quat=local_quat,
            approach_standoff=standoff,
            approach_axis=approach_axis,
        )


def default_library_path(project_root: str) -> str:
    return os.path.join(project_root, "config", "grasp_poses_3d.yaml")
