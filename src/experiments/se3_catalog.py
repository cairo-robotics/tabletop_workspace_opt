"""Canonical SE(3) shared-autonomy experiment catalog.

The repo has several SE(3) scripts that need the same scene/task/object
definitions. Keeping the definitions here avoids silent drift between
optimization, evaluation, threshold sweeps, and grasp audits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


START_POS_3D = np.array([0.452, 0.160, 1.05])


@dataclass(frozen=True)
class SE3Tier:
    label: str
    scene: str
    task_2d: str
    task_3d: str
    objects: Tuple[str, ...]
    maxiter: int
    popsize: int
    yaw_steps: int

    @property
    def task(self) -> str:
        """3D task basename used by the SE(3) optimizer scripts."""
        return self.task_3d.rsplit("/", 1)[-1].replace(".yaml", "")

    @property
    def se3_me_optimized_scene(self) -> str:
        return f"{self.scene}_se3_me_optimized"


SE3_TIERS: Tuple[SE3Tier, ...] = (
    SE3Tier(
        label="Easy",
        scene="scene_breakfast_easy",
        task_2d="config/tasks/breakfast_easy_pick_and_return_sa.yaml",
        task_3d="config/tasks/breakfast_easy_pick_and_return_sa_3d.yaml",
        objects=("cereal", "banana"),
        maxiter=25,
        popsize=10,
        yaw_steps=15,
    ),
    SE3Tier(
        label="Med-A",
        scene="scene_desk",
        task_2d="config/tasks/desk_pick_and_return_sa.yaml",
        task_3d="config/tasks/desk_pick_and_return_sa_3d.yaml",
        objects=("mug", "stapler", "pen_cup"),
        maxiter=30,
        popsize=10,
        yaw_steps=15,
    ),
    SE3Tier(
        label="Med-B",
        scene="scene_breakfast",
        task_2d="config/tasks/breakfast_pick_and_return_sa.yaml",
        task_3d="config/tasks/breakfast_pick_and_return_sa_3d.yaml",
        objects=("cereal", "banana", "milk_carton"),
        maxiter=30,
        popsize=10,
        yaw_steps=15,
    ),
    SE3Tier(
        label="Med-C",
        scene="scene_kitchen_prep",
        task_2d="config/tasks/kitchen_pick_and_return_sa.yaml",
        task_3d="config/tasks/kitchen_pick_and_return_sa_3d.yaml",
        objects=("apple", "can", "bottle"),
        maxiter=30,
        popsize=10,
        yaw_steps=15,
    ),
    SE3Tier(
        label="Hard-A",
        scene="scene_meal_assembly",
        task_2d="config/tasks/meal_pick_and_return_sa.yaml",
        task_3d="config/tasks/meal_pick_and_return_sa_3d.yaml",
        objects=("cereal", "banana", "apple", "can", "bottle"),
        maxiter=35,
        popsize=12,
        yaw_steps=12,
    ),
    SE3Tier(
        label="Hard-B",
        scene="scene_cluttered",
        task_2d="config/tasks/cluttered_pick_and_return_sa.yaml",
        task_3d="config/tasks/cluttered_pick_and_return_sa_3d.yaml",
        objects=(
            "red_block", "blue_block", "green_cylinder", "yellow_block",
            "orange_cylinder", "purple_block", "white_cylinder", "pink_block",
        ),
        maxiter=40,
        popsize=12,
        yaw_steps=12,
    ),
)


SE3_TIER_LABELS: Tuple[str, ...] = tuple(t.label for t in SE3_TIERS)


def tier_by_scene(scene: str) -> SE3Tier:
    for tier in SE3_TIERS:
        if tier.scene == scene:
            return tier
    raise KeyError(f"Unknown SE(3) scene: {scene}")


def compare_scene_tiers() -> List[Tuple[str, str, str, List[str]]]:
    """Legacy tuple shape used by comparison/sweep scripts."""
    return [
        (tier.task_2d, tier.scene, tier.label, list(tier.objects))
        for tier in SE3_TIERS
    ]


def grasp_audit_scenes() -> List[Tuple[str, str, List[str]]]:
    """Return (optimized_scene_yaml_name, base_scene_name, pick_objects)."""
    return [
        (tier.se3_me_optimized_scene, tier.scene, list(tier.objects))
        for tier in SE3_TIERS
    ]


def eval_scene_tiers() -> List[Dict]:
    """Legacy dict shape used by cross-observer evaluation scripts."""
    return [
        {
            "name": tier.scene,
            "task_2d": tier.task_2d,
            "task_3d": tier.task_3d,
            "objects": list(tier.objects),
        }
        for tier in SE3_TIERS
    ]


def scene_names(tiers: Iterable[SE3Tier] = SE3_TIERS) -> Tuple[str, ...]:
    return tuple(t.scene for t in tiers)
