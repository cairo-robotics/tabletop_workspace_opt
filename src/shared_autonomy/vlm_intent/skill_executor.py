"""Adapter that executes an inferred goal_spec via the existing TaskExecutor.

Reuses scripts/run_task.py without modifying it: the scripts directory is
added to sys.path, the module-level object maps are set from the scene
config, and TaskExecutor is instantiated once (it waits for the MoveIt and
gripper services at construction).
"""
import os
import sys
from typing import Any, Dict

import rospkg

DEFAULT_ORIENTATION = {"qx": 1.0, "qy": 0.0, "qz": 0.0, "qw": 0.0}
DEFAULT_POUR_ORIENTATION = {"qx": 0.924, "qy": 0.0, "qz": 0.0, "qw": 0.383}
DEFAULT_POUR_HOLD_S = 3.0


class SkillExecutor:
    def __init__(self, scene_name: str, grasp_library_path: str):
        pkg_root = rospkg.RosPack().get_path("tabletop_workspace_opt")
        scripts_dir = os.path.join(pkg_root, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import run_task  # noqa: E402  (deliberate late import)
        self._run_task = run_task

        det_ids, mujoco_names, half_extents = \
            run_task._load_scene_config(scene_name)
        if det_ids is not None:
            run_task.OBJECT_DET_IDS = det_ids
            run_task.OBJECT_MUJOCO_NAMES = mujoco_names
            run_task.OBJECT_HALF_EXTENTS = half_extents

        if not os.path.isabs(grasp_library_path):
            grasp_library_path = os.path.join(pkg_root, grasp_library_path)
        self.executor = run_task.TaskExecutor(grasp_library_path)

    @property
    def holding(self):
        return self.executor.holding

    def execute(self, goal_spec: Dict[str, Any]) -> bool:
        """Run one goal_spec (from the task YAML state machine).
        Returns True on skill success. Never raises."""
        action = goal_spec.get("action")
        try:
            if action == "pick":
                return bool(self.executor.pick(goal_spec["object"]))
            if action == "place":
                return bool(self.executor.place(
                    goal_spec["destination"],
                    goal_spec.get("orientation", dict(DEFAULT_ORIENTATION)),
                    is_last_step=bool(goal_spec.get("is_last_step", False)),
                ))
            if action == "pour":
                return bool(self.executor.pour(
                    goal_spec["destination"],
                    goal_spec.get("orientation", dict(DEFAULT_ORIENTATION)),
                    goal_spec.get("pour_orientation",
                                  dict(DEFAULT_POUR_ORIENTATION)),
                    hold_time=float(goal_spec.get("hold_time",
                                                  DEFAULT_POUR_HOLD_S)),
                ))
            import rospy
            rospy.logwarn("SkillExecutor: unknown action %r", action)
            return False
        except Exception as exc:
            import rospy
            rospy.logerr("SkillExecutor: %s failed: %r", action, exc)
            return False
