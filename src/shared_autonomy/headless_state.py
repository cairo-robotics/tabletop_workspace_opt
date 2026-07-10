"""ROS-free task state machines for headless shared-autonomy evaluation."""
from __future__ import annotations

import numpy as np


class HeadlessTaskStateMachine:
    def __init__(self, task_config, scene_cfg):
        self.states = task_config["states"]
        self.current_state = "initial"
        self.holding = None
        self.scene_cfg = scene_cfg
        self.mujoco_names = {o["short_name"]: o["mujoco_name"]
                             for o in scene_cfg["objects"]}

    def get_valid_goals(self, sim):
        state = self.states.get(self.current_state, {})
        goals = state.get("valid_goals", [])
        result = []
        for g in goals:
            pos = self._compute_goal_pos(g, sim)
            if pos is not None:
                result.append((g, pos))
        return result

    def _compute_goal_pos(self, goal_spec, sim):
        action = goal_spec["action"]
        if action == "pick":
            mname = self.mujoco_names.get(goal_spec["object"])
            if mname:
                return sim.get_object_pos(mname)
        elif action in ("place", "pour"):
            dest = goal_spec["destination"]
            if "absolute" in dest:
                ab = dest["absolute"]
                return np.array([ab["x"], ab["y"], ab["z"]])
            ref = dest.get("reference")
            if ref:
                mname = self.mujoco_names.get(ref)
                if mname:
                    ref_pos = sim.get_object_pos(mname)
                    off = dest.get("offset", {})
                    return ref_pos + np.array([
                        off.get("x", 0), off.get("y", 0), off.get("z", 0)])
        return None

    def transition(self, goal_spec):
        next_state = goal_spec.get("next_state", self.current_state)
        action = goal_spec["action"]
        if action == "pick":
            self.holding = goal_spec["object"]
        elif action == "place":
            self.holding = None
        self.current_state = next_state

    def is_done(self):
        state = self.states.get(self.current_state, {})
        return len(state.get("valid_goals", [])) == 0


class PickAndReturnStateMachine:
    """Dynamic state machine for pick-and-return benchmark tasks."""

    def __init__(self, task_config, scene_cfg):
        self.pick_objects = list(task_config["pick_objects"])
        self.scene_cfg = scene_cfg
        self.mujoco_names = {o["short_name"]: o["mujoco_name"]
                             for o in scene_cfg["objects"]}

        self.picked = set()
        self.holding = None
        self.held_origin = None
        self.current_state = "pick"

    def get_valid_goals(self, sim):
        if self.current_state == "pick":
            goals = []
            for obj in self.pick_objects:
                mname = self.mujoco_names.get(obj)
                if mname:
                    pos = sim.get_object_pos(mname)
                    goals.append(({
                        "id": f"pick_{obj}",
                        "action": "pick",
                        "object": obj,
                    }, pos))
            return goals

        if self.current_state == "return":
            if self.holding and self.held_origin is not None:
                return [({
                    "id": f"return_{self.holding}",
                    "action": "place",
                    "object": self.holding,
                }, self.held_origin)]
            return []

        return []

    def transition(self, goal_spec):
        action = goal_spec["action"]
        if action == "pick":
            self.holding = goal_spec["object"]
            self.current_state = "return"
        elif action == "place":
            self.picked.add(self.holding)
            self.holding = None
            self.held_origin = None
            self.current_state = "pick"

    def save_origin(self, pos):
        self.held_origin = pos.copy()

    def is_done(self):
        return (self.current_state == "pick" and
                len(self.picked) >= len(self.pick_objects))
