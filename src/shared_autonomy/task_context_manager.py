#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple task-phase manager for category-gated AprilTag intent inference."""

import os

import rospy
import yaml
from std_msgs.msg import Int32MultiArray, String


class TaskContextManager:
    def __init__(self):
        rospy.init_node("task_context_manager")

        self.object_map_yaml = os.path.expanduser(
            rospy.get_param(
                "~object_map_yaml",
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    "config",
                    "apriltag_object_map.yaml",
                ),
            )
        )
        self.tasks_yaml = os.path.expanduser(
            rospy.get_param(
                "~tasks_yaml",
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    "config",
                    "user_study_tasks.yaml",
                ),
            )
        )
        self.command_topic = str(rospy.get_param("~command_topic", "/task_context/command")).strip()
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "/task_context/phase")).strip()
        self.prompt_topic = str(rospy.get_param("~prompt_topic", "/task_context/prompt")).strip()
        self.initial_phase = str(rospy.get_param("~initial_phase", "scan_workspace")).strip().lower()

        self.tag_meta = self._load_object_map()
        self.phase_step_map = self._load_phase_step_map()
        self.phase = ""

        self.pub_allowed = rospy.Publisher(self.allowed_ids_topic, Int32MultiArray, queue_size=1, latch=True)
        self.pub_phase = rospy.Publisher(self.phase_topic, String, queue_size=1, latch=True)
        self.pub_prompt = rospy.Publisher(self.prompt_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=10)
        self._set_phase(self.initial_phase)

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            rospy.logwarn("[task_context_manager] object map YAML not found: %s", self.object_map_yaml)
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict) and isinstance(data.get("tag_objects"), dict):
            raw = data.get("tag_objects", {}) or {}
        elif isinstance(data, dict) and isinstance(data.get("candidate_objects"), dict):
            raw = data.get("candidate_objects", {}) or {}
        else:
            raw = {}
        parsed = {}
        for key, meta in raw.items():
            try:
                tag_id = int(key)
            except Exception:
                continue
            parsed[tag_id] = meta if isinstance(meta, dict) else {}
        return parsed

    def _load_phase_step_map(self):
        if not os.path.exists(self.tasks_yaml):
            rospy.logwarn("[task_context_manager] tasks YAML not found: %s", self.tasks_yaml)
            return {}
        with open(self.tasks_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw_tasks = data.get("tasks", {}) if isinstance(data, dict) else {}
        phase_steps = {}
        for task_id, task in raw_tasks.items():
            if not isinstance(task, dict):
                continue
            task_name = str(task.get("display_name", task_id)).strip()
            for step in list(task.get("steps", []) or []):
                if not isinstance(step, dict):
                    continue
                step_info = {
                    "task_id": str(task_id).strip(),
                    "task_name": task_name,
                    "step_id": str(step.get("id", "")).strip(),
                    "title": str(step.get("title", "")).strip(),
                    "description": str(step.get("description", "")).strip(),
                    "allowed_tag_ids": [int(v) for v in list(step.get("allowed_tag_ids", []) or [])],
                }
                for phase_name in (step_info["step_id"], str(step.get("command", "")).strip()):
                    phase_key = phase_name.lower()
                    if phase_key:
                        phase_steps[phase_key] = step_info
        return phase_steps

    def _yaml_step_for_phase(self, phase):
        return self.phase_step_map.get(str(phase).strip().lower(), {})

    def _allowed_ids_for_phase(self, phase):
        yaml_step = self._yaml_step_for_phase(phase)
        if yaml_step:
            return sorted([int(v) for v in list(yaml_step.get("allowed_tag_ids", []) or [])])
        if phase in ("scan_workspace", "scan", "all", "select_all"):
            return sorted(self.tag_meta.keys())
        if phase in ("select_breakfast_ingredient", "breakfast_ingredient"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("breakfast_group", "")).strip() == "ingredient"
                ]
            )
        if phase in ("select_breakfast_milk", "breakfast_milk"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("breakfast_group", "")).strip() == "milk"
                ]
            )
        if phase in ("select_sort_object", "sort_object"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("sorting_group", "")).strip() in ("fruit", "cereal_like")
                ]
            )
        if phase in ("select_lego_brick", "lego_brick"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("category", "")).strip() == "lego"
                ]
            )
        if phase in ("select_sort_destination", "sort_destination"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("destination_group", "")).strip() == "sorting_target"
                ]
            )
        if phase in ("grasp_fruit", "select_fruit", "fruit"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("category", "")).strip() == "fruit"
                ]
            )
        if phase in ("grasp_destination", "select_destination", "destination"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("category", "")).strip() == "destination"
                ]
            )
        if phase in ("avoid_soy_milk", "select_soy_free_milk", "soy_free_milk"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("object_name", "")).strip() in ("pure_milk", "oat_milk")
                ]
            )
        if phase in ("avoid_dairy_milk", "select_dairy_free_milk", "dairy_free_milk"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("object_name", "")).strip() in ("oat_milk", "soy_milk")
                ]
            )
        if phase in ("grasp_condiment", "select_condiment", "condiment"):
            return sorted(
                [
                    tid
                    for tid, meta in self.tag_meta.items()
                    if str(meta.get("category", "")).strip() in ("cereal", "chocolate")
                ]
            )
        if phase in ("grasp_chocolate", "select_chocolate", "chocolate"):
            return sorted([tid for tid, meta in self.tag_meta.items() if str(meta.get("category", "")).strip() == "chocolate"])
        if phase in ("grasp_milk", "select_milk", "milk"):
            return sorted([tid for tid, meta in self.tag_meta.items() if str(meta.get("category", "")).strip() == "milk"])
        if phase in ("grasp_cereal", "select_cereal", "cereal"):
            return sorted([tid for tid, meta in self.tag_meta.items() if str(meta.get("category", "")).strip() == "cereal"])
        return sorted(self.tag_meta.keys())

    def _prompt_for_phase(self, phase, allowed_ids):
        yaml_step = self._yaml_step_for_phase(phase)
        if yaml_step:
            task_name = str(yaml_step.get("task_name", "")).strip() or "Current Task"
            title = str(yaml_step.get("title", "")).strip() or str(yaml_step.get("step_id", phase)).strip()
            description = str(yaml_step.get("description", "")).strip()
            prompt = "Task: {}. Step: {}. Allowed tags: {}.".format(task_name, title, allowed_ids)
            if description:
                prompt += " " + description
            return prompt
        if phase in ("select_breakfast_ingredient", "breakfast_ingredient"):
            return "Task phase: SELECT_BREAKFAST_INGREDIENT. Intent is limited to breakfast ingredient tags {}. Choose cereal box 1, cereal box 2, or chocolate powder.".format(allowed_ids)
        if phase in ("select_breakfast_milk", "breakfast_milk"):
            return "Task phase: SELECT_BREAKFAST_MILK. Intent is limited to milk tags {}. Choose one milk carton for breakfast.".format(allowed_ids)
        if phase in ("select_sort_object", "sort_object"):
            return "Task phase: SELECT_SORT_OBJECT. Intent is limited to sorting item tags {}. Choose one fruit or cereal-related item.".format(allowed_ids)
        if phase in ("select_lego_brick", "lego_brick"):
            return "Task phase: SELECT_LEGO_BRICK. Intent is limited to LEGO tags {}. Choose one LEGO brick to pick up before sorting.".format(allowed_ids)
        if phase in ("select_sort_destination", "sort_destination"):
            return "Task phase: SELECT_SORT_DESTINATION. Intent is limited to destination tags {}. Guide the grasped item toward one of the available sorting destinations, including the plate, bowl, or tagged sorting containers.".format(allowed_ids)
        if phase in ("grasp_fruit", "select_fruit", "fruit"):
            return "Task phase: SELECT_FRUIT. Intent is limited to fruit tag(s) {}. Choose the apple, lemon, or orange and confirm when prompted.".format(allowed_ids)
        if phase in ("grasp_destination", "select_destination", "destination"):
            return "Task phase: SELECT_DESTINATION. Intent is limited to destination tag(s) {}. Choose the plate or bowl, then release the fruit there.".format(allowed_ids)
        if phase in ("avoid_soy_milk", "select_soy_free_milk", "soy_free_milk"):
            return "Task phase: SELECT_SOY_FREE_MILK. Intent is limited to soy-free milk tags {}. Choose pure milk or oat milk.".format(allowed_ids)
        if phase in ("avoid_dairy_milk", "select_dairy_free_milk", "dairy_free_milk"):
            return "Task phase: SELECT_DAIRY_FREE_MILK. Intent is limited to dairy-free milk tags {}. Choose oat milk or soy milk.".format(allowed_ids)
        if phase in ("grasp_condiment", "select_condiment", "condiment"):
            return "Task phase: SELECT_CONDIMENT. Intent is limited to condiment tags {}. Choose cereal box 1, cereal box 2, or chocolate can.".format(allowed_ids)
        if phase in ("grasp_chocolate", "select_chocolate", "chocolate"):
            return "Task phase: SELECT_CHOCOLATE. Intent is limited to chocolate tags {}.".format(allowed_ids)
        if phase in ("grasp_milk", "select_milk", "milk"):
            return "Task phase: SELECT_MILK. Intent is limited to milk tags {}.".format(allowed_ids)
        if phase in ("grasp_cereal", "select_cereal", "cereal"):
            return "Task phase: SELECT_CEREAL. Intent is limited to cereal tags {}.".format(allowed_ids)
        return "Task phase: SCAN_WORKSPACE. Scan and record all visible tags."

    def _set_phase(self, phase):
        phase = str(phase).strip().lower()
        allowed_ids = self._allowed_ids_for_phase(phase)
        self.phase = phase
        self.pub_phase.publish(String(data=phase))
        self.pub_allowed.publish(Int32MultiArray(data=allowed_ids))
        self.pub_prompt.publish(String(data=self._prompt_for_phase(phase, allowed_ids)))
        rospy.loginfo("[task_context_manager] phase=%s allowed_ids=%s", phase, allowed_ids)

    def _command_cb(self, msg):
        cmd = str(msg.data).strip().lower()
        if not cmd:
            return
        if cmd in self.phase_step_map:
            self._set_phase(cmd)
            return
        if cmd in (
            "scan_workspace", "scan", "all", "select_all",
            "select_breakfast_ingredient", "breakfast_ingredient",
            "select_breakfast_milk", "breakfast_milk",
            "select_sort_object", "sort_object",
            "select_lego_brick", "lego_brick",
            "select_sort_destination", "sort_destination",
            "grasp_fruit", "select_fruit", "fruit",
            "grasp_destination", "select_destination", "destination",
            "avoid_soy_milk", "select_soy_free_milk", "soy_free_milk",
            "avoid_dairy_milk", "select_dairy_free_milk", "dairy_free_milk",
            "grasp_condiment", "select_condiment", "condiment",
            "grasp_chocolate", "select_chocolate", "chocolate",
            "grasp_milk", "select_milk", "milk",
            "grasp_cereal", "select_cereal", "cereal",
            "reset_task", "reset",
        ):
            self._set_phase(self.initial_phase if cmd in ("reset_task", "reset") else cmd)
            return
        rospy.logwarn("[task_context_manager] ignoring unknown command: %s", cmd)


if __name__ == "__main__":
    TaskContextManager()
    rospy.spin()
