#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple task-phase manager for category-gated AprilTag intent inference."""

import os

import rospy
import yaml
from sensor_msgs.msg import Joy
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
        self.command_topic = str(rospy.get_param("~command_topic", "/task_context/command")).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "/task_context/phase")).strip()
        self.prompt_topic = str(rospy.get_param("~prompt_topic", "/task_context/prompt")).strip()
        self.initial_phase = str(rospy.get_param("~initial_phase", "scan_workspace")).strip().lower()
        self.dpad_horizontal_axis = int(rospy.get_param("~dpad_horizontal_axis", 6))
        self.dpad_vertical_axis = int(rospy.get_param("~dpad_vertical_axis", 7))
        self.axis_trigger_threshold = float(rospy.get_param("~axis_trigger_threshold", 0.5))

        self.tag_meta = self._load_object_map()
        self.phase = ""
        self.latest_axes = []
        self.prev_axes = []

        self.pub_allowed = rospy.Publisher(self.allowed_ids_topic, Int32MultiArray, queue_size=1, latch=True)
        self.pub_phase = rospy.Publisher(self.phase_topic, String, queue_size=1, latch=True)
        self.pub_prompt = rospy.Publisher(self.prompt_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=10)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        self._set_phase(self.initial_phase)

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            rospy.logwarn("[task_context_manager] object map YAML not found: %s", self.object_map_yaml)
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw = data.get("tag_objects", {}) if isinstance(data, dict) else {}
        parsed = {}
        for key, meta in raw.items():
            try:
                tag_id = int(key)
            except Exception:
                continue
            parsed[tag_id] = meta if isinstance(meta, dict) else {}
        return parsed

    def _allowed_ids_for_phase(self, phase):
        if phase in ("scan_workspace", "scan", "all", "select_all"):
            return sorted(self.tag_meta.keys())
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
        if phase in ("grasp_condiment", "select_condiment", "condiment"):
            return "Task phase: SELECT_CONDIMENT. Intent is limited to condiment tags {}. Choose cereal box 1, cereal box 2, or chocolate can. D-pad up=milk, down=condiment, left=scan.".format(allowed_ids)
        if phase in ("grasp_chocolate", "select_chocolate", "chocolate"):
            return "Task phase: SELECT_CHOCOLATE. Intent is limited to chocolate tags {}. D-pad up=milk, down=condiment, left=scan.".format(allowed_ids)
        if phase in ("grasp_milk", "select_milk", "milk"):
            return "Task phase: SELECT_MILK. Intent is limited to milk tags {}. D-pad down=condiment, left=scan.".format(allowed_ids)
        if phase in ("grasp_cereal", "select_cereal", "cereal"):
            return "Task phase: SELECT_CEREAL. Intent is limited to cereal tags {}. D-pad up=milk, left=scan.".format(allowed_ids)
        return "Task phase: SCAN_WORKSPACE. Scan and record all visible tags. D-pad up=milk, down=condiment."

    def _set_phase(self, phase):
        phase = str(phase).strip().lower()
        allowed_ids = self._allowed_ids_for_phase(phase)
        self.phase = phase
        self.pub_phase.publish(String(data=phase))
        self.pub_allowed.publish(Int32MultiArray(data=allowed_ids))
        self.pub_prompt.publish(String(data=self._prompt_for_phase(phase, allowed_ids)))
        rospy.loginfo("[task_context_manager] phase=%s allowed_ids=%s", phase, allowed_ids)

    def _axis_value(self, axes, idx):
        if idx < 0 or idx >= len(axes):
            return 0.0
        return float(axes[idx])

    def _axis_edge(self, idx, direction):
        cur = self._axis_value(self.latest_axes, idx)
        prev = self._axis_value(self.prev_axes, idx)
        thr = self.axis_trigger_threshold
        if direction > 0:
            return cur >= thr and prev < thr
        return cur <= -thr and prev > -thr

    def _joy_cb(self, msg):
        self.prev_axes = list(self.latest_axes)
        self.latest_axes = list(msg.axes)

        if self._axis_edge(self.dpad_vertical_axis, +1):
            self._set_phase("grasp_milk")
            return
        if self._axis_edge(self.dpad_vertical_axis, -1):
            self._set_phase("grasp_condiment")
            return
        if self._axis_edge(self.dpad_horizontal_axis, -1):
            self._set_phase("scan_workspace")
            return

    def _command_cb(self, msg):
        cmd = str(msg.data).strip().lower()
        if not cmd:
            return
        if cmd in (
            "scan_workspace", "scan", "all", "select_all",
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
