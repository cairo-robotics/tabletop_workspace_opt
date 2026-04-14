#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge AprilTag intent selection into the existing filtered executor."""

import copy
import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32, String


def _parse_int_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [int(v) for v in txt.split() if v]
    return [int(v) for v in default]


class AprilTagIntentGraspBridge:
    def __init__(self):
        rospy.init_node("apriltag_intent_grasp_bridge")

        self.tag_ids = _parse_int_list_param("~tag_ids", [0, 1, 2])
        self.input_namespace_prefix = str(rospy.get_param("~input_namespace_prefix", "/apriltag_candidates/tag_")).strip()
        self.top_goal_topic = str(rospy.get_param("~top_goal_topic", "/apriltag_intent_inference/top_goal")).strip()
        self.top_probability_topic = str(
            rospy.get_param("~top_probability_topic", "/apriltag_intent_inference/top_probability")
        ).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()
        self.output_pregrasp_topic = str(rospy.get_param("~output_pregrasp_topic", "/tag_grasp_demo/pregrasp_pose")).strip()
        self.output_grasp_topic = str(rospy.get_param("~output_grasp_topic", "/tag_grasp_demo/grasp_pose")).strip()
        self.prompt_topic = str(
            rospy.get_param("~prompt_topic", "/apriltag_intent_inference/confirmation_prompt")
        ).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
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
        self.select_threshold = float(rospy.get_param("~select_threshold", 0.60))
        self.confirm_button_index = int(rospy.get_param("~confirm_button_index", 2))
        self.cancel_button_index = int(rospy.get_param("~cancel_button_index", 3))

        self.top_goal = None
        self.top_prob = 0.0
        self.latest_buttons = []
        self.prev_buttons = []
        self.tag_poses = {}
        self.object_map = self._load_object_map()
        self.last_status = ""

        self.pub_pre = rospy.Publisher(self.output_pregrasp_topic, PoseStamped, queue_size=1, latch=True)
        self.pub_grasp = rospy.Publisher(self.output_grasp_topic, PoseStamped, queue_size=1, latch=True)
        self.pub_prompt = rospy.Publisher(self.prompt_topic, String, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.pub_selected = rospy.Publisher(self.selected_grasp_label_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.top_goal_topic, String, self._top_goal_cb, queue_size=1)
        rospy.Subscriber(self.top_probability_topic, Float32, self._top_prob_cb, queue_size=1)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)

        for tag_id in self.tag_ids:
            ns = "{}{}".format(self.input_namespace_prefix, tag_id)
            rospy.Subscriber("{}/pregrasp_pose".format(ns), PoseStamped, self._pre_cb, callback_args=tag_id, queue_size=1)
            rospy.Subscriber("{}/grasp_pose".format(ns), PoseStamped, self._grasp_cb, callback_args=tag_id, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self._tick)
        self._publish_status("waiting_for_intent_selection")

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("tag_objects", {}) if isinstance(data, dict) else {}

    def _entry(self, tag_id):
        if tag_id not in self.tag_poses:
            self.tag_poses[tag_id] = {"pregrasp": None, "grasp": None}
        return self.tag_poses[tag_id]

    def _pre_cb(self, msg, tag_id):
        self._entry(tag_id)["pregrasp"] = copy.deepcopy(msg)

    def _grasp_cb(self, msg, tag_id):
        self._entry(tag_id)["grasp"] = copy.deepcopy(msg)

    def _top_goal_cb(self, msg):
        txt = str(msg.data).strip()
        self.top_goal = int(txt) if txt and txt.lstrip("-").isdigit() else None

    def _top_prob_cb(self, msg):
        self.top_prob = float(msg.data)

    def _joy_cb(self, msg):
        self.prev_buttons = list(self.latest_buttons)
        self.latest_buttons = list(msg.buttons)

    def _pressed_edge(self, idx):
        cur = idx >= 0 and idx < len(self.latest_buttons) and bool(self.latest_buttons[idx])
        prev = idx >= 0 and idx < len(self.prev_buttons) and bool(self.prev_buttons[idx])
        return cur and not prev

    def _label_for(self, tag_id):
        meta = self.object_map.get(tag_id, self.object_map.get(str(tag_id), {}))
        if isinstance(meta, dict) and meta.get("grasp_complete_label"):
            return str(meta["grasp_complete_label"]).strip()
        return "apriltag_id_{}".format(tag_id)

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[apriltag_intent_grasp_bridge] %s", text)
        self.pub_status.publish(String(data=text))

    def _tick(self, _evt):
        if self.top_goal is None:
            self.pub_prompt.publish(String(data="Scan tags, then move toward a candidate grasp."))
            return

        poses = self.tag_poses.get(self.top_goal)
        if poses is None or poses.get("pregrasp") is None or poses.get("grasp") is None:
            self.pub_prompt.publish(String(data="Top tag {} has no grasp recorded yet.".format(self.top_goal)))
            return

        if self.top_prob >= self.select_threshold:
            self.pub_prompt.publish(
                String(
                    data="Top tag {} prob {:.2f}. Press X to execute grasp.".format(
                        self.top_goal,
                        self.top_prob,
                    )
                )
            )
            if self._pressed_edge(self.confirm_button_index):
                self.pub_pre.publish(copy.deepcopy(poses["pregrasp"]))
                self.pub_grasp.publish(copy.deepcopy(poses["grasp"]))
                self.pub_selected.publish(String(data=self._label_for(self.top_goal)))
                self._publish_status("loaded_grasp_for_tag={} prob={:.2f}".format(self.top_goal, self.top_prob))
        else:
            self.pub_prompt.publish(
                String(
                    data="Top tag {} prob {:.2f}. Move closer or align joystick.".format(
                        self.top_goal,
                        self.top_prob,
                    )
                )
            )

        if self._pressed_edge(self.cancel_button_index):
            self._publish_status("selection_cancelled")


if __name__ == "__main__":
    AprilTagIntentGraspBridge()
    rospy.spin()
