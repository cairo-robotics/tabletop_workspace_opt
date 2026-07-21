#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge AprilTag intent selection into the existing filtered executor."""

import copy
import json
import math
import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32, String


def _parse_int_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [int(v) for v in txt.split() if v]
    return [int(v) for v in default]


def _parse_string_set_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return {str(v).strip().lower() for v in raw if str(v).strip()}
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return {str(v).strip().lower() for v in txt.split() if str(v).strip()}
    return {str(v).strip().lower() for v in default if str(v).strip()}


def _parse_bool_param(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _consume_edge_state(latest_buttons):
    return list(latest_buttons)


def _quat_multiply(a, b):
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


class AprilTagIntentGraspBridge:
    def __init__(self):
        rospy.init_node("apriltag_intent_grasp_bridge")

        self.tag_ids = _parse_int_list_param("~tag_ids", [0, 1, 2])
        self.input_namespace_prefix = str(rospy.get_param("~input_namespace_prefix", "/apriltag_candidates/tag_")).strip()
        self.alternate_input_namespace_prefix = str(
            rospy.get_param("~alternate_input_namespace_prefix", "")
        ).strip()
        self.alternate_tag_ids = set(_parse_int_list_param("~alternate_tag_ids", []))
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
        self.destination_pose_yaml = os.path.expanduser(
            rospy.get_param(
                "~destination_pose_yaml",
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    "config",
                    "fixed_grasp_candidates_grouped.yaml",
                ),
            )
        )
        self.select_threshold = float(rospy.get_param("~select_threshold", 0.60))
        self.lego_select_threshold = float(rospy.get_param("~lego_select_threshold", self.select_threshold))
        self.destination_select_threshold = float(
            rospy.get_param("~destination_select_threshold", self.select_threshold)
        )
        self.auto_select_pregrasp = _parse_bool_param(rospy.get_param("~auto_select_pregrasp", False))
        self.auto_select_hold_sec = float(rospy.get_param("~auto_select_hold_sec", 0.50))
        self.auto_select_phases = _parse_string_set_param(
            "~auto_select_phases",
            ["select_sandwich_item", "select_breakfast_ingredient", "select_breakfast_milk"],
        )
        self.intent_select_mode = str(rospy.get_param("~intent_select_mode", "threshold")).strip().lower()
        self.joystick_event_axis_indices = _parse_int_list_param("~joystick_event_axis_indices", [0, 1])
        self.joystick_event_threshold = float(rospy.get_param("~joystick_event_threshold", 0.25))
        self.joystick_event_hold_sec = float(rospy.get_param("~joystick_event_hold_sec", 0.30))
        self.joystick_event_cooldown_sec = float(rospy.get_param("~joystick_event_cooldown_sec", 0.35))
        self.joystick_event_sample_delay_sec = float(rospy.get_param("~joystick_event_sample_delay_sec", 0.0))
        self.joystick_retarget_threshold = float(rospy.get_param("~joystick_retarget_threshold", 0.0))
        self.joystick_retarget_hold_sec = float(rospy.get_param("~joystick_retarget_hold_sec", 0.15))
        self.joystick_retarget_input_threshold = float(
            rospy.get_param("~joystick_retarget_input_threshold", 0.10)
        )
        self.joystick_event_min_probability = float(rospy.get_param("~joystick_event_min_probability", 0.20))
        self.joystick_start_on_release = _parse_bool_param(rospy.get_param("~joystick_start_on_release", True))
        self.joystick_release_threshold = float(rospy.get_param("~joystick_release_threshold", 0.08))
        self.joystick_recent_input_window_sec = float(rospy.get_param("~joystick_recent_input_window_sec", 1.25))
        self.joystick_stable_select_hold_sec = float(rospy.get_param("~joystick_stable_select_hold_sec", 0.12))
        self.joystick_retarget_on_release = _parse_bool_param(
            rospy.get_param("~joystick_retarget_on_release", False)
        )
        self.joystick_retarget_release_window_sec = float(
            rospy.get_param("~joystick_retarget_release_window_sec", 0.70)
        )
        self.selection_lock_start_timeout_sec = float(rospy.get_param("~selection_lock_start_timeout_sec", 1.0))
        self.task_phase_topic = str(rospy.get_param("~task_phase_topic", "/task_context/phase")).strip()
        self.confirm_button_index = int(rospy.get_param("~confirm_button_index", 2))
        self.cancel_button_index = int(rospy.get_param("~cancel_button_index", 3))
        self.pause_topic = str(rospy.get_param("~pause_topic", "/shared_autonomy/home_motion_active")).strip()
        self.study_event_topic = str(
            rospy.get_param("~study_event_topic", "/user_study/events")
        ).strip()
        self.selection_ready_topic = str(
            rospy.get_param("~selection_ready_topic", "/intent_inference/selection_ready")
        ).strip()
        self.execution_state_topic = str(
            rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")
        ).strip()
        self.snapshot_yaml = os.path.expanduser(str(rospy.get_param("~snapshot_yaml", "")).strip())
        self.load_snapshot_yaml = bool(rospy.get_param("~load_snapshot_yaml", False))
        self.destination_topdown_quat = self._parse_quat_param(
            "~destination_topdown_quat_xyzw",
            [0.0, 1.0, 0.0, 0.0],
        )
        self.force_topdown_categories = _parse_string_set_param(
            "~force_topdown_categories",
            [],
        )
        self.force_topdown_quat = self._parse_quat_param(
            "~force_topdown_quat_xyzw",
            self.destination_topdown_quat,
        )
        self.breakfast_grasp_orientation_yaml = os.path.expanduser(
            rospy.get_param(
                "~breakfast_grasp_orientation_yaml",
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    "config",
                    "fixed_grasp_candidates.yaml",
                ),
            )
        )
        self.breakfast_grasp_orientation_grasp_id = str(
            rospy.get_param("~breakfast_grasp_orientation_grasp_id", "carry_pose")
        ).strip()
        self.breakfast_grasp_orientation_stage = str(
            rospy.get_param("~breakfast_grasp_orientation_stage", "carry_pose")
        ).strip()
        self.breakfast_grasp_match_pour_orientation = bool(
            rospy.get_param("~breakfast_grasp_match_pour_orientation", False)
        )
        self.breakfast_grasp_roll_flip = bool(rospy.get_param("~breakfast_grasp_roll_flip", False))
        self.breakfast_milk_pose_lift_z = float(rospy.get_param("~breakfast_milk_pose_lift_z", 0.05))
        self.breakfast_milk_min_pregrasp_z = float(rospy.get_param("~breakfast_milk_min_pregrasp_z", 0.08))

        self.top_goal = None
        self.top_prob = 0.0
        self.current_phase = ""
        self.latest_axes = []
        self.prev_axes = []
        self.latest_buttons = []
        self.prev_buttons = []
        self.tag_poses = {}
        self.alt_tag_poses = {}
        self.object_map = self._load_object_map()
        self.destination_pose_library = self._load_destination_pose_library()
        self.breakfast_orientation_pose = self._load_breakfast_orientation_pose()
        self.snapshot_tag_poses = self._load_snapshot_tag_poses()
        self.last_status = ""
        self.paused = False
        self.execution_state = ""
        self.ready_top_goal = None
        self.ready_since = None
        self.last_auto_selected_goal = None
        self.last_published_selected_label = None
        self.selection_locked_tag_id = None
        self.selection_lock_seen_active_execution = False
        self.selection_lock_started_at = None
        self.pending_joystick_event_time = None
        self.last_joystick_select_time = None
        self.joystick_event_active_since = None
        self.joystick_event_triggered_for_hold = False
        self.retarget_top_goal = None
        self.retarget_since = None
        self.pending_start_tag_id = None
        self.pending_start_poses = None
        self.pending_start_trigger = None
        self.last_joystick_input_time = None
        self.joystick_input_rearmed = True
        self.joystick_stable_top_goal = None
        self.joystick_stable_since = None

        self.pub_pre = rospy.Publisher(self.output_pregrasp_topic, PoseStamped, queue_size=1, latch=True)
        self.pub_grasp = rospy.Publisher(self.output_grasp_topic, PoseStamped, queue_size=1, latch=True)
        self.pub_prompt = rospy.Publisher(self.prompt_topic, String, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.pub_selected = rospy.Publisher(self.selected_grasp_label_topic, String, queue_size=1, latch=True)
        self.pub_study_event = rospy.Publisher(self.study_event_topic, String, queue_size=20)
        self.pub_selection_ready = rospy.Publisher(self.selection_ready_topic, Bool, queue_size=1, latch=True)

        rospy.Subscriber(self.top_goal_topic, String, self._top_goal_cb, queue_size=1)
        rospy.Subscriber(self.top_probability_topic, Float32, self._top_prob_cb, queue_size=1)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        rospy.Subscriber(self.pause_topic, Bool, self._pause_cb, queue_size=1)
        rospy.Subscriber(self.task_phase_topic, String, self._phase_cb, queue_size=10)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=10)

        for tag_id in self.tag_ids:
            ns = "{}{}".format(self.input_namespace_prefix, tag_id)
            rospy.Subscriber("{}/pregrasp_pose".format(ns), PoseStamped, self._pre_cb, callback_args=tag_id, queue_size=1)
            rospy.Subscriber("{}/grasp_pose".format(ns), PoseStamped, self._grasp_cb, callback_args=tag_id, queue_size=1)
            if self.alternate_input_namespace_prefix and tag_id in self.alternate_tag_ids:
                alt_ns = "{}{}".format(self.alternate_input_namespace_prefix, tag_id)
                rospy.Subscriber("{}/pregrasp_pose".format(alt_ns), PoseStamped, self._alt_pre_cb, callback_args=tag_id, queue_size=1)
                rospy.Subscriber("{}/grasp_pose".format(alt_ns), PoseStamped, self._alt_grasp_cb, callback_args=tag_id, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self._tick)
        self.pub_selection_ready.publish(Bool(data=False))
        self._publish_status("waiting_for_intent_selection")

    def _publish_selected_label(self, label):
        label = str(label).strip()
        if self.last_published_selected_label == label:
            return
        self.last_published_selected_label = label
        self.pub_selected.publish(String(data=label))

    def _clear_selection_lock(self):
        self.selection_locked_tag_id = None
        self.selection_lock_seen_active_execution = False
        self.selection_lock_started_at = None

    @staticmethod
    def _parse_quat_param(name, default):
        raw = rospy.get_param(name, default)
        if isinstance(raw, (list, tuple)):
            values = [float(v) for v in raw]
        elif isinstance(raw, str):
            txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
            values = [float(v) for v in txt.split() if v]
        else:
            values = [float(raw)]
        if len(values) != 4:
            values = list(default)
        norm = math.sqrt(sum(v * v for v in values))
        if norm < 1e-9:
            return [float(v) for v in default]
        return [float(v) / norm for v in values]

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}
        if isinstance(data.get("tag_objects"), dict):
            return data.get("tag_objects", {}) or {}
        if isinstance(data.get("candidate_objects"), dict):
            return data.get("candidate_objects", {}) or {}
        return {}

    @staticmethod
    def _dict_to_pose_stamped(data):
        msg = PoseStamped()
        position = list(data.get("position", []))
        orientation = list(data.get("orientation", []))
        if len(position) != 3 or len(orientation) != 4:
            raise ValueError("pose entry must contain position[3] and orientation[4]")
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])
        return msg

    def _load_snapshot_tag_poses(self):
        if not self.load_snapshot_yaml or not self.snapshot_yaml or not os.path.exists(self.snapshot_yaml):
            return {}
        with open(self.snapshot_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        entries = raw.get("tag_grasps", {}) if isinstance(raw, dict) else {}
        parsed = {}
        for key, pose_dict in entries.items():
            try:
                tag_id = int(key)
                pair = {"pregrasp": None, "grasp": None}
                if isinstance(pose_dict, dict) and ("pregrasp_pose" in pose_dict or "grasp_pose" in pose_dict):
                    if isinstance(pose_dict.get("pregrasp_pose"), dict):
                        pair["pregrasp"] = self._dict_to_pose_stamped(pose_dict.get("pregrasp_pose"))
                    if isinstance(pose_dict.get("grasp_pose"), dict):
                        pair["grasp"] = self._dict_to_pose_stamped(pose_dict.get("grasp_pose"))
                if pair["pregrasp"] is not None and pair["grasp"] is not None:
                    parsed[tag_id] = pair
            except Exception as exc:
                rospy.logwarn(
                    "[apriltag_intent_grasp_bridge] skipping invalid snapshot entry %s: %s",
                    key,
                    exc,
                )
        if parsed:
            rospy.loginfo(
                "[apriltag_intent_grasp_bridge] loaded snapshot poses for tags %s from %s",
                sorted(parsed.keys()),
                self.snapshot_yaml,
            )
        return parsed

    def _load_destination_pose_library(self):
        if not os.path.exists(self.destination_pose_yaml):
            return {}
        with open(self.destination_pose_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        grasps = raw.get("grasps", []) if isinstance(raw, dict) else []
        library = {}
        for grasp in grasps:
            if not isinstance(grasp, dict):
                continue
            grasp_id = str(grasp.get("grasp_id", "")).strip()
            if grasp_id:
                library[grasp_id] = grasp
        return library

    def _load_breakfast_orientation_pose(self):
        if not self.breakfast_grasp_orientation_yaml or not os.path.exists(self.breakfast_grasp_orientation_yaml):
            return None
        with open(self.breakfast_grasp_orientation_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        grasps = raw.get("grasps", []) if isinstance(raw, dict) else []
        for grasp in grasps:
            if not isinstance(grasp, dict):
                continue
            grasp_id = str(grasp.get("grasp_id", "")).strip()
            if grasp_id != self.breakfast_grasp_orientation_grasp_id:
                continue
            pose_dict = grasp.get(self.breakfast_grasp_orientation_stage)
            if not isinstance(pose_dict, dict):
                return None
            pose_msg = self._dict_to_pose_stamped(pose_dict)
            pose_msg.header.frame_id = str(grasp.get("frame_id", "base")).strip() or "base"
            return pose_msg
        return None

    def _entry(self, tag_id):
        if tag_id not in self.tag_poses:
            self.tag_poses[tag_id] = {"pregrasp": None, "grasp": None}
        return self.tag_poses[tag_id]

    def _pre_cb(self, msg, tag_id):
        self._entry(tag_id)["pregrasp"] = copy.deepcopy(msg)

    def _grasp_cb(self, msg, tag_id):
        self._entry(tag_id)["grasp"] = copy.deepcopy(msg)

    def _alt_entry(self, tag_id):
        if tag_id not in self.alt_tag_poses:
            self.alt_tag_poses[tag_id] = {"pregrasp": None, "grasp": None}
        return self.alt_tag_poses[tag_id]

    def _alt_pre_cb(self, msg, tag_id):
        self._alt_entry(tag_id)["pregrasp"] = copy.deepcopy(msg)

    def _alt_grasp_cb(self, msg, tag_id):
        self._alt_entry(tag_id)["grasp"] = copy.deepcopy(msg)

    def _top_goal_cb(self, msg):
        txt = str(msg.data).strip()
        next_goal = int(txt) if txt and txt.lstrip("-").isdigit() else None
        if next_goal != self.top_goal and self.selection_locked_tag_id is None:
            self.ready_top_goal = None
            self.ready_since = None
        self.top_goal = next_goal

    def _top_prob_cb(self, msg):
        self.top_prob = float(msg.data)

    def _joy_cb(self, msg):
        self.prev_axes = list(self.latest_axes)
        self.latest_axes = list(msg.axes)
        self.prev_buttons = list(self.latest_buttons)
        self.latest_buttons = list(msg.buttons)

    def _pause_cb(self, msg):
        self.paused = bool(msg.data)
        if self.paused:
            self._publish_status("paused_for_home_motion")

    def _phase_cb(self, msg):
        next_phase = str(msg.data).strip().lower()
        if next_phase != self.current_phase:
            self._clear_selection_lock()
            self.ready_top_goal = None
            self.ready_since = None
            self.last_auto_selected_goal = None
            self._clear_joystick_event(reset_hold=True)
        self.current_phase = next_phase
        if not self._selection_enabled_for_phase():
            self.ready_top_goal = None
            self.ready_since = None
            self.last_auto_selected_goal = None
            self._clear_joystick_event(reset_hold=True)
            self._clear_selection_lock()
            self.pub_selection_ready.publish(Bool(data=False))
            self._publish_selected_label("")
            self._publish_status("selection_disabled_phase={}".format(self.current_phase or "none"))

    def _execution_state_cb(self, msg):
        was_unlocked = self._selection_unlocked()
        self.execution_state = str(msg.data).strip().lower()
        is_unlocked = self._selection_unlocked()
        if self.selection_locked_tag_id is not None and not is_unlocked:
            self.selection_lock_seen_active_execution = True
        if is_unlocked and not was_unlocked:
            self.last_auto_selected_goal = None
            self.ready_top_goal = None
            self.ready_since = None
            self.last_joystick_input_time = None
            self.joystick_input_rearmed = False
            self._clear_joystick_event(reset_hold=True)
            self._clear_pending_start()
            if self.selection_lock_seen_active_execution:
                self._clear_selection_lock()

    def _pressed_edge(self, idx):
        cur = idx >= 0 and idx < len(self.latest_buttons) and bool(self.latest_buttons[idx])
        prev = idx >= 0 and idx < len(self.prev_buttons) and bool(self.prev_buttons[idx])
        return cur and not prev

    def _label_for(self, tag_id):
        meta = self.object_map.get(tag_id, self.object_map.get(str(tag_id), {}))
        if isinstance(meta, dict) and meta.get("grasp_complete_label"):
            return str(meta["grasp_complete_label"]).strip()
        return "apriltag_id_{}".format(tag_id)

    def _meta_for(self, tag_id):
        meta = self.object_map.get(tag_id, self.object_map.get(str(tag_id), {}))
        return meta if isinstance(meta, dict) else {}

    def _is_destination_tag(self, tag_id):
        meta = self._meta_for(tag_id)
        if str(meta.get("destination_group", "")).strip() == "sorting_target":
            return True
        return str(meta.get("category", "")).strip() == "destination"

    def _destination_named_pose_pair(self, tag_id):
        meta = self._meta_for(tag_id)
        grasp_id = str(meta.get("destination_pose_grasp_id", "")).strip()
        stage = str(meta.get("destination_pose_stage", "")).strip()
        if not grasp_id or not stage:
            return None
        grasp_entry = self.destination_pose_library.get(grasp_id, {})
        pose_dict = grasp_entry.get(stage)
        if not isinstance(pose_dict, dict):
            return None
        pose_msg = self._dict_to_pose_stamped(pose_dict)
        pose_msg.header.frame_id = str(grasp_entry.get("frame_id", "base")).strip() or "base"
        return {
            "pregrasp": copy.deepcopy(pose_msg),
            "grasp": copy.deepcopy(pose_msg),
        }

    def _apply_destination_topdown_orientation(self, pose_stamped):
        if pose_stamped is None:
            return None
        out = copy.deepcopy(pose_stamped)
        out.pose.orientation.x = float(self.destination_topdown_quat[0])
        out.pose.orientation.y = float(self.destination_topdown_quat[1])
        out.pose.orientation.z = float(self.destination_topdown_quat[2])
        out.pose.orientation.w = float(self.destination_topdown_quat[3])
        return out

    def _apply_forced_topdown_orientation(self, pose_stamped):
        if pose_stamped is None:
            return None
        out = copy.deepcopy(pose_stamped)
        out.pose.orientation.x = float(self.force_topdown_quat[0])
        out.pose.orientation.y = float(self.force_topdown_quat[1])
        out.pose.orientation.z = float(self.force_topdown_quat[2])
        out.pose.orientation.w = float(self.force_topdown_quat[3])
        return out

    def _apply_orientation_template(self, pose_stamped, template_pose):
        if pose_stamped is None or template_pose is None:
            return pose_stamped
        out = copy.deepcopy(pose_stamped)
        out.pose.orientation = copy.deepcopy(template_pose.pose.orientation)
        return out

    def _lift_pose_z(self, pose_stamped, lift_z, min_z=None):
        if pose_stamped is None:
            return None
        out = copy.deepcopy(pose_stamped)
        out.pose.position.z = float(out.pose.position.z) + float(lift_z)
        if min_z is not None:
            out.pose.position.z = max(float(out.pose.position.z), float(min_z))
        return out

    def _is_breakfast_phase(self):
        return self.current_phase in (
            "select_breakfast_ingredient",
            "breakfast_ingredient",
            "select_breakfast_milk",
            "breakfast_milk",
        )

    def _prepare_selected_poses(self, tag_id, poses):
        if poses is None:
            return poses
        prepared = {
            "pregrasp": copy.deepcopy(poses.get("pregrasp")),
            "grasp": copy.deepcopy(poses.get("grasp")),
        }
        meta = self._meta_for(tag_id)
        category = str(meta.get("category", "")).strip().lower() if isinstance(meta, dict) else ""
        if category and category in self.force_topdown_categories:
            prepared["pregrasp"] = self._apply_forced_topdown_orientation(prepared["pregrasp"])
            prepared["grasp"] = self._apply_forced_topdown_orientation(prepared["grasp"])
        elif self._is_destination_tag(tag_id):
            prepared["pregrasp"] = self._apply_destination_topdown_orientation(prepared["pregrasp"])
            prepared["grasp"] = self._apply_destination_topdown_orientation(prepared["grasp"])
        elif self._is_breakfast_phase() and (self.breakfast_grasp_match_pour_orientation or self.breakfast_grasp_roll_flip):
            if self.breakfast_grasp_match_pour_orientation or self.breakfast_grasp_roll_flip:
                prepared["pregrasp"] = self._apply_local_roll_flip(prepared["pregrasp"])
                prepared["grasp"] = self._apply_local_roll_flip(prepared["grasp"])
        if self._is_breakfast_phase() and category == "milk" and self.breakfast_milk_pose_lift_z != 0.0:
            prepared["pregrasp"] = self._lift_pose_z(
                prepared["pregrasp"],
                self.breakfast_milk_pose_lift_z,
                self.breakfast_milk_min_pregrasp_z,
            )
        return prepared

    def _apply_local_roll_flip(self, pose_stamped):
        if pose_stamped is None:
            return None
        out = copy.deepcopy(pose_stamped)
        current = [
            out.pose.orientation.x,
            out.pose.orientation.y,
            out.pose.orientation.z,
            out.pose.orientation.w,
        ]
        flipped = _quat_multiply(current, [0.0, 0.0, 1.0, 0.0])
        out.pose.orientation.x = float(flipped[0])
        out.pose.orientation.y = float(flipped[1])
        out.pose.orientation.z = float(flipped[2])
        out.pose.orientation.w = float(flipped[3])
        return out

    def _active_select_threshold(self):
        if self.current_phase in ("select_lego_brick", "lego_brick"):
            return self.lego_select_threshold
        if self.current_phase in ("select_sort_destination", "sort_destination", "select_lego_destination"):
            return self.destination_select_threshold
        return self.select_threshold

    def _selection_enabled_for_phase(self):
        phase = str(self.current_phase).strip().lower()
        if not phase:
            return False
        return phase not in ("scan_workspace", "scan", "all", "select_all")

    def _joystick_event_mode(self):
        return self.intent_select_mode in ("joystick_event", "event", "input_event")

    def _auto_select_enabled_for_phase(self):
        phase = str(self.current_phase).strip().lower()
        return bool(self.auto_select_pregrasp and phase and phase in self.auto_select_phases)

    def _joystick_event_enabled_for_phase(self):
        phase = str(self.current_phase).strip().lower()
        return bool(self._joystick_event_mode() and phase and phase in self.auto_select_phases)

    def _selection_unlocked(self):
        state = str(self.execution_state).strip().lower()
        if not state:
            return True
        if state in ("idle", "done", "wait_pregrasp_confirm"):
            return True
        if state.startswith("grasp_complete") or state.startswith("release_complete"):
            return True
        return False

    def _update_ready_hold(self, selection_ready):
        if not selection_ready or self.top_goal is None:
            self.ready_top_goal = None
            self.ready_since = None
            return False
        now = rospy.Time.now()
        if self.ready_top_goal != self.top_goal:
            self.ready_top_goal = self.top_goal
            self.ready_since = now
            return self.auto_select_hold_sec <= 0.0
        if self.ready_since is None:
            self.ready_since = now
            return self.auto_select_hold_sec <= 0.0
        return (now - self.ready_since).to_sec() >= self.auto_select_hold_sec

    def _axis_norm(self, axes):
        total = 0.0
        for idx in self.joystick_event_axis_indices:
            if idx < 0 or idx >= len(axes):
                continue
            total += float(axes[idx]) * float(axes[idx])
        return math.sqrt(total)

    def _update_joystick_event_hold(self, now):
        axis_norm = self._axis_norm(self.latest_axes)
        if not self.joystick_input_rearmed:
            if axis_norm <= self.joystick_release_threshold:
                self.joystick_input_rearmed = True
            else:
                self.joystick_event_active_since = None
                self.joystick_event_triggered_for_hold = False
                return
        if axis_norm < self.joystick_event_threshold:
            self.joystick_event_active_since = None
            self.joystick_event_triggered_for_hold = False
            return
        self.last_joystick_input_time = now
        if self.joystick_event_active_since is None:
            self.joystick_event_active_since = now
            self.joystick_event_triggered_for_hold = False
            return
        held_sec = (now - self.joystick_event_active_since).to_sec()
        if held_sec < self.joystick_event_hold_sec:
            return
        if self.joystick_event_triggered_for_hold:
            return
        if self.pending_joystick_event_time is not None:
            return
        if not self._joystick_event_cooldown_elapsed(now):
            return
        self.pending_joystick_event_time = now
        self.joystick_event_triggered_for_hold = True

    def _joystick_event_cooldown_elapsed(self, now):
        if self.last_joystick_select_time is None:
            return True
        return (now - self.last_joystick_select_time).to_sec() >= self.joystick_event_cooldown_sec

    def _joystick_event_ready(self):
        if self.pending_joystick_event_time is None:
            return False
        return (rospy.Time.now() - self.pending_joystick_event_time).to_sec() >= self.joystick_event_sample_delay_sec

    def _clear_joystick_event(self, reset_hold=False):
        self.pending_joystick_event_time = None
        if reset_hold:
            self.joystick_event_active_since = None
            self.joystick_event_triggered_for_hold = False
            self.retarget_top_goal = None
            self.retarget_since = None
            self.joystick_active_for_retarget = False
            self.last_joystick_release_time = None
            self.joystick_stable_top_goal = None
            self.joystick_stable_since = None

    def _update_recent_joystick_input(self, now):
        axis_norm = self._axis_norm(self.latest_axes)
        if not self.joystick_input_rearmed:
            if axis_norm <= self.joystick_release_threshold:
                self.joystick_input_rearmed = True
            else:
                return
        if axis_norm >= self.joystick_retarget_input_threshold:
            self.last_joystick_input_time = now
            self.joystick_active_for_retarget = True
        elif self.joystick_active_for_retarget and axis_norm <= self.joystick_release_threshold:
            self.last_joystick_release_time = now
            self.joystick_active_for_retarget = False

    def _recent_joystick_input(self, now):
        if not self.joystick_input_rearmed or self.last_joystick_input_time is None:
            return False
        return (now - self.last_joystick_input_time).to_sec() <= self.joystick_recent_input_window_sec

    def _recent_joystick_release(self, now):
        if not self.joystick_input_rearmed or self.last_joystick_release_time is None:
            return False
        return (now - self.last_joystick_release_time).to_sec() <= self.joystick_retarget_release_window_sec

    def _joystick_event_probability_ready(self):
        return self.joystick_event_min_probability <= 0.0 or self.top_prob >= self.joystick_event_min_probability

    def _joystick_stable_top_ready(self, now, hold_sec=None):
        if hold_sec is None:
            hold_sec = self.joystick_stable_select_hold_sec
        if self.top_goal is None or not self._recent_joystick_input(now) or not self._joystick_event_probability_ready():
            self.joystick_stable_top_goal = None
            self.joystick_stable_since = None
            return False
        if self.joystick_stable_top_goal != self.top_goal:
            self.joystick_stable_top_goal = self.top_goal
            self.joystick_stable_since = now
            return hold_sec <= 0.0
        if self.joystick_stable_since is None:
            self.joystick_stable_since = now
            return hold_sec <= 0.0
        return (now - self.joystick_stable_since).to_sec() >= hold_sec

    def _clear_pending_start(self):
        self.pending_start_tag_id = None
        self.pending_start_poses = None
        self.pending_start_trigger = None

    def _joystick_released_for_start(self):
        return self._axis_norm(self.latest_axes) <= self.joystick_release_threshold

    def _start_or_queue_selected_target(self, tag_id, poses, trigger):
        if not self.joystick_start_on_release or trigger != "joystick_event":
            self._load_selected_target(tag_id, poses, trigger)
            return True
        self.pending_start_tag_id = int(tag_id)
        self.pending_start_poses = copy.deepcopy(poses)
        self.pending_start_trigger = trigger
        self.last_joystick_select_time = rospy.Time.now()
        self._clear_joystick_event(reset_hold=True)
        self._publish_status(
            "pending_{}_tag={} release_joystick_to_start".format(trigger, tag_id)
        )
        self._publish_study_event(
            "joystick_pending_select",
            grasp_id=self._label_for(tag_id),
            stage="selection",
            tag_id=tag_id,
            probability=self.top_prob,
        )
        return False

    def _maybe_start_pending_target(self):
        if self.pending_start_tag_id is None or self.pending_start_poses is None:
            return False
        if not self._joystick_released_for_start():
            return False
        tag_id = self.pending_start_tag_id
        poses = self.pending_start_poses
        trigger = self.pending_start_trigger or "joystick_event"
        self._clear_pending_start()
        self._publish_study_event(
            "joystick_release_start",
            grasp_id=self._label_for(tag_id),
            stage="selection",
            tag_id=tag_id,
            probability=self.top_prob,
        )
        self._load_selected_target(tag_id, poses, trigger)
        return True

    def _reselection_allowed_for_state(self):
        state = str(self.execution_state).strip().lower()
        return state in ("", "idle", "wait_pregrasp_confirm")

    def _retarget_allowed_for_state(self):
        state = str(self.execution_state).strip().lower()
        return state == "exec_pregrasp"

    def _retarget_ready(self, locked_tag_id):
        now = rospy.Time.now()
        if not self._joystick_event_enabled_for_phase() or not self._retarget_allowed_for_state():
            self.retarget_top_goal = None
            self.retarget_since = None
            return False
        if self.top_goal is None or int(self.top_goal) == int(locked_tag_id):
            self.retarget_top_goal = None
            self.retarget_since = None
            return False
        if self.joystick_retarget_on_release:
            has_retarget_input = self._recent_joystick_release(now)
        else:
            has_retarget_input = self._recent_joystick_input(now)
        if not has_retarget_input:
            self.retarget_top_goal = None
            self.retarget_since = None
            return False
        if not self._joystick_event_probability_ready():
            self.retarget_top_goal = None
            self.retarget_since = None
            return False
        if self.joystick_retarget_threshold > 0.0 and self.top_prob < self.joystick_retarget_threshold:
            self.retarget_top_goal = None
            self.retarget_since = None
            return False
        if self.retarget_top_goal != self.top_goal:
            self.retarget_top_goal = self.top_goal
            self.retarget_since = now
            return self.joystick_retarget_hold_sec <= 0.0
        if self.retarget_since is None:
            self.retarget_since = now
            return self.joystick_retarget_hold_sec <= 0.0
        return (now - self.retarget_since).to_sec() >= self.joystick_retarget_hold_sec

    def _load_selected_target(self, tag_id, poses, trigger):
        self.selection_locked_tag_id = int(tag_id)
        self.selection_lock_seen_active_execution = False
        self.selection_lock_started_at = rospy.Time.now()
        self.pub_pre.publish(copy.deepcopy(poses["pregrasp"]))
        self.pub_grasp.publish(copy.deepcopy(poses["grasp"]))
        self._publish_selected_label(self._label_for(tag_id))
        self._publish_status(
            "loaded_grasp_for_tag={} prob={:.2f} threshold={:.2f} phase={} trigger={}".format(
                tag_id,
                self.top_prob,
                self._active_select_threshold(),
                self.current_phase or "unknown",
                trigger,
            )
        )
        self._publish_study_event(
            "auto_select_pregrasp"
            if trigger == "auto"
            else (
                "joystick_event_select"
                if trigger == "joystick_event"
                else ("joystick_retarget_select" if trigger == "joystick_retarget" else "confirm_accept")
            ),
            grasp_id=self._label_for(tag_id),
            stage="selection",
            tag_id=tag_id,
            probability=self.top_prob,
        )
        if trigger in ("joystick_event", "joystick_retarget"):
            self.last_joystick_select_time = rospy.Time.now()
            self._clear_joystick_event(reset_hold=True)
            self._clear_pending_start()
        self.prev_buttons = _consume_edge_state(self.latest_buttons)

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[apriltag_intent_grasp_bridge] %s", text)
        self.pub_status.publish(String(data=text))

    def _publish_study_event(self, event_type, **fields):
        payload = {
            "event": str(event_type),
            "node": rospy.get_name(),
            "stamp": rospy.Time.now().to_sec(),
        }
        for key, value in fields.items():
            if value is None:
                continue
            payload[str(key)] = value
        self.pub_study_event.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _resolve_poses(self, tag_id):
        if self._is_destination_tag(tag_id):
            named_pair = self._destination_named_pose_pair(tag_id)
            if named_pair is not None:
                return named_pair
        live = self.tag_poses.get(tag_id)
        if self.current_phase in ("select_lego_brick", "lego_brick"):
            if live is not None and live.get("pregrasp") is not None and live.get("grasp") is not None:
                return live
        if tag_id in self.alternate_tag_ids:
            alt = self.alt_tag_poses.get(tag_id)
            if alt is not None and alt.get("pregrasp") is not None and alt.get("grasp") is not None:
                return alt
            if live is not None and live.get("pregrasp") is not None and live.get("grasp") is not None:
                return live
            snap = self.snapshot_tag_poses.get(tag_id)
            if snap is not None and snap.get("pregrasp") is not None and snap.get("grasp") is not None:
                return snap
            return alt if alt is not None else live
        if live is not None and live.get("pregrasp") is not None and live.get("grasp") is not None:
            return live
        snap = self.snapshot_tag_poses.get(tag_id)
        if snap is not None and snap.get("pregrasp") is not None and snap.get("grasp") is not None:
            return snap
        return live

    def _tick(self, _evt):
        now = rospy.Time.now()
        self._update_recent_joystick_input(now)

        if self.paused:
            self.pub_selection_ready.publish(Bool(data=False))
            self.pub_prompt.publish(String(data="Home motion active. Intent selection paused."))
            return

        if not self._selection_enabled_for_phase():
            self.pub_selection_ready.publish(Bool(data=False))
            self.ready_top_goal = None
            self.ready_since = None
            self.last_auto_selected_goal = None
            self._clear_joystick_event(reset_hold=True)
            self._clear_pending_start()
            self._clear_selection_lock()
            self._publish_selected_label("")
            self.pub_prompt.publish(String(data="Scanning workspace. Target selection and auto pregrasp are disabled."))
            return

        if self.pending_start_tag_id is not None:
            self.pub_selection_ready.publish(Bool(data=False))
            if self._joystick_stable_top_ready(now) and self.top_goal is not None:
                poses = self._prepare_selected_poses(self.top_goal, self._resolve_poses(self.top_goal))
                if poses is not None and poses.get("pregrasp") is not None and poses.get("grasp") is not None:
                    if int(self.top_goal) != int(self.pending_start_tag_id):
                        self.pending_start_tag_id = int(self.top_goal)
                        self.pending_start_poses = copy.deepcopy(poses)
                        self._publish_status(
                            "pending_target_updated tag={} release_joystick_to_start".format(self.top_goal)
                        )
            if self._maybe_start_pending_target():
                self.last_auto_selected_goal = None
                return
            self.pub_prompt.publish(
                String(
                    data="Target tag {} selected. Release joystick to start autonomous pregrasp.".format(
                        self.pending_start_tag_id
                    )
                )
            )
            if self._pressed_edge(self.cancel_button_index):
                self._publish_status("pending_selection_cancelled")
                self._clear_pending_start()
                self._clear_joystick_event(reset_hold=True)
                self.prev_buttons = _consume_edge_state(self.latest_buttons)
            return

        if self.selection_locked_tag_id is not None:
            locked_tag_id = self.selection_locked_tag_id
            if (
                not self.selection_lock_seen_active_execution
                and self._selection_unlocked()
                and self.selection_lock_started_at is not None
                and (rospy.Time.now() - self.selection_lock_started_at).to_sec()
                >= self.selection_lock_start_timeout_sec
            ):
                self._publish_status("selection_lock_expired_no_executor_start")
                self._clear_selection_lock()
                self.last_auto_selected_goal = None
                self._clear_joystick_event(reset_hold=True)
                self._clear_pending_start()
                self._publish_selected_label("")
                self.pub_selection_ready.publish(Bool(data=False))
                self.pub_prompt.publish(String(data="Selection did not start. Move joystick clearly toward a target again."))
                return
            self.pub_selection_ready.publish(Bool(data=False))
            self.ready_top_goal = None
            self.ready_since = None
            if self._retarget_ready(locked_tag_id):
                poses = self._prepare_selected_poses(self.top_goal, self._resolve_poses(self.top_goal))
                if poses is not None and poses.get("pregrasp") is not None and poses.get("grasp") is not None:
                    self._publish_status(
                        "joystick_retarget tag {} -> {} prob {:.2f}".format(
                            locked_tag_id,
                            self.top_goal,
                            self.top_prob,
                        )
                    )
                    self._start_or_queue_selected_target(self.top_goal, poses, "joystick_retarget")
                    self.last_auto_selected_goal = None
                    return
            self.pub_prompt.publish(
                String(
                    data=(
                        "Target tag {} locked. Retargeting is allowed during approach. "
                        "At suggested pregrasp, nudge joystick to retarget automatically or press X to accept; after X, press Y to cancel/reselect."
                    ).format(
                        locked_tag_id,
                    )
                )
            )
            state = str(self.execution_state).strip().lower()
            cancel_allowed = state != "exec_pregrasp"
            if cancel_allowed and self._pressed_edge(self.cancel_button_index):
                cancelled_tag_id = int(locked_tag_id)
                self._publish_status("selection_cancelled")
                self._clear_selection_lock()
                self.last_auto_selected_goal = None
                self._clear_joystick_event(reset_hold=True)
                self._clear_pending_start()
                self._publish_selected_label("")
                self._publish_study_event(
                    "confirm_cancel",
                    grasp_id=self._label_for(cancelled_tag_id),
                    stage="selection",
                )
                self.prev_buttons = _consume_edge_state(self.latest_buttons)
            return

        if self._joystick_event_mode():
            self._update_joystick_event_hold(now)

        if self.top_goal is None:
            self.pub_selection_ready.publish(Bool(data=False))
            self.pub_prompt.publish(String(data="Scan tags, then move toward a candidate grasp."))
            return

        poses = self._prepare_selected_poses(self.top_goal, self._resolve_poses(self.top_goal))
        if poses is None or poses.get("pregrasp") is None or poses.get("grasp") is None:
            self.pub_selection_ready.publish(Bool(data=False))
            if self.load_snapshot_yaml:
                self.pub_prompt.publish(
                    String(
                        data="Top tag {} is missing a saved pregrasp/grasp pair. Re-scan and save this scene first.".format(
                            self.top_goal
                        )
                    )
                )
            else:
                self.pub_prompt.publish(String(data="Top tag {} has no grasp recorded yet.".format(self.top_goal)))
            return

        active_threshold = self._active_select_threshold()
        selection_ready = bool(self.top_prob >= active_threshold)
        selection_unlocked = self._selection_unlocked() and self.selection_locked_tag_id is None
        auto_select_active = self._auto_select_enabled_for_phase()
        joystick_event_active = self._joystick_event_enabled_for_phase()
        joystick_event_ready = self._joystick_event_ready() and self._joystick_event_probability_ready()
        joystick_stable_ready = self._joystick_stable_top_ready(now) if joystick_event_active else False
        self.pub_selection_ready.publish(
            Bool(data=bool(selection_ready if not joystick_event_active else (joystick_event_ready or joystick_stable_ready)))
        )
        ready_held = self._update_ready_hold(selection_ready and selection_unlocked and auto_select_active)
        if joystick_event_active:
            if (joystick_event_ready or joystick_stable_ready) and selection_unlocked:
                self.pub_prompt.publish(
                    String(
                        data="Intent stable. Selected top tag {} prob {:.2f}; release joystick to start.".format(
                            self.top_goal,
                            self.top_prob,
                        )
                    )
                )
                self._start_or_queue_selected_target(self.top_goal, poses, "joystick_event")
                self.last_auto_selected_goal = None
            else:
                self.pub_prompt.publish(
                    String(
                        data="Nudge joystick toward a target; stable top tag will start. Current top tag {} prob {:.2f}.".format(
                            self.top_goal,
                            self.top_prob,
                        )
                    )
                )
        elif selection_ready:
            if auto_select_active:
                action_text = "Auto-moving to pregrasp; press Y to cancel." if ready_held else "Hold intent steady for auto pregrasp."
            else:
                action_text = "Press X to move above the selected container." if self.current_phase in (
                    "select_sort_destination",
                    "sort_destination",
                ) else "Press X to execute grasp."
            self.pub_prompt.publish(
                String(
                    data=(
                        "Top tag {} prob {:.2f} (threshold {:.2f}). {}".format(
                            self.top_goal,
                            self.top_prob,
                            active_threshold,
                            action_text
                            if selection_unlocked
                            else "Target locked. Finish or cancel the active execution before choosing a new target.",
                        )
                    )
                )
            )
            if (
                auto_select_active
                and selection_unlocked
                and ready_held
                and self.last_auto_selected_goal != self.top_goal
            ):
                self._load_selected_target(self.top_goal, poses, "auto")
                self.last_auto_selected_goal = self.top_goal
            elif selection_unlocked and self._pressed_edge(self.confirm_button_index):
                self._load_selected_target(self.top_goal, poses, "x")
                self.last_auto_selected_goal = None
        else:
            self.last_auto_selected_goal = None
            self.pub_prompt.publish(
                String(
                    data="Top tag {} prob {:.2f} below threshold {:.2f}. Move closer or align joystick.".format(
                        self.top_goal,
                        self.top_prob,
                        active_threshold,
                    )
                )
            )

        if self._pressed_edge(self.cancel_button_index):
            self._publish_status("selection_cancelled")
            self.ready_top_goal = None
            self.ready_since = None
            self.last_auto_selected_goal = None
            self._clear_joystick_event(reset_hold=True)
            self._clear_selection_lock()
            self._publish_selected_label("")
            self.pub_selection_ready.publish(Bool(data=False))
            self.pub_prompt.publish(String(data="Selection cancelled. Move toward a target again."))
            self._publish_study_event(
                "confirm_cancel",
                grasp_id="" if self.top_goal is None else self._label_for(self.top_goal),
                stage="selection",
            )
            self.prev_buttons = _consume_edge_state(self.latest_buttons)


if __name__ == "__main__":
    AprilTagIntentGraspBridge()
    rospy.spin()
