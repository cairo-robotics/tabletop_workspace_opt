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

        self.top_goal = None
        self.top_prob = 0.0
        self.current_phase = ""
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
        self.top_goal = int(txt) if txt and txt.lstrip("-").isdigit() else None

    def _top_prob_cb(self, msg):
        self.top_prob = float(msg.data)

    def _joy_cb(self, msg):
        self.prev_buttons = list(self.latest_buttons)
        self.latest_buttons = list(msg.buttons)

    def _pause_cb(self, msg):
        self.paused = bool(msg.data)
        if self.paused:
            self._publish_status("paused_for_home_motion")

    def _phase_cb(self, msg):
        self.current_phase = str(msg.data).strip().lower()

    def _execution_state_cb(self, msg):
        self.execution_state = str(msg.data).strip().lower()

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

    def _apply_orientation_template(self, pose_stamped, template_pose):
        if pose_stamped is None or template_pose is None:
            return pose_stamped
        out = copy.deepcopy(pose_stamped)
        out.pose.orientation = copy.deepcopy(template_pose.pose.orientation)
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
        if self._is_destination_tag(tag_id):
            prepared["pregrasp"] = self._apply_destination_topdown_orientation(prepared["pregrasp"])
            prepared["grasp"] = self._apply_destination_topdown_orientation(prepared["grasp"])
        elif self._is_breakfast_phase() and (self.breakfast_grasp_match_pour_orientation or self.breakfast_grasp_roll_flip):
            if self.breakfast_grasp_match_pour_orientation or self.breakfast_grasp_roll_flip:
                prepared["pregrasp"] = self._apply_local_roll_flip(prepared["pregrasp"])
                prepared["grasp"] = self._apply_local_roll_flip(prepared["grasp"])
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

    def _selection_unlocked(self):
        state = str(self.execution_state).strip().lower()
        if not state:
            return True
        if state in ("idle", "done", "wait_pregrasp_confirm"):
            return True
        if state.startswith("grasp_complete") or state.startswith("release_complete"):
            return True
        return False

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
        if self.paused:
            self.pub_selection_ready.publish(Bool(data=False))
            self.pub_prompt.publish(String(data="Home motion active. Intent selection paused."))
            return

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
        selection_unlocked = self._selection_unlocked()
        self.pub_selection_ready.publish(Bool(data=selection_ready))
        if selection_ready:
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
            if selection_unlocked and self._pressed_edge(self.confirm_button_index):
                self.pub_pre.publish(copy.deepcopy(poses["pregrasp"]))
                self.pub_grasp.publish(copy.deepcopy(poses["grasp"]))
                self.pub_selected.publish(String(data=self._label_for(self.top_goal)))
                self._publish_status(
                    "loaded_grasp_for_tag={} prob={:.2f} threshold={:.2f} phase={}".format(
                        self.top_goal,
                        self.top_prob,
                        active_threshold,
                        self.current_phase or "unknown",
                    )
                )
                self.prev_buttons = _consume_edge_state(self.latest_buttons)
        else:
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
            self.pub_selected.publish(String(data=""))
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
