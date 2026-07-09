#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record and retain AprilTag grasp candidates discovered during scanning."""

import copy
import math
import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def _quat_angle_deg(a, b):
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    dot = max(0.0, min(1.0, float(dot)))
    return math.degrees(2.0 * math.acos(dot))


def _pose_distance(a, b):
    dx = float(a.position.x - b.position.x)
    dy = float(a.position.y - b.position.y)
    dz = float(a.position.z - b.position.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _parse_int_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [int(v) for v in txt.split() if v]
    return [int(v) for v in default]


def _parse_float_list_param(name, default, expected_len=None):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        values = [float(v) for v in raw]
    elif isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        values = [float(v) for v in txt.split() if v]
    else:
        values = [float(raw)]
    if expected_len is not None and len(values) != expected_len:
        return list(default)
    return values


def _parse_string_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [str(v).strip().lower() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [s.strip().lower() for s in txt.split() if s.strip()]
    return [str(v).strip().lower() for v in default if str(v).strip()]


class AprilTagGraspRegistry:
    def __init__(self):
        rospy.init_node("apriltag_grasp_registry")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.input_topic = str(rospy.get_param("~input_topic", "/apriltag_candidate_manager/detections")).strip()
        self.output_topic = str(rospy.get_param("~output_topic", "/apriltag_grasp_registry/detections")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.command_topic = str(rospy.get_param("~command_topic", "/task_context/command")).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.stable_count_required = int(rospy.get_param("~stable_count_required", 3))
        self.max_pos_jump_m = float(rospy.get_param("~max_pos_jump_m", 0.05))
        self.max_rot_jump_deg = float(rospy.get_param("~max_rot_jump_deg", 20.0))
        self.overwrite_recorded = bool(rospy.get_param("~overwrite_recorded", True))
        self.max_recorded_update_pos_jump_m = float(rospy.get_param("~max_recorded_update_pos_jump_m", -1.0))
        self.max_recorded_update_rot_jump_deg = float(rospy.get_param("~max_recorded_update_rot_jump_deg", -1.0))
        self.expected_recorded_count = int(rospy.get_param("~expected_recorded_count", 0))
        self.freeze_when_full = bool(rospy.get_param("~freeze_when_full", False))
        self.clear_commands = set(
            _parse_string_list_param("~clear_commands", ["scan_workspace", "scan", "reset_task", "reset"])
        )
        self.record_only_phases = set(_parse_string_list_param("~record_only_phases", []))
        self.freeze_outside_record_only_phases = bool(
            rospy.get_param("~freeze_outside_record_only_phases", False)
        )
        self.tag_ids = _parse_int_list_param("~tag_ids", [0, 1, 2, 3, 4, 5])
        self.synthesize_grasp_from_pregrasp_ids = set(
            _parse_int_list_param("~synthesize_grasp_from_pregrasp_ids", [])
        )
        self.synthesized_grasp_axis = _parse_float_list_param(
            "~synthesized_grasp_axis", [0.0, 0.0, 1.0], 3
        )
        self.synthesized_grasp_delta_m = float(rospy.get_param("~synthesized_grasp_delta_m", 0.0))
        self.input_namespace_prefix = str(
            rospy.get_param("~input_namespace_prefix", "/apriltag_candidates/tag_")
        ).strip()
        self.output_namespace_prefix = str(
            rospy.get_param("~output_namespace_prefix", "")
        ).strip()
        self.snapshot_yaml = os.path.expanduser(str(rospy.get_param("~snapshot_yaml", "")).strip())
        self.load_from_yaml = bool(rospy.get_param("~load_from_yaml", False))
        self.auto_save_yaml = bool(rospy.get_param("~auto_save_yaml", False))
        self.log_status_to_console = bool(rospy.get_param("~log_status_to_console", False))

        self.observed = {}
        self.recorded = {}
        self.recorded_pairs = {}
        self.live_pairs = {}
        self.recorded_pose_pubs = {}
        self.last_status = ""
        self.last_saved_count = 0
        self.registration_frozen = False
        self.current_phase = ""

        axis_norm = math.sqrt(sum(v * v for v in self.synthesized_grasp_axis))
        if axis_norm < 1e-9:
            self.synthesized_grasp_axis = [0.0, 0.0, 1.0]
        else:
            self.synthesized_grasp_axis = [v / axis_norm for v in self.synthesized_grasp_axis]

        self.pub = rospy.Publisher(self.output_topic, Detection2DArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        self._load_snapshot_if_requested()
        for tag_id in self.tag_ids:
            ns = "{}{}".format(self.input_namespace_prefix, tag_id)
            rospy.Subscriber("{}/pregrasp_pose".format(ns), PoseStamped, self._pregrasp_cb, callback_args=tag_id, queue_size=1)
            rospy.Subscriber("{}/grasp_pose".format(ns), PoseStamped, self._grasp_cb, callback_args=tag_id, queue_size=1)
        rospy.Subscriber(self.input_topic, Detection2DArray, self._input_cb, queue_size=1)
        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=10)
        if self.phase_topic:
            rospy.Subscriber(self.phase_topic, String, self._phase_cb, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._timer_cb)
        if self.recorded:
            self._publish_status("loaded_recorded_snapshot count={}".format(len(self.recorded)))
        else:
            self._publish_status("waiting_for_scan_candidates")

    @staticmethod
    def _pose_to_dict(pose):
        return {
            "position": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "orientation": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }

    @staticmethod
    def _dict_to_pose(data):
        pose = PoseStamped().pose
        position = list(data.get("position", []))
        orientation = list(data.get("orientation", []))
        if len(position) != 3 or len(orientation) != 4:
            raise ValueError("pose entry must contain position[3] and orientation[4]")
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])
        return pose

    @staticmethod
    def _pair_to_dict(pregrasp_pose, grasp_pose):
        data = {}
        if pregrasp_pose is not None:
            data["pregrasp_pose"] = AprilTagGraspRegistry._pose_to_dict(pregrasp_pose)
        if grasp_pose is not None:
            data["grasp_pose"] = AprilTagGraspRegistry._pose_to_dict(grasp_pose)
        return data

    def _pair_entry(self, tag_id):
        if tag_id not in self.live_pairs:
            self.live_pairs[tag_id] = {"pregrasp_pose": None, "grasp_pose": None}
        return self.live_pairs[tag_id]

    def _make_consistent_pair(self, tag_id, pregrasp_pose, grasp_pose):
        pair = {
            "pregrasp_pose": copy.deepcopy(pregrasp_pose),
            "grasp_pose": copy.deepcopy(grasp_pose),
        }
        if int(tag_id) not in self.synthesize_grasp_from_pregrasp_ids:
            return pair
        if pair["pregrasp_pose"] is None:
            return pair
        synthesized = copy.deepcopy(pair["pregrasp_pose"])
        synthesized.position.x -= float(self.synthesized_grasp_axis[0]) * self.synthesized_grasp_delta_m
        synthesized.position.y -= float(self.synthesized_grasp_axis[1]) * self.synthesized_grasp_delta_m
        synthesized.position.z -= float(self.synthesized_grasp_axis[2]) * self.synthesized_grasp_delta_m
        pair["grasp_pose"] = synthesized
        return pair

    def _pregrasp_cb(self, msg, tag_id):
        self._pair_entry(tag_id)["pregrasp_pose"] = copy.deepcopy(msg.pose)

    def _grasp_cb(self, msg, tag_id):
        self._pair_entry(tag_id)["grasp_pose"] = copy.deepcopy(msg.pose)

    def _load_snapshot_if_requested(self):
        if not self.load_from_yaml or not self.snapshot_yaml:
            return
        if not os.path.exists(self.snapshot_yaml):
            rospy.logwarn(
                "[apriltag_grasp_registry] snapshot_yaml does not exist: %s",
                self.snapshot_yaml,
            )
            return
        with open(self.snapshot_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        entries = raw.get("tag_grasps", {}) if isinstance(raw, dict) else {}
        loaded = {}
        for key, pose_dict in entries.items():
            try:
                tag_id = int(key)
                pair = {"pregrasp_pose": None, "grasp_pose": None}
                if isinstance(pose_dict, dict) and ("pregrasp_pose" in pose_dict or "grasp_pose" in pose_dict):
                    if isinstance(pose_dict.get("pregrasp_pose"), dict):
                        pair["pregrasp_pose"] = self._dict_to_pose(pose_dict.get("pregrasp_pose"))
                    if isinstance(pose_dict.get("grasp_pose"), dict):
                        pair["grasp_pose"] = self._dict_to_pose(pose_dict.get("grasp_pose"))
                else:
                    pair["pregrasp_pose"] = self._dict_to_pose(pose_dict if isinstance(pose_dict, dict) else {})
                loaded[tag_id] = pair
            except Exception as exc:
                rospy.logwarn(
                    "[apriltag_grasp_registry] skipping invalid snapshot entry %s: %s",
                    key,
                    exc,
                )
        for tag_id, pair in loaded.items():
            self.recorded_pairs[tag_id] = pair
            preferred_pose = pair.get("pregrasp_pose") or pair.get("grasp_pose")
            if preferred_pose is not None:
                self.recorded[tag_id] = copy.deepcopy(preferred_pose)
        self.last_saved_count = len(self.recorded_pairs)
        rospy.loginfo(
            "[apriltag_grasp_registry] loaded %d recorded tag grasps from %s",
            len(loaded),
            self.snapshot_yaml,
        )

    def _save_snapshot_if_enabled(self):
        if not self.auto_save_yaml or not self.snapshot_yaml:
            return
        data = {
            "base_frame": self.base_frame,
            "tag_grasps": {
                str(tag_id): self._pair_to_dict(
                    self.recorded_pairs.get(tag_id, {}).get("pregrasp_pose"),
                    self.recorded_pairs.get(tag_id, {}).get("grasp_pose"),
                )
                for tag_id in sorted(self.recorded_pairs.keys())
            },
        }
        parent = os.path.dirname(self.snapshot_yaml)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = self.snapshot_yaml + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=True)
        os.replace(tmp_path, self.snapshot_yaml)
        self.last_saved_count = len(self.recorded_pairs)
        rospy.loginfo(
            "[apriltag_grasp_registry] saved %d recorded tag grasps to %s",
            self.last_saved_count,
            self.snapshot_yaml,
        )

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        if self.log_status_to_console:
            rospy.loginfo("[apriltag_grasp_registry] %s", text)
        self.status_pub.publish(String(data=text))

    def _clear_registry(self):
        self.observed = {}
        self.recorded = {}
        self.recorded_pairs = {}
        self.live_pairs = {}
        self.last_saved_count = 0
        self.registration_frozen = False
        self._publish_status("registry_cleared_waiting_for_scan_candidates")

    def _remove_recorded_tag(self, tag_id):
        tag_id = int(tag_id)
        removed = False
        if tag_id in self.observed:
            del self.observed[tag_id]
        if tag_id in self.live_pairs:
            del self.live_pairs[tag_id]
        if tag_id in self.recorded:
            del self.recorded[tag_id]
            removed = True
        if tag_id in self.recorded_pairs:
            del self.recorded_pairs[tag_id]
            removed = True
        if removed:
            self.registration_frozen = False
            self._save_snapshot_if_enabled()
            self._publish_status(
                "removed_tag={} recorded={}/{} frozen={}".format(
                    tag_id,
                    len(self.recorded),
                    self.expected_recorded_count if self.expected_recorded_count > 0 else len(self.recorded),
                    self.registration_frozen,
                )
            )
        return removed

    def _command_cb(self, msg):
        cmd = str(msg.data).strip().lower()
        if cmd in self.clear_commands:
            self._clear_registry()
            return
        if cmd.startswith("remove_tag:"):
            try:
                tag_id = int(cmd.split(":", 1)[1].strip())
            except Exception:
                rospy.logwarn("[apriltag_grasp_registry] invalid remove_tag command: %s", cmd)
                return
            if not self._remove_recorded_tag(tag_id):
                rospy.loginfo("[apriltag_grasp_registry] remove_tag ignored; tag %s not recorded", tag_id)

    def _phase_cb(self, msg):
        self.current_phase = str(msg.data).strip().lower()
        if (
            self.freeze_outside_record_only_phases
            and self.record_only_phases
            and self.current_phase
            and self.current_phase not in self.record_only_phases
            and self.recorded
        ):
            self.registration_frozen = True

    def _pose_publishers(self, tag_id):
        if not self.output_namespace_prefix:
            return None
        if tag_id not in self.recorded_pose_pubs:
            ns = "{}{}".format(self.output_namespace_prefix, int(tag_id))
            self.recorded_pose_pubs[tag_id] = {
                "pregrasp": rospy.Publisher("{}/pregrasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
                "grasp": rospy.Publisher("{}/grasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
            }
        return self.recorded_pose_pubs[tag_id]

    def _input_cb(self, msg):
        if self.record_only_phases:
            phase = str(self.current_phase).strip().lower()
            if phase not in self.record_only_phases:
                if self.freeze_outside_record_only_phases and self.recorded:
                    self.registration_frozen = True
                self._publish_status(
                    "holding_recorded phase={} recorded={}/{} frozen={}".format(
                        phase or "unknown",
                        len(self.recorded),
                        self.expected_recorded_count if self.expected_recorded_count > 0 else len(self.recorded),
                        self.registration_frozen,
                    )
                )
                return
        if self.registration_frozen:
            self._publish_status(
                "registry_frozen recorded={}/{}".format(
                    len(self.recorded),
                    self.expected_recorded_count if self.expected_recorded_count > 0 else len(self.recorded),
                )
            )
            return
        seen_ids = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            tag_id = int(hyp.id)
            seen_ids.append(tag_id)
            pose = hyp.pose.pose

            obs = self.observed.get(tag_id)
            if obs is None:
                self.observed[tag_id] = {
                    "pose": copy.deepcopy(pose),
                    "stable_count": 1,
                }
            else:
                pos_jump = _pose_distance(obs["pose"], pose)
                rot_jump = _quat_angle_deg(obs["pose"].orientation, pose.orientation)
                if pos_jump <= self.max_pos_jump_m and rot_jump <= self.max_rot_jump_deg:
                    obs["stable_count"] += 1
                else:
                    obs["stable_count"] = 1
                obs["pose"] = copy.deepcopy(pose)

            obs = self.observed[tag_id]
            if obs["stable_count"] >= self.stable_count_required:
                previous_pose = self.recorded.get(tag_id)
                should_update = tag_id not in self.recorded or self.overwrite_recorded
                if should_update and previous_pose is not None:
                    pos_jump = _pose_distance(previous_pose, obs["pose"])
                    rot_jump = _quat_angle_deg(previous_pose.orientation, obs["pose"].orientation)
                    if self.max_recorded_update_pos_jump_m > 0.0 and pos_jump > self.max_recorded_update_pos_jump_m:
                        should_update = False
                    if self.max_recorded_update_rot_jump_deg > 0.0 and rot_jump > self.max_recorded_update_rot_jump_deg:
                        should_update = False
                if should_update:
                    self.recorded[tag_id] = copy.deepcopy(obs["pose"])
                    pair = self.live_pairs.get(tag_id, {})
                    self.recorded_pairs[tag_id] = self._make_consistent_pair(
                        tag_id,
                        pair.get("pregrasp_pose"),
                        pair.get("grasp_pose"),
                    )
                pose_changed = previous_pose is None
                if previous_pose is not None:
                    pos_jump = _pose_distance(previous_pose, obs["pose"])
                    rot_jump = _quat_angle_deg(previous_pose.orientation, obs["pose"].orientation)
                    pose_changed = pos_jump > 1e-6 or rot_jump > 1e-3
                if pose_changed:
                    self._save_snapshot_if_enabled()

                if (
                    self.freeze_when_full
                    and self.expected_recorded_count > 0
                    and len(self.recorded) >= self.expected_recorded_count
                ):
                    self.registration_frozen = True

        self._publish_status(
            "scanning seen={} recorded={}/{} frozen={}".format(
                sorted(seen_ids),
                len(self.recorded),
                self.expected_recorded_count if self.expected_recorded_count > 0 else len(self.recorded),
                self.registration_frozen,
            )
        )

    def _timer_cb(self, _evt):
        now = rospy.Time.now()
        out = Detection2DArray()
        out.header.stamp = now
        out.header.frame_id = self.base_frame

        for tag_id in sorted(self.recorded.keys()):
            pose = self.recorded[tag_id]
            det = Detection2D()
            det.header = out.header
            hyp = ObjectHypothesisWithPose()
            hyp.id = int(tag_id)
            hyp.score = 1.0
            hyp.pose.pose = copy.deepcopy(pose)
            det.results.append(hyp)
            out.detections.append(det)

            pubs = self._pose_publishers(tag_id)
            if pubs is not None:
                pair = self.recorded_pairs.get(tag_id, {})
                pre = pair.get("pregrasp_pose")
                grasp = pair.get("grasp_pose")
                if pre is not None:
                    msg = PoseStamped()
                    msg.header = out.header
                    msg.pose = copy.deepcopy(pre)
                    pubs["pregrasp"].publish(msg)
                if grasp is not None:
                    msg = PoseStamped()
                    msg.header = out.header
                    msg.pose = copy.deepcopy(grasp)
                    pubs["grasp"].publish(msg)

        self.pub.publish(out)


if __name__ == "__main__":
    AprilTagGraspRegistry()
    rospy.spin()
