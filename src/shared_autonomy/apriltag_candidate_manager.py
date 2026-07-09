#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate multiple AprilTag grasp candidates into a single dynamic list."""

import copy
import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def _parse_int_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        parts = [p for p in txt.split() if p]
        return [int(v) for v in parts]
    return [int(v) for v in default]


class AprilTagCandidateManager:
    def __init__(self):
        rospy.init_node("apriltag_candidate_manager")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.tag_ids = _parse_int_list_param("~tag_ids", [0, 1, 2])
        self.input_namespace_prefix = str(rospy.get_param("~input_namespace_prefix", "/apriltag_candidates/tag_")).strip()
        self.output_topic = str(rospy.get_param("~output_topic", "/apriltag_candidate_manager/detections")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
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
        self.stage = str(rospy.get_param("~stage", "pregrasp")).strip().lower()
        self.stale_timeout_sec = float(rospy.get_param("~stale_timeout_sec", 1.0))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.log_status_to_console = bool(rospy.get_param("~log_status_to_console", False))

        self.object_map = self._load_object_map()
        self.candidates = {}
        self.last_status = ""

        self.pub = rospy.Publisher(self.output_topic, Detection2DArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        for tag_id in self.tag_ids:
            ns = "{}{}".format(self.input_namespace_prefix, tag_id)
            rospy.Subscriber("{}/base_tag_pose".format(ns), PoseStamped, self._base_tag_cb, callback_args=tag_id, queue_size=1)
            rospy.Subscriber("{}/pregrasp_pose".format(ns), PoseStamped, self._pregrasp_cb, callback_args=tag_id, queue_size=1)
            rospy.Subscriber("{}/grasp_pose".format(ns), PoseStamped, self._grasp_cb, callback_args=tag_id, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._timer_cb)
        self._publish_status("waiting_for_candidates tag_ids={}".format(self.tag_ids))

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            rospy.logwarn("[apriltag_candidate_manager] object map YAML not found: %s", self.object_map_yaml)
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("tag_objects", {}) if isinstance(data, dict) else {}

    def _candidate(self, tag_id):
        if tag_id not in self.candidates:
            self.candidates[tag_id] = {
                "base_tag_pose": None,
                "pregrasp_pose": None,
                "grasp_pose": None,
                "updated_at": rospy.Time(0),
            }
        return self.candidates[tag_id]

    def _base_tag_cb(self, msg, tag_id):
        candidate = self._candidate(tag_id)
        candidate["base_tag_pose"] = copy.deepcopy(msg)
        candidate["updated_at"] = rospy.Time.now()

    def _pregrasp_cb(self, msg, tag_id):
        candidate = self._candidate(tag_id)
        candidate["pregrasp_pose"] = copy.deepcopy(msg)
        candidate["updated_at"] = rospy.Time.now()

    def _grasp_cb(self, msg, tag_id):
        candidate = self._candidate(tag_id)
        candidate["grasp_pose"] = copy.deepcopy(msg)
        candidate["updated_at"] = rospy.Time.now()

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        if self.log_status_to_console:
            rospy.loginfo("[apriltag_candidate_manager] %s", text)
        self.status_pub.publish(String(data=text))

    def _is_fresh(self, candidate, now):
        updated = candidate.get("updated_at", rospy.Time(0))
        if updated == rospy.Time(0):
            return False
        return (now - updated).to_sec() <= self.stale_timeout_sec

    def _label_for(self, tag_id):
        meta = self.object_map.get(tag_id, self.object_map.get(str(tag_id), {}))
        if isinstance(meta, dict) and meta.get("grasp_complete_label"):
            return str(meta["grasp_complete_label"]).strip()
        return "apriltag_id_{}".format(tag_id)

    def _pose_for_stage(self, candidate):
        if self.stage == "grasp":
            return candidate.get("grasp_pose")
        return candidate.get("pregrasp_pose")

    def _timer_cb(self, _evt):
        now = rospy.Time.now()
        out = Detection2DArray()
        out.header.stamp = now
        out.header.frame_id = self.base_frame

        active_ids = []
        for tag_id in self.tag_ids:
            candidate = self.candidates.get(tag_id)
            if candidate is None or not self._is_fresh(candidate, now):
                continue
            pose_msg = self._pose_for_stage(candidate)
            if pose_msg is None:
                continue

            det = Detection2D()
            det.header = out.header
            hyp = ObjectHypothesisWithPose()
            hyp.id = int(tag_id)
            hyp.score = 1.0
            hyp.pose.pose = copy.deepcopy(pose_msg.pose)
            det.results.append(hyp)
            out.detections.append(det)
            active_ids.append(tag_id)

        self.pub.publish(out)
        self._publish_status("active_tags={} stage={} count={}".format(active_ids, self.stage, len(active_ids)))


if __name__ == "__main__":
    AprilTagCandidateManager()
    rospy.spin()
