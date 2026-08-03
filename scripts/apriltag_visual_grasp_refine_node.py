#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish grasp refinement offsets from live AprilTag pose drift."""

import os

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


def _parse_int_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [int(v) for v in txt.split() if v]
    return [int(v) for v in default]


def _parse_csv_set(raw):
    if raw is None:
        return set()
    return {chunk.strip().lower() for chunk in str(raw).split(",") if chunk.strip()}


def _pos(msg):
    p = msg.pose.position
    return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)


class AprilTagVisualGraspRefineNode:
    def __init__(self):
        rospy.init_node("apriltag_visual_grasp_refine")

        pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param("~object_map_yaml", os.path.join(pkg_dir, "config", "apriltag_object_map.yaml"))
        )
        self.tag_ids = _parse_int_list_param("~tag_ids", [0, 1, 2, 3, 4, 5])
        self.observed_namespace_prefix = str(
            rospy.get_param("~observed_namespace_prefix", "/apriltag_candidates/tag_")
        ).strip()
        self.reference_detections_topic = str(
            rospy.get_param("~reference_detections_topic", "/apriltag_grasp_registry/detections")
        ).strip()
        self.offset_topic = str(rospy.get_param("~offset_topic", "/visual_grasp_refine/offset")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.execution_state_topic = str(
            rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")
        ).strip()
        self.selected_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.allowed_categories = _parse_csv_set(rospy.get_param("~allowed_categories", "milk,cereal,chocolate"))
        self.active_states = _parse_csv_set(rospy.get_param("~active_states", "exec_pregrasp,visual_align"))
        self.max_observed_age_sec = float(rospy.get_param("~max_observed_age_sec", 0.35))
        self.max_reference_age_sec = float(rospy.get_param("~max_reference_age_sec", 5.0))
        self.max_xy_m = float(rospy.get_param("~max_xy_m", 0.035))
        self.max_z_m = float(rospy.get_param("~max_z_m", 0.0))
        self.publish_rate_hz = max(1.0, float(rospy.get_param("~publish_rate_hz", 20.0)))

        self.label_to_meta = self._load_label_metadata()
        self.execution_state = ""
        self.selected_label = ""
        self.observed = {}
        self.references = {}
        self.captured_reference = None
        self.captured_reference_label = ""
        self.last_status = ""

        self.pub = rospy.Publisher(self.offset_topic, Vector3Stamped, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(self.reference_detections_topic, Detection2DArray, self._reference_cb, queue_size=1)
        rospy.Subscriber(self.execution_state_topic, String, self._state_cb, queue_size=1)
        rospy.Subscriber(self.selected_label_topic, String, self._label_cb, queue_size=1)
        for tag_id in self.tag_ids:
            ns = "{}{}".format(self.observed_namespace_prefix, tag_id)
            rospy.Subscriber("{}/base_tag_pose".format(ns), PoseStamped, self._observed_cb, callback_args=tag_id, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / self.publish_rate_hz), self._tick)
        self._publish_status("waiting_for_selection")
        rospy.loginfo(
            "[apriltag_visual_grasp_refine] observed_prefix=%s reference=%s offset=%s",
            self.observed_namespace_prefix,
            self.reference_detections_topic,
            self.offset_topic,
        )

    def _load_label_metadata(self):
        if not os.path.exists(self.object_map_yaml):
            rospy.logwarn("[apriltag_visual_grasp_refine] object map YAML not found: %s", self.object_map_yaml)
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict) and isinstance(data.get("tag_objects"), dict):
            objects = data.get("tag_objects", {}) or {}
        elif isinstance(data, dict) and isinstance(data.get("candidate_objects"), dict):
            objects = data.get("candidate_objects", {}) or {}
        else:
            objects = {}
        mapping = {}
        for key, meta in objects.items():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("grasp_complete_label", "")).strip()
            if not label:
                continue
            enriched = dict(meta)
            try:
                enriched["tag_id"] = int(key)
            except Exception:
                enriched["tag_id"] = key
            mapping[label] = enriched
        return mapping

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        self.status_pub.publish(String(data=text))

    def _reference_cb(self, msg):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        for det in msg.detections:
            if not det.results:
                continue
            try:
                tag_id = int(det.results[0].id)
            except Exception:
                continue
            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.pose = det.results[0].pose.pose
            self.references[tag_id] = (pose_msg, stamp)

    def _observed_cb(self, msg, tag_id):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        self.observed[int(tag_id)] = (msg, stamp)

    def _state_cb(self, msg):
        next_state = str(msg.data).strip().lower()
        if next_state not in self.active_states:
            self.captured_reference = None
            self.captured_reference_label = ""
        self.execution_state = next_state

    def _label_cb(self, msg):
        next_label = str(msg.data).strip()
        if next_label != self.selected_label:
            self.captured_reference = None
            self.captured_reference_label = ""
        self.selected_label = next_label

    def _selected_meta(self):
        if not self.selected_label:
            return None
        meta = self.label_to_meta.get(self.selected_label)
        if not isinstance(meta, dict):
            return None
        category = str(meta.get("category", "")).strip().lower()
        if self.allowed_categories and category not in self.allowed_categories:
            return None
        return meta

    def _fresh_pose(self, store, tag_id, max_age_sec, now):
        item = store.get(int(tag_id))
        if item is None:
            return None
        msg, stamp = item
        if stamp is None or (now - stamp).to_sec() > max_age_sec:
            return None
        return msg

    def _reference_pose(self, tag_id, observed_pose, now):
        reference = self._fresh_pose(self.references, tag_id, self.max_reference_age_sec, now)
        if reference is not None:
            return reference
        if self.captured_reference is None or self.captured_reference_label != self.selected_label:
            self.captured_reference = observed_pose
            self.captured_reference_label = self.selected_label
            self._publish_status("captured_live_reference:{}".format(self.selected_label))
        return self.captured_reference

    def _clamped_offset(self, delta):
        out = np.array(delta, dtype=np.float64)
        xy_norm = float(np.linalg.norm(out[:2]))
        if self.max_xy_m > 0.0 and xy_norm > self.max_xy_m:
            out[:2] *= self.max_xy_m / xy_norm
        if self.max_z_m <= 0.0:
            out[2] = 0.0
        else:
            out[2] = float(np.clip(out[2], -self.max_z_m, self.max_z_m))
        return out

    def _tick(self, _evt):
        if self.execution_state not in self.active_states:
            self._publish_status("inactive_state:{}".format(self.execution_state or "unknown"))
            return
        meta = self._selected_meta()
        if meta is None:
            self._publish_status("waiting_for_allowed_selection")
            return
        try:
            tag_id = int(meta.get("tag_id"))
        except Exception:
            self._publish_status("selected_label_without_numeric_tag:{}".format(self.selected_label))
            return

        now = rospy.Time.now()
        observed = self._fresh_pose(self.observed, tag_id, self.max_observed_age_sec, now)
        if observed is None:
            self._publish_status("waiting_for_live_tag:{}".format(tag_id))
            return
        reference = self._reference_pose(tag_id, observed, now)
        if reference is None:
            self._publish_status("waiting_for_reference_tag:{}".format(tag_id))
            return

        offset = self._clamped_offset(_pos(observed) - _pos(reference))
        msg = Vector3Stamped()
        msg.header.stamp = now
        msg.header.frame_id = observed.header.frame_id or self.base_frame
        msg.vector.x = float(offset[0])
        msg.vector.y = float(offset[1])
        msg.vector.z = float(offset[2])
        self.pub.publish(msg)
        self._publish_status(
            "offset label={} tag={} dx={:.3f} dy={:.3f} dz={:.3f}".format(
                self.selected_label,
                tag_id,
                offset[0],
                offset[1],
                offset[2],
            )
        )


if __name__ == "__main__":
    AprilTagVisualGraspRefineNode()
    rospy.spin()
