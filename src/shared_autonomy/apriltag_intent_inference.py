#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trajectory-based intent inference over scanned AprilTag grasp candidates."""

import os
import math
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from intera_core_msgs.msg import EndpointState
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray, MultiArrayDimension, String
from vision_msgs.msg import Detection2DArray
import yaml


class AprilTagIntentInference:
    def __init__(self):
        rospy.init_node("apriltag_intent_inference")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.end_effector_topic = str(
            rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        ).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()
        self.candidates_topic = str(
            rospy.get_param("~candidates_topic", "/apriltag_candidate_manager/detections")
        ).strip()
        self.allowed_ids_topic = str(
            rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")
        ).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.task_phase_topic = str(
            rospy.get_param("~task_phase_topic", "/task_context/phase")
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

        self.beta = float(rospy.get_param("~beta", 3.0))
        self.window_s = float(rospy.get_param("~window_sec", 1.2))
        self.speed_eps = float(rospy.get_param("~stationary_speed_mps", 0.03))
        self.reset_hold_sec = float(rospy.get_param("~reset_hold_sec", 2.0))
        self.warmup_sec = float(rospy.get_param("~warmup_sec", 0.0))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))
        self.probability_smoothing_alpha = float(rospy.get_param("~probability_smoothing_alpha", 0.2))
        self.intent_action_threshold = float(rospy.get_param("~intent_action_threshold", 1.1))

        self.latest_ee = None
        self.latest_buttons = []
        self.candidates = []
        self.allowed_tag_ids = set()
        self.history = deque()
        self.start_point = None
        self.last_move_t = None
        self.start_time = rospy.get_time()
        self.smoothed_probs = {}
        self.selected_grasp_label = ""
        self.current_phase = "scan_workspace"
        self.label_to_meta, self.tag_id_to_meta = self._load_object_map()

        self.pub_dist = rospy.Publisher("~distribution", Float32MultiArray, queue_size=1)
        self.pub_dist_labels = rospy.Publisher("~distribution_labels", Int32MultiArray, queue_size=1)
        self.pub_top = rospy.Publisher("~top_goal", String, queue_size=1)
        self.pub_top_prob = rospy.Publisher("~top_probability", Float32, queue_size=1)
        self.pub_toppose = rospy.Publisher("~top_pose", PoseStamped, queue_size=1)
        self.pub_tracker = rospy.Publisher("~current_tracker_point", PointStamped, queue_size=1)
        self.pub_prompt = rospy.Publisher("~confirmation_prompt", String, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher("~status", String, queue_size=1, latch=True)

        rospy.Subscriber(self.end_effector_topic, EndpointState, self._ee_cb, queue_size=10)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        rospy.Subscriber(self.candidates_topic, Detection2DArray, self._candidates_cb, queue_size=1)
        rospy.Subscriber(self.allowed_ids_topic, Int32MultiArray, self._allowed_ids_cb, queue_size=1)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_grasp_label_cb, queue_size=1)
        rospy.Subscriber(self.task_phase_topic, String, self._task_phase_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._tick)

        rospy.loginfo(
            "[apriltag_intent_inference] ready candidates=%s ee=%s joy=%s",
            self.candidates_topic,
            self.end_effector_topic,
            self.joy_topic,
        )

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}, {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        by_label = {}
        by_tag_id = {}
        if isinstance(raw.get("tag_objects"), dict):
            entries = raw.get("tag_objects", {}) or {}
        elif isinstance(raw.get("candidate_objects"), dict):
            entries = raw.get("candidate_objects", {}) or {}
        else:
            entries = {}
        for key, meta in entries.items():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("grasp_complete_label", "")).strip()
            if label:
                by_label[label] = meta
            try:
                by_tag_id[int(key)] = meta
            except Exception:
                continue
        return by_label, by_tag_id

    def _ee_cb(self, msg):
        self.latest_ee = msg.pose
        pt = PointStamped()
        pt.header = msg.header
        pt.header.frame_id = self.base_frame
        pt.point = msg.pose.position
        self.pub_tracker.publish(pt)
        self._update_reach_state(pt)

    def _joy_cb(self, msg):
        self.latest_buttons = list(msg.buttons)

    def _selected_grasp_label_cb(self, msg):
        next_label = str(msg.data).strip()
        if next_label == self.selected_grasp_label:
            return
        previous_label = self.selected_grasp_label
        self.selected_grasp_label = next_label
        if not next_label:
            self._reset_intent_state("selected_grasp_cleared")
            return
        if previous_label and previous_label != next_label:
            self._reset_intent_state("selected_grasp_changed")

    def _task_phase_cb(self, msg):
        self.current_phase = str(msg.data).strip().lower() or "scan_workspace"

    def _candidates_cb(self, msg):
        parsed = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            label = str(int(hyp.id))
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = hyp.pose.pose
            parsed.append((label, pose))
        parsed.sort(key=lambda item: int(item[0]) if str(item[0]).lstrip("-").isdigit() else item[0])
        self.candidates = parsed

    def _allowed_ids_cb(self, msg):
        next_allowed = set(int(v) for v in list(msg.data))
        if next_allowed != self.allowed_tag_ids:
            self.allowed_tag_ids = next_allowed
            self._reset_intent_state("allowed_ids_changed")
            return
        self.allowed_tag_ids = next_allowed

    def _reset_intent_state(self, reason):
        self.history.clear()
        self.start_point = None
        self.last_move_t = None
        self.smoothed_probs = {}
        self.pub_dist.publish(Float32MultiArray(data=[]))
        self.pub_top.publish(String(data=""))
        self.pub_top_prob.publish(Float32(data=0.0))
        self.pub_status.publish(String(data=reason))
        if self.allowed_tag_ids:
            self.pub_prompt.publish(
                String(
                    data="Task changed. Move the end-effector to begin intent inference for tags {}.".format(
                        sorted(self.allowed_tag_ids)
                    )
                )
            )
        else:
            self.pub_prompt.publish(String(data="Task changed. Move the wrist camera to discover tags."))

    def _update_reach_state(self, msg):
        stamp_sec = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time(0) else rospy.Time.now().to_sec()
        p_tuple = (msg.point.x, msg.point.y, msg.point.z)
        self.history.append((stamp_sec, p_tuple))
        t_min = stamp_sec - self.window_s
        while self.history and self.history[0][0] < t_min:
            self.history.popleft()

        if rospy.get_time() - self.start_time < self.warmup_sec:
            return

        speed = 0.0
        if len(self.history) >= 2:
            (t0, p0), (t1, p1) = self.history[-2], self.history[-1]
            dt = max(1e-6, t1 - t0)
            speed = float(np.linalg.norm(np.subtract(p1, p0)) / dt)

        if speed > self.speed_eps:
            self.last_move_t = stamp_sec
            if self.start_point is None:
                self.start_point = msg.point
                rospy.loginfo("[apriltag_intent_inference] reach detected, start intent inference")
        elif self.last_move_t is not None and (stamp_sec - self.last_move_t) > self.reset_hold_sec:
            if self.start_point is not None:
                rospy.loginfo("[apriltag_intent_inference] reach ended, resetting intent inference")
            self.start_point = None

    def _compute_scores(self):
        if self.latest_ee is None or not self.candidates or self.start_point is None:
            return []

        start = (self.start_point.x, self.start_point.y, self.start_point.z)
        current = (
            float(self.latest_ee.position.x),
            float(self.latest_ee.position.y),
            float(self.latest_ee.position.z),
        )
        observed_path_length = self._path_length_observed()

        scores = []
        for label, pose_msg in self.candidates:
            if self.allowed_tag_ids:
                try:
                    if int(label) not in self.allowed_tag_ids:
                        continue
                except Exception:
                    continue
            goal = (
                float(pose_msg.pose.position.x),
                float(pose_msg.pose.position.y),
                float(pose_msg.pose.position.z),
            )
            d_start_goal = self._vec_dist(start, goal)
            if d_start_goal < 1e-3:
                continue
            d_current_goal = self._vec_dist(current, goal)
            score = -self.beta * (observed_path_length + d_current_goal) / d_start_goal
            score += self._prior_log_bias_for_candidate(label)
            scores.append((label, pose_msg, score))
        return scores

    def _prior_log_bias_for_candidate(self, label):
        if self.current_phase not in ("select_sort_destination", "sort_destination"):
            return 0.0
        if not self.selected_grasp_label:
            return 0.0
        object_meta = self.label_to_meta.get(self.selected_grasp_label, {})
        priors = object_meta.get("sorting_destination_priors", {}) if isinstance(object_meta, dict) else {}
        if not isinstance(priors, dict):
            return 0.0
        try:
            tag_id = int(label)
        except Exception:
            return 0.0
        destination_meta = self.tag_id_to_meta.get(tag_id, {})
        destination_name = str(destination_meta.get("destination_name", destination_meta.get("object_name", ""))).strip().lower()
        if not destination_name:
            return 0.0
        prior = float(priors.get(destination_name, 0.5))
        prior = max(1e-3, min(1.0, prior))
        return math.log(prior)

    def _path_length_observed(self):
        if len(self.history) < 2:
            return 0.0
        points = [p for (_, p) in self.history]
        return sum(float(np.linalg.norm(np.subtract(points[i], points[i - 1]))) for i in range(1, len(points)))

    def _vec_dist(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _smooth_probabilities(self, labels, raw_probs):
        alpha = max(0.0, min(1.0, self.probability_smoothing_alpha))
        if alpha <= 0.0:
            self.smoothed_probs = {label: float(prob) for label, prob in zip(labels, raw_probs)}
            return list(raw_probs)

        next_probs = {}
        for label, raw_prob in zip(labels, raw_probs):
            prev = self.smoothed_probs.get(label, float(raw_prob))
            next_probs[label] = (1.0 - alpha) * float(prev) + alpha * float(raw_prob)

        total = sum(next_probs.values())
        if total > 1e-9:
            for label in list(next_probs.keys()):
                next_probs[label] /= total

        self.smoothed_probs = next_probs
        return [next_probs[label] for label in labels]

    def _tick(self, _evt):
        scores = self._compute_scores()
        if not scores:
            if self.allowed_tag_ids:
                if self.start_point is None:
                    self.pub_prompt.publish(String(data="Move the end-effector to begin intent inference for tags {}.".format(sorted(self.allowed_tag_ids))))
                    self.pub_status.publish(String(data="waiting_for_reach_start"))
                else:
                    self.pub_prompt.publish(
                        String(data="No recorded candidates in current task set {}.".format(sorted(self.allowed_tag_ids)))
                    )
                    self.pub_status.publish(String(data="waiting_for_candidates_in_allowed_set"))
            else:
                self.pub_prompt.publish(String(data="Move the wrist camera to discover tags."))
                self.pub_status.publish(String(data="waiting_for_candidates_or_ee"))
            return

        max_score = max(score for _, _, score in scores)
        exp_scores = [math.exp(score - max_score) for _, _, score in scores]
        z = max(sum(exp_scores), 1e-9)
        labels = [label for label, _, _ in scores]
        probs = [v / z for v in exp_scores]
        probs = self._smooth_probabilities(labels, probs)

        dist_msg = Float32MultiArray()
        dist_msg.layout.dim.append(MultiArrayDimension(label="objects", size=len(probs), stride=len(probs)))
        dist_msg.data = probs
        self.pub_dist.publish(dist_msg)
        labels_msg = Int32MultiArray()
        labels_msg.data = [int(label) for label in labels if str(label).lstrip("-").isdigit()]
        self.pub_dist_labels.publish(labels_msg)

        top_idx = int(np.argmax(probs))
        top_label, top_pose, _ = scores[top_idx]
        top_prob = probs[top_idx]
        self.pub_top.publish(String(data=top_label))
        self.pub_top_prob.publish(Float32(data=top_prob))
        self.pub_toppose.publish(top_pose)
        self.pub_status.publish(String(data="top_goal={} prob={:.2f}".format(top_label, top_prob)))
        self.pub_prompt.publish(
            String(
                data="Top tag {} ({:.2f}). Use joystick to indicate intent.".format(
                    top_label, top_prob
                )
            )
        )


if __name__ == "__main__":
    AprilTagIntentInference()
    rospy.spin()
