#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intent inference over dynamic AprilTag grasp candidates."""

import math
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from intera_core_msgs.msg import EndpointState
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension, String
from vision_msgs.msg import Detection2DArray


def _normalize(vec):
    arr = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64), 0.0
    return arr / norm, norm


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

        self.beta = float(rospy.get_param("~beta", 10.0))
        self.distance_scale = float(rospy.get_param("~distance_scale", 0.12))
        self.w_alignment = float(rospy.get_param("~w_alignment", 1.5))
        self.w_distance = float(rospy.get_param("~w_distance", 1.0))
        self.w_stability = float(rospy.get_param("~w_stability", 0.2))
        self.min_input_norm = float(rospy.get_param("~min_input_norm", 0.05))
        self.input_hold_sec = float(rospy.get_param("~input_hold_sec", 0.35))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))
        self.intent_action_threshold = float(rospy.get_param("~intent_action_threshold", 1.1))

        self.joy_deadzone = float(rospy.get_param("~joy_deadzone", 0.1))
        self.joy_linear_axes = list(rospy.get_param("~joy_linear_axes", [1, 0, 4]))
        self.joy_axis_signs = list(rospy.get_param("~joy_axis_signs", [1.0, 1.0, 1.0]))

        self.latest_ee = None
        self.latest_buttons = []
        self.latest_input_vector = np.zeros(3, dtype=np.float64)
        self.latest_input_stamp = None
        self.candidates = []
        self.last_scores = {}
        self.prob_history = {}
        self.history = deque(maxlen=20)

        self.pub_dist = rospy.Publisher("~distribution", Float32MultiArray, queue_size=1)
        self.pub_top = rospy.Publisher("~top_goal", String, queue_size=1)
        self.pub_top_prob = rospy.Publisher("~top_probability", Float32, queue_size=1)
        self.pub_toppose = rospy.Publisher("~top_pose", PoseStamped, queue_size=1)
        self.pub_tracker = rospy.Publisher("~current_tracker_point", PointStamped, queue_size=1)
        self.pub_prompt = rospy.Publisher("~confirmation_prompt", String, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher("~status", String, queue_size=1, latch=True)

        rospy.Subscriber(self.end_effector_topic, EndpointState, self._ee_cb, queue_size=10)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        rospy.Subscriber(self.candidates_topic, Detection2DArray, self._candidates_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._tick)

        rospy.loginfo(
            "[apriltag_intent_inference] ready candidates=%s ee=%s joy=%s",
            self.candidates_topic,
            self.end_effector_topic,
            self.joy_topic,
        )

    def _ee_cb(self, msg):
        self.latest_ee = msg.pose
        pt = PointStamped()
        pt.header = msg.header
        pt.header.frame_id = self.base_frame
        pt.point = msg.pose.position
        self.pub_tracker.publish(pt)

    def _joy_cb(self, msg):
        self.latest_buttons = list(msg.buttons)
        axes = list(msg.axes)
        vec = np.zeros(3, dtype=np.float64)
        for i in range(min(3, len(self.joy_linear_axes))):
            axis_idx = int(self.joy_linear_axes[i])
            sign = float(self.joy_axis_signs[i]) if i < len(self.joy_axis_signs) else 1.0
            if 0 <= axis_idx < len(axes):
                val = float(axes[axis_idx])
                if abs(val) < self.joy_deadzone:
                    val = 0.0
                vec[i] = sign * val
        self.latest_input_vector = vec
        _, norm = _normalize(vec)
        if norm >= self.min_input_norm:
            self.latest_input_stamp = rospy.Time.now()
        self.history.append((rospy.Time.now().to_sec(), vec.copy()))

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
        self.candidates = parsed

    def _have_recent_input(self, now):
        return self.latest_input_stamp is not None and (now - self.latest_input_stamp).to_sec() <= self.input_hold_sec

    def _compute_scores(self):
        if self.latest_ee is None or not self.candidates:
            return []

        ee_pos = np.array(
            [self.latest_ee.position.x, self.latest_ee.position.y, self.latest_ee.position.z],
            dtype=np.float64,
        )
        input_dir, input_norm = _normalize(self.latest_input_vector)
        use_alignment = self._have_recent_input(rospy.Time.now()) and input_norm >= self.min_input_norm

        scores = []
        for label, pose_msg in self.candidates:
            goal = np.array(
                [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z],
                dtype=np.float64,
            )
            delta = goal - ee_pos
            dir_to_goal, dist = _normalize(delta)

            alignment = float(np.dot(input_dir, dir_to_goal)) if use_alignment else 0.0
            distance_term = -dist / max(self.distance_scale, 1e-6)
            stability_term = self.prob_history.get(label, 0.0)
            score = self.w_distance * distance_term
            if use_alignment:
                score += self.w_alignment * alignment
            score += self.w_stability * stability_term
            scores.append((label, pose_msg, score))
        return scores

    def _tick(self, _evt):
        scores = self._compute_scores()
        if not scores:
            self.pub_prompt.publish(String(data="Move the wrist camera to discover tags."))
            self.pub_status.publish(String(data="waiting_for_candidates_or_ee"))
            return

        max_score = max(score for _, _, score in scores)
        exp_scores = [math.exp(self.beta * (score - max_score)) for _, _, score in scores]
        z = max(sum(exp_scores), 1e-9)
        probs = [v / z for v in exp_scores]

        labels = [label for label, _, _ in scores]
        for label, prob in zip(labels, probs):
            self.prob_history[label] = 0.7 * self.prob_history.get(label, prob) + 0.3 * prob

        dist_msg = Float32MultiArray()
        dist_msg.layout.dim.append(MultiArrayDimension(label="objects", size=len(probs), stride=len(probs)))
        dist_msg.data = probs
        self.pub_dist.publish(dist_msg)

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
