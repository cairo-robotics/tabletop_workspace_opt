#!/usr/bin/env python3
"""Shared autonomy grasp selector.

Selects the most likely intended grasp from a set of candidates using a robust MAP-like approach:
1. Cumulative joystick likelihood (input vs goal-conditioned direction)
2. Prior / belief carryover across frames
3. Alignment, upstream grasp score, and feasibility as auxiliary terms
"""

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from sensor_msgs.msg import Joy
from tabletop_workspace_opt.msg import (
    GraspCandidate,
    GraspCandidateArray,
    GraspScore,
    GraspScoreArray,
)

try:
    from relaxed_ik_ros1.msg import EEVelGoals
except ImportError:
    EEVelGoals = None


def _as_numpy_xyz(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


def _normalize(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64), 0.0
    return vec / norm, norm


class SharedAutonomyGraspSelector:
    def __init__(self):
        rospy.init_node("shared_autonomy_grasp_selector")

        self.base_frame = rospy.get_param("~base_frame", "")
        self.ee_pose_timeout = float(rospy.get_param("~ee_pose_timeout_sec", 0.5))
        self.input_timeout = float(rospy.get_param("~input_timeout_sec", 0.5))
        self.min_input_norm = float(rospy.get_param("~min_input_norm", 1e-4))

        self.w_alignment = float(rospy.get_param("~w_alignment", 0.6))
        self.w_grasp_score = float(rospy.get_param("~w_grasp_score", 0.3))
        self.w_feasibility = float(rospy.get_param("~w_feasibility", 0.1))
        self.w_high_z_penalty = float(rospy.get_param("~w_high_z_penalty", 0.25))
        self.clamp_negative_alignment = bool(rospy.get_param("~clamp_negative_alignment", False))
        self.high_z_penalty_start = float(rospy.get_param("~high_z_penalty_start", 0.92))
        self.high_z_penalty_scale = float(rospy.get_param("~high_z_penalty_scale", 0.08))

        self.joy_deadzone = float(rospy.get_param("~joy_deadzone", 0.1))
        self.joy_linear_axes = list(rospy.get_param("~joy_linear_axes", [1, 0, 4]))
        self.joy_axis_signs = list(rospy.get_param("~joy_axis_signs", [1.0, 1.0, 1.0]))

        ee_pose_topic = rospy.get_param("~ee_pose_topic", "/ee_pose")
        candidate_topic = rospy.get_param("~candidate_grasps_topic", "/candidate_grasps")
        joy_topic = rospy.get_param("~joy_topic", "/joy")
        twist_topic = rospy.get_param("~joystick_twist_topic", "")
        twist_stamped_topic = rospy.get_param("~joystick_twist_stamped_topic", "")
        ee_vel_goals_topic = rospy.get_param("~ee_vel_goals_topic", "")

        selected_grasp_topic = rospy.get_param("~selected_grasp_topic", "/selected_grasp")
        selected_pose_topic = rospy.get_param("~selected_grasp_pose_topic", "/selected_grasp_pose")
        score_topic = rospy.get_param("~grasp_scores_topic", "/grasp_scores")

        self.latest_ee_pose: Optional[PoseStamped] = None
        self.latest_candidates: Optional[GraspCandidateArray] = None
        self.latest_input_vector = np.zeros(3, dtype=np.float64)
        self.latest_input_stamp: Optional[rospy.Time] = None
        self.input_history: Deque[Tuple[rospy.Time, np.ndarray]] = deque()

        self.obs_window_sec = float(rospy.get_param("~obs_window_sec", 0.5))
        self.noise_cov_diag = list(rospy.get_param("~noise_cov_diag", [0.1, 0.1, 0.1]))
        self.noise_cov_inv = np.linalg.inv(np.diag(np.array(self.noise_cov_diag, dtype=np.float64)))
        self.belief_decay = float(rospy.get_param("~belief_decay", 0.98))
        self.w_likelihood = float(rospy.get_param("~w_likelihood", 1.0))

        self.log_beliefs: Dict[str, float] = {}

        self.selected_grasp_pub = rospy.Publisher(selected_grasp_topic, GraspCandidate, queue_size=1)
        self.selected_pose_pub = rospy.Publisher(selected_pose_topic, PoseStamped, queue_size=1)
        self.score_pub = rospy.Publisher(score_topic, GraspScoreArray, queue_size=1)

        rospy.Subscriber(ee_pose_topic, PoseStamped, self.ee_pose_cb, queue_size=1)
        rospy.Subscriber(candidate_topic, GraspCandidateArray, self.candidate_cb, queue_size=1)
        rospy.Subscriber(joy_topic, Joy, self.joy_cb, queue_size=1)

        if twist_topic:
            rospy.Subscriber(twist_topic, Twist, self.twist_cb, queue_size=1)
        if twist_stamped_topic:
            rospy.Subscriber(twist_stamped_topic, TwistStamped, self.twist_stamped_cb, queue_size=1)
        if ee_vel_goals_topic and EEVelGoals is not None:
            rospy.Subscriber(ee_vel_goals_topic, EEVelGoals, self.ee_vel_goals_cb, queue_size=1)
        elif ee_vel_goals_topic and EEVelGoals is None:
            rospy.logwarn("Requested ~ee_vel_goals_topic but relaxed_ik_ros1.msg.EEVelGoals is unavailable.")

        self.timer = rospy.Timer(rospy.Duration(1.0 / 20.0), self.timer_cb)

        rospy.loginfo(
            "Shared autonomy grasp selector ready. "
            "ee_pose=%s candidate_grasps=%s joy=%s",
            ee_pose_topic,
            candidate_topic,
            joy_topic,
        )

    def ee_pose_cb(self, msg: PoseStamped):
        self.latest_ee_pose = msg

    def candidate_cb(self, msg: GraspCandidateArray):
        self.latest_candidates = msg

    def joy_cb(self, msg: Joy):
        direction = np.zeros(3, dtype=np.float64)
        for i in range(3):
            axis_index = int(self.joy_linear_axes[i])
            if axis_index < len(msg.axes):
                value = float(msg.axes[axis_index])
                if abs(value) < self.joy_deadzone:
                    value = 0.0
                direction[i] = float(self.joy_axis_signs[i]) * value

        self.latest_input_vector = direction
        self.latest_input_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        self.input_history.append((self.latest_input_stamp, self.latest_input_vector.copy()))
        self._prune_input_history()

    def twist_cb(self, msg: Twist):
        self.latest_input_vector = _as_numpy_xyz(msg.linear.x, msg.linear.y, msg.linear.z)
        self.latest_input_stamp = rospy.Time.now()
        self.input_history.append((self.latest_input_stamp, self.latest_input_vector.copy()))
        self._prune_input_history()

    def twist_stamped_cb(self, msg: TwistStamped):
        self.latest_input_vector = _as_numpy_xyz(msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z)
        self.latest_input_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        self.input_history.append((self.latest_input_stamp, self.latest_input_vector.copy()))
        self._prune_input_history()

    def ee_vel_goals_cb(self, msg):
        if not msg.ee_vels:
            return
        twist = msg.ee_vels[0]
        self.latest_input_vector = _as_numpy_xyz(twist.linear.x, twist.linear.y, twist.linear.z)
        self.latest_input_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    def timer_cb(self, _event):
        if self.latest_ee_pose is None or self.latest_candidates is None:
            return

        now = rospy.Time.now()
        ee_stamp = self.latest_ee_pose.header.stamp
        if ee_stamp == rospy.Time():
            ee_stamp = now

        if self.base_frame:
            if self.latest_ee_pose.header.frame_id and self.latest_ee_pose.header.frame_id != self.base_frame:
                rospy.logwarn_throttle(
                    2.0,
                    "Skipping grasp selection: ee pose frame '%s' does not match base frame '%s'.",
                    self.latest_ee_pose.header.frame_id,
                    self.base_frame,
                )
                return
            if self.latest_candidates.header.frame_id and self.latest_candidates.header.frame_id != self.base_frame:
                rospy.logwarn_throttle(
                    2.0,
                    "Skipping grasp selection: candidate frame '%s' does not match base frame '%s'.",
                    self.latest_candidates.header.frame_id,
                    self.base_frame,
                )
                return

        if (now - ee_stamp).to_sec() > self.ee_pose_timeout:
            rospy.logwarn_throttle(2.0, "Skipping grasp selection: ee pose is stale.")
            return

        if not self.latest_candidates.grasps:
            rospy.logwarn_throttle(2.0, "Skipping grasp selection: no candidate grasps received.")
            return

        input_direction = self._resolve_input_direction(now)
        score_array, selected_index = self._score_candidates(
            ee_pose=self.latest_ee_pose,
            candidates=self.latest_candidates,
            input_direction=input_direction,
        )

        if selected_index < 0:
            return

        selected_candidate = self.latest_candidates.grasps[selected_index]
        selected_pose = PoseStamped()
        selected_pose.header = self.latest_candidates.header
        if selected_pose.header.stamp == rospy.Time():
            selected_pose.header.stamp = now
        selected_pose.pose = selected_candidate.pose

        self.score_pub.publish(score_array)
        self.selected_grasp_pub.publish(selected_candidate)
        self.selected_pose_pub.publish(selected_pose)

    def _resolve_input_direction(self, now: rospy.Time) -> np.ndarray:
        if self.latest_input_stamp is None:
            return np.zeros(3, dtype=np.float64)

        if (now - self.latest_input_stamp).to_sec() > self.input_timeout:
            return np.zeros(3, dtype=np.float64)

        unit_direction, norm = _normalize(self.latest_input_vector)
        if norm < self.min_input_norm:
            return np.zeros(3, dtype=np.float64)
        return unit_direction

    def _prune_input_history(self):
        now = rospy.Time.now()
        while self.input_history and (now - self.input_history[0][0]).to_sec() > self.obs_window_sec:
            self.input_history.popleft()

    def _score_candidates(
        self,
        ee_pose: PoseStamped,
        candidates: GraspCandidateArray,
        input_direction: np.ndarray,
    ) -> Tuple[GraspScoreArray, int]:
        ee_position = _as_numpy_xyz(
            ee_pose.pose.position.x,
            ee_pose.pose.position.y,
            ee_pose.pose.position.z,
        )

        # Prune old candidates and maintain deterministic keys
        current_keys = []
        for index, candidate in enumerate(candidates.grasps):
            candidate_key = str(candidate.grasp_id) if candidate.grasp_id is not None else f"idx_{index}"
            current_keys.append(candidate_key)
            if candidate_key not in self.log_beliefs:
                initial_prior = 1.0 / max(len(candidates.grasps), 1)
                self.log_beliefs[candidate_key] = math.log(initial_prior)
        self.log_beliefs = {k: v for k, v in self.log_beliefs.items() if k in current_keys}

        score_msg = GraspScoreArray()
        score_msg.header = candidates.header
        score_msg.selected_index = -1

        best_index = -1
        best_total = -math.inf

        for index, candidate in enumerate(candidates.grasps):
            candidate_key = str(candidate.grasp_id) if candidate.grasp_id is not None else f"idx_{index}"
            grasp_position = _as_numpy_xyz(
                candidate.pose.position.x,
                candidate.pose.position.y,
                candidate.pose.position.z,
            )

            direction_to_grasp, distance = _normalize(grasp_position - ee_position)
            alignment = float(np.dot(input_direction, direction_to_grasp))
            if np.allclose(input_direction, 0.0):
                alignment = 0.0
            if self.clamp_negative_alignment:
                alignment = max(0.0, alignment)

            normalized_grasp_score = float(np.clip(candidate.grasp_score, 0.0, 1.0))
            feasibility_score = 1.0 if candidate.feasible else 0.0
            z_world = float(candidate.pose.position.z)
            if self.high_z_penalty_scale > 1e-9:
                high_z_penalty = -max(0.0, z_world - self.high_z_penalty_start) / self.high_z_penalty_scale
            else:
                high_z_penalty = 0.0

            # Likelihood-based incremental evidence from recent joystick history (MAP-style)
            evidence = 0.0
            if self.input_history:
                for _, commanded in self.input_history:
                    residual = commanded - direction_to_grasp
                    evidence += -0.5 * float(np.dot(residual, self.noise_cov_inv.dot(residual)))

            self.log_beliefs[candidate_key] = (
                self.log_beliefs.get(candidate_key, 0.0) * self.belief_decay
                + self.w_likelihood * evidence
            )

            total_score = (
                self.log_beliefs[candidate_key]
                + self.w_alignment * alignment
                + self.w_grasp_score * normalized_grasp_score
                + self.w_feasibility * feasibility_score
                + self.w_high_z_penalty * high_z_penalty
            )

            grasp_score_msg = GraspScore()
            grasp_score_msg.grasp_id = candidate.grasp_id
            grasp_score_msg.candidate_index = index
            grasp_score_msg.total_score = total_score
            grasp_score_msg.alignment_score = alignment
            grasp_score_msg.grasp_score = normalized_grasp_score
            grasp_score_msg.feasibility_score = feasibility_score
            grasp_score_msg.distance_to_grasp = distance
            grasp_score_msg.feasible = candidate.feasible
            score_msg.scores.append(grasp_score_msg)

            if total_score > best_total:
                best_total = total_score
                best_index = index

        score_msg.selected_index = best_index
        return score_msg, best_index


if __name__ == "__main__":
    SharedAutonomyGraspSelector()
    rospy.spin()
