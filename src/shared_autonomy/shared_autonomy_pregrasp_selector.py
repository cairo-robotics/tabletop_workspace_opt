#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared-autonomy selector for fixed pregrasp candidates.

Loads fixed grasp candidates from YAML and selects the most likely intended
pregrasp based on joystick/end-effector motion evidence. This keeps the MAP-like
idea from the draft selector but uses only standard ROS messages so it can slot
into the existing launch stack.
"""

import math
import os
from collections import deque

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PointStamped, PoseStamped
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEPoseGoals, EEVelGoals
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from visualization_msgs.msg import Marker, MarkerArray


def _as_numpy_xyz(x, y, z):
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64), 0.0
    return vec / norm, norm


class SharedAutonomyPregraspSelector:
    def __init__(self):
        rospy.init_node("shared_autonomy_pregrasp_selector")

        self.base_frame = rospy.get_param("~base_frame", "base")
        self.fixed_grasp_stage = rospy.get_param("~fixed_grasp_stage", "pregrasp_pose")
        self.fixed_grasp_yaml = self._resolve_yaml_path()

        self.ee_pose_timeout = float(rospy.get_param("~ee_pose_timeout_sec", 0.5))
        self.input_timeout = float(rospy.get_param("~input_timeout_sec", 0.5))
        self.min_input_norm = float(rospy.get_param("~min_input_norm", 1e-4))
        self.obs_window_sec = float(rospy.get_param("~obs_window_sec", 0.5))

        self.w_alignment = float(rospy.get_param("~w_alignment", 0.6))
        self.w_prior = float(rospy.get_param("~w_prior", 0.25))
        self.w_distance = float(rospy.get_param("~w_distance", 1.2))
        self.w_grasp_score = float(rospy.get_param("~w_grasp_score", 0.3))
        self.w_feasibility = float(rospy.get_param("~w_feasibility", 0.1))
        self.w_high_z_penalty = float(rospy.get_param("~w_high_z_penalty", 0.25))
        self.w_likelihood = float(rospy.get_param("~w_likelihood", 1.0))
        self.belief_decay = float(rospy.get_param("~belief_decay", 0.90))
        self.distance_scale = float(rospy.get_param("~distance_scale", 0.12))

        self.clamp_negative_alignment = bool(rospy.get_param("~clamp_negative_alignment", False))
        self.high_z_penalty_start = float(rospy.get_param("~high_z_penalty_start", 0.92))
        self.high_z_penalty_scale = float(rospy.get_param("~high_z_penalty_scale", 0.08))

        self.intent_action_threshold = float(rospy.get_param("~intent_action_threshold", 0.85))
        self.use_goal_orientation = bool(rospy.get_param("~use_goal_orientation", True))
        self.action_z_offset = float(rospy.get_param("~action_z_offset", 0.0))
        self.enable_locking = bool(rospy.get_param("~enable_locking", False))
        self.select_threshold = float(rospy.get_param("~select_threshold", 0.80))
        self.release_threshold = float(rospy.get_param("~release_threshold", 0.55))
        self.hold_time_sec = float(rospy.get_param("~hold_time_sec", 0.30))

        noise_cov_diag = rospy.get_param("~noise_cov_diag", [0.1, 0.1, 0.1])
        noise_cov = np.diag(np.array(noise_cov_diag, dtype=np.float64))
        self.noise_cov_inv = np.linalg.inv(noise_cov)

        self.endpoint_topic = rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        self.ee_vel_goals_topic = rospy.get_param("~ee_vel_goals_topic", "relaxed_ik/ee_vel_goals")
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "shared_autonomy")).strip()

        self.latest_ee_pose = None
        self.latest_ee_stamp = None
        self.latest_input_vector = np.zeros(3, dtype=np.float64)
        self.latest_input_stamp = None
        self.input_history = deque()

        self.candidates = self._load_candidates()
        self.log_beliefs = {}
        initial_prior = 1.0 / max(len(self.candidates), 1)
        for candidate in self.candidates:
            self.log_beliefs[candidate["grasp_id"]] = math.log(initial_prior)

        self.commanded_goal_label = None
        self.locked_goal_label = None
        self.locked_goal_index = -1
        self.candidate_enter_times = {}

        self.pub_dist = rospy.Publisher("~distribution", Float32MultiArray, queue_size=1)
        self.pub_top = rospy.Publisher("~top_goal", String, queue_size=1)
        self.pub_toppose = rospy.Publisher("~top_pose", PoseStamped, queue_size=1)
        self.pub_current_tracker_point = rospy.Publisher("~current_tracker_point", PointStamped, queue_size=1)
        self.pub_ee_goal = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.pub_score_text = rospy.Publisher("~score_debug", String, queue_size=1)
        self.pub_markers = rospy.Publisher("~selection_markers", MarkerArray, queue_size=1)

        rospy.Subscriber(self.endpoint_topic, EndpointState, self._endpoint_cb, queue_size=10)
        rospy.Subscriber(self.ee_vel_goals_topic, EEVelGoals, self._ee_vel_goals_cb, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0 / 20.0), self._timer_cb)
        self.guard_timer = rospy.Timer(rospy.Duration(0.5), self._control_mode_guard)

        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] ready. candidates=%d yaml=%s endpoint=%s ee_vel_goals=%s",
            len(self.candidates),
            self.fixed_grasp_yaml,
            self.endpoint_topic,
            self.ee_vel_goals_topic,
        )
        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] locking=%s select_threshold=%.2f release_threshold=%.2f hold_time=%.2fs",
            "on" if self.enable_locking else "off",
            self.select_threshold,
            self.release_threshold,
            self.hold_time_sec,
        )

    def _control_mode_guard(self, _event):
        current_mode = str(rospy.get_param("/tabletop_workspace_opt/control_mode", "")).strip()
        if current_mode and current_mode != self.required_control_mode:
            rospy.logwarn(
                "[shared_autonomy_pregrasp_selector] control_mode=%s but required=%s. Shutting down to avoid command conflicts.",
                current_mode,
                self.required_control_mode,
            )
            rospy.signal_shutdown("control mode mismatch")

    def _resolve_yaml_path(self):
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_yaml = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")
        return os.path.expanduser(rospy.get_param("~fixed_grasp_yaml", default_yaml))

    def _pose_dict_to_pose_stamped(self, pose_dict):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.base_frame
        pose_msg.pose.position.x = float(pose_dict["position"][0])
        pose_msg.pose.position.y = float(pose_dict["position"][1])
        pose_msg.pose.position.z = float(pose_dict["position"][2])
        pose_msg.pose.orientation.x = float(pose_dict["orientation"][0])
        pose_msg.pose.orientation.y = float(pose_dict["orientation"][1])
        pose_msg.pose.orientation.z = float(pose_dict["orientation"][2])
        pose_msg.pose.orientation.w = float(pose_dict["orientation"][3])
        return pose_msg

    def _load_candidates(self):
        if not os.path.exists(self.fixed_grasp_yaml):
            raise RuntimeError(f"Fixed grasp YAML not found: {self.fixed_grasp_yaml}")

        with open(self.fixed_grasp_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        yaml_frame = str(data.get("frame_id", self.base_frame))
        if yaml_frame != self.base_frame:
            rospy.logwarn(
                "fixed_grasp_yaml frame_id=%s but selector base_frame=%s. Current implementation assumes they match.",
                yaml_frame,
                self.base_frame,
            )

        candidates = []
        for index, grasp in enumerate(data.get("grasps", [])):
            if not isinstance(grasp, dict):
                continue
            grasp_id = str(grasp.get("grasp_id", f"candidate_{index}")).strip()
            pose_dict = grasp.get(self.fixed_grasp_stage)
            if not isinstance(pose_dict, dict):
                continue
            position = pose_dict.get("position", [])
            orientation = pose_dict.get("orientation", [])
            if len(position) != 3 or len(orientation) != 4:
                rospy.logwarn("Skipping malformed candidate '%s'.", grasp_id)
                continue

            candidates.append(
                {
                    "grasp_id": grasp_id,
                    "object_name": str(grasp.get("object_name", grasp_id)),
                    "position": tuple(float(v) for v in position),
                    "pose": self._pose_dict_to_pose_stamped(pose_dict),
                    "grasp_score": float(grasp.get("grasp_score", 1.0)),
                    "feasible": bool(grasp.get("feasible", True)),
                }
            )

        if not candidates:
            raise RuntimeError(f"No valid fixed pregrasp candidates found in {self.fixed_grasp_yaml}")
        return candidates

    def _endpoint_cb(self, msg):
        self.latest_ee_pose = msg.pose
        self.latest_ee_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

        tracker_point = PointStamped()
        tracker_point.header.stamp = self.latest_ee_stamp
        tracker_point.header.frame_id = self.base_frame
        tracker_point.point = msg.pose.position
        self.pub_current_tracker_point.publish(tracker_point)

    def _ee_vel_goals_cb(self, msg):
        if not msg.ee_vels:
            return
        twist = msg.ee_vels[0]
        self.latest_input_vector = _as_numpy_xyz(
            twist.linear.x, twist.linear.y, twist.linear.z
        )
        self.latest_input_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        self.input_history.append((self.latest_input_stamp, self.latest_input_vector.copy()))
        self._prune_input_history()

    def _prune_input_history(self):
        now = rospy.Time.now()
        while self.input_history and (now - self.input_history[0][0]).to_sec() > self.obs_window_sec:
            self.input_history.popleft()

    def _resolve_input_direction(self, now):
        if self.latest_input_stamp is None:
            return np.zeros(3, dtype=np.float64)
        if (now - self.latest_input_stamp).to_sec() > self.input_timeout:
            return np.zeros(3, dtype=np.float64)
        unit_direction, norm = _normalize(self.latest_input_vector)
        if norm < self.min_input_norm:
            return np.zeros(3, dtype=np.float64)
        return unit_direction

    def _timer_cb(self, _event):
        if self.latest_ee_pose is None or self.latest_ee_stamp is None:
            return

        now = rospy.Time.now()
        if (now - self.latest_ee_stamp).to_sec() > self.ee_pose_timeout:
            rospy.logwarn_throttle(2.0, "[shared_autonomy_pregrasp_selector] skipping: ee pose is stale.")
            return

        input_direction = self._resolve_input_direction(now)
        score_msg, raw_best_index, raw_best_prob, debug_scores = self._score_candidates(self.latest_ee_pose, input_direction)
        if raw_best_index < 0:
            return

        if self.enable_locking:
            best_index, best_prob = self._update_locked_selection(raw_best_index, raw_best_prob, debug_scores, now)
            if best_index < 0:
                return
        else:
            self.locked_goal_label = None
            self.locked_goal_index = -1
            best_index, best_prob = raw_best_index, raw_best_prob

        selected = self.candidates[best_index]
        top_pose = PoseStamped()
        top_pose.header.frame_id = self.base_frame
        top_pose.header.stamp = now
        top_pose.pose = selected["pose"].pose

        self.pub_dist.publish(score_msg)
        self.pub_top.publish(String(data=selected["grasp_id"]))
        self.pub_toppose.publish(top_pose)
        self.pub_markers.publish(self._build_selection_markers(debug_scores, best_index, now))
        self.pub_score_text.publish(
            String(
                data=(
                    f"raw={self.candidates[raw_best_index]['grasp_id']} raw_prob={raw_best_prob:.3f} "
                    f"selected={selected['grasp_id']} selected_prob={best_prob:.3f} "
                    f"locked={self.locked_goal_label if self.locked_goal_label else 'none'} "
                    f"locking={'on' if self.enable_locking else 'off'}"
                )
            )
        )

        if best_prob >= self.intent_action_threshold and selected["grasp_id"] != self.commanded_goal_label:
            goal_msg = EEPoseGoals()
            goal_msg.header = top_pose.header
            goal_msg.ee_poses.append(top_pose.pose)
            if not self.use_goal_orientation:
                goal_msg.ee_poses[0].orientation.x = 0.0
                goal_msg.ee_poses[0].orientation.y = 0.0
                goal_msg.ee_poses[0].orientation.z = 0.0
                goal_msg.ee_poses[0].orientation.w = 1.0
            goal_msg.ee_poses[0].position.z += self.action_z_offset
            self.pub_ee_goal.publish(goal_msg)
            self.commanded_goal_label = selected["grasp_id"]
            rospy.loginfo(
                "[shared_autonomy_pregrasp_selector] selected %s with prob=%.3f",
                selected["grasp_id"],
                best_prob,
            )

    def _update_locked_selection(self, raw_best_index, raw_best_prob, debug_scores, now):
        raw_best_label = self.candidates[raw_best_index]["grasp_id"]

        if self.locked_goal_label is not None:
            locked_index = next(
                (idx for idx, candidate in enumerate(self.candidates) if candidate["grasp_id"] == self.locked_goal_label),
                -1,
            )
            if locked_index >= 0:
                locked_prob = debug_scores[locked_index].get("probability", 0.0)
                if locked_prob >= self.release_threshold:
                    self.locked_goal_index = locked_index
                    return locked_index, locked_prob
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] released lock on %s (prob=%.3f < %.3f)",
                    self.locked_goal_label,
                    locked_prob,
                    self.release_threshold,
                )
            self.locked_goal_label = None
            self.locked_goal_index = -1

        if raw_best_prob < self.select_threshold:
            self.candidate_enter_times.pop(raw_best_label, None)
            return raw_best_index, raw_best_prob

        entered_at = self.candidate_enter_times.get(raw_best_label)
        if entered_at is None:
            self.candidate_enter_times = {raw_best_label: now}
            return raw_best_index, raw_best_prob

        held_for = (now - entered_at).to_sec()
        if held_for < self.hold_time_sec:
            return raw_best_index, raw_best_prob

        self.locked_goal_label = raw_best_label
        self.locked_goal_index = raw_best_index
        self.candidate_enter_times = {raw_best_label: entered_at}
        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] locked onto %s after %.2fs above threshold (prob=%.3f)",
            raw_best_label,
            held_for,
            raw_best_prob,
        )
        return raw_best_index, raw_best_prob

    def _score_candidates(self, ee_pose, input_direction):
        ee_position = _as_numpy_xyz(
            ee_pose.position.x,
            ee_pose.position.y,
            ee_pose.position.z,
        )

        raw_scores = []
        best_index = -1
        best_total = -math.inf

        for index, candidate in enumerate(self.candidates):
            candidate_key = candidate["grasp_id"]
            grasp_position = np.array(candidate["position"], dtype=np.float64)

            direction_to_grasp, distance = _normalize(grasp_position - ee_position)
            alignment = float(np.dot(input_direction, direction_to_grasp))
            if np.allclose(input_direction, 0.0):
                alignment = 0.0
            if self.clamp_negative_alignment:
                alignment = max(0.0, alignment)

            normalized_grasp_score = float(np.clip(candidate["grasp_score"], 0.0, 1.0))
            feasibility_score = 1.0 if candidate["feasible"] else 0.0
            if self.distance_scale > 1e-9:
                distance_score = -distance / self.distance_scale
            else:
                distance_score = -distance
            z_world = float(candidate["position"][2])
            if self.high_z_penalty_scale > 1e-9:
                high_z_penalty = -max(0.0, z_world - self.high_z_penalty_start) / self.high_z_penalty_scale
            else:
                high_z_penalty = 0.0

            evidence = 0.0
            if self.input_history:
                for _, commanded in self.input_history:
                    commanded_unit, commanded_norm = _normalize(commanded)
                    if commanded_norm < self.min_input_norm:
                        continue
                    residual = commanded_unit - direction_to_grasp
                    evidence += -0.5 * float(np.dot(residual, self.noise_cov_inv.dot(residual)))

            self.log_beliefs[candidate_key] = (
                self.log_beliefs.get(candidate_key, 0.0) * self.belief_decay
                + self.w_likelihood * evidence
            )

            total_score = (
                self.w_prior * self.log_beliefs[candidate_key]
                + self.w_alignment * alignment
                + self.w_distance * distance_score
                + self.w_grasp_score * normalized_grasp_score
                + self.w_feasibility * feasibility_score
                + self.w_high_z_penalty * high_z_penalty
            )

            raw_scores.append({
                "grasp_id": candidate_key,
                "total_score": total_score,
                "distance": distance,
                "distance_score": distance_score,
                "position": candidate["position"],
                "pose": candidate["pose"],
                "alignment": alignment,
            })

            if total_score > best_total:
                best_total = total_score
                best_index = index

        if not raw_scores:
            return Float32MultiArray(), -1, 0.0

        max_score = max(item["total_score"] for item in raw_scores)
        exp_scores = [math.exp(item["total_score"] - max_score) for item in raw_scores]
        z = sum(exp_scores)
        probs = [score / z for score in exp_scores]
        for idx, prob in enumerate(probs):
            raw_scores[idx]["probability"] = prob

        dist_msg = Float32MultiArray()
        dist_msg.layout.dim.append(
            MultiArrayDimension(label="grasps", size=len(probs), stride=len(probs))
        )
        dist_msg.data = probs

        best_prob = probs[best_index] if 0 <= best_index < len(probs) else 0.0
        return dist_msg, best_index, best_prob, raw_scores

    def _build_selection_markers(self, debug_scores, best_index, stamp):
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        text_height = 0.12
        for index, item in enumerate(debug_scores):
            label_marker = Marker()
            label_marker.header.frame_id = self.base_frame
            label_marker.header.stamp = stamp
            label_marker.ns = "selector_scores"
            label_marker.id = index
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position.x = float(item["position"][0])
            label_marker.pose.position.y = float(item["position"][1])
            label_marker.pose.position.z = float(item["position"][2]) + text_height + 0.03 * (index % 3)
            label_marker.pose.orientation.w = 1.0
            label_marker.scale.z = 0.05
            if index == best_index:
                label_marker.color.r = 1.0
                label_marker.color.g = 0.95
                label_marker.color.b = 0.2
                label_marker.color.a = 1.0
            else:
                label_marker.color.r = 1.0
                label_marker.color.g = 1.0
                label_marker.color.b = 1.0
                label_marker.color.a = 0.95
            label_marker.text = (
                f"{item['grasp_id']}\n"
                f"p={item.get('probability', 0.0):.2f} "
                f"s={item['total_score']:.2f}\n"
                f"d={item['distance']:.2f}"
            )
            markers.markers.append(label_marker)

            if index == best_index:
                selected_marker = Marker()
                selected_marker.header.frame_id = self.base_frame
                selected_marker.header.stamp = stamp
                selected_marker.ns = "selector_selected"
                selected_marker.id = 1000 + index
                selected_marker.type = Marker.SPHERE
                selected_marker.action = Marker.ADD
                selected_marker.pose = item["pose"].pose
                selected_marker.scale.x = 0.05
                selected_marker.scale.y = 0.05
                selected_marker.scale.z = 0.05
                selected_marker.color.r = 1.0
                selected_marker.color.g = 0.95
                selected_marker.color.b = 0.2
                selected_marker.color.a = 0.9
                markers.markers.append(selected_marker)

                pose_marker = Marker()
                pose_marker.header.frame_id = self.base_frame
                pose_marker.header.stamp = stamp
                pose_marker.ns = "selector_selected_pose"
                pose_marker.id = 2000 + index
                pose_marker.type = Marker.ARROW
                pose_marker.action = Marker.ADD
                pose_marker.pose = item["pose"].pose
                pose_marker.scale.x = 0.11
                pose_marker.scale.y = 0.018
                pose_marker.scale.z = 0.018
                pose_marker.color.r = 1.0
                pose_marker.color.g = 0.95
                pose_marker.color.b = 0.2
                pose_marker.color.a = 0.95
                markers.markers.append(pose_marker)

        return markers


def main():
    SharedAutonomyPregraspSelector()
    rospy.spin()


if __name__ == "__main__":
    main()
