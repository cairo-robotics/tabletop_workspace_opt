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
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from visualization_msgs.msg import Marker, MarkerArray


def _as_numpy_xyz(x, y, z):
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64), 0.0
    return vec / norm, norm


def _normalize_quaternion(quat):
    quat = np.array(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _quat_angular_distance_rad(quat_a, quat_b):
    quat_a = _normalize_quaternion(quat_a)
    quat_b = _normalize_quaternion(quat_b)
    dot = float(np.clip(abs(np.dot(quat_a, quat_b)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


class SharedAutonomyPregraspSelector:
    def __init__(self):
        rospy.init_node("shared_autonomy_pregrasp_selector")

        self.base_frame = rospy.get_param("~base_frame", "base")
        self.fixed_grasp_stage = rospy.get_param("~fixed_grasp_stage", "pregrasp_pose")
        self.fixed_grasp_yaml = self._resolve_yaml_path()

        self.ee_pose_timeout = float(rospy.get_param("~ee_pose_timeout_sec", 0.5))
        self.input_timeout = float(rospy.get_param("~input_timeout_sec", 0.5))
        self.min_input_norm = float(rospy.get_param("~min_input_norm", 1e-4))
        self.obs_window_sec = float(rospy.get_param("~obs_window_sec", rospy.get_param("~window_sec", 1.2)))
        self.debug_top_k = int(rospy.get_param("~debug_top_k", 3))
        self.warmup_sec = float(rospy.get_param("~warmup_sec", 0.0))

        self.intent_method = str(rospy.get_param("~intent_method", "legacy_path")).strip().lower()
        self.beta = float(rospy.get_param("~beta", 25.0))
        self.stationary_speed_mps = float(rospy.get_param("~stationary_speed_mps", 0.03))
        self.reset_hold_sec = float(rospy.get_param("~reset_hold_sec", 2.0))
        self.w_path_efficiency = float(rospy.get_param("~w_path_efficiency", 1.0))

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
        self.auto_approach_max_speed_mps = float(rospy.get_param("~auto_approach_max_speed_mps", 0.08))
        self.auto_approach_max_angular_step = float(rospy.get_param("~auto_approach_max_angular_step", 0.08))
        self.auto_approach_position_tolerance = float(rospy.get_param("~auto_approach_position_tolerance", 0.01))
        self.auto_approach_orientation_tolerance_rad = float(rospy.get_param("~auto_approach_orientation_tolerance_rad", 0.18))
        self.pregrasp_completion_orientation_tolerance_rad = float(
            rospy.get_param("~pregrasp_completion_orientation_tolerance_rad", 0.35)
        )
        self.pregrasp_grasp_confirmation_distance_m = float(
            rospy.get_param("~pregrasp_grasp_confirmation_distance_m", 0.03)
        )
        self.orientation_align_distance_m = float(rospy.get_param("~orientation_align_distance_m", 0.10))
        self.manual_override_input_norm = float(rospy.get_param("~manual_override_input_norm", 0.35))
        self.manual_override_hold_sec = float(rospy.get_param("~manual_override_hold_sec", 1.0))
        self.require_confirmation = bool(rospy.get_param("~require_confirmation", True))
        self.confirmation_timeout_sec = float(rospy.get_param("~confirmation_timeout_sec", 3.0))
        self.require_grasp_confirmation = bool(rospy.get_param("~require_grasp_confirmation", True))
        self.grasp_confirmation_timeout_sec = float(rospy.get_param("~grasp_confirmation_timeout_sec", 0.0))
        self.keep_pregrasp_orientation_for_grasp = bool(
            rospy.get_param("~keep_pregrasp_orientation_for_grasp", True)
        )
        self.pause_after_grasp_complete = bool(rospy.get_param("~pause_after_grasp_complete", False))
        self.pause_after_grasp_complete_label = str(
            rospy.get_param("~pause_after_grasp_complete_label", "")
        ).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.grasp_close_button_index = int(rospy.get_param("~grasp_close_button_index", 0))
        self.confirm_button_index = int(rospy.get_param("~confirm_button_index", 2))
        self.cancel_button_index = int(rospy.get_param("~cancel_button_index", 3))
        self.enable_locking = bool(rospy.get_param("~enable_locking", False))
        self.select_threshold = float(rospy.get_param("~select_threshold", 0.80))
        self.release_threshold = float(rospy.get_param("~release_threshold", 0.55))
        self.hold_time_sec = float(rospy.get_param("~hold_time_sec", 0.30))

        noise_cov_diag = rospy.get_param("~noise_cov_diag", [0.1, 0.1, 0.1])
        noise_cov = np.diag(np.array(noise_cov_diag, dtype=np.float64))
        self.noise_cov_inv = np.linalg.inv(noise_cov)

        self.endpoint_topic = rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        self.ee_vel_goals_topic = rospy.get_param("~ee_vel_goals_topic", "relaxed_ik/ee_vel_goals")
        self.joy_topic = rospy.get_param("~joy_topic", "joy")
        self.joy_deadzone = float(rospy.get_param("~joy_deadzone", 0.1))
        self.joy_linear_axes = list(rospy.get_param("~joy_linear_axes", [1, 0, 4]))
        self.joy_axis_signs = list(rospy.get_param("~joy_axis_signs", [1.0, 1.0, 1.0]))
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "shared_autonomy")).strip()

        self.latest_ee_pose = None
        self.latest_ee_stamp = None
        self.latest_input_vector = np.zeros(3, dtype=np.float64)
        self.latest_input_stamp = None
        self.latest_input_source = "none"
        self.latest_buttons = []
        self.input_history = deque()
        self.ee_history = deque()
        self.start_time = rospy.get_time()
        self.reach_start_position = None
        self.last_move_time = None

        self.candidates = self._load_candidates()
        self.candidate_map = {candidate["grasp_id"]: candidate for candidate in self.candidates}
        self.log_beliefs = {}
        initial_prior = 1.0 / max(len(self.candidates), 1)
        for candidate in self.candidates:
            self.log_beliefs[candidate["grasp_id"]] = math.log(initial_prior)

        self.commanded_goal_label = None
        self.last_auto_command_time = None
        self.manual_override_until = None
        self.pending_goal_label = None
        self.pending_goal_stage = None
        self.pending_goal_prob = 0.0
        self.pending_goal_since = None
        self.approved_goal_label = None
        self.approved_goal_stage = None
        self.completed_pregrasp_label = None
        self.autonomy_paused = False
        self.locked_goal_label = None
        self.locked_goal_index = -1
        self.candidate_enter_times = {}

        self.pub_dist = rospy.Publisher("~distribution", Float32MultiArray, queue_size=1)
        self.pub_top = rospy.Publisher("~top_goal", String, queue_size=1)
        self.pub_toppose = rospy.Publisher("~top_pose", PoseStamped, queue_size=1)
        self.pub_current_tracker_point = rospy.Publisher("~current_tracker_point", PointStamped, queue_size=1)
        self.pub_ee_goal = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.pub_score_text = rospy.Publisher("~score_debug", String, queue_size=1)
        self.pub_confirmation_prompt = rospy.Publisher("~confirmation_prompt", String, queue_size=1)
        self.pub_execution_state = rospy.Publisher("~execution_state", String, queue_size=1)
        self.pub_markers = rospy.Publisher("~selection_markers", MarkerArray, queue_size=1)

        rospy.Subscriber(self.endpoint_topic, EndpointState, self._endpoint_cb, queue_size=10)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        rospy.Subscriber(self.ee_vel_goals_topic, EEVelGoals, self._ee_vel_goals_cb, queue_size=10)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_grasp_label_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / 20.0), self._timer_cb)
        self.guard_timer = rospy.Timer(rospy.Duration(0.5), self._control_mode_guard)

        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] ready. candidates=%d yaml=%s endpoint=%s joy=%s ee_vel_goals=%s",
            len(self.candidates),
            self.fixed_grasp_yaml,
            self.endpoint_topic,
            self.joy_topic,
            self.ee_vel_goals_topic,
        )
        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] locking=%s select_threshold=%.2f release_threshold=%.2f hold_time=%.2fs",
            "on" if self.enable_locking else "off",
            self.select_threshold,
            self.release_threshold,
            self.hold_time_sec,
        )
        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] intent_method=%s beta=%.2f stationary_speed=%.3f obs_window=%.2f auto_speed=%.3f orient_align=%.3f orient_tol=%.2fdeg pregrasp_orient_tol=%.2fdeg grasp_confirm_dist=%.3fm keep_pregrasp_orient=%s manual_override=%.2f/%.2fs confirmation=%s grasp_confirmation=%s timeout=%.1fs grasp_timeout=%.1fs",
            self.intent_method,
            self.beta,
            self.stationary_speed_mps,
            self.obs_window_sec,
            self.auto_approach_max_speed_mps,
            self.orientation_align_distance_m,
            math.degrees(self.auto_approach_orientation_tolerance_rad),
            math.degrees(self.pregrasp_completion_orientation_tolerance_rad),
            self.pregrasp_grasp_confirmation_distance_m,
            "on" if self.keep_pregrasp_orientation_for_grasp else "off",
            self.manual_override_input_norm,
            self.manual_override_hold_sec,
            "on" if self.require_confirmation else "off",
            "on" if self.require_grasp_confirmation else "off",
            self.confirmation_timeout_sec,
            self.grasp_confirmation_timeout_sec,
        )
        if self.pause_after_grasp_complete:
            rospy.loginfo(
                "[shared_autonomy_pregrasp_selector] will pause autonomous inference after grasp_complete for label=%s",
                self.pause_after_grasp_complete_label if self.pause_after_grasp_complete_label else "<any>",
            )

    def _selected_grasp_label_cb(self, msg):
        selected_label = str(msg.data).strip()
        if not selected_label:
            return
        if self.pause_after_grasp_complete_label == selected_label:
            return
        self.pause_after_grasp_complete_label = selected_label
        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] updated pause-after-grasp label to %s from selector topic.",
            selected_label,
        )

    @staticmethod
    def _paired_grasp_id(grasp_id):
        if "_pregrasp_" in grasp_id:
            return grasp_id.replace("_pregrasp_", "_grasp_", 1)
        if "_grasp_" in grasp_id:
            return grasp_id.replace("_grasp_", "_pregrasp_", 1)
        return None

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

        all_grasps = [grasp for grasp in data.get("grasps", []) if isinstance(grasp, dict)]
        grasp_lookup = {
            str(grasp.get("grasp_id", f"candidate_{index}")).strip(): grasp
            for index, grasp in enumerate(all_grasps)
        }

        candidates = []
        for index, grasp in enumerate(all_grasps):
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

            paired_entry = grasp_lookup.get(self._paired_grasp_id(grasp_id) or "")
            final_grasp_pose = grasp.get("grasp_pose")
            if not isinstance(final_grasp_pose, dict) and isinstance(paired_entry, dict):
                final_grasp_pose = paired_entry.get("grasp_pose")

            candidates.append(
                {
                    "grasp_id": grasp_id,
                    "object_name": str(grasp.get("object_name", grasp_id)),
                    "position": tuple(float(v) for v in position),
                    "pose": self._pose_dict_to_pose_stamped(pose_dict),
                    "grasp_pose": self._pose_dict_to_pose_stamped(final_grasp_pose) if isinstance(final_grasp_pose, dict) else None,
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
        self._update_reach_state(self.latest_ee_stamp, msg.pose)

        tracker_point = PointStamped()
        tracker_point.header.stamp = self.latest_ee_stamp
        tracker_point.header.frame_id = self.base_frame
        tracker_point.point = msg.pose.position
        self.pub_current_tracker_point.publish(tracker_point)

    def _record_input_vector(self, vector, stamp, source):
        self.latest_input_vector = np.array(vector, dtype=np.float64)
        self.latest_input_stamp = stamp if stamp != rospy.Time() else rospy.Time.now()
        self.latest_input_source = source
        self.input_history.append((self.latest_input_stamp, self.latest_input_vector.copy()))
        self._prune_input_history()

    def _joy_cb(self, msg):
        direction = np.zeros(3, dtype=np.float64)
        for i in range(3):
            if i >= len(self.joy_linear_axes):
                break
            axis_index = int(self.joy_linear_axes[i])
            if axis_index < 0 or axis_index >= len(msg.axes):
                continue
            value = float(msg.axes[axis_index])
            if abs(value) < self.joy_deadzone:
                value = 0.0
            sign = float(self.joy_axis_signs[i]) if i < len(self.joy_axis_signs) else 1.0
            direction[i] = sign * value
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        self._record_input_vector(direction, stamp, "joy")
        self.latest_buttons = list(msg.buttons)

    def _ee_vel_goals_cb(self, msg):
        if not msg.ee_vels:
            return
        twist = msg.ee_vels[0]
        fallback_vector = _as_numpy_xyz(twist.linear.x, twist.linear.y, twist.linear.z)
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        if self.latest_input_source == "joy" and self.latest_input_stamp is not None:
            if (stamp - self.latest_input_stamp).to_sec() <= self.input_timeout:
                return
        self._record_input_vector(fallback_vector, stamp, "ee_vel_goals")

    def _prune_input_history(self):
        now = rospy.Time.now()
        while self.input_history and (now - self.input_history[0][0]).to_sec() > self.obs_window_sec:
            self.input_history.popleft()

    def _prune_reach_history(self, now_sec):
        while self.ee_history and (now_sec - self.ee_history[0][0].to_sec()) > self.obs_window_sec:
            self.ee_history.popleft()

    def _update_reach_state(self, stamp, pose):
        position = _as_numpy_xyz(pose.position.x, pose.position.y, pose.position.z)
        self.ee_history.append((stamp, position.copy()))
        self._prune_reach_history(stamp.to_sec())

        if rospy.get_time() - self.start_time < self.warmup_sec:
            return

        if len(self.ee_history) < 2:
            return

        previous_stamp, previous_position = self.ee_history[-2]
        dt = max(1e-6, (stamp - previous_stamp).to_sec())
        speed = float(np.linalg.norm(position - previous_position) / dt)

        stamp_sec = stamp.to_sec()
        if speed > self.stationary_speed_mps:
            self.last_move_time = stamp_sec
            if self.reach_start_position is None:
                self.reach_start_position = position.copy()
                rospy.loginfo("[shared_autonomy_pregrasp_selector] reach detected. Starting legacy-path inference.")
        elif self.last_move_time is not None and (stamp_sec - self.last_move_time) > self.reset_hold_sec:
            autonomous_flow_active = (
                self.approved_goal_label is not None
                or self.commanded_goal_label is not None
                or self.pending_goal_stage == "grasp"
            )
            if self.reach_start_position is not None:
                if autonomous_flow_active:
                    rospy.loginfo(
                        "[shared_autonomy_pregrasp_selector] reach ended, but keeping autonomous execution state active."
                    )
                else:
                    rospy.loginfo("[shared_autonomy_pregrasp_selector] reach ended. Resetting legacy-path state.")
            self.reach_start_position = None
            self.last_move_time = None
            if autonomous_flow_active:
                return
            self.commanded_goal_label = None
            self.last_auto_command_time = None
            self.manual_override_until = None
            self.pending_goal_label = None
            self.pending_goal_stage = None
            self.pending_goal_prob = 0.0
            self.pending_goal_since = None
            self.approved_goal_label = None
            self.approved_goal_stage = None
            self.completed_pregrasp_label = None
            self.locked_goal_label = None
            self.locked_goal_index = -1
            self.candidate_enter_times = {}

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

        if self.autonomy_paused:
            self.pub_confirmation_prompt.publish(
                String(data="Autonomous grasp paused after grasp completion; pouring sequence may take over.")
            )
            return

        now = rospy.Time.now()
        if (now - self.latest_ee_stamp).to_sec() > self.ee_pose_timeout:
            rospy.logwarn_throttle(2.0, "[shared_autonomy_pregrasp_selector] skipping: ee pose is stale.")
            return

        autonomous_flow_active = (
            self.pending_goal_label is not None
            or self.approved_goal_label is not None
            or self.commanded_goal_label is not None
        )
        if self.intent_method == "legacy_path" and self.reach_start_position is None and not autonomous_flow_active:
            return

        input_direction = self._resolve_input_direction(now)
        if self._manual_override_active(now):
            self._log_debug_summary([], input_direction)
            return
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

        status_suffix, prompt_text = self._update_confirmation_gate(selected["grasp_id"], best_prob, now)
        self.pub_confirmation_prompt.publish(String(data=prompt_text))
        self.pub_score_text.publish(
            String(
                data=(
                    f"raw={self.candidates[raw_best_index]['grasp_id']} raw_prob={raw_best_prob:.3f} "
                        f"selected={selected['grasp_id']} selected_prob={best_prob:.3f} "
                        f"locked={self.locked_goal_label if self.locked_goal_label else 'none'} "
                        f"locking={'on' if self.enable_locking else 'off'} "
                        f"manual_override={'yes' if self._manual_override_active(now) else 'no'} "
                        f"reach_active={'yes' if self.reach_start_position is not None else 'no'} "
                        f"input_source={self.latest_input_source} "
                        f"input_norm={np.linalg.norm(self.latest_input_vector):.3f}"
                        f"{status_suffix}"
                    )
            )
        )
        self._log_debug_summary(debug_scores, input_direction)

        goal_to_execute, goal_stage = self._goal_to_execute()
        if goal_to_execute is not None:
            goal_candidate = self.candidate_map.get(goal_to_execute)
            if goal_candidate is None:
                self.approved_goal_label = None
                self.approved_goal_stage = None
                self.pending_goal_label = None
                self.pending_goal_stage = None
                self.last_auto_command_time = None
                return
            target_pose_stamped = self._pose_for_stage(goal_candidate, goal_stage)
            if target_pose_stamped is None:
                self.approved_goal_label = None
                self.approved_goal_stage = None
                self.last_auto_command_time = None
                return
            goal_msg = EEPoseGoals()
            goal_msg.header = top_pose.header
            safe_goal_pose = self._build_safe_goal_pose(self.latest_ee_pose, target_pose_stamped.pose, now)
            goal_msg.ee_poses.append(safe_goal_pose)
            self.pub_ee_goal.publish(goal_msg)
            goal_position = _as_numpy_xyz(
                target_pose_stamped.pose.position.x,
                target_pose_stamped.pose.position.y,
                target_pose_stamped.pose.position.z + self.action_z_offset,
            )
            current_position = _as_numpy_xyz(
                self.latest_ee_pose.position.x,
                self.latest_ee_pose.position.y,
                self.latest_ee_pose.position.z,
            )
            remaining_distance = float(np.linalg.norm(goal_position - current_position))
            orientation_error_rad = self._orientation_error_rad(self.latest_ee_pose, target_pose_stamped.pose)
            if goal_to_execute != self.commanded_goal_label:
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] executing %s stage=%s with prob=%.3f (safe approach enabled, remaining_dist=%.3f m orient_err=%.1fdeg)",
                    goal_to_execute,
                    goal_stage,
                    best_prob,
                    remaining_distance,
                    math.degrees(orientation_error_rad),
                )
            self.commanded_goal_label = goal_to_execute
            if goal_stage == "grasp" and self._button_pressed(self.grasp_close_button_index):
                self._complete_grasp_motion(goal_to_execute)
                return
            if (
                goal_stage == "pregrasp"
                and self.require_grasp_confirmation
                and goal_candidate.get("grasp_pose") is not None
                and remaining_distance <= self.pregrasp_grasp_confirmation_distance_m
                and self.pending_goal_label is None
                and self.approved_goal_label == goal_to_execute
            ):
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] near pregrasp for %s (dist=%.3f m). Requesting grasp confirmation.",
                    goal_to_execute,
                    remaining_distance,
                )
                self.completed_pregrasp_label = goal_to_execute
                self.approved_goal_label = None
                self.approved_goal_stage = None
                self.pending_goal_label = goal_to_execute
                self.pending_goal_stage = "grasp"
                self.pending_goal_prob = 1.0
                self.pending_goal_since = now
                self.last_auto_command_time = None
                self.commanded_goal_label = None
                return
            orientation_tolerance_for_stage = self.auto_approach_orientation_tolerance_rad
            if goal_stage == "pregrasp":
                orientation_tolerance_for_stage = self.pregrasp_completion_orientation_tolerance_rad
            reached_orientation = (
                (not self.use_goal_orientation)
                or orientation_error_rad <= orientation_tolerance_for_stage
            )
            if (
                remaining_distance <= self.auto_approach_position_tolerance
                and reached_orientation
                and self.approved_goal_label == goal_to_execute
            ):
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] reached approved goal %s stage=%s (remaining_dist=%.3f m orient_err=%.1fdeg)",
                    goal_to_execute,
                    goal_stage,
                    remaining_distance,
                    math.degrees(orientation_error_rad),
                )
                self._handle_stage_completion(goal_candidate, goal_stage, now)
        else:
            self.last_auto_command_time = None
            self.commanded_goal_label = None

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

    def _manual_override_active(self, now):
        input_norm = float(np.linalg.norm(self.latest_input_vector))
        if self.latest_input_source == "joy" and input_norm >= self.manual_override_input_norm:
            self.manual_override_until = now + rospy.Duration(self.manual_override_hold_sec)
            if self.commanded_goal_label is not None or self.locked_goal_label is not None:
                rospy.loginfo_throttle(
                    0.5,
                    "[shared_autonomy_pregrasp_selector] manual override active (input_norm=%.3f). Suspending autonomous pose goals.",
                    input_norm,
                )
            self.commanded_goal_label = None
            self.last_auto_command_time = None
            self.pending_goal_label = None
            self.pending_goal_stage = None
            self.pending_goal_prob = 0.0
            self.pending_goal_since = None
            self.approved_goal_label = None
            self.approved_goal_stage = None
            self.completed_pregrasp_label = None
            self.locked_goal_label = None
            self.locked_goal_index = -1
            self.candidate_enter_times = {}
            return True

        if self.manual_override_until is None:
            return False

        if now < self.manual_override_until:
            return True

        self.manual_override_until = None
        return False

    def _button_pressed(self, button_index):
        if button_index < 0 or button_index >= len(self.latest_buttons):
            return False
        return bool(self.latest_buttons[button_index])

    def _clear_pending_goal(self):
        self.pending_goal_label = None
        self.pending_goal_stage = None
        self.pending_goal_prob = 0.0
        self.pending_goal_since = None

    def _pose_for_stage(self, candidate, goal_stage):
        if goal_stage == "grasp":
            grasp_pose = candidate.get("grasp_pose")
            if grasp_pose is None:
                return None
            if not self.keep_pregrasp_orientation_for_grasp:
                return grasp_pose

            blended_pose = PoseStamped()
            blended_pose.header.frame_id = grasp_pose.header.frame_id
            blended_pose.pose.position = grasp_pose.pose.position
            pregrasp_pose = candidate.get("pose")
            if pregrasp_pose is not None:
                blended_pose.pose.orientation = pregrasp_pose.pose.orientation
            else:
                blended_pose.pose.orientation = grasp_pose.pose.orientation
            return blended_pose
        return candidate.get("pose")


    def _handle_stage_completion(self, goal_candidate, goal_stage, now):
        goal_label = goal_candidate["grasp_id"]
        self.last_auto_command_time = None
        self.commanded_goal_label = None
        self._reset_execution_watchdog()

        if goal_stage == "pregrasp":
            self.completed_pregrasp_label = goal_label
            if self.require_grasp_confirmation and goal_candidate.get("grasp_pose") is not None:
                self.approved_goal_label = None
                self.approved_goal_stage = None
                self.pending_goal_label = goal_label
                self.pending_goal_stage = "grasp"
                self.pending_goal_prob = 1.0
                self.pending_goal_since = now
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] pregrasp reached for %s. Awaiting confirmation for final grasp motion.",
                    goal_label,
                )
                return

        self.approved_goal_label = None
        self.approved_goal_stage = None

    def _complete_grasp_motion(self, goal_label):
        execution_label = goal_label
        paired_goal_label = self._paired_grasp_id(goal_label)
        if paired_goal_label and "_pregrasp_" in goal_label:
            execution_label = paired_goal_label

        self.approved_goal_label = None
        self.approved_goal_stage = None
        self.commanded_goal_label = None
        self.last_auto_command_time = None
        self.pending_goal_label = None
        self.pending_goal_stage = None
        self.pending_goal_prob = 0.0
        self.pending_goal_since = None
        self.completed_pregrasp_label = goal_label
        self.pub_confirmation_prompt.publish(
            String(data=f"Grasp confirmed for {execution_label}. Autonomous grasp motion ended.")
        )
        self.pub_execution_state.publish(String(data=f"grasp_complete:{execution_label}"))
        rospy.loginfo(
            "[shared_autonomy_pregrasp_selector] grasp manually confirmed for %s. Autonomous grasp motion ended.",
            execution_label,
        )
        if self.pause_after_grasp_complete:
            if (
                not self.pause_after_grasp_complete_label
                or self.pause_after_grasp_complete_label == execution_label
            ):
                self.autonomy_paused = True
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] autonomous inference paused after grasp completion for %s.",
                    execution_label,
                )

    def _orientation_error_rad(self, current_pose, target_pose):
        return _quat_angular_distance_rad(
            [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ],
            [
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w,
            ],
        )

    def _update_confirmation_gate(self, selected_goal_label, selected_prob, now):
        if not self.require_confirmation:
            if selected_prob >= self.intent_action_threshold:
                self.approved_goal_label = selected_goal_label
                self.approved_goal_stage = "pregrasp"
                return " confirmation=off", ""
            self.approved_goal_label = None
            self.approved_goal_stage = None
            return " confirmation=off", ""

        if self._button_pressed(self.cancel_button_index):
            if self.pending_goal_label is not None or self.approved_goal_label is not None:
                rospy.loginfo("[shared_autonomy_pregrasp_selector] operator cancelled autonomous execution.")
            self.approved_goal_label = None
            self.approved_goal_stage = None
            self.commanded_goal_label = None
            self.last_auto_command_time = None
            if self.pending_goal_stage == "grasp":
                self.completed_pregrasp_label = self.pending_goal_label
            self._clear_pending_goal()
            return " pending=cancelled", "Execution cancelled."

        if self.pending_goal_label is not None:
            timeout_sec = self.grasp_confirmation_timeout_sec if self.pending_goal_stage == "grasp" else self.confirmation_timeout_sec
            timed_confirmation = timeout_sec > 1e-6
            if timed_confirmation and self.pending_goal_since is not None and (now - self.pending_goal_since).to_sec() > timeout_sec:
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] confirmation timeout for %s after %.1fs",
                    self.pending_goal_label,
                    timeout_sec,
                )
                if self.pending_goal_stage == "grasp":
                    self.completed_pregrasp_label = self.pending_goal_label
                self._clear_pending_goal()
                return " pending=timeout", "Confirmation timed out."

            if self._button_pressed(self.confirm_button_index):
                self.approved_goal_label = self.pending_goal_label
                self.approved_goal_stage = self.pending_goal_stage or "pregrasp"
                rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] operator approved execution to %s stage=%s",
                    self.approved_goal_label,
                    self.approved_goal_stage,
                )
                self._clear_pending_goal()
                return (
                    f" approved={self.approved_goal_label}:{self.approved_goal_stage}",
                    f"Executing {self.approved_goal_stage} for {self.approved_goal_label}.",
                )

            elapsed = 0.0 if self.pending_goal_since is None else (now - self.pending_goal_since).to_sec()
            prompt_stage = self.pending_goal_stage or "pregrasp"
            if timed_confirmation:
                remaining = max(0.0, timeout_sec - elapsed)
                timeout_text = f"{remaining:.1f}s"
                prompt_text = f"Execute {prompt_stage} for {self.pending_goal_label}? X=yes Y=no ({remaining:.1f}s)"
            else:
                timeout_text = "wait"
                prompt_text = f"Execute {prompt_stage} for {self.pending_goal_label}? X=yes Y=no"
            return (
                f" pending={self.pending_goal_label}:{prompt_stage} p={self.pending_goal_prob:.3f} "
                f"confirm=X cancel=Y timeout={timeout_text}",
                prompt_text,
            )

        if self.approved_goal_label is not None:
            stage = self.approved_goal_stage or "pregrasp"
            return f" approved={self.approved_goal_label}:{stage}", f"Executing {stage} for {self.approved_goal_label}."

        if self.completed_pregrasp_label == selected_goal_label:
            return " pending=none", ""

        if selected_prob >= self.intent_action_threshold:
            self.pending_goal_label = selected_goal_label
            self.pending_goal_stage = "pregrasp"
            self.pending_goal_prob = selected_prob
            self.pending_goal_since = now
            rospy.loginfo(
                    "[shared_autonomy_pregrasp_selector] pending confirmation for %s stage=pregrasp (prob=%.3f). Press X to execute, Y to cancel.",
                    selected_goal_label,
                    selected_prob,
                )
            return (
                f" pending={self.pending_goal_label}:pregrasp p={self.pending_goal_prob:.3f} "
                f"confirm=X cancel=Y timeout={self.confirmation_timeout_sec:.1f}s",
                f"Execute pregrasp for {self.pending_goal_label}? X=yes Y=no ({self.confirmation_timeout_sec:.1f}s)",
            )

        return " pending=none", ""

    def _goal_to_execute(self):
        if self.require_confirmation:
            return self.approved_goal_label, self.approved_goal_stage
        if self.commanded_goal_label is not None:
            return self.commanded_goal_label, "pregrasp"
        return self.approved_goal_label, self.approved_goal_stage

    def _score_candidates(self, ee_pose, input_direction):
        ee_position = _as_numpy_xyz(
            ee_pose.position.x,
            ee_pose.position.y,
            ee_pose.position.z,
        )
        path_length = self._path_length_observed()
        has_legacy_context = self.reach_start_position is not None
        use_input_evidence = self.intent_method in ("hybrid", "input_evidence")
        use_legacy_path = self.intent_method in ("legacy_path", "hybrid") and has_legacy_context

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

            path_efficiency_score = 0.0
            if use_legacy_path:
                start_to_goal = float(np.linalg.norm(grasp_position - self.reach_start_position))
                current_to_goal = float(np.linalg.norm(grasp_position - ee_position))
                start_to_goal = max(start_to_goal, 1e-3)
                path_efficiency_score = -self.beta * ((path_length + current_to_goal) / start_to_goal)

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
            if use_input_evidence:
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
                self.w_path_efficiency * path_efficiency_score
                + self.w_prior * self.log_beliefs[candidate_key]
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
                "path_efficiency_score": path_efficiency_score,
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

    def _log_debug_summary(self, debug_scores, input_direction):
        if not debug_scores:
            rospy.loginfo_throttle(
                1.0,
                "[shared_autonomy_pregrasp_selector] manual override active=%s input_source=%s input_dir=[%.2f %.2f %.2f]",
                "yes" if self.manual_override_until is not None else "no",
                self.latest_input_source,
                float(input_direction[0]),
                float(input_direction[1]),
                float(input_direction[2]),
            )
            return
        top_scores = sorted(
            debug_scores,
            key=lambda item: item.get("probability", 0.0),
            reverse=True,
        )[: max(1, self.debug_top_k)]
        summary = " | ".join(
            f"{item['grasp_id']}:p={item.get('probability', 0.0):.2f},path={item.get('path_efficiency_score', 0.0):.2f},a={item['alignment']:.2f},d={item['distance']:.2f}"
            for item in top_scores
        )
        rospy.loginfo_throttle(
            1.0,
            "[shared_autonomy_pregrasp_selector] method=%s reach_active=%s input_source=%s input_dir=[%.2f %.2f %.2f] top=%s",
            self.intent_method,
            "yes" if self.reach_start_position is not None else "no",
            self.latest_input_source,
            float(input_direction[0]),
            float(input_direction[1]),
            float(input_direction[2]),
            summary,
        )

    def _path_length_observed(self):
        history_points = [entry[1].copy() for entry in list(self.ee_history)]
        if len(history_points) < 2:
            return 0.0
        path_length = 0.0
        for index in range(1, len(history_points)):
            previous_position = history_points[index - 1]
            current_position = history_points[index]
            path_length += float(np.linalg.norm(current_position - previous_position))
        return path_length

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
                f"s={item['total_score']:.2f} "
                f"path={item.get('path_efficiency_score', 0.0):.2f}\n"
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

    def _compute_command_dt(self, now):
        if self.last_auto_command_time is None:
            self.last_auto_command_time = now
            return 1.0 / 20.0
        dt = max(1e-3, (now - self.last_auto_command_time).to_sec())
        self.last_auto_command_time = now
        return dt

    def _build_safe_goal_pose(self, current_pose, target_pose, now):
        dt = self._compute_command_dt(now)
        current_position = _as_numpy_xyz(
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z,
        )
        target_position = _as_numpy_xyz(
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z + self.action_z_offset,
        )
        delta = target_position - current_position
        distance = float(np.linalg.norm(delta))
        max_step = max(1e-4, self.auto_approach_max_speed_mps * dt)
        if distance > max_step:
            commanded_position = current_position + (delta / distance) * max_step
        else:
            commanded_position = target_position

        commanded_pose = PoseStamped().pose
        commanded_pose.position.x = float(commanded_position[0])
        commanded_pose.position.y = float(commanded_position[1])
        commanded_pose.position.z = float(commanded_position[2])

        if not self.use_goal_orientation:
            commanded_pose.orientation.x = 0.0
            commanded_pose.orientation.y = 0.0
            commanded_pose.orientation.z = 0.0
            commanded_pose.orientation.w = 1.0
            return commanded_pose

        current_quat = _normalize_quaternion(
            [
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w,
            ]
        )
        target_quat = _normalize_quaternion(
            [
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
                target_pose.orientation.w,
            ]
        )
        if distance > self.orientation_align_distance_m:
            commanded_pose.orientation.x = float(current_quat[0])
            commanded_pose.orientation.y = float(current_quat[1])
            commanded_pose.orientation.z = float(current_quat[2])
            commanded_pose.orientation.w = float(current_quat[3])
            return commanded_pose

        dot = float(np.dot(current_quat, target_quat))
        if dot < 0.0:
            target_quat = -target_quat
            dot = -dot

        if dot > 0.9995:
            blended_quat = target_quat
        else:
            blend_alpha = min(1.0, self.auto_approach_max_angular_step / math.acos(min(1.0, max(-1.0, dot))))
            blended_quat = _normalize_quaternion(current_quat + blend_alpha * (target_quat - current_quat))

        commanded_pose.orientation.x = float(blended_quat[0])
        commanded_pose.orientation.y = float(blended_quat[1])
        commanded_pose.orientation.z = float(blended_quat[2])
        commanded_pose.orientation.w = float(blended_quat[3])
        return commanded_pose


def main():
    SharedAutonomyPregraspSelector()
    rospy.spin()


if __name__ == "__main__":
    main()
