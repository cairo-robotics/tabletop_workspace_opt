#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guarded thin-layer placement controller for sandwich-style stacking."""

import json
import math
import os

import rospy
import yaml
from geometry_msgs.msg import Pose, PoseStamped, Twist
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEPoseGoals
from sensor_msgs.msg import Joy
from std_msgs.msg import String

try:
    from intera_interface import Gripper, RobotEnable
except Exception:
    Gripper = None
    RobotEnable = None


def _normalize_quat(q):
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(v) / norm for v in q]


def _pose_from_dict(data, default_orientation=None):
    pose = Pose()
    data = data if isinstance(data, dict) else {}
    position = list(data.get("position", []))
    orientation = list(data.get("orientation", []))
    if len(position) == 3:
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
    q = _normalize_quat(
        orientation if len(orientation) == 4 else (default_orientation or [0.0, 0.0, 0.0, 1.0])
    )
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def _copy_pose(pose):
    out = Pose()
    out.position.x = float(pose.position.x)
    out.position.y = float(pose.position.y)
    out.position.z = float(pose.position.z)
    out.orientation.x = float(pose.orientation.x)
    out.orientation.y = float(pose.orientation.y)
    out.orientation.z = float(pose.orientation.z)
    out.orientation.w = float(pose.orientation.w)
    return out


def _quat_angle_rad(a, b):
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = max(0.0, min(1.0, abs(sum(float(x) * float(y) for x, y in zip(qa, qb)))))
    return 2.0 * math.acos(dot)


class ThinLayerPlacementController:
    def __init__(self):
        rospy.init_node("thin_layer_placement_controller")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_candidate_metadata = os.path.join(
            package_root,
            "config",
            "sandwich_candidate_metadata.example.yaml",
        )
        default_stack_config = os.path.join(package_root, "config", "sandwich_stack_profiles.example.yaml")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.end_effector_topic = str(rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()
        self.command_topic = str(rospy.get_param("~command_topic", "/sandwich_assembly/command")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.study_event_topic = str(rospy.get_param("~study_event_topic", "/user_study/events")).strip()
        self.execution_state_topic = str(rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.candidate_metadata_yaml = os.path.expanduser(
            rospy.get_param(
                "~candidate_metadata_yaml",
                rospy.get_param("~object_map_yaml", default_candidate_metadata),
            )
        )
        self.stack_config_yaml = os.path.expanduser(rospy.get_param("~stack_config_yaml", default_stack_config))
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "shared_autonomy")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))
        self.position_tolerance_m = float(rospy.get_param("~position_tolerance_m", 0.008))
        self.orientation_tolerance_rad = float(rospy.get_param("~orientation_tolerance_rad", 0.30))
        self.release_hold_sec = float(rospy.get_param("~release_hold_sec", 0.45))
        self.guard_step_wait_sec = float(rospy.get_param("~guard_step_wait_sec", 0.18))
        self.manual_release_button_index = int(rospy.get_param("~manual_release_button_index", 1))
        self.enable_robot_interface = bool(rospy.get_param("~enable_robot_interface", True))
        self.enable_gripper_actions = bool(rospy.get_param("~enable_gripper_actions", True))
        self.auto_prepare_gripper = bool(rospy.get_param("~auto_prepare_gripper", True))
        self.limb = str(rospy.get_param("~limb", "right")).strip()
        self.auto_trigger_on_grasp_complete = bool(rospy.get_param("~auto_trigger_on_grasp_complete", False))
        self.require_lift_before_auto_place = bool(rospy.get_param("~require_lift_before_auto_place", True))
        self.auto_place_min_lift_delta_m = float(rospy.get_param("~auto_place_min_lift_delta_m", 0.025))
        self.auto_trigger_categories = {
            str(item).strip()
            for item in list(rospy.get_param("~auto_trigger_categories", ["sandwich_bread", "sandwich_filling"]))
            if str(item).strip()
        }

        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.study_event_pub = rospy.Publisher(self.study_event_topic, String, queue_size=20)

        self.current_pose = None
        self.current_label = ""
        self.latest_buttons = []
        self.prev_buttons = []
        self.phase = "IDLE"
        self.phase_started_at = rospy.Time.now()
        self.active_stack_id = "default"
        self.active_object_name = ""
        self.active_profile = {}
        self.hover_pose = None
        self.approach_pose = None
        self.retreat_pose = None
        self.guard_target_pose = None
        self.guard_step_started_at = rospy.Time.now()
        self.guard_step_start_z = None
        self.guard_nonprogress_count = 0
        self.awaiting_manual_release = False
        self.release_event_sent = False
        self.gripper = None
        self.pending_auto_place_label = ""
        self.pending_auto_place_stack_id = "default"
        self.pending_auto_place_start_z = None

        self.label_to_meta = self._load_candidate_metadata()
        self.stack_config = self._load_stack_config()
        self.stack_states = self._build_stack_states()

        if self.enable_robot_interface and RobotEnable is not None and Gripper is not None:
            self._init_robot_interface()

        rospy.Subscriber(self.end_effector_topic, EndpointState, self._endpoint_cb, queue_size=20)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_label_cb, queue_size=10)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=20)
        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=20)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=20)
        rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate_hz, 1.0)), self._tick)
        rospy.Timer(rospy.Duration(0.5), self._guard)

        self._publish_status("idle_ready")

    def _init_robot_interface(self):
        try:
            rs = RobotEnable(False)
            rs.enable()
        except BaseException as exc:
            rospy.logwarn("[thin_layer_placement_controller] could not enable robot interface: %s", exc)
        try:
            self.gripper = Gripper(self.limb + "_gripper", calibrate=False)
            self.gripper.set_dead_zone(0.001)
            if self.auto_prepare_gripper:
                if self.gripper.has_error():
                    self.gripper.reboot()
                    rospy.sleep(1.0)
                if not self.gripper.is_calibrated():
                    self.gripper.calibrate()
                    rospy.sleep(0.5)
        except BaseException as exc:
            rospy.logwarn("[thin_layer_placement_controller] could not initialize gripper: %s", exc)
            self.gripper = None

    def _load_candidate_metadata(self):
        if not os.path.exists(self.candidate_metadata_yaml):
            rospy.logwarn(
                "[thin_layer_placement_controller] candidate metadata not found: %s",
                self.candidate_metadata_yaml,
            )
            return {}
        with open(self.candidate_metadata_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        mapping = {}
        entries = {}
        if isinstance(raw, dict):
            if isinstance(raw.get("candidate_objects"), dict):
                entries = raw.get("candidate_objects", {}) or {}
            elif isinstance(raw.get("tag_objects"), dict):
                entries = raw.get("tag_objects", {}) or {}
        for meta in entries.values():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("grasp_complete_label", "")).strip()
            if label:
                mapping[label] = dict(meta)
        return mapping

    def _load_stack_config(self):
        if not os.path.exists(self.stack_config_yaml):
            rospy.logwarn("[thin_layer_placement_controller] stack config not found: %s", self.stack_config_yaml)
            return {}
        with open(self.stack_config_yaml, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _build_stack_states(self):
        raw_stacks = self.stack_config.get("stacks", {}) if isinstance(self.stack_config, dict) else {}
        if not raw_stacks:
            raw_stacks = {"default": {}}
        states = {}
        for stack_id, raw in raw_stacks.items():
            raw = raw if isinstance(raw, dict) else {}
            anchor_pose = _pose_from_dict(raw.get("anchor_pose", {}), default_orientation=[0.0, 0.0, 0.0, 1.0])
            states[str(stack_id)] = {
                "anchor_pose": anchor_pose,
                "stack_height_m": float(raw.get("initial_stack_height_m", 0.0) or 0.0),
                "last_observed_top_z": None,
                "placed_layers": [],
            }
        return states

    def _guard(self, _event):
        mode = str(rospy.get_param("/tabletop_workspace_opt/control_mode", "")).strip()
        if mode and mode != self.required_control_mode:
            rospy.logwarn(
                "[thin_layer_placement_controller] control_mode=%s required=%s; shutting down",
                mode,
                self.required_control_mode,
            )
            rospy.signal_shutdown("control mode mismatch")

    def _endpoint_cb(self, msg):
        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.header.frame_id = self.base_frame
        stamped.pose = msg.pose
        self.current_pose = stamped

    def _selected_label_cb(self, msg):
        self.current_label = str(msg.data).strip()

    def _execution_state_cb(self, msg):
        if not self.auto_trigger_on_grasp_complete or self.phase != "IDLE":
            return
        text = str(msg.data).strip()
        if not text.startswith("grasp_complete:"):
            return
        label = text.split(":", 1)[1].strip()
        meta = self.label_to_meta.get(label, {})
        if str(meta.get("category", "")).strip() not in self.auto_trigger_categories:
            return
        if self.require_lift_before_auto_place:
            self.pending_auto_place_label = label
            self.pending_auto_place_stack_id = "default"
            self.pending_auto_place_start_z = None if self.current_pose is None else float(self.current_pose.pose.position.z)
            self._publish_status(
                "waiting_for_lift_before_place:{} dz>={:.3f}".format(
                    label,
                    self.auto_place_min_lift_delta_m,
                )
            )
            return
        self._start_place(label=label, stack_id="default")

    def _joy_cb(self, msg):
        self.prev_buttons = list(self.latest_buttons)
        self.latest_buttons = list(msg.buttons)

    def _pressed_edge(self, idx):
        cur = idx >= 0 and idx < len(self.latest_buttons) and bool(self.latest_buttons[idx])
        prev = idx >= 0 and idx < len(self.prev_buttons) and bool(self.prev_buttons[idx])
        return cur and not prev

    def _publish_status(self, text):
        rospy.loginfo("[thin_layer_placement_controller] %s", text)
        self.status_pub.publish(String(data=str(text)))

    def _publish_event(self, event_name, **fields):
        payload = {
            "event": str(event_name),
            "node": rospy.get_name(),
            "stamp": rospy.Time.now().to_sec(),
            "grasp_id": self.current_label,
            "object_name": self.active_object_name,
            "stack_id": self.active_stack_id,
        }
        for key, value in fields.items():
            payload[str(key)] = value
        self.study_event_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _command_cb(self, msg):
        raw = str(msg.data).strip()
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            rospy.logwarn("[thin_layer_placement_controller] bad command json: %s", raw)
            return
        action = str(payload.get("action", "")).strip().lower()
        if action == "reset_stack":
            self._reset_stack(str(payload.get("stack_id", "default")).strip() or "default")
            return
        if action == "set_stack_anchor":
            self._set_stack_anchor(payload)
            return
        if action == "confirm_release":
            if self.awaiting_manual_release:
                self._perform_release()
            return
        if action in ("place", "place_from_selected"):
            label = str(payload.get("grasp_label", "")).strip()
            object_name = str(payload.get("object_name", "")).strip()
            if action == "place_from_selected" and not label:
                label = self.current_label
            stack_id = str(payload.get("stack_id", "default")).strip() or "default"
            self._start_place(label=label, object_name=object_name, stack_id=stack_id)
            return
        rospy.logwarn("[thin_layer_placement_controller] unknown action: %s", action)

    def _reset_stack(self, stack_id):
        state = self.stack_states.get(stack_id)
        if state is None:
            rospy.logwarn("[thin_layer_placement_controller] unknown stack_id=%s", stack_id)
            return
        state["stack_height_m"] = 0.0
        state["last_observed_top_z"] = None
        state["placed_layers"] = []
        self._publish_event("reset_stack", stack_id=stack_id)
        self._publish_status("stack_reset:{}".format(stack_id))

    def _set_stack_anchor(self, payload):
        stack_id = str(payload.get("stack_id", "default")).strip() or "default"
        state = self.stack_states.get(stack_id)
        if state is None:
            state = {
                "anchor_pose": Pose(),
                "stack_height_m": 0.0,
                "last_observed_top_z": None,
                "placed_layers": [],
            }
            state["anchor_pose"].orientation.w = 1.0
            self.stack_states[stack_id] = state
        state["anchor_pose"] = _pose_from_dict(payload.get("anchor_pose", {}), default_orientation=[0.0, 0.0, 0.0, 1.0])
        self._publish_event("set_stack_anchor", stack_id=stack_id)
        self._publish_status("stack_anchor_updated:{}".format(stack_id))

    def _profile_for_object(self, object_name):
        profiles = self.stack_config.get("object_profiles", {}) if isinstance(self.stack_config, dict) else {}
        profile = profiles.get(object_name, {}) if isinstance(profiles, dict) else {}
        return profile if isinstance(profile, dict) else {}

    def _current_stack_state(self):
        return self.stack_states.get(self.active_stack_id, {})

    def _set_phase(self, new_phase):
        self.phase = str(new_phase)
        self.phase_started_at = rospy.Time.now()

    def _publish_goal(self, pose):
        msg = EEPoseGoals()
        msg.header.stamp = rospy.Time.now()
        msg.ee_poses.append(_copy_pose(pose))
        msg.tolerances.append(Twist())
        self.goal_pub.publish(msg)

    def _at_target(self, pose):
        if self.current_pose is None:
            return False
        dx = float(self.current_pose.pose.position.x) - float(pose.position.x)
        dy = float(self.current_pose.pose.position.y) - float(pose.position.y)
        dz = float(self.current_pose.pose.position.z) - float(pose.position.z)
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        angle = _quat_angle_rad(
            [
                self.current_pose.pose.orientation.x,
                self.current_pose.pose.orientation.y,
                self.current_pose.pose.orientation.z,
                self.current_pose.pose.orientation.w,
            ],
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
        )
        return dist <= self.position_tolerance_m and angle <= self.orientation_tolerance_rad

    def _enter_next_guard_step(self):
        next_pose = _copy_pose(self.guard_target_pose)
        next_pose.position.z = max(
            float(self.active_profile.get("guarded_min_z", next_pose.position.z)),
            float(next_pose.position.z) - float(self.active_profile.get("guarded_step_m", 0.0015)),
        )
        self.guard_target_pose = next_pose
        self.guard_step_started_at = rospy.Time.now()
        self.guard_step_start_z = float(self.current_pose.pose.position.z) if self.current_pose is not None else None

    def _start_place(self, label="", object_name="", stack_id="default"):
        if self.phase != "IDLE":
            rospy.logwarn("[thin_layer_placement_controller] ignoring place request because phase=%s", self.phase)
            return
        if self.current_pose is None:
            self._publish_status("cannot_start_place_no_endpoint_pose")
            return
        label = str(label).strip() or self.current_label
        meta = self.label_to_meta.get(label, {})
        if not object_name:
            object_name = str(meta.get("object_name", "")).strip()
        if not object_name:
            self._publish_status("cannot_start_place_unknown_object")
            return
        state = self.stack_states.get(stack_id)
        if state is None:
            self._publish_status("cannot_start_place_unknown_stack")
            return

        profile = self._profile_for_object(object_name)
        anchor_pose = _copy_pose(state["anchor_pose"])
        predicted_top_z = float(anchor_pose.position.z) + float(state.get("stack_height_m", 0.0))
        observed_top_z = state.get("last_observed_top_z")
        if observed_top_z is not None:
            predicted_top_z = max(predicted_top_z, float(observed_top_z))

        release_safety_offset = float(
            profile.get("release_safety_offset_m", self.stack_config.get("release_safety_offset_m", 0.0015)) or 0.0
        )
        hover_offset = float(profile.get("hover_offset_m", self.stack_config.get("hover_offset_m", 0.050)) or 0.050)
        slow_approach_offset = float(
            profile.get("slow_approach_offset_m", self.stack_config.get("slow_approach_offset_m", 0.006)) or 0.006
        )
        max_guarded_descent = float(
            profile.get("max_guarded_descent_m", self.stack_config.get("max_guarded_descent_m", 0.010)) or 0.010
        )
        nominal_thickness = float(profile.get("nominal_thickness_m", 0.003) or 0.003)

        self.current_label = label
        self.active_stack_id = stack_id
        self.active_object_name = object_name
        self.active_profile = {
            "nominal_thickness_m": nominal_thickness,
            "contact_to_surface_offset_m": float(profile.get("contact_to_surface_offset_m", 0.0) or 0.0),
            "guarded_step_m": float(
                profile.get("guarded_step_m", self.stack_config.get("guarded_step_m", 0.0015)) or 0.0015
            ),
            "contact_progress_epsilon_m": float(
                profile.get(
                    "contact_progress_epsilon_m",
                    self.stack_config.get("contact_progress_epsilon_m", 0.0007),
                )
                or 0.0007
            ),
            "contact_hold_steps": int(
                profile.get("contact_hold_steps", self.stack_config.get("contact_hold_steps", 3)) or 3
            ),
            "guarded_min_z": predicted_top_z + release_safety_offset - max_guarded_descent,
        }

        release_z = predicted_top_z + release_safety_offset
        self.hover_pose = _copy_pose(anchor_pose)
        self.hover_pose.position.z = release_z + hover_offset
        self.approach_pose = _copy_pose(anchor_pose)
        self.approach_pose.position.z = release_z + slow_approach_offset
        self.retreat_pose = _copy_pose(anchor_pose)
        self.retreat_pose.position.z = release_z + float(
            profile.get("retreat_offset_m", self.stack_config.get("retreat_offset_m", 0.045)) or 0.045
        )
        self.guard_target_pose = _copy_pose(self.approach_pose)
        self.guard_step_started_at = rospy.Time.now()
        self.guard_step_start_z = float(self.current_pose.pose.position.z)
        self.guard_nonprogress_count = 0
        self.awaiting_manual_release = False
        self.release_event_sent = False
        self.pending_auto_place_label = ""
        self.pending_auto_place_stack_id = "default"
        self.pending_auto_place_start_z = None

        self._set_phase("MOVE_TO_HOVER")
        self._publish_event(
            "thin_place_start",
            predicted_top_z=predicted_top_z,
            release_z=release_z,
            nominal_thickness_m=nominal_thickness,
        )
        self._publish_status("thin_place_start:{}".format(object_name))

    def _perform_release(self):
        self.awaiting_manual_release = False
        if self.enable_gripper_actions and self.gripper is not None:
            try:
                self.gripper.open()
            except BaseException as exc:
                rospy.logwarn("[thin_layer_placement_controller] gripper open failed: %s", exc)
        self._set_phase("RELEASE_SETTLE")
        if not self.release_event_sent:
            self.release_event_sent = True
            self._publish_event("release_complete")
        self._publish_status("release_complete")

    def _update_stack_after_release(self):
        state = self._current_stack_state()
        if not state or self.current_pose is None:
            return
        nominal = float(self.active_profile.get("nominal_thickness_m", 0.0))
        contact_offset = float(self.active_profile.get("contact_to_surface_offset_m", 0.0))
        observed_top_z = float(self.current_pose.pose.position.z) - contact_offset
        anchor_z = float(state["anchor_pose"].position.z)
        observed_height = max(0.0, observed_top_z - anchor_z)
        state["last_observed_top_z"] = observed_top_z
        state["stack_height_m"] = max(float(state.get("stack_height_m", 0.0)) + nominal, observed_height)
        state["placed_layers"].append(
            {
                "label": self.current_label,
                "object_name": self.active_object_name,
                "stack_height_m": float(state["stack_height_m"]),
                "observed_top_z": observed_top_z,
            }
        )

    def _finish_cycle(self):
        self._update_stack_after_release()
        self._publish_event("thin_place_retreat_complete")
        self._publish_status("thin_place_done:{}".format(self.active_object_name or "none"))
        self.phase = "IDLE"
        self.active_object_name = ""
        self.active_profile = {}
        self.hover_pose = None
        self.approach_pose = None
        self.retreat_pose = None
        self.guard_target_pose = None
        self.awaiting_manual_release = False
        self.release_event_sent = False

    def _maybe_start_pending_auto_place(self):
        if not self.pending_auto_place_label or self.phase != "IDLE" or self.current_pose is None:
            return
        start_z = self.pending_auto_place_start_z
        if start_z is None:
            self.pending_auto_place_start_z = float(self.current_pose.pose.position.z)
            return
        current_z = float(self.current_pose.pose.position.z)
        if (current_z - float(start_z)) < self.auto_place_min_lift_delta_m:
            return
        label = self.pending_auto_place_label
        stack_id = self.pending_auto_place_stack_id or "default"
        self.pending_auto_place_label = ""
        self.pending_auto_place_stack_id = "default"
        self.pending_auto_place_start_z = None
        self._publish_status("lift_detected_start_place:{}".format(label))
        self._start_place(label=label, stack_id=stack_id)

    def _tick(self, _event):
        if self.phase == "IDLE":
            self._maybe_start_pending_auto_place()
            return
        if self.current_pose is None:
            return

        if self.phase == "MOVE_TO_HOVER":
            self._publish_goal(self.hover_pose)
            if self._at_target(self.hover_pose):
                self._set_phase("MOVE_TO_APPROACH")
                self._publish_status("thin_place_hover_reached")
            return

        if self.phase == "MOVE_TO_APPROACH":
            self._publish_goal(self.approach_pose)
            if self._at_target(self.approach_pose):
                self.guard_target_pose = _copy_pose(self.approach_pose)
                self.guard_step_started_at = rospy.Time.now()
                self.guard_step_start_z = float(self.current_pose.pose.position.z)
                self.guard_nonprogress_count = 0
                self._enter_next_guard_step()
                self._set_phase("GUARDED_DESCENT")
                self._publish_status("thin_place_guarded_descent")
            return

        if self.phase == "GUARDED_DESCENT":
            self._publish_goal(self.guard_target_pose)
            if (rospy.Time.now() - self.guard_step_started_at).to_sec() < self.guard_step_wait_sec:
                return
            if self.guard_step_start_z is None:
                return
            actual_z = float(self.current_pose.pose.position.z)
            achieved_drop = float(self.guard_step_start_z) - actual_z
            if achieved_drop < float(self.active_profile.get("contact_progress_epsilon_m", 0.0007)):
                self.guard_nonprogress_count += 1
            else:
                self.guard_nonprogress_count = 0

            reached_min_z = actual_z <= float(self.active_profile.get("guarded_min_z", actual_z))
            if reached_min_z or self.guard_nonprogress_count >= int(self.active_profile.get("contact_hold_steps", 3)):
                self._set_phase("WAIT_RELEASE")
                self._publish_event(
                    "thin_place_contact",
                    nonprogress_count=self.guard_nonprogress_count,
                    contact_z=actual_z,
                )
                self._publish_status("thin_place_contact_detected")
                return
            self._enter_next_guard_step()
            return

        if self.phase == "WAIT_RELEASE":
            if self.enable_gripper_actions and self.gripper is not None:
                self._perform_release()
                return
            self.awaiting_manual_release = True
            self._publish_status("manual_release_required_press_b")
            if self._pressed_edge(self.manual_release_button_index):
                self._perform_release()
            return

        if self.phase == "RELEASE_SETTLE":
            if (rospy.Time.now() - self.phase_started_at).to_sec() >= self.release_hold_sec:
                self._set_phase("RETREAT")
                self._publish_status("thin_place_retreat")
            return

        if self.phase == "RETREAT":
            self._publish_goal(self.retreat_pose)
            if self._at_target(self.retreat_pose):
                self._finish_cycle()


if __name__ == "__main__":
    ThinLayerPlacementController()
    rospy.spin()
