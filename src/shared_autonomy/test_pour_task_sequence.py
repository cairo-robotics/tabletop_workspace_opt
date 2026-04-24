#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a simple fixed-pose pouring sequence from the YAML pose library."""

import copy
import math
import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from intera_core_msgs.msg import EndpointState
from intera_interface import Gripper, RobotEnable
from relaxed_ik_ros1.msg import EEPoseGoals
from std_msgs.msg import String


class PourTaskSequenceTest:
    def __init__(self):
        rospy.init_node("test_pour_task_sequence")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_yaml = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")

        self.fixed_grasp_yaml = os.path.expanduser(rospy.get_param("~fixed_grasp_yaml", default_yaml))
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 15.0))
        self.end_effector_topic = str(
            rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        ).strip()
        self.limb = str(rospy.get_param("~limb", "right")).strip()
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "fixed_goal_test")).strip()
        self.loop_sequence = bool(rospy.get_param("~loop_sequence", False))
        self.wait_for_grasp_complete = bool(rospy.get_param("~wait_for_grasp_complete", True))
        self.execution_state_topic = str(
            rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")
        ).strip()
        self.grasp_complete_label = str(rospy.get_param("~grasp_complete_label", "side_grasp_milk")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param(
                "~object_map_yaml",
                os.path.join(package_root, "config", "apriltag_object_map.yaml"),
            )
        )

        self.carry_grasp_id = str(rospy.get_param("~carry_grasp_id", "carry_pose")).strip()
        self.carry_stage = str(rospy.get_param("~carry_stage", "carry_pose")).strip()

        self.pre_pour_grasp_id = str(rospy.get_param("~pre_pour_grasp_id", "pre_pour_pose")).strip()
        self.pre_pour_stage = str(rospy.get_param("~pre_pour_stage", "pour_pre_pose")).strip()
        self.pour_grasp_id = str(rospy.get_param("~pour_grasp_id", "pour_pose")).strip()
        self.pour_stage = str(rospy.get_param("~pour_stage", "pour_pose")).strip()
        self.return_grasp_id = str(rospy.get_param("~return_grasp_id", "return_upright_pose")).strip()
        self.return_stage = str(rospy.get_param("~return_stage", "return_upright_pose")).strip()
        self.place_back_grasp_id = str(rospy.get_param("~place_back_grasp_id", "place_back_pose")).strip()
        self.place_back_stage = str(rospy.get_param("~place_back_stage", "place_back_pose")).strip()

        self.start_hold_sec = float(rospy.get_param("~start_hold_sec", 1.0))
        self.hover_hold_sec = float(rospy.get_param("~hover_hold_sec", 0.5))
        self.pre_pour_move_sec = float(rospy.get_param("~pre_pour_move_sec", 4.0))
        self.pre_pour_hold_sec = float(rospy.get_param("~pre_pour_hold_sec", 1.0))
        self.pre_pour_transit_move_sec = float(rospy.get_param("~pre_pour_transit_move_sec", 2.0))
        self.pre_pour_transit_hold_sec = float(rospy.get_param("~pre_pour_transit_hold_sec", 0.5))
        self.pre_pour_align_move_sec = float(rospy.get_param("~pre_pour_align_move_sec", 2.0))
        self.pre_pour_align_hold_sec = float(rospy.get_param("~pre_pour_align_hold_sec", 0.5))
        self.pour_move_sec = float(rospy.get_param("~pour_move_sec", 2.5))
        self.pour_hold_sec = float(rospy.get_param("~pour_hold_sec", 3.0))
        self.return_move_sec = float(rospy.get_param("~return_move_sec", 2.5))
        self.return_hold_sec = float(rospy.get_param("~return_hold_sec", 1.0))
        self.return_lift_move_sec = float(rospy.get_param("~return_lift_move_sec", 1.5))
        self.return_lift_hold_sec = float(rospy.get_param("~return_lift_hold_sec", 0.5))
        self.place_back_move_sec = float(rospy.get_param("~place_back_move_sec", 4.0))
        self.release_hold_sec = float(rospy.get_param("~release_hold_sec", 1.0))
        self.final_hold_sec = float(rospy.get_param("~final_hold_sec", 2.0))
        self.safe_hover_offset_z = float(rospy.get_param("~safe_hover_offset_z", 0.10))
        self.safe_travel_min_z = float(rospy.get_param("~safe_travel_min_z", 0.08))
        self.place_release_offset_z = float(rospy.get_param("~place_release_offset_z", 0.10))
        self.return_lift_offset_z = float(rospy.get_param("~return_lift_offset_z", 0.06))
        self.enable_robot_interface = bool(rospy.get_param("~enable_robot_interface", True))
        self.enable_gripper_actions = bool(rospy.get_param("~enable_gripper_actions", True))
        self.auto_prepare_gripper = bool(rospy.get_param("~auto_prepare_gripper", True))

        self.goal_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.carry_pub = rospy.Publisher("~carry_pose", PoseStamped, queue_size=1, latch=True)
        self.pre_pour_pub = rospy.Publisher("~pre_pour_pose", PoseStamped, queue_size=1, latch=True)
        self.pour_pub = rospy.Publisher("~pour_pose", PoseStamped, queue_size=1, latch=True)
        self.return_pub = rospy.Publisher("~return_upright_pose", PoseStamped, queue_size=1, latch=True)
        self.place_back_pub = rospy.Publisher("~place_back_pose", PoseStamped, queue_size=1, latch=True)

        self.current_pose = None
        self.phase_name = "WAIT_FOR_TRIGGER" if self.wait_for_grasp_complete else "WAIT_FOR_POSE"
        self.phase_started_at = rospy.Time.now()
        self.phase_start_pose = None
        self.command_pose = None
        self.grasp_triggered = not self.wait_for_grasp_complete

        self.carry_pose = self._load_pose(self.carry_grasp_id, self.carry_stage)
        self.pre_pour_pose = self._load_pose(self.pre_pour_grasp_id, self.pre_pour_stage)
        self.pour_pose = self._load_pose(self.pour_grasp_id, self.pour_stage)
        self.return_pose = self._load_pose(self.return_grasp_id, self.return_stage)
        self.place_back_pose = self._load_pose(self.place_back_grasp_id, self.place_back_stage)
        self.grasp_reference_pose = None
        self.carry_hover_pose = None
        self.pre_pour_transit_pose = None
        self.pre_pour_hover_pose = None
        self.pre_pour_align_pose = None
        self.pour_hover_pose = None
        self.return_hover_pose = None
        self.return_lift_pose = None
        self.place_back_hover_pose = None
        self.release_pose = self._make_release_pose(self.place_back_pose)
        self.gripper = None
        self.label_to_meta = self._load_label_metadata()
        self.active_task_policy_name = "default"

        if self.enable_robot_interface:
            try:
                rs = RobotEnable(False)
                rs.enable()
            except BaseException as exc:
                rospy.logwarn("[test_pour_task_sequence] could not enable robot interface: %s", exc)
            if self.enable_gripper_actions:
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
                    rospy.logwarn("[test_pour_task_sequence] could not initialize gripper: %s", exc)

        rospy.Subscriber(self.end_effector_topic, EndpointState, self._endpoint_cb, queue_size=10)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=10)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_grasp_label_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), self._control_mode_guard)

    def _endpoint_cb(self, msg):
        self.current_pose = PoseStamped()
        self.current_pose.header = msg.header
        self.current_pose.header.frame_id = self.base_frame
        self.current_pose.pose = msg.pose

    def _control_mode_guard(self, _event):
        current_mode = str(rospy.get_param("/tabletop_workspace_opt/control_mode", "")).strip()
        if current_mode and current_mode != self.required_control_mode:
            rospy.logwarn(
                "[test_pour_task_sequence] control_mode=%s but required=%s. Shutting down to avoid command conflicts.",
                current_mode,
                self.required_control_mode,
            )
            rospy.signal_shutdown("control mode mismatch")

    def _execution_state_cb(self, msg):
        expected = "grasp_complete:{}".format(self.grasp_complete_label)
        if str(msg.data).strip() != expected:
            return
        if self.grasp_triggered:
            return
        self.grasp_triggered = True
        rospy.loginfo(
            "[test_pour_task_sequence] received trigger %s. Starting pour sequence.",
            expected,
        )
        if self.phase_name == "WAIT_FOR_TRIGGER":
            self._set_phase("WAIT_FOR_POSE")

    def _selected_grasp_label_cb(self, msg):
        selected_label = str(msg.data).strip()
        if not selected_label:
            return
        meta = self.label_to_meta.get(selected_label, {})
        if str(meta.get("task", "")).strip().lower() != "pour":
            rospy.loginfo(
                "[test_pour_task_sequence] ignoring selected label %s because task=%s",
                selected_label,
                str(meta.get("task", "")).strip() or "unknown",
            )
            return
        if self.grasp_complete_label == selected_label:
            return
        self.grasp_complete_label = selected_label
        self._apply_task_policy_from_meta(meta)
        rospy.loginfo(
            "[test_pour_task_sequence] updated grasp_complete_label to %s from selector topic.",
            selected_label,
        )

    def _load_label_metadata(self):
        if not os.path.exists(self.object_map_yaml):
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        tag_objects = data.get("tag_objects", {}) if isinstance(data, dict) else {}
        mapping = {}
        for meta in tag_objects.values():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("grasp_complete_label", "")).strip()
            if label:
                mapping[label] = meta
        return mapping

    def _policy_value(self, meta, meta_key, default_value):
        value = str(meta.get(meta_key, "")).strip() if isinstance(meta, dict) else ""
        return value or default_value

    def _apply_task_policy_from_meta(self, meta):
        if not isinstance(meta, dict):
            return

        next_carry_grasp_id = self._policy_value(meta, "carry_grasp_id", self.carry_grasp_id)
        next_carry_stage = self._policy_value(meta, "carry_stage", self.carry_stage)
        next_pre_pour_grasp_id = self._policy_value(meta, "pre_pour_grasp_id", self.pre_pour_grasp_id)
        next_pre_pour_stage = self._policy_value(meta, "pre_pour_stage", self.pre_pour_stage)
        next_pour_grasp_id = self._policy_value(meta, "pour_grasp_id", self.pour_grasp_id)
        next_pour_stage = self._policy_value(meta, "pour_stage", self.pour_stage)
        next_return_grasp_id = self._policy_value(meta, "return_grasp_id", self.return_grasp_id)
        next_return_stage = self._policy_value(meta, "return_stage", self.return_stage)
        next_place_back_grasp_id = self._policy_value(meta, "place_back_grasp_id", self.place_back_grasp_id)
        next_place_back_stage = self._policy_value(meta, "place_back_stage", self.place_back_stage)

        policy_name = str(meta.get("task_policy", "")).strip() or str(meta.get("category", "")).strip() or "default"

        changed = (
            next_carry_grasp_id != self.carry_grasp_id
            or next_carry_stage != self.carry_stage
            or next_pre_pour_grasp_id != self.pre_pour_grasp_id
            or next_pre_pour_stage != self.pre_pour_stage
            or next_pour_grasp_id != self.pour_grasp_id
            or next_pour_stage != self.pour_stage
            or next_return_grasp_id != self.return_grasp_id
            or next_return_stage != self.return_stage
            or next_place_back_grasp_id != self.place_back_grasp_id
            or next_place_back_stage != self.place_back_stage
        )
        if not changed and self.active_task_policy_name == policy_name:
            return

        self.carry_grasp_id = next_carry_grasp_id
        self.carry_stage = next_carry_stage
        self.pre_pour_grasp_id = next_pre_pour_grasp_id
        self.pre_pour_stage = next_pre_pour_stage
        self.pour_grasp_id = next_pour_grasp_id
        self.pour_stage = next_pour_stage
        self.return_grasp_id = next_return_grasp_id
        self.return_stage = next_return_stage
        self.place_back_grasp_id = next_place_back_grasp_id
        self.place_back_stage = next_place_back_stage

        self.carry_pose = self._load_pose(self.carry_grasp_id, self.carry_stage)
        self.pre_pour_pose = self._load_pose(self.pre_pour_grasp_id, self.pre_pour_stage)
        self.pour_pose = self._load_pose(self.pour_grasp_id, self.pour_stage)
        self.return_pose = self._load_pose(self.return_grasp_id, self.return_stage)
        self.place_back_pose = self._load_pose(self.place_back_grasp_id, self.place_back_stage)
        self.release_pose = self._make_release_pose(self.place_back_pose)

        self.carry_pub.publish(self.carry_pose)
        self.pre_pour_pub.publish(self.pre_pour_pose)
        self.pour_pub.publish(self.pour_pose)
        self.return_pub.publish(self.return_pose)
        self.place_back_pub.publish(self.place_back_pose)

        self.active_task_policy_name = policy_name
        rospy.loginfo(
            "[test_pour_task_sequence] task_policy=%s carry=%s/%s pre_pour=%s/%s pour=%s/%s return=%s/%s place_back=%s/%s",
            self.active_task_policy_name,
            self.carry_grasp_id,
            self.carry_stage,
            self.pre_pour_grasp_id,
            self.pre_pour_stage,
            self.pour_grasp_id,
            self.pour_stage,
            self.return_grasp_id,
            self.return_stage,
            self.place_back_grasp_id,
            self.place_back_stage,
        )

    def _load_pose_from_dict(self, pose_dict):
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

    def _load_pose(self, grasp_id, stage_name):
        if not os.path.exists(self.fixed_grasp_yaml):
            raise RuntimeError(f"Fixed grasp YAML not found: {self.fixed_grasp_yaml}")

        with open(self.fixed_grasp_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        for grasp in data.get("grasps", []):
            if str(grasp.get("grasp_id", "")).strip() != grasp_id:
                continue
            pose_dict = grasp.get(stage_name)
            if not isinstance(pose_dict, dict):
                raise RuntimeError(
                    f"Candidate '{grasp_id}' does not contain stage '{stage_name}'."
                )
            return self._load_pose_from_dict(pose_dict)

        raise RuntimeError(f"Could not find grasp_id '{grasp_id}' in {self.fixed_grasp_yaml}")

    @staticmethod
    def _lerp(a, b, alpha):
        return a + (b - a) * alpha

    @staticmethod
    def _normalize_quat(quat):
        norm = math.sqrt(sum(float(v) * float(v) for v in quat))
        if norm < 1e-9:
            return [0.0, 0.0, 0.0, 1.0]
        return [float(v) / norm for v in quat]

    @classmethod
    def _quat_slerp(cls, q0, q1, alpha):
        t = max(0.0, min(1.0, float(alpha)))
        qa = cls._normalize_quat(q0)
        qb = cls._normalize_quat(q1)
        dot = sum(a * b for a, b in zip(qa, qb))
        if dot < 0.0:
            qb = [-v for v in qb]
            dot = -dot
        dot = max(-1.0, min(1.0, dot))
        if dot > 0.9995:
            return cls._normalize_quat([(1.0 - t) * a + t * b for a, b in zip(qa, qb)])
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        if abs(sin_theta_0) < 1e-9:
            return qa
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return cls._normalize_quat([s0 * a + s1 * b for a, b in zip(qa, qb)])

    def _blend_pose(self, from_pose, to_pose, alpha):
        pose = copy.deepcopy(from_pose)
        pose.position.x = self._lerp(from_pose.position.x, to_pose.position.x, alpha)
        pose.position.y = self._lerp(from_pose.position.y, to_pose.position.y, alpha)
        pose.position.z = self._lerp(from_pose.position.z, to_pose.position.z, alpha)
        blended_q = self._quat_slerp(
            [
                from_pose.orientation.x,
                from_pose.orientation.y,
                from_pose.orientation.z,
                from_pose.orientation.w,
            ],
            [
                to_pose.orientation.x,
                to_pose.orientation.y,
                to_pose.orientation.z,
                to_pose.orientation.w,
            ],
            alpha,
        )
        pose.orientation.x = blended_q[0]
        pose.orientation.y = blended_q[1]
        pose.orientation.z = blended_q[2]
        pose.orientation.w = blended_q[3]
        return pose

    def _publish_pose(self, pose_stamped):
        msg = EEPoseGoals()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        msg.ee_poses.append(copy.deepcopy(pose_stamped.pose))
        msg.tolerances.append(Twist())
        self.goal_pub.publish(msg)

    def _should_publish_goal(self):
        if self.phase_name == "WAIT_FOR_TRIGGER":
            return False
        return True

    def _make_hover_pose(self, target_pose, source_pose=None):
        hover_pose = copy.deepcopy(target_pose)
        hover_pose.pose.position.z = max(
            hover_pose.pose.position.z + self.safe_hover_offset_z,
            self.safe_travel_min_z,
        )
        if source_pose is not None:
            hover_pose.pose.orientation = copy.deepcopy(source_pose.orientation)
        return hover_pose

    def _make_release_pose(self, place_back_pose):
        release_pose = copy.deepcopy(place_back_pose)
        release_pose.pose.position.z += self.place_release_offset_z
        return release_pose

    def _make_lift_pose(self, base_pose, offset_z):
        lift_pose = copy.deepcopy(base_pose)
        lift_pose.pose.position.z = max(
            lift_pose.pose.position.z + offset_z,
            self.safe_travel_min_z,
        )
        return lift_pose

    def _make_return_lift_pose(self, return_pose):
        return self._make_lift_pose(return_pose, self.return_lift_offset_z)

    def _make_pre_pour_transit_pose(self, source_pose, target_pose):
        transit_pose = copy.deepcopy(target_pose)
        source = source_pose if source_pose is not None else target_pose
        transit_pose.pose.position.x = float(source.pose.position.x)
        transit_pose.pose.position.y = float(source.pose.position.y)
        transit_pose.pose.position.z = max(
            float(source.pose.position.z),
            float(target_pose.pose.position.z) + self.safe_hover_offset_z,
            self.safe_travel_min_z,
        )
        transit_pose.pose.orientation = copy.deepcopy(source.pose.orientation)
        return transit_pose

    def _make_orientation_align_pose(self, reference_pose, target_pose):
        align_pose = copy.deepcopy(reference_pose)
        align_pose.pose.orientation = copy.deepcopy(target_pose.pose.orientation)
        return align_pose

    def _capture_grasp_reference_pose(self):
        if self.current_pose is None:
            return
        self.grasp_reference_pose = copy.deepcopy(self.current_pose)
        rospy.loginfo(
            "[test_pour_task_sequence] captured grasp reference pose at x=%.3f y=%.3f z=%.3f",
            self.grasp_reference_pose.pose.position.x,
            self.grasp_reference_pose.pose.position.y,
            self.grasp_reference_pose.pose.position.z,
        )

    def _phase_elapsed(self):
        return max(0.0, (rospy.Time.now() - self.phase_started_at).to_sec())

    def _set_phase(self, name, command_pose=None):
        self.phase_name = name
        self.phase_started_at = rospy.Time.now()
        self.command_pose = command_pose
        if self.current_pose is not None:
            self.phase_start_pose = copy.deepcopy(self.current_pose.pose)
        else:
            self.phase_start_pose = None

        if name == "START_HOLD":
            self._capture_grasp_reference_pose()

        if name == "MOVE_TO_CARRY_HOVER":
            carry_target = self._make_lift_pose(
                self.grasp_reference_pose if self.grasp_reference_pose is not None else self.carry_pose,
                self.safe_hover_offset_z,
            )
            self.carry_hover_pose = carry_target
            self.command_pose = self.carry_hover_pose
        elif name == "MOVE_TO_PRE_POUR_TRANSIT":
            self.pre_pour_transit_pose = self._make_pre_pour_transit_pose(
                self.carry_hover_pose if self.carry_hover_pose is not None else self.phase_start_pose,
                self.pre_pour_pose,
            )
            self.command_pose = self.pre_pour_transit_pose
        elif name == "MOVE_TO_PRE_POUR_HOVER":
            self.pre_pour_hover_pose = self._make_hover_pose(self.pre_pour_pose, self.phase_start_pose)
            self.command_pose = self.pre_pour_hover_pose
        elif name == "MOVE_TO_PRE_POUR_ALIGN":
            self.pre_pour_align_pose = self._make_orientation_align_pose(
                self.pre_pour_hover_pose if self.pre_pour_hover_pose is not None else self.pre_pour_pose,
                self.pre_pour_pose,
            )
            self.command_pose = self.pre_pour_align_pose
        elif name == "MOVE_TO_POUR_HOVER":
            self.pour_hover_pose = self._make_hover_pose(self.pour_pose, self.phase_start_pose)
            self.command_pose = self.pour_hover_pose
        elif name == "MOVE_TO_RETURN_HOVER":
            self.return_hover_pose = self._make_hover_pose(self.return_pose, self.phase_start_pose)
            self.command_pose = self.return_hover_pose
        elif name == "MOVE_TO_RETURN_LIFT":
            self.return_lift_pose = self._make_return_lift_pose(self.return_pose)
            self.command_pose = self.return_lift_pose
        elif name == "MOVE_TO_PLACE_BACK_HOVER":
            place_back_base = self.grasp_reference_pose if self.grasp_reference_pose is not None else self.place_back_pose
            self.place_back_hover_pose = self._make_lift_pose(place_back_base, self.safe_hover_offset_z)
            self.command_pose = self.place_back_hover_pose
        elif name == "MOVE_TO_RELEASE":
            self.release_pose = (
                copy.deepcopy(self.grasp_reference_pose)
                if self.grasp_reference_pose is not None
                else self._make_release_pose(self.place_back_pose)
            )
            self.command_pose = self.release_pose

        self.status_pub.publish(String(data=name))
        rospy.loginfo("[test_pour_task_sequence] phase -> %s", name)
        if name == "OPEN_GRIPPER" and self.gripper is not None:
            try:
                rospy.loginfo("[test_pour_task_sequence] opening gripper for release")
                self.gripper.open()
            except Exception as exc:
                rospy.logwarn("[test_pour_task_sequence] gripper open failed: %s", exc)

    def _pose_for_motion_phase(self, target_pose, elapsed, duration_sec):
        if self.phase_start_pose is None:
            return target_pose
        alpha = min(1.0, elapsed / max(duration_sec, 1e-6))
        target_msg = copy.deepcopy(target_pose)
        target_msg.pose = self._blend_pose(self.phase_start_pose, target_pose.pose, alpha)
        return target_msg

    def _commanded_pose_for_phase(self, elapsed):
        if self.phase_name == "WAIT_FOR_TRIGGER":
            if self.current_pose is not None:
                return self.current_pose
            return self.carry_pose

        if self.phase_name in ("WAIT_FOR_POSE", "START_HOLD"):
            if self.current_pose is not None:
                return self.current_pose
            return self.carry_pose

        if self.phase_name == "MOVE_TO_CARRY_HOVER":
            return self._pose_for_motion_phase(self.carry_hover_pose, elapsed, self.pre_pour_move_sec)
        if self.phase_name == "HOLD_CARRY_HOVER":
            return self.carry_hover_pose
        if self.phase_name == "MOVE_TO_CARRY":
            return self.carry_hover_pose
        if self.phase_name == "HOLD_CARRY":
            return self.carry_hover_pose

        if self.phase_name == "MOVE_TO_PRE_POUR_TRANSIT":
            return self._pose_for_motion_phase(self.pre_pour_transit_pose, elapsed, self.pre_pour_transit_move_sec)
        if self.phase_name == "HOLD_PRE_POUR_TRANSIT":
            return self.pre_pour_transit_pose
        if self.phase_name == "MOVE_TO_PRE_POUR_HOVER":
            return self._pose_for_motion_phase(self.pre_pour_hover_pose, elapsed, self.pre_pour_move_sec)
        if self.phase_name == "HOLD_PRE_POUR_HOVER":
            return self.pre_pour_hover_pose
        if self.phase_name == "MOVE_TO_PRE_POUR_ALIGN":
            return self._pose_for_motion_phase(self.pre_pour_align_pose, elapsed, self.pre_pour_align_move_sec)
        if self.phase_name == "HOLD_PRE_POUR_ALIGN":
            return self.pre_pour_align_pose
        if self.phase_name == "MOVE_TO_PRE_POUR":
            return self._pose_for_motion_phase(self.pre_pour_pose, elapsed, self.pre_pour_move_sec)
        if self.phase_name == "HOLD_PRE_POUR":
            return self.pre_pour_pose

        if self.phase_name == "MOVE_TO_POUR_HOVER":
            return self._pose_for_motion_phase(self.pour_hover_pose, elapsed, self.pour_move_sec)
        if self.phase_name == "HOLD_POUR_HOVER":
            return self.pour_hover_pose
        if self.phase_name == "MOVE_TO_POUR":
            return self._pose_for_motion_phase(self.pour_pose, elapsed, self.pour_move_sec)
        if self.phase_name == "HOLD_POUR":
            return self.pour_pose

        if self.phase_name == "MOVE_TO_RETURN_HOVER":
            return self._pose_for_motion_phase(self.return_hover_pose, elapsed, self.return_move_sec)
        if self.phase_name == "HOLD_RETURN_HOVER":
            return self.return_hover_pose
        if self.phase_name == "MOVE_TO_RETURN":
            return self._pose_for_motion_phase(self.return_pose, elapsed, self.return_move_sec)
        if self.phase_name == "HOLD_RETURN":
            return self.return_pose
        if self.phase_name == "MOVE_TO_RETURN_LIFT":
            return self._pose_for_motion_phase(self.return_lift_pose, elapsed, self.return_lift_move_sec)
        if self.phase_name == "HOLD_RETURN_LIFT":
            return self.return_lift_pose

        if self.phase_name == "MOVE_TO_PLACE_BACK_HOVER":
            return self._pose_for_motion_phase(self.place_back_hover_pose, elapsed, self.place_back_move_sec)
        if self.phase_name == "HOLD_PLACE_BACK_HOVER":
            return self.place_back_hover_pose
        if self.phase_name == "MOVE_TO_RELEASE":
            return self._pose_for_motion_phase(self.release_pose, elapsed, self.place_back_move_sec)
        if self.phase_name in ("HOLD_RELEASE", "OPEN_GRIPPER", "FINAL_HOLD"):
            return self.release_pose

        return self.release_pose

    def _advance_after_final_hold(self):
        if self.loop_sequence:
            self._set_phase("START_HOLD")
        else:
            rospy.loginfo("[test_pour_task_sequence] sequence complete. Returning to WAIT_FOR_TRIGGER.")
            self.grasp_triggered = False
            self.grasp_reference_pose = None
            self.carry_hover_pose = None
            self.pre_pour_transit_pose = None
            self.pre_pour_hover_pose = None
            self.pre_pour_align_pose = None
            self.pour_hover_pose = None
            self.return_hover_pose = None
            self.return_lift_pose = None
            self.place_back_hover_pose = None
            self._set_phase("WAIT_FOR_TRIGGER")

    def run(self):
        rate = rospy.Rate(self.publish_rate_hz)

        self.carry_pub.publish(self.carry_pose)
        self.pre_pour_pub.publish(self.pre_pour_pose)
        self.pour_pub.publish(self.pour_pose)
        self.return_pub.publish(self.return_pose)
        self.place_back_pub.publish(self.place_back_pose)
        self.status_pub.publish(String(data="waiting_for_grasp_complete" if self.wait_for_grasp_complete else "waiting_for_endpoint_state"))

        rospy.loginfo(
            "[test_pour_task_sequence] carry=%s/%s pre_pour=%s/%s pour=%s/%s return=%s/%s place_back=%s/%s release_z_offset=%.3f wait_for_grasp_complete=%s trigger=%s",
            self.carry_grasp_id,
            self.carry_stage,
            self.pre_pour_grasp_id,
            self.pre_pour_stage,
            self.pour_grasp_id,
            self.pour_stage,
            self.return_grasp_id,
            self.return_stage,
            self.place_back_grasp_id,
            self.place_back_stage,
            self.place_release_offset_z,
            str(self.wait_for_grasp_complete).lower(),
            self.grasp_complete_label,
        )

        while not rospy.is_shutdown():
            elapsed = self._phase_elapsed()
            if self._should_publish_goal():
                self._publish_pose(self._commanded_pose_for_phase(elapsed))

            if self.phase_name == "WAIT_FOR_TRIGGER":
                pass

            elif self.phase_name == "WAIT_FOR_POSE":
                self._set_phase("START_HOLD")

            elif self.phase_name == "START_HOLD":
                if elapsed >= self.start_hold_sec:
                    self._set_phase("MOVE_TO_CARRY_HOVER")

            elif self.phase_name == "MOVE_TO_CARRY_HOVER":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pre_pour_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_CARRY_HOVER", self.carry_hover_pose)

            elif self.phase_name == "HOLD_CARRY_HOVER":
                if elapsed >= self.hover_hold_sec:
                    self._set_phase("MOVE_TO_CARRY", self.carry_pose)

            elif self.phase_name == "MOVE_TO_CARRY":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pre_pour_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_CARRY", self.carry_pose)

            elif self.phase_name == "HOLD_CARRY":
                if elapsed >= self.pre_pour_hold_sec:
                    self._set_phase("MOVE_TO_PRE_POUR_TRANSIT")

            elif self.phase_name == "MOVE_TO_PRE_POUR_TRANSIT":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pre_pour_transit_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_PRE_POUR_TRANSIT", self.pre_pour_transit_pose)

            elif self.phase_name == "HOLD_PRE_POUR_TRANSIT":
                if elapsed >= self.pre_pour_transit_hold_sec:
                    self._set_phase("MOVE_TO_PRE_POUR_HOVER")

            elif self.phase_name == "MOVE_TO_PRE_POUR_HOVER":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pre_pour_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_PRE_POUR_HOVER", self.pre_pour_hover_pose)

            elif self.phase_name == "HOLD_PRE_POUR_HOVER":
                if elapsed >= self.hover_hold_sec:
                    self._set_phase("MOVE_TO_PRE_POUR_ALIGN")

            elif self.phase_name == "MOVE_TO_PRE_POUR_ALIGN":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pre_pour_align_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_PRE_POUR_ALIGN", self.pre_pour_align_pose)

            elif self.phase_name == "HOLD_PRE_POUR_ALIGN":
                if elapsed >= self.pre_pour_align_hold_sec:
                    self._set_phase("MOVE_TO_PRE_POUR", self.pre_pour_pose)

            elif self.phase_name == "MOVE_TO_PRE_POUR":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pre_pour_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_PRE_POUR", self.pre_pour_pose)

            elif self.phase_name == "HOLD_PRE_POUR":
                if elapsed >= self.pre_pour_hold_sec:
                    self._set_phase("MOVE_TO_POUR_HOVER")

            elif self.phase_name == "MOVE_TO_POUR_HOVER":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pour_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_POUR_HOVER", self.pour_hover_pose)

            elif self.phase_name == "HOLD_POUR_HOVER":
                if elapsed >= self.hover_hold_sec:
                    self._set_phase("MOVE_TO_POUR", self.pour_pose)

            elif self.phase_name == "MOVE_TO_POUR":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pour_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_POUR", self.pour_pose)

            elif self.phase_name == "HOLD_POUR":
                if elapsed >= self.pour_hold_sec:
                    self._set_phase("MOVE_TO_RETURN_HOVER")

            elif self.phase_name == "MOVE_TO_RETURN_HOVER":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.return_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_RETURN_HOVER", self.return_hover_pose)

            elif self.phase_name == "HOLD_RETURN_HOVER":
                if elapsed >= self.hover_hold_sec:
                    self._set_phase("MOVE_TO_RETURN", self.return_pose)

            elif self.phase_name == "MOVE_TO_RETURN":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.return_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_RETURN", self.return_pose)

            elif self.phase_name == "HOLD_RETURN":
                if elapsed >= self.return_hold_sec:
                    self._set_phase("MOVE_TO_RETURN_LIFT")

            elif self.phase_name == "MOVE_TO_RETURN_LIFT":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.return_lift_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_RETURN_LIFT", self.return_lift_pose)

            elif self.phase_name == "HOLD_RETURN_LIFT":
                if elapsed >= self.return_lift_hold_sec:
                    self._set_phase("MOVE_TO_PLACE_BACK_HOVER")

            elif self.phase_name == "MOVE_TO_PLACE_BACK_HOVER":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.place_back_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_PLACE_BACK_HOVER", self.place_back_hover_pose)

            elif self.phase_name == "HOLD_PLACE_BACK_HOVER":
                if elapsed >= self.hover_hold_sec:
                    self._set_phase("MOVE_TO_RELEASE", self.release_pose)

            elif self.phase_name == "MOVE_TO_RELEASE":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.place_back_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_RELEASE", self.release_pose)

            elif self.phase_name == "HOLD_RELEASE":
                if elapsed >= self.release_hold_sec:
                    self._set_phase("OPEN_GRIPPER", self.release_pose)

            elif self.phase_name == "OPEN_GRIPPER":
                if elapsed >= self.release_hold_sec:
                    self._set_phase("FINAL_HOLD", self.release_pose)

            elif self.phase_name == "FINAL_HOLD":
                if elapsed >= self.final_hold_sec:
                    self._advance_after_final_hold()

            else:
                rospy.logwarn("[test_pour_task_sequence] unknown phase '%s'", self.phase_name)
                rospy.signal_shutdown("unknown phase")

            rate.sleep()


def main():
    PourTaskSequenceTest().run()


if __name__ == "__main__":
    main()
