#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a simple fixed-grasp test: open, move to pregrasp, move to grasp, close."""

import copy
import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped, Twist
from intera_core_msgs.msg import EndpointState
from intera_interface import Gripper, RobotEnable
from relaxed_ik_ros1.msg import EEPoseGoals
from std_msgs.msg import String


class SequentialFixedGraspTest:
    def __init__(self):
        rospy.init_node("sequential_fixed_grasp_test")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_yaml = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")

        self.fixed_grasp_yaml = os.path.expanduser(rospy.get_param("~fixed_grasp_yaml", default_yaml))
        self.grasp_id = str(rospy.get_param("~grasp_id", "side_pregrasp_oat")).strip()
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.limb = str(rospy.get_param("~limb", "right")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 15.0))
        self.end_effector_topic = str(
            rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        ).strip()
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "fixed_goal_test")).strip()
        self.enable_robot_interface = bool(rospy.get_param("~enable_robot_interface", True))
        self.enable_gripper_actions = bool(rospy.get_param("~enable_gripper_actions", False))
        self.auto_prepare_gripper = bool(rospy.get_param("~auto_prepare_gripper", True))
        self.loop_sequence = bool(rospy.get_param("~loop_sequence", False))

        self.open_pause_sec = float(rospy.get_param("~open_pause_sec", 1.0))
        self.stage_move_sec = float(rospy.get_param("~stage_move_sec", 3.0))
        self.stage_hold_sec = float(rospy.get_param("~stage_hold_sec", 0.5))
        self.pregrasp_move_sec = float(rospy.get_param("~pregrasp_move_sec", 4.0))
        self.pregrasp_hold_sec = float(rospy.get_param("~pregrasp_hold_sec", 1.0))
        self.grasp_move_sec = float(rospy.get_param("~grasp_move_sec", 2.5))
        self.grasp_hold_sec = float(rospy.get_param("~grasp_hold_sec", 0.5))
        self.close_pause_sec = float(rospy.get_param("~close_pause_sec", 1.0))
        self.final_hold_sec = float(rospy.get_param("~final_hold_sec", 2.0))

        self.goal_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.staging_pub = rospy.Publisher("~staging_pose", PoseStamped, queue_size=1, latch=True)
        self.pregrasp_pub = rospy.Publisher("~pregrasp_pose", PoseStamped, queue_size=1, latch=True)
        self.grasp_pub = rospy.Publisher("~grasp_pose", PoseStamped, queue_size=1, latch=True)

        self.current_pose = None
        self.phase_name = "WAIT_FOR_POSE"
        self.phase_started_at = rospy.Time.now()
        self.phase_start_pose = None
        self.command_pose = None

        self.resolved_staging_offset = (0.0, 0.0, 0.0)
        self.pregrasp_pose, self.grasp_pose = self._load_grasp_pair()
        self.staging_pose = self._build_staging_pose(self.pregrasp_pose)

        self.gripper = None
        if self.enable_robot_interface:
            try:
                # Skip strict SDK/robot version gating here so this utility test
                # does not exit immediately on benign version mismatches.
                rs = RobotEnable(False)
                rs.enable()
            except BaseException as exc:
                rospy.logwarn("[sequential_fixed_grasp_test] could not enable robot interface: %s", exc)
            if self.enable_gripper_actions:
                try:
                    self.gripper = Gripper(self.limb + "_gripper", calibrate=False)
                    self.gripper.set_dead_zone(0.001)
                    rospy.loginfo(
                        "[sequential_fixed_grasp_test] gripper detected: calibrated=%s error=%s ready=%s",
                        self.gripper.is_calibrated(),
                        self.gripper.has_error(),
                        self.gripper.is_ready(),
                    )
                    if self.auto_prepare_gripper:
                        if self.gripper.has_error():
                            rospy.logwarn("[sequential_fixed_grasp_test] gripper has error, rebooting...")
                            self.gripper.reboot()
                            rospy.sleep(1.0)
                        if not self.gripper.is_calibrated():
                            rospy.logwarn("[sequential_fixed_grasp_test] gripper not calibrated, calibrating...")
                            self.gripper.calibrate()
                            rospy.sleep(0.5)
                        rospy.loginfo(
                            "[sequential_fixed_grasp_test] gripper after prepare: calibrated=%s error=%s ready=%s",
                            self.gripper.is_calibrated(),
                            self.gripper.has_error(),
                            self.gripper.is_ready(),
                        )
                except BaseException as exc:
                    rospy.logwarn("[sequential_fixed_grasp_test] could not initialize gripper: %s", exc)
            else:
                rospy.logwarn("[sequential_fixed_grasp_test] gripper actions disabled for this run.")
        else:
            rospy.logwarn("[sequential_fixed_grasp_test] robot interface disabled, gripper actions disabled.")

        rospy.Subscriber(self.end_effector_topic, EndpointState, self._endpoint_cb, queue_size=10)
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
                "[sequential_fixed_grasp_test] control_mode=%s but required=%s. Shutting down to avoid command conflicts.",
                current_mode,
                self.required_control_mode,
            )
            rospy.signal_shutdown("control mode mismatch")

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

    @staticmethod
    def _paired_grasp_id(grasp_id):
        if "_pregrasp_" in grasp_id:
            return grasp_id.replace("_pregrasp_", "_grasp_", 1)
        if "_grasp_" in grasp_id:
            return grasp_id.replace("_grasp_", "_pregrasp_", 1)
        return None

    @staticmethod
    def _find_grasp_entry(entries, grasp_id):
        for grasp in entries:
            if str(grasp.get("grasp_id", "")).strip() == grasp_id:
                return grasp
        return None

    def _extract_staging_offset(self, *entries):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            offset = entry.get("staging_offset")
            if isinstance(offset, list) and len(offset) == 3:
                try:
                    return float(offset[0]), float(offset[1]), float(offset[2])
                except (TypeError, ValueError):
                    pass
        return 0.0, 0.0, 0.0

    def _build_staging_pose(self, pregrasp_pose):
        stage = copy.deepcopy(pregrasp_pose)
        ox, oy, oz = self.resolved_staging_offset
        stage.pose.position.x += ox
        stage.pose.position.y += oy
        stage.pose.position.z += oz
        return stage

    def _load_grasp_pair(self):
        if not os.path.exists(self.fixed_grasp_yaml):
            raise RuntimeError(f"Fixed grasp YAML not found: {self.fixed_grasp_yaml}")

        with open(self.fixed_grasp_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        grasps = data.get("grasps", [])
        grasp = self._find_grasp_entry(grasps, self.grasp_id)
        if grasp is None:
            raise RuntimeError(f"Could not find grasp_id '{self.grasp_id}' in {self.fixed_grasp_yaml}")

        pregrasp_dict = grasp.get("pregrasp_pose")
        grasp_dict = grasp.get("grasp_pose")
        if isinstance(pregrasp_dict, dict) and isinstance(grasp_dict, dict):
            return self._load_pose_from_dict(pregrasp_dict), self._load_pose_from_dict(grasp_dict)

        paired_id = self._paired_grasp_id(self.grasp_id)
        paired_entry = self._find_grasp_entry(grasps, paired_id) if paired_id else None
        if paired_entry is not None:
            if pregrasp_dict is None:
                pregrasp_dict = paired_entry.get("pregrasp_pose")
            if grasp_dict is None:
                grasp_dict = paired_entry.get("grasp_pose")

        if not isinstance(pregrasp_dict, dict) or not isinstance(grasp_dict, dict):
            raise RuntimeError(
                f"Grasp '{self.grasp_id}' must resolve to both pregrasp_pose and grasp_pose."
            )

        if paired_entry is not None:
            rospy.loginfo(
                "[sequential_fixed_grasp_test] paired %s with %s for pregrasp/grasp sequence.",
                self.grasp_id,
                paired_id,
            )
        self.resolved_staging_offset = self._extract_staging_offset(grasp, paired_entry)
        return self._load_pose_from_dict(pregrasp_dict), self._load_pose_from_dict(grasp_dict)

    @staticmethod
    def _lerp(a, b, alpha):
        return a + (b - a) * alpha

    def _blend_pose(self, from_pose, to_pose, alpha):
        pose = copy.deepcopy(from_pose)
        pose.position.x = self._lerp(from_pose.position.x, to_pose.position.x, alpha)
        pose.position.y = self._lerp(from_pose.position.y, to_pose.position.y, alpha)
        pose.position.z = self._lerp(from_pose.position.z, to_pose.position.z, alpha)
        pose.orientation.x = self._lerp(from_pose.orientation.x, to_pose.orientation.x, alpha)
        pose.orientation.y = self._lerp(from_pose.orientation.y, to_pose.orientation.y, alpha)
        pose.orientation.z = self._lerp(from_pose.orientation.z, to_pose.orientation.z, alpha)
        pose.orientation.w = self._lerp(from_pose.orientation.w, to_pose.orientation.w, alpha)
        return pose

    def _publish_pose(self, pose_stamped):
        msg = EEPoseGoals()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        msg.ee_poses.append(copy.deepcopy(pose_stamped.pose))
        msg.tolerances.append(Twist())
        self.goal_pub.publish(msg)

    def _phase_elapsed(self):
        return max(0.0, (rospy.Time.now() - self.phase_started_at).to_sec())

    def _set_phase(self, name, command_pose=None):
        self.phase_name = name
        self.phase_started_at = rospy.Time.now()
        self.command_pose = command_pose
        if self.current_pose is not None:
            self.phase_start_pose = copy.deepcopy(self.current_pose.pose)
        elif command_pose is not None:
            self.phase_start_pose = None
        else:
            self.phase_start_pose = None

        self.status_pub.publish(String(data=name))
        rospy.loginfo("[sequential_fixed_grasp_test] phase -> %s", name)

        if name == "OPEN_GRIPPER" and self.gripper is not None:
            try:
                rospy.loginfo("[sequential_fixed_grasp_test] opening gripper")
                self.gripper.open()
            except Exception as exc:
                rospy.logwarn("[sequential_fixed_grasp_test] gripper open failed: %s", exc)
        elif name == "CLOSE_GRIPPER" and self.gripper is not None:
            try:
                rospy.loginfo("[sequential_fixed_grasp_test] closing gripper")
                self.gripper.close()
            except Exception as exc:
                rospy.logwarn("[sequential_fixed_grasp_test] gripper close failed: %s", exc)

    def _commanded_pose_for_phase(self, elapsed):
        if self.phase_name in ("WAIT_FOR_POSE", "OPEN_GRIPPER"):
            if self.current_pose is not None:
                return self.current_pose
            return self.staging_pose

        if self.phase_name == "MOVE_TO_STAGING":
            if self.phase_start_pose is None:
                return self.staging_pose
            alpha = min(1.0, elapsed / max(self.stage_move_sec, 1e-6))
            target_msg = copy.deepcopy(self.staging_pose)
            target_msg.pose = self._blend_pose(self.phase_start_pose, self.staging_pose.pose, alpha)
            return target_msg

        if self.phase_name == "HOLD_STAGING":
            return self.staging_pose

        if self.phase_name == "MOVE_TO_PREGRASP":
            if self.phase_start_pose is None:
                return self.pregrasp_pose
            alpha = min(1.0, elapsed / max(self.pregrasp_move_sec, 1e-6))
            target_msg = copy.deepcopy(self.pregrasp_pose)
            target_msg.pose = self._blend_pose(self.phase_start_pose, self.pregrasp_pose.pose, alpha)
            return target_msg

        if self.phase_name == "HOLD_PREGRASP":
            return self.pregrasp_pose

        if self.phase_name == "MOVE_TO_GRASP":
            if self.phase_start_pose is None:
                return self.grasp_pose
            alpha = min(1.0, elapsed / max(self.grasp_move_sec, 1e-6))
            target_msg = copy.deepcopy(self.grasp_pose)
            target_msg.pose = self._blend_pose(self.phase_start_pose, self.grasp_pose.pose, alpha)
            return target_msg

        if self.phase_name in ("HOLD_GRASP", "CLOSE_GRIPPER", "FINAL_HOLD"):
            return self.grasp_pose

        return self.grasp_pose

    def _advance_after_final_hold(self):
        if self.loop_sequence:
            self._set_phase("OPEN_GRIPPER")
        else:
            rospy.loginfo("[sequential_fixed_grasp_test] sequence complete.")
            rospy.signal_shutdown("sequence complete")

    def run(self):
        rate = rospy.Rate(self.publish_rate_hz)

        self.pregrasp_pub.publish(self.pregrasp_pose)
        self.grasp_pub.publish(self.grasp_pose)
        self.staging_pub.publish(self.staging_pose)
        self.status_pub.publish(String(data="waiting_for_endpoint_state"))

        rospy.loginfo(
            "[sequential_fixed_grasp_test] grasp_id=%s stage_offset=[%.3f, %.3f, %.3f] stage_move=%.2fs pregrasp_move=%.2fs grasp_move=%.2fs loop=%s",
            self.grasp_id,
            self.resolved_staging_offset[0],
            self.resolved_staging_offset[1],
            self.resolved_staging_offset[2],
            self.stage_move_sec,
            self.pregrasp_move_sec,
            self.grasp_move_sec,
            str(self.loop_sequence).lower(),
        )

        while not rospy.is_shutdown():
            elapsed = self._phase_elapsed()
            self._publish_pose(self._commanded_pose_for_phase(elapsed))

            if self.phase_name == "WAIT_FOR_POSE":
                self._set_phase("OPEN_GRIPPER")

            elif self.phase_name == "OPEN_GRIPPER":
                if elapsed >= self.open_pause_sec:
                    self._set_phase("MOVE_TO_STAGING", self.staging_pose)

            elif self.phase_name == "MOVE_TO_STAGING":
                alpha = 1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.stage_move_sec, 1e-6))
                if alpha >= 1.0:
                    self._set_phase("HOLD_STAGING", self.staging_pose)

            elif self.phase_name == "HOLD_STAGING":
                if elapsed >= self.stage_hold_sec:
                    self._set_phase("MOVE_TO_PREGRASP", self.pregrasp_pose)

            elif self.phase_name == "MOVE_TO_PREGRASP":
                alpha = 1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pregrasp_move_sec, 1e-6))
                if alpha >= 1.0:
                    self._set_phase("HOLD_PREGRASP", self.pregrasp_pose)

            elif self.phase_name == "HOLD_PREGRASP":
                if elapsed >= self.pregrasp_hold_sec:
                    self._set_phase("MOVE_TO_GRASP", self.grasp_pose)

            elif self.phase_name == "MOVE_TO_GRASP":
                alpha = 1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.grasp_move_sec, 1e-6))
                if alpha >= 1.0:
                    self._set_phase("HOLD_GRASP", self.grasp_pose)

            elif self.phase_name == "HOLD_GRASP":
                if elapsed >= self.grasp_hold_sec:
                    self._set_phase("CLOSE_GRIPPER", self.grasp_pose)

            elif self.phase_name == "CLOSE_GRIPPER":
                if elapsed >= self.close_pause_sec:
                    self._set_phase("FINAL_HOLD", self.grasp_pose)

            elif self.phase_name == "FINAL_HOLD":
                if elapsed >= self.final_hold_sec:
                    self._advance_after_final_hold()

            else:
                rospy.logwarn("[sequential_fixed_grasp_test] unknown phase '%s'", self.phase_name)
                rospy.signal_shutdown("unknown phase")

            rate.sleep()


def main():
    SequentialFixedGraspTest().run()


if __name__ == "__main__":
    main()
