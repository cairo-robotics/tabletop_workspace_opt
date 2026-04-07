#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a simple fixed-pose pouring sequence from the YAML pose library."""

import copy
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
        self.pour_move_sec = float(rospy.get_param("~pour_move_sec", 2.5))
        self.pour_hold_sec = float(rospy.get_param("~pour_hold_sec", 3.0))
        self.return_move_sec = float(rospy.get_param("~return_move_sec", 2.5))
        self.return_hold_sec = float(rospy.get_param("~return_hold_sec", 1.0))
        self.place_back_move_sec = float(rospy.get_param("~place_back_move_sec", 4.0))
        self.release_hold_sec = float(rospy.get_param("~release_hold_sec", 1.0))
        self.final_hold_sec = float(rospy.get_param("~final_hold_sec", 2.0))
        self.safe_hover_offset_z = float(rospy.get_param("~safe_hover_offset_z", 0.10))
        self.safe_travel_min_z = float(rospy.get_param("~safe_travel_min_z", 0.08))
        self.place_release_offset_z = float(rospy.get_param("~place_release_offset_z", 0.10))
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
        self.carry_hover_pose = None
        self.pre_pour_hover_pose = None
        self.pour_hover_pose = None
        self.return_hover_pose = None
        self.place_back_hover_pose = None
        self.release_pose = self._make_release_pose(self.place_back_pose)
        self.gripper = None

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

        if name == "MOVE_TO_CARRY_HOVER":
            self.carry_hover_pose = self._make_hover_pose(self.carry_pose, self.phase_start_pose)
            self.command_pose = self.carry_hover_pose
        elif name == "MOVE_TO_PRE_POUR_HOVER":
            self.pre_pour_hover_pose = self._make_hover_pose(self.pre_pour_pose, self.phase_start_pose)
            self.command_pose = self.pre_pour_hover_pose
        elif name == "MOVE_TO_POUR_HOVER":
            self.pour_hover_pose = self._make_hover_pose(self.pour_pose, self.phase_start_pose)
            self.command_pose = self.pour_hover_pose
        elif name == "MOVE_TO_RETURN_HOVER":
            self.return_hover_pose = self._make_hover_pose(self.return_pose, self.phase_start_pose)
            self.command_pose = self.return_hover_pose
        elif name == "MOVE_TO_PLACE_BACK_HOVER":
            self.place_back_hover_pose = self._make_hover_pose(self.place_back_pose, self.phase_start_pose)
            self.command_pose = self.place_back_hover_pose

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
            return self._pose_for_motion_phase(self.carry_pose, elapsed, self.pre_pour_move_sec)
        if self.phase_name == "HOLD_CARRY":
            return self.carry_pose

        if self.phase_name == "MOVE_TO_PRE_POUR_HOVER":
            return self._pose_for_motion_phase(self.pre_pour_hover_pose, elapsed, self.pre_pour_move_sec)
        if self.phase_name == "HOLD_PRE_POUR_HOVER":
            return self.pre_pour_hover_pose
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
            rospy.loginfo("[test_pour_task_sequence] sequence complete.")
            rospy.signal_shutdown("sequence complete")

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
                    self._set_phase("MOVE_TO_PRE_POUR_HOVER")

            elif self.phase_name == "MOVE_TO_PRE_POUR_HOVER":
                if (1.0 if self.phase_start_pose is None else min(1.0, elapsed / max(self.pre_pour_move_sec, 1e-6))) >= 1.0:
                    self._set_phase("HOLD_PRE_POUR_HOVER", self.pre_pour_hover_pose)

            elif self.phase_name == "HOLD_PRE_POUR_HOVER":
                if elapsed >= self.hover_hold_sec:
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
