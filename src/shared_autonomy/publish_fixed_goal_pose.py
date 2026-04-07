#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish one fixed goal pose from the grasp YAML to RelaxedIK."""

import os

import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from intera_core_msgs.msg import EndpointState
from geometry_msgs.msg import Twist
from relaxed_ik_ros1.msg import EEPoseGoals
from std_msgs.msg import String


class FixedGoalPublisher:
    def __init__(self):
        rospy.init_node("publish_fixed_goal_pose")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_yaml = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")

        self.fixed_grasp_yaml = os.path.expanduser(rospy.get_param("~fixed_grasp_yaml", default_yaml))
        self.grasp_id = str(rospy.get_param("~grasp_id", "side_pregrasp_oat")).strip()
        self.fixed_grasp_stage = str(rospy.get_param("~fixed_grasp_stage", "pregrasp_pose")).strip()
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.action_z_offset = float(rospy.get_param("~action_z_offset", 0.0))
        self.enable_z_clamp = bool(rospy.get_param("~enable_z_clamp", False))
        self.min_goal_z = float(rospy.get_param("~min_goal_z", -0.05))
        self.ramp_duration_sec = float(rospy.get_param("~ramp_duration_sec", 4.0))
        self.end_effector_topic = str(rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")).strip()
        self.use_staging_pose = bool(rospy.get_param("~use_staging_pose", True))
        self.staging_offset_x = float(rospy.get_param("~staging_offset_x", -0.08))
        self.staging_offset_y = float(rospy.get_param("~staging_offset_y", 0.0))
        self.staging_offset_z = float(rospy.get_param("~staging_offset_z", 0.04))
        self.stage1_duration_sec = float(rospy.get_param("~stage1_duration_sec", 4.0))
        self.stage2_duration_sec = float(rospy.get_param("~stage2_duration_sec", 3.0))
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "fixed_goal_test")).strip()
        self.selected_grasp_config = {}
        self.resolved_staging_offset = (
            self.staging_offset_x,
            self.staging_offset_y,
            self.staging_offset_z,
        )

        self.goal_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.pose_pub = rospy.Publisher("~goal_pose", PoseStamped, queue_size=1, latch=True)
        self.staging_pose_pub = rospy.Publisher("~staging_pose", PoseStamped, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)

        self.current_pose = None
        self.start_pose = None
        self.start_time = None

        rospy.Subscriber(self.end_effector_topic, EndpointState, self._endpoint_cb, queue_size=10)

        self.goal_pose = self._load_goal_pose()
        self.staging_pose = self._build_staging_pose(self.goal_pose)
        self.status_pub.publish(
            String(data=f"loaded grasp_id={self.grasp_id} stage={self.fixed_grasp_stage}")
        )
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
                "[publish_fixed_goal_pose] control_mode=%s but required=%s. Shutting down to avoid command conflicts.",
                current_mode,
                self.required_control_mode,
            )
            rospy.signal_shutdown("control mode mismatch")

    def _load_goal_pose(self):
        if not os.path.exists(self.fixed_grasp_yaml):
            raise RuntimeError(f"Fixed grasp YAML not found: {self.fixed_grasp_yaml}")

        with open(self.fixed_grasp_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        for grasp in data.get("grasps", []):
            if str(grasp.get("grasp_id", "")).strip() != self.grasp_id:
                continue

            self.selected_grasp_config = grasp
            pose_dict = grasp.get(self.fixed_grasp_stage)
            if not isinstance(pose_dict, dict):
                raise RuntimeError(
                    f"Candidate '{self.grasp_id}' does not contain stage '{self.fixed_grasp_stage}'."
                )

            position = pose_dict.get("position", [])
            orientation = pose_dict.get("orientation", [])
            if len(position) != 3 or len(orientation) != 4:
                raise RuntimeError(f"Malformed pose for '{self.grasp_id}'.")

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.base_frame
            pose_msg.pose.position.x = float(position[0])
            pose_msg.pose.position.y = float(position[1])
            requested_z = float(position[2]) + self.action_z_offset
            pose_msg.pose.position.z = max(requested_z, self.min_goal_z) if self.enable_z_clamp else requested_z
            pose_msg.pose.orientation.x = float(orientation[0])
            pose_msg.pose.orientation.y = float(orientation[1])
            pose_msg.pose.orientation.z = float(orientation[2])
            pose_msg.pose.orientation.w = float(orientation[3])
            if self.enable_z_clamp and pose_msg.pose.position.z != requested_z:
                rospy.logwarn(
                    "[publish_fixed_goal_pose] clamped goal z from %.4f to %.4f for safety.",
                    requested_z,
                    pose_msg.pose.position.z,
                )
            return pose_msg

        raise RuntimeError(f"Could not find grasp_id '{self.grasp_id}' in {self.fixed_grasp_yaml}")

    def _get_staging_offset(self):
        grasp_offset = self.selected_grasp_config.get("staging_offset")
        if isinstance(grasp_offset, list) and len(grasp_offset) == 3:
            try:
                return float(grasp_offset[0]), float(grasp_offset[1]), float(grasp_offset[2])
            except (TypeError, ValueError):
                rospy.logwarn(
                    "[publish_fixed_goal_pose] ignoring malformed staging_offset for %s: %s",
                    self.grasp_id,
                    grasp_offset,
                )
        return self.staging_offset_x, self.staging_offset_y, self.staging_offset_z

    def _build_staging_pose(self, goal_pose):
        offset_x, offset_y, offset_z = self._get_staging_offset()
        self.resolved_staging_offset = (offset_x, offset_y, offset_z)
        stage = PoseStamped()
        stage.header.frame_id = goal_pose.header.frame_id
        stage.pose.position.x = goal_pose.pose.position.x + offset_x
        stage.pose.position.y = goal_pose.pose.position.y + offset_y
        stage_requested_z = goal_pose.pose.position.z + offset_z
        stage.pose.position.z = max(stage_requested_z, self.min_goal_z) if self.enable_z_clamp else stage_requested_z
        stage.pose.orientation = goal_pose.pose.orientation
        return stage

    @staticmethod
    def _lerp(a, b, alpha):
        return a + (b - a) * alpha

    def _blend_pose(self, from_pose, to_pose, alpha):
        pose = type(self.goal_pose.pose)()
        pose.position.x = self._lerp(from_pose.position.x, to_pose.position.x, alpha)
        pose.position.y = self._lerp(from_pose.position.y, to_pose.position.y, alpha)
        pose.position.z = self._lerp(from_pose.position.z, to_pose.position.z, alpha)
        pose.orientation.x = self._lerp(from_pose.orientation.x, to_pose.orientation.x, alpha)
        pose.orientation.y = self._lerp(from_pose.orientation.y, to_pose.orientation.y, alpha)
        pose.orientation.z = self._lerp(from_pose.orientation.z, to_pose.orientation.z, alpha)
        pose.orientation.w = self._lerp(from_pose.orientation.w, to_pose.orientation.w, alpha)
        return pose

    def _interpolated_pose(self, now):
        if self.current_pose is None:
            return self.goal_pose.pose

        if self.start_pose is None:
            self.start_pose = self.current_pose.pose
            self.start_time = now
            return self.start_pose

        if self.ramp_duration_sec <= 1e-6 and not self.use_staging_pose:
            return self.goal_pose.pose

        elapsed = max(0.0, (now - self.start_time).to_sec())

        if not self.use_staging_pose:
            alpha = min(1.0, elapsed / max(self.ramp_duration_sec, 1e-6))
            return self._blend_pose(self.start_pose, self.goal_pose.pose, alpha)

        if elapsed <= self.stage1_duration_sec:
            alpha = min(1.0, elapsed / max(self.stage1_duration_sec, 1e-6))
            return self._blend_pose(self.start_pose, self.staging_pose.pose, alpha)

        stage2_elapsed = elapsed - self.stage1_duration_sec
        alpha = min(1.0, stage2_elapsed / max(self.stage2_duration_sec, 1e-6))
        return self._blend_pose(self.staging_pose.pose, self.goal_pose.pose, alpha)

    def run(self):
        rate = rospy.Rate(self.publish_rate_hz)
        rospy.loginfo(
            "[publish_fixed_goal_pose] publishing grasp_id=%s stage=%s to /relaxed_ik/ee_pose_goals (enable_z_clamp=%s min_goal_z=%.3f use_staging=%s stage_offset=[%.3f, %.3f, %.3f] stage1=%.2fs stage2=%.2fs rate=%.1fHz)",
            self.grasp_id,
            self.fixed_grasp_stage,
            str(self.enable_z_clamp).lower(),
            self.min_goal_z,
            str(self.use_staging_pose).lower(),
            self.resolved_staging_offset[0],
            self.resolved_staging_offset[1],
            self.resolved_staging_offset[2],
            self.stage1_duration_sec,
            self.stage2_duration_sec,
            self.publish_rate_hz,
        )
        self.pose_pub.publish(self.goal_pose)
        self.staging_pose_pub.publish(self.staging_pose)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            msg = EEPoseGoals()
            msg.header.stamp = now
            msg.header.frame_id = self.base_frame
            msg.ee_poses.append(self._interpolated_pose(now))
            msg.tolerances.append(Twist())
            self.goal_pub.publish(msg)
            self.pose_pub.publish(self.goal_pose)
            self.staging_pose_pub.publish(self.staging_pose)
            rate.sleep()


def main():
    FixedGoalPublisher().run()


if __name__ == "__main__":
    main()
