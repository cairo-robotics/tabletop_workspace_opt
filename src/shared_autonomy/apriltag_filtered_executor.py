#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute filtered AprilTag pregrasp/grasp with joystick confirmation."""

import copy
import math

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEPoseGoals
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from visualization_msgs.msg import Marker


def _as_np_pos(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)


def _normalize_quat(q):
    q = np.array(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_angle_rad(a, b):
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


class AprilTagFilteredExecutor:
    def __init__(self):
        rospy.init_node("apriltag_filtered_executor")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.endpoint_topic = str(rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")).strip()
        self.pregrasp_topic = str(rospy.get_param("~pregrasp_topic", "/tag_grasp_demo/pregrasp_pose")).strip()
        self.grasp_topic = str(rospy.get_param("~grasp_topic", "/tag_grasp_demo/grasp_pose")).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.prompt_topic = str(
            rospy.get_param("~prompt_topic", "/intent_inference/confirmation_prompt")
        ).strip()
        self.execution_state_topic = str(
            rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")
        ).strip()
        self.grasp_complete_label = str(rospy.get_param("~grasp_complete_label", "apriltag_id_0")).strip()
        self.prompt_marker_topic = str(rospy.get_param("~prompt_marker_topic", "~prompt_marker")).strip()
        self.prompt_marker_offset = float(rospy.get_param("~prompt_marker_offset_z", 0.15))

        self.confirm_button_index = int(rospy.get_param("~confirm_button_index", 2))
        self.cancel_button_index = int(rospy.get_param("~cancel_button_index", 3))
        self.close_button_index = int(rospy.get_param("~close_button_index", 0))
        self.max_speed_mps = float(rospy.get_param("~max_speed_mps", 0.10))
        self.max_angular_step = float(rospy.get_param("~max_angular_step", 0.12))
        self.pos_tol = float(rospy.get_param("~position_tolerance_m", 0.01))
        self.rot_tol = float(rospy.get_param("~orientation_tolerance_rad", 0.25))
        self.use_goal_orientation = bool(rospy.get_param("~use_goal_orientation", True))
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "shared_autonomy")).strip()

        self.latest_ee = None
        self.latest_buttons = []
        self.prev_buttons = []
        self.last_joy_time = None
        self.pregrasp = None
        self.grasp = None
        self.exec_pregrasp = None
        self.exec_grasp = None
        self.state = "WAIT_PREGRASP_CONFIRM"
        self.last_cmd_time = None
        self.last_status = ""

        self.goal_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.prompt_pub = rospy.Publisher(self.prompt_topic, String, queue_size=1, latch=True)
        self.exec_state_pub = rospy.Publisher(self.execution_state_topic, String, queue_size=1, latch=True)
        self.prompt_marker_pub = rospy.Publisher(self.prompt_marker_topic, Marker, queue_size=1, latch=True)

        rospy.Subscriber(self.endpoint_topic, EndpointState, self._ee_cb, queue_size=10)
        rospy.Subscriber(self.pregrasp_topic, PoseStamped, self._pre_cb, queue_size=1)
        rospy.Subscriber(self.grasp_topic, PoseStamped, self._grasp_cb, queue_size=1)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        rospy.Timer(rospy.Duration(0.05), self._tick)
        rospy.Timer(rospy.Duration(0.5), self._guard)

        self._publish_status("waiting_for_targets")
        rospy.loginfo(
            "[apriltag_filtered_executor] ready pre=%s grasp=%s endpoint=%s joy=%s",
            self.pregrasp_topic,
            self.grasp_topic,
            self.endpoint_topic,
            self.joy_topic,
        )

    def _guard(self, _evt):
        mode = str(rospy.get_param("/tabletop_workspace_opt/control_mode", "")).strip()
        if mode and mode != self.required_control_mode:
            rospy.logwarn(
                "[apriltag_filtered_executor] control_mode=%s required=%s; shutting down",
                mode,
                self.required_control_mode,
            )
            rospy.signal_shutdown("control mode mismatch")

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[apriltag_filtered_executor] %s", text)
        self.status_pub.publish(String(data=text))
        self.prompt_pub.publish(String(data=text))
        self._publish_prompt_marker(text)

    def _publish_prompt_marker(self, text):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "apriltag_exec_prompt"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        if self.latest_ee is not None:
            marker.pose.position.x = float(self.latest_ee.position.x)
            marker.pose.position.y = float(self.latest_ee.position.y)
            marker.pose.position.z = float(self.latest_ee.position.z + self.prompt_marker_offset)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = text
        self.prompt_marker_pub.publish(marker)

    def _ee_cb(self, msg):
        self.latest_ee = msg.pose

    def _pre_cb(self, msg):
        self.pregrasp = copy.deepcopy(msg)

    def _grasp_cb(self, msg):
        self.grasp = copy.deepcopy(msg)

    def _joy_cb(self, msg):
        self.prev_buttons = list(self.latest_buttons)
        self.latest_buttons = list(msg.buttons)
        self.last_joy_time = rospy.Time.now()
        if len(self.prev_buttons) == 0 and len(self.latest_buttons) > 0:
            rospy.loginfo(
                "[apriltag_filtered_executor] joy connected on %s (buttons=%d, confirm=%d cancel=%d close=%d)",
                self.joy_topic,
                len(self.latest_buttons),
                self.confirm_button_index,
                self.cancel_button_index,
                self.close_button_index,
            )

    def _pressed(self, idx):
        if idx < 0 or idx >= len(self.latest_buttons):
            return False
        return bool(self.latest_buttons[idx])

    def _compute_dt(self, now):
        if self.last_cmd_time is None:
            self.last_cmd_time = now
            return 0.05
        dt = max(0.005, (now - self.last_cmd_time).to_sec())
        self.last_cmd_time = now
        return dt

    def _build_cmd_pose(self, target_pose, now):
        cur = self.latest_ee
        dt = self._compute_dt(now)
        cur_p = _as_np_pos(cur)
        tgt_p = _as_np_pos(target_pose)
        delta = tgt_p - cur_p
        dist = float(np.linalg.norm(delta))
        max_step = max(1e-4, self.max_speed_mps * dt)
        if dist > max_step:
            cmd_p = cur_p + delta / dist * max_step
        else:
            cmd_p = tgt_p

        out = Pose()
        out.position.x = float(cmd_p[0])
        out.position.y = float(cmd_p[1])
        out.position.z = float(cmd_p[2])

        if not self.use_goal_orientation:
            out.orientation = copy.deepcopy(cur.orientation)
            return out

        qa = _normalize_quat([cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w])
        qb = _normalize_quat([target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w])
        if float(np.dot(qa, qb)) < 0.0:
            qb = -qb
        q = _normalize_quat(qa + min(1.0, self.max_angular_step) * (qb - qa))
        out.orientation.x = float(q[0])
        out.orientation.y = float(q[1])
        out.orientation.z = float(q[2])
        out.orientation.w = float(q[3])
        return out

    def _at_target(self, target_pose):
        cur = self.latest_ee
        if cur is None:
            return False
        dist = float(np.linalg.norm(_as_np_pos(cur) - _as_np_pos(target_pose)))
        if not self.use_goal_orientation:
            return dist <= self.pos_tol
        ang = _quat_angle_rad(
            [cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w],
            [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w],
        )
        return dist <= self.pos_tol and ang <= self.rot_tol

    def _publish_goal(self, pose, now):
        msg = EEPoseGoals()
        msg.header.stamp = now
        msg.header.frame_id = self.base_frame
        msg.ee_poses.append(pose)
        self.goal_pub.publish(msg)

    def _tick(self, _evt):
        now = rospy.Time.now()
        if self.latest_ee is None or self.pregrasp is None or self.grasp is None:
            self._publish_status(
                "waiting_for_targets ee={} pregrasp={} grasp={} endpoint_topic={} pre_topic={} grasp_topic={}".format(
                    int(self.latest_ee is not None),
                    int(self.pregrasp is not None),
                    int(self.grasp is not None),
                    self.endpoint_topic,
                    self.pregrasp_topic,
                    self.grasp_topic,
                )
            )
            return

        if self.last_joy_time is None:
            self._publish_status("waiting_for_joy topic={}".format(self.joy_topic))
            return

        if self._pressed(self.cancel_button_index):
            self.state = "WAIT_PREGRASP_CONFIRM"
            self.exec_pregrasp = None
            self.exec_grasp = None
            self._publish_status("cancelled_wait_pregrasp")
            return

        if self.state == "WAIT_PREGRASP_CONFIRM":
            self._publish_status("Execute pregrasp? Press X to continue, Y to cancel.")
            if self._pressed(self.confirm_button_index):
                self.exec_pregrasp = copy.deepcopy(self.pregrasp)
                self.exec_grasp = None
                self.state = "EXEC_PREGRASP"
                self._publish_status("confirmed_pregrasp_start")
            return

        if self.state == "EXEC_PREGRASP":
            target_pre = self.exec_pregrasp.pose if self.exec_pregrasp is not None else self.pregrasp.pose
            cmd = self._build_cmd_pose(target_pre, now)
            self._publish_goal(cmd, now)
            self._publish_status("Executing pregrasp...")
            if self._at_target(target_pre):
                self.state = "WAIT_GRASP_CONFIRM"
            return

        if self.state == "WAIT_GRASP_CONFIRM":
            self._publish_status("Execute grasp? Press X to continue, Y to cancel.")
            if self._pressed(self.confirm_button_index):
                self.exec_grasp = copy.deepcopy(self.grasp)
                self.state = "EXEC_GRASP"
                self._publish_status("confirmed_grasp_start")
            return

        if self.state == "EXEC_GRASP":
            target_grasp = self.exec_grasp.pose if self.exec_grasp is not None else self.grasp.pose
            cmd = self._build_cmd_pose(target_grasp, now)
            self._publish_goal(cmd, now)
            self._publish_status("Executing grasp...")
            if self._at_target(target_grasp):
                self.state = "WAIT_CLOSE_A"
            return

        if self.state == "WAIT_CLOSE_A":
            self._publish_status("At grasp pose. Press A to close gripper.")
            if self._pressed(self.close_button_index):
                self.exec_state_pub.publish(String(data="grasp_complete:{}".format(self.grasp_complete_label)))
                self.state = "DONE"
            return

        self._publish_status("Done.")


if __name__ == "__main__":
    AprilTagFilteredExecutor()
    rospy.spin()
