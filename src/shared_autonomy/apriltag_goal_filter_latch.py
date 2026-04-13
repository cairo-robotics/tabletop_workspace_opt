#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter and latch AprilTag grasp goals for stable execution."""

import copy
import math

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
from std_msgs.msg import String


def _quat_normalize(q):
    q = np.array(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_angle_deg(q0, q1):
    a = _quat_normalize(q0)
    b = _quat_normalize(q1)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _pose_distance(a, b):
    dx = float(a.pose.position.x - b.pose.position.x)
    dy = float(a.pose.position.y - b.pose.position.y)
    dz = float(a.pose.position.z - b.pose.position.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pose_quat(msg):
    return np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )


def _ema_pose(prev, cur, alpha):
    out = copy.deepcopy(cur)
    out.pose.position.x = (1.0 - alpha) * prev.pose.position.x + alpha * cur.pose.position.x
    out.pose.position.y = (1.0 - alpha) * prev.pose.position.y + alpha * cur.pose.position.y
    out.pose.position.z = (1.0 - alpha) * prev.pose.position.z + alpha * cur.pose.position.z
    qa = _quat_normalize(_pose_quat(prev))
    qb = _quat_normalize(_pose_quat(cur))
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    q = _quat_normalize((1.0 - alpha) * qa + alpha * qb)
    out.pose.orientation.x = float(q[0])
    out.pose.orientation.y = float(q[1])
    out.pose.orientation.z = float(q[2])
    out.pose.orientation.w = float(q[3])
    out.header.stamp = rospy.Time.now()
    return out


class AprilTagGoalFilterLatch:
    def __init__(self):
        rospy.init_node("apriltag_goal_filter_latch")

        self.input_pregrasp_topic = str(rospy.get_param("~input_pregrasp_topic", "/tag_grasp_demo/pregrasp_pose")).strip()
        self.input_grasp_topic = str(rospy.get_param("~input_grasp_topic", "/tag_grasp_demo/grasp_pose")).strip()
        self.input_status_topic = str(rospy.get_param("~input_status_topic", "/tag_grasp_demo/status")).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()

        self.output_pregrasp_topic = str(rospy.get_param("~output_pregrasp_topic", "/tag_grasp_filtered/pregrasp_pose")).strip()
        self.output_grasp_topic = str(rospy.get_param("~output_grasp_topic", "/tag_grasp_filtered/grasp_pose")).strip()
        self.output_status_topic = str(rospy.get_param("~output_status_topic", "/tag_grasp_filtered/status")).strip()

        self.alpha = float(rospy.get_param("~ema_alpha", 0.35))
        self.max_pos_jump_m = float(rospy.get_param("~max_pos_jump_m", 0.05))
        self.max_rot_jump_deg = float(rospy.get_param("~max_rot_jump_deg", 20.0))
        self.stable_required = int(rospy.get_param("~stable_count_required", 3))
        self.confirm_button_index = int(rospy.get_param("~confirm_button_index", 2))
        self.cancel_button_index = int(rospy.get_param("~cancel_button_index", 3))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 20.0))

        self.raw_pre = None
        self.raw_grasp = None
        self.filtered_pre = None
        self.filtered_grasp = None
        self.latched_pre = None
        self.latched_grasp = None
        self.stable_count = 0
        self.last_buttons = []
        self.last_raw_status = ""
        self.last_status_msg = ""

        self.pub_pre = rospy.Publisher(self.output_pregrasp_topic, PoseStamped, queue_size=1, latch=True)
        self.pub_grasp = rospy.Publisher(self.output_grasp_topic, PoseStamped, queue_size=1, latch=True)
        self.pub_status = rospy.Publisher(self.output_status_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.input_pregrasp_topic, PoseStamped, self._pre_cb, queue_size=1)
        rospy.Subscriber(self.input_grasp_topic, PoseStamped, self._grasp_cb, queue_size=1)
        rospy.Subscriber(self.input_status_topic, String, self._status_cb, queue_size=1)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._timer_cb)
        self._publish_status("waiting_for_input")
        rospy.loginfo(
            "[apriltag_goal_filter_latch] ready. in=(%s,%s) out=(%s,%s) alpha=%.2f jump_pos=%.3f jump_rot=%.1f stable=%d",
            self.input_pregrasp_topic,
            self.input_grasp_topic,
            self.output_pregrasp_topic,
            self.output_grasp_topic,
            self.alpha,
            self.max_pos_jump_m,
            self.max_rot_jump_deg,
            self.stable_required,
        )

    def _publish_status(self, text):
        if text == self.last_status_msg:
            return
        self.last_status_msg = text
        rospy.loginfo("[apriltag_goal_filter_latch] %s", text)
        self.pub_status.publish(String(data=text))

    def _pre_cb(self, msg):
        self.raw_pre = copy.deepcopy(msg)

    def _grasp_cb(self, msg):
        self.raw_grasp = copy.deepcopy(msg)

    def _status_cb(self, msg):
        self.last_raw_status = str(msg.data).strip()

    def _joy_cb(self, msg):
        self.last_buttons = list(msg.buttons)

    def _button_pressed(self, index):
        if index < 0 or index >= len(self.last_buttons):
            return False
        return bool(self.last_buttons[index])

    def _update_filter(self):
        if self.raw_pre is None or self.raw_grasp is None:
            return False
        if self.filtered_pre is None or self.filtered_grasp is None:
            self.filtered_pre = copy.deepcopy(self.raw_pre)
            self.filtered_grasp = copy.deepcopy(self.raw_grasp)
            self.stable_count = 1
            return True

        jump_pre_pos = _pose_distance(self.filtered_pre, self.raw_pre)
        jump_grasp_pos = _pose_distance(self.filtered_grasp, self.raw_grasp)
        jump_pre_rot = _quat_angle_deg(_pose_quat(self.filtered_pre), _pose_quat(self.raw_pre))
        jump_grasp_rot = _quat_angle_deg(_pose_quat(self.filtered_grasp), _pose_quat(self.raw_grasp))

        too_big = (
            jump_pre_pos > self.max_pos_jump_m
            or jump_grasp_pos > self.max_pos_jump_m
            or jump_pre_rot > self.max_rot_jump_deg
            or jump_grasp_rot > self.max_rot_jump_deg
        )
        if too_big:
            self.stable_count = 0
            self._publish_status(
                "reject_jump pre_pos={:.3f} grasp_pos={:.3f} pre_rot={:.1f} grasp_rot={:.1f} raw={}".format(
                    jump_pre_pos, jump_grasp_pos, jump_pre_rot, jump_grasp_rot, self.last_raw_status
                )
            )
            return False

        a = min(1.0, max(0.0, self.alpha))
        self.filtered_pre = _ema_pose(self.filtered_pre, self.raw_pre, a)
        self.filtered_grasp = _ema_pose(self.filtered_grasp, self.raw_grasp, a)
        self.stable_count += 1
        return True

    def _timer_cb(self, _event):
        updated = self._update_filter()
        if not updated:
            if self.latched_pre is not None and self.latched_grasp is not None:
                self.pub_pre.publish(self.latched_pre)
                self.pub_grasp.publish(self.latched_grasp)
            return

        stable = self.stable_count >= max(1, self.stable_required)
        if self._button_pressed(self.cancel_button_index):
            self.latched_pre = None
            self.latched_grasp = None
            self._publish_status("latch_cleared")

        if self._button_pressed(self.confirm_button_index) and stable:
            self.latched_pre = copy.deepcopy(self.filtered_pre)
            self.latched_grasp = copy.deepcopy(self.filtered_grasp)
            self._publish_status("latched stable_count={} raw={}".format(self.stable_count, self.last_raw_status))

        if self.latched_pre is not None and self.latched_grasp is not None:
            self.pub_pre.publish(self.latched_pre)
            self.pub_grasp.publish(self.latched_grasp)
            if stable:
                self._publish_status("latched stable_count={} raw={}".format(self.stable_count, self.last_raw_status))
        else:
            self.pub_pre.publish(self.filtered_pre)
            self.pub_grasp.publish(self.filtered_grasp)
            self._publish_status("tracking stable_count={} raw={}".format(self.stable_count, self.last_raw_status))


if __name__ == "__main__":
    AprilTagGoalFilterLatch()
    rospy.spin()
