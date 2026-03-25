#!/usr/bin/env python3
"""Adapt shared-autonomy grasp candidates into robot EE pregrasp/grasp target poses."""

import copy
import math

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from tabletop_workspace_opt.msg import GraspCandidate


def _normalize(vec):
    arr = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return arr / norm


def _quat_normalize_xyzw(q):
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    return [float(v) / norm for v in q]


def _quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def _rotate_vec_by_quat_xyzw(v, q):
    x, y, z = v
    qx, qy, qz, qw = q
    ix = qw * x + qy * z - qz * y
    iy = qw * y + qz * x - qx * z
    iz = qw * z + qx * y - qy * x
    iw = -qx * x - qy * y - qz * z
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return np.array([rx, ry, rz], dtype=np.float64)


class SharedAutonomyGraspPoseAdapter:
    def __init__(self):
        rospy.init_node("shared_autonomy_grasp_pose_adapter")

        self.base_frame = rospy.get_param("~base_frame", "world")
        self.execution_mode = rospy.get_param("~execution_mode", "top_down")
        self.selected_grasp_topic = rospy.get_param("~selected_grasp_topic", "/shared_autonomy/selected_grasp")
        self.adapted_grasp_pose_topic = rospy.get_param(
            "~adapted_grasp_pose_topic", "/shared_autonomy/adapted_grasp_pose"
        )
        self.adapted_pregrasp_pose_topic = rospy.get_param(
            "~adapted_pregrasp_pose_topic", "/shared_autonomy/adapted_pregrasp_pose"
        )
        self.grasp_center_offset_m = float(rospy.get_param("~grasp_center_offset_m", 0.0))
        self.pregrasp_offset_m = float(rospy.get_param("~pregrasp_offset_m", 0.12))
        self.pregrasp_world_z_offset_m = float(rospy.get_param("~pregrasp_world_z_offset_m", 0.04))
        self.grasp_to_ee_translation = [
            float(rospy.get_param("~grasp_to_ee_tx", 0.0)),
            float(rospy.get_param("~grasp_to_ee_ty", 0.0)),
            float(rospy.get_param("~grasp_to_ee_tz", 0.0)),
        ]
        self.grasp_to_ee_quaternion_xyzw = _quat_normalize_xyzw(
            [
                float(rospy.get_param("~grasp_to_ee_qx", 0.0)),
                float(rospy.get_param("~grasp_to_ee_qy", 0.0)),
                float(rospy.get_param("~grasp_to_ee_qz", 0.0)),
                float(rospy.get_param("~grasp_to_ee_qw", 1.0)),
            ]
        )
        self.top_down_quaternion_xyzw = _quat_normalize_xyzw(
            [
                float(rospy.get_param("~top_down_qx", 1.0)),
                float(rospy.get_param("~top_down_qy", 0.0)),
                float(rospy.get_param("~top_down_qz", 0.0)),
                float(rospy.get_param("~top_down_qw", 0.0)),
            ]
        )

        self.grasp_pub = rospy.Publisher(self.adapted_grasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.pregrasp_pub = rospy.Publisher(
            self.adapted_pregrasp_pose_topic, PoseStamped, queue_size=1, latch=True
        )
        rospy.Subscriber(self.selected_grasp_topic, GraspCandidate, self.selected_grasp_cb, queue_size=1)

        rospy.loginfo(
            "Shared autonomy grasp pose adapter ready. mode=%s selected_grasp=%s adapted_grasp=%s adapted_pregrasp=%s",
            self.execution_mode,
            self.selected_grasp_topic,
            self.adapted_grasp_pose_topic,
            self.adapted_pregrasp_pose_topic,
        )

    def selected_grasp_cb(self, msg: GraspCandidate):
        approach_dir = _normalize(
            [
                msg.approach_direction.x,
                msg.approach_direction.y,
                msg.approach_direction.z,
            ]
        )

        grasp_orientation = _quat_normalize_xyzw(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        grasp_position = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        grasp_position = grasp_position + float(self.grasp_center_offset_m) * approach_dir

        if self.execution_mode == "top_down":
            base_orientation = self.top_down_quaternion_xyzw
            rotated_translation = _rotate_vec_by_quat_xyzw(self.grasp_to_ee_translation, base_orientation)
            ee_position = grasp_position + rotated_translation
            ee_orientation = base_orientation
        else:
            rotated_translation = _rotate_vec_by_quat_xyzw(self.grasp_to_ee_translation, grasp_orientation)
            ee_position = grasp_position + rotated_translation
            ee_orientation = _quat_normalize_xyzw(
                _quat_mul_xyzw(grasp_orientation, self.grasp_to_ee_quaternion_xyzw)
            )

        grasp_pose = PoseStamped()
        grasp_pose.header.stamp = rospy.Time.now()
        grasp_pose.header.frame_id = self.base_frame
        grasp_pose.pose.position.x = float(ee_position[0])
        grasp_pose.pose.position.y = float(ee_position[1])
        grasp_pose.pose.position.z = float(ee_position[2])
        grasp_pose.pose.orientation.x = float(ee_orientation[0])
        grasp_pose.pose.orientation.y = float(ee_orientation[1])
        grasp_pose.pose.orientation.z = float(ee_orientation[2])
        grasp_pose.pose.orientation.w = float(ee_orientation[3])

        pregrasp_pose = copy.deepcopy(grasp_pose)
        if self.execution_mode == "top_down":
            pregrasp_pose.pose.position.z += float(self.pregrasp_offset_m + self.pregrasp_world_z_offset_m)
        else:
            pregrasp_pose.pose.position.x += float(self.pregrasp_offset_m * approach_dir[0])
            pregrasp_pose.pose.position.y += float(self.pregrasp_offset_m * approach_dir[1])
            pregrasp_pose.pose.position.z += float(
                self.pregrasp_offset_m * approach_dir[2] + self.pregrasp_world_z_offset_m
            )

        self.grasp_pub.publish(grasp_pose)
        self.pregrasp_pub.publish(pregrasp_pose)


if __name__ == "__main__":
    SharedAutonomyGraspPoseAdapter()
    rospy.spin()
