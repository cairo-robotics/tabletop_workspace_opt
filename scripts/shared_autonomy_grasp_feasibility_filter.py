#!/usr/bin/env python3
"""Filter grasp candidates using MoveIt IK/state validity on adapted EE targets."""

import copy
import math
import threading

import moveit_commander
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from tabletop_workspace_opt.msg import GraspCandidate, GraspCandidateArray


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


class SharedAutonomyGraspFeasibilityFilter:
    def __init__(self):
        rospy.init_node("shared_autonomy_grasp_feasibility_filter")
        moveit_commander.roscpp_initialize([])

        default_ready_joint_names = [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]
        default_ready_joint_positions = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]

        self.base_frame = rospy.get_param("~base_frame", "world")
        self.group_name = rospy.get_param("~group_name", "right_arm")
        self.ee_link = rospy.get_param("~ee_link", "right_gripper_tip")
        self.execution_mode = rospy.get_param("~execution_mode", "top_down")
        self.raw_candidate_topic = rospy.get_param("~candidate_grasps_topic", "/shared_autonomy/candidate_grasps")
        self.filtered_candidate_topic = rospy.get_param(
            "~filtered_candidate_grasps_topic", "/shared_autonomy/feasible_candidate_grasps"
        )
        self.publish_only_feasible = bool(rospy.get_param("~publish_only_feasible", True))
        self.filter_grasp_pose = bool(rospy.get_param("~filter_grasp_pose", True))
        self.ik_timeout_sec = float(rospy.get_param("~ik_timeout_sec", 0.08))
        self.process_rate_hz = float(rospy.get_param("~process_rate_hz", 0.5))
        self.reuse_last_nonempty_sec = float(rospy.get_param("~reuse_last_nonempty_sec", 1.5))

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
        self.ready_joint_names = list(rospy.get_param("~ready_joint_names", default_ready_joint_names))
        self.ready_joint_positions = [
            float(value) for value in rospy.get_param("~ready_joint_positions", default_ready_joint_positions)
        ]
        self.ready_joint_map = {
            joint_name: joint_position
            for joint_name, joint_position in zip(self.ready_joint_names, self.ready_joint_positions)
        }

        self.robot = moveit_commander.RobotCommander()
        self.filtered_pub = rospy.Publisher(
            self.filtered_candidate_topic, GraspCandidateArray, queue_size=1, latch=True
        )

        rospy.loginfo("Waiting for MoveIt IK/state validity services...")
        rospy.wait_for_service("/compute_ik")
        rospy.wait_for_service("/check_state_validity")
        self.compute_ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
        self.state_validity_srv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)

        self.lock = threading.Lock()
        self.latest_candidates = None
        self.processing = False
        self.last_nonempty_filtered_msg = None
        self.last_nonempty_stamp = None

        rospy.Subscriber(self.raw_candidate_topic, GraspCandidateArray, self.candidate_cb, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.process_rate_hz, 1e-3)), self.timer_cb)
        rospy.loginfo(
            "Shared autonomy grasp feasibility filter ready. mode=%s raw=%s filtered=%s",
            self.execution_mode,
            self.raw_candidate_topic,
            self.filtered_candidate_topic,
        )

    def _seed_robot_state(self) -> RobotState:
        state = copy.deepcopy(self.robot.get_current_state())
        state.joint_state.name = list(state.joint_state.name)
        state.joint_state.position = list(state.joint_state.position)
        state.joint_state.velocity = list(state.joint_state.velocity)
        state.joint_state.effort = list(state.joint_state.effort)
        name_to_index = {name: idx for idx, name in enumerate(state.joint_state.name)}
        for joint_name, joint_position in self.ready_joint_map.items():
            if joint_name in name_to_index:
                state.joint_state.position[name_to_index[joint_name]] = float(joint_position)
        state.joint_state.header.stamp = rospy.Time.now()
        return state

    def _adapt_candidate(self, candidate: GraspCandidate):
        approach_dir = _normalize(
            [
                candidate.approach_direction.x,
                candidate.approach_direction.y,
                candidate.approach_direction.z,
            ]
        )
        grasp_orientation = _quat_normalize_xyzw(
            [
                candidate.pose.orientation.x,
                candidate.pose.orientation.y,
                candidate.pose.orientation.z,
                candidate.pose.orientation.w,
            ]
        )
        grasp_position = np.array(
            [candidate.pose.position.x, candidate.pose.position.y, candidate.pose.position.z],
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
        return pregrasp_pose, grasp_pose

    def _compute_collision_free_ik(self, pose: PoseStamped, seed_state: RobotState):
        request = GetPositionIKRequest()
        request.ik_request.group_name = self.group_name
        request.ik_request.robot_state = copy.deepcopy(seed_state)
        request.ik_request.avoid_collisions = True
        request.ik_request.ik_link_name = self.ee_link
        request.ik_request.pose_stamped = copy.deepcopy(pose)
        request.ik_request.timeout = rospy.Duration.from_sec(self.ik_timeout_sec)
        response = self.compute_ik_srv(request)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, None

        validity_request = GetStateValidityRequest()
        validity_request.robot_state = response.solution
        validity_request.group_name = self.group_name
        validity_response = self.state_validity_srv(validity_request)
        if not validity_response.valid:
            return False, None
        return True, response.solution

    def candidate_cb(self, msg: GraspCandidateArray):
        with self.lock:
            self.latest_candidates = copy.deepcopy(msg)

    def timer_cb(self, _event):
        with self.lock:
            if self.processing or self.latest_candidates is None:
                return
            msg = copy.deepcopy(self.latest_candidates)
            self.processing = True

        try:
            self._process_candidates(msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Grasp feasibility filter cycle failed: %s", exc)
        finally:
            with self.lock:
                self.processing = False

    def _process_candidates(self, msg: GraspCandidateArray):
        seed_state = self._seed_robot_state()
        filtered_msg = GraspCandidateArray()
        filtered_msg.header = msg.header

        feasible_count = 0
        for candidate in msg.grasps:
            pregrasp_pose, grasp_pose = self._adapt_candidate(candidate)
            try:
                pregrasp_ok, pregrasp_solution = self._compute_collision_free_ik(pregrasp_pose, seed_state)
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(2.0, "compute_ik failed for pregrasp: %s", exc)
                pregrasp_ok, pregrasp_solution = False, None

            grasp_ok = True
            if pregrasp_ok and self.filter_grasp_pose:
                try:
                    grasp_ok, _ = self._compute_collision_free_ik(
                        grasp_pose,
                        pregrasp_solution if pregrasp_solution is not None else seed_state,
                    )
                except rospy.ServiceException as exc:
                    rospy.logwarn_throttle(2.0, "compute_ik failed for grasp: %s", exc)
                    grasp_ok = False

            filtered_candidate = copy.deepcopy(candidate)
            filtered_candidate.feasible = bool(candidate.feasible and pregrasp_ok and grasp_ok)
            if filtered_candidate.feasible:
                feasible_count += 1
            if (not self.publish_only_feasible) or filtered_candidate.feasible:
                filtered_msg.grasps.append(filtered_candidate)

        msg_to_publish = filtered_msg
        if filtered_msg.grasps:
            self.last_nonempty_filtered_msg = copy.deepcopy(filtered_msg)
            self.last_nonempty_stamp = rospy.Time.now()
        elif (
            self.last_nonempty_filtered_msg is not None
            and self.last_nonempty_stamp is not None
            and (rospy.Time.now() - self.last_nonempty_stamp).to_sec() <= self.reuse_last_nonempty_sec
        ):
            msg_to_publish = copy.deepcopy(self.last_nonempty_filtered_msg)
            msg_to_publish.header = filtered_msg.header
            rospy.loginfo_throttle(
                2.0,
                "Grasp feasibility filter reused last non-empty candidate set for %.2fs.",
                self.reuse_last_nonempty_sec,
            )

        self.filtered_pub.publish(msg_to_publish)
        rospy.loginfo_throttle(
            2.0,
            "Grasp feasibility filter kept %d/%d candidates on %s.",
            feasible_count,
            len(msg.grasps),
            self.filtered_candidate_topic,
        )


if __name__ == "__main__":
    SharedAutonomyGraspFeasibilityFilter()
    rospy.spin()
