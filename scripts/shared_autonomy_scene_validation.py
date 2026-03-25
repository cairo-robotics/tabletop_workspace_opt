#!/usr/bin/env python3
"""Scene-level validation node for shared autonomy grasp selection.

Uses MuJoCo detections from the scene_swapped simulation, synthesizes a small
set of task-relevant toaster grasp candidates, bridges the end-effector pose to
PoseStamped, and publishes RViz markers for:
- target object
- candidate grasps
- selected grasp
- current operator input direction
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEVelGoals
from std_msgs.msg import ColorRGBA
from tabletop_workspace_opt.msg import GraspCandidate, GraspCandidateArray
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray


def _point(x: float, y: float, z: float) -> Point:
    return Point(x=float(x), y=float(y), z=float(z))


def _vector3(values: Tuple[float, float, float]) -> Vector3:
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _normalize(values: np.ndarray) -> Tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(values))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64), 0.0
    return values / norm, norm


def _pose_with_identity_orientation(position: np.ndarray) -> Pose:
    pose = Pose()
    pose.position = _point(position[0], position[1], position[2])
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return pose


class SharedAutonomySceneValidation:
    def __init__(self):
        rospy.init_node("shared_autonomy_scene_validation")

        self.world_frame = rospy.get_param("~world_frame", "world")
        self.target_detection_id = int(rospy.get_param("~target_detection_id", 6))
        self.target_name = str(rospy.get_param("~target_name", "toaster"))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 15.0))

        self.object_size = np.array(rospy.get_param("~target_box_size", [0.20, 0.14, 0.14]), dtype=np.float64)
        self.default_candidate_scores = rospy.get_param(
            "~candidate_scores",
            {
                "front_center": 0.92,
                "left_side": 0.78,
                "right_side": 0.80,
                "top_center": 0.70,
            },
        )
        self.default_candidate_feasibility = rospy.get_param(
            "~candidate_feasibility",
            {
                "front_center": True,
                "left_side": True,
                "right_side": True,
                "top_center": True,
            },
        )

        self.detections_topic = rospy.get_param("~detections_topic", "/mujoco_sim/detections")
        self.endpoint_state_topic = rospy.get_param("~endpoint_state_topic", "/mujoco_sim/endpoint_state")
        self.ee_vel_goals_topic = rospy.get_param("~ee_vel_goals_topic", "/relaxed_ik/ee_vel_goals")

        self.ee_pose_topic = rospy.get_param("~ee_pose_topic", "/scene_validation/ee_pose")
        self.candidate_grasps_topic = rospy.get_param("~candidate_grasps_topic", "/scene_validation/candidate_grasps")
        self.selected_grasp_topic = rospy.get_param("~selected_grasp_topic", "/scene_validation/selected_grasp")
        self.marker_topic = rospy.get_param("~marker_topic", "/scene_validation/markers")

        self.last_target_pose: Optional[Pose] = None
        self.last_ee_pose: Optional[PoseStamped] = None
        self.last_selected_grasp: Optional[GraspCandidate] = None
        self.last_input_vector = np.zeros(3, dtype=np.float64)

        self.ee_pose_pub = rospy.Publisher(self.ee_pose_topic, PoseStamped, queue_size=1)
        self.candidate_pub = rospy.Publisher(self.candidate_grasps_topic, GraspCandidateArray, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        rospy.Subscriber(self.detections_topic, Detection2DArray, self.detections_cb, queue_size=1)
        rospy.Subscriber(self.endpoint_state_topic, EndpointState, self.endpoint_state_cb, queue_size=1)
        rospy.Subscriber(self.ee_vel_goals_topic, EEVelGoals, self.ee_vel_goals_cb, queue_size=1)
        rospy.Subscriber(self.selected_grasp_topic, GraspCandidate, self.selected_grasp_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate_hz), self.publish_cb)

        rospy.loginfo(
            "Scene validation ready. target=%s detections=%s endpoint_state=%s markers=%s",
            self.target_name,
            self.detections_topic,
            self.endpoint_state_topic,
            self.marker_topic,
        )

    def detections_cb(self, msg: Detection2DArray):
        for detection in msg.detections:
            if not detection.results:
                continue
            result = detection.results[0]
            if int(result.id) != self.target_detection_id:
                continue
            self.last_target_pose = result.pose.pose
            return

    def endpoint_state_cb(self, msg: EndpointState):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        pose_msg.header.frame_id = msg.header.frame_id or self.world_frame
        pose_msg.pose = msg.pose
        self.last_ee_pose = pose_msg

    def ee_vel_goals_cb(self, msg: EEVelGoals):
        if not msg.ee_vels:
            return
        twist = msg.ee_vels[0]
        self.last_input_vector = np.array(
            [twist.linear.x, twist.linear.y, twist.linear.z],
            dtype=np.float64,
        )

    def selected_grasp_cb(self, msg: GraspCandidate):
        self.last_selected_grasp = msg

    def publish_cb(self, _event):
        if self.last_target_pose is None or self.last_ee_pose is None:
            return

        ee_pose = PoseStamped()
        ee_pose.header = self.last_ee_pose.header
        ee_pose.pose = self.last_ee_pose.pose
        if ee_pose.header.stamp == rospy.Time():
            ee_pose.header.stamp = rospy.Time.now()
        self.ee_pose_pub.publish(ee_pose)

        candidates = self._build_candidate_message()
        self.candidate_pub.publish(candidates)

        marker_array = self._build_marker_array(candidates)
        self.marker_pub.publish(marker_array)

    def _build_candidate_message(self) -> GraspCandidateArray:
        toaster_center = np.array(
            [
                self.last_target_pose.position.x,
                self.last_target_pose.position.y,
                self.last_target_pose.position.z,
            ],
            dtype=np.float64,
        )

        candidates = GraspCandidateArray()
        candidates.header.frame_id = self.world_frame
        candidates.header.stamp = rospy.Time.now()

        for grasp_id, offset, approach in self._candidate_specs():
            position = toaster_center + offset
            msg = GraspCandidate()
            msg.grasp_id = grasp_id
            msg.pose = _pose_with_identity_orientation(position)
            msg.approach_direction = _vector3(tuple(approach.tolist()))
            msg.grasp_score = float(self.default_candidate_scores.get(grasp_id, 0.75))
            msg.feasible = bool(self.default_candidate_feasibility.get(grasp_id, True))
            candidates.grasps.append(msg)

        return candidates

    def _candidate_specs(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        sx, sy, sz = self.object_size
        return [
            (
                "front_center",
                np.array([0.5 * sx + 0.07, 0.0, 0.03], dtype=np.float64),
                np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            ),
            (
                "left_side",
                np.array([0.0, 0.5 * sy + 0.08, 0.02], dtype=np.float64),
                np.array([0.0, -1.0, 0.0], dtype=np.float64),
            ),
            (
                "right_side",
                np.array([0.0, -(0.5 * sy + 0.08), 0.02], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            ),
            (
                "top_center",
                np.array([0.0, 0.0, 0.5 * sz + 0.08], dtype=np.float64),
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
            ),
        ]

    def _build_marker_array(self, candidates: GraspCandidateArray) -> MarkerArray:
        msg = MarkerArray()
        msg.markers.append(self._delete_all_marker())

        if self.last_target_pose is not None:
            msg.markers.extend(self._target_object_markers())

        msg.markers.extend(self._candidate_markers(candidates))
        msg.markers.extend(self._input_markers())

        if self.last_selected_grasp is not None:
            selected = next(
                (candidate for candidate in candidates.grasps if candidate.grasp_id == self.last_selected_grasp.grasp_id),
                self.last_selected_grasp,
            )
            msg.markers.extend(self._selected_markers(selected))

        return msg

    def _delete_all_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.action = Marker.DELETEALL
        return marker

    def _target_object_markers(self) -> List[Marker]:
        pose = Pose()
        pose.position = self.last_target_pose.position
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        cube = Marker()
        cube.header.frame_id = self.world_frame
        cube.header.stamp = rospy.Time.now()
        cube.ns = "scene_validation_target"
        cube.id = 0
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose = pose
        cube.scale = _vector3(tuple(self.object_size.tolist()))
        cube.color = ColorRGBA(r=0.95, g=0.6, b=0.15, a=0.55)
        cube.lifetime = rospy.Duration(0.0)

        text = Marker()
        text.header.frame_id = self.world_frame
        text.header.stamp = cube.header.stamp
        text.ns = "scene_validation_target"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose = pose
        text.pose.position.z += 0.18
        text.scale.z = 0.06
        text.color = ColorRGBA(r=1.0, g=0.95, b=0.9, a=1.0)
        text.text = self.target_name
        text.lifetime = rospy.Duration(0.0)

        return [cube, text]

    def _candidate_markers(self, candidates: GraspCandidateArray) -> List[Marker]:
        markers: List[Marker] = []
        for index, candidate in enumerate(candidates.grasps):
            is_selected = (
                self.last_selected_grasp is not None
                and candidate.grasp_id == self.last_selected_grasp.grasp_id
            )
            color = ColorRGBA(
                r=0.2 if not is_selected else 0.1,
                g=0.7 if not is_selected else 1.0,
                b=1.0 if not is_selected else 0.1,
                a=0.85,
            )
            if not candidate.feasible:
                color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.55)

            sphere = Marker()
            sphere.header.frame_id = self.world_frame
            sphere.header.stamp = rospy.Time.now()
            sphere.ns = "scene_validation_candidate"
            sphere.id = 10 + index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = candidate.pose
            sphere.scale = _vector3((0.045, 0.045, 0.045))
            sphere.color = color
            sphere.lifetime = rospy.Duration(0.0)
            markers.append(sphere)

            arrow = Marker()
            arrow.header.frame_id = self.world_frame
            arrow.header.stamp = sphere.header.stamp
            arrow.ns = "scene_validation_candidate"
            arrow.id = 100 + index
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.015
            arrow.scale.y = 0.03
            arrow.scale.z = 0.03
            arrow.color = color
            arrow.lifetime = rospy.Duration(0.0)
            start = candidate.pose.position
            direction = np.array(
                [
                    candidate.approach_direction.x,
                    candidate.approach_direction.y,
                    candidate.approach_direction.z,
                ],
                dtype=np.float64,
            )
            direction, _ = _normalize(direction)
            end = _point(
                start.x + 0.12 * direction[0],
                start.y + 0.12 * direction[1],
                start.z + 0.12 * direction[2],
            )
            arrow.points = [start, end]
            markers.append(arrow)

            text = Marker()
            text.header.frame_id = self.world_frame
            text.header.stamp = sphere.header.stamp
            text.ns = "scene_validation_candidate_text"
            text.id = 200 + index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = candidate.pose
            text.pose.position.z += 0.07
            text.scale.z = 0.045
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = "%s %.2f" % (candidate.grasp_id, candidate.grasp_score)
            text.lifetime = rospy.Duration(0.0)
            markers.append(text)

        return markers

    def _selected_markers(self, candidate: GraspCandidate) -> List[Marker]:
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "scene_validation_selected"
        marker.id = 300
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = candidate.pose
        marker.scale = _vector3((0.08, 0.08, 0.08))
        marker.color = ColorRGBA(r=1.0, g=0.95, b=0.1, a=0.35)
        marker.lifetime = rospy.Duration(0.0)

        text = Marker()
        text.header.frame_id = self.world_frame
        text.header.stamp = marker.header.stamp
        text.ns = "scene_validation_selected"
        text.id = 301
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose = candidate.pose
        text.pose.position.z += 0.13
        text.scale.z = 0.06
        text.color = ColorRGBA(r=1.0, g=0.95, b=0.3, a=1.0)
        text.text = "selected: %s" % candidate.grasp_id
        text.lifetime = rospy.Duration(0.0)

        return [marker, text]

    def _input_markers(self) -> List[Marker]:
        if self.last_ee_pose is None:
            return []

        direction, norm = _normalize(self.last_input_vector)
        if norm < 1e-5:
            return []

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "scene_validation_input"
        marker.id = 400
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.scale.y = 0.035
        marker.scale.z = 0.04
        marker.color = ColorRGBA(r=0.95, g=0.25, b=0.85, a=0.95)
        marker.lifetime = rospy.Duration(0.0)
        start = self.last_ee_pose.pose.position
        end = _point(
            start.x + 0.20 * direction[0],
            start.y + 0.20 * direction[1],
            start.z + 0.20 * direction[2],
        )
        marker.points = [start, end]

        text = Marker()
        text.header.frame_id = self.world_frame
        text.header.stamp = marker.header.stamp
        text.ns = "scene_validation_input"
        text.id = 401
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = end
        text.pose.orientation.w = 1.0
        text.scale.z = 0.05
        text.color = ColorRGBA(r=1.0, g=0.8, b=0.95, a=1.0)
        text.text = "input"
        text.lifetime = rospy.Duration(0.0)

        return [marker, text]


if __name__ == "__main__":
    SharedAutonomySceneValidation()
    rospy.spin()
