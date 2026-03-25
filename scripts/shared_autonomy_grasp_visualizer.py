#!/usr/bin/env python3
"""RViz markers for shared autonomy grasp candidates and selected grasp."""

import copy
from typing import List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Vector3
from relaxed_ik_ros1.msg import EEVelGoals
from std_msgs.msg import ColorRGBA
from tabletop_workspace_opt.msg import GraspCandidate, GraspCandidateArray
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


class SharedAutonomyGraspVisualizer:
    def __init__(self):
        rospy.init_node("shared_autonomy_grasp_visualizer")

        self.world_frame = rospy.get_param("~world_frame", "world")
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 15.0))
        self.marker_topic = rospy.get_param("~marker_topic", "/shared_autonomy/markers")
        self.candidate_topic = rospy.get_param("~candidate_grasps_topic", "/shared_autonomy/candidate_grasps")
        self.selected_grasp_topic = rospy.get_param("~selected_grasp_topic", "/shared_autonomy/selected_grasp")
        self.ee_pose_topic = rospy.get_param("~ee_pose_topic", "/shared_autonomy/ee_pose")
        self.ee_vel_goals_topic = rospy.get_param("~ee_vel_goals_topic", "/relaxed_ik/ee_vel_goals")

        self.last_candidates: Optional[GraspCandidateArray] = None
        self.last_selected_grasp: Optional[GraspCandidate] = None
        self.last_ee_pose: Optional[PoseStamped] = None
        self.last_input_vector = np.zeros(3, dtype=np.float64)

        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        rospy.Subscriber(self.candidate_topic, GraspCandidateArray, self.candidate_cb, queue_size=1)
        rospy.Subscriber(self.selected_grasp_topic, GraspCandidate, self.selected_grasp_cb, queue_size=1)
        rospy.Subscriber(self.ee_pose_topic, PoseStamped, self.ee_pose_cb, queue_size=1)
        rospy.Subscriber(self.ee_vel_goals_topic, EEVelGoals, self.ee_vel_goals_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate_hz), self.publish_cb)

        rospy.loginfo(
            "Shared autonomy grasp visualizer ready. candidates=%s selected=%s markers=%s",
            self.candidate_topic,
            self.selected_grasp_topic,
            self.marker_topic,
        )

    def candidate_cb(self, msg: GraspCandidateArray):
        self.last_candidates = msg

    def selected_grasp_cb(self, msg: GraspCandidate):
        self.last_selected_grasp = msg

    def ee_pose_cb(self, msg: PoseStamped):
        self.last_ee_pose = msg

    def ee_vel_goals_cb(self, msg: EEVelGoals):
        if not msg.ee_vels:
            return
        twist = msg.ee_vels[0]
        self.last_input_vector = np.array(
            [twist.linear.x, twist.linear.y, twist.linear.z],
            dtype=np.float64,
        )

    def publish_cb(self, _event):
        msg = MarkerArray()
        msg.markers.append(self._delete_all_marker())

        if self.last_candidates is not None:
            msg.markers.extend(self._candidate_markers(self.last_candidates))

        if self.last_selected_grasp is not None:
            selected = self.last_selected_grasp
            if self.last_candidates is not None:
                selected = next(
                    (candidate for candidate in self.last_candidates.grasps if candidate.grasp_id == self.last_selected_grasp.grasp_id),
                    self.last_selected_grasp,
                )
            msg.markers.extend(self._selected_markers(selected))

        msg.markers.extend(self._input_markers())
        self.marker_pub.publish(msg)

    def _delete_all_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time(0)
        marker.action = Marker.DELETEALL
        return marker

    def _candidate_markers(self, candidates: GraspCandidateArray) -> List[Marker]:
        markers: List[Marker] = []
        frame_id = candidates.header.frame_id or self.world_frame

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
            sphere.header.frame_id = frame_id
            sphere.header.stamp = rospy.Time(0)
            sphere.ns = "shared_autonomy_candidate"
            sphere.id = 10 + index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = copy.deepcopy(candidate.pose)
            sphere.scale = _vector3((0.045, 0.045, 0.045))
            sphere.color = color
            sphere.lifetime = rospy.Duration(0.0)
            markers.append(sphere)

            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.header.stamp = sphere.header.stamp
            arrow.ns = "shared_autonomy_candidate"
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
            # GraspNet uses the first rotation axis as the grasp approach axis,
            # while execution approaches the object along the opposite direction.
            end = _point(
                start.x - 0.12 * direction[0],
                start.y - 0.12 * direction[1],
                start.z - 0.12 * direction[2],
            )
            arrow.points = [start, end]
            markers.append(arrow)

            text = Marker()
            text.header.frame_id = frame_id
            text.header.stamp = sphere.header.stamp
            text.ns = "shared_autonomy_candidate_text"
            text.id = 200 + index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose = copy.deepcopy(candidate.pose)
            text.pose.position.z += 0.07
            text.scale.z = 0.045
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = "%s %.2f" % (candidate.grasp_id, candidate.grasp_score)
            text.lifetime = rospy.Duration(0.0)
            markers.append(text)

        return markers

    def _selected_markers(self, candidate: GraspCandidate) -> List[Marker]:
        frame_id = self.world_frame
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time(0)
        marker.ns = "shared_autonomy_selected"
        marker.id = 300
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = copy.deepcopy(candidate.pose)
        marker.scale = _vector3((0.08, 0.08, 0.08))
        marker.color = ColorRGBA(r=1.0, g=0.95, b=0.1, a=0.35)
        marker.lifetime = rospy.Duration(0.0)

        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = marker.header.stamp
        text.ns = "shared_autonomy_selected"
        text.id = 301
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose = copy.deepcopy(candidate.pose)
        text.pose.position.z += 0.13
        text.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
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

        frame_id = self.last_ee_pose.header.frame_id or self.world_frame

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time(0)
        marker.ns = "shared_autonomy_input"
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
        text.header.frame_id = frame_id
        text.header.stamp = marker.header.stamp
        text.ns = "shared_autonomy_input"
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
    SharedAutonomyGraspVisualizer()
    rospy.spin()
