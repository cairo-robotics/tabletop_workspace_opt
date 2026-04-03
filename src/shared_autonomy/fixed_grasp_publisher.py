#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish fixed grasp candidates from YAML for RViz visualization."""

import os
import rospy
import yaml
from geometry_msgs.msg import Pose, PoseArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def make_pose(pose_dict):
    pose = Pose()
    pos = pose_dict.get("position", [0.0, 0.0, 0.0])
    ori = pose_dict.get("orientation", [0.0, 0.0, 0.0, 1.0])
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])
    pose.orientation.x = float(ori[0])
    pose.orientation.y = float(ori[1])
    pose.orientation.z = float(ori[2])
    pose.orientation.w = float(ori[3])
    return pose


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def add_vectors(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale_vector(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


def make_point(xyz):
    point = Point()
    point.x = float(xyz[0])
    point.y = float(xyz[1])
    point.z = float(xyz[2])
    return point


class FixedGraspPublisher:
    def __init__(self):
        rospy.init_node("fixed_grasp_publisher")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_yaml = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")

        self.yaml_path = os.path.expanduser(rospy.get_param("~fixed_grasp_yaml", default_yaml))
        self.rate_hz = float(rospy.get_param("~rate_hz", 2.0))
        self.pregrasp_scale = float(rospy.get_param("~pregrasp_scale", 0.12))
        self.grasp_scale = float(rospy.get_param("~grasp_scale", 0.08))
        self.text_scale = float(rospy.get_param("~text_scale", 0.04))
        self.axis_radius = float(rospy.get_param("~axis_radius", 0.008))
        self.approach_scale = float(rospy.get_param("~approach_scale", 0.14))
        self.approach_axis = str(rospy.get_param("~approach_axis", "z")).strip().lower()

        self.pub_markers = rospy.Publisher("~markers", MarkerArray, queue_size=1, latch=True)
        self.pub_pregrasp = rospy.Publisher("~pregrasp_poses", PoseArray, queue_size=1, latch=True)
        self.pub_grasp = rospy.Publisher("~grasp_poses", PoseArray, queue_size=1, latch=True)

        self.frame_id = "base"
        self.grasps = []
        self._load_grasps()

    def _load_grasps(self):
        if not os.path.exists(self.yaml_path):
            raise RuntimeError(f"YAML file not found: {self.yaml_path}")

        with open(self.yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        self.frame_id = str(data.get("frame_id", "base"))
        grasps = data.get("grasps", [])
        if not isinstance(grasps, list):
            raise RuntimeError("Expected `grasps` to be a list in the YAML file.")

        self.grasps = grasps
        rospy.loginfo(
            "[fixed_grasp_publisher] loaded %d grasp candidates from %s (approach_axis=%s)",
            len(self.grasps),
            self.yaml_path,
            self.approach_axis,
        )

    def _append_approach_marker(self, markers, marker_id, stamp, ns, pose_dict, scale):
        pose = make_pose(pose_dict)
        rotation = quaternion_to_rotation_matrix(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        origin = (pose.position.x, pose.position.y, pose.position.z)
        axis_lookup = {
            "x": rotation[0],
            "-x": scale_vector(rotation[0], -1.0),
            "y": rotation[1],
            "-y": scale_vector(rotation[1], -1.0),
            "z": rotation[2],
            "-z": scale_vector(rotation[2], -1.0),
        }
        direction = axis_lookup.get(self.approach_axis, scale_vector(rotation[2], -1.0))
        tip = add_vectors(origin, scale_vector(direction, scale))

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = self.axis_radius * 1.4
        marker.scale.y = self.axis_radius * 2.4
        marker.scale.z = self.axis_radius * 2.8
        marker.color = ColorRGBA(1.0, 1.0, 0.1, 0.98)
        marker.points = [make_point(origin), make_point(tip)]
        markers.markers.append(marker)
        return marker_id + 1

    def _make_pose_arrays(self):
        pregrasp_array = PoseArray()
        pregrasp_array.header.frame_id = self.frame_id
        pregrasp_array.header.stamp = rospy.Time.now()

        grasp_array = PoseArray()
        grasp_array.header.frame_id = self.frame_id
        grasp_array.header.stamp = pregrasp_array.header.stamp

        for grasp in self.grasps:
            if "pregrasp_pose" in grasp:
                pregrasp_array.poses.append(make_pose(grasp["pregrasp_pose"]))
            if "grasp_pose" in grasp:
                grasp_array.poses.append(make_pose(grasp["grasp_pose"]))

        return pregrasp_array, grasp_array

    def _make_markers(self):
        markers = MarkerArray()
        stamp = rospy.Time.now()
        marker_id = 0

        text_color = ColorRGBA(1.0, 1.0, 1.0, 0.95)

        for grasp in self.grasps:
            grasp_id = str(grasp.get("grasp_id", f"grasp_{marker_id}"))

            for stage_name, scale in (
                ("pregrasp_pose", self.pregrasp_scale),
                ("grasp_pose", self.grasp_scale),
            ):
                pose_dict = grasp.get(stage_name)
                if not pose_dict:
                    continue

                marker_id = self._append_approach_marker(
                    markers,
                    marker_id,
                    stamp,
                    f"{stage_name}_approach",
                    pose_dict,
                    self.approach_scale if stage_name == "pregrasp_pose" else self.approach_scale * 0.8,
                )

                text = Marker()
                text.header.frame_id = self.frame_id
                text.header.stamp = stamp
                text.ns = f"{stage_name}_label"
                text.id = marker_id
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose = make_pose(pose_dict)
                text.pose.position.z += 0.05
                text.scale.z = self.text_scale
                text.color = text_color
                stage_label = "pre" if stage_name == "pregrasp_pose" else "grasp"
                text.text = f"{grasp_id} ({stage_label})"
                markers.markers.append(text)
                marker_id += 1

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.insert(0, delete_all)
        return markers

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            pregrasp_array, grasp_array = self._make_pose_arrays()
            markers = self._make_markers()
            self.pub_pregrasp.publish(pregrasp_array)
            self.pub_grasp.publish(grasp_array)
            self.pub_markers.publish(markers)
            rate.sleep()


def main():
    FixedGraspPublisher().run()


if __name__ == "__main__":
    main()
