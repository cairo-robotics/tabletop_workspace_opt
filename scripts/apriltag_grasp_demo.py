#!/usr/bin/env python3
"""Simple wrist-camera AprilTag grasp demo."""

import copy
import math
import os
import sys

import numpy as np
import rospy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

try:
    from apriltag_camera_calibration import AprilTagCameraCalibration
except ImportError:
    import rospkg

    pkg_dir = rospkg.RosPack().get_path("tabletop_workspace_opt")
    scripts_dir = os.path.join(pkg_dir, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from apriltag_camera_calibration import AprilTagCameraCalibration


def _normalize(vec):
    arr = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return arr / norm


def _quat_normalize_xyzw(q):
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([float(v) / norm for v in q], dtype=np.float64)


def _quat_to_matrix(q):
    x, y, z, w = _quat_normalize_xyzw(q)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(rot):
    m = np.array(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize_xyzw([x, y, z, w])


def _pose_dict_to_matrix(pose_dict):
    position = pose_dict.get("position", [0.0, 0.0, 0.0])
    orientation = pose_dict.get("orientation", [0.0, 0.0, 0.0, 1.0])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix(orientation)
    matrix[:3, 3] = np.array(position, dtype=np.float64)
    return matrix


def _matrix_to_pose(matrix):
    pose = Pose()
    quat = _matrix_to_quat_xyzw(matrix[:3, :3])
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


def _make_pose_stamped(frame_id, stamp, matrix):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose = _matrix_to_pose(matrix)
    return msg


def _make_point(xyz):
    point = Point()
    point.x = float(xyz[0])
    point.y = float(xyz[1])
    point.z = float(xyz[2])
    return point


def _tf_to_matrix(tf_msg: TransformStamped):
    tx = tf_msg.transform.translation.x
    ty = tf_msg.transform.translation.y
    tz = tf_msg.transform.translation.z
    qx = tf_msg.transform.rotation.x
    qy = tf_msg.transform.rotation.y
    qz = tf_msg.transform.rotation.z
    qw = tf_msg.transform.rotation.w
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix([qx, qy, qz, qw])
    matrix[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return matrix


class AprilTagGraspDemo:
    def __init__(self):
        rospy.init_node("apriltag_grasp_demo")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_yaml = os.path.join(package_root, "config", "apriltag_grasp_demo.yaml")

        self.tag_size = float(rospy.get_param("~tag_size", 0.05))
        self.tag_family = str(rospy.get_param("~tag_family", "tag36h11")).strip()
        self.target_tag_id = int(rospy.get_param("~target_tag_id", -1))
        self.selected_template_id = str(rospy.get_param("~selected_template_id", "")).strip()
        self.template_yaml = os.path.expanduser(rospy.get_param("~template_yaml", default_yaml))
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.camera_frame = str(rospy.get_param("~camera_frame", "camera_color_optical_frame")).strip()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info")).strip()
        self.use_image_header_frame = bool(rospy.get_param("~use_image_header_frame", True))
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 0.2))
        self.use_latest_tf = bool(rospy.get_param("~use_latest_tf", True))
        self.stale_timeout_sec = float(rospy.get_param("~stale_timeout_sec", 1.0))
        self.base_tag_pose_topic = str(rospy.get_param("~base_tag_pose_topic", "~base_tag_pose")).strip()
        self.selected_template_topic = str(rospy.get_param("~selected_template_topic", "~selected_template")).strip()
        self.pregrasp_pose_topic = str(rospy.get_param("~pregrasp_pose_topic", "~pregrasp_pose")).strip()
        self.grasp_pose_topic = str(rospy.get_param("~grasp_pose_topic", "~grasp_pose")).strip()
        self.markers_topic = str(rospy.get_param("~markers_topic", "~markers")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()

        self.bridge = CvBridge()
        self.detector = AprilTagCameraCalibration(self.tag_size, self.tag_family)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cam_matrix = None
        self.cam_dist = None
        self.templates = self._load_templates()
        self.last_detection_time = None
        self.last_status = ""

        self.base_tag_pose_pub = rospy.Publisher(self.base_tag_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.selected_template_pub = rospy.Publisher(self.selected_template_topic, String, queue_size=1, latch=True)
        self.pregrasp_pose_pub = rospy.Publisher(self.pregrasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.grasp_pose_pub = rospy.Publisher(self.grasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        self.stale_timer = rospy.Timer(rospy.Duration(0.2), self.stale_timer_cb)

        self._publish_status(
            "waiting_for_camera_info image_topic={} camera_info_topic={} template_count={}".format(
                self.image_topic,
                self.camera_info_topic,
                len(self.templates),
            )
        )
        rospy.loginfo(
            "AprilTag grasp demo ready. base_frame=%s camera_frame=%s target_tag_id=%d selected_template_id=%s use_latest_tf=%s",
            self.base_frame,
            self.camera_frame,
            self.target_tag_id,
            self.selected_template_id or "<first>",
            self.use_latest_tf,
        )

    def _load_templates(self):
        if not os.path.exists(self.template_yaml):
            raise RuntimeError("Template YAML not found: {}".format(self.template_yaml))

        with open(self.template_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        raw_templates = data.get("templates", [])
        if not isinstance(raw_templates, list) or not raw_templates:
            raise RuntimeError("Expected a non-empty `templates` list in {}".format(self.template_yaml))

        templates = []
        for index, item in enumerate(raw_templates):
            if not isinstance(item, dict):
                raise RuntimeError("Template {} is not a dictionary.".format(index))
            grasp_id = str(item.get("grasp_id", "template_{}".format(index))).strip()
            if not grasp_id:
                raise RuntimeError("Template {} has an empty grasp_id.".format(index))
            grasp_pose = item.get("grasp_pose")
            if not isinstance(grasp_pose, dict):
                raise RuntimeError("Template {} is missing grasp_pose.".format(grasp_id))
            pregrasp_pose = item.get("pregrasp_pose")
            template = {
                "grasp_id": grasp_id,
                "description": str(item.get("description", "")).strip(),
                "pregrasp_pose": copy.deepcopy(pregrasp_pose) if isinstance(pregrasp_pose, dict) else None,
                "grasp_pose": copy.deepcopy(grasp_pose),
            }
            explicit_approach = item.get("approach_direction")
            if isinstance(explicit_approach, list) and len(explicit_approach) == 3:
                template["approach_direction"] = _normalize(explicit_approach)
            else:
                template["approach_direction"] = self._infer_approach_direction(template)
            templates.append(template)

        return templates

    def _infer_approach_direction(self, template):
        pregrasp = template.get("pregrasp_pose")
        grasp = template.get("grasp_pose")
        if isinstance(pregrasp, dict) and isinstance(grasp, dict):
            pre = np.array(pregrasp.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            post = np.array(grasp.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            delta = post - pre
            direction = _normalize(delta)
            if np.linalg.norm(direction) > 1e-9:
                return direction
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        self.status_pub.publish(String(data=text))

    def camera_info_cb(self, msg: CameraInfo):
        self.cam_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self.cam_dist = np.array(msg.D, dtype=np.float64)

    def image_cb(self, msg: Image):
        if self.cam_matrix is None:
            return

        if self.use_image_header_frame and msg.header.frame_id:
            self.camera_frame = str(msg.header.frame_id)

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to decode image: %s", exc)
            return

        detection = self.detector.detect_apriltag(image, self.cam_matrix, self.cam_dist)
        if detection is None or detection[0] is None:
            return

        T_tag_cam, tag_id = detection
        if self.target_tag_id >= 0 and tag_id != self.target_tag_id:
            self._publish_status("detected_tag_id={} waiting_for={}".format(tag_id, self.target_tag_id))
            return

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time(0)
        tf_stamp = rospy.Time(0) if self.use_latest_tf else stamp
        T_base_cam = self._lookup_base_to_camera(tf_stamp)
        if T_base_cam is None:
            return

        T_base_tag = T_base_cam @ np.linalg.inv(T_tag_cam)
        self.last_detection_time = rospy.Time.now()
        self._publish_transformed_templates(T_base_tag, tag_id, stamp)

    def _lookup_base_to_camera(self, stamp):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                stamp,
                rospy.Duration(self.tf_timeout_sec),
            )
            return _tf_to_matrix(tf_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_timeout_sec),
            )
            return _tf_to_matrix(tf_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0,
                "Failed to lookup TF %s -> %s: %s",
                self.base_frame,
                self.camera_frame,
                exc,
            )
            self._publish_status("tf_lookup_failed {}->{}".format(self.base_frame, self.camera_frame))
            return None

    def _publish_transformed_templates(self, T_base_tag, tag_id, stamp):
        base_tag_pose = _make_pose_stamped(self.base_frame, stamp, T_base_tag)
        self.base_tag_pose_pub.publish(base_tag_pose)

        selected_template_name = None
        selected_pregrasp_pose = None
        selected_grasp_pose = None
        marker_specs = []

        for index, template in enumerate(self.templates):
            T_tag_grasp = _pose_dict_to_matrix(template["grasp_pose"])
            T_base_grasp = T_base_tag @ T_tag_grasp
            grasp_pose = _make_pose_stamped(self.base_frame, stamp, T_base_grasp)

            pregrasp_pose = None
            if template["pregrasp_pose"] is not None:
                T_tag_pregrasp = _pose_dict_to_matrix(template["pregrasp_pose"])
                T_base_pregrasp = T_base_tag @ T_tag_pregrasp
                pregrasp_pose = _make_pose_stamped(self.base_frame, stamp, T_base_pregrasp)

            marker_specs.append(
                {
                    "index": index,
                    "grasp_id": template["grasp_id"],
                    "tag_id": tag_id,
                    "pregrasp_pose": pregrasp_pose,
                    "grasp_pose": grasp_pose,
                }
            )

            if self._is_selected_template(template, index):
                selected_template_name = template["grasp_id"]
                selected_pregrasp_pose = pregrasp_pose
                selected_grasp_pose = grasp_pose

        if selected_template_name is None and self.templates:
            selected_template_name = self.templates[0]["grasp_id"]
            selected_grasp_pose = _make_pose_stamped(
                self.base_frame,
                stamp,
                T_base_tag @ _pose_dict_to_matrix(self.templates[0]["grasp_pose"]),
            )
            if self.templates[0]["pregrasp_pose"] is not None:
                selected_pregrasp_pose = _make_pose_stamped(
                    self.base_frame,
                    stamp,
                    T_base_tag @ _pose_dict_to_matrix(self.templates[0]["pregrasp_pose"]),
                )

        if selected_template_name is not None:
            self.selected_template_pub.publish(String(data=selected_template_name))
        if selected_pregrasp_pose is not None:
            self.pregrasp_pose_pub.publish(selected_pregrasp_pose)
        if selected_grasp_pose is not None:
            self.grasp_pose_pub.publish(selected_grasp_pose)

        self.markers_pub.publish(self._make_markers(stamp, T_base_tag, tag_id, marker_specs))
        self._publish_status(
            "tracking_tag_id={} selected_template={} templates={}".format(
                tag_id,
                selected_template_name if selected_template_name is not None else "none",
                len(self.templates),
            )
        )

    def _is_selected_template(self, template, index):
        if self.selected_template_id:
            return template["grasp_id"] == self.selected_template_id
        return index == 0

    def _make_markers(self, stamp, T_base_tag, tag_id, marker_specs):
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        tag_marker = Marker()
        tag_marker.header.frame_id = self.base_frame
        tag_marker.header.stamp = stamp
        tag_marker.ns = "tag_origin"
        tag_marker.id = 0
        tag_marker.type = Marker.SPHERE
        tag_marker.action = Marker.ADD
        tag_marker.pose = _matrix_to_pose(T_base_tag)
        tag_marker.scale.x = 0.03
        tag_marker.scale.y = 0.03
        tag_marker.scale.z = 0.03
        tag_marker.color = ColorRGBA(0.1, 0.8, 0.2, 0.95)
        markers.markers.append(tag_marker)

        text = Marker()
        text.header.frame_id = self.base_frame
        text.header.stamp = stamp
        text.ns = "tag_label"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose = _matrix_to_pose(T_base_tag)
        text.pose.position.z += 0.05
        text.scale.z = 0.035
        text.color = ColorRGBA(1.0, 1.0, 1.0, 0.95)
        text.text = "tag {}".format(tag_id)
        markers.markers.append(text)

        marker_id = 10
        for spec in marker_specs:
            if spec["pregrasp_pose"] is not None:
                pre_marker = Marker()
                pre_marker.header.frame_id = self.base_frame
                pre_marker.header.stamp = stamp
                pre_marker.ns = "pregrasp"
                pre_marker.id = marker_id
                pre_marker.type = Marker.SPHERE
                pre_marker.action = Marker.ADD
                pre_marker.pose = spec["pregrasp_pose"].pose
                pre_marker.scale.x = 0.025
                pre_marker.scale.y = 0.025
                pre_marker.scale.z = 0.025
                pre_marker.color = ColorRGBA(0.2, 0.6, 1.0, 0.85)
                markers.markers.append(pre_marker)
                marker_id += 1

            grasp_marker = Marker()
            grasp_marker.header.frame_id = self.base_frame
            grasp_marker.header.stamp = stamp
            grasp_marker.ns = "grasp"
            grasp_marker.id = marker_id
            grasp_marker.type = Marker.CUBE
            grasp_marker.action = Marker.ADD
            grasp_marker.pose = spec["grasp_pose"].pose
            grasp_marker.scale.x = 0.035
            grasp_marker.scale.y = 0.02
            grasp_marker.scale.z = 0.02
            grasp_marker.color = ColorRGBA(1.0, 0.4, 0.1, 0.9)
            markers.markers.append(grasp_marker)
            marker_id += 1

            if spec["pregrasp_pose"] is not None:
                approach_marker = Marker()
                approach_marker.header.frame_id = self.base_frame
                approach_marker.header.stamp = stamp
                approach_marker.ns = "approach"
                approach_marker.id = marker_id
                approach_marker.type = Marker.ARROW
                approach_marker.action = Marker.ADD
                approach_marker.scale.x = 0.008
                approach_marker.scale.y = 0.015
                approach_marker.scale.z = 0.02
                approach_marker.color = ColorRGBA(1.0, 0.9, 0.1, 0.9)
                approach_marker.points = [
                    _make_point(
                        [
                            spec["pregrasp_pose"].pose.position.x,
                            spec["pregrasp_pose"].pose.position.y,
                            spec["pregrasp_pose"].pose.position.z,
                        ]
                    ),
                    _make_point(
                        [
                            spec["grasp_pose"].pose.position.x,
                            spec["grasp_pose"].pose.position.y,
                            spec["grasp_pose"].pose.position.z,
                        ]
                    ),
                ]
                markers.markers.append(approach_marker)
                marker_id += 1

            label = Marker()
            label.header.frame_id = self.base_frame
            label.header.stamp = stamp
            label.ns = "labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose = copy.deepcopy(spec["grasp_pose"].pose)
            label.pose.position.z += 0.04
            label.scale.z = 0.03
            label.color = ColorRGBA(1.0, 1.0, 1.0, 0.95)
            label.text = spec["grasp_id"]
            markers.markers.append(label)
            marker_id += 1

        return markers

    def stale_timer_cb(self, _event):
        if self.last_detection_time is None:
            return
        if (rospy.Time.now() - self.last_detection_time).to_sec() <= self.stale_timeout_sec:
            return
        self.last_detection_time = None
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        self.markers_pub.publish(markers)
        self._publish_status("tag_detection_stale")


if __name__ == "__main__":
    AprilTagGraspDemo()
    rospy.spin()
