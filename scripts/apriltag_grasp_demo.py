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
from intera_core_msgs.msg import EndpointState
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


def _rpy_deg_to_matrix(roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


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


def _quat_angle_deg_xyzw(q0, q1):
    a = _quat_normalize_xyzw(q0)
    b = _quat_normalize_xyzw(q1)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _quat_slerp_xyzw(q0, q1, t):
    qa = _quat_normalize_xyzw(q0)
    qb = _quat_normalize_xyzw(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _quat_normalize_xyzw((1.0 - t) * qa + t * qb)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    if abs(sin_theta_0) < 1e-9:
        return qa
    theta = theta_0 * float(np.clip(t, 0.0, 1.0))
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize_xyzw(s0 * qa + s1 * qb)


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


def _format_matrix_4x4(mat):
    rows = []
    for row in mat:
        rows.append("[{: .6f} {: .6f} {: .6f} {: .6f}]".format(float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return "\n".join(rows)


def _safe_normalize(vec, fallback):
    arr = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.array(fallback, dtype=np.float64)
    return arr / norm


def _parse_vec3_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)
    if isinstance(raw, str):
        txt = raw.strip()
        if "$(" in txt:
            rospy.logwarn("[apriltag_grasp_demo] unresolved launch arg for %s: %s. using default=%s", name, txt, default)
            return np.array(default, dtype=np.float64)
        txt = txt.replace("[", "").replace("]", "").replace(",", " ")
        parts = [p for p in txt.split() if p]
        if len(parts) == 3:
            try:
                return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
            except ValueError:
                pass
    rospy.logwarn("[apriltag_grasp_demo] invalid vec3 param %s=%r. using default=%s", name, raw, default)
    return np.array(default, dtype=np.float64)


def _compute_axis_alignment_rotation(forward_w, up_ref, ee_forward_axis, ee_up_axis):
    forward_w = _safe_normalize(forward_w, [0.0, 0.0, -1.0])
    up_ref = _safe_normalize(up_ref, [0.0, 0.0, 1.0])
    right_w = np.cross(forward_w, up_ref)
    if np.linalg.norm(right_w) < 1e-6:
        right_w = np.cross(forward_w, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(right_w) < 1e-6:
            right_w = np.cross(forward_w, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    right_w = _safe_normalize(right_w, [1.0, 0.0, 0.0])
    up_w = _safe_normalize(np.cross(right_w, forward_w), [0.0, 0.0, 1.0])
    right_w = _safe_normalize(np.cross(forward_w, up_w), [1.0, 0.0, 0.0])

    f_l = _safe_normalize(ee_forward_axis, [0.0, 0.0, 1.0])
    u_l_raw = _safe_normalize(ee_up_axis, [0.0, 1.0, 0.0])
    u_l = _safe_normalize(u_l_raw - np.dot(u_l_raw, f_l) * f_l, [0.0, 1.0, 0.0])
    r_l = _safe_normalize(np.cross(f_l, u_l), [1.0, 0.0, 0.0])
    u_l = _safe_normalize(np.cross(r_l, f_l), [0.0, 1.0, 0.0])

    L = np.column_stack((f_l, u_l, r_l))
    W = np.column_stack((forward_w, up_w, right_w))
    return W @ L.T


def _compute_look_at_rotation(origin, target, world_up, ee_forward_axis, ee_up_axis):
    return _compute_axis_alignment_rotation(
        forward_w=np.array(target, dtype=np.float64) - np.array(origin, dtype=np.float64),
        up_ref=world_up,
        ee_forward_axis=ee_forward_axis,
        ee_up_axis=ee_up_axis,
    )


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
        self.end_effector_topic = str(rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")).strip()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info")).strip()
        self.use_image_header_frame = bool(rospy.get_param("~use_image_header_frame", True))
        self.tf_timeout_sec = float(rospy.get_param("~tf_timeout_sec", 0.2))
        self.use_latest_tf = bool(rospy.get_param("~use_latest_tf", True))
        self.invert_tag_transform = bool(rospy.get_param("~invert_tag_transform", False))
        self.invert_base_camera_transform = bool(rospy.get_param("~invert_base_camera_transform", False))
        self.stale_timeout_sec = float(rospy.get_param("~stale_timeout_sec", 1.0))
        self.base_tag_pose_topic = str(rospy.get_param("~base_tag_pose_topic", "~base_tag_pose")).strip()
        self.selected_template_topic = str(rospy.get_param("~selected_template_topic", "~selected_template")).strip()
        self.pregrasp_pose_topic = str(rospy.get_param("~pregrasp_pose_topic", "~pregrasp_pose")).strip()
        self.grasp_pose_topic = str(rospy.get_param("~grasp_pose_topic", "~grasp_pose")).strip()
        self.markers_topic = str(rospy.get_param("~markers_topic", "~markers")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.print_tag2base_matrix = bool(rospy.get_param("~print_tag2base_matrix", True))
        self.tag2base_print_period_sec = float(rospy.get_param("~tag2base_print_period_sec", 5.0))
        self.template_roll_deg = float(rospy.get_param("~template_roll_deg", 0.0))
        self.template_pitch_deg = float(rospy.get_param("~template_pitch_deg", 0.0))
        self.template_yaw_deg = float(rospy.get_param("~template_yaw_deg", 0.0))
        self.template_rot_correction = _rpy_deg_to_matrix(
            self.template_roll_deg,
            self.template_pitch_deg,
            self.template_yaw_deg,
        )
        self.procedural_tag_grasp = bool(rospy.get_param("~procedural_tag_grasp", False))
        self.procedural_grasp_id = str(rospy.get_param("~procedural_grasp_id", "tag_axis_grasp")).strip()
        self.procedural_offset_mode = str(rospy.get_param("~procedural_offset_mode", "camera_ray")).strip().lower()
        self.procedural_grasp_distance_m = float(rospy.get_param("~procedural_grasp_distance_m", 0.10))
        self.procedural_min_grasp_z = float(rospy.get_param("~procedural_min_grasp_z", -1.0))
        self.procedural_grasp_offset_tag = _parse_vec3_param("~procedural_grasp_offset_tag", [0.0, 0.0, 0.10])
        self.procedural_pregrasp_offset_axis_tag = _parse_vec3_param("~procedural_pregrasp_offset_axis_tag", [0.0, 0.0, 1.0])
        self.procedural_pregrasp_offset_distance_m = float(rospy.get_param("~procedural_pregrasp_offset_distance_m", 0.08))
        self.procedural_world_up_axis = _parse_vec3_param("~procedural_world_up_axis", [0.0, 0.0, 1.0])
        self.procedural_ee_forward_axis = _parse_vec3_param("~procedural_ee_forward_axis", [0.0, 0.0, 1.0])
        self.procedural_ee_up_axis = _parse_vec3_param("~procedural_ee_up_axis", [0.0, 1.0, 0.0])
        self.procedural_tag_up_axis_tag = _parse_vec3_param("~procedural_tag_up_axis_tag", [1.0, 0.0, 0.0])
        self.procedural_orientation_mode = str(
            rospy.get_param("~procedural_orientation_mode", "tag_axes_current_limited")
        ).strip().lower()
        self.procedural_orientation_max_delta_deg = float(
            rospy.get_param("~procedural_orientation_max_delta_deg", 20.0)
        )
        self.procedural_topdown_dot_threshold = float(
            rospy.get_param("~procedural_topdown_dot_threshold", 0.75)
        )

        self.bridge = CvBridge()
        self.detector = AprilTagCameraCalibration(self.tag_size, self.tag_family)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cam_matrix = None
        self.cam_dist = None
        self.templates = self._load_templates()
        self.last_detection_time = None
        self.last_status = ""
        self.last_tag2base_print_time = rospy.Time(0)
        self.latest_ee_pose = None

        self.base_tag_pose_pub = rospy.Publisher(self.base_tag_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.selected_template_pub = rospy.Publisher(self.selected_template_topic, String, queue_size=1, latch=True)
        self.pregrasp_pose_pub = rospy.Publisher(self.pregrasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.grasp_pose_pub = rospy.Publisher(self.grasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.end_effector_topic, EndpointState, self.endpoint_cb, queue_size=10)
        self.stale_timer = rospy.Timer(rospy.Duration(0.2), self.stale_timer_cb)

        self._publish_status(
            "waiting_for_camera_info image_topic={} camera_info_topic={} template_count={}".format(
            self.image_topic,
            self.camera_info_topic,
            len(self.templates),
        )
        )
        rospy.loginfo(
            "AprilTag grasp demo ready. base_frame=%s camera_frame=%s target_tag_id=%d selected_template_id=%s use_latest_tf=%s invert_tag_transform=%s invert_base_camera_transform=%s template_rpy_deg=[%.1f, %.1f, %.1f]",
            self.base_frame,
            self.camera_frame,
            self.target_tag_id,
            self.selected_template_id or "<first>",
            self.use_latest_tf,
            str(self.invert_tag_transform),
            str(self.invert_base_camera_transform),
            self.template_roll_deg,
            self.template_pitch_deg,
            self.template_yaw_deg,
        )

    def endpoint_cb(self, msg):
        self.latest_ee_pose = copy.deepcopy(msg.pose)

    def _maybe_limit_orientation(self, R_base_target):
        if self.latest_ee_pose is None:
            return R_base_target

        current_q = [
            self.latest_ee_pose.orientation.x,
            self.latest_ee_pose.orientation.y,
            self.latest_ee_pose.orientation.z,
            self.latest_ee_pose.orientation.w,
        ]
        current_R = _quat_to_matrix(current_q)
        mode = self.procedural_orientation_mode
        if mode in ("current_ee_locked", "current_locked", "current"):
            return current_R
        if mode not in ("tag_axes_current_limited", "current_limited", "limited"):
            return R_base_target

        target_q = _matrix_to_quat_xyzw(R_base_target)
        delta_deg = _quat_angle_deg_xyzw(current_q, target_q)
        max_delta = max(0.0, float(self.procedural_orientation_max_delta_deg))
        if delta_deg <= max_delta or max_delta <= 1e-6:
            if max_delta <= 1e-6:
                return current_R
            return R_base_target
        blend = max_delta / max(delta_deg, 1e-6)
        limited_q = _quat_slerp_xyzw(current_q, target_q, blend)
        return _quat_to_matrix(limited_q)

    def _choose_rotation_closest_to_current(self, rotations):
        if not rotations:
            raise ValueError("rotations must be non-empty")
        if self.latest_ee_pose is None or len(rotations) == 1:
            return rotations[0]

        current_q = [
            self.latest_ee_pose.orientation.x,
            self.latest_ee_pose.orientation.y,
            self.latest_ee_pose.orientation.z,
            self.latest_ee_pose.orientation.w,
        ]
        best_rot = rotations[0]
        best_delta = None
        for rot in rotations:
            delta = _quat_angle_deg_xyzw(current_q, _matrix_to_quat_xyzw(rot))
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_rot = rot
        return best_rot

    def _resolve_tag_up_reference(self, axis_base, R_base_tag, world_up):
        tag_up_axis = R_base_tag @ _safe_normalize(self.procedural_tag_up_axis_tag, [1.0, 0.0, 0.0])
        approach_dir = _safe_normalize(-axis_base, [0.0, 0.0, -1.0])
        mode = self.procedural_orientation_mode

        if mode in ("side_grasp_world_aligned", "side_world_aligned", "world_up"):
            return world_up
        if mode in ("tag_axes", "tag_axes_current_limited", "current_limited", "limited", "current_ee_locked", "current_locked", "current"):
            return tag_up_axis
        if mode in ("auto_surface_align", "auto_grasp", "auto"):
            dot_up = abs(float(np.dot(approach_dir, _safe_normalize(world_up, [0.0, 0.0, 1.0]))))
            if dot_up >= self.procedural_topdown_dot_threshold:
                return tag_up_axis
            return world_up
        return tag_up_axis

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
        rospy.loginfo("[apriltag_grasp_demo] %s", text)
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

        detections = self.detector.detect_apriltags(image, self.cam_matrix, self.cam_dist)
        if not detections:
            rospy.loginfo_throttle(2.0, "[apriltag_grasp_demo] no_tag_detected image_topic=%s", self.image_topic)
            return

        chosen = None
        if self.target_tag_id >= 0:
            for T_tag_cam_i, tag_id_i in detections:
                if int(tag_id_i) == int(self.target_tag_id):
                    chosen = (T_tag_cam_i, tag_id_i)
                    break
            if chosen is None:
                visible_ids = [int(tag_id_i) for _, tag_id_i in detections]
                self._publish_status(
                    "visible_tags={} waiting_for={}".format(visible_ids, self.target_tag_id)
                )
                return
        else:
            chosen = detections[0]

        T_tag_cam, tag_id = chosen

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time(0)
        tf_stamp = rospy.Time(0) if self.use_latest_tf else stamp
        T_base_cam = self._lookup_base_to_camera(tf_stamp)
        if T_base_cam is None:
            return
        if self.invert_base_camera_transform:
            T_base_cam = np.linalg.inv(T_base_cam)

        # detect_apriltag() returns tag->camera (OpenCV solvePnP convention),
        # so base->tag is directly: base->camera * tag->camera.
        if self.invert_tag_transform:
            T_base_tag = T_base_cam @ np.linalg.inv(T_tag_cam)
        else:
            T_base_tag = T_base_cam @ T_tag_cam
        self.last_detection_time = rospy.Time.now()
        self._maybe_log_tag2base_matrix(T_base_tag, tag_id)
        rospy.loginfo_throttle(
            2.0,
            "[apriltag_grasp_demo] tag_detected id=%d base_frame=%s camera_frame=%s",
            int(tag_id),
            self.base_frame,
            self.camera_frame,
        )
        self._publish_transformed_templates(T_base_tag, T_base_cam, tag_id, stamp)

    def _maybe_log_tag2base_matrix(self, T_base_tag, tag_id):
        if not self.print_tag2base_matrix:
            return
        now = rospy.Time.now()
        if (now - self.last_tag2base_print_time).to_sec() < self.tag2base_print_period_sec:
            return
        self.last_tag2base_print_time = now
        rospy.loginfo(
            "tag2base transform (tag_id=%d, base_frame=%s):\n%s",
            int(tag_id),
            self.base_frame,
            _format_matrix_4x4(T_base_tag),
        )

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

    def _publish_transformed_templates(self, T_base_tag, T_base_cam, tag_id, stamp):
        base_tag_pose = _make_pose_stamped(self.base_frame, stamp, T_base_tag)
        self.base_tag_pose_pub.publish(base_tag_pose)

        if self.procedural_tag_grasp:
            self._publish_procedural_grasp(T_base_tag, T_base_cam, tag_id, stamp)
            return

        selected_template_name = None
        selected_pregrasp_pose = None
        selected_grasp_pose = None
        marker_specs = []

        for index, template in enumerate(self.templates):
            T_tag_grasp = _pose_dict_to_matrix(template["grasp_pose"])
            T_tag_grasp[:3, :3] = T_tag_grasp[:3, :3] @ self.template_rot_correction
            T_base_grasp = T_base_tag @ T_tag_grasp
            grasp_pose = _make_pose_stamped(self.base_frame, stamp, T_base_grasp)

            pregrasp_pose = None
            if template["pregrasp_pose"] is not None:
                T_tag_pregrasp = _pose_dict_to_matrix(template["pregrasp_pose"])
                T_tag_pregrasp[:3, :3] = T_tag_pregrasp[:3, :3] @ self.template_rot_correction
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
            fallback_grasp = _pose_dict_to_matrix(self.templates[0]["grasp_pose"])
            fallback_grasp[:3, :3] = fallback_grasp[:3, :3] @ self.template_rot_correction
            selected_grasp_pose = _make_pose_stamped(
                self.base_frame,
                stamp,
                T_base_tag @ fallback_grasp,
            )
            if self.templates[0]["pregrasp_pose"] is not None:
                fallback_pregrasp = _pose_dict_to_matrix(self.templates[0]["pregrasp_pose"])
                fallback_pregrasp[:3, :3] = fallback_pregrasp[:3, :3] @ self.template_rot_correction
                selected_pregrasp_pose = _make_pose_stamped(
                    self.base_frame,
                    stamp,
                    T_base_tag @ fallback_pregrasp,
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

    def _publish_procedural_grasp(self, T_base_tag, T_base_cam, tag_id, stamp):
        R_base_tag = T_base_tag[:3, :3]
        p_base_tag = T_base_tag[:3, 3]
        p_base_cam = T_base_cam[:3, 3]
        world_up = _safe_normalize(self.procedural_world_up_axis, [0.0, 0.0, 1.0])

        if self.procedural_offset_mode == "camera_ray":
            ray_dir = _safe_normalize(p_base_cam - p_base_tag, [0.0, 0.0, 1.0])
            p_base_grasp = p_base_tag + ray_dir * float(self.procedural_grasp_distance_m)
            p_base_pregrasp = p_base_grasp + ray_dir * float(self.procedural_pregrasp_offset_distance_m)
            R_base_grasp = _compute_look_at_rotation(
                origin=p_base_grasp,
                target=p_base_tag,
                world_up=world_up,
                ee_forward_axis=self.procedural_ee_forward_axis,
                ee_up_axis=self.procedural_ee_up_axis,
            )
        elif self.procedural_offset_mode in ("world_up", "vertical", "top_down"):
            p_base_grasp = p_base_tag + world_up * float(self.procedural_grasp_distance_m)
            p_base_pregrasp = p_base_grasp + world_up * float(self.procedural_pregrasp_offset_distance_m)
            R_base_grasp = _compute_look_at_rotation(
                origin=p_base_grasp,
                target=p_base_tag,
                world_up=world_up,
                ee_forward_axis=self.procedural_ee_forward_axis,
                ee_up_axis=self.procedural_ee_up_axis,
            )
        elif self.procedural_offset_mode in ("tag_normal", "tag_frame_normal", "surface_normal"):
            axis_tag = _safe_normalize(self.procedural_pregrasp_offset_axis_tag, [0.0, 0.0, 1.0])
            axis_base = R_base_tag @ axis_tag
            p_base_grasp = p_base_tag + (R_base_tag @ self.procedural_grasp_offset_tag)
            p_base_pregrasp = p_base_grasp + axis_base * float(self.procedural_pregrasp_offset_distance_m)

            # Build both roll-equivalent solutions and keep the one closest to the
            # current wrist orientation to avoid sudden flips near convergence.
            tag_up_axis = self._resolve_tag_up_reference(axis_base, R_base_tag, world_up)
            if abs(np.dot(_safe_normalize(tag_up_axis, [0.0, 1.0, 0.0]), _safe_normalize(-axis_base, [0.0, 0.0, -1.0]))) > 0.98:
                tag_up_axis = R_base_tag[:, 0]
            rotation_candidates = []
            for up_candidate in (tag_up_axis, -tag_up_axis):
                rotation_candidates.append(
                    _compute_axis_alignment_rotation(
                        forward_w=-axis_base,
                        up_ref=up_candidate,
                        ee_forward_axis=self.procedural_ee_forward_axis,
                        ee_up_axis=self.procedural_ee_up_axis,
                    )
                )
            R_base_grasp = self._choose_rotation_closest_to_current(rotation_candidates)
        else:
            p_base_grasp = p_base_tag + (R_base_tag @ self.procedural_grasp_offset_tag)
            axis_base = R_base_tag @ _safe_normalize(self.procedural_pregrasp_offset_axis_tag, [0.0, 0.0, 1.0])
            p_base_pregrasp = p_base_grasp + axis_base * float(self.procedural_pregrasp_offset_distance_m)
            R_base_grasp = _compute_look_at_rotation(
                origin=p_base_grasp,
                target=p_base_tag,
                world_up=world_up,
                ee_forward_axis=self.procedural_ee_forward_axis,
                ee_up_axis=self.procedural_ee_up_axis,
            )

        if self.procedural_min_grasp_z > -0.9:
            p_base_grasp[2] = max(p_base_grasp[2], self.procedural_min_grasp_z)
            p_base_pregrasp[2] = max(p_base_pregrasp[2], self.procedural_min_grasp_z)

        R_base_grasp = self._maybe_limit_orientation(R_base_grasp)
        R_base_grasp = R_base_grasp @ self.template_rot_correction

        T_base_grasp = np.eye(4, dtype=np.float64)
        T_base_grasp[:3, :3] = R_base_grasp
        T_base_grasp[:3, 3] = p_base_grasp

        T_base_pregrasp = np.eye(4, dtype=np.float64)
        T_base_pregrasp[:3, :3] = R_base_grasp
        T_base_pregrasp[:3, 3] = p_base_pregrasp

        selected_name = self.procedural_grasp_id or "tag_axis_grasp"
        pregrasp_pose = _make_pose_stamped(self.base_frame, stamp, T_base_pregrasp)
        grasp_pose = _make_pose_stamped(self.base_frame, stamp, T_base_grasp)

        self.selected_template_pub.publish(String(data=selected_name))
        self.pregrasp_pose_pub.publish(pregrasp_pose)
        self.grasp_pose_pub.publish(grasp_pose)

        marker_specs = [
            {
                "index": 0,
                "grasp_id": selected_name,
                "tag_id": tag_id,
                "pregrasp_pose": pregrasp_pose,
                "grasp_pose": grasp_pose,
            }
        ]
        self.markers_pub.publish(self._make_markers(stamp, T_base_tag, tag_id, marker_specs))
        rospy.loginfo_throttle(
            2.0,
            "[apriltag_grasp_demo] published procedural grasp_id=%s pre_topic=%s grasp_topic=%s",
            selected_name,
            self.pregrasp_pose_topic,
            self.grasp_pose_topic,
        )
        self._publish_status(
            "tracking_tag_id={} selected_template={} templates=procedural".format(
                tag_id,
                selected_name,
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

        axis_origin = T_base_tag[:3, 3]
        axis_length = 0.06
        axis_specs = [
            ("tag_axes", 2, T_base_tag[:3, 0], ColorRGBA(1.0, 0.1, 0.1, 0.95), "x"),
            ("tag_axes", 3, T_base_tag[:3, 1], ColorRGBA(0.1, 1.0, 0.1, 0.95), "y"),
            ("tag_axes", 4, T_base_tag[:3, 2], ColorRGBA(0.1, 0.4, 1.0, 0.95), "z"),
        ]
        for ns, marker_id, axis_dir, color, label_text in axis_specs:
            axis_tip = axis_origin + _safe_normalize(axis_dir, [1.0, 0.0, 0.0]) * axis_length

            axis_marker = Marker()
            axis_marker.header.frame_id = self.base_frame
            axis_marker.header.stamp = stamp
            axis_marker.ns = ns
            axis_marker.id = marker_id
            axis_marker.type = Marker.ARROW
            axis_marker.action = Marker.ADD
            axis_marker.scale.x = 0.005
            axis_marker.scale.y = 0.01
            axis_marker.scale.z = 0.015
            axis_marker.color = color
            axis_marker.points = [_make_point(axis_origin), _make_point(axis_tip)]
            markers.markers.append(axis_marker)

            axis_label = Marker()
            axis_label.header.frame_id = self.base_frame
            axis_label.header.stamp = stamp
            axis_label.ns = "{}_labels".format(ns)
            axis_label.id = marker_id
            axis_label.type = Marker.TEXT_VIEW_FACING
            axis_label.action = Marker.ADD
            axis_label.pose.orientation.w = 1.0
            axis_label.pose.position.x = float(axis_tip[0])
            axis_label.pose.position.y = float(axis_tip[1])
            axis_label.pose.position.z = float(axis_tip[2] + 0.01)
            axis_label.scale.z = 0.025
            axis_label.color = color
            axis_label.text = label_text
            markers.markers.append(axis_label)

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
