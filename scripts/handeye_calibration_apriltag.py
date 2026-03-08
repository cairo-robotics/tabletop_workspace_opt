#!/usr/bin/env python3
"""
Hand-eye calibration (eye-in-hand) with checkerboard detections.

Supports:
- Manual sample capture from live image ('a' key)
- Automatic plane traversal with RelaxedIK pose goals
- Image/depth capture for each accepted sample
- Confirmation gate via key ('s') or service (~confirm_and_solve)
- Optional fused final point cloud visualization after solving
"""

import os
import threading
from typing import List, Tuple
from collections import deque

import cv2
import numpy as np
import open3d as o3d
import rospy
import tf.transformations as tft
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray
from relaxed_ik_ros1.msg import EEPoseGoals
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

METHOD_MAP = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def tf_to_matrix(tf_msg):
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def matrix_to_R_t(T):
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].reshape(3, 1).astype(np.float64)
    return R, t


def R_t_to_matrix(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def xyz_quat_to_matrix(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = tft.quaternion_matrix(quat_xyzw.tolist())
    T[:3, 3] = xyz.reshape(3).tolist()
    return T.astype(np.float64)


def parse_vec3_csv(value: str, name: str) -> np.ndarray:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Parameter '{name}' must be 'x,y,z'. Got: {value}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def parse_quat_xyzw_csv(value: str, name: str) -> np.ndarray:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"Parameter '{name}' must be 'x,y,z,w'. Got: {value}")
    q = np.array([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 1e-9:
        raise ValueError(f"Parameter '{name}' quaternion norm is zero.")
    return q / n


def parse_float_csv(value: str, name: str) -> List[float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Parameter '{name}' must contain at least one float.")
    vals = []
    for p in parts:
        vals.append(float(p))
    return vals


class HandEyeCalibrator:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # Core calibration params
        self.checkerboard_rows = int(rospy.get_param("~checkerboard_rows", 6))
        self.checkerboard_cols = int(rospy.get_param("~checkerboard_cols", 9))
        self.square_size_m = float(rospy.get_param("~square_size_m", 0.025))
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.eef_frame = rospy.get_param("~eef_frame", "right_hand")
        self.camera_link = rospy.get_param("~camera_link", "camera_link")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.info_topic = rospy.get_param("~info_topic", "/camera/color/camera_info")
        self.min_samples = int(rospy.get_param("~min_samples", 8))
        method_str = str(rospy.get_param("~calib_method", "TSAI")).upper()
        self.calib_method = METHOD_MAP.get(method_str, cv2.CALIB_HAND_EYE_TSAI)
        self.output_file = rospy.get_param("~output_file", "handeye_calibration_result.yaml")

        # Automation params
        self.auto_mode = bool(rospy.get_param("~auto_mode", False))
        self.enable_gui = bool(rospy.get_param("~enable_gui", not self.auto_mode))
        self.drive_with_relaxedik = bool(rospy.get_param("~drive_with_relaxedik", True))
        self.motion_settle_sec = float(rospy.get_param("~motion_settle_sec", 1.5))
        self.capture_timeout_sec = float(rospy.get_param("~capture_timeout_sec", 2.0))
        self.capture_dir = rospy.get_param("~capture_dir", os.path.join(os.getcwd(), "handeye_captures"))
        self.auto_confirm = bool(rospy.get_param("~auto_confirm", False))
        self.max_color_depth_dt_sec = float(rospy.get_param("~max_color_depth_dt_sec", 0.05))
        self.save_all_waypoint_images = bool(rospy.get_param("~save_all_waypoint_images", True))
        self.max_good_samples = int(rospy.get_param("~max_good_samples", 30))
        self.stabilize_min_frames = int(rospy.get_param("~stabilize_min_frames", 5))
        self.stabilize_max_trans_delta_m = float(rospy.get_param("~stabilize_max_trans_delta_m", 0.004))
        self.stabilize_max_rot_delta_deg = float(rospy.get_param("~stabilize_max_rot_delta_deg", 1.5))

        # Checkerboard quality gating (quality over quantity).
        self.max_reproj_error_px = float(rospy.get_param("~max_reproj_error_px", 0.8))
        self.min_board_area_px = float(rospy.get_param("~min_board_area_px", 1500.0))
        self.min_board_distance_m = float(rospy.get_param("~min_board_distance_m", 0.20))
        self.max_board_distance_m = float(rospy.get_param("~max_board_distance_m", 1.20))
        self.require_sb_detector = bool(rospy.get_param("~require_sb_detector", False))

        # Plane definition (3 corners: p0, p1, p2)
        self.plane_p0 = parse_vec3_csv(rospy.get_param("~plane_p0", "0.60,-0.10,0.40"), "plane_p0")
        self.plane_p1 = parse_vec3_csv(rospy.get_param("~plane_p1", "0.70,-0.10,0.40"), "plane_p1")
        self.plane_p2 = parse_vec3_csv(rospy.get_param("~plane_p2", "0.60,0.10,0.35"), "plane_p2")
        self.plane_rows = int(rospy.get_param("~plane_rows", 3))
        self.plane_cols = int(rospy.get_param("~plane_cols", 4))
        self.flat_grid = bool(rospy.get_param("~flat_grid", True))
        self.plane_height_m = float(rospy.get_param("~plane_height_m", self.plane_p0[2]))
        self.plane_height_layers_m_csv = parse_float_csv(
            rospy.get_param("~plane_height_layers_m_csv", str(self.plane_height_m)),
            "plane_height_layers_m_csv",
        )
        self.plane_orientation_xyzw = parse_quat_xyzw_csv(
            rospy.get_param("~plane_orientation_xyzw", "0,1,0,0"), "plane_orientation_xyzw"
        )
        self.orientation_roll_deg_csv = parse_float_csv(
            rospy.get_param("~orientation_roll_deg_csv", "0"), "orientation_roll_deg_csv"
        )
        self.orientation_pitch_deg_csv = parse_float_csv(
            rospy.get_param("~orientation_pitch_deg_csv", "0"), "orientation_pitch_deg_csv"
        )
        self.orientation_yaw_deg_csv = parse_float_csv(
            rospy.get_param("~orientation_yaw_deg_csv", "0"), "orientation_yaw_deg_csv"
        )
        self.auto_center_grid_on_checkerboard = bool(rospy.get_param("~auto_center_grid_on_checkerboard", True))
        self.auto_center_use_known_checkerboard_center = bool(
            rospy.get_param("~auto_center_use_known_checkerboard_center", False)
        )
        self.auto_center_known_checkerboard_center_xyz = parse_vec3_csv(
            rospy.get_param("~auto_center_known_checkerboard_center_xyz", "0,0,0"),
            "auto_center_known_checkerboard_center_xyz",
        )
        self.auto_center_timeout_sec = float(rospy.get_param("~auto_center_timeout_sec", 8.0))
        self.auto_center_span_x_m = float(rospy.get_param("~auto_center_span_x_m", 0.30))
        self.auto_center_span_y_m = float(rospy.get_param("~auto_center_span_y_m", 0.30))
        self.auto_center_plane_height_from_board = bool(rospy.get_param("~auto_center_plane_height_from_board", True))
        self.auto_center_plane_height_offset_m = float(rospy.get_param("~auto_center_plane_height_offset_m", 0.30))
        self.auto_center_layer_offsets_m_csv = parse_float_csv(
            rospy.get_param("~auto_center_layer_offsets_m_csv", "-0.05,0.0,0.05"),
            "auto_center_layer_offsets_m_csv",
        )
        self.initial_guess_from_output_file = bool(rospy.get_param("~initial_guess_from_output_file", True))
        self.initial_guess_xyz = parse_vec3_csv(
            rospy.get_param("~initial_guess_gripper_to_camera_xyz", "0,0,0"),
            "initial_guess_gripper_to_camera_xyz",
        )
        self.initial_guess_xyzw = parse_quat_xyzw_csv(
            rospy.get_param("~initial_guess_gripper_to_camera_xyzw", "0,0,0,1"),
            "initial_guess_gripper_to_camera_xyzw",
        )

        # Point cloud output
        self.show_final_pointcloud = bool(rospy.get_param("~show_final_pointcloud", True))
        self.max_points_per_frame = int(rospy.get_param("~max_points_per_frame", 25000))
        self.fusion_use_icp = bool(rospy.get_param("~fusion_use_icp", True))
        self.fusion_voxel_m = float(rospy.get_param("~fusion_voxel_m", 0.005))
        self.icp_max_corr_m = float(rospy.get_param("~icp_max_corr_m", 0.03))
        self.calib_consistency_warn_m = float(rospy.get_param("~calib_consistency_warn_m", 0.03))

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # Live frame buffers
        self.latest_color = None
        self.latest_color_stamp = None
        self.latest_depth = None
        self.latest_depth_stamp = None

        # Checkerboard object points (target frame).
        # OpenCV expects patternSize = (cols, rows) for inner corners.
        objp = np.zeros((self.checkerboard_rows * self.checkerboard_cols, 3), dtype=np.float32)
        grid = np.mgrid[0:self.checkerboard_cols, 0:self.checkerboard_rows].T.reshape(-1, 2)
        objp[:, :2] = grid
        objp *= self.square_size_m
        self.board_obj_points = objp

        # Calibration dataset
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        self.sample_records = []
        self.n_samples = 0

        self.solved_T_gripper2cam = None
        self.solved_T_cam2gripper = None
        self.total_waypoint_captures = 0
        self.initial_T_gripper2cam_guess = self._load_initial_gripper_to_camera_guess()

        # ROS I/O
        self.color_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback, queue_size=1)
        self.info_sub = rospy.Subscriber(self.info_topic, CameraInfo, self.camera_info_callback, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.ee_pose_pub = None
        if self.auto_mode and self.drive_with_relaxedik:
            self.ee_pose_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=5)

        # RViz outputs (latched so they persist in RViz without continuous re-publish).
        self.plane_markers_pub = rospy.Publisher("~plane_markers", MarkerArray, queue_size=1, latch=True)
        self.planner_markers_pub = rospy.Publisher("~planner_markers", MarkerArray, queue_size=1, latch=True)
        self.waypoints_pub = rospy.Publisher("~planned_waypoints", PoseArray, queue_size=1, latch=True)
        self.calibrated_cloud_pub = rospy.Publisher("~calibrated_cloud", PointCloud2, queue_size=1, latch=True)

        self.confirm_srv = rospy.Service("~confirm_and_solve", Trigger, self.confirm_and_solve_srv)

        os.makedirs(self.capture_dir, exist_ok=True)

        self._log_config()
        self.publish_plane_markers()
        self.publish_motion_plan_markers(self.generate_plane_waypoints(), active_idx=-1)

    def _log_config(self):
        rospy.loginfo("=" * 60)
        rospy.loginfo("Hand-Eye Calibration - Checkerboard")
        rospy.loginfo("=" * 60)
        rospy.loginfo(
            f"Checkerboard: rows={self.checkerboard_rows}, cols={self.checkerboard_cols}, square={self.square_size_m:.4f} m"
        )
        rospy.loginfo(f"Frames: base={self.base_frame}, eef={self.eef_frame}, camera={self.camera_link}")
        rospy.loginfo(f"Topics: color={self.image_topic}, depth={self.depth_topic}, info={self.info_topic}")
        rospy.loginfo(f"Method: {self._get_method_name(self.calib_method)}")
        rospy.loginfo(f"GUI enabled: {self.enable_gui}")
        rospy.loginfo(f"Min samples: {self.min_samples}")
        rospy.loginfo(f"Capture dir: {self.capture_dir}")
        rospy.loginfo(f"Max color-depth dt: {self.max_color_depth_dt_sec:.3f}s")
        rospy.loginfo(f"Save all waypoint images: {self.save_all_waypoint_images}")
        rospy.loginfo(f"Max good samples: {self.max_good_samples}")
        rospy.loginfo(
            "Capture stabilization: min_frames=%d, max_trans_delta=%.4fm, max_rot_delta=%.2fdeg",
            self.stabilize_min_frames,
            self.stabilize_max_trans_delta_m,
            self.stabilize_max_rot_delta_deg,
        )
        rospy.loginfo(
            "Checkerboard quality gates: reproj<=%.3fpx, area>=%.1fpx, dist=[%.2f, %.2f]m",
            self.max_reproj_error_px,
            self.min_board_area_px,
            self.min_board_distance_m,
            self.max_board_distance_m,
        )
        rospy.loginfo(
            "Auto-center grid: enabled=%s span=[%.3f, %.3f]m timeout=%.1fs",
            self.auto_center_grid_on_checkerboard,
            self.auto_center_span_x_m,
            self.auto_center_span_y_m,
            self.auto_center_timeout_sec,
        )
        if self.auto_center_use_known_checkerboard_center:
            c = self.auto_center_known_checkerboard_center_xyz
            rospy.loginfo("Auto-center uses known checkerboard center in base: [%.4f, %.4f, %.4f]", c[0], c[1], c[2])
        rospy.loginfo(
            f"Fusion: use_icp={self.fusion_use_icp}, voxel={self.fusion_voxel_m:.4f}m, icp_max_corr={self.icp_max_corr_m:.4f}m"
        )
        if self.auto_mode:
            rospy.loginfo(
                f"Auto mode ON: rows={self.plane_rows}, cols={self.plane_cols}, "
                f"drive_with_relaxedik={self.drive_with_relaxedik}"
            )
            rospy.loginfo(
                "Plane points: p0=%s p1=%s p2=%s",
                np.array2string(self.plane_p0, precision=4),
                np.array2string(self.plane_p1, precision=4),
                np.array2string(self.plane_p2, precision=4),
            )
            rospy.loginfo(f"Flat grid: {self.flat_grid} (height z={self.plane_height_m:.4f})")
            rospy.loginfo(f"Grid height layers z={self.plane_height_layers_m_csv}")
            rospy.loginfo(
                "Orientation sweep deg: roll=%s pitch=%s yaw=%s",
                self.orientation_roll_deg_csv,
                self.orientation_pitch_deg_csv,
                self.orientation_yaw_deg_csv,
            )
        else:
            rospy.loginfo("Manual mode: press 'a' add sample, 's' solve, 'c' clear, 'q' quit")
        rospy.loginfo("Service confirmation: rosservice call %s/confirm_and_solve", rospy.get_name())
        rospy.loginfo("=" * 60)

    def _get_method_name(self, method):
        names = {
            cv2.CALIB_HAND_EYE_TSAI: "Tsai",
            cv2.CALIB_HAND_EYE_PARK: "Park",
            cv2.CALIB_HAND_EYE_HORAUD: "Horaud",
            cv2.CALIB_HAND_EYE_ANDREFF: "Andreff",
            cv2.CALIB_HAND_EYE_DANIILIDIS: "Daniilidis",
        }
        return names.get(method, "Unknown")

    def _load_initial_gripper_to_camera_guess(self) -> np.ndarray:
        if self.initial_guess_from_output_file and os.path.isfile(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                g2c = data.get("gripper_to_camera", {})
                tr = g2c.get("translation", {})
                q = g2c.get("rotation_quaternion_xyzw", None)
                if q is not None:
                    xyz = np.array([float(tr.get("x", 0.0)), float(tr.get("y", 0.0)), float(tr.get("z", 0.0))])
                    quat = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])], dtype=np.float64)
                    n = np.linalg.norm(quat)
                    if n > 1e-9:
                        quat = quat / n
                        rospy.loginfo("Loaded initial gripper->camera guess from output file")
                        return xyz_quat_to_matrix(xyz, quat)
            except Exception as e:
                rospy.logwarn(f"Failed loading initial guess from {self.output_file}: {e}")

        rospy.loginfo("Using param-based initial gripper->camera guess")
        return xyz_quat_to_matrix(self.initial_guess_xyz, self.initial_guess_xyzw)

    def get_plane_corners(self, z_height: float = None):
        p0 = self.plane_p0.copy()
        p1 = self.plane_p1.copy()
        p2 = self.plane_p2.copy()
        if self.flat_grid:
            z = self.plane_height_m if z_height is None else float(z_height)
            p0[2] = z
            p1[2] = z
            p2[2] = z
        p3 = p1 + p2 - p0
        return p0, p1, p2, p3

    def _point_msg(self, xyz: np.ndarray) -> Point:
        p = Point()
        p.x = float(xyz[0])
        p.y = float(xyz[1])
        p.z = float(xyz[2])
        return p

    def _new_marker(self, ns: str, mid: int, mtype: int, scale_xyz, rgba):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = int(mid)
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(scale_xyz[0])
        m.scale.y = float(scale_xyz[1])
        m.scale.z = float(scale_xyz[2])
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])
        return m

    def publish_plane_markers(self):
        heights = self.plane_height_layers_m_csv if self.flat_grid else [None]
        markers = []
        for i, z in enumerate(heights):
            p0, p1, p2, p3 = self.get_plane_corners(z)
            border = self._new_marker("plane", 100 + 2 * i, Marker.LINE_STRIP, (0.003, 0.0, 0.0), (0.2, 0.8, 1.0, 0.9))
            border.points = [
                self._point_msg(p0),
                self._point_msg(p1),
                self._point_msg(p3),
                self._point_msg(p2),
                self._point_msg(p0),
            ]
            markers.append(border)

            corners = self._new_marker(
                "plane", 101 + 2 * i, Marker.SPHERE_LIST, (0.015, 0.015, 0.015), (0.1, 0.9, 0.4, 0.9)
            )
            corners.points = [self._point_msg(p0), self._point_msg(p1), self._point_msg(p2), self._point_msg(p3)]
            markers.append(corners)

        self.plane_markers_pub.publish(MarkerArray(markers=markers))

    def publish_motion_plan_markers(self, waypoints: List[np.ndarray], active_idx: int = -1):
        markers = []

        line = self._new_marker("planner", 10, Marker.LINE_STRIP, (0.004, 0.0, 0.0), (1.0, 0.8, 0.2, 0.95))
        line.points = [self._point_msg(wp[0]) for wp in waypoints]
        markers.append(line)

        spheres = self._new_marker("planner", 11, Marker.SPHERE_LIST, (0.012, 0.012, 0.012), (1.0, 0.4, 0.2, 0.95))
        spheres.points = [self._point_msg(wp[0]) for wp in waypoints]
        markers.append(spheres)

        if 0 <= active_idx < len(waypoints):
            active = self._new_marker("planner", 12, Marker.SPHERE, (0.03, 0.03, 0.03), (0.2, 1.0, 0.2, 0.95))
            active.pose.position = self._point_msg(waypoints[active_idx][0])
            markers.append(active)

        self.planner_markers_pub.publish(MarkerArray(markers=markers))

        pa = PoseArray()
        pa.header.frame_id = self.base_frame
        pa.header.stamp = rospy.Time.now()
        for p, q in waypoints:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            pose.orientation.x = float(q[0])
            pose.orientation.y = float(q[1])
            pose.orientation.z = float(q[2])
            pose.orientation.w = float(q[3])
            pa.poses.append(pose)
        self.waypoints_pub.publish(pa)

    def publish_calibrated_cloud(self, points: np.ndarray):
        if points is None or len(points) == 0:
            return
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.base_frame
        cloud = pc2.create_cloud_xyz32(header, points.astype(np.float32))
        self.calibrated_cloud_pub.publish(cloud)

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]

        self.camera_matrix = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )

        d = np.array(msg.D, dtype=np.float64).reshape(-1, 1) if msg.D else np.zeros((5, 1), dtype=np.float64)
        self.dist_coeffs = d if d.size > 0 else np.zeros((5, 1), dtype=np.float64)

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr_throttle(2.0, f"Failed depth conversion: {e}")
            return

        with self.lock:
            self.latest_depth = depth.copy()
            self.latest_depth_stamp = msg.header.stamp

    def image_callback(self, msg: Image):
        if self.camera_matrix is None:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr_throttle(2.0, f"Failed image conversion: {e}")
            return

        with self.lock:
            self.latest_color = image.copy()
            self.latest_color_stamp = msg.header.stamp

        R_target2cam, t_target2cam, corners, reproj_err, area_px = self.detect_checkerboard(image)
        display = self.draw_detection(image, R_target2cam, t_target2cam, corners, reproj_err, area_px)
        cv2.putText(
            display,
            f"Samples: {self.n_samples}/{self.min_samples}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            display,
            f"Mode: {'AUTO' if self.auto_mode else 'MANUAL'}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        if self.enable_gui:
            cv2.imshow("Hand-Eye Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):
                if R_target2cam is None:
                    rospy.logwarn("Cannot add sample: checkerboard not detected")
                    return
                color, depth, stamp = self.get_latest_frame_copy()
                self.add_sample(stamp, R_target2cam, t_target2cam, color, depth)

            elif key == ord("s"):
                self.solve_calibration()

            elif key == ord("c"):
                with self.lock:
                    self.R_gripper2base = []
                    self.t_gripper2base = []
                    self.R_target2cam = []
                    self.t_target2cam = []
                    self.sample_records = []
                    self.n_samples = 0
                rospy.loginfo("All samples cleared")

            elif key == ord("q"):
                rospy.signal_shutdown("User requested shutdown")

    def detect_checkerboard(self, image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self.camera_matrix is None:
            return None, None, None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pattern_size = (self.checkerboard_cols, self.checkerboard_rows)
        corners = None

        if self.require_sb_detector:
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        else:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if not found:
                found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not found or corners is None:
            return None, None, None, None, None

        corners_refined = corners.astype(np.float32)
        # SB output is already subpixel-quality; refine only on the classic detector path.
        if not self.require_sb_detector:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
            cv2.cornerSubPix(gray, corners_refined, (11, 11), (-1, -1), term)

        ok, rvec, tvec = cv2.solvePnP(
            self.board_obj_points,
            corners_refined,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None, None, None, None

        proj, _ = cv2.projectPoints(self.board_obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners_refined.reshape(-1, 2), axis=1)))
        area_px = float(cv2.contourArea(corners_refined.reshape(-1, 2).astype(np.float32)))
        dist_m = float(np.linalg.norm(tvec))

        if reproj_err > self.max_reproj_error_px:
            rospy.logwarn_throttle(1.0, "Reject board: reproj %.3fpx > %.3fpx", reproj_err, self.max_reproj_error_px)
            return None, None, None, None, None
        if area_px < self.min_board_area_px:
            rospy.logwarn_throttle(1.0, "Reject board: area %.1fpx < %.1fpx", area_px, self.min_board_area_px)
            return None, None, None, None, None
        if dist_m < self.min_board_distance_m or dist_m > self.max_board_distance_m:
            rospy.logwarn_throttle(
                1.0,
                "Reject board: distance %.3fm outside [%.2f, %.2f]m",
                dist_m,
                self.min_board_distance_m,
                self.max_board_distance_m,
            )
            return None, None, None, None, None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1).astype(np.float64)
        return R.astype(np.float64), t, corners_refined.reshape(-1, 2), reproj_err, area_px

    def draw_detection(self, image, R, t, corners, reproj_err=None, area_px=None):
        if R is None or corners is None:
            return image

        out = image.copy()
        pattern_size = (self.checkerboard_cols, self.checkerboard_rows)
        corners_cb = corners.reshape(-1, 1, 2).astype(np.float32)
        cv2.drawChessboardCorners(out, pattern_size, corners_cb, True)

        rvec, _ = cv2.Rodrigues(R)
        axis_len = self.square_size_m * max(1.0, 0.5 * float(min(self.checkerboard_cols, self.checkerboard_rows)))
        cv2.drawFrameAxes(out, self.camera_matrix, self.dist_coeffs, rvec, t, axis_len)
        cv2.putText(out, "Checkerboard detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"Distance: {np.linalg.norm(t):.3f}m",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        if reproj_err is not None:
            cv2.putText(
                out,
                f"Reproj: {reproj_err:.3f}px",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        if area_px is not None:
            cv2.putText(
                out,
                f"Area: {area_px:.0f}px",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        return out

    def get_gripper_pose(self, timestamp):
        try:
            # lookup_transform(target, source): transform that maps source-frame points into target frame.
            # With target=base, source=eef, this returns ^base T_gripper (gripper->base).
            tf_msg = self.tf_buffer.lookup_transform(self.base_frame, self.eef_frame, timestamp, rospy.Duration(1.0))
            return tf_to_matrix(tf_msg)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Failed to get gripper pose: {e}")
            return None

    def get_latest_frame_copy(self):
        with self.lock:
            color = None if self.latest_color is None else self.latest_color.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            stamp = self.latest_color_stamp
            depth_stamp = self.latest_depth_stamp
        if stamp is not None and depth_stamp is not None:
            dt = abs((stamp - depth_stamp).to_sec())
            if dt > self.max_color_depth_dt_sec:
                rospy.logwarn_throttle(
                    2.0,
                    f"Color/depth not synchronized (dt={dt:.3f}s > {self.max_color_depth_dt_sec:.3f}s); skipping depth for this sample.",
                )
                depth = None
        return color, depth, stamp

    def wait_for_sensor_ready(self, timeout_sec=20.0):
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                ready = self.camera_matrix is not None and self.latest_color is not None and self.latest_depth is not None
            if ready:
                return True
            if (rospy.Time.now() - start).to_sec() > timeout_sec:
                return False
            rate.sleep()
        return False

    def save_capture(self, idx: int, color: np.ndarray, depth: np.ndarray):
        if color is None:
            return

        color_path = os.path.join(self.capture_dir, f"sample_{idx:03d}_color.png")
        cv2.imwrite(color_path, color)

        if depth is not None:
            if depth.dtype == np.uint16:
                depth_path = os.path.join(self.capture_dir, f"sample_{idx:03d}_depth.png")
                cv2.imwrite(depth_path, depth)
            else:
                depth_path = os.path.join(self.capture_dir, f"sample_{idx:03d}_depth.npy")
                np.save(depth_path, depth)

    def save_waypoint_capture(self, waypoint_idx: int, ok: bool, color: np.ndarray, depth: np.ndarray, stamp):
        if not self.save_all_waypoint_images or color is None:
            return
        status = "ok" if ok else "miss"
        stamp_ns = str(stamp.to_nsec()) if stamp is not None else "nostamp"
        base = f"waypoint_{waypoint_idx + 1:03d}_{status}_{stamp_ns}"
        color_path = os.path.join(self.capture_dir, base + "_color.png")
        cv2.imwrite(color_path, color)
        if depth is not None:
            if depth.dtype == np.uint16:
                cv2.imwrite(os.path.join(self.capture_dir, base + "_depth.png"), depth)
            else:
                np.save(os.path.join(self.capture_dir, base + "_depth.npy"), depth)
        self.total_waypoint_captures += 1

    def save_session_manifest(self):
        manifest_path = os.path.join(self.capture_dir, "session_manifest.yaml")
        data = {
            "timestamp": float(rospy.get_time()),
            "num_samples": int(self.n_samples),
            "num_waypoint_captures": int(self.total_waypoint_captures),
            "target_type": "checkerboard",
            "checkerboard_rows": int(self.checkerboard_rows),
            "checkerboard_cols": int(self.checkerboard_cols),
            "square_size_m": float(self.square_size_m),
            "camera_info": {
                "fx": None if self.fx is None else float(self.fx),
                "fy": None if self.fy is None else float(self.fy),
                "cx": None if self.cx is None else float(self.cx),
                "cy": None if self.cy is None else float(self.cy),
            },
        }
        try:
            with open(manifest_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            rospy.logwarn(f"Failed to save session manifest: {e}")

    def add_sample(self, timestamp, R_target2cam, t_target2cam, color=None, depth=None):
        T_gripper2base = self.get_gripper_pose(timestamp)
        if T_gripper2base is None:
            rospy.logwarn("Failed to get gripper pose, sample not added")
            return False

        R_gripper2base, t_gripper2base = matrix_to_R_t(T_gripper2base)

        with self.lock:
            self.R_gripper2base.append(R_gripper2base.copy())
            self.t_gripper2base.append(t_gripper2base.copy())
            self.R_target2cam.append(R_target2cam.copy())
            self.t_target2cam.append(t_target2cam.copy())

            self.sample_records.append(
                {
                    "timestamp": timestamp,
                    "T_gripper2base": T_gripper2base.copy(),
                    "color": None if color is None else color.copy(),
                    "depth": None if depth is None else depth.copy(),
                }
            )
            self.n_samples += 1
            sample_idx = self.n_samples

        self.save_capture(sample_idx, color, depth)
        rospy.loginfo(f"Sample {sample_idx} added")

        if self.n_samples >= self.min_samples:
            rospy.loginfo("Reached minimum sample count. Press 's' or call ~confirm_and_solve")

        return True

    def publish_ee_goal(self, position_xyz: np.ndarray, quat_xyzw: np.ndarray):
        if self.ee_pose_pub is None:
            return

        pose = Pose()
        pose.position.x = float(position_xyz[0])
        pose.position.y = float(position_xyz[1])
        pose.position.z = float(position_xyz[2])
        pose.orientation.x = float(quat_xyzw[0])
        pose.orientation.y = float(quat_xyzw[1])
        pose.orientation.z = float(quat_xyzw[2])
        pose.orientation.w = float(quat_xyzw[3])

        msg = EEPoseGoals()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        msg.ee_poses = [pose]
        self.ee_pose_pub.publish(msg)

    def generate_orientation_quaternions(self) -> List[np.ndarray]:
        base_q = self.plane_orientation_xyzw
        quats = []
        for r_deg in self.orientation_roll_deg_csv:
            for p_deg in self.orientation_pitch_deg_csv:
                for y_deg in self.orientation_yaw_deg_csv:
                    q_delta = tft.quaternion_from_euler(
                        np.deg2rad(r_deg), np.deg2rad(p_deg), np.deg2rad(y_deg), axes="sxyz"
                    )
                    q = tft.quaternion_multiply(base_q, q_delta)
                    n = np.linalg.norm(q)
                    if n > 1e-9:
                        q = q / n
                    quats.append(q.astype(np.float64))
        return quats

    def generate_plane_waypoints(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        rows = max(1, self.plane_rows)
        cols = max(1, self.plane_cols)
        quat_list = self.generate_orientation_quaternions()
        heights = self.plane_height_layers_m_csv if self.flat_grid else [None]

        waypoints = []
        for h_idx, z in enumerate(heights):
            p0, p1, p2, _ = self.get_plane_corners(z)
            for i in range(rows):
                v = 0.0 if rows == 1 else float(i) / float(rows - 1)
                row_points = []
                for j in range(cols):
                    u = 0.0 if cols == 1 else float(j) / float(cols - 1)
                    p = p0 + u * (p1 - p0) + v * (p2 - p0)
                    row_points.append(p)
                if (i + h_idx) % 2 == 1:
                    row_points.reverse()
                for p in row_points:
                    for q in quat_list:
                        waypoints.append((p.copy(), q.copy()))
        return waypoints

    def _lookup_T_base2cam(self, stamp: rospy.Time) -> np.ndarray:
        # Prefer TF if an existing calibration TF is already published.
        try:
            tf_msg = self.tf_buffer.lookup_transform(self.base_frame, self.camera_link, stamp, rospy.Duration(0.05))
            return tf_to_matrix(tf_msg)
        except Exception:
            pass

        T_gripper2base = self.get_gripper_pose(stamp)
        if T_gripper2base is None:
            return None
        # ^bT_c = ^bT_g * ^gT_c
        return T_gripper2base @ self.initial_T_gripper2cam_guess

    def _apply_auto_center(self, center: np.ndarray):
        span_x = max(0.05, self.auto_center_span_x_m)
        span_y = max(0.05, self.auto_center_span_y_m)
        center_z = self.plane_height_m
        if self.auto_center_plane_height_from_board:
            center_z = float(center[2] + self.auto_center_plane_height_offset_m)

        self.plane_height_m = center_z
        self.plane_p0 = np.array([center[0] - 0.5 * span_x, center[1] - 0.5 * span_y, center_z], dtype=np.float64)
        self.plane_p1 = np.array([center[0] + 0.5 * span_x, center[1] - 0.5 * span_y, center_z], dtype=np.float64)
        self.plane_p2 = np.array([center[0] - 0.5 * span_x, center[1] + 0.5 * span_y, center_z], dtype=np.float64)

        if self.flat_grid:
            self.plane_height_layers_m_csv = [center_z + dz for dz in self.auto_center_layer_offsets_m_csv]

    def auto_center_grid_from_checkerboard(self) -> bool:
        if not self.auto_center_grid_on_checkerboard:
            return False

        if self.auto_center_use_known_checkerboard_center:
            center = self.auto_center_known_checkerboard_center_xyz.copy()
            self._apply_auto_center(center)
            rospy.loginfo(
                "Auto-centered grid from known checkerboard center [%.3f, %.3f, %.3f]; z layers=%s",
                center[0],
                center[1],
                center[2],
                [round(z, 4) for z in self.plane_height_layers_m_csv],
            )
            self.publish_plane_markers()
            return True

        start = rospy.Time.now()
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() <= self.auto_center_timeout_sec:
            color, _, stamp = self.get_latest_frame_copy()
            if color is None or stamp is None:
                rate.sleep()
                continue

            R_t2c, t_t2c, _, _, _ = self.detect_checkerboard(color)
            if R_t2c is None:
                rate.sleep()
                continue

            T_base2cam = self._lookup_T_base2cam(stamp)
            if T_base2cam is None:
                rate.sleep()
                continue

            T_target2cam = R_t_to_matrix(R_t2c, t_t2c)
            # Pose of checkerboard target in base frame.
            T_base2target = T_base2cam @ T_target2cam
            center = T_base2target[:3, 3].copy()
            self._apply_auto_center(center)

            rospy.loginfo(
                "Auto-centered grid around checkerboard center [%.3f, %.3f, %.3f]; z layers=%s",
                center[0],
                center[1],
                center[2],
                [round(z, 4) for z in self.plane_height_layers_m_csv],
            )
            self.publish_plane_markers()
            return True

        rospy.logwarn("Auto-center skipped: checkerboard center could not be estimated in time")
        return False

    def capture_sample_at_current_view(self, waypoint_idx: int) -> bool:
        t0 = rospy.Time.now()
        rate = rospy.Rate(20)
        last_color = None
        last_depth = None
        last_stamp = None
        stable_buf = deque(maxlen=max(2, self.stabilize_min_frames))
        while not rospy.is_shutdown() and (rospy.Time.now() - t0).to_sec() <= self.capture_timeout_sec:
            color, depth, stamp = self.get_latest_frame_copy()
            if color is None or stamp is None:
                rate.sleep()
                continue
            last_color = color
            last_depth = depth
            last_stamp = stamp

            R_t2c, t_t2c, _, _, _ = self.detect_checkerboard(color)
            if R_t2c is not None:
                stable_buf.append((R_t2c.copy(), t_t2c.copy(), color, depth, stamp))
                if self._is_stable_detection_window(stable_buf):
                    R_use, t_use, c_use, d_use, s_use = stable_buf[-1]
                    self.save_waypoint_capture(waypoint_idx, True, c_use, d_use, s_use)
                    return self.add_sample(s_use, R_use, t_use, color=c_use, depth=d_use)
            rate.sleep()

        self.save_waypoint_capture(waypoint_idx, False, last_color, last_depth, last_stamp)
        rospy.logwarn("No valid checkerboard detection within capture timeout")
        return False

    def _rotation_delta_deg(self, R_prev: np.ndarray, R_curr: np.ndarray) -> float:
        R = R_prev.T @ R_curr
        tr = float(np.trace(R))
        cos_theta = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
        return float(np.rad2deg(np.arccos(cos_theta)))

    def _is_stable_detection_window(self, stable_buf: deque) -> bool:
        if len(stable_buf) < max(2, self.stabilize_min_frames):
            return False

        max_dt = 0.0
        max_dr = 0.0
        for i in range(1, len(stable_buf)):
            R_prev, t_prev, _, _, _ = stable_buf[i - 1]
            R_curr, t_curr, _, _, _ = stable_buf[i]
            dt = float(np.linalg.norm((t_curr - t_prev).reshape(3)))
            dr = self._rotation_delta_deg(R_prev, R_curr)
            if dt > max_dt:
                max_dt = dt
            if dr > max_dr:
                max_dr = dr

        return max_dt <= self.stabilize_max_trans_delta_m and max_dr <= self.stabilize_max_rot_delta_deg

    def run_auto_sequence(self):
        rospy.loginfo("Starting automatic plane traversal for hand-eye capture")
        if not self.wait_for_sensor_ready(timeout_sec=20.0):
            rospy.logerr("Camera data/intrinsics not ready; aborting auto sequence")
            return

        self.auto_center_grid_from_checkerboard()

        waypoints = self.generate_plane_waypoints()
        rospy.loginfo(f"Generated {len(waypoints)} plane waypoints")
        self.publish_motion_plan_markers(waypoints, active_idx=-1)

        for idx, (p, q) in enumerate(waypoints):
            if rospy.is_shutdown():
                return
            if self.max_good_samples > 0 and self.n_samples >= self.max_good_samples:
                rospy.loginfo(f"Reached max_good_samples={self.max_good_samples}; stopping traversal early.")
                break

            self.publish_motion_plan_markers(waypoints, active_idx=idx)
            rospy.loginfo(
                f"[{idx + 1}/{len(waypoints)}] waypoint: x={p[0]:.4f}, y={p[1]:.4f}, z={p[2]:.4f}"
            )
            if self.drive_with_relaxedik and self.ee_pose_pub is not None:
                self.publish_ee_goal(p, q)
                rospy.sleep(self.motion_settle_sec)
            else:
                rospy.logwarn_throttle(2.0, "drive_with_relaxedik disabled; not publishing motion goals")
                rospy.sleep(0.5)

            self.capture_sample_at_current_view(idx)

        rospy.loginfo(f"Auto traversal complete. Captured {self.n_samples} samples")
        self.save_session_manifest()
        self.publish_motion_plan_markers(waypoints, active_idx=-1)

        if self.auto_confirm:
            rospy.loginfo("auto_confirm=true: solving now")
            self.solve_calibration()
        else:
            rospy.loginfo("Waiting for user confirmation: press 's' or call ~confirm_and_solve")

    def confirm_and_solve_srv(self, _req):
        ok = self.solve_calibration()
        return TriggerResponse(success=bool(ok), message="Solved" if ok else "Solve failed")

    def solve_calibration(self):
        with self.lock:
            n = len(self.R_gripper2base)
            if n < self.min_samples:
                rospy.logwarn(f"Need at least {self.min_samples} samples, have {n}")
                return False

            R_g2b = list(self.R_gripper2base)
            t_g2b = list(self.t_gripper2base)
            R_t2c = list(self.R_target2cam)
            t_t2c = list(self.t_target2cam)

        rospy.loginfo("=" * 60)
        rospy.loginfo(f"Solving hand-eye with {n} samples ({self._get_method_name(self.calib_method)})")

        try:
            # OpenCV returns camera->gripper for this API. Invert to get gripper->camera.
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_g2b,
                t_g2b,
                R_t2c,
                t_t2c,
                method=self.calib_method,
            )
            T_cam2gripper = R_t_to_matrix(R_cam2gripper, t_cam2gripper)
            T_gripper2cam = np.linalg.inv(T_cam2gripper)

            self.solved_T_gripper2cam = T_gripper2cam
            self.solved_T_cam2gripper = T_cam2gripper

            self.print_results(T_gripper2cam, T_cam2gripper)
            self.save_results(T_gripper2cam, T_cam2gripper)
            self.evaluate_calibration_consistency(T_gripper2cam)

            fused_pcd = self.build_fused_pointcloud_o3d(T_gripper2cam)
            if fused_pcd is not None:
                fused_points = np.asarray(fused_pcd.points)
                self.publish_calibrated_cloud(fused_points)

            if self.show_final_pointcloud:
                self.show_fused_pointcloud(T_gripper2cam, fused_pcd=fused_pcd)

            rospy.loginfo("Calibration successful")
            rospy.loginfo("=" * 60)
            return True

        except Exception as e:
            rospy.logerr(f"Calibration failed: {e}")
            rospy.logerr("Try collecting more diverse samples")
            return False

    def print_results(self, T_gripper2cam, T_cam2gripper):
        print("\n" + "=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)

        t = T_gripper2cam[:3, 3]
        quat = tft.quaternion_from_matrix(T_gripper2cam)
        rpy = tft.euler_from_matrix(T_gripper2cam[:3, :3], "sxyz")

        print(f"\n1. Gripper -> Camera ({self.eef_frame} -> {self.camera_link})")
        print("-" * 60)
        print(f"Translation (m):  x={t[0]:7.4f}  y={t[1]:7.4f}  z={t[2]:7.4f}")
        print(f"Quaternion (xyzw): [{quat[0]:8.5f}, {quat[1]:8.5f}, {quat[2]:8.5f}, {quat[3]:8.5f}]")
        print(f"RPY (deg):         [{np.rad2deg(rpy[0]):7.2f}, {np.rad2deg(rpy[1]):7.2f}, {np.rad2deg(rpy[2]):7.2f}]")

        print("\nStatic TF Publisher Command:")
        print("-" * 60)
        print("rosrun tf2_ros static_transform_publisher \\")
        print(f"  {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} \\")
        print(f"  {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} \\")
        print(f"  {self.eef_frame} {self.camera_link}")

        t_inv = T_cam2gripper[:3, 3]
        quat_inv = tft.quaternion_from_matrix(T_cam2gripper)
        print("\n2. Camera -> Gripper (inverse)")
        print("-" * 60)
        print(f"Translation (m):  x={t_inv[0]:7.4f}  y={t_inv[1]:7.4f}  z={t_inv[2]:7.4f}")
        print(f"Quaternion (xyzw): [{quat_inv[0]:8.5f}, {quat_inv[1]:8.5f}, {quat_inv[2]:8.5f}, {quat_inv[3]:8.5f}]")

        print("\n" + "=" * 60 + "\n")

    def save_results(self, T_gripper2cam, T_cam2gripper):
        t = T_gripper2cam[:3, 3]
        quat = tft.quaternion_from_matrix(T_gripper2cam)
        rpy = tft.euler_from_matrix(T_gripper2cam[:3, :3], "sxyz")

        t_inv = T_cam2gripper[:3, 3]
        quat_inv = tft.quaternion_from_matrix(T_cam2gripper)
        rpy_inv = tft.euler_from_matrix(T_cam2gripper[:3, :3], "sxyz")

        data = {
            "calibration_date": rospy.get_time(),
            "num_samples": self.n_samples,
            "method": self._get_method_name(self.calib_method),
            "target_type": "checkerboard",
            "checkerboard_rows": int(self.checkerboard_rows),
            "checkerboard_cols": int(self.checkerboard_cols),
            "square_size_m": float(self.square_size_m),
            "gripper_to_camera": {
                "parent_frame": self.eef_frame,
                "child_frame": self.camera_link,
                "translation": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
                "rotation_quaternion_xyzw": [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],
                "rotation_rpy_rad": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
                "rotation_rpy_deg": [float(np.rad2deg(rpy[0])), float(np.rad2deg(rpy[1])), float(np.rad2deg(rpy[2]))],
                "transformation_matrix": T_gripper2cam.tolist(),
            },
            "camera_to_gripper_inverse": {
                "parent_frame": self.camera_link,
                "child_frame": self.eef_frame,
                "translation": {"x": float(t_inv[0]), "y": float(t_inv[1]), "z": float(t_inv[2])},
                "rotation_quaternion_xyzw": [
                    float(quat_inv[0]),
                    float(quat_inv[1]),
                    float(quat_inv[2]),
                    float(quat_inv[3]),
                ],
                "rotation_rpy_rad": [float(rpy_inv[0]), float(rpy_inv[1]), float(rpy_inv[2])],
                "transformation_matrix": T_cam2gripper.tolist(),
            },
            "static_transform_publisher_command": (
                f"rosrun tf2_ros static_transform_publisher "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
                f"{self.eef_frame} {self.camera_link}"
            ),
        }

        try:
            with open(self.output_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            rospy.loginfo(f"Results saved to: {self.output_file}")
        except Exception as e:
            rospy.logerr(f"Failed to save results: {e}")

    def depth_to_meters(self, depth: np.ndarray) -> np.ndarray:
        if depth is None:
            return None
        if depth.dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        return depth.astype(np.float32)

    def evaluate_calibration_consistency(self, T_gripper2cam: np.ndarray):
        base_target_positions = []
        with self.lock:
            records = list(self.sample_records)
            R_t2c = list(self.R_target2cam)
            t_t2c = list(self.t_target2cam)

        if not records or len(records) != len(R_t2c):
            rospy.logwarn("Consistency check skipped: inconsistent sample arrays")
            return

        for rec, Rct, tct in zip(records, R_t2c, t_t2c):
            T_gripper2base = rec.get("T_gripper2base")
            if T_gripper2base is None:
                continue
            T_base2gripper = np.linalg.inv(T_gripper2base)
            T_cam2target = R_t_to_matrix(Rct, tct)
            T_base2target = T_base2gripper @ T_gripper2cam @ T_cam2target
            base_target_positions.append(T_base2target[:3, 3])

        if len(base_target_positions) < 2:
            rospy.logwarn("Consistency check skipped: not enough valid checkerboard poses")
            return

        pts = np.asarray(base_target_positions)
        ctr = np.mean(pts, axis=0)
        d = np.linalg.norm(pts - ctr, axis=1)
        rmse = float(np.sqrt(np.mean(d**2)))
        std_xyz = np.std(pts, axis=0)
        rospy.loginfo(
            "Calibration consistency (fixed checkerboard in base): rmse=%.4fm std_xyz=[%.4f, %.4f, %.4f] m",
            rmse,
            float(std_xyz[0]),
            float(std_xyz[1]),
            float(std_xyz[2]),
        )
        if rmse > self.calib_consistency_warn_m:
            rospy.logwarn(
                "Calibration consistency RMSE %.4fm exceeds warning threshold %.4fm",
                rmse,
                self.calib_consistency_warn_m,
            )

    def build_frame_pcds_o3d(self, T_gripper2cam: np.ndarray):
        pcds = []
        with self.lock:
            records = list(self.sample_records)

        for rec in records:
            color = rec.get("color")
            depth = rec.get("depth")
            T_gripper2base = rec.get("T_gripper2base")

            if color is None or depth is None or T_gripper2base is None:
                continue

            depth_m = self.depth_to_meters(depth)
            valid = np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 2.0)
            rows, cols = np.where(valid)
            if rows.size == 0:
                continue

            if rows.size > self.max_points_per_frame:
                idx = np.random.choice(rows.size, self.max_points_per_frame, replace=False)
                rows = rows[idx]
                cols = cols[idx]

            z = depth_m[rows, cols]
            x = (cols.astype(np.float32) - self.cx) * z / self.fx
            y = (rows.astype(np.float32) - self.cy) * z / self.fy
            pts_cam = np.stack([x, y, z], axis=1)

            # T_gripper2base = ^bT_g; therefore ^bT_c = ^bT_g * ^gT_c.
            T_base2cam = T_gripper2base @ T_gripper2cam
            R = T_base2cam[:3, :3]
            t = T_base2cam[:3, 3]
            pts_base = (R @ pts_cam.T).T + t

            rgb = color[rows, cols, ::-1].astype(np.float32) / 255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_base.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
            pcds.append(pcd)

        return pcds

    def merge_pcds_open3d(self, pcds):
        if not pcds:
            return None

        merged = pcds[0]
        if self.fusion_voxel_m > 0:
            merged = merged.voxel_down_sample(self.fusion_voxel_m)

        for idx in range(1, len(pcds)):
            src = pcds[idx]
            if self.fusion_voxel_m > 0:
                src = src.voxel_down_sample(self.fusion_voxel_m)

            if self.fusion_use_icp and len(np.asarray(merged.points)) > 20 and len(np.asarray(src.points)) > 20:
                reg = o3d.pipelines.registration.registration_icp(
                    src,
                    merged,
                    self.icp_max_corr_m,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                src.transform(reg.transformation)
                rospy.loginfo(
                    "Fusion ICP[%d]: fitness=%.4f rmse=%.4f",
                    idx,
                    float(reg.fitness),
                    float(reg.inlier_rmse),
                )

            merged += src
            if self.fusion_voxel_m > 0:
                merged = merged.voxel_down_sample(self.fusion_voxel_m)

        return merged

    def build_fused_pointcloud_o3d(self, T_gripper2cam: np.ndarray):
        if self.camera_matrix is None:
            rospy.logwarn("Cannot build point cloud: missing intrinsics")
            return None
        pcds = self.build_frame_pcds_o3d(T_gripper2cam)
        if not pcds:
            rospy.logwarn("No points available to visualize")
            return None
        merged = self.merge_pcds_open3d(pcds)
        if merged is None:
            return None
        rospy.loginfo("Merged %d frames into %d points", len(pcds), len(np.asarray(merged.points)))
        return merged

    def build_fused_pointcloud_data(self, T_gripper2cam: np.ndarray):
        if self.camera_matrix is None:
            rospy.logwarn("Cannot build point cloud: missing intrinsics")
            return None, None

        all_points = []
        all_colors = []

        with self.lock:
            records = list(self.sample_records)

        for rec in records:
            color = rec.get("color")
            depth = rec.get("depth")
            T_gripper2base = rec.get("T_gripper2base")

            if color is None or depth is None or T_gripper2base is None:
                continue

            depth_m = self.depth_to_meters(depth)
            valid = np.isfinite(depth_m) & (depth_m > 0.1) & (depth_m < 2.0)
            rows, cols = np.where(valid)
            if rows.size == 0:
                continue

            if rows.size > self.max_points_per_frame:
                idx = np.random.choice(rows.size, self.max_points_per_frame, replace=False)
                rows = rows[idx]
                cols = cols[idx]

            z = depth_m[rows, cols]
            x = (cols.astype(np.float32) - self.cx) * z / self.fx
            y = (rows.astype(np.float32) - self.cy) * z / self.fy
            pts_cam = np.stack([x, y, z], axis=1)

            # T_gripper2base = ^bT_g; therefore ^bT_c = ^bT_g * ^gT_c.
            T_base2cam = T_gripper2base @ T_gripper2cam
            R = T_base2cam[:3, :3]
            t = T_base2cam[:3, 3]
            pts_base = (R @ pts_cam.T).T + t

            rgb = color[rows, cols, ::-1].astype(np.float32) / 255.0
            all_points.append(pts_base)
            all_colors.append(rgb)

        if not all_points:
            rospy.logwarn("No points available to visualize")
            return None, None

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        return points, colors

    def show_fused_pointcloud(self, T_gripper2cam: np.ndarray, fused_pcd=None):
        pcd = fused_pcd if fused_pcd is not None else self.build_fused_pointcloud_o3d(T_gripper2cam)
        if pcd is None:
            return

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        rospy.loginfo(f"Visualizing fused cloud with {len(pcd.points)} points")
        o3d.visualization.draw_geometries([pcd, frame], window_name="HandEye Final PointCloud")


def main():
    rospy.init_node("handeye_calibration_apriltag")

    try:
        calibrator = HandEyeCalibrator()
        if calibrator.auto_mode:
            calibrator.run_auto_sequence()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
