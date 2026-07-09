#!/home/gyanig/anaconda3/bin/python3
"""Generate top-down LEGO grasp candidates from SAM instance segmentation."""

import copy
import math
import os
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from tabletop_workspace_opt.msg import ValidGoal, ValidGoals

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def _normalize(vec, fallback=None):
    arr = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        if fallback is None:
            return np.zeros(3, dtype=np.float64)
        return np.array(fallback, dtype=np.float64)
    return arr / norm


def _parse_float_list_param(name, default, expected_len=None):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        values = [float(v) for v in raw]
    elif isinstance(raw, str):
        text = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        values = [float(v) for v in text.split() if v]
    else:
        values = [float(raw)]
    if expected_len is not None and len(values) != expected_len:
        return np.array(default, dtype=np.float64)
    return np.array(values, dtype=np.float64)


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


def _make_point(xyz):
    p = Point()
    p.x = float(xyz[0])
    p.y = float(xyz[1])
    p.z = float(xyz[2])
    return p


def _make_pose_stamped(frame_id, stamp, position, rotation):
    quat = _matrix_to_quat_xyzw(rotation)
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose = Pose()
    msg.pose.position = _make_point(position)
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def _compute_axis_alignment_rotation(forward_w, up_ref, ee_forward_axis, ee_up_axis):
    forward_w = _normalize(forward_w, [0.0, 0.0, -1.0])
    up_ref = _normalize(up_ref, [0.0, 1.0, 0.0])
    right_w = np.cross(forward_w, up_ref)
    if np.linalg.norm(right_w) < 1e-6:
        right_w = np.cross(forward_w, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    right_w = _normalize(right_w, [1.0, 0.0, 0.0])
    up_w = _normalize(np.cross(right_w, forward_w), [0.0, 1.0, 0.0])
    right_w = _normalize(np.cross(forward_w, up_w), [1.0, 0.0, 0.0])

    f_l = _normalize(ee_forward_axis, [0.0, 0.0, 1.0])
    u_l = _normalize(ee_up_axis, [0.0, 1.0, 0.0])
    u_l = _normalize(u_l - np.dot(u_l, f_l) * f_l, [0.0, 1.0, 0.0])
    r_l = _normalize(np.cross(f_l, u_l), [1.0, 0.0, 0.0])
    u_l = _normalize(np.cross(r_l, f_l), [0.0, 1.0, 0.0])

    world_basis = np.column_stack((right_w, up_w, forward_w))
    local_basis = np.column_stack((r_l, u_l, f_l))
    return world_basis @ local_basis.T


def _tf_to_matrix(tf_msg):
    q = tf_msg.transform.rotation
    t = tf_msg.transform.translation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix([q.x, q.y, q.z, q.w])
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def _principal_axis_from_mask(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size < 8:
        return None, None
    pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    center = np.mean(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2):
        return center, None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    major = major / max(np.linalg.norm(major), 1e-9)
    return center, major


def _project_base_point_to_image(xyz_base, T_base_cam, camera_matrix):
    T_cam_base = np.linalg.inv(T_base_cam)
    point_base_h = np.ones(4, dtype=np.float64)
    point_base_h[:3] = np.array(xyz_base, dtype=np.float64)
    point_cam = T_cam_base @ point_base_h
    if point_cam[2] <= 1e-6:
        return None
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    u = fx * (point_cam[0] / point_cam[2]) + cx
    v = fy * (point_cam[1] / point_cam[2]) + cy
    return np.array([u, v], dtype=np.float32)


def _long_axis_from_rect(rect):
    (_, _), (w, h), angle_deg = rect
    if max(w, h) < 1e-6:
        return None
    angle_rad = math.radians(float(angle_deg))
    axis = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
    if h > w:
        axis = np.array([-math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


class SamLegoGraspNode:
    def __init__(self):
        rospy.init_node("sam_lego_grasp_node")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.camera_frame = str(rospy.get_param("~camera_frame", "camera_color_optical_frame")).strip()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info")).strip()
        self.sam_model_path = str(rospy.get_param("~sam_model_path", "/home/gyanig/catkin_ws/src/graspnet-baseline/sam_vit_b_01ec64.pth")).strip()
        self.sam_model_type = str(rospy.get_param("~sam_model_type", "vit_b")).strip()
        self.sam_image_max_side = int(rospy.get_param("~sam_image_max_side", 800))
        self.sam_points_per_side = int(rospy.get_param("~sam_points_per_side", 16))
        self.sam_crop_n_layers = int(rospy.get_param("~sam_crop_n_layers", 0))
        self.sam_pred_iou_thresh = float(rospy.get_param("~sam_pred_iou_thresh", 0.88))
        self.sam_stability_score_thresh = float(rospy.get_param("~sam_stability_score_thresh", 0.92))
        self.sam_device = str(rospy.get_param("~sam_device", "cuda" if torch.cuda.is_available() else "cpu")).strip()
        self.process_rate_hz = float(rospy.get_param("~process_rate_hz", 1.0))
        self.allowed_color_groups = [s.strip().lower() for s in str(rospy.get_param("~allowed_color_groups", "warm,cool")).split(",") if s.strip()]
        self.warm_hsv_lower = _parse_float_list_param("~warm_hsv_lower", [8, 70, 70], 3).astype(np.uint8)
        self.warm_hsv_upper = _parse_float_list_param("~warm_hsv_upper", [42, 255, 255], 3).astype(np.uint8)
        self.cool_hsv_lower = _parse_float_list_param("~cool_hsv_lower", [90, 50, 50], 3).astype(np.uint8)
        self.cool_hsv_upper = _parse_float_list_param("~cool_hsv_upper", [160, 255, 255], 3).astype(np.uint8)
        self.min_mask_area_px = float(rospy.get_param("~min_mask_area_px", 800.0))
        self.max_mask_area_px = float(rospy.get_param("~max_mask_area_px", 250000.0))
        self.min_rect_aspect_ratio = float(rospy.get_param("~min_rect_aspect_ratio", 1.0))
        self.max_rect_aspect_ratio = float(rospy.get_param("~max_rect_aspect_ratio", 10.0))
        self.min_fill_ratio = float(rospy.get_param("~min_fill_ratio", 0.45))
        self.contained_mask_overlap_threshold = float(rospy.get_param("~contained_mask_overlap_threshold", 0.80))
        self.small_round_area_px = float(rospy.get_param("~small_round_area_px", 4000.0))
        self.small_round_aspect_threshold = float(rospy.get_param("~small_round_aspect_threshold", 1.35))
        self.max_candidates = int(rospy.get_param("~max_candidates", 12))
        self.min_relative_candidate_area = float(rospy.get_param("~min_relative_candidate_area", 0.25))
        self.candidate_id_sort_mode = str(rospy.get_param("~candidate_id_sort_mode", "image_position")).strip().lower()
        self.candidate_id_offset = int(rospy.get_param("~candidate_id_offset", 0))
        self.track_match_distance_m = float(rospy.get_param("~track_match_distance_m", 0.08))
        self.track_max_missing_frames = int(rospy.get_param("~track_max_missing_frames", 8))
        self.track_position_alpha = float(rospy.get_param("~track_position_alpha", 0.35))
        self.publish_missing_tracks = bool(rospy.get_param("~publish_missing_tracks", False))
        self.table_z = float(rospy.get_param("~fixed_table_z", -0.255))
        self.object_height_m = float(rospy.get_param("~object_height_m", 0.0192))
        self.grasp_height_offset_m = float(rospy.get_param("~grasp_height_offset_m", 0.135))
        self.pregrasp_offset_m = float(rospy.get_param("~pregrasp_offset_m", 0.240))
        self.world_up_axis = _normalize(_parse_float_list_param("~world_up_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_forward_axis = _normalize(_parse_float_list_param("~ee_forward_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_up_axis = _normalize(_parse_float_list_param("~ee_up_axis", [0.0, 1.0, 0.0], 3), [0.0, 1.0, 0.0])
        self.selection_reference_uv = _parse_float_list_param("~selection_reference_uv", [], None)
        self.table_roi_min_xy = _parse_float_list_param("~table_roi_min_xy", [0.25, -0.30], 2)
        self.table_roi_max_xy = _parse_float_list_param("~table_roi_max_xy", [0.85, 0.30], 2)
        self.birdseye_resolution_m = float(rospy.get_param("~birdseye_resolution_m", 0.0025))

        self.markers_topic = str(rospy.get_param("~markers_topic", "~markers")).strip()
        self.debug_image_topic = str(rospy.get_param("~debug_image_topic", "~debug_image")).strip()
        self.birdseye_debug_topic = str(rospy.get_param("~birdseye_debug_topic", "~birdseye_debug")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.grasp_pose_topic = str(rospy.get_param("~grasp_pose_topic", "~grasp_pose")).strip()
        self.pregrasp_pose_topic = str(rospy.get_param("~pregrasp_pose_topic", "~pregrasp_pose")).strip()
        self.valid_goals_topic = str(rospy.get_param("~valid_goals_topic", "~valid_goals")).strip()
        self.detections_topic = str(rospy.get_param("~detections_topic", "~detections")).strip()
        self.candidate_namespace_prefix = str(rospy.get_param("~candidate_namespace_prefix", "/sam_lego_candidates/candidate_")).strip()

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_matrix = None
        self.latest_image = None
        self.latest_image_stamp = rospy.Time(0)
        self.last_process_time = rospy.Time(0)
        self.processing_lock = threading.Lock()
        self.last_status = ""
        self.track_slots = {}

        self.markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1, latch=True)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.birdseye_debug_pub = rospy.Publisher(self.birdseye_debug_topic, Image, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.grasp_pub = rospy.Publisher(self.grasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.pregrasp_pub = rospy.Publisher(self.pregrasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.valid_goals_pub = rospy.Publisher(self.valid_goals_topic, ValidGoals, queue_size=1, latch=True)
        self.detections_pub = rospy.Publisher(self.detections_topic, Detection2DArray, queue_size=1, latch=True)
        self.candidate_pose_pubs = {}

        self.sam_mask_generator = None
        self._load_sam_model()

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self._process_timer_cb)

        self._publish_status("waiting_for_image topic={}".format(self.image_topic))

    def _load_sam_model(self):
        if not self.sam_model_path:
            self._publish_status("sam_model_missing set ~sam_model_path to local sam_vit_*.pth")
            return
        if not os.path.exists(self.sam_model_path):
            self._publish_status("sam_model_missing path={}".format(self.sam_model_path))
            return
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path)
            sam.to(device=self.sam_device)
            sam.eval()
            self.sam_mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=self.sam_points_per_side,
                pred_iou_thresh=self.sam_pred_iou_thresh,
                stability_score_thresh=self.sam_stability_score_thresh,
                crop_n_layers=self.sam_crop_n_layers,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=int(self.min_mask_area_px),
            )
            rospy.loginfo("SAM model loaded from %s on %s", self.sam_model_path, self.sam_device)
        except Exception as exc:
            self._publish_status("sam_model_load_failed {}".format(exc))

    def camera_info_cb(self, msg):
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
            if msg.header.frame_id:
                self.camera_frame = str(msg.header.frame_id).strip()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[sam_lego_grasp_node] image decode failed: %s", exc)

    def _process_timer_cb(self, _event):
        if self.camera_matrix is None or self.sam_mask_generator is None or self.latest_image is None:
            return
        now = rospy.Time.now()
        if self.process_rate_hz > 0.0 and (now - self.last_process_time).to_sec() < (1.0 / self.process_rate_hz):
            return
        if not self.processing_lock.acquire(False):
            return
        try:
            self.last_process_time = now
            self._process_image(self.latest_image_stamp)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[sam_lego_grasp_node] image processing failed: %s", exc)
        finally:
            self.processing_lock.release()

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[sam_lego_grasp_node] %s", text)
        self.status_pub.publish(String(data=text))

    def _process_image(self, stamp):
        image = self.latest_image
        if image is None:
            return
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb_small, scale = self._downscale(rgb)
            masks_dicts = self.sam_mask_generator.generate(rgb_small)
        except Exception as exc:
            self._publish_status("sam_predict_failed {}".format(exc))
            return
        T_base_cam = self._lookup_base_to_camera(stamp)
        if T_base_cam is None:
            self._publish_status("tf_lookup_failed {}->{}".format(self.base_frame, self.camera_frame))
            return
        birdseye = self._compute_birdseye_geometry(T_base_cam)
        if birdseye is None:
            self._publish_status("birdseye_geometry_failed")
            return
        candidates = self._extract_candidates(image, masks_dicts, scale, birdseye)
        if not candidates:
            self.markers_pub.publish(self._make_empty_markers())
            self._publish_debug_image(image, [])
            self._publish_birdseye_debug_image(birdseye, [])
            self._publish_valid_goals([], stamp)
            self._publish_status("no_sam_candidates")
            return
        solutions = []
        for candidate in candidates[: self.max_candidates]:
            solution = self._build_solution(candidate, stamp)
            if solution is not None:
                solutions.append(solution)
        if not solutions:
            self.markers_pub.publish(self._make_empty_markers())
            self._publish_debug_image(image, [])
            self._publish_birdseye_debug_image(birdseye, [])
            self._publish_valid_goals([], stamp)
            self._publish_status("sam_candidates_projection_failed")
            return

        solutions = self._order_solutions(solutions)
        solutions = self._update_tracks(solutions)
        if not solutions:
            self.markers_pub.publish(self._make_empty_markers())
            self._publish_debug_image(image, [])
            self._publish_birdseye_debug_image(birdseye, [])
            self._publish_valid_goals([], stamp)
            self._publish_status("sam_tracks_unavailable")
            return

        selected = self._select_primary_candidate(solutions, image.shape[1], image.shape[0])
        self.markers_pub.publish(self._make_markers(solutions, selected, stamp))
        self._publish_debug_image(image, solutions)
        self._publish_birdseye_debug_image(birdseye, solutions)
        self._publish_valid_goals(solutions, stamp)
        self._publish_detection_candidates(solutions, stamp)
        self._publish_candidate_pose_topics(solutions, stamp)
        self.grasp_pub.publish(selected["grasp_pose"])
        self.pregrasp_pub.publish(selected["pregrasp_pose"])
        self._publish_status("tracking_sam_candidates total={} selected={} color_group={}".format(
            len(solutions),
            int(selected["candidate_id"]),
            selected["color_group"],
        ))

    def _extract_candidates(self, image, masks_dicts, scale, birdseye):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        candidates = []
        if masks_dicts is None:
            return candidates
        for md in masks_dicts:
            mask = md.get("segmentation")
            if mask is None:
                continue
            mask_u8 = mask.astype(np.uint8)
            if scale != 1.0:
                h, w = image.shape[:2]
                mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if contour.shape[0] < 4:
                continue
            area = float(cv2.contourArea(contour))
            if area < self.min_mask_area_px or area > self.max_mask_area_px:
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (w, h), _ = rect
            if min(w, h) < 2.0:
                continue
            rect_area = float(w * h)
            if rect_area < 1e-6:
                continue
            aspect = max(w, h) / max(min(w, h), 1e-6)
            fill_ratio = area / rect_area
            if aspect < self.min_rect_aspect_ratio or aspect > self.max_rect_aspect_ratio:
                continue
            if fill_ratio < self.min_fill_ratio:
                continue
            contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            group = self._classify_color_group(hsv, contour_mask)
            if group not in self.allowed_color_groups:
                continue
            refined_mask = self._refine_mask_by_color_group(hsv, contour_mask, group)
            if refined_mask is None:
                continue
            refined_contours, _ = cv2.findContours(refined_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not refined_contours:
                continue
            contour = max(refined_contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area < self.min_mask_area_px or area > self.max_mask_area_px:
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (w, h), _ = rect
            if min(w, h) < 2.0:
                continue
            rect_area = float(w * h)
            if rect_area < 1e-6:
                continue
            aspect = max(w, h) / max(min(w, h), 1e-6)
            fill_ratio = area / rect_area
            if aspect < self.min_rect_aspect_ratio or aspect > self.max_rect_aspect_ratio:
                continue
            if fill_ratio < self.min_fill_ratio:
                continue
            if area <= self.small_round_area_px and aspect <= self.small_round_aspect_threshold:
                continue
            moments = cv2.moments(contour)
            if abs(float(moments.get("m00", 0.0))) > 1e-6:
                center_uv = np.array(
                    [
                        float(moments["m10"]) / float(moments["m00"]),
                        float(moments["m01"]) / float(moments["m00"]),
                    ],
                    dtype=np.float64,
                )
            else:
                center_uv = np.array(rect[0], dtype=np.float64)
            pca_center_uv, major_dir_uv = _principal_axis_from_mask(refined_mask.astype(bool))
            if pca_center_uv is not None:
                center_uv = pca_center_uv
            birdseye_mask = cv2.warpPerspective(
                refined_mask.astype(np.uint8) * 255,
                birdseye["H_img_to_be"],
                (birdseye["width_px"], birdseye["height_px"]),
                flags=cv2.INTER_NEAREST,
            )
            if int(np.count_nonzero(birdseye_mask)) < max(20, int(self.min_mask_area_px * 0.05)):
                continue
            birdseye_contours, _ = cv2.findContours(birdseye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not birdseye_contours:
                continue
            be_contour = max(birdseye_contours, key=cv2.contourArea)
            be_area = float(cv2.contourArea(be_contour))
            if be_area < 20.0:
                continue
            be_rect = cv2.minAreaRect(be_contour)
            (_, _), (be_w, be_h), _ = be_rect
            if min(be_w, be_h) < 2.0:
                continue
            be_center_px = np.array(be_rect[0], dtype=np.float64)
            be_major_dir_px = _long_axis_from_rect(be_rect)
            if be_major_dir_px is None:
                continue
            center_xy = self._birdseye_pixel_to_table_xy(be_center_px, birdseye)
            if center_xy is None:
                continue
            long_axis = np.array(
                [
                    float(be_major_dir_px[0]),
                    -float(be_major_dir_px[1]),
                    0.0,
                ],
                dtype=np.float64,
            )
            long_axis = _normalize(long_axis, [1.0, 0.0, 0.0])
            candidates.append(
                {
                    "contour": contour,
                    "birdseye_contour": be_contour,
                    "rect": rect,
                    "center_uv": center_uv,
                    "major_dir_uv": major_dir_uv,
                    "birdseye_center_px": be_center_px,
                    "birdseye_major_dir_px": be_major_dir_px,
                    "center_xy": center_xy,
                    "long_axis_table": long_axis,
                    "area": area,
                    "aspect": aspect,
                    "fill_ratio": fill_ratio,
                    "birdseye_area": be_area,
                    "color_group": group,
                    "pred_iou": float(md.get("predicted_iou", 0.0)),
                    "stability": float(md.get("stability_score", 0.0)),
                    "mask": refined_mask.astype(bool),
                    "birdseye_mask": birdseye_mask > 0,
                }
            )
        candidates = sorted(
            candidates,
            key=lambda item: (float(item["birdseye_area"]), float(item["area"]), float(item["pred_iou"])),
            reverse=True,
        )
        candidates = self._suppress_contained_candidates(candidates)
        candidates = self._filter_relative_area_candidates(candidates)
        return candidates

    def _suppress_contained_candidates(self, candidates):
        kept = []
        for candidate in candidates:
            suppressed = False
            mask_small = candidate["mask"]
            area_small = max(float(candidate["area"]), 1.0)
            for other in kept:
                if candidate["color_group"] != other["color_group"]:
                    continue
                mask_big = other["mask"]
                overlap = float(np.logical_and(mask_small, mask_big).sum()) / area_small
                if overlap >= self.contained_mask_overlap_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(candidate)
        return kept

    def _filter_relative_area_candidates(self, candidates):
        if not candidates:
            return candidates
        largest_area = max(float(c.get("birdseye_area", 0.0)) for c in candidates)
        if largest_area <= 1e-6:
            return candidates
        min_area = largest_area * max(0.0, float(self.min_relative_candidate_area))
        return [c for c in candidates if float(c.get("birdseye_area", 0.0)) >= min_area]

    def _downscale(self, rgb):
        h, w = rgb.shape[:2]
        max_side = max(h, w)
        if self.sam_image_max_side <= 0 or max_side <= self.sam_image_max_side:
            return rgb, 1.0
        scale = float(self.sam_image_max_side) / float(max_side)
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    def _classify_color_group(self, hsv_image, mask):
        pixels = hsv_image[mask > 0]
        if pixels.size == 0:
            return "unknown"
        mean_h = float(np.mean(pixels[:, 0]))
        if 8.0 <= mean_h <= 42.0:
            return "warm"
        if 90.0 <= mean_h <= 160.0:
            return "cool"
        return "unknown"

    def _refine_mask_by_color_group(self, hsv_image, contour_mask, group_name):
        if group_name == "warm":
            color_mask = cv2.inRange(hsv_image, self.warm_hsv_lower, self.warm_hsv_upper)
        elif group_name == "cool":
            color_mask = cv2.inRange(hsv_image, self.cool_hsv_lower, self.cool_hsv_upper)
        else:
            return contour_mask.astype(bool)

        refined = cv2.bitwise_and(color_mask, contour_mask)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8), connectivity=8)
        if n_labels <= 1:
            return None
        largest_label = None
        largest_area = 0
        for label_idx in range(1, n_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area > largest_area:
                largest_area = area
                largest_label = label_idx
        if largest_label is None or largest_area < self.min_mask_area_px:
            return None
        return labels == largest_label

    def _build_solution(self, candidate, stamp):
        center_base = np.array(
            [
                float(candidate["center_xy"][0]),
                float(candidate["center_xy"][1]),
                float(self.table_z),
            ],
            dtype=np.float64,
        )
        center_base = center_base + self.world_up_axis * (0.5 * self.object_height_m)
        long_axis = candidate.get("long_axis_table")
        if long_axis is None:
            return None
        long_axis = _normalize(long_axis - np.dot(long_axis, self.world_up_axis) * self.world_up_axis, [1.0, 0.0, 0.0])
        short_axis = _normalize(np.cross(self.world_up_axis, long_axis), [0.0, 1.0, 0.0])
        if np.dot(np.cross(long_axis, short_axis), self.world_up_axis) < 0.0:
            short_axis = -short_axis
        rotation = _compute_axis_alignment_rotation(
            forward_w=-self.world_up_axis,
            up_ref=long_axis,
            ee_forward_axis=self.ee_forward_axis,
            ee_up_axis=self.ee_up_axis,
        )
        grasp_pos = center_base + self.world_up_axis * self.grasp_height_offset_m
        pregrasp_pos = grasp_pos + self.world_up_axis * self.pregrasp_offset_m
        return {
            "center_uv": candidate["center_uv"],
            "centroid": center_base,
            "long_axis": long_axis,
            "short_axis": short_axis,
            "up_axis": self.world_up_axis.copy(),
            "contour": candidate["contour"],
            "birdseye_contour": candidate["birdseye_contour"],
            "birdseye_center_px": candidate["birdseye_center_px"],
            "color_group": candidate["color_group"],
            "grasp_pose": _make_pose_stamped(self.base_frame, stamp, grasp_pos, rotation),
            "pregrasp_pose": _make_pose_stamped(self.base_frame, stamp, pregrasp_pos, rotation),
        }

    def _select_primary_candidate(self, solutions, width, height):
        if not solutions:
            return None
        if self.selection_reference_uv.size == 2:
            ref_uv = self.selection_reference_uv
        else:
            ref_uv = np.array([0.5 * width, 0.5 * height], dtype=np.float64)
        return min(solutions, key=lambda s: float(np.linalg.norm(s["center_uv"] - ref_uv)))

    def _order_solutions(self, solutions):
        if self.candidate_id_sort_mode == "image_position":
            return sorted(
                solutions,
                key=lambda s: (
                    float(s["center_uv"][1]),
                    float(s["center_uv"][0]),
                ),
            )
        if self.candidate_id_sort_mode == "x_then_y":
            return sorted(
                solutions,
                key=lambda s: (
                    float(s["centroid"][0]),
                    float(s["centroid"][1]),
                ),
            )
        if self.candidate_id_sort_mode == "y_then_x":
            return sorted(
                solutions,
                key=lambda s: (
                    float(s["centroid"][1]),
                    float(s["centroid"][0]),
                ),
            )
        return list(solutions)

    def _update_tracks(self, solutions):
        active_tracks = {slot: data for slot, data in self.track_slots.items() if data.get("solution") is not None}
        unmatched_slots = set(active_tracks.keys())
        unmatched_solution_indices = set(range(len(solutions)))
        matches = []

        candidate_pairs = []
        for sol_idx, solution in enumerate(solutions):
            for slot, track in active_tracks.items():
                prev = track.get("solution")
                if prev is None:
                    continue
                dist = float(np.linalg.norm(solution["centroid"] - prev["centroid"]))
                if dist <= self.track_match_distance_m:
                    candidate_pairs.append((dist, slot, sol_idx))
        candidate_pairs.sort(key=lambda item: item[0])

        for _, slot, sol_idx in candidate_pairs:
            if slot not in unmatched_slots or sol_idx not in unmatched_solution_indices:
                continue
            matches.append((slot, sol_idx, True))
            unmatched_slots.remove(slot)
            unmatched_solution_indices.remove(sol_idx)

        new_slots = [slot for slot in range(self.max_candidates) if slot not in self.track_slots]
        free_slots = sorted(list(unmatched_slots) + new_slots)
        for sol_idx in sorted(unmatched_solution_indices):
            if not free_slots:
                break
            slot = free_slots.pop(0)
            matches.append((slot, sol_idx, False))

        next_tracks = {}
        matched_slots = set()
        for slot, sol_idx, matched_by_distance in matches:
            matched_slots.add(slot)
            current_solution = copy.deepcopy(solutions[sol_idx])
            previous_track = self.track_slots.get(slot, {})
            previous_solution = previous_track.get("solution")
            if matched_by_distance and previous_solution is not None:
                current_solution = self._smooth_solution(previous_solution, current_solution)
            current_solution["candidate_index"] = slot
            current_solution["candidate_id"] = self.candidate_id_offset + slot
            current_solution["is_visible"] = True
            next_tracks[slot] = {
                "solution": current_solution,
                "missing": 0,
            }

        for slot, track in self.track_slots.items():
            if slot in matched_slots:
                continue
            missing = int(track.get("missing", 0)) + 1
            if missing > self.track_max_missing_frames:
                continue
            carried = copy.deepcopy(track.get("solution"))
            if carried is None:
                continue
            carried["candidate_index"] = slot
            carried["candidate_id"] = self.candidate_id_offset + slot
            carried["is_visible"] = False
            next_tracks[slot] = {
                "solution": carried,
                "missing": missing,
            }

        self.track_slots = next_tracks
        ordered_slots = sorted(self.track_slots.keys())
        solutions_out = []
        for slot in ordered_slots:
            solution = self.track_slots[slot]["solution"]
            if self.publish_missing_tracks or solution.get("is_visible", True):
                solutions_out.append(solution)
        return solutions_out

    def _smooth_solution(self, previous, current):
        alpha = float(np.clip(self.track_position_alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return current
        out = copy.deepcopy(current)
        out["center_uv"] = (1.0 - alpha) * previous["center_uv"] + alpha * current["center_uv"]
        out["centroid"] = (1.0 - alpha) * previous["centroid"] + alpha * current["centroid"]

        for key in ("grasp_pose", "pregrasp_pose"):
            prev_pose = previous[key]
            cur_pose = out[key]
            cur_pose.pose.position.x = float((1.0 - alpha) * prev_pose.pose.position.x + alpha * cur_pose.pose.position.x)
            cur_pose.pose.position.y = float((1.0 - alpha) * prev_pose.pose.position.y + alpha * cur_pose.pose.position.y)
            cur_pose.pose.position.z = float((1.0 - alpha) * prev_pose.pose.position.z + alpha * cur_pose.pose.position.z)
        return out

    def _lookup_base_to_camera(self, stamp):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0) if stamp == rospy.Time() else stamp,
                rospy.Duration(0.2),
            )
            return _tf_to_matrix(tf_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def _pixel_to_plane(self, u, v, T_base_cam, plane_height):
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        ray_cam = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
        ray_base = _normalize(T_base_cam[:3, :3] @ ray_cam, [0.0, 0.0, 1.0])
        origin_base = T_base_cam[:3, 3]
        denom = float(np.dot(self.world_up_axis, ray_base))
        if abs(denom) < 1e-6:
            return None
        t = (float(plane_height) - float(np.dot(self.world_up_axis, origin_base))) / denom
        if t <= 0.0:
            return None
        return origin_base + t * ray_base

    def _compute_birdseye_geometry(self, T_base_cam):
        x_min = float(self.table_roi_min_xy[0])
        y_min = float(self.table_roi_min_xy[1])
        x_max = float(self.table_roi_max_xy[0])
        y_max = float(self.table_roi_max_xy[1])
        resolution = max(float(self.birdseye_resolution_m), 1e-4)
        width_px = max(32, int(math.ceil((x_max - x_min) / resolution)))
        height_px = max(32, int(math.ceil((y_max - y_min) / resolution)))
        src_points = []
        dst_points = np.array(
            [
                [0.0, 0.0],
                [float(width_px - 1), 0.0],
                [float(width_px - 1), float(height_px - 1)],
                [0.0, float(height_px - 1)],
            ],
            dtype=np.float32,
        )
        corners_base = [
            [x_min, y_max, self.table_z],
            [x_max, y_max, self.table_z],
            [x_max, y_min, self.table_z],
            [x_min, y_min, self.table_z],
        ]
        for corner in corners_base:
            uv = _project_base_point_to_image(corner, T_base_cam, self.camera_matrix)
            if uv is None:
                return None
            src_points.append(uv)
        src_points = np.array(src_points, dtype=np.float32)
        H_img_to_be = cv2.getPerspectiveTransform(src_points, dst_points)
        return {
            "H_img_to_be": H_img_to_be,
            "width_px": width_px,
            "height_px": height_px,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "resolution": resolution,
        }

    def _birdseye_pixel_to_table_xy(self, center_px, birdseye):
        if center_px is None:
            return None
        x = float(birdseye["x_min"]) + float(center_px[0]) * float(birdseye["resolution"])
        y = float(birdseye["y_max"]) - float(center_px[1]) * float(birdseye["resolution"])
        return np.array([x, y], dtype=np.float64)

    def _make_empty_markers(self):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        return markers

    def _publish_birdseye_debug_image(self, birdseye, solutions):
        canvas = np.zeros((birdseye["height_px"], birdseye["width_px"], 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        for solution in solutions:
            contour = solution.get("birdseye_contour")
            if contour is not None:
                cv2.drawContours(canvas, [contour], -1, (0, 0, 255), 2)
            center_px = solution.get("birdseye_center_px")
            if center_px is not None:
                center_int = (int(round(center_px[0])), int(round(center_px[1])))
                cv2.circle(canvas, center_int, 4, (0, 255, 0), -1)
                cv2.putText(
                    canvas,
                    str(int(solution.get("candidate_id", -1))),
                    (center_int[0] + 6, center_int[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        try:
            msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            self.birdseye_debug_pub.publish(msg)
        except Exception:
            pass

    def _make_markers(self, solutions, selected, stamp):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        for idx, solution in enumerate(solutions):
            centroid = solution["centroid"]
            is_selected = solution is selected
            alpha = 0.95 if is_selected else 0.40
            base_id = idx * 100

            sphere = Marker()
            sphere.header.frame_id = self.base_frame
            sphere.header.stamp = stamp
            sphere.ns = "sam_lego_candidates"
            sphere.id = base_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = _make_point(centroid)
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.018
            sphere.scale.y = 0.018
            sphere.scale.z = 0.018
            sphere.color = ColorRGBA(1.0, 0.85, 0.1, alpha)
            markers.markers.append(sphere)

            for offset, pose_msg, color, label_text in [
                (20, solution["pregrasp_pose"], ColorRGBA(1.0, 0.6, 0.1, alpha), "pregrasp"),
                (21, solution["grasp_pose"], ColorRGBA(0.2, 1.0, 0.8, alpha), "grasp"),
            ]:
                arrow = Marker()
                arrow.header.frame_id = self.base_frame
                arrow.header.stamp = stamp
                arrow.ns = "sam_lego_grasp_pose"
                arrow.id = base_id + offset
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose = pose_msg.pose
                arrow.scale.x = 0.08 if is_selected else 0.06
                arrow.scale.y = 0.012 if is_selected else 0.009
                arrow.scale.z = 0.012 if is_selected else 0.009
                arrow.color = color
                markers.markers.append(arrow)

            label = Marker()
            label.header.frame_id = self.base_frame
            label.header.stamp = stamp
            label.ns = "sam_lego_labels"
            label.id = base_id + 90
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = _make_point(centroid + np.array([0.0, 0.0, 0.03], dtype=np.float64))
            label.pose.orientation.w = 1.0
            label.scale.z = 0.02
            label.color = ColorRGBA(1.0, 1.0, 1.0, alpha)
            label.text = "{}_{}".format(solution["color_group"], int(solution["candidate_id"]))
            markers.markers.append(label)
        return markers

    def _publish_debug_image(self, image, solutions):
        if self.debug_image_pub.get_num_connections() <= 0:
            return
        vis = image.copy()
        for idx, solution in enumerate(solutions):
            contour = solution["contour"].astype(np.int32)
            color = (0, 0, 255) if idx == solution.get("candidate_index", -1) else (0, 255, 255)
            cv2.drawContours(vis, [contour], -1, color, 2)
            center_uv = solution["center_uv"]
            cv2.circle(vis, (int(center_uv[0]), int(center_uv[1])), 5, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "{}_{}".format(solution["color_group"][:1], int(solution["candidate_id"])),
                (int(center_uv[0]) + 6, int(center_uv[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))
        except Exception:
            pass

    def _publish_valid_goals(self, solutions, stamp):
        msg = ValidGoals()
        msg.header.stamp = stamp
        msg.current_state = "lego_pick"
        for solution in solutions:
            goal = ValidGoal()
            goal.goal_id = "lego_{}".format(int(solution["candidate_id"]))
            goal.action_type = "pick"
            goal.target_position = _make_point(
                np.array([
                    solution["grasp_pose"].pose.position.x,
                    solution["grasp_pose"].pose.position.y,
                    solution["grasp_pose"].pose.position.z,
                ], dtype=np.float64)
            )
            goal.object_name = "{}_lego".format(solution["color_group"])
            msg.goals.append(goal)
        self.valid_goals_pub.publish(msg)

    def _publish_detection_candidates(self, solutions, stamp):
        msg = Detection2DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        for solution in solutions:
            det = Detection2D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.id = int(solution["candidate_id"])
            hyp.score = 1.0
            hyp.pose.pose = solution["pregrasp_pose"].pose
            det.results.append(hyp)
            msg.detections.append(det)
        self.detections_pub.publish(msg)

    def _candidate_publishers(self, idx):
        if idx not in self.candidate_pose_pubs:
            ns = "{}{}".format(self.candidate_namespace_prefix, idx)
            self.candidate_pose_pubs[idx] = {
                "pregrasp": rospy.Publisher("{}/pregrasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
                "grasp": rospy.Publisher("{}/grasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
            }
        return self.candidate_pose_pubs[idx]

    def _publish_candidate_pose_topics(self, solutions, stamp):
        for solution in solutions:
            pubs = self._candidate_publishers(int(solution["candidate_id"]))
            pubs["pregrasp"].publish(solution["pregrasp_pose"])
            pubs["grasp"].publish(solution["grasp_pose"])


def main():
    SamLegoGraspNode()
    rospy.spin()


if __name__ == "__main__":
    main()
