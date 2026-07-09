#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate top-down sandwich grasp candidates from SAM masks."""

import json
import math
import os
import threading
from datetime import datetime

import cv2
import numpy as np
import rospy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA, Int32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from tabletop_workspace_opt.msg import ValidGoal, ValidGoals

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def _is_cuda_oom(exc):
    text = str(exc).lower()
    return "cuda" in text and "out of memory" in text


def _normalize(vec, fallback=None):
    arr = np.array(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.array(fallback if fallback is not None else [0.0, 0.0, 0.0], dtype=np.float64)
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


def _parse_hsv_triplet(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
    except Exception:
        return None


def _hue_distance_deg(a, b):
    delta = abs(float(a) - float(b))
    return min(delta, 180.0 - delta)


def _hsv_distance(obs_hsv, ref_hsv):
    if obs_hsv is None or ref_hsv is None:
        return 0.0
    hue_term = _hue_distance_deg(obs_hsv[0], ref_hsv[0]) / 90.0
    sat_term = abs(float(obs_hsv[1]) - float(ref_hsv[1])) / 255.0
    val_term = abs(float(obs_hsv[2]) - float(ref_hsv[2])) / 255.0
    return float(hue_term + 0.35 * sat_term + 0.20 * val_term)


def _ratio_in_hsv_range(pixels, hue_ranges=None, sat_min=None, sat_max=None, val_min=None, val_max=None):
    if pixels is None or len(pixels) == 0:
        return 0.0
    mask = np.ones((pixels.shape[0],), dtype=bool)
    if hue_ranges:
        hue_mask = np.zeros((pixels.shape[0],), dtype=bool)
        hue = pixels[:, 0]
        for lo, hi in hue_ranges:
            if lo <= hi:
                hue_mask |= (hue >= lo) & (hue <= hi)
            else:
                hue_mask |= (hue >= lo) | (hue <= hi)
        mask &= hue_mask
    if sat_min is not None:
        mask &= pixels[:, 1] >= sat_min
    if sat_max is not None:
        mask &= pixels[:, 1] <= sat_max
    if val_min is not None:
        mask &= pixels[:, 2] >= val_min
    if val_max is not None:
        mask &= pixels[:, 2] <= val_max
    return float(np.count_nonzero(mask)) / float(len(pixels))


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


class SamSandwichGraspNode:
    def __init__(self):
        rospy.init_node("sam_sandwich_grasp_node")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.camera_frame = str(rospy.get_param("~camera_frame", "camera_color_optical_frame")).strip()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info")).strip()
        self.candidate_metadata_yaml = os.path.expanduser(
            str(
                rospy.get_param(
                    "~candidate_metadata_yaml",
                    os.path.join(package_root, "config", "sandwich_candidate_metadata.example.yaml"),
                )
            ).strip()
        )
        self.sam_model_path = str(
            rospy.get_param(
                "~sam_model_path",
                "/home/gyanig/catkin_ws/src/graspnet-baseline/sam_vit_b_01ec64.pth",
            )
        ).strip()
        self.sam_model_type = str(rospy.get_param("~sam_model_type", "vit_b")).strip()
        self.sam_image_max_side = int(rospy.get_param("~sam_image_max_side", 800))
        self.sam_points_per_side = int(rospy.get_param("~sam_points_per_side", 16))
        self.sam_crop_n_layers = int(rospy.get_param("~sam_crop_n_layers", 0))
        self.scan_sam_points_per_side = int(rospy.get_param("~scan_sam_points_per_side", 20))
        self.scan_sam_crop_n_layers = int(rospy.get_param("~scan_sam_crop_n_layers", 1))
        self.sam_pred_iou_thresh = float(rospy.get_param("~sam_pred_iou_thresh", 0.88))
        self.sam_stability_score_thresh = float(rospy.get_param("~sam_stability_score_thresh", 0.92))
        self.sam_device = str(rospy.get_param("~sam_device", "cuda" if torch.cuda.is_available() else "cpu")).strip()
        self.oom_retry_image_max_side = int(rospy.get_param("~oom_retry_image_max_side", 512))
        self.oom_retry_points_per_side = int(rospy.get_param("~oom_retry_points_per_side", 8))
        self.oom_fallback_device = str(rospy.get_param("~oom_fallback_device", "cpu")).strip()
        self.color_match_weight = float(rospy.get_param("~color_match_weight", 0.65))
        self.process_rate_hz = float(rospy.get_param("~process_rate_hz", 1.0))
        self.min_mask_area_px = float(rospy.get_param("~min_mask_area_px", 600.0))
        self.max_mask_area_px = float(rospy.get_param("~max_mask_area_px", 250000.0))
        self.max_mask_image_fraction = float(rospy.get_param("~max_mask_image_fraction", 0.18))
        self.border_reject_margin_px = int(rospy.get_param("~border_reject_margin_px", 6))
        self.max_aspect_ratio = float(rospy.get_param("~max_aspect_ratio", 1.45))
        self.min_circularity = float(rospy.get_param("~min_circularity", 0.55))
        self.min_fill_ratio = float(rospy.get_param("~min_fill_ratio", 0.35))
        self.contained_mask_overlap_threshold = float(rospy.get_param("~contained_mask_overlap_threshold", 0.80))
        self.max_candidates = int(rospy.get_param("~max_candidates", 6))
        self.min_relative_candidate_area = float(rospy.get_param("~min_relative_candidate_area", 0.10))
        self.enable_relaxed_single_candidate_fallback = bool(
            rospy.get_param("~enable_relaxed_single_candidate_fallback", True)
        )
        self.relaxed_min_mask_area_px = float(rospy.get_param("~relaxed_min_mask_area_px", 250.0))
        self.relaxed_min_fill_ratio = float(rospy.get_param("~relaxed_min_fill_ratio", 0.18))
        self.relaxed_max_aspect_ratio = float(rospy.get_param("~relaxed_max_aspect_ratio", 1.90))
        self.scan_min_mask_area_px = float(rospy.get_param("~scan_min_mask_area_px", 220.0))
        self.scan_max_aspect_ratio = float(rospy.get_param("~scan_max_aspect_ratio", 3.20))
        self.scan_min_fill_ratio = float(rospy.get_param("~scan_min_fill_ratio", 0.08))
        self.scan_min_circularity = float(rospy.get_param("~scan_min_circularity", 0.10))
        self.scan_min_candidate_radius_m = float(rospy.get_param("~scan_min_candidate_radius_m", 0.010))
        self.scan_max_candidate_radius_m = float(rospy.get_param("~scan_max_candidate_radius_m", 0.070))
        self.scan_border_reject_margin_px = int(rospy.get_param("~scan_border_reject_margin_px", 0))
        self.scan_max_candidates = int(rospy.get_param("~scan_max_candidates", 12))
        self.min_candidate_radius_m = float(rospy.get_param("~min_candidate_radius_m", 0.015))
        self.max_candidate_radius_m = float(rospy.get_param("~max_candidate_radius_m", 0.045))
        self.radius_match_tolerance_m = float(rospy.get_param("~radius_match_tolerance_m", 0.012))
        self.zone_match_tolerance_m = float(rospy.get_param("~zone_match_tolerance_m", 0.035))
        self.table_z = float(rospy.get_param("~fixed_table_z", -0.255))
        self.grasp_height_offset_m = float(rospy.get_param("~grasp_height_offset_m", 0.135))
        self.pregrasp_offset_m = float(rospy.get_param("~pregrasp_offset_m", 0.240))
        self.grasp_xy_offset_m = _parse_float_list_param("~grasp_xy_offset_m", [0.0, 0.0], 2)
        self.selection_reference_uv = _parse_float_list_param("~selection_reference_uv", [], None)
        self.world_up_axis = _normalize(_parse_float_list_param("~world_up_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_forward_axis = _normalize(_parse_float_list_param("~ee_forward_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_up_axis = _normalize(_parse_float_list_param("~ee_up_axis", [0.0, 1.0, 0.0], 3), [0.0, 1.0, 0.0])
        self.table_roi_min_xy = _parse_float_list_param("~table_roi_min_xy", [0.25, -0.30], 2)
        self.table_roi_max_xy = _parse_float_list_param("~table_roi_max_xy", [0.85, 0.30], 2)
        self.birdseye_resolution_m = float(rospy.get_param("~birdseye_resolution_m", 0.0025))
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "/task_context/phase")).strip()

        self.markers_topic = str(rospy.get_param("~markers_topic", "~markers")).strip()
        self.debug_image_topic = str(rospy.get_param("~debug_image_topic", "~debug_image")).strip()
        self.debug_show_mask_fill = bool(rospy.get_param("~debug_show_mask_fill", True))
        self.debug_dump_dir = os.path.expanduser(str(rospy.get_param("~debug_dump_dir", "")).strip())
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.grasp_pose_topic = str(rospy.get_param("~grasp_pose_topic", "~grasp_pose")).strip()
        self.pregrasp_pose_topic = str(rospy.get_param("~pregrasp_pose_topic", "~pregrasp_pose")).strip()
        self.valid_goals_topic = str(rospy.get_param("~valid_goals_topic", "~valid_goals")).strip()
        self.detections_topic = str(rospy.get_param("~detections_topic", "~detections")).strip()
        self.candidate_namespace_prefix = str(
            rospy.get_param("~candidate_namespace_prefix", "/sam_sandwich_candidates/candidate_")
        ).strip()

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_matrix = None
        self.latest_image = None
        self.latest_image_stamp = rospy.Time(0)
        self.last_process_time = rospy.Time(0)
        self.processing_lock = threading.Lock()
        self.last_status = ""
        self.candidate_pose_pubs = {}
        self.metadata_entries = self._load_candidate_metadata()
        self.current_allowed_ids = set()
        self.current_phase = "scan_workspace"
        self._oom_recovery_attempted = False
        self._last_debug_dump_signature = ""

        self.markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1, latch=True)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.grasp_pub = rospy.Publisher(self.grasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.pregrasp_pub = rospy.Publisher(self.pregrasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.valid_goals_pub = rospy.Publisher(self.valid_goals_topic, ValidGoals, queue_size=1, latch=True)
        self.detections_pub = rospy.Publisher(self.detections_topic, Detection2DArray, queue_size=1, latch=True)

        self.sam_model = None
        self.sam_mask_generator = None
        self.scan_sam_mask_generator = None
        self._load_sam_model()

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)
        rospy.Subscriber(self.allowed_ids_topic, Int32MultiArray, self._allowed_ids_cb, queue_size=1)
        rospy.Subscriber(self.phase_topic, String, self._phase_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self._process_timer_cb)
        self._publish_status("waiting_for_image topic={}".format(self.image_topic))

    def _allowed_ids_cb(self, msg):
        parsed = set()
        for value in list(msg.data):
            try:
                parsed.add(int(value))
            except Exception:
                continue
        self.current_allowed_ids = parsed

    def _phase_cb(self, msg):
        self.current_phase = str(msg.data).strip().lower() or "scan_workspace"

    def _load_candidate_metadata(self):
        if not os.path.exists(self.candidate_metadata_yaml):
            rospy.logwarn("[sam_sandwich_grasp_node] candidate metadata not found: %s", self.candidate_metadata_yaml)
            return []
        with open(self.candidate_metadata_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if isinstance(raw.get("candidate_objects"), dict):
            entries = raw.get("candidate_objects", {}) or {}
        else:
            entries = raw.get("tag_objects", {}) or {}
        parsed = []
        for key, meta in entries.items():
            if not isinstance(meta, dict):
                continue
            try:
                candidate_id = int(key)
            except Exception:
                continue
            parsed.append(
                {
                    "candidate_id": candidate_id,
                    "object_name": str(meta.get("object_name", "")).strip(),
                    "category": str(meta.get("category", "")).strip(),
                    "grasp_complete_label": str(meta.get("grasp_complete_label", "")).strip(),
                    "nominal_radius_m": float(meta.get("nominal_radius_m", 0.0) or 0.0),
                    "nominal_thickness_m": float(meta.get("nominal_thickness_m", 0.0) or 0.0),
                    "expected_hsv": _parse_hsv_triplet(meta.get("expected_hsv")),
                    "staging_zone_center_xy": self._parse_xy(meta.get("staging_zone_center_xy")),
                    "staging_zone_half_extents_xy": self._parse_xy(meta.get("staging_zone_half_extents_xy")),
                    "staging_zone_radius_m": float(meta.get("staging_zone_radius_m", 0.0) or 0.0),
                }
            )
        parsed.sort(key=lambda item: item["candidate_id"])
        return parsed

    @staticmethod
    def _parse_xy(value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return np.array([float(value[0]), float(value[1])], dtype=np.float64)
            except Exception:
                return None
        return None

    def _load_sam_model(self):
        if not self.sam_model_path:
            self._publish_status("sam_model_missing set ~sam_model_path")
            return
        if not os.path.exists(self.sam_model_path):
            self._publish_status("sam_model_missing path={}".format(self.sam_model_path))
            return
        try:
            self.sam_model = None
            self.sam_mask_generator = None
            self.scan_sam_mask_generator = None
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path)
            sam.to(device=self.sam_device)
            sam.eval()
            self.sam_model = sam
            self.sam_mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=self.sam_points_per_side,
                pred_iou_thresh=self.sam_pred_iou_thresh,
                stability_score_thresh=self.sam_stability_score_thresh,
                crop_n_layers=self.sam_crop_n_layers,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=int(self.min_mask_area_px),
            )
            self.scan_sam_mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=max(self.sam_points_per_side, self.scan_sam_points_per_side),
                pred_iou_thresh=self.sam_pred_iou_thresh,
                stability_score_thresh=self.sam_stability_score_thresh,
                crop_n_layers=max(self.sam_crop_n_layers, self.scan_sam_crop_n_layers),
                crop_n_points_downscale_factor=2,
                min_mask_region_area=int(min(self.min_mask_area_px, self.scan_min_mask_area_px)),
            )
            rospy.loginfo("[sam_sandwich_grasp_node] SAM model loaded from %s on %s", self.sam_model_path, self.sam_device)
        except Exception as exc:
            self._publish_status("sam_model_load_failed {}".format(exc))

    def _handle_cuda_oom(self, exc):
        if not _is_cuda_oom(exc):
            return False
        if self.sam_device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if self._oom_recovery_attempted:
            return False
        self._oom_recovery_attempted = True

        retry_image_max_side = max(128, int(self.oom_retry_image_max_side))
        retry_points_per_side = max(4, int(self.oom_retry_points_per_side))
        lowered = False

        if self.sam_image_max_side > retry_image_max_side:
            self.sam_image_max_side = retry_image_max_side
            lowered = True
        if self.sam_points_per_side > retry_points_per_side:
            self.sam_points_per_side = retry_points_per_side
            lowered = True

        if lowered:
            self._publish_status(
                "sam_cuda_oom retrying lower_memory_mode device={} max_side={} points_per_side={}".format(
                    self.sam_device,
                    self.sam_image_max_side,
                    self.sam_points_per_side,
                )
            )
            self._load_sam_model()
            return self.sam_mask_generator is not None

        fallback_device = self.oom_fallback_device.strip()
        if fallback_device and fallback_device != self.sam_device:
            self.sam_device = fallback_device
            self._publish_status("sam_cuda_oom switching_device={}".format(self.sam_device))
            self._load_sam_model()
            return self.sam_mask_generator is not None
        return False

    def camera_info_cb(self, msg):
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
            if msg.header.frame_id:
                self.camera_frame = str(msg.header.frame_id).strip()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[sam_sandwich_grasp_node] image decode failed: %s", exc)

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
            rospy.logwarn_throttle(2.0, "[sam_sandwich_grasp_node] image processing failed: %s", exc)
        finally:
            self.processing_lock.release()

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[sam_sandwich_grasp_node] %s", text)
        self.status_pub.publish(String(data=text))

    def _downscale(self, rgb):
        h, w = rgb.shape[:2]
        max_side = max(h, w)
        if self.sam_image_max_side <= 0 or max_side <= self.sam_image_max_side:
            return rgb, 1.0
        scale = float(self.sam_image_max_side) / float(max_side)
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

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
        H_img_to_be = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)
        return {
            "H_img_to_be": H_img_to_be,
            "width_px": width_px,
            "height_px": height_px,
            "x_min": x_min,
            "y_max": y_max,
            "resolution": resolution,
        }

    def _birdseye_pixel_to_table_xy(self, center_px, birdseye):
        x = float(birdseye["x_min"]) + float(center_px[0]) * float(birdseye["resolution"])
        y = float(birdseye["y_max"]) - float(center_px[1]) * float(birdseye["resolution"])
        return np.array([x, y], dtype=np.float64)

    def _process_image(self, stamp):
        image = self.latest_image
        if image is None or not self.metadata_entries:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks_dicts = None
        scale = 1.0
        scan_mode = self.current_phase == "scan_workspace"
        generator = self.scan_sam_mask_generator if scan_mode and self.scan_sam_mask_generator is not None else self.sam_mask_generator
        for attempt in range(2):
            try:
                rgb_small, scale = self._downscale(rgb)
                masks_dicts = generator.generate(rgb_small)
                self._oom_recovery_attempted = False
                break
            except Exception as exc:
                if attempt == 0 and self._handle_cuda_oom(exc):
                    continue
                self._publish_status("sam_predict_failed {}".format(exc))
                return
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        T_base_cam = self._lookup_base_to_camera(stamp)
        if T_base_cam is None:
            self._publish_status("tf_lookup_failed {}->{}".format(self.base_frame, self.camera_frame))
            return
        birdseye = self._compute_birdseye_geometry(T_base_cam)
        if birdseye is None:
            self._publish_status("birdseye_geometry_failed")
            return
        observed = self._extract_observed_candidates(image, hsv_image, masks_dicts, scale, birdseye)
        if not observed:
            self._publish_empty_outputs()
            self._publish_status("no_sandwich_candidates")
            return
        assigned = self._assign_metadata(observed, stamp)
        if not assigned:
            self._publish_empty_outputs()
            self._publish_status("no_metadata_matched_candidates")
            return
        selected = self._select_primary_candidate(assigned, image.shape[1], image.shape[0])
        self.markers_pub.publish(self._make_markers(assigned, selected, stamp))
        self._publish_debug_image(image, assigned)
        self._maybe_dump_debug_snapshot(image, assigned)
        self._publish_valid_goals(assigned, stamp)
        self._publish_detection_candidates(assigned, stamp)
        self._publish_candidate_pose_topics(assigned)
        if selected is not None:
            self.grasp_pub.publish(selected["grasp_pose"])
            self.pregrasp_pub.publish(selected["pregrasp_pose"])
        self._publish_status(
            "tracking_sandwich_candidates observed={} assigned={} selected={}".format(
                len(observed),
                len(assigned),
                int(selected["candidate_id"]) if selected is not None else -1,
            )
        )

    def _extract_observed_candidates(self, image, hsv_image, masks_dicts, scale, birdseye):
        candidates = []
        relaxed_candidates = []
        if masks_dicts is None:
            return candidates
        scan_mode = self.current_phase == "scan_workspace"
        image_h, image_w = image.shape[:2]
        image_area = float(image_h * image_w)
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
            area = float(cv2.contourArea(contour))
            if area > self.max_mask_area_px:
                continue
            if image_area > 1.0 and (area / image_area) > self.max_mask_image_fraction:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            margin = max(0, int(self.scan_border_reject_margin_px if scan_mode else self.border_reject_margin_px))
            if (
                x <= margin
                or y <= margin
                or (x + bw) >= (image_w - margin)
                or (y + bh) >= (image_h - margin)
            ):
                continue
            rect = cv2.minAreaRect(contour)
            (_, _), (w, h), _ = rect
            if min(w, h) < 2.0:
                continue
            aspect = max(w, h) / max(min(w, h), 1e-6)
            (_, _), radius_px = cv2.minEnclosingCircle(contour)
            circle_area = math.pi * float(radius_px) * float(radius_px)
            fill_ratio = area / max(circle_area, 1e-6)
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
            birdseye_mask = cv2.warpPerspective(
                mask_u8 * 255,
                birdseye["H_img_to_be"],
                (birdseye["width_px"], birdseye["height_px"]),
                flags=cv2.INTER_NEAREST,
            )
            birdseye_contours, _ = cv2.findContours(birdseye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not birdseye_contours:
                continue
            be_contour = max(birdseye_contours, key=cv2.contourArea)
            be_area = float(cv2.contourArea(be_contour))
            if be_area < 10.0:
                continue
            (be_cx, be_cy), be_radius_px = cv2.minEnclosingCircle(be_contour)
            radius_m = float(be_radius_px) * float(birdseye["resolution"])
            min_radius_m = self.scan_min_candidate_radius_m if scan_mode else self.min_candidate_radius_m
            max_radius_m = self.scan_max_candidate_radius_m if scan_mode else self.max_candidate_radius_m
            if radius_m < min_radius_m or radius_m > max_radius_m:
                continue
            center_xy = self._birdseye_pixel_to_table_xy([be_cx, be_cy], birdseye)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = float((4.0 * math.pi * area) / max(perimeter * perimeter, 1e-6)) if perimeter >= 1e-6 else 0.0
            candidate = {
                "contour": contour,
                "mask": mask_u8.astype(bool),
                "center_uv": center_uv,
                "center_xy": center_xy,
                "radius_m": radius_m,
                "area": area,
                "birdseye_area": be_area,
                "circularity": circularity,
                "pred_iou": float(md.get("predicted_iou", 0.0)),
                "aspect_ratio": float(aspect),
                "fill_ratio": float(fill_ratio),
                "median_hsv": self._masked_median_hsv(hsv_image, mask_u8.astype(bool)),
                "color_features": self._masked_color_features(hsv_image, mask_u8.astype(bool)),
            }
            min_area_px = self.scan_min_mask_area_px if scan_mode else self.min_mask_area_px
            max_aspect_ratio = self.scan_max_aspect_ratio if scan_mode else self.max_aspect_ratio
            min_circularity = self.scan_min_circularity if scan_mode else self.min_circularity
            min_fill_ratio = self.scan_min_fill_ratio if scan_mode else self.min_fill_ratio
            if (
                contour.shape[0] >= 5
                and area >= min_area_px
                and aspect <= max_aspect_ratio
                and circularity >= min_circularity
                and fill_ratio >= min_fill_ratio
            ):
                candidates.append(candidate)
                continue
            if (
                self.enable_relaxed_single_candidate_fallback
                and area >= self.relaxed_min_mask_area_px
                and aspect <= self.relaxed_max_aspect_ratio
                and fill_ratio >= self.relaxed_min_fill_ratio
            ):
                relaxed_candidates.append(candidate)
        candidates.sort(key=lambda item: (float(item["birdseye_area"]), float(item["pred_iou"])), reverse=True)
        candidates = self._suppress_contained_candidates(candidates)
        if not scan_mode:
            candidates = self._filter_relative_area_candidates(candidates)
        if candidates or not self.enable_relaxed_single_candidate_fallback:
            return candidates[: (self.scan_max_candidates if scan_mode else self.max_candidates)]
        relaxed_candidates.sort(key=lambda item: (float(item["birdseye_area"]), float(item["pred_iou"])), reverse=True)
        relaxed_candidates = self._suppress_contained_candidates(relaxed_candidates)
        if not scan_mode:
            relaxed_candidates = self._filter_relative_area_candidates(relaxed_candidates)
        if not relaxed_candidates:
            return []
        self._publish_status(
            "relaxed_candidate_fallback observed={} area_px={:.1f} fill={:.2f} circ={:.2f}".format(
                len(relaxed_candidates),
                float(relaxed_candidates[0].get("area", 0.0)),
                float(relaxed_candidates[0].get("fill_ratio", 0.0)),
                float(relaxed_candidates[0].get("circularity", 0.0)),
            )
        )
        return relaxed_candidates[: (self.scan_max_candidates if scan_mode else 1)]

    def _suppress_contained_candidates(self, candidates):
        kept = []
        for candidate in candidates:
            suppressed = False
            area_small = max(float(candidate["area"]), 1.0)
            for other in kept:
                overlap = float(np.logical_and(candidate["mask"], other["mask"]).sum()) / area_small
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

    @staticmethod
    def _masked_median_hsv(hsv_image, mask):
        try:
            pixels = hsv_image[mask]
        except Exception:
            return None
        if pixels is None or len(pixels) == 0:
            return None
        return np.median(pixels.astype(np.float64), axis=0)

    @staticmethod
    def _masked_color_features(hsv_image, mask):
        try:
            pixels = hsv_image[mask]
        except Exception:
            return {}
        if pixels is None or len(pixels) == 0:
            return {}
        pixels = pixels.astype(np.float64)
        return {
            "green_ratio": _ratio_in_hsv_range(pixels, hue_ranges=[(35, 95)], sat_min=70, val_min=40),
            "red_ratio": _ratio_in_hsv_range(pixels, hue_ranges=[(0, 12), (168, 179)], sat_min=90, val_min=70),
            "yellow_ratio": _ratio_in_hsv_range(pixels, hue_ranges=[(15, 38)], sat_min=80, val_min=90),
            "purple_ratio": _ratio_in_hsv_range(pixels, hue_ranges=[(125, 175)], sat_min=45, val_min=80),
            "brown_ratio": _ratio_in_hsv_range(pixels, hue_ranges=[(5, 28)], sat_min=40, sat_max=190, val_min=60, val_max=190),
            "white_ratio": _ratio_in_hsv_range(pixels, sat_max=55, val_min=150),
            "dark_ratio": _ratio_in_hsv_range(pixels, val_max=105),
            "mean_sat": float(np.mean(pixels[:, 1])),
            "mean_val": float(np.mean(pixels[:, 2])),
        }

    @staticmethod
    def _heuristic_identity_penalty(candidate, meta):
        name = str(meta.get("object_name", "")).strip().lower()
        feat = dict(candidate.get("color_features") or {})
        if not feat:
            return 0.0
        green_ratio = float(feat.get("green_ratio", 0.0))
        red_ratio = float(feat.get("red_ratio", 0.0))
        yellow_ratio = float(feat.get("yellow_ratio", 0.0))
        purple_ratio = float(feat.get("purple_ratio", 0.0))
        brown_ratio = float(feat.get("brown_ratio", 0.0))
        white_ratio = float(feat.get("white_ratio", 0.0))
        dark_ratio = float(feat.get("dark_ratio", 0.0))
        mean_sat = float(feat.get("mean_sat", 0.0))
        mean_val = float(feat.get("mean_val", 0.0))

        if name == "lettuce":
            return 2.0 * max(0.0, 0.22 - green_ratio) + 0.5 * max(0.0, 90.0 - mean_sat) / 255.0
        if name == "tomato":
            return 2.0 * max(0.0, 0.18 - red_ratio) + 0.4 * max(0.0, 110.0 - mean_sat) / 255.0
        if name == "cheese":
            return 1.8 * max(0.0, 0.16 - yellow_ratio) + 0.3 * max(0.0, 120.0 - mean_val) / 255.0
        if name == "beef_patty":
            return 2.0 * max(0.0, 0.18 - dark_ratio) + 0.8 * max(0.0, mean_val - 135.0) / 255.0
        if name == "onion":
            onion_signal = max(purple_ratio, 0.75 * white_ratio)
            return 2.0 * max(0.0, 0.12 - onion_signal) + 0.4 * max(0.0, 120.0 - mean_val) / 255.0
        if name.startswith("bread"):
            bread_signal = max(brown_ratio, 0.60 * white_ratio)
            non_bread_signal = max(green_ratio, red_ratio, yellow_ratio, purple_ratio, dark_ratio)
            return 1.7 * max(0.0, 0.14 - bread_signal) + 0.9 * non_bread_signal
        return 0.0

    def _candidate_match_cost(self, candidate, meta, include_zone_distance=False):
        radius_err = abs(float(candidate.get("radius_m", 0.0)) - float(meta.get("nominal_radius_m", 0.0)))
        color_err = _hsv_distance(candidate.get("median_hsv"), meta.get("expected_hsv"))
        heuristic_penalty = self._heuristic_identity_penalty(candidate, meta)
        cost = radius_err + self.color_match_weight * color_err + heuristic_penalty
        if include_zone_distance and meta.get("staging_zone_center_xy") is not None:
            zone_center = np.array(meta.get("staging_zone_center_xy"), dtype=np.float64)
            zone_dist = float(np.linalg.norm(candidate["center_xy"] - zone_center))
            cost += zone_dist
        return (
            round(cost, 6),
            round(radius_err, 6),
            round(color_err, 6),
            round(heuristic_penalty, 6),
            round(float(candidate["center_uv"][1]), 3),
            round(float(candidate["center_uv"][0]), 3),
        )

    def _assign_metadata(self, observed, stamp):
        limit = self.scan_max_candidates if self.current_phase == "scan_workspace" else self.max_candidates
        assigned = self._assign_single_allowed_candidate(observed, stamp)
        if assigned:
            return assigned[:limit]
        if any(meta.get("staging_zone_center_xy") is not None for meta in self.metadata_entries):
            assigned = self._assign_by_zone(observed, stamp)
            if assigned:
                return assigned[:limit]
        assigned = []
        unused_indices = set(range(len(observed)))
        for meta in self.metadata_entries:
            nominal_radius = float(meta.get("nominal_radius_m", 0.0))
            best_idx = None
            best_cost = None
            for idx in sorted(unused_indices):
                candidate = observed[idx]
                radius_err = abs(float(candidate["radius_m"]) - nominal_radius)
                if radius_err > self.radius_match_tolerance_m:
                    continue
                cost = self._candidate_match_cost(candidate, meta, include_zone_distance=False)
                if best_cost is None or cost < best_cost:
                    best_idx = idx
                    best_cost = cost
            if best_idx is None:
                continue
            unused_indices.remove(best_idx)
            solution = self._build_solution(observed[best_idx], meta, stamp)
            if solution is not None:
                assigned.append(solution)
        assigned.sort(key=lambda item: item["candidate_id"])
        return assigned[:limit]

    def _assign_single_allowed_candidate(self, observed, stamp):
        if len(self.current_allowed_ids) != 1 or len(observed) != 1:
            return []
        allowed_id = next(iter(self.current_allowed_ids))
        meta = None
        for entry in self.metadata_entries:
            if int(entry.get("candidate_id", -1)) == int(allowed_id):
                meta = entry
                break
        if meta is None:
            return []
        solution = self._build_solution(observed[0], meta, stamp)
        if solution is None:
            return []
        self._publish_status(
            "single_allowed_candidate_fallback id={} object_name={}".format(
                int(allowed_id),
                str(meta.get("object_name", "")),
            )
        )
        return [solution]

    def _assign_by_zone(self, observed, stamp):
        assigned = []
        unused_indices = set(range(len(observed)))
        for meta in self.metadata_entries:
            zone_center = meta.get("staging_zone_center_xy")
            if zone_center is None:
                continue
            best_idx = None
            best_cost = None
            for idx in sorted(unused_indices):
                candidate = observed[idx]
                if not self._candidate_in_zone(candidate, meta):
                    continue
                area_bias = -float(candidate.get("birdseye_area", 0.0))
                match_cost = self._candidate_match_cost(candidate, meta, include_zone_distance=True)
                cost = match_cost + (round(area_bias, 3),)
                if best_cost is None or cost < best_cost:
                    best_idx = idx
                    best_cost = cost
            if best_idx is None:
                continue
            unused_indices.remove(best_idx)
            solution = self._build_solution(observed[best_idx], meta, stamp)
            if solution is not None:
                assigned.append(solution)
        assigned.sort(key=lambda item: item["candidate_id"])
        return assigned

    def _candidate_in_zone(self, candidate, meta):
        center_xy = candidate.get("center_xy")
        zone_center = meta.get("staging_zone_center_xy")
        if center_xy is None or zone_center is None:
            return False
        delta = np.array(center_xy, dtype=np.float64) - np.array(zone_center, dtype=np.float64)
        half_extents = meta.get("staging_zone_half_extents_xy")
        if half_extents is not None:
            if abs(float(delta[0])) <= float(half_extents[0]) and abs(float(delta[1])) <= float(half_extents[1]):
                return True
        zone_radius = float(meta.get("staging_zone_radius_m", 0.0) or 0.0)
        if zone_radius > 0.0:
            return float(np.linalg.norm(delta)) <= zone_radius
        return float(np.linalg.norm(delta)) <= self.zone_match_tolerance_m

    def _build_solution(self, observed, meta, stamp):
        nominal_thickness = float(meta.get("nominal_thickness_m", 0.0))
        center_base = np.array(
            [
                float(observed["center_xy"][0]) + float(self.grasp_xy_offset_m[0]),
                float(observed["center_xy"][1]) + float(self.grasp_xy_offset_m[1]),
                float(self.table_z + 0.5 * nominal_thickness),
            ],
            dtype=np.float64,
        )
        rotation = _compute_axis_alignment_rotation(
            forward_w=-self.world_up_axis,
            up_ref=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            ee_forward_axis=self.ee_forward_axis,
            ee_up_axis=self.ee_up_axis,
        )
        grasp_pos = center_base + self.world_up_axis * self.grasp_height_offset_m
        pregrasp_pos = grasp_pos + self.world_up_axis * self.pregrasp_offset_m
        return {
            "candidate_id": int(meta["candidate_id"]),
            "object_name": str(meta.get("object_name", "")).strip(),
            "grasp_complete_label": str(meta.get("grasp_complete_label", "")).strip(),
            "category": str(meta.get("category", "")).strip(),
            "nominal_radius_m": float(meta.get("nominal_radius_m", 0.0)),
            "observed_radius_m": float(observed.get("radius_m", 0.0)),
            "median_hsv": observed.get("median_hsv"),
            "color_features": dict(observed.get("color_features") or {}),
            "center_uv": observed["center_uv"],
            "centroid": center_base,
            "contour": observed["contour"],
            "mask": observed.get("mask"),
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

    def _publish_empty_outputs(self):
        self.markers_pub.publish(self._make_empty_markers())
        self.detections_pub.publish(Detection2DArray())
        self.valid_goals_pub.publish(ValidGoals())

    def _make_empty_markers(self):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        return markers

    def _make_markers(self, solutions, selected, stamp):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        for idx, solution in enumerate(solutions):
            centroid = solution["centroid"]
            is_selected = solution is selected
            alpha = 0.95 if is_selected else 0.45
            base_id = idx * 100

            sphere = Marker()
            sphere.header.frame_id = self.base_frame
            sphere.header.stamp = stamp
            sphere.ns = "sam_sandwich_candidates"
            sphere.id = base_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = _make_point(centroid)
            sphere.pose.orientation.w = 1.0
            diameter = max(0.01, 2.0 * float(solution["observed_radius_m"]))
            sphere.scale.x = diameter
            sphere.scale.y = diameter
            sphere.scale.z = 0.01
            sphere.color = ColorRGBA(0.95, 0.65, 0.10, alpha)
            markers.markers.append(sphere)

            arrow = Marker()
            arrow.header.frame_id = self.base_frame
            arrow.header.stamp = stamp
            arrow.ns = "sam_sandwich_grasp_pose"
            arrow.id = base_id + 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose = solution["grasp_pose"].pose
            arrow.scale.x = 0.08 if is_selected else 0.06
            arrow.scale.y = 0.012 if is_selected else 0.009
            arrow.scale.z = 0.012 if is_selected else 0.009
            arrow.color = ColorRGBA(0.2, 1.0, 0.8, alpha)
            markers.markers.append(arrow)

            label = Marker()
            label.header.frame_id = self.base_frame
            label.header.stamp = stamp
            label.ns = "sam_sandwich_labels"
            label.id = base_id + 2
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = _make_point(centroid + np.array([0.0, 0.0, 0.03], dtype=np.float64))
            label.pose.orientation.w = 1.0
            label.scale.z = 0.02
            label.color = ColorRGBA(1.0, 1.0, 1.0, alpha)
            label.text = "{} ({})".format(solution["object_name"], int(solution["candidate_id"]))
            markers.markers.append(label)
        return markers

    def _publish_debug_image(self, image, solutions):
        if self.debug_image_pub.get_num_connections() <= 0:
            return
        vis = image.copy()
        if self.debug_show_mask_fill:
            overlay = vis.copy()
            palette = [
                (48, 180, 255),
                (90, 220, 90),
                (255, 170, 60),
                (190, 120, 255),
                (80, 220, 220),
                (220, 120, 120),
            ]
            for idx, solution in enumerate(solutions):
                mask = solution.get("mask")
                if mask is None:
                    continue
                color = palette[idx % len(palette)]
                overlay[mask.astype(bool)] = color
            vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0.0)
        for solution in solutions:
            contour = solution["contour"].astype(np.int32)
            center_uv = solution["center_uv"]
            feats = dict(solution.get("color_features") or {})
            green_ratio = float(feats.get("green_ratio", 0.0))
            red_ratio = float(feats.get("red_ratio", 0.0))
            yellow_ratio = float(feats.get("yellow_ratio", 0.0))
            purple_ratio = float(feats.get("purple_ratio", 0.0))
            brown_ratio = float(feats.get("brown_ratio", 0.0))
            white_ratio = float(feats.get("white_ratio", 0.0))
            dark_ratio = float(feats.get("dark_ratio", 0.0))
            cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)
            cv2.circle(vis, (int(center_uv[0]), int(center_uv[1])), 5, (0, 255, 0), -1)
            cv2.putText(
                vis,
                "{}:{}".format(solution["object_name"], int(solution["candidate_id"])),
                (int(center_uv[0]) + 6, int(center_uv[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "g{:.2f} r{:.2f} y{:.2f} p{:.2f} b{:.2f} w{:.2f} d{:.2f}".format(
                    green_ratio,
                    red_ratio,
                    yellow_ratio,
                    purple_ratio,
                    brown_ratio,
                    white_ratio,
                    dark_ratio,
                ),
                (int(center_uv[0]) + 6, int(center_uv[1]) + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))
        except Exception:
            pass

    def _maybe_dump_debug_snapshot(self, image, solutions):
        dump_dir = self.debug_dump_dir.strip()
        if not dump_dir or image is None or not solutions:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
        except Exception:
            return
        signature = "|".join(
            "{}:{}:{:.3f}:{:.3f}".format(
                int(solution.get("candidate_id", -1)),
                str(solution.get("object_name", "")),
                float(solution.get("center_uv", [0.0, 0.0])[0]),
                float(solution.get("center_uv", [0.0, 0.0])[1]),
            )
            for solution in solutions
        )
        if signature == self._last_debug_dump_signature:
            return
        self._last_debug_dump_signature = signature
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(dump_dir, "scan_{}.png".format(stamp))
        json_path = os.path.join(dump_dir, "scan_{}.json".format(stamp))
        payload = {"candidates": []}
        for solution in solutions:
            payload["candidates"].append(
                {
                    "candidate_id": int(solution.get("candidate_id", -1)),
                    "object_name": str(solution.get("object_name", "")),
                    "center_uv": [float(solution["center_uv"][0]), float(solution["center_uv"][1])],
                    "observed_radius_m": float(solution.get("observed_radius_m", 0.0)),
                    "median_hsv": None if solution.get("median_hsv") is None else [float(v) for v in solution["median_hsv"]],
                    "color_features": dict(solution.get("color_features") or {}),
                }
            )
        try:
            cv2.imwrite(image_path, image)
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except Exception:
            return

    def _publish_valid_goals(self, solutions, stamp):
        msg = ValidGoals()
        msg.header.stamp = stamp
        msg.current_state = "sandwich_pick"
        for solution in solutions:
            goal = ValidGoal()
            goal.goal_id = "sandwich_{}".format(int(solution["candidate_id"]))
            goal.action_type = "pick"
            goal.target_position = _make_point(
                [
                    solution["grasp_pose"].pose.position.x,
                    solution["grasp_pose"].pose.position.y,
                    solution["grasp_pose"].pose.position.z,
                ]
            )
            goal.object_name = solution["object_name"]
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

    def _candidate_publishers(self, candidate_id):
        if candidate_id not in self.candidate_pose_pubs:
            ns = "{}{}".format(self.candidate_namespace_prefix, int(candidate_id))
            self.candidate_pose_pubs[candidate_id] = {
                "pregrasp": rospy.Publisher("{}/pregrasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
                "grasp": rospy.Publisher("{}/grasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
            }
        return self.candidate_pose_pubs[candidate_id]

    def _publish_candidate_pose_topics(self, solutions):
        for solution in solutions:
            pubs = self._candidate_publishers(solution["candidate_id"])
            pubs["pregrasp"].publish(solution["pregrasp_pose"])
            pubs["grasp"].publish(solution["grasp_pose"])


if __name__ == "__main__":
    SamSandwichGraspNode()
    rospy.spin()
