#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual point-prompt SAM labeler for sandwich objects."""

import copy
import json
import math
import os
from contextlib import nullcontext

import cv2
import numpy as np
import rospy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

import torch
from segment_anything import SamPredictor, sam_model_registry


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
        values = [float(v) for v in raw.strip().replace("[", "").replace("]", "").replace(",", " ").split() if v]
    else:
        values = [float(raw)]
    if expected_len is not None and len(values) != expected_len:
        return np.array(default, dtype=np.float64)
    return np.array(values, dtype=np.float64)


def _parse_string_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [str(v).strip().lower() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [s.strip().lower() for s in txt.split() if s.strip()]
    return [str(v).strip().lower() for v in default if str(v).strip()]


def _parse_int_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [int(v) for v in txt.split() if v]
    return [int(v) for v in default]


def _quat_normalize_xyzw(q):
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.array([float(v) / norm for v in q], dtype=np.float64)


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
    msg = Point()
    msg.x = float(xyz[0])
    msg.y = float(xyz[1])
    msg.z = float(xyz[2])
    return msg


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


def _tf_to_matrix(tf_msg):
    tx = float(tf_msg.transform.translation.x)
    ty = float(tf_msg.transform.translation.y)
    tz = float(tf_msg.transform.translation.z)
    qx = float(tf_msg.transform.rotation.x)
    qy = float(tf_msg.transform.rotation.y)
    qz = float(tf_msg.transform.rotation.z)
    qw = float(tf_msg.transform.rotation.w)
    x, y, z, w = _quat_normalize_xyzw([qx, qy, qz, qw])
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    mat[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return mat


def _project_base_point_to_image(point_base, T_base_cam, camera_matrix):
    point_base_h = np.array([point_base[0], point_base[1], point_base[2], 1.0], dtype=np.float64)
    T_cam_base = np.linalg.inv(T_base_cam)
    point_cam = T_cam_base.dot(point_base_h)
    if point_cam[2] <= 1e-6:
        return None
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    u = fx * (point_cam[0] / point_cam[2]) + cx
    v = fy * (point_cam[1] / point_cam[2]) + cy
    return np.array([u, v], dtype=np.float32)


def _compute_axis_alignment_rotation(forward_w, up_ref, ee_forward_axis, ee_up_axis):
    forward = _normalize(forward_w, [0.0, 0.0, -1.0])
    up_seed = _normalize(up_ref, [1.0, 0.0, 0.0])
    side = np.cross(forward, up_seed)
    if float(np.linalg.norm(side)) < 1e-6:
        up_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        side = np.cross(forward, up_seed)
    side = _normalize(side, [0.0, 1.0, 0.0])
    up = _normalize(np.cross(side, forward), [1.0, 0.0, 0.0])
    world_basis = np.column_stack((forward, up, side))

    ef = _normalize(ee_forward_axis, [0.0, 0.0, 1.0])
    eu_seed = _normalize(ee_up_axis, [0.0, 1.0, 0.0])
    es = np.cross(ef, eu_seed)
    if float(np.linalg.norm(es)) < 1e-6:
        eu_seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        es = np.cross(ef, eu_seed)
    es = _normalize(es, [1.0, 0.0, 0.0])
    eu = _normalize(np.cross(es, ef), [0.0, 1.0, 0.0])
    ee_basis = np.column_stack((ef, eu, es))
    return world_basis.dot(np.linalg.inv(ee_basis))


class SamManualLabelerNode:
    def __init__(self):
        rospy.init_node("sam_manual_labeler_node")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.camera_frame = str(rospy.get_param("~camera_frame", "camera_color_optical_frame")).strip()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info")).strip()
        self.command_topic = str(rospy.get_param("~command_topic", "/sam_manual_labeler/command")).strip()
        self.task_command_topic = str(rospy.get_param("~task_command_topic", "/task_context/command")).strip()
        self.state_topic = str(rospy.get_param("~state_topic", "/sam_manual_labeler/state")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "/sam_manual_labeler/status")).strip()
        self.debug_image_topic = str(rospy.get_param("~debug_image_topic", "/sam_manual_labeler/debug_image")).strip()
        self.detections_topic = str(rospy.get_param("~detections_topic", "/sam_manual_labeler/detections")).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "/task_context/phase")).strip()
        self.clear_commands = set(_parse_string_list_param("~clear_commands", ["scan_workspace", "scan", "reset_task", "reset"]))
        self.preserve_labels_on_task_command = bool(rospy.get_param("~preserve_labels_on_task_command", True))
        self.candidate_metadata_yaml = os.path.expanduser(
            str(rospy.get_param("~candidate_metadata_yaml", os.path.join(package_root, "config", "sandwich_candidate_metadata.example.yaml"))).strip()
        )
        self.sam_model_path = str(rospy.get_param("~sam_model_path", "/home/gyanig/catkin_ws/src/graspnet-baseline/sam_vit_b_01ec64.pth")).strip()
        self.sam_model_type = str(rospy.get_param("~sam_model_type", "vit_b")).strip()
        self.sam_device = str(rospy.get_param("~sam_device", "cpu")).strip()
        self.sam_cpu_fallback_on_oom = bool(rospy.get_param("~sam_cpu_fallback_on_oom", True))
        self.sam_image_max_side = int(rospy.get_param("~sam_image_max_side", 640))
        self.sam_retry_max_sides = [
            int(v)
            for v in _parse_int_list_param("~sam_retry_max_sides", [640, 512, 384, 256])
            if int(v) > 0
        ]
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 5.0))
        self.tf_lookup_timeout_sec = float(rospy.get_param("~tf_lookup_timeout_sec", 0.35))
        self.table_z = float(rospy.get_param("~fixed_table_z", -0.255))
        self.grasp_height_offset_m = float(rospy.get_param("~grasp_height_offset_m", 0.135))
        self.pregrasp_offset_m = float(rospy.get_param("~pregrasp_offset_m", 0.240))
        self.grasp_xy_offset_m = _parse_float_list_param("~grasp_xy_offset_m", [0.0, 0.0], 2)
        self.table_roi_min_xy = _parse_float_list_param("~table_roi_min_xy", [0.25, -0.30], 2)
        self.table_roi_max_xy = _parse_float_list_param("~table_roi_max_xy", [0.85, 0.30], 2)
        self.birdseye_resolution_m = float(rospy.get_param("~birdseye_resolution_m", 0.0025))
        self.world_up_axis = _normalize(_parse_float_list_param("~world_up_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_forward_axis = _normalize(_parse_float_list_param("~ee_forward_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_up_axis = _normalize(_parse_float_list_param("~ee_up_axis", [0.0, 1.0, 0.0], 3), [0.0, 1.0, 0.0])
        self.output_namespace_prefix = str(rospy.get_param("~output_namespace_prefix", "/apriltag_candidates/tag_")).strip()
        self.snapshot_yaml = os.path.expanduser(str(rospy.get_param("~snapshot_yaml", "")).strip())
        self.auto_save_yaml = bool(rospy.get_param("~auto_save_yaml", True))
        self.load_from_yaml = bool(rospy.get_param("~load_from_yaml", False))

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.camera_matrix = None
        self.latest_image = None
        self.latest_image_stamp = rospy.Time(0)
        self.current_phase = "scan_workspace"
        self.pending_observed = None
        self.pending_click_uv = None
        self.last_status = ""
        self.predictor = None
        self.active_sam_device = self.sam_device
        self.metadata_by_name, self.metadata_by_id = self._load_candidate_metadata()
        self.labeled_objects = {}
        self.pose_pubs = {}

        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.state_pub = rospy.Publisher(self.state_topic, String, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.detections_pub = rospy.Publisher(self.detections_topic, Detection2DArray, queue_size=1, latch=True)

        self._load_predictor()
        self._load_snapshot_if_requested()

        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=10)
        rospy.Subscriber(self.task_command_topic, String, self._task_command_cb, queue_size=10)
        rospy.Subscriber(self.phase_topic, String, self._phase_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._timer_cb)
        self._publish_status("waiting_for_image topic={}".format(self.image_topic))
        self._publish_state()

    def _load_candidate_metadata(self):
        if not os.path.exists(self.candidate_metadata_yaml):
            rospy.logwarn("[sam_manual_labeler_node] candidate metadata not found: %s", self.candidate_metadata_yaml)
            return {}, {}
        with open(self.candidate_metadata_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        entries = raw.get("candidate_objects", {}) if isinstance(raw.get("candidate_objects"), dict) else raw.get("tag_objects", {})
        by_name = {}
        by_id = {}
        for key, meta in (entries or {}).items():
            if not isinstance(meta, dict):
                continue
            try:
                candidate_id = int(key)
            except Exception:
                continue
            item = dict(meta)
            item["candidate_id"] = candidate_id
            object_name = str(item.get("object_name", "")).strip()
            by_id[candidate_id] = item
            if object_name:
                by_name[object_name] = item
        return by_name, by_id

    def _load_predictor(self, device_override=None):
        if not self.sam_model_path or not os.path.exists(self.sam_model_path):
            self._publish_status("sam_model_missing")
            return
        target_device = str(device_override or self.sam_device).strip()
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path)
            sam.to(device=target_device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            self.active_sam_device = target_device
            rospy.loginfo("[sam_manual_labeler_node] SAM predictor loaded on %s", target_device)
        except Exception as exc:
            self._publish_status("sam_model_load_failed {}".format(exc))

    def _load_snapshot_if_requested(self):
        if not self.load_from_yaml or not self.snapshot_yaml or not os.path.exists(self.snapshot_yaml):
            return
        try:
            with open(self.snapshot_yaml, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except Exception as exc:
            rospy.logwarn("[sam_manual_labeler_node] failed to load snapshot: %s", exc)
            return
        for key, entry in (raw.get("objects", {}) or {}).items():
            try:
                candidate_id = int(key)
            except Exception:
                continue
            meta = self.metadata_by_id.get(candidate_id)
            if meta is None or not isinstance(entry, dict):
                continue
            center_xy = np.array(entry.get("center_xy", [0.0, 0.0]), dtype=np.float64)
            center_uv = np.array(entry.get("center_uv", [0.0, 0.0]), dtype=np.float64)
            observed = {
                "center_xy": center_xy,
                "center_uv": center_uv,
                "radius_m": float(entry.get("observed_radius_m", meta.get("nominal_radius_m", 0.0))),
                "mask": None,
                "contour": np.zeros((0, 1, 2), dtype=np.int32),
            }
            self.labeled_objects[candidate_id] = self._build_solution(observed, meta, rospy.Time.now())

    def _save_snapshot_if_enabled(self):
        if not self.auto_save_yaml or not self.snapshot_yaml:
            return
        data = {"objects": {}}
        for candidate_id, solution in sorted(self.labeled_objects.items()):
            data["objects"][str(candidate_id)] = {
                "object_name": str(solution.get("object_name", "")),
                "center_uv": [float(solution["center_uv"][0]), float(solution["center_uv"][1])],
                "center_xy": [float(solution["center_xy"][0]), float(solution["center_xy"][1])],
                "observed_radius_m": float(solution.get("observed_radius_m", 0.0)),
            }
        parent = os.path.dirname(self.snapshot_yaml)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = self.snapshot_yaml + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=True)
        os.replace(tmp_path, self.snapshot_yaml)

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[sam_manual_labeler_node] %s", text)
        self.status_pub.publish(String(data=text))

    def _phase_cb(self, msg):
        self.current_phase = str(msg.data).strip().lower() or "scan_workspace"

    def _camera_info_cb(self, msg):
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def _image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
            if msg.header.frame_id:
                self.camera_frame = str(msg.header.frame_id).strip()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[sam_manual_labeler_node] image decode failed: %s", exc)

    def _command_cb(self, msg):
        try:
            payload = json.loads(str(msg.data))
        except Exception as exc:
            self._publish_status("bad_command {}".format(exc))
            return
        action = str(payload.get("action", "")).strip().lower()
        if action == "click":
            self._segment_click(payload)
        elif action == "assign":
            self._assign_label(str(payload.get("object_name", "")).strip())
        elif action == "remove":
            self._remove_label(str(payload.get("object_name", "")).strip())
        elif action == "clear":
            self.pending_observed = None
            self.pending_click_uv = None
            self.labeled_objects = {}
            self._save_snapshot_if_enabled()
            self._publish_status("manual_labels_cleared")
        self._publish_state()

    def _task_command_cb(self, msg):
        cmd = str(msg.data).strip().lower()
        if cmd.startswith("remove_tag:"):
            try:
                candidate_id = int(cmd.split(":", 1)[1].strip())
            except Exception:
                self._publish_status("manual_remove_tag_invalid")
                return
            self._remove_label_by_id(candidate_id)
            self._publish_state()
            return
        if cmd not in self.clear_commands:
            return
        self.pending_observed = None
        self.pending_click_uv = None
        if self.preserve_labels_on_task_command:
            self._publish_status("manual_pending_cleared_labels_preserved")
        else:
            self.labeled_objects = {}
            self._save_snapshot_if_enabled()
            self._publish_status("manual_labels_cleared")
        self._publish_state()

    def _lookup_base_to_camera(self, stamp):
        timeout = rospy.Duration(max(0.05, self.tf_lookup_timeout_sec))
        attempts = []
        if stamp not in (None, rospy.Time()):
            attempts.append(("image_stamp", stamp))
        attempts.append(("latest", rospy.Time(0)))
        last_error = ""
        for label, query_stamp in attempts:
            try:
                tf_msg = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, query_stamp, timeout)
                return _tf_to_matrix(tf_msg), label
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
                last_error = "{}: {}".format(label, exc)
        return None, last_error

    def _compute_birdseye_geometry(self, T_base_cam):
        x_min = float(self.table_roi_min_xy[0])
        y_min = float(self.table_roi_min_xy[1])
        x_max = float(self.table_roi_max_xy[0])
        y_max = float(self.table_roi_max_xy[1])
        resolution = max(float(self.birdseye_resolution_m), 1e-4)
        width_px = max(32, int(math.ceil((x_max - x_min) / resolution)))
        height_px = max(32, int(math.ceil((y_max - y_min) / resolution)))
        src_points = []
        dst_points = np.array([[0.0, 0.0], [float(width_px - 1), 0.0], [float(width_px - 1), float(height_px - 1)], [0.0, float(height_px - 1)]], dtype=np.float32)
        corners_base = [[x_min, y_max, self.table_z], [x_max, y_max, self.table_z], [x_max, y_min, self.table_z], [x_min, y_min, self.table_z]]
        for corner in corners_base:
            uv = _project_base_point_to_image(corner, T_base_cam, self.camera_matrix)
            if uv is None:
                return None
            src_points.append(uv)
        H_img_to_be = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)
        return {"H_img_to_be": H_img_to_be, "width_px": width_px, "height_px": height_px, "x_min": x_min, "y_max": y_max, "resolution": resolution}

    def _birdseye_pixel_to_table_xy(self, center_px, birdseye):
        x = float(birdseye["x_min"]) + float(center_px[0]) * float(birdseye["resolution"])
        y = float(birdseye["y_max"]) - float(center_px[1]) * float(birdseye["resolution"])
        return np.array([x, y], dtype=np.float64)

    def _downscale(self, rgb):
        h, w = rgb.shape[:2]
        max_side = max(h, w)
        if self.sam_image_max_side <= 0 or max_side <= self.sam_image_max_side:
            return rgb, 1.0
        scale = float(self.sam_image_max_side) / float(max_side)
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    def _downscale_to_limit(self, rgb, max_side_limit):
        h, w = rgb.shape[:2]
        max_side = max(h, w)
        if max_side_limit <= 0 or max_side <= max_side_limit:
            return rgb, 1.0
        scale = float(max_side_limit) / float(max_side)
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    def _is_cuda_oom(self, exc):
        text = str(exc).lower()
        return "out of memory" in text and "cuda" in text

    def _maybe_cuda_empty_cache(self):
        if self.active_sam_device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _sam_inference_context(self):
        if self.active_sam_device.startswith("cuda") and torch.cuda.is_available():
            try:
                return torch.autocast(device_type="cuda", dtype=torch.float16)
            except Exception:
                return nullcontext()
        return nullcontext()

    def _switch_predictor_to_cpu(self):
        if self.active_sam_device == "cpu":
            return False
        rospy.logwarn("[sam_manual_labeler_node] switching SAM predictor to CPU after repeated CUDA OOM")
        self._maybe_cuda_empty_cache()
        self.predictor = None
        self._load_predictor(device_override="cpu")
        return self.predictor is not None and self.active_sam_device == "cpu"

    def _predict_with_retries(self, rgb, u, v):
        retry_limits = []
        seen = set()
        for value in [self.sam_image_max_side] + list(self.sam_retry_max_sides):
            try:
                limit = int(value)
            except Exception:
                continue
            if limit <= 0 or limit in seen:
                continue
            seen.add(limit)
            retry_limits.append(limit)
        if not retry_limits:
            retry_limits = [640, 512, 384, 256]

        last_exc = None
        for limit in retry_limits:
            rgb_small, scale = self._downscale_to_limit(rgb, limit)
            point = np.array([[u * scale, v * scale]], dtype=np.float32)
            try:
                with torch.inference_mode():
                    with self._sam_inference_context():
                        self.predictor.set_image(rgb_small)
                        masks, scores, _ = self.predictor.predict(
                            point_coords=point,
                            point_labels=np.array([1], dtype=np.int32),
                            multimask_output=True,
                        )
                self._maybe_cuda_empty_cache()
                if limit != retry_limits[0]:
                    rospy.logwarn(
                        "[sam_manual_labeler_node] recovered from SAM OOM by retrying with max_side=%d",
                        limit,
                    )
                return masks, scores, scale
            except RuntimeError as exc:
                last_exc = exc
                if not self._is_cuda_oom(exc):
                    raise
                rospy.logwarn(
                    "[sam_manual_labeler_node] SAM CUDA OOM at max_side=%d, retrying smaller resolution",
                    limit,
                )
                self._maybe_cuda_empty_cache()
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("sam_predict_failed_no_retry_limit")

    def _segment_click(self, payload):
        if self.predictor is None or self.latest_image is None or self.camera_matrix is None:
            self._publish_status("manual_click_rejected_not_ready")
            return
        try:
            u = float(payload.get("u"))
            v = float(payload.get("v"))
        except Exception:
            self._publish_status("manual_click_invalid_uv")
            return
        image = self.latest_image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            masks, scores, scale = self._predict_with_retries(rgb, u, v)
        except Exception as exc:
            if self._is_cuda_oom(exc) and self.sam_cpu_fallback_on_oom and self._switch_predictor_to_cpu():
                try:
                    masks, scores, scale = self._predict_with_retries(rgb, u, v)
                except Exception as retry_exc:
                    self._publish_status("manual_segment_failed {}".format(retry_exc))
                    return
            else:
                self._publish_status("manual_segment_failed {}".format(exc))
                return
        if masks is None or len(masks) == 0:
            self._publish_status("manual_segment_no_mask")
            return
        best_idx = int(np.argmax(scores))
        mask_u8 = masks[best_idx].astype(np.uint8)
        if scale != 1.0:
            h, w = image.shape[:2]
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._publish_status("manual_segment_no_contour")
            return
        contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask_u8)
        cv2.drawContours(clean_mask, [contour], -1, 1, thickness=-1)
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) > 1e-6:
            center_uv = np.array([float(moments["m10"]) / float(moments["m00"]), float(moments["m01"]) / float(moments["m00"])], dtype=np.float64)
        else:
            rect = cv2.minAreaRect(contour)
            center_uv = np.array(rect[0], dtype=np.float64)
        T_base_cam, tf_info = self._lookup_base_to_camera(self.latest_image_stamp)
        if T_base_cam is None:
            self._publish_status("manual_segment_tf_failed {}".format(tf_info))
            return
        if tf_info != "image_stamp":
            rospy.logwarn_throttle(2.0, "[sam_manual_labeler_node] using %s TF fallback for click segmentation", tf_info)
        birdseye = self._compute_birdseye_geometry(T_base_cam)
        if birdseye is None:
            self._publish_status("manual_segment_birdseye_failed")
            return
        birdseye_mask = cv2.warpPerspective(clean_mask * 255, birdseye["H_img_to_be"], (birdseye["width_px"], birdseye["height_px"]), flags=cv2.INTER_NEAREST)
        be_contours, _ = cv2.findContours(birdseye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not be_contours:
            self._publish_status("manual_segment_be_contour_failed")
            return
        be_contour = max(be_contours, key=cv2.contourArea)
        (be_cx, be_cy), be_radius_px = cv2.minEnclosingCircle(be_contour)
        observed = {
            "mask": clean_mask.astype(bool),
            "contour": contour,
            "center_uv": center_uv,
            "center_xy": self._birdseye_pixel_to_table_xy([be_cx, be_cy], birdseye),
            "radius_m": float(be_radius_px) * float(birdseye["resolution"]),
        }
        self.pending_observed = observed
        self.pending_click_uv = [float(u), float(v)]
        self._publish_status("manual_segment_ready")

    def _assign_label(self, object_name):
        object_name = str(object_name).strip()
        meta = self.metadata_by_name.get(object_name)
        if meta is None or self.pending_observed is None:
            self._publish_status("manual_assign_failed")
            return
        candidate_id = int(meta["candidate_id"])
        self.labeled_objects[candidate_id] = self._build_solution(self.pending_observed, meta, self.latest_image_stamp)
        self.pending_observed = None
        self.pending_click_uv = None
        self._save_snapshot_if_enabled()
        self._publish_status("manual_assigned {}".format(object_name))

    def _remove_label(self, object_name):
        meta = self.metadata_by_name.get(str(object_name).strip())
        if meta is None:
            self._publish_status("manual_remove_failed")
            return
        self._remove_label_by_id(int(meta["candidate_id"]))

    def _remove_label_by_id(self, candidate_id):
        candidate_id = int(candidate_id)
        meta = self.metadata_by_id.get(candidate_id, {})
        object_name = str(meta.get("object_name", candidate_id)).strip()
        if candidate_id in self.labeled_objects:
            del self.labeled_objects[candidate_id]
            self._save_snapshot_if_enabled()
            self._publish_status("manual_removed {}".format(object_name))
        else:
            self._publish_status("manual_remove_ignored {}".format(object_name))

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
        rotation = _compute_axis_alignment_rotation(-self.world_up_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64), self.ee_forward_axis, self.ee_up_axis)
        grasp_pos = center_base + self.world_up_axis * self.grasp_height_offset_m
        pregrasp_pos = grasp_pos + self.world_up_axis * self.pregrasp_offset_m
        return {
            "candidate_id": int(meta["candidate_id"]),
            "object_name": str(meta.get("object_name", "")).strip(),
            "center_uv": np.array(observed["center_uv"], dtype=np.float64),
            "center_xy": np.array(observed["center_xy"], dtype=np.float64),
            "observed_radius_m": float(observed.get("radius_m", 0.0)),
            "mask": observed.get("mask"),
            "contour": observed.get("contour"),
            "pregrasp_pose": _make_pose_stamped(self.base_frame, stamp, pregrasp_pos, rotation),
            "grasp_pose": _make_pose_stamped(self.base_frame, stamp, grasp_pos, rotation),
        }

    def _pose_publishers(self, candidate_id):
        if candidate_id not in self.pose_pubs:
            ns = "{}{}".format(self.output_namespace_prefix, int(candidate_id))
            self.pose_pubs[candidate_id] = {
                "pregrasp": rospy.Publisher("{}/pregrasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
                "grasp": rospy.Publisher("{}/grasp_pose".format(ns), PoseStamped, queue_size=1, latch=True),
            }
        return self.pose_pubs[candidate_id]

    def _publish_detections(self):
        msg = Detection2DArray()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.base_frame
        for candidate_id, solution in sorted(self.labeled_objects.items()):
            det = Detection2D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.id = int(candidate_id)
            hyp.score = 1.0
            hyp.pose.pose = copy.deepcopy(solution["pregrasp_pose"].pose)
            det.results.append(hyp)
            msg.detections.append(det)
            pubs = self._pose_publishers(candidate_id)
            pubs["pregrasp"].publish(solution["pregrasp_pose"])
            pubs["grasp"].publish(solution["grasp_pose"])
        self.detections_pub.publish(msg)

    def _publish_debug_image(self):
        if self.latest_image is None:
            return
        vis = self.latest_image.copy()
        for candidate_id, solution in sorted(self.labeled_objects.items()):
            contour = solution.get("contour")
            if contour is not None and len(contour) > 0:
                cv2.drawContours(vis, [contour.astype(np.int32)], -1, (0, 255, 255), 2)
            center_uv = solution["center_uv"]
            cv2.circle(vis, (int(center_uv[0]), int(center_uv[1])), 6, (0, 200, 0), -1)
            cv2.putText(vis, "{}:{}".format(solution["object_name"], int(candidate_id)), (int(center_uv[0]) + 8, int(center_uv[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        if self.pending_observed is not None:
            mask = self.pending_observed.get("mask")
            if mask is not None:
                overlay = vis.copy()
                overlay[mask.astype(bool)] = (255, 120, 80)
                vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0.0)
            center_uv = self.pending_observed["center_uv"]
            cv2.circle(vis, (int(center_uv[0]), int(center_uv[1])), 6, (255, 120, 80), -1)
            cv2.putText(vis, "pending", (int(center_uv[0]) + 8, int(center_uv[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 80), 2, cv2.LINE_AA)
        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))
        except Exception:
            pass

    def _publish_state(self):
        payload = {
            "enabled": True,
            "current_phase": self.current_phase,
            "status": self.last_status,
            "pending_ready": bool(self.pending_observed is not None),
            "pending_click_uv": self.pending_click_uv,
            "available_labels": sorted(self.metadata_by_name.keys()),
            "labeled_objects": [
                {
                    "candidate_id": int(candidate_id),
                    "object_name": str(solution.get("object_name", "")),
                    "center_uv": [float(solution["center_uv"][0]), float(solution["center_uv"][1])],
                }
                for candidate_id, solution in sorted(self.labeled_objects.items())
            ],
        }
        self.state_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _timer_cb(self, _evt):
        self._publish_detections()
        self._publish_debug_image()
        self._publish_state()


if __name__ == "__main__":
    SamManualLabelerNode()
    rospy.spin()
