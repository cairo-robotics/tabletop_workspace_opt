#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM + RGB-D final approach correction for selected grasp targets."""

import copy
import math
import os
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from intera_core_msgs.msg import EndpointState
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

import torch
from segment_anything import SamPredictor, sam_model_registry


def _parse_int_list_param(name, default):
    raw = rospy.get_param(name, default)
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
        return [int(v) for v in txt.split() if v]
    return [int(v) for v in default]


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


def _parse_csv_set(raw):
    if raw is None:
        return set()
    return {chunk.strip().lower() for chunk in str(raw).split(",") if chunk.strip()}


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


def _tf_to_matrix(tf_msg):
    q = tf_msg.transform.rotation
    t = tf_msg.transform.translation
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_to_matrix([q.x, q.y, q.z, q.w])
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def _pose_position(msg):
    p = msg.pose.position
    return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)


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
    return np.array([u, v], dtype=np.float64)


def _depth_to_meters(depth):
    if depth is None:
        return None
    arr = np.asarray(depth)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) * 0.001
    return arr.astype(np.float32)


class SamFinalApproachNode:
    def __init__(self):
        rospy.init_node("sam_final_approach")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.camera_frame = str(rospy.get_param("~camera_frame", "camera_color_optical_frame")).strip()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.depth_topic = str(rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")).strip()
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info")).strip()
        self.end_effector_topic = str(rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")).strip()
        self.execution_state_topic = str(rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")).strip()
        self.selected_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.observed_namespace_prefix = str(
            rospy.get_param("~observed_namespace_prefix", "/apriltag_candidates/tag_")
        ).strip()
        self.object_map_yaml = os.path.expanduser(
            str(rospy.get_param("~object_map_yaml", os.path.join(package_root, "config", "apriltag_object_map.yaml"))).strip()
        )
        self.tag_ids = _parse_int_list_param("~tag_ids", [0, 1, 2, 3, 4, 5])
        self.allowed_categories = _parse_csv_set(rospy.get_param("~allowed_categories", "milk,cereal,chocolate"))
        self.active_states = _parse_csv_set(
            rospy.get_param("~active_states", "exec_pregrasp,visual_align,wait_pregrasp_confirm,wait_close_a")
        )
        self.approach_mode = str(rospy.get_param("~approach_mode", "tag_local")).strip().lower()

        self.sam_model_path = str(
            rospy.get_param("~sam_model_path", "/home/gyanig/catkin_ws/src/graspnet-baseline/sam_vit_b_01ec64.pth")
        ).strip()
        self.sam_model_type = str(rospy.get_param("~sam_model_type", "vit_b")).strip()
        self.sam_device = self._resolve_sam_device(str(rospy.get_param("~sam_device", "auto")).strip())
        self.sam_cpu_fallback_on_oom = bool(rospy.get_param("~sam_cpu_fallback_on_oom", True))

        self.offset_topic = str(rospy.get_param("~offset_topic", "/visual_grasp_refine/offset")).strip()
        self.ready_topic = str(rospy.get_param("~ready_topic", "/visual_grasp_refine/ready")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "/visual_grasp_refine/status")).strip()
        self.debug_image_topic = str(rospy.get_param("~debug_image_topic", "/visual_grasp_refine/debug_image")).strip()
        self.publish_rate_hz = max(0.5, float(rospy.get_param("~publish_rate_hz", 2.0)))
        self.tf_lookup_timeout_sec = float(rospy.get_param("~tf_lookup_timeout_sec", 0.2))
        self.max_image_age_sec = float(rospy.get_param("~max_image_age_sec", 0.5))
        self.max_tag_age_sec = float(rospy.get_param("~max_tag_age_sec", 1.5))
        self.roi_half_size_px = int(rospy.get_param("~roi_half_size_px", 120))
        self.roi_min_size_px = int(rospy.get_param("~roi_min_size_px", 96))
        self.roi_max_size_px = int(rospy.get_param("~roi_max_size_px", 320))
        self.sam_prompt_mode = str(rospy.get_param("~sam_prompt_mode", "box")).strip().lower()
        self.min_mask_area_px = float(rospy.get_param("~min_mask_area_px", 300.0))
        self.max_mask_area_fraction = float(rospy.get_param("~max_mask_area_fraction", 0.98))
        self.min_depth_points = int(rospy.get_param("~min_depth_points", 80))
        self.local_depth_window_px = int(rospy.get_param("~local_depth_window_px", 21))
        self.local_offset_sign = float(rospy.get_param("~local_offset_sign", 1.0))
        self.local_pixel_deadband_px = float(rospy.get_param("~local_pixel_deadband_px", 8.0))
        self.local_max_pixel_error_px = float(rospy.get_param("~local_max_pixel_error_px", 80.0))
        self.depth_trim_quantile = float(rospy.get_param("~depth_trim_quantile", 0.12))
        self.max_xy_correction_m = float(rospy.get_param("~max_xy_correction_m", 0.035))
        self.max_z_correction_m = float(rospy.get_param("~max_z_correction_m", 0.0))
        self.publish_zero_until_reference = bool(rospy.get_param("~publish_zero_until_reference", True))
        self.category_grasp_center_offset_xy = _parse_float_list_param("~category_grasp_center_offset_xy", [0.0, 0.0], 2)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.lock = threading.Lock()
        self.camera_matrix = None
        self.latest_image = None
        self.latest_image_stamp = rospy.Time(0)
        self.latest_depth = None
        self.latest_depth_stamp = rospy.Time(0)
        self.latest_depth_frame = ""
        self.latest_ee = None
        self.execution_state = ""
        self.selected_label = ""
        self.observed_tags = {}
        self.reference_center = None
        self.reference_uv = None
        self.reference_depth_m = None
        self.reference_label = ""
        self.last_status = ""
        self.predictor = self._load_sam_predictor() if self.approach_mode == "sam_mask_center" else None
        self.label_to_meta = self._load_label_metadata()

        self.offset_pub = rospy.Publisher(self.offset_topic, Vector3Stamped, queue_size=1)
        self.ready_pub = rospy.Publisher(self.ready_topic, Bool, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb, queue_size=1)
        rospy.Subscriber(self.end_effector_topic, EndpointState, self._ee_cb, queue_size=5)
        rospy.Subscriber(self.execution_state_topic, String, self._state_cb, queue_size=1)
        rospy.Subscriber(self.selected_label_topic, String, self._label_cb, queue_size=1)
        for tag_id in self.tag_ids:
            ns = "{}{}".format(self.observed_namespace_prefix, tag_id)
            rospy.Subscriber("{}/base_tag_pose".format(ns), PoseStamped, self._tag_cb, callback_args=tag_id, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / self.publish_rate_hz), self._tick)
        self._publish_status("waiting_for_inputs")
        rospy.loginfo(
            "[sam_final_approach] image=%s depth=%s selected=%s offset=%s",
            self.image_topic,
            self.depth_topic,
            self.selected_label_topic,
            self.offset_topic,
        )

    def _load_sam_predictor(self):
        if not self.sam_model_path or not os.path.exists(self.sam_model_path):
            rospy.logwarn("[sam_final_approach] SAM checkpoint not found: %s", self.sam_model_path)
            return None
        try:
            sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path)
            sam.to(device=self.sam_device)
            return SamPredictor(sam)
        except Exception as exc:
            if self.sam_cpu_fallback_on_oom and self.sam_device != "cpu":
                rospy.logwarn("[sam_final_approach] SAM load on %s failed: %s; falling back to cpu", self.sam_device, exc)
                sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_model_path)
                sam.to(device="cpu")
                self.sam_device = "cpu"
                return SamPredictor(sam)
            raise

    @staticmethod
    def _resolve_sam_device(requested):
        requested = str(requested or "auto").strip().lower()
        if requested in ("auto", "cuda_else_cpu", "cuda_if_available"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            rospy.logwarn("[sam_final_approach] CUDA requested but unavailable; falling back to cpu")
            return "cpu"
        return requested

    def _load_label_metadata(self):
        if not os.path.exists(self.object_map_yaml):
            rospy.logwarn("[sam_final_approach] object map YAML not found: %s", self.object_map_yaml)
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        objects = data.get("tag_objects", {}) if isinstance(data, dict) else {}
        mapping = {}
        for key, meta in objects.items():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("grasp_complete_label", "")).strip()
            if not label:
                continue
            enriched = dict(meta)
            try:
                enriched["tag_id"] = int(key)
            except Exception:
                enriched["tag_id"] = key
            mapping[label] = enriched
        return mapping

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[sam_final_approach] %s", text)
        self.status_pub.publish(String(data=text))

    def _image_cb(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest_image = image
                self.latest_image_stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
                if msg.header.frame_id:
                    self.camera_frame = str(msg.header.frame_id).strip()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[sam_final_approach] image decode failed: %s", exc)

    def _depth_cb(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.latest_depth = _depth_to_meters(depth)
                self.latest_depth_stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
                self.latest_depth_frame = str(msg.header.frame_id).strip()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[sam_final_approach] depth decode failed: %s", exc)

    def _camera_info_cb(self, msg):
        K = np.array(msg.K, dtype=np.float64).reshape((3, 3))
        if msg.width > 0 and msg.height > 0:
            with self.lock:
                self.camera_matrix = K
                if msg.header.frame_id:
                    self.camera_frame = str(msg.header.frame_id).strip()

    def _ee_cb(self, msg):
        with self.lock:
            self.latest_ee = copy.deepcopy(msg.pose)

    def _state_cb(self, msg):
        state = str(msg.data).strip().lower()
        if state not in self.active_states:
            self.reference_center = None
            self.reference_uv = None
            self.reference_depth_m = None
            self.reference_label = ""
            self.ready_pub.publish(Bool(data=False))
        self.execution_state = state

    def _label_cb(self, msg):
        label = str(msg.data).strip()
        if label != self.selected_label:
            self.reference_center = None
            self.reference_uv = None
            self.reference_depth_m = None
            self.reference_label = ""
            self.ready_pub.publish(Bool(data=False))
        self.selected_label = label

    def _tag_cb(self, msg, tag_id):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        self.observed_tags[int(tag_id)] = (copy.deepcopy(msg), stamp)

    def _selected_meta(self):
        meta = self.label_to_meta.get(self.selected_label)
        if not isinstance(meta, dict):
            return None
        category = str(meta.get("category", "")).strip().lower()
        if self.allowed_categories and category not in self.allowed_categories:
            return None
        return meta

    def _lookup_base_to_camera(self, stamp):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0) if stamp == rospy.Time() else stamp,
                rospy.Duration(self.tf_lookup_timeout_sec),
            )
            return _tf_to_matrix(tf_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def _make_roi(self, image_shape, center_uv):
        h, w = image_shape[:2]
        half = max(self.roi_min_size_px // 2, min(self.roi_half_size_px, self.roi_max_size_px // 2))
        u = int(round(float(center_uv[0])))
        v = int(round(float(center_uv[1])))
        x0 = max(0, u - half)
        y0 = max(0, v - half)
        x1 = min(w, u + half)
        y1 = min(h, v + half)
        if x1 - x0 < self.roi_min_size_px or y1 - y0 < self.roi_min_size_px:
            return None
        return x0, y0, x1, y1

    def _predict_mask(self, image_bgr, roi, center_uv):
        if self.predictor is None:
            return None
        x0, y0, x1, y1 = roi
        crop_bgr = image_bgr[y0:y1, x0:x1]
        if crop_bgr.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        point = np.array([[float(center_uv[0] - x0), float(center_uv[1] - y0)]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        box = np.array([0.0, 0.0, float(crop_rgb.shape[1] - 1), float(crop_rgb.shape[0] - 1)], dtype=np.float32)
        use_point = self.sam_prompt_mode in ("point_box", "point", "tag_point")
        use_box = self.sam_prompt_mode in ("box", "point_box", "roi_box")
        try:
            self.predictor.set_image(crop_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=point if use_point else None,
                point_labels=labels if use_point else None,
                box=box if use_box else None,
                multimask_output=True,
            )
        except Exception as exc:
            if self.sam_cpu_fallback_on_oom and "out of memory" in str(exc).lower() and self.sam_device != "cpu":
                rospy.logwarn("[sam_final_approach] SAM OOM; falling back to CPU")
                self.sam_device = "cpu"
                self.predictor.model.to(device="cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.predictor.set_image(crop_rgb)
                masks, scores, _ = self.predictor.predict(
                    point_coords=point if use_point else None,
                    point_labels=labels if use_point else None,
                    box=box if use_box else None,
                    multimask_output=True,
                )
            else:
                self._publish_status("sam_predict_failed {}".format(exc))
                return None
        if masks is None or len(masks) == 0:
            return None
        roi_area = float(max(1, crop_rgb.shape[0] * crop_rgb.shape[1]))
        valid_indices = []
        for idx, candidate in enumerate(masks):
            area_fraction = float(np.count_nonzero(candidate)) / roi_area
            if area_fraction <= self.max_mask_area_fraction:
                valid_indices.append(idx)
        if not valid_indices:
            valid_indices = list(range(len(masks)))
        if scores is not None and len(scores) == len(masks):
            best_idx = max(valid_indices, key=lambda idx: float(scores[idx]))
        else:
            best_idx = max(valid_indices, key=lambda idx: int(np.count_nonzero(masks[idx])))
        mask_crop = masks[best_idx].astype(bool)
        full = np.zeros(image_bgr.shape[:2], dtype=bool)
        full[y0:y1, x0:x1] = mask_crop
        return full

    def _mask_center_base(self, mask, depth_m, T_base_cam, camera_matrix):
        if mask is None or depth_m is None:
            return None
        if depth_m.shape[:2] != mask.shape[:2]:
            depth_m = cv2.resize(depth_m, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mask)
        if len(xs) < self.min_depth_points:
            return None
        z = depth_m[ys, xs].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.05) & (z < 2.0)
        if int(np.count_nonzero(valid)) < self.min_depth_points:
            return None
        xs = xs[valid].astype(np.float64)
        ys = ys[valid].astype(np.float64)
        z = z[valid]
        if 0.0 < self.depth_trim_quantile < 0.45 and len(z) > 20:
            lo = np.quantile(z, self.depth_trim_quantile)
            hi = np.quantile(z, 1.0 - self.depth_trim_quantile)
            keep = (z >= lo) & (z <= hi)
            xs, ys, z = xs[keep], ys[keep], z[keep]
        if len(z) < self.min_depth_points:
            return None
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        x_cam = (xs - cx) * z / fx
        y_cam = (ys - cy) * z / fy
        pts_cam = np.vstack((x_cam, y_cam, z, np.ones_like(z)))
        pts_base = T_base_cam @ pts_cam
        center = np.median(pts_base[:3, :], axis=1)
        return center.astype(np.float64)

    def _local_depth_at_uv(self, depth_m, uv):
        if depth_m is None or uv is None:
            return None
        h, w = depth_m.shape[:2]
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        half = max(1, int(self.local_depth_window_px) // 2)
        x0 = max(0, u - half)
        y0 = max(0, v - half)
        x1 = min(w, u + half + 1)
        y1 = min(h, v + half + 1)
        patch = depth_m[y0:y1, x0:x1].astype(np.float64)
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 2.0)]
        if len(valid) < max(5, min(self.min_depth_points, 20)):
            return None
        return float(np.median(valid))

    def _tag_local_offset(self, center_uv, depth_m, T_base_cam, camera_matrix):
        depth = self._local_depth_at_uv(depth_m, center_uv)
        if depth is None:
            return None, "tag_local_depth_failed"
        if self.reference_uv is None or self.reference_label != self.selected_label:
            self.reference_uv = np.array(center_uv, dtype=np.float64)
            self.reference_depth_m = depth
            self.reference_label = self.selected_label
            self.ready_pub.publish(Bool(data=False))
            return np.zeros(3, dtype=np.float64), "captured_tag_local_reference:{}".format(self.selected_label)
        uv_error = np.array(center_uv, dtype=np.float64) - np.array(self.reference_uv, dtype=np.float64)
        raw_uv_error = np.array(uv_error, dtype=np.float64)
        pixel_error_norm = float(np.linalg.norm(uv_error))
        if pixel_error_norm <= self.local_pixel_deadband_px:
            uv_error[:] = 0.0
        elif self.local_max_pixel_error_px > 0.0 and pixel_error_norm > self.local_max_pixel_error_px:
            uv_error *= self.local_max_pixel_error_px / pixel_error_norm
        z = depth if depth > 0.05 else float(self.reference_depth_m or 0.5)
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        offset_cam = np.array(
            [
                self.local_offset_sign * uv_error[0] * z / max(fx, 1e-6),
                self.local_offset_sign * uv_error[1] * z / max(fy, 1e-6),
                0.0,
            ],
            dtype=np.float64,
        )
        offset_base = T_base_cam[:3, :3].dot(offset_cam)
        offset_base = self._clamp_offset(offset_base)
        self.ready_pub.publish(Bool(data=True))
        status = "tag_local_offset label={} du={:.1f}/{:.1f} dv={:.1f}/{:.1f} z={:.3f} dx={:.3f} dy={:.3f}".format(
            self.selected_label,
            raw_uv_error[0],
            uv_error[0],
            raw_uv_error[1],
            uv_error[1],
            z,
            offset_base[0],
            offset_base[1],
        )
        return offset_base, status

    def _clamp_offset(self, offset):
        out = np.array(offset, dtype=np.float64)
        xy_norm = float(np.linalg.norm(out[:2]))
        if self.max_xy_correction_m > 0.0 and xy_norm > self.max_xy_correction_m:
            out[:2] *= self.max_xy_correction_m / xy_norm
        if self.max_z_correction_m <= 0.0:
            out[2] = 0.0
        else:
            out[2] = float(np.clip(out[2], -self.max_z_correction_m, self.max_z_correction_m))
        return out

    def _publish_debug(self, image, roi, mask, center_uv, status_text, reference_uv=None):
        if image is None or self.debug_pub.get_num_connections() <= 0:
            return
        debug = image.copy()
        if roi is not None:
            x0, y0, x1, y1 = roi
            cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 255), 2)
        if mask is not None:
            overlay = np.zeros_like(debug)
            overlay[mask] = (0, 128, 255)
            debug = cv2.addWeighted(debug, 0.75, overlay, 0.25, 0.0)
        if center_uv is not None:
            cv2.circle(debug, (int(round(center_uv[0])), int(round(center_uv[1]))), 5, (0, 255, 0), -1)
        if reference_uv is not None:
            cv2.circle(debug, (int(round(reference_uv[0])), int(round(reference_uv[1]))), 7, (255, 0, 255), 2)
        if center_uv is not None and reference_uv is not None:
            p0 = (int(round(reference_uv[0])), int(round(reference_uv[1])))
            p1 = (int(round(center_uv[0])), int(round(center_uv[1])))
            cv2.arrowedLine(debug, p0, p1, (255, 0, 255), 2, tipLength=0.25)
        cv2.putText(debug, status_text[:80], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except Exception:
            pass

    def _tick(self, _evt):
        with self.lock:
            preview_image = None if self.latest_image is None else self.latest_image.copy()
        if self.execution_state not in self.active_states:
            self._publish_debug(
                preview_image,
                None,
                None,
                None,
                "inactive_state:{}".format(self.execution_state or "unknown"),
            )
            return
        if self.approach_mode == "sam_mask_center" and self.predictor is None:
            self._publish_status("sam_not_loaded")
            self._publish_debug(preview_image, None, None, None, "sam_not_loaded")
            return
        meta = self._selected_meta()
        if meta is None:
            self._publish_status("waiting_for_allowed_selection")
            self._publish_debug(preview_image, None, None, None, "waiting_for_allowed_selection")
            return
        try:
            tag_id = int(meta.get("tag_id"))
        except Exception:
            self._publish_status("selected_label_without_tag:{}".format(self.selected_label))
            return
        now = rospy.Time.now()
        tag_item = self.observed_tags.get(tag_id)
        if tag_item is None or (now - tag_item[1]).to_sec() > self.max_tag_age_sec:
            self._publish_status("waiting_for_live_tag:{}".format(tag_id))
            return

        with self.lock:
            image = None if self.latest_image is None else self.latest_image.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            image_stamp = self.latest_image_stamp
            depth_stamp = self.latest_depth_stamp
            camera_matrix = None if self.camera_matrix is None else self.camera_matrix.copy()
        if image is None or depth is None or camera_matrix is None:
            self._publish_status("waiting_for_rgbd")
            return
        if (now - image_stamp).to_sec() > self.max_image_age_sec or (now - depth_stamp).to_sec() > self.max_image_age_sec:
            self._publish_status("stale_rgbd")
            return

        T_base_cam = self._lookup_base_to_camera(image_stamp)
        if T_base_cam is None:
            self._publish_status("tf_lookup_failed {}->{}".format(self.base_frame, self.camera_frame))
            return
        tag_pose = tag_item[0]
        center_uv = _project_base_point_to_image(_pose_position(tag_pose), T_base_cam, camera_matrix)
        if center_uv is None:
            self._publish_status("selected_tag_not_in_camera:{}".format(tag_id))
            return
        roi = self._make_roi(image.shape, center_uv)
        if roi is None:
            self._publish_status("roi_out_of_bounds:{}".format(tag_id))
            return
        if self.approach_mode == "tag_local":
            offset, status = self._tag_local_offset(center_uv, depth, T_base_cam, camera_matrix)
            if offset is None:
                self._publish_status(status)
                self._publish_debug(image, roi, None, center_uv, status, reference_uv=self.reference_uv)
                return
            self._publish_status(status)
            msg = Vector3Stamped()
            msg.header.stamp = now
            msg.header.frame_id = self.base_frame
            msg.vector.x = float(offset[0])
            msg.vector.y = float(offset[1])
            msg.vector.z = float(offset[2])
            self.offset_pub.publish(msg)
            self._publish_debug(image, roi, None, center_uv, status, reference_uv=self.reference_uv)
            return
        if self.approach_mode != "sam_mask_center":
            self._publish_status("unknown_approach_mode:{}".format(self.approach_mode))
            self._publish_debug(image, roi, None, center_uv, self.last_status, reference_uv=self.reference_uv)
            return
        mask = self._predict_mask(image, roi, center_uv)
        if mask is None:
            self._publish_status("no_sam_mask:{}".format(tag_id))
            self._publish_debug(image, roi, None, center_uv, "no_sam_mask")
            return
        mask_area = float(np.count_nonzero(mask))
        roi_area = float(max(1, (roi[2] - roi[0]) * (roi[3] - roi[1])))
        if mask_area < self.min_mask_area_px or mask_area / roi_area > self.max_mask_area_fraction:
            self._publish_status("bad_mask_area area={:.0f} frac={:.2f}".format(mask_area, mask_area / roi_area))
            self._publish_debug(image, roi, mask, center_uv, "bad_mask_area")
            return
        center_base = self._mask_center_base(mask, depth, T_base_cam, camera_matrix)
        if center_base is None:
            self._publish_status("depth_center_failed")
            self._publish_debug(image, roi, mask, center_uv, "depth_center_failed")
            return
        center_base[:2] += self.category_grasp_center_offset_xy[:2]
        if self.reference_center is None or self.reference_label != self.selected_label:
            self.reference_center = center_base
            self.reference_label = self.selected_label
            self.ready_pub.publish(Bool(data=False))
            self._publish_status("captured_visual_reference:{}".format(self.selected_label))
            if self.publish_zero_until_reference:
                offset = np.zeros(3, dtype=np.float64)
            else:
                return
        else:
            offset = self._clamp_offset(center_base - self.reference_center)
            self.ready_pub.publish(Bool(data=True))
            self._publish_status(
                "offset label={} dx={:.3f} dy={:.3f} dz={:.3f}".format(
                    self.selected_label,
                    offset[0],
                    offset[1],
                    offset[2],
                )
            )

        msg = Vector3Stamped()
        msg.header.stamp = now
        msg.header.frame_id = self.base_frame
        msg.vector.x = float(offset[0])
        msg.vector.y = float(offset[1])
        msg.vector.z = float(offset[2])
        self.offset_pub.publish(msg)
        self._publish_debug(image, roi, mask, center_uv, self.last_status)


if __name__ == "__main__":
    SamFinalApproachNode()
    rospy.spin()
