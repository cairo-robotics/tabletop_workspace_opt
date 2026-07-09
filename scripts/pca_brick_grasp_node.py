#!/usr/bin/env python3
"""Generate top-down LEGO grasp poses from RGB-only segmentation.

Pipeline:
  1. Subscribe to RGB image and CameraInfo
  2. Estimate tabletop/background color from image borders
  3. Segment foreground objects in RGB
  4. Find candidate contours and fit minAreaRect to each
  5. Select one candidate in image space
  6. Back-project the 2D center to the known tabletop plane
  7. Build a top-down pregrasp/grasp pose aligned with the long edge
"""

import math
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
        parts = [p for p in text.split() if p]
        values = [float(v) for v in parts]
    else:
        values = [float(raw)]

    if expected_len is not None and len(values) != expected_len:
        rospy.logwarn(
            "[pca_brick_grasp_node] param %s expected len=%d, got %r. using default=%r",
            name,
            expected_len,
            raw,
            default,
        )
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


def _quat_dot_abs(q0, q1):
    qa = _quat_normalize_xyzw(q0)
    qb = _quat_normalize_xyzw(q1)
    return float(abs(np.dot(qa, qb)))


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


def _make_pose_stamped(frame_id, stamp, position, rotation):
    quat = _matrix_to_quat_xyzw(rotation)
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.pose = Pose()
    msg.pose.position.x = float(position[0])
    msg.pose.position.y = float(position[1])
    msg.pose.position.z = float(position[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def _make_point(xyz):
    point = Point()
    point.x = float(xyz[0])
    point.y = float(xyz[1])
    point.z = float(xyz[2])
    return point


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


class PCABrickGraspNode:
    def __init__(self):
        rospy.init_node("pca_brick_grasp_node")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.camera_frame = str(rospy.get_param("~camera_frame", "camera_color_optical_frame")).strip()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info")).strip()
        self.pregrasp_pose_topic = str(rospy.get_param("~pregrasp_pose_topic", "~pregrasp_pose")).strip()
        self.grasp_pose_topic = str(rospy.get_param("~grasp_pose_topic", "~grasp_pose")).strip()
        self.markers_topic = str(rospy.get_param("~markers_topic", "~markers")).strip()
        self.debug_image_topic = str(rospy.get_param("~debug_image_topic", "~debug_image")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.selected_template_topic = str(rospy.get_param("~selected_template_topic", "~selected_template")).strip()

        self.world_up_axis = _normalize(_parse_float_list_param("~world_up_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_forward_axis = _normalize(_parse_float_list_param("~ee_forward_axis", [0.0, 0.0, 1.0], 3), [0.0, 0.0, 1.0])
        self.ee_up_axis = _normalize(_parse_float_list_param("~ee_up_axis", [0.0, 1.0, 0.0], 3), [0.0, 1.0, 0.0])

        self.image_roi_min_uv = _parse_float_list_param("~image_roi_min_uv", [0.0, 0.0], 2)
        self.image_roi_max_uv = _parse_float_list_param("~image_roi_max_uv", [-1.0, -1.0], 2)
        self.selection_reference_uv = _parse_float_list_param("~selection_reference_uv", [], None)
        self.selection_reference_weight = float(rospy.get_param("~selection_reference_weight", 1.0))
        self.selection_distance_scale_px = float(rospy.get_param("~selection_distance_scale_px", 120.0))
        self.border_sample_px = int(rospy.get_param("~border_sample_px", 20))
        self.foreground_lab_threshold = float(rospy.get_param("~foreground_lab_threshold", 18.0))
        self.foreground_min_saturation = float(rospy.get_param("~foreground_min_saturation", 35.0))
        self.use_hsv_color_mask = bool(rospy.get_param("~use_hsv_color_mask", True))
        self.use_grouped_hsv_mask = bool(rospy.get_param("~use_grouped_hsv_mask", True))
        self.target_color_group = str(rospy.get_param("~target_color_group", "warm")).strip().lower()
        self.strict_group_mask = bool(rospy.get_param("~strict_group_mask", True))
        self.hsv_lower = _parse_float_list_param("~hsv_lower", [5, 80, 80], 3).astype(np.uint8)
        self.hsv_upper = _parse_float_list_param("~hsv_upper", [35, 255, 255], 3).astype(np.uint8)
        self.warm_hsv_lower = _parse_float_list_param("~warm_hsv_lower", [8, 70, 70], 3).astype(np.uint8)
        self.warm_hsv_upper = _parse_float_list_param("~warm_hsv_upper", [40, 255, 255], 3).astype(np.uint8)
        self.cool_hsv_lower = _parse_float_list_param("~cool_hsv_lower", [95, 50, 50], 3).astype(np.uint8)
        self.cool_hsv_upper = _parse_float_list_param("~cool_hsv_upper", [155, 255, 255], 3).astype(np.uint8)
        self.reject_specular = bool(rospy.get_param("~reject_specular", True))
        self.specular_value_threshold = float(rospy.get_param("~specular_value_threshold", 235.0))
        self.specular_saturation_threshold = float(rospy.get_param("~specular_saturation_threshold", 40.0))
        self.morph_kernel_px = int(rospy.get_param("~morph_kernel_px", 5))
        self.min_contour_area_px = float(rospy.get_param("~min_contour_area_px", 150.0))
        self.max_contour_area_px = float(rospy.get_param("~max_contour_area_px", 50000.0))
        self.min_rect_aspect_ratio = float(rospy.get_param("~min_rect_aspect_ratio", 1.1))
        self.max_rect_aspect_ratio = float(rospy.get_param("~max_rect_aspect_ratio", 8.0))
        self.min_fill_ratio = float(rospy.get_param("~min_fill_ratio", 0.55))
        self.reject_border_contours = bool(rospy.get_param("~reject_border_contours", True))
        self.border_reject_margin_px = int(rospy.get_param("~border_reject_margin_px", 6))
        self.max_visualized_candidates = int(rospy.get_param("~max_visualized_candidates", 12))

        self.grasp_height_offset_m = float(rospy.get_param("~grasp_height_offset_m", 0.090))
        self.pregrasp_offset_m = float(rospy.get_param("~pregrasp_offset_m", 0.080))
        self.fixed_table_z = float(rospy.get_param("~fixed_table_z", -0.255))
        self.object_height_m = float(rospy.get_param("~object_height_m", 0.0192))
        self.process_rate_hz = float(rospy.get_param("~process_rate_hz", 3.0))
        self.position_ema_alpha = float(rospy.get_param("~position_ema_alpha", 0.25))
        self.orientation_ema_alpha = float(rospy.get_param("~orientation_ema_alpha", 0.25))
        self.stable_position_threshold_m = float(rospy.get_param("~stable_position_threshold_m", 0.01))
        self.stable_orientation_dot_threshold = float(rospy.get_param("~stable_orientation_dot_threshold", 0.995))
        self.stable_count_required = int(rospy.get_param("~stable_count_required", 4))
        self.publish_unstable_pose = bool(rospy.get_param("~publish_unstable_pose", False))

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.latest_image = None
        self.latest_image_frame = self.camera_frame
        self.camera_matrix = None
        self.last_process_time = rospy.Time(0)
        self.processing_lock = threading.Lock()
        self.last_status = ""

        self.filtered_centroid = None
        self.filtered_quat = None
        self.filtered_long_axis = None
        self.filtered_short_axis = None
        self.filtered_uv = None
        self.previous_candidate_uv = None
        self.previous_candidate_area = None
        self.previous_raw_centroid = None
        self.current_raw_centroid = None
        self.stable_count = 0
        self.last_segmentation_summary = "uninitialized"

        self.pregrasp_pub = rospy.Publisher(self.pregrasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.grasp_pub = rospy.Publisher(self.grasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.markers_pub = rospy.Publisher(self.markers_topic, MarkerArray, queue_size=1, latch=True)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.selected_template_pub = rospy.Publisher(self.selected_template_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_cb, queue_size=1)

        self._publish_status("waiting_for_image topic={}".format(self.image_topic))
        rospy.loginfo(
            "RGB brick grasp node ready. image_topic=%s camera_info_topic=%s base_frame=%s fixed_table_z=%.3f",
            self.image_topic,
            self.camera_info_topic,
            self.base_frame,
            self.fixed_table_z,
        )

    def camera_info_cb(self, msg):
        self.camera_matrix = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def image_cb(self, msg):
        if self.camera_matrix is None:
            return
        if self.process_rate_hz > 0.0:
            now = rospy.Time.now()
            if (now - self.last_process_time).to_sec() < (1.0 / self.process_rate_hz):
                return
            self.last_process_time = now
        if not self.processing_lock.acquire(False):
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = image
            if msg.header.frame_id:
                self.latest_image_frame = str(msg.header.frame_id).strip()
                self.camera_frame = self.latest_image_frame
            self._process_image(msg.header.stamp)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[pca_brick_grasp_node] image processing failed: %s", exc)
        finally:
            self.processing_lock.release()

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[pca_brick_grasp_node] %s", text)
        self.status_pub.publish(String(data=text))

    def _process_image(self, stamp):
        image = self.latest_image
        if image is None:
            self._publish_status("waiting_for_image topic={}".format(self.image_topic))
            return

        candidates = self._segment_candidates(image)
        if not candidates:
            self._reset_tracking()
            self.markers_pub.publish(self._make_empty_markers())
            self._publish_debug_image(image, [], None)
            self._publish_status("no_rgb_candidate {}".format(self.last_segmentation_summary))
            return

        candidate = self._choose_candidate(candidates)
        if candidate is None:
            self._reset_tracking()
            self.markers_pub.publish(self._make_empty_markers())
            self._publish_debug_image(image, candidates, None)
            self._publish_status("no_rgb_candidate_selected")
            return

        visual_solutions = []
        for idx, item in enumerate(candidates[: max(1, self.max_visualized_candidates)]):
            visual_solution = self._build_solution_from_candidate(item, stamp, apply_temporal_filter=False)
            if visual_solution is None:
                continue
            visual_solution["visual_index"] = idx
            visual_solution["selected"] = item is candidate
            visual_solutions.append(visual_solution)

        solution = self._build_solution_from_candidate(candidate, stamp, apply_temporal_filter=True)
        if solution is None or not visual_solutions:
            self._reset_tracking()
            self.markers_pub.publish(self._make_empty_markers())
            self._publish_debug_image(image, candidates, candidate)
            self._publish_status("projection_failed")
            return

        self.markers_pub.publish(self._make_markers(visual_solutions, solution, stamp))
        self._publish_debug_image(image, candidates, candidate)
        if solution["stable"] or self.publish_unstable_pose:
            self.selected_template_pub.publish(String(data="rgb_lego_topdown"))
            self.pregrasp_pub.publish(solution["pregrasp_pose"])
            self.grasp_pub.publish(solution["grasp_pose"])

        status_prefix = "tracking_cluster" if solution["stable"] else "tracking_cluster_unstable"
        self._publish_status(
            "{} selected={} total_candidates={} center_uv=[{:.1f},{:.1f}] center=[{:.3f},{:.3f},{:.3f}] stable_count={}/{}".format(
                status_prefix,
                int(candidates.index(candidate)),
                int(len(candidates)),
                float(solution["center_uv"][0]),
                float(solution["center_uv"][1]),
                float(solution["centroid"][0]),
                float(solution["centroid"][1]),
                float(solution["centroid"][2]),
                int(self.stable_count),
                int(self.stable_count_required),
            )
        )

    def _segment_candidates(self, image):
        roi, (x0, y0) = self._crop_image_roi(image)
        if roi is None or roi.size == 0:
            self.last_segmentation_summary = "empty_roi"
            return []

        candidates = []
        ref_uv = self._selection_reference_uv(image.shape[1], image.shape[0])
        mask_groups = self._foreground_masks_by_group(roi)
        raw_contour_count = 0
        candidate_index = 0
        for group_name, mask in mask_groups:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            raw_contour_count += len(contours)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_contour_area_px or area > self.max_contour_area_px:
                    continue
                if self.reject_border_contours and self._touches_roi_border(contour, roi.shape[1], roi.shape[0]):
                    continue
                rect = cv2.minAreaRect(contour)
                (cx, cy), (w, h), _ = rect
                if min(w, h) < 2.0:
                    continue
                rect_area = float(w * h)
                if rect_area < 1e-6:
                    continue
                aspect = max(w, h) / max(min(w, h), 1e-6)
                if aspect < self.min_rect_aspect_ratio or aspect > self.max_rect_aspect_ratio:
                    continue
                fill_ratio = area / rect_area
                if fill_ratio < self.min_fill_ratio:
                    continue
                center_uv = np.array([cx + x0, cy + y0], dtype=np.float64)
                if self._is_duplicate_candidate(center_uv, area, candidates):
                    continue
                distance = float(np.linalg.norm(center_uv - ref_uv))
                temporal_distance = 0.0 if self.previous_candidate_uv is None else float(np.linalg.norm(center_uv - self.previous_candidate_uv))
                temporal_area_delta = 0.0 if self.previous_candidate_area is None else abs(area - self.previous_candidate_area) / max(self.previous_candidate_area, 1.0)
                candidates.append(
                    {
                        "contour_local": contour,
                        "contour": contour + np.array([[[x0, y0]]], dtype=np.int32),
                        "rect_local": rect,
                        "center_uv": center_uv,
                        "distance_px": distance,
                        "temporal_distance_px": temporal_distance,
                        "temporal_area_delta": temporal_area_delta,
                        "area": area,
                        "aspect": aspect,
                        "fill_ratio": fill_ratio,
                        "roi_origin": (x0, y0),
                        "candidate_index": candidate_index,
                        "color_group": group_name,
                    }
                )
                candidate_index += 1

        if not candidates:
            self.last_segmentation_summary = "raw_contours={}".format(int(raw_contour_count))
            return []
        self.last_segmentation_summary = "raw_contours={} kept_candidates={}".format(int(raw_contour_count), int(len(candidates)))
        return sorted(candidates, key=lambda item: float(item["area"]), reverse=True)

    def _is_duplicate_candidate(self, center_uv, area, candidates):
        for item in candidates:
            center_dist = float(np.linalg.norm(center_uv - item["center_uv"]))
            area_ratio = abs(area - float(item["area"])) / max(float(item["area"]), 1.0)
            if center_dist < 25.0 and area_ratio < 0.35:
                return True
        return False

    def _choose_candidate(self, candidates):
        if not candidates:
            return None
        chosen = None
        if not self._has_explicit_reference_uv():
            chosen = min(candidates, key=self._candidate_cost_without_reference)
        else:
            chosen = min(candidates, key=self._candidate_cost)
        self.previous_candidate_uv = chosen["center_uv"].copy()
        self.previous_candidate_area = float(chosen["area"])
        return chosen

    def _touches_roi_border(self, contour, roi_width, roi_height):
        x, y, w, h = cv2.boundingRect(contour)
        margin = max(0, int(self.border_reject_margin_px))
        if x <= margin:
            return True
        if y <= margin:
            return True
        if (x + w) >= (roi_width - margin):
            return True
        if (y + h) >= (roi_height - margin):
            return True
        return False

    def _publish_debug_image(self, image, candidates, selected_candidate):
        if self.debug_image_pub.get_num_connections() <= 0:
            return
        vis = image.copy()
        height, width = vis.shape[:2]

        ref_uv = self._selection_reference_uv(width, height)
        cv2.circle(vis, (int(ref_uv[0]), int(ref_uv[1])), 6, (255, 255, 0), -1)
        cv2.putText(vis, "ref", (int(ref_uv[0]) + 8, int(ref_uv[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        x0 = int(np.clip(self.image_roi_min_uv[0], 0, width - 1))
        y0 = int(np.clip(self.image_roi_min_uv[1], 0, height - 1))
        x1 = width if self.image_roi_max_uv[0] < 0.0 else int(np.clip(self.image_roi_max_uv[0], x0 + 1, width))
        y1 = height if self.image_roi_max_uv[1] < 0.0 else int(np.clip(self.image_roi_max_uv[1], y0 + 1, height))
        cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 200), 1)

        for idx, candidate in enumerate(candidates[: max(1, self.max_visualized_candidates)]):
            is_selected = candidate is selected_candidate
            group_name = str(candidate.get("color_group", ""))
            if group_name == "warm":
                contour_color = (0, 255, 255)
                box_color = (0, 0, 255) if is_selected else (0, 180, 255)
                point_color = (0, 255, 0) if is_selected else (0, 180, 255)
            elif group_name == "cool":
                contour_color = (255, 255, 0)
                box_color = (255, 0, 0) if is_selected else (255, 160, 0)
                point_color = (0, 255, 0) if is_selected else (255, 160, 0)
            else:
                contour_color = (180, 255, 180)
                box_color = (180, 180, 180)
                point_color = (180, 255, 180)
            thickness = 2 if is_selected else 1
            cv2.drawContours(vis, [candidate["contour"]], -1, contour_color, thickness)
            box = cv2.boxPoints(candidate["rect_local"])
            box[:, 0] += candidate["roi_origin"][0]
            box[:, 1] += candidate["roi_origin"][1]
            box = np.round(box).astype(np.int32)
            cv2.polylines(vis, [box], True, box_color, thickness)
            center_uv = candidate["center_uv"]
            cv2.circle(vis, (int(center_uv[0]), int(center_uv[1])), 6, point_color, -1)
            cv2.putText(
                vis,
                "target" if is_selected else "{}_c{}".format(group_name[:1] if group_name else "g", idx),
                (int(center_uv[0]) + 8, int(center_uv[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                point_color,
                1,
                cv2.LINE_AA,
            )

        try:
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))
        except Exception:
            pass

    def _crop_image_roi(self, image):
        height, width = image.shape[:2]
        x0 = int(np.clip(self.image_roi_min_uv[0], 0, width - 1))
        y0 = int(np.clip(self.image_roi_min_uv[1], 0, height - 1))
        if self.image_roi_max_uv[0] < 0.0:
            x1 = width
        else:
            x1 = int(np.clip(self.image_roi_max_uv[0], x0 + 1, width))
        if self.image_roi_max_uv[1] < 0.0:
            y1 = height
        else:
            y1 = int(np.clip(self.image_roi_max_uv[1], y0 + 1, height))
        if x1 <= x0 or y1 <= y0:
            return None, (0, 0)
        return image[y0:y1, x0:x1], (x0, y0)

    def _foreground_masks_by_group(self, roi):
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        border = max(2, int(self.border_sample_px))
        border_pixels = np.concatenate(
            [
                lab[:border, :, :].reshape(-1, 3),
                lab[-border:, :, :].reshape(-1, 3),
                lab[:, :border, :].reshape(-1, 3),
                lab[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        table_color = np.median(border_pixels.astype(np.float32), axis=0)
        diff = np.linalg.norm(lab.astype(np.float32) - table_color.reshape(1, 1, 3), axis=2)
        sat = hsv[:, :, 1].astype(np.float32)

        mask_generic = ((diff > self.foreground_lab_threshold) | (sat > self.foreground_min_saturation)).astype(np.uint8) * 255
        masks = []
        if self.use_hsv_color_mask and self.use_grouped_hsv_mask:
            mask_warm = cv2.inRange(hsv, self.warm_hsv_lower, self.warm_hsv_upper)
            mask_cool = cv2.inRange(hsv, self.cool_hsv_lower, self.cool_hsv_upper)
            warm_mask = mask_warm if self.strict_group_mask else cv2.bitwise_or(mask_generic, mask_warm)
            cool_mask = mask_cool if self.strict_group_mask else cv2.bitwise_or(mask_generic, mask_cool)
            grouped_masks = {
                "warm": warm_mask,
                "cool": cool_mask,
            }
            if self.target_color_group == "all":
                masks.extend(grouped_masks.items())
            elif self.target_color_group in grouped_masks:
                masks.append((self.target_color_group, grouped_masks[self.target_color_group]))
            else:
                masks.append(("warm", grouped_masks["warm"]))
        elif self.use_hsv_color_mask:
            mask_color = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            masks.append(("color", cv2.bitwise_or(mask_generic, mask_color)))
        else:
            masks.append(("generic", mask_generic))
        if self.reject_specular:
            specular_mask = (
                (hsv[:, :, 2].astype(np.float32) >= self.specular_value_threshold)
                & (hsv[:, :, 1].astype(np.float32) <= self.specular_saturation_threshold)
            )
        kernel_size = max(1, int(self.morph_kernel_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        output_masks = []
        for group_name, mask in masks:
            mask = mask.copy()
            if self.reject_specular:
                mask[specular_mask] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            output_masks.append((group_name, mask))
        return output_masks

    def _selection_reference_uv(self, width, height):
        if self.selection_reference_uv.size == 2:
            return self.selection_reference_uv
        roi, (x0, y0) = self._crop_image_roi(np.zeros((height, width, 3), dtype=np.uint8))
        if roi is None:
            return np.array([0.5 * width, 0.5 * height], dtype=np.float64)
        return np.array([x0 + 0.5 * roi.shape[1], y0 + 0.5 * roi.shape[0]], dtype=np.float64)

    def _has_explicit_reference_uv(self):
        return self.selection_reference_uv.size == 2

    def _candidate_cost(self, item):
        scale = max(self.selection_distance_scale_px, 1e-3)
        return (
            float(self.selection_reference_weight) * float(item["distance_px"]) / scale
            + 0.35 * float(item["temporal_distance_px"]) / scale
            + 0.50 * float(item["temporal_area_delta"])
            - 0.002 * float(item["area"])
            - 0.75 * float(item["fill_ratio"])
        )

    def _candidate_cost_without_reference(self, item):
        scale = max(self.selection_distance_scale_px, 1e-3)
        return (
            0.65 * float(item["temporal_distance_px"]) / scale
            + 0.50 * float(item["temporal_area_delta"])
            - 0.003 * float(item["area"])
            - 1.00 * float(item["fill_ratio"])
        )

    def _build_solution_from_candidate(self, candidate, stamp, apply_temporal_filter):
        center_uv = candidate["center_uv"]
        T_base_cam = self._lookup_base_to_camera(stamp)
        if T_base_cam is None:
            return None

        center_base = self._pixel_to_plane(center_uv[0], center_uv[1], T_base_cam, self.fixed_table_z)
        if center_base is None:
            return None
        center_base = center_base + self.world_up_axis * (0.5 * self.object_height_m)

        long_axis_base, short_axis_base = self._candidate_axes_to_base(candidate, T_base_cam)
        if long_axis_base is None or short_axis_base is None:
            return None

        if apply_temporal_filter:
            self.previous_raw_centroid = None if self.current_raw_centroid is None else self.current_raw_centroid.copy()
            self.current_raw_centroid = center_base.copy()
            centroid, long_axis_base, short_axis_base = self._apply_temporal_filter(center_base, long_axis_base, short_axis_base)
        else:
            centroid = center_base.copy()

        if np.dot(np.cross(long_axis_base, short_axis_base), self.world_up_axis) < 0.0:
            short_axis_base = -short_axis_base

        grasp_rotation = _compute_axis_alignment_rotation(
            forward_w=-self.world_up_axis,
            up_ref=short_axis_base,
            ee_forward_axis=self.ee_forward_axis,
            ee_up_axis=self.ee_up_axis,
        )
        grasp_quat = _matrix_to_quat_xyzw(grasp_rotation)
        stable = self._update_stability(grasp_quat) if apply_temporal_filter else False

        grasp_position = centroid + self.world_up_axis * self.grasp_height_offset_m
        pregrasp_position = grasp_position + self.world_up_axis * self.pregrasp_offset_m

        return {
            "center_uv": center_uv,
            "centroid": centroid,
            "long_axis": long_axis_base,
            "short_axis": short_axis_base,
            "up_axis": self.world_up_axis.copy(),
            "contour": candidate["contour"],
            "area": float(candidate["area"]),
            "grasp_pose": _make_pose_stamped(self.base_frame, stamp, grasp_position, grasp_rotation),
            "pregrasp_pose": _make_pose_stamped(self.base_frame, stamp, pregrasp_position, grasp_rotation),
            "stable": stable,
        }

    def _candidate_axes_to_base(self, candidate, T_base_cam):
        box = cv2.boxPoints(candidate["rect_local"])
        center = np.array(candidate["rect_local"][0], dtype=np.float64)
        lengths = []
        dirs = []
        for idx in range(4):
            p0 = box[idx]
            p1 = box[(idx + 1) % 4]
            vec = p1 - p0
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            lengths.append(length)
            dirs.append(vec / length)
        if not lengths:
            return None, None
        long_dir_2d = dirs[int(np.argmax(lengths))]
        half = 0.5 * max(lengths)
        p0_uv = center + np.array([0.0, 0.0], dtype=np.float64) - long_dir_2d * half
        p1_uv = center + np.array([0.0, 0.0], dtype=np.float64) + long_dir_2d * half

        # rect_local is in ROI coordinates, contour/global center_uv are global.
        global_center = candidate["center_uv"]
        p0_uv = global_center - long_dir_2d * half
        p1_uv = global_center + long_dir_2d * half

        p0_base = self._pixel_to_plane(p0_uv[0], p0_uv[1], T_base_cam, self.fixed_table_z)
        p1_base = self._pixel_to_plane(p1_uv[0], p1_uv[1], T_base_cam, self.fixed_table_z)
        if p0_base is None or p1_base is None:
            return None, None

        long_axis = _normalize(p1_base - p0_base, [1.0, 0.0, 0.0])
        long_axis = _normalize(long_axis - np.dot(long_axis, self.world_up_axis) * self.world_up_axis, [1.0, 0.0, 0.0])
        short_axis = _normalize(np.cross(self.world_up_axis, long_axis), [0.0, 1.0, 0.0])
        return long_axis, short_axis

    def _lookup_base_to_camera(self, stamp):
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rospy.Time(0) if stamp == rospy.Time() else stamp,
                rospy.Duration(0.2),
            )
            return _tf_to_matrix(tf_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "[pca_brick_grasp_node] TF lookup failed %s -> %s: %s", self.base_frame, self.camera_frame, exc)
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

    def _apply_temporal_filter(self, raw_centroid, raw_long_axis, raw_short_axis):
        a = float(np.clip(self.position_ema_alpha, 0.0, 1.0))
        if self.filtered_centroid is None:
            self.filtered_centroid = raw_centroid.copy()
            self.filtered_long_axis = raw_long_axis.copy()
            self.filtered_short_axis = raw_short_axis.copy()
            return self.filtered_centroid.copy(), self.filtered_long_axis.copy(), self.filtered_short_axis.copy()

        self.filtered_centroid = (1.0 - a) * self.filtered_centroid + a * raw_centroid
        if np.dot(raw_long_axis, self.filtered_long_axis) < 0.0:
            raw_long_axis = -raw_long_axis
        if np.dot(raw_short_axis, self.filtered_short_axis) < 0.0:
            raw_short_axis = -raw_short_axis

        self.filtered_long_axis = _normalize((1.0 - a) * self.filtered_long_axis + a * raw_long_axis, raw_long_axis)
        self.filtered_short_axis = _normalize((1.0 - a) * self.filtered_short_axis + a * raw_short_axis, raw_short_axis)
        self.filtered_long_axis = _normalize(
            self.filtered_long_axis - np.dot(self.filtered_long_axis, self.world_up_axis) * self.world_up_axis,
            raw_long_axis,
        )
        self.filtered_short_axis = _normalize(np.cross(self.world_up_axis, self.filtered_long_axis), raw_short_axis)
        return self.filtered_centroid.copy(), self.filtered_long_axis.copy(), self.filtered_short_axis.copy()

    def _update_stability(self, current_quat):
        if self.filtered_quat is None:
            self.filtered_quat = current_quat.copy()
            self.stable_count = 1
            return self.stable_count >= self.stable_count_required

        prev_quat = self.filtered_quat.copy()
        t = float(np.clip(self.orientation_ema_alpha, 0.0, 1.0))
        self.filtered_quat = _quat_slerp_xyzw(self.filtered_quat, current_quat, t)

        if self.previous_raw_centroid is None or self.current_raw_centroid is None:
            pos_delta = 0.0
        else:
            pos_delta = float(np.linalg.norm(self.current_raw_centroid - self.previous_raw_centroid))
        quat_dot = _quat_dot_abs(prev_quat, current_quat)

        if pos_delta <= self.stable_position_threshold_m and quat_dot >= self.stable_orientation_dot_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 1
        return self.stable_count >= self.stable_count_required

    def _reset_tracking(self):
        self.filtered_centroid = None
        self.filtered_quat = None
        self.filtered_long_axis = None
        self.filtered_short_axis = None
        self.filtered_uv = None
        self.previous_candidate_uv = None
        self.previous_candidate_area = None
        self.previous_raw_centroid = None
        self.current_raw_centroid = None
        self.stable_count = 0

    def _make_empty_markers(self):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        return markers

    def _make_markers(self, visual_solutions, selected_solution, stamp):
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        axis_scale = 0.05
        for idx, solution in enumerate(visual_solutions):
            centroid = solution["centroid"]
            is_selected = bool(solution.get("selected", False))
            base_id = idx * 100
            alpha = 0.95 if is_selected else 0.40

            centroid_marker = Marker()
            centroid_marker.header.frame_id = self.base_frame
            centroid_marker.header.stamp = stamp
            centroid_marker.ns = "rgb_brick_candidates"
            centroid_marker.id = base_id + 0
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD
            centroid_marker.pose.position = _make_point(centroid)
            centroid_marker.pose.orientation.w = 1.0
            centroid_marker.scale.x = 0.018 if is_selected else 0.014
            centroid_marker.scale.y = 0.018 if is_selected else 0.014
            centroid_marker.scale.z = 0.018 if is_selected else 0.014
            centroid_marker.color = ColorRGBA(1.0, 0.8, 0.1, alpha)
            markers.markers.append(centroid_marker)

            axis_specs = [
                (1, solution["long_axis"], ColorRGBA(1.0, 0.1, 0.1, alpha), "long"),
                (2, solution["short_axis"], ColorRGBA(0.1, 1.0, 0.1, alpha), "short"),
                (3, solution["up_axis"], ColorRGBA(0.1, 0.4, 1.0, alpha), "up"),
            ]
            for marker_offset, axis, color, label_text in axis_specs:
                tip = centroid + _normalize(axis, [1.0, 0.0, 0.0]) * axis_scale
                arrow = Marker()
                arrow.header.frame_id = self.base_frame
                arrow.header.stamp = stamp
                arrow.ns = "rgb_candidate_axes"
                arrow.id = base_id + marker_offset
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.scale.x = 0.005
                arrow.scale.y = 0.010
                arrow.scale.z = 0.012
                arrow.color = color
                arrow.points = [_make_point(centroid), _make_point(tip)]
                markers.markers.append(arrow)

                if is_selected:
                    label = Marker()
                    label.header.frame_id = self.base_frame
                    label.header.stamp = stamp
                    label.ns = "rgb_axis_labels"
                    label.id = base_id + 10 + marker_offset
                    label.type = Marker.TEXT_VIEW_FACING
                    label.action = Marker.ADD
                    label.pose.position = _make_point(tip + np.array([0.0, 0.0, 0.015], dtype=np.float64))
                    label.pose.orientation.w = 1.0
                    label.scale.z = 0.02
                    label.color = ColorRGBA(1.0, 1.0, 1.0, 0.95)
                    label.text = label_text
                    markers.markers.append(label)

            pose_specs = [
                (20, solution["pregrasp_pose"], ColorRGBA(1.0, 0.6, 0.1, alpha), "pregrasp"),
                (21, solution["grasp_pose"], ColorRGBA(0.2, 1.0, 0.8, alpha), "grasp"),
            ]
            for marker_offset, pose_msg, color, label_text in pose_specs:
                arrow = Marker()
                arrow.header.frame_id = self.base_frame
                arrow.header.stamp = stamp
                arrow.ns = "rgb_candidate_grasp_pose"
                arrow.id = base_id + marker_offset
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose = pose_msg.pose
                arrow.scale.x = 0.08 if is_selected else 0.06
                arrow.scale.y = 0.012 if is_selected else 0.009
                arrow.scale.z = 0.012 if is_selected else 0.009
                arrow.color = color
                markers.markers.append(arrow)

                if is_selected:
                    label = Marker()
                    label.header.frame_id = self.base_frame
                    label.header.stamp = stamp
                    label.ns = "rgb_grasp_pose_label"
                    label.id = base_id + 30 + marker_offset
                    label.type = Marker.TEXT_VIEW_FACING
                    label.action = Marker.ADD
                    label.pose.position = _make_point(
                        np.array(
                            [
                                pose_msg.pose.position.x,
                                pose_msg.pose.position.y,
                                pose_msg.pose.position.z + 0.03,
                            ],
                            dtype=np.float64,
                        )
                    )
                    label.pose.orientation.w = 1.0
                    label.scale.z = 0.02
                    label.color = ColorRGBA(1.0, 1.0, 1.0, 0.95)
                    label.text = label_text
                    markers.markers.append(label)

            candidate_label = Marker()
            candidate_label.header.frame_id = self.base_frame
            candidate_label.header.stamp = stamp
            candidate_label.ns = "rgb_candidate_ids"
            candidate_label.id = base_id + 90
            candidate_label.type = Marker.TEXT_VIEW_FACING
            candidate_label.action = Marker.ADD
            candidate_label.pose.position = _make_point(centroid + np.array([0.0, 0.0, 0.03], dtype=np.float64))
            candidate_label.pose.orientation.w = 1.0
            candidate_label.scale.z = 0.018
            candidate_label.color = ColorRGBA(1.0, 1.0, 1.0, alpha)
            candidate_label.text = "brick_{}".format(idx)
            markers.markers.append(candidate_label)

        return markers


def main():
    PCABrickGraspNode()
    rospy.spin()


if __name__ == "__main__":
    main()
