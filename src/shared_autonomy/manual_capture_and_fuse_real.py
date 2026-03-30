#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual multi-view capture for real RealSense point clouds plus Open3D fusion.
 
Workflow:
- manually move the robot to a good viewpoint
- press `s` in this terminal to save one transformed point-cloud frame
- repeat around the object
- press `f` to run the existing pose-graph fusion pipeline
- press `q` to quit
 
This script does not command the robot. It only captures point clouds, stores
`scan_XXXX.ply`, and optionally fuses them with the existing ICP/pose-graph
pipeline.
 
Running command:
python3 -u /home/aaquib/sawyer_ws/src/tabletop_workspace_opt/src/shared_autonomy/manual_capture_and_fuse_real.py
"""
 
import ast
from collections import deque
import glob
import inspect
import os
import shutil
import struct
import sys
import termios
import threading
import time
import tty
 
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
 
try:
    from auto_scan_ring_and_fuse import (
        ensure_dir,
        fuse_scans,
        keep_largest_cluster,
        remove_plane,
    )
    HAS_FUSE = True
except ImportError:
    HAS_FUSE = False
 
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
 
    def remove_plane(pcd, dist):
        return pcd
 
    def keep_largest_cluster(pcd, eps, min_points):
        return pcd
 
try:
    import tf2_sensor_msgs.tf2_sensor_msgs as tf2sm
    HAS_TF2_SENSOR_MSGS = True
except Exception:
    HAS_TF2_SENSOR_MSGS = False
 
 
def _parse_float_list(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return []
    return out
 
 
def _header_stamp(header):
    if header.stamp is not None and header.stamp != rospy.Time():
        return header.stamp
    return rospy.Time.now()


def _decode_rgb_value(value):
    if value is None:
        return None
    try:
        if isinstance(value, float):
            packed = struct.unpack("I", struct.pack("f", value))[0]
        else:
            packed = int(value) & 0xFFFFFFFF
        r = (packed >> 16) & 0xFF
        g = (packed >> 8) & 0xFF
        b = packed & 0xFF
        return [r / 255.0, g / 255.0, b / 255.0]
    except Exception:
        return None


def _count_cloud_points(cloud_msg):
    count = 0
    for _ in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
        count += 1
    return count


def _normalize_output_dir(path):
    path = os.path.expanduser(str(path).strip())
    legacy_home = "/home/heyang"
    current_home = os.path.expanduser("~")
    if path == legacy_home or path.startswith(legacy_home + os.sep):
        suffix = path[len(legacy_home):].lstrip(os.sep)
        path = os.path.join(current_home, suffix) if suffix else current_home
    return path
 
 
class _RawTerminal:
    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None
 
    def __enter__(self):
        if not sys.stdin.isatty():
            rospy.logwarn("[manual_capture] stdin is not a TTY; use `s` + Enter, or publish to the command topic.")
            return self
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.enabled = True
        return self
 
    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
 
 
class _StdinCommandReader:
    def __init__(self):
        self._commands = deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._raw_terminal = _RawTerminal()
 
    def __enter__(self):
        self._raw_terminal.__enter__()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self
 
    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        self._raw_terminal.__exit__(exc_type, exc, tb)
        if self._thread is not None:
            self._thread.join(timeout=0.2)
 
    def _push(self, cmd):
        if not cmd:
            return
        with self._lock:
            self._commands.append(cmd[:1].lower())
 
    def pop(self):
        with self._lock:
            if not self._commands:
                return None
            return self._commands.popleft()
 
    def _reader_loop(self):
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            try:
                if self._raw_terminal.enabled:
                    ch = sys.stdin.read(1)
                    if ch:
                        self._push(ch)
                else:
                    line = sys.stdin.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    self._push(line.strip())
            except Exception:
                time.sleep(0.05)
 
 
class ManualCaptureAndFuseReal:
    def __init__(self):
        self.out_dir = _normalize_output_dir(rospy.get_param("~out_dir", "~/scans_manual_real"))
        self.cloud_topic = rospy.get_param("~cloud_topic", "/camera/depth/color/points")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.base_frame_candidates = self._parse_string_list(
            rospy.get_param("~base_frame_candidates", ["base", "base_link", "sawyer_base_link", "world"])
        )
        self.use_latest_tf_for_cloud = bool(rospy.get_param("~use_latest_tf_for_cloud", True))
        self.allow_untransformed_capture = bool(rospy.get_param("~allow_untransformed_capture", True))
        self.clear_out_dir_on_start = bool(rospy.get_param("~clear_out_dir_on_start", False))
        self.capture_in_sensor_frame = False
 
        self.require_fresh_cloud = bool(rospy.get_param("~require_fresh_cloud", True))
        self.fresh_cloud_timeout = float(rospy.get_param("~fresh_cloud_timeout", 2.0))
        self.cloud_stale_sec = float(rospy.get_param("~cloud_stale_sec", 1.0))
        self.capture_delay_sec = float(rospy.get_param("~capture_delay_sec", 0.15))
        self.take_multiple_frames = max(1, int(rospy.get_param("~take_multiple_frames", 3)))
        self.inter_frame_delay_sec = float(rospy.get_param("~inter_frame_delay_sec", 0.05))
 
        self.capture_voxel = float(rospy.get_param("~capture_voxel", rospy.get_param("~voxel", 0.0)))
        self.min_points = int(rospy.get_param("~min_points", 200))
        self.crop_box = _parse_float_list(rospy.get_param("~crop_box", []))
        if len(self.crop_box) != 6:
            self.crop_box = []
        self.fuse_crop_box = _parse_float_list(rospy.get_param("~fuse_crop_box", self.crop_box))
        if len(self.fuse_crop_box) != 6:
            self.fuse_crop_box = []
 
        self.capture_remove_plane = bool(rospy.get_param("~capture_remove_plane", True))
        self.capture_plane_dist = float(rospy.get_param("~capture_plane_dist", max(self.capture_voxel * 2.0, 0.004)))
        self.capture_keep_largest_cluster = bool(rospy.get_param("~capture_keep_largest_cluster", True))
        self.capture_cluster_eps = float(rospy.get_param("~capture_cluster_eps", max(self.capture_voxel * 8.0, 0.015)))
        self.capture_cluster_min_points = int(rospy.get_param("~capture_cluster_min_points", max(self.min_points, 150)))
        self.sensor_depth_window = _parse_float_list(rospy.get_param("~sensor_depth_window", []))
        if len(self.sensor_depth_window) != 2:
            self.sensor_depth_window = []
 
        ensure_dir(self.out_dir)
        if self.clear_out_dir_on_start:
            self._clear_output_dir()
 
        self.last_cloud = None
        self.last_cloud_stamp = rospy.Time(0)
        self.last_cloud_receipt = rospy.Time(0)
        self.capture_count = 0
        self.pending_commands = deque()
        self.last_capture_raw_counts = []
 
        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        self.command_topic = rospy.get_param("~command_topic", "~command")
        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
 
        self.capture_count = self._find_next_index()
 
        rospy.loginfo("[manual_capture] waiting for first cloud on %s ...", self.cloud_topic)
        self._wait_for_first_cloud()
        try:
            self.base_frame = self._resolve_base_frame(timeout=5.0)
        except RuntimeError as exc:
            if not self.allow_untransformed_capture:
                raise
            self.capture_in_sensor_frame = True
            self.base_frame = self.last_cloud.header.frame_id
            rospy.logwarn("[manual_capture] %s", str(exc))
            rospy.logwarn(
                "[manual_capture] No camera-to-robot TF available. Capturing in raw sensor frame '%s' and relying on pose-graph registration.",
                self.base_frame,
            )
            if len(self.crop_box) == 6:
                rospy.logwarn("[manual_capture] Disabling crop_box because it is defined in robot/base coordinates.")
                self.crop_box = []
 
        rospy.loginfo(
            "[manual_capture] ready. base=%s mode=%s cloud_topic=%s command_topic=%s out_dir=%s next_index=%d",
            self.base_frame,
            "sensor_frame" if self.capture_in_sensor_frame else "robot_base_tf",
            self.cloud_topic,
            rospy.resolve_name(self.command_topic),
            self.out_dir,
            self.capture_count,
        )
        if not HAS_TF2_SENSOR_MSGS:
            rospy.logwarn("[manual_capture] tf2_sensor_msgs not found; using slow point-wise cloud transform fallback.")
 
    def _cloud_cb(self, msg):
        self.last_cloud = msg
        self.last_cloud_stamp = _header_stamp(msg.header)
        self.last_cloud_receipt = rospy.Time.now()
 
    def _command_cb(self, msg):
        cmd = str(msg.data).strip().lower()[:1]
        if cmd in ("s", "f", "p", "h", "q"):
            self.pending_commands.append(cmd)
 
    def _find_next_index(self):
        files = sorted(glob.glob(os.path.join(self.out_dir, "scan_*.ply")))
        if not files:
            return 0
        last_name = os.path.basename(files[-1])
        try:
            prefix = last_name.split("_")[1]
            return int(prefix) + 1
        except Exception:
            return len(files)
 
    def _clear_output_dir(self):
        patterns = [
            "scan_*.ply",
            "merged_posegraph_before_drop.ply",
            "merged_posegraph_after_drop.ply",
        ]
        removed = 0
        for pattern in patterns:
            for path in glob.glob(os.path.join(self.out_dir, pattern)):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        removed += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        removed += 1
                except Exception as exc:
                    rospy.logwarn("[manual_capture] failed to remove %s: %s", path, str(exc))
        if removed > 0:
            rospy.loginfo("[manual_capture] cleared %d previous scan/fusion files from %s", removed, self.out_dir)
 
    def _wait_for_first_cloud(self, timeout=10.0):
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout:
            if self.last_cloud is not None and self.last_cloud.header.frame_id:
                return
            rospy.sleep(0.05)
        raise RuntimeError("Timed out waiting for first PointCloud2 message.")
 
    def _parse_string_list(self, value):
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    value = parsed
                else:
                    value = [value]
            except Exception:
                value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item).strip() for item in value if str(item).strip()]
 
    def _lookup_transform(self, target_frame, source_frame, stamp, timeout_sec=0.6, use_latest=False):
        query_time = rospy.Time(0) if use_latest else stamp
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            query_time,
            rospy.Duration(timeout_sec),
        )
 
    def _can_transform(self, target_frame, source_frame, timeout_sec=0.2):
        try:
            return self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(timeout_sec),
            )
        except Exception:
            return False
 
    def _resolve_base_frame(self, timeout=5.0):
        source_frame = self.last_cloud.header.frame_id
        candidates = []
        for frame in [self.base_frame] + self.base_frame_candidates:
            if frame and frame not in candidates:
                candidates.append(frame)
 
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout:
            for frame in candidates:
                if self._can_transform(frame, source_frame, timeout_sec=0.15):
                    if frame != self.base_frame:
                        rospy.logwarn(
                            "[manual_capture] requested base_frame '%s' is disconnected from '%s'; using '%s' instead.",
                            self.base_frame,
                            source_frame,
                            frame,
                        )
                    return frame
            rospy.sleep(0.1)
 
        raise RuntimeError(
            "No TF path from cloud frame '%s' to any candidate base frame %s. "
            "Publish the camera extrinsic or pass a connected _base_frame:=..."
            % (source_frame, candidates)
        )
 
    def _wait_for_fresh_cloud(self, after_stamp):
        rate = rospy.Rate(30.0)
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < self.fresh_cloud_timeout:
            if self.last_cloud is None:
                rate.sleep()
                continue
 
            cloud_stamp = self.last_cloud_stamp
            cloud_age = (rospy.Time.now() - self.last_cloud_receipt).to_sec()
 
            if self.require_fresh_cloud and cloud_stamp <= after_stamp:
                rate.sleep()
                continue
 
            if cloud_age > self.cloud_stale_sec:
                rate.sleep()
                continue
 
            return self.last_cloud
 
        if self.require_fresh_cloud:
            raise RuntimeError("Timed out waiting for a fresh RealSense point cloud.")
        if self.last_cloud is None:
            raise RuntimeError("No RealSense point cloud available.")
        return self.last_cloud
 
    def _transform_cloud_to_base(self, cloud_msg):
        field_names = [field.name for field in cloud_msg.fields]
        color_field = "rgb" if "rgb" in field_names else ("rgba" if "rgba" in field_names else None)

        if self.capture_in_sensor_frame:
            pts = []
            colors = []
            read_fields = ("x", "y", "z", color_field) if color_field is not None else ("x", "y", "z")
            for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=read_fields):
                pts.append([point[0], point[1], point[2]])
                if color_field is not None:
                    rgb = _decode_rgb_value(point[3])
                    colors.append(rgb if rgb is not None else [0.0, 0.0, 0.0])
            pts = np.asarray(pts, dtype=np.float64)
            if len(pts) < self.min_points:
                return None
            if color_field is not None and len(colors) == len(pts):
                return {
                    "points": pts,
                    "colors": np.asarray(colors, dtype=np.float64),
                }
            return pts

        try:
            tf_msg = self._lookup_transform(
                self.base_frame,
                cloud_msg.header.frame_id,
                _header_stamp(cloud_msg.header),
                timeout_sec=0.6,
                use_latest=self.use_latest_tf_for_cloud,
            )
        except Exception as exc:
            rospy.logwarn("[manual_capture] cloud TF lookup failed: %s", str(exc))
            return None
 
        if HAS_TF2_SENSOR_MSGS:
            try:
                return tf2sm.do_transform_cloud(cloud_msg, tf_msg)
            except Exception as exc:
                rospy.logwarn("[manual_capture] tf2_sensor_msgs failed, using point-wise fallback: %s", str(exc))

        pts = []
        colors = []
        read_fields = ("x", "y", "z", color_field) if color_field is not None else None
        iterator = (
            pc2.read_points(cloud_msg, skip_nans=True, field_names=read_fields)
            if read_fields is not None
            else pc2.read_points(cloud_msg, skip_nans=True)
        )
        for point in iterator:
            stamped = PointStamped()
            stamped.header = cloud_msg.header
            stamped.point.x = point[0]
            stamped.point.y = point[1]
            stamped.point.z = point[2]
            world_pt = tf2_geometry_msgs.do_transform_point(stamped, tf_msg)
            pts.append([world_pt.point.x, world_pt.point.y, world_pt.point.z])
            if color_field is not None:
                rgb = _decode_rgb_value(point[3])
                colors.append(rgb if rgb is not None else [0.0, 0.0, 0.0])

        if len(pts) < self.min_points:
            return None
        pts = np.asarray(pts, dtype=np.float64)
        if color_field is not None and len(colors) == len(pts):
            return {
                "points": pts,
                "colors": np.asarray(colors, dtype=np.float64),
            }
        return pts

    def _cloud_to_o3d(self, cloud_msg_or_pts):
        colors = None
        if isinstance(cloud_msg_or_pts, dict):
            pts = np.asarray(cloud_msg_or_pts.get("points", []), dtype=np.float64)
            color_arr = cloud_msg_or_pts.get("colors")
            if color_arr is not None:
                colors = np.asarray(color_arr, dtype=np.float64)
                if colors.shape[0] != pts.shape[0]:
                    colors = None
        elif isinstance(cloud_msg_or_pts, np.ndarray):
            pts = cloud_msg_or_pts
        else:
            field_names = [field.name for field in cloud_msg_or_pts.fields]
            color_field = "rgb" if "rgb" in field_names else ("rgba" if "rgba" in field_names else None)
            pts = []
            color_list = []
            read_fields = ("x", "y", "z", color_field) if color_field is not None else ("x", "y", "z")
            for point in pc2.read_points(cloud_msg_or_pts, skip_nans=True, field_names=read_fields):
                pts.append([point[0], point[1], point[2]])
                if color_field is not None:
                    rgb = _decode_rgb_value(point[3])
                    color_list.append(rgb if rgb is not None else [0.0, 0.0, 0.0])
            pts = np.asarray(pts, dtype=np.float64)
            if color_field is not None and len(color_list) == len(pts):
                colors = np.asarray(color_list, dtype=np.float64)
            if pts.shape[0] < self.min_points:
                return None

        if pts.shape[0] < self.min_points:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None and colors.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)
 
        if len(self.crop_box) == 6:
            xmin, xmax, ymin, ymax, zmin, zmax = self.crop_box
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(xmin, ymin, zmin),
                max_bound=(xmax, ymax, zmax),
            )
            pcd = pcd.crop(aabb)
 
        if self.capture_voxel > 0.0:
            pcd = pcd.voxel_down_sample(self.capture_voxel)
 
        if len(pcd.points) < self.min_points:
            return None
        return pcd
 
    def _refine_capture_cloud(self, pcd):
        if pcd is None or len(pcd.points) < self.min_points:
            return pcd
 
        if self.capture_in_sensor_frame and len(self.sensor_depth_window) == 2:
            zmin, zmax = sorted(self.sensor_depth_window)
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(-10.0, -10.0, zmin),
                max_bound=(10.0, 10.0, zmax),
            )
            pcd = pcd.crop(aabb)
 
        if self.capture_remove_plane:
            pcd = remove_plane(pcd, self.capture_plane_dist)
 
        if self.capture_keep_largest_cluster:
            pcd = keep_largest_cluster(
                pcd,
                eps=self.capture_cluster_eps,
                min_points=self.capture_cluster_min_points,
            )
 
        if self.capture_voxel > 0.0 and len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(self.capture_voxel)
 
        if len(pcd.points) < self.min_points:
            return None
        return pcd
 
    def _capture_burst_clouds(self):
        captured = []
        raw_counts = []
        gate_stamp = rospy.Time.now()
        for idx in range(self.take_multiple_frames):
            if idx > 0 and self.inter_frame_delay_sec > 0.0:
                rospy.sleep(self.inter_frame_delay_sec)
            cloud = self._wait_for_fresh_cloud(gate_stamp)
            raw_counts.append(_count_cloud_points(cloud))
            gate_stamp = self.last_cloud_stamp
            transformed = self._transform_cloud_to_base(cloud)
            if transformed is None:
                raise RuntimeError("Failed to transform cloud into base frame.")
            captured.append(transformed)
        self.last_capture_raw_counts = raw_counts
        return captured
 
    def capture_once(self):
        if self.capture_delay_sec > 0.0:
            rospy.sleep(self.capture_delay_sec)

        transformed_clouds = self._capture_burst_clouds()
        raw_counts = list(self.last_capture_raw_counts)
        transformed_counts = []
        for transformed in transformed_clouds:
            if isinstance(transformed, dict):
                transformed_counts.append(int(np.asarray(transformed.get("points", [])).shape[0]))
            elif isinstance(transformed, np.ndarray):
                transformed_counts.append(int(transformed.shape[0]))
            else:
                transformed_counts.append(_count_cloud_points(transformed))

        if len(transformed_clouds) == 1:
            pcd = self._cloud_to_o3d(transformed_clouds[0])
        else:
            merged = o3d.geometry.PointCloud()
            for transformed in transformed_clouds:
                pcd_part = self._cloud_to_o3d(transformed)
                if pcd_part is None:
                    continue
                merged += pcd_part
            pcd = merged if len(merged.points) >= self.min_points else None
        pre_refine_points = len(pcd.points) if pcd is not None else 0

        pcd = self._refine_capture_cloud(pcd)
        if pcd is None:
            raise RuntimeError("Cropped/downsampled cloud has too few points.")

        out_path = os.path.join(self.out_dir, f"scan_{self.capture_count:04d}.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        rospy.loginfo(
            "[manual_capture] saved %s (points=%d, merged_frames=%d, capture_voxel=%.4f)",
            out_path,
            len(pcd.points),
            len(transformed_clouds),
            self.capture_voxel,
        )
        rospy.loginfo(
            "[manual_capture] capture stats raw=%s transformed=%s pre_refine=%d post_refine=%d "
            "remove_plane=%s keep_largest_cluster=%s",
            raw_counts if raw_counts else "n/a",
            transformed_counts,
            pre_refine_points,
            len(pcd.points),
            self.capture_remove_plane,
            self.capture_keep_largest_cluster,
        )
        self.capture_count += 1
        return out_path
 
    def _make_fuse_kwargs(self):
        signature = inspect.signature(fuse_scans)
        params = signature.parameters
 
        raw_crop_box = self.fuse_crop_box if len(self.fuse_crop_box) == 6 else None
        if self.capture_in_sensor_frame and raw_crop_box is not None:
            rospy.logwarn("[manual_capture] Disabling fuse raw crop box because captures are in sensor coordinates.")
            raw_crop_box = None
 
        kwargs = {
            "ply_dir": self.out_dir,
            "pattern": str(rospy.get_param("~fuse_pattern", "scan_*.ply")),
            "out_before": str(rospy.get_param("~out_before", "merged_posegraph_before_drop.ply")),
            "out_after": str(rospy.get_param("~out_after", "merged_posegraph_after_drop.ply")),
            "fusion_strategy": str(
                rospy.get_param("~fusion_strategy", "posegraph" if self.capture_in_sensor_frame else "base_icp")
            ),
            "voxel_reg": float(rospy.get_param("~voxel_reg", 0.005)),
            "voxel_final": float(rospy.get_param("~voxel_final", 0.003)),
            "fitness_gate": float(rospy.get_param("~fitness_gate", 0.30)),
            "rmse_gate": float(rospy.get_param("~rmse_gate", 0.015)),
            "loop_k": int(rospy.get_param("~loop_k", 3 if self.capture_in_sensor_frame else 6)),
            "loop_fitness_gate": float(rospy.get_param("~loop_fitness_gate", 0.40)),
            "loop_rmse_gate": float(rospy.get_param("~loop_rmse_gate", 0.012)),
            "opt_max_corr": float(rospy.get_param("~opt_max_corr", 0.02)),
            "drop_k": int(rospy.get_param("~drop_k", 2 if self.capture_in_sensor_frame else 1)),
            "drop_fitness_th": float(rospy.get_param("~drop_fitness_th", 0.35)),
            "drop_rmse_th": float(rospy.get_param("~drop_rmse_th", 0.012)),
            "reg_remove_plane": bool(rospy.get_param("~reg_remove_plane", True)),
            "reg_keep_largest_cluster": bool(rospy.get_param("~reg_keep_largest_cluster", True)),
            "final_remove_outlier": bool(rospy.get_param("~final_remove_outlier", True)),
            "final_outlier_nb_neighbors": int(rospy.get_param("~final_outlier_nb_neighbors", 30)),
            "final_outlier_std_ratio": float(rospy.get_param("~final_outlier_std_ratio", 2.5)),
            "verbose": bool(rospy.get_param("~fuse_verbose", True)),
        }
 
        if "raw_crop_box" in params:
            kwargs["raw_crop_box"] = raw_crop_box
        if "final_use_registration_clouds" in params:
            kwargs["final_use_registration_clouds"] = bool(
                rospy.get_param("~final_use_registration_clouds", self.capture_in_sensor_frame)
            )
        return kwargs
 
    def fuse(self):
        if not HAS_FUSE:
            raise RuntimeError("Fusion module not available. Please install auto_scan_ring_and_fuse.")
 
        files = sorted(glob.glob(os.path.join(self.out_dir, "scan_*.ply")))
        if len(files) < 2:
            raise RuntimeError("Need at least 2 saved scans before fusion.")
 
        rospy.loginfo("[manual_capture] starting fusion with %d scans ...", len(files))
        before_path, after_path = fuse_scans(**self._make_fuse_kwargs())
        rospy.loginfo("[manual_capture] fusion done. before=%s after=%s", before_path, after_path)
        return before_path, after_path
 
    @staticmethod
    def print_help():
        print(
            "\nManual Capture Controls\n"
            "  s : save one point-cloud frame\n"
            "  f : fuse saved frames\n"
            "  p : print saved-scan count\n"
            "  h : print this help\n"
            "  q : quit\n"
            "\nROS command topic:\n"
            "  rostopic pub -1 /manual_capture_and_fuse_real/command std_msgs/String 's'\n"
        )
 
    def run(self):
        self.print_help()
        print(f"Listening on {self.cloud_topic}, saving into {self.out_dir}")
 
        with _StdinCommandReader() as terminal:
            while not rospy.is_shutdown():
                key = self.pending_commands.popleft() if self.pending_commands else terminal.pop()
                if key is None:
                    rospy.sleep(0.05)
                    continue
 
                key = key.lower()
                if key == "s":
                    try:
                        self.capture_once()
                    except Exception as exc:
                        rospy.logwarn("[manual_capture] capture failed: %s", str(exc))
                elif key == "f":
                    try:
                        self.fuse()
                    except Exception as exc:
                        rospy.logwarn("[manual_capture] fusion failed: %s", str(exc))
                elif key == "p":
                    print(f"Saved scans: {self.capture_count}")
                elif key == "h":
                    self.print_help()
                elif key == "q":
                    rospy.loginfo("[manual_capture] quitting.")
                    return
 
 
def main():
    node = ManualCaptureAndFuseReal()
    node.run()
 
 
if __name__ == "__main__":
    rospy.init_node("manual_capture_and_fuse_real")
    main()
 
 
