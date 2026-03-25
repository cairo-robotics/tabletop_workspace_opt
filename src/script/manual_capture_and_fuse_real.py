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
"""

import ast
import glob
import os
import select
import sys
import termios
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

from auto_scan_ring_and_fuse import ensure_dir, fuse_scans

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


class _RawTerminal:
    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            rospy.logwarn("[manual_capture] stdin is not a TTY; falling back to line input.")
            return self
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout_sec=0.1):
        if not self.enabled:
            if timeout_sec > 0.0:
                time.sleep(timeout_sec)
            return None
        ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
        if not ready:
            return None
        return sys.stdin.read(1)


class ManualCaptureAndFuseReal:
    def __init__(self):
        self.out_dir = os.path.expanduser(rospy.get_param("~out_dir", "~/scans_manual_real"))
        self.cloud_topic = rospy.get_param("~cloud_topic", "/camera/depth/color/points")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.use_latest_tf_for_cloud = bool(rospy.get_param("~use_latest_tf_for_cloud", False))

        self.require_fresh_cloud = bool(rospy.get_param("~require_fresh_cloud", True))
        self.fresh_cloud_timeout = float(rospy.get_param("~fresh_cloud_timeout", 2.0))
        self.cloud_stale_sec = float(rospy.get_param("~cloud_stale_sec", 1.0))
        self.capture_delay_sec = float(rospy.get_param("~capture_delay_sec", 0.15))

        self.voxel = float(rospy.get_param("~voxel", 0.0))
        self.min_points = int(rospy.get_param("~min_points", 200))
        self.crop_box = _parse_float_list(rospy.get_param("~crop_box", []))
        if len(self.crop_box) != 6:
            self.crop_box = []

        ensure_dir(self.out_dir)

        self.last_cloud = None
        self.last_cloud_stamp = rospy.Time(0)
        self.last_cloud_receipt = rospy.Time(0)
        self.capture_count = 0

        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.capture_count = self._find_next_index()

        rospy.loginfo("[manual_capture] waiting for first cloud on %s ...", self.cloud_topic)
        self._wait_for_first_cloud()
        rospy.loginfo(
            "[manual_capture] ready. base=%s cloud_topic=%s out_dir=%s next_index=%d",
            self.base_frame,
            self.cloud_topic,
            self.out_dir,
            self.capture_count,
        )
        if not HAS_TF2_SENSOR_MSGS:
            rospy.logwarn("[manual_capture] tf2_sensor_msgs not found; using slow point-wise cloud transform fallback.")

    def _cloud_cb(self, msg):
        self.last_cloud = msg
        self.last_cloud_stamp = _header_stamp(msg.header)
        self.last_cloud_receipt = rospy.Time.now()

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

    def _wait_for_first_cloud(self, timeout=10.0):
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout:
            if self.last_cloud is not None:
                return
            rospy.sleep(0.05)
        raise RuntimeError("Timed out waiting for first PointCloud2 message.")

    def _lookup_transform(self, target_frame, source_frame, stamp, timeout_sec=0.6, use_latest=False):
        query_time = rospy.Time(0) if use_latest else stamp
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            query_time,
            rospy.Duration(timeout_sec),
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
        for point in pc2.read_points(cloud_msg, skip_nans=True):
            stamped = PointStamped()
            stamped.header = cloud_msg.header
            stamped.point.x = point[0]
            stamped.point.y = point[1]
            stamped.point.z = point[2]
            world_pt = tf2_geometry_msgs.do_transform_point(stamped, tf_msg)
            pts.append([world_pt.point.x, world_pt.point.y, world_pt.point.z])

        if len(pts) < self.min_points:
            return None
        return np.asarray(pts, dtype=np.float64)

    def _cloud_to_o3d(self, cloud_msg_or_pts):
        if isinstance(cloud_msg_or_pts, np.ndarray):
            pts = cloud_msg_or_pts
        else:
            pts = []
            for point in pc2.read_points(cloud_msg_or_pts, skip_nans=True, field_names=("x", "y", "z")):
                pts.append([point[0], point[1], point[2]])
            pts = np.asarray(pts, dtype=np.float64)

        if pts.shape[0] < self.min_points:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if len(self.crop_box) == 6:
            xmin, xmax, ymin, ymax, zmin, zmax = self.crop_box
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(xmin, ymin, zmin),
                max_bound=(xmax, ymax, zmax),
            )
            pcd = pcd.crop(aabb)

        if self.voxel > 0.0:
            pcd = pcd.voxel_down_sample(self.voxel)

        if len(pcd.points) < self.min_points:
            return None
        return pcd

    def capture_once(self):
        if self.capture_delay_sec > 0.0:
            rospy.sleep(self.capture_delay_sec)
        gate_stamp = rospy.Time.now()
        cloud = self._wait_for_fresh_cloud(gate_stamp)
        transformed = self._transform_cloud_to_base(cloud)
        if transformed is None:
            raise RuntimeError("Transformed cloud is empty or too small.")

        pcd = self._cloud_to_o3d(transformed)
        if pcd is None:
            raise RuntimeError("Cropped/downsampled cloud has too few points.")

        out_path = os.path.join(self.out_dir, f"scan_{self.capture_count:04d}.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        rospy.loginfo("[manual_capture] saved %s (points=%d)", out_path, len(pcd.points))
        self.capture_count += 1
        return out_path

    def fuse(self):
        files = sorted(glob.glob(os.path.join(self.out_dir, "scan_*.ply")))
        if len(files) < 2:
            raise RuntimeError("Need at least 2 saved scans before fusion.")

        rospy.loginfo("[manual_capture] starting fusion with %d scans ...", len(files))
        before_path, after_path = fuse_scans(
            ply_dir=self.out_dir,
            pattern=str(rospy.get_param("~fuse_pattern", "scan_*.ply")),
            out_before=str(rospy.get_param("~out_before", "merged_posegraph_before_drop.ply")),
            out_after=str(rospy.get_param("~out_after", "merged_posegraph_after_drop.ply")),
            voxel_reg=float(rospy.get_param("~voxel_reg", 0.005)),
            voxel_final=float(rospy.get_param("~voxel_final", 0.001)),
            fitness_gate=float(rospy.get_param("~fitness_gate", 0.30)),
            rmse_gate=float(rospy.get_param("~rmse_gate", 0.015)),
            loop_k=int(rospy.get_param("~loop_k", 0)),
            loop_fitness_gate=float(rospy.get_param("~loop_fitness_gate", 0.40)),
            loop_rmse_gate=float(rospy.get_param("~loop_rmse_gate", 0.012)),
            opt_max_corr=float(rospy.get_param("~opt_max_corr", 0.02)),
            drop_k=int(rospy.get_param("~drop_k", 0)),
            drop_fitness_th=float(rospy.get_param("~drop_fitness_th", 0.35)),
            drop_rmse_th=float(rospy.get_param("~drop_rmse_th", 0.012)),
            reg_remove_plane=bool(rospy.get_param("~reg_remove_plane", True)),
            reg_keep_largest_cluster=bool(rospy.get_param("~reg_keep_largest_cluster", False)),
            final_remove_outlier=bool(rospy.get_param("~final_remove_outlier", False)),
            final_outlier_nb_neighbors=int(rospy.get_param("~final_outlier_nb_neighbors", 30)),
            final_outlier_std_ratio=float(rospy.get_param("~final_outlier_std_ratio", 2.5)),
            raw_crop_box=_parse_float_list(rospy.get_param("~fuse_crop_box", self.crop_box)),
            verbose=bool(rospy.get_param("~fuse_verbose", True)),
        )
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
        )

    def run(self):
        self.print_help()
        print(f"Listening on {self.cloud_topic}, saving into {self.out_dir}")

        with _RawTerminal() as terminal:
            while not rospy.is_shutdown():
                key = terminal.read_key(timeout_sec=0.1)
                if key is None:
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
