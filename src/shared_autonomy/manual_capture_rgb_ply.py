#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual capture of raw RGB point clouds from RealSense and save them as PLY files.

This script intentionally does not do TF transforms, filtering, cropping, or fusion.
It saves the raw point cloud in the camera frame, preserving RGB when available.

Controls:
- press `s` to save one fresh RGB PLY
- press `p` to print saved count
- press `q` to quit

ROS fallback command topic:
- rostopic pub -1 /manual_capture_rgb_ply/command std_msgs/String "data: 's'"
"""

import glob
import os
import struct
import sys
import termios
import threading
import time
import tty
from collections import deque

import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


class _RawTerminal:
    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            rospy.logwarn("[rgb_capture] stdin is not a TTY; use `s` + Enter, or publish to the command topic.")
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


class ManualCaptureRgbPly:
    def __init__(self):
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.out_dir = os.path.expanduser(
            rospy.get_param("~out_dir", os.path.join(package_root, "captured_rgb_ply"))
        )
        self.cloud_topic = rospy.get_param("~cloud_topic", "/camera/depth/color/points")
        self.require_fresh_cloud = bool(rospy.get_param("~require_fresh_cloud", True))
        self.fresh_cloud_timeout = float(rospy.get_param("~fresh_cloud_timeout", 2.0))
        self.cloud_stale_sec = float(rospy.get_param("~cloud_stale_sec", 1.0))
        self.capture_delay_sec = float(rospy.get_param("~capture_delay_sec", 0.05))
        self.min_points = int(rospy.get_param("~min_points", 100))
        self.clear_out_dir_on_start = bool(rospy.get_param("~clear_out_dir_on_start", False))

        ensure_dir(self.out_dir)
        if self.clear_out_dir_on_start:
            self._clear_output_dir()

        self.last_cloud = None
        self.last_cloud_stamp = rospy.Time(0)
        self.last_cloud_receipt = rospy.Time(0)
        self.capture_count = 0
        self.pending_commands = deque()

        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        self.command_topic = rospy.get_param("~command_topic", "~command")
        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=10)

        self.capture_count = self._find_next_index()

        rospy.loginfo("[rgb_capture] waiting for first cloud on %s ...", self.cloud_topic)
        self._wait_for_first_cloud()
        rospy.loginfo(
            "[rgb_capture] ready. cloud_topic=%s command_topic=%s out_dir=%s next_index=%d",
            self.cloud_topic,
            rospy.resolve_name(self.command_topic),
            self.out_dir,
            self.capture_count,
        )

    def _cloud_cb(self, msg):
        self.last_cloud = msg
        self.last_cloud_stamp = _header_stamp(msg.header)
        self.last_cloud_receipt = rospy.Time.now()

    def _command_cb(self, msg):
        cmd = str(msg.data).strip().lower()[:1]
        if cmd in ("s", "p", "q", "h"):
            self.pending_commands.append(cmd)

    def _find_next_index(self):
        files = sorted(glob.glob(os.path.join(self.out_dir, "scan_rgb_*.ply")))
        if not files:
            return 0
        last_name = os.path.basename(files[-1])
        try:
            return int(last_name.split("_")[-1].split(".")[0]) + 1
        except Exception:
            return len(files)

    def _clear_output_dir(self):
        removed = 0
        for path in glob.glob(os.path.join(self.out_dir, "scan_rgb_*.ply")):
            try:
                os.remove(path)
                removed += 1
            except Exception as exc:
                rospy.logwarn("[rgb_capture] failed to remove %s: %s", path, str(exc))
        if removed > 0:
            rospy.loginfo("[rgb_capture] cleared %d previous RGB PLY files from %s", removed, self.out_dir)

    def _wait_for_first_cloud(self, timeout=10.0):
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout:
            if self.last_cloud is not None and self.last_cloud.header.frame_id:
                return
            rospy.sleep(0.05)
        raise RuntimeError("Timed out waiting for first PointCloud2 message.")

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

    def _cloud_to_o3d(self, cloud_msg):
        field_names = [field.name for field in cloud_msg.fields]
        color_field = "rgb" if "rgb" in field_names else ("rgba" if "rgba" in field_names else None)
        read_fields = ("x", "y", "z", color_field) if color_field is not None else ("x", "y", "z")

        points = []
        colors = []
        has_valid_color = color_field is not None

        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=read_fields):
            points.append([point[0], point[1], point[2]])
            if color_field is not None:
                rgb = _decode_rgb_value(point[3])
                if rgb is None:
                    has_valid_color = False
                else:
                    colors.append(rgb)

        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] < self.min_points:
            return None, False

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if has_valid_color and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        return pcd, pcd.has_colors()

    def capture_once(self):
        if self.capture_delay_sec > 0.0:
            rospy.sleep(self.capture_delay_sec)
        gate_stamp = rospy.Time.now()
        cloud = self._wait_for_fresh_cloud(gate_stamp)
        pcd, has_color = self._cloud_to_o3d(cloud)
        if pcd is None:
            raise RuntimeError("Raw point cloud has too few valid points.")

        out_path = os.path.join(self.out_dir, f"scan_rgb_{self.capture_count:04d}.ply")
        ok = o3d.io.write_point_cloud(out_path, pcd, write_ascii=False, compressed=False)
        if not ok:
            raise RuntimeError("Failed to write PLY file.")

        rospy.loginfo(
            "[rgb_capture] saved %s (points=%d, has_rgb=%s, frame=%s)",
            out_path,
            len(pcd.points),
            "true" if has_color else "false",
            cloud.header.frame_id,
        )
        self.capture_count += 1
        return out_path

    @staticmethod
    def print_help():
        print(
            "\nManual RGB PLY Capture Controls\n"
            "  s : save one raw RGB point cloud\n"
            "  p : print saved-scan count\n"
            "  h : print this help\n"
            "  q : quit\n"
            "\nROS command topic:\n"
            "  rostopic pub -1 /manual_capture_rgb_ply/command std_msgs/String \"data: 's'\"\n"
        )

    def run(self):
        self.print_help()
        print(f"Listening on {self.cloud_topic}, saving raw RGB PLYs into {self.out_dir}")

        with _StdinCommandReader() as terminal:
            while not rospy.is_shutdown():
                key = self.pending_commands.popleft() if self.pending_commands else terminal.pop()
                if key is None:
                    rospy.sleep(0.05)
                    continue

                if key == "s":
                    try:
                        self.capture_once()
                    except Exception as exc:
                        rospy.logwarn("[rgb_capture] capture failed: %s", str(exc))
                elif key == "p":
                    print(f"Saved scans: {self.capture_count}")
                elif key == "h":
                    self.print_help()
                elif key == "q":
                    rospy.loginfo("[rgb_capture] quitting.")
                    return


def main():
    node = ManualCaptureRgbPly()
    node.run()


if __name__ == "__main__":
    rospy.init_node("manual_capture_rgb_ply")
    main()
