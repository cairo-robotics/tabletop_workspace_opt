#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto ring scan for real Sawyer + wrist-mounted RealSense, followed by Open3D
pose-graph fusion.

This script keeps the original scan-and-fuse structure but adapts the hardware
interface for a real robot:
- motion command: publishes RelaxedIK `EEVelGoals`
- EE feedback: uses TF or Sawyer `EndpointState`
- optional orientation hold: keeps the end effector in a top-down pose
- cloud capture: waits for fresh RealSense PointCloud2 frames before saving

Typical use:
  1. bring up Sawyer + RelaxedIK + RealSense + hand-eye TF
  2. move the robot to a safe "scan center" pose
  3. run this script to traverse an XY ring and save `scan_*.ply`
  4. optionally fuse the saved frames into `merged_posegraph_*.ply`
"""

import ast
import math
import os
import time

import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEVelGoals
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2

from auto_scan_ring_and_fuse import clamp, ensure_dir, fuse_scans

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


def _normalize_quat_xyzw(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-9:
        raise ValueError("Quaternion norm is too small.")
    return q / n


class RealAutoRingScanAndFuse:
    def __init__(self):
        # IO/topics
        self.out_dir = os.path.expanduser(rospy.get_param("~out_dir", "~/scans_real"))
        self.cloud_topic = rospy.get_param("~cloud_topic", "/camera/depth/color/points")
        self.vel_topic = rospy.get_param("~vel_topic", "/relaxed_ik/ee_vel_goals")
        self.endpoint_topic = rospy.get_param("~endpoint_topic", "/robot/limb/right/endpoint_state")

        # Frames and pose source
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.tip_frame = rospy.get_param("~tip_frame", "right_hand")
        self.pose_source = str(rospy.get_param("~pose_source", "auto")).strip().lower()
        self.use_latest_tf_for_pose = bool(rospy.get_param("~use_latest_tf_for_pose", True))
        self.use_latest_tf_for_cloud = bool(rospy.get_param("~use_latest_tf_for_cloud", False))

        # Ring path
        self.radius = float(rospy.get_param("~radius", 0.08))
        self.n_poses = int(rospy.get_param("~n_poses", 12))
        self.z_offset = float(rospy.get_param("~z_offset", 0.0))
        self.start_angle_deg = float(rospy.get_param("~start_angle_deg", 0.0))
        self.scan_arc_deg = float(rospy.get_param("~scan_arc_deg", 360.0))
        self.center_xyz = _parse_float_list(rospy.get_param("~center_xyz", []))
        self.center_offset_xyz = _parse_float_list(rospy.get_param("~center_offset_xyz", [0.0, 0.0, 0.0]))
        if len(self.center_offset_xyz) != 3:
            self.center_offset_xyz = [0.0, 0.0, 0.0]

        # Position controller
        self.rate_hz = float(rospy.get_param("~rate", 60.0))
        self.kp_pos = float(rospy.get_param("~kp_pos", 1.2))
        self.vmax = float(rospy.get_param("~vmax", 0.025))
        self.pos_tol = float(rospy.get_param("~pos_tol", 0.008))
        self.settle_time = float(rospy.get_param("~settle_time", 0.35))
        self.timeout = float(rospy.get_param("~timeout", 12.0))
        self.return_to_start = bool(rospy.get_param("~return_to_start", True))

        # Orientation controller
        self.orientation_mode = str(rospy.get_param("~orientation_mode", "topdown")).strip().lower()
        self.kp_rot = float(rospy.get_param("~kp_rot", 2.0))
        self.wmax = float(rospy.get_param("~wmax", 0.6))
        self.ang_tol = math.radians(float(rospy.get_param("~ang_tol_deg", 6.0)))
        self.topdown_roll_deg = float(rospy.get_param("~topdown_roll_deg", 180.0))
        self.topdown_pitch_deg = float(rospy.get_param("~topdown_pitch_deg", 0.0))
        self.topdown_yaw_deg = float(rospy.get_param("~topdown_yaw_deg", 0.0))
        self.custom_target_quat_xyzw = _parse_float_list(rospy.get_param("~target_quat_xyzw", []))
        if self.orientation_mode in ("off", "free", "none"):
            self.control_orientation = False
        else:
            self.control_orientation = True

        # Capture
        self.take_multiple_frames = int(rospy.get_param("~take_multiple_frames", 1))
        self.capture_spacing = float(rospy.get_param("~capture_spacing", 0.10))
        self.capture_wait_after_stop = float(rospy.get_param("~capture_wait_after_stop", 0.20))
        self.require_fresh_cloud = bool(rospy.get_param("~require_fresh_cloud", True))
        self.fresh_cloud_timeout = float(rospy.get_param("~fresh_cloud_timeout", 2.0))
        self.cloud_stale_sec = float(rospy.get_param("~cloud_stale_sec", 1.0))
        self.voxel = float(rospy.get_param("~voxel", 0.003))
        self.min_points = int(rospy.get_param("~min_points", 200))
        self.crop_box = _parse_float_list(rospy.get_param("~crop_box", []))
        if len(self.crop_box) != 6:
            self.crop_box = []

        ensure_dir(self.out_dir)

        self.pub_vel = rospy.Publisher(self.vel_topic, EEVelGoals, queue_size=1)

        self.last_cloud = None
        self.last_cloud_stamp = rospy.Time(0)
        self.last_cloud_receipt = rospy.Time(0)
        self.last_endpoint = None
        self.last_endpoint_stamp = rospy.Time(0)

        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        rospy.Subscriber(self.endpoint_topic, EndpointState, self._endpoint_cb, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("[real_ring_scan] waiting for first cloud on %s ...", self.cloud_topic)
        self._wait_for_first_cloud()
        self._wait_for_pose_feedback()

        self.start_xyz, self.start_quat_xyzw = self._get_ee_pose()
        self.scan_center_xyz = self._resolve_scan_center()
        self.scan_quat_xyzw = self._resolve_scan_orientation(self.start_quat_xyzw)

        rospy.loginfo(
            "[real_ring_scan] pose_source=%s base=%s tip=%s center=%s radius=%.3f n_poses=%d",
            self.pose_source,
            self.base_frame,
            self.tip_frame,
            np.round(self.scan_center_xyz, 4),
            self.radius,
            self.n_poses,
        )
        rospy.loginfo(
            "[real_ring_scan] orientation_mode=%s control_orientation=%s target_quat=%s",
            self.orientation_mode,
            self.control_orientation,
            None if self.scan_quat_xyzw is None else np.round(self.scan_quat_xyzw, 4).tolist(),
        )
        rospy.loginfo(
            "[real_ring_scan] cloud_topic=%s endpoint_topic=%s out_dir=%s",
            self.cloud_topic,
            self.endpoint_topic,
            self.out_dir,
        )

        if not HAS_TF2_SENSOR_MSGS:
            rospy.logwarn("[real_ring_scan] tf2_sensor_msgs not found; cloud transforms will fall back to point-wise conversion.")

    def _cloud_cb(self, msg):
        self.last_cloud = msg
        self.last_cloud_stamp = _header_stamp(msg.header)
        self.last_cloud_receipt = rospy.Time.now()

    def _endpoint_cb(self, msg):
        self.last_endpoint = msg
        self.last_endpoint_stamp = _header_stamp(msg.header)

    def _wait_for_first_cloud(self, timeout=10.0):
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout:
            if self.last_cloud is not None:
                return
            rospy.sleep(0.05)
        raise RuntimeError("Timed out waiting for first PointCloud2 message.")

    def _wait_for_pose_feedback(self, timeout=10.0):
        t0 = time.time()
        while not rospy.is_shutdown() and (time.time() - t0) < timeout:
            try:
                self._get_ee_pose()
                return
            except Exception:
                rospy.sleep(0.05)
        raise RuntimeError("Timed out waiting for end-effector pose feedback from TF/EndpointState.")

    def _lookup_transform(self, target_frame, source_frame, stamp, timeout_sec=0.2, use_latest=False):
        query_time = rospy.Time(0) if use_latest else stamp
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            query_time,
            rospy.Duration(timeout_sec),
        )

    def _get_tip_pose_from_tf(self):
        tf_msg = self._lookup_transform(
            self.base_frame,
            self.tip_frame,
            rospy.Time.now(),
            timeout_sec=0.2,
            use_latest=self.use_latest_tf_for_pose,
        )
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        pos = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat = _normalize_quat_xyzw([q.x, q.y, q.z, q.w])
        return pos, quat

    def _get_tip_pose_from_endpoint(self):
        if self.last_endpoint is None:
            raise RuntimeError("No EndpointState received yet.")

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.last_endpoint_stamp
        pose_msg.header.frame_id = self.last_endpoint.header.frame_id or self.base_frame
        pose_msg.pose = self.last_endpoint.pose

        if pose_msg.header.frame_id != self.base_frame:
            tf_msg = self._lookup_transform(
                self.base_frame,
                pose_msg.header.frame_id,
                pose_msg.header.stamp,
                timeout_sec=0.2,
                use_latest=self.use_latest_tf_for_pose,
            )
            pose_msg = tf2_geometry_msgs.do_transform_pose(pose_msg, tf_msg)

        p = pose_msg.pose.position
        q = pose_msg.pose.orientation
        pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        quat = _normalize_quat_xyzw([q.x, q.y, q.z, q.w])
        return pos, quat

    def _get_ee_pose(self):
        errors = []

        if self.pose_source in ("tf", "auto"):
            try:
                return self._get_tip_pose_from_tf()
            except Exception as exc:
                errors.append(f"tf:{exc}")
                if self.pose_source == "tf":
                    raise

        if self.pose_source in ("endpoint", "endpoint_state", "auto"):
            try:
                return self._get_tip_pose_from_endpoint()
            except Exception as exc:
                errors.append(f"endpoint:{exc}")
                if self.pose_source in ("endpoint", "endpoint_state"):
                    raise

        raise RuntimeError("Unable to obtain EE pose (%s)" % ", ".join(errors))

    def _resolve_scan_center(self):
        if len(self.center_xyz) == 3:
            center = np.array(self.center_xyz, dtype=np.float64)
        else:
            center = self.start_xyz.copy()

        center += np.array(self.center_offset_xyz, dtype=np.float64)
        center[2] += self.z_offset
        return center

    def _resolve_scan_orientation(self, current_quat_xyzw):
        if not self.control_orientation:
            return None

        if self.orientation_mode in ("keep", "keep_current", "current"):
            return current_quat_xyzw.copy()

        if self.orientation_mode == "custom":
            if len(self.custom_target_quat_xyzw) != 4:
                raise ValueError("orientation_mode=custom requires ~target_quat_xyzw: [x,y,z,w]")
            return _normalize_quat_xyzw(self.custom_target_quat_xyzw)

        if self.orientation_mode == "topdown":
            quat = R.from_euler(
                "xyz",
                [self.topdown_roll_deg, self.topdown_pitch_deg, self.topdown_yaw_deg],
                degrees=True,
            ).as_quat()
            return _normalize_quat_xyzw(quat)

        raise ValueError("Unsupported orientation_mode='%s'" % self.orientation_mode)

    @staticmethod
    def _clip_norm(vec, max_norm):
        vec = np.asarray(vec, dtype=np.float64)
        n = np.linalg.norm(vec)
        if n < 1e-9:
            return vec
        if n > max_norm:
            return vec / n * max_norm
        return vec

    def _publish_twist(self, linear_xyz, angular_xyz=None):
        if angular_xyz is None:
            angular_xyz = np.zeros(3, dtype=np.float64)

        msg = EEVelGoals()
        tw = Twist()
        tw.linear.x = float(linear_xyz[0])
        tw.linear.y = float(linear_xyz[1])
        tw.linear.z = float(linear_xyz[2])
        tw.angular.x = float(angular_xyz[0])
        tw.angular.y = float(angular_xyz[1])
        tw.angular.z = float(angular_xyz[2])
        msg.ee_vels.append(tw)
        msg.tolerances.append(Twist())
        self.pub_vel.publish(msg)

    def _stop(self, duration=0.2):
        t_end = time.time() + duration
        rate = rospy.Rate(self.rate_hz)
        zeros = np.zeros(3, dtype=np.float64)
        while not rospy.is_shutdown() and time.time() < t_end:
            self._publish_twist(zeros, zeros)
            rate.sleep()

    def _drive_to_pose(self, target_xyz, target_quat_xyzw):
        t0 = time.time()
        rate = rospy.Rate(self.rate_hz)
        reached_since = None
        target_rot = R.from_quat(target_quat_xyzw) if target_quat_xyzw is not None else None

        while not rospy.is_shutdown() and (time.time() - t0) < self.timeout:
            cur_xyz, cur_quat_xyzw = self._get_ee_pose()
            pos_err = target_xyz - cur_xyz
            linear_cmd = np.array(
                [
                    clamp(self.kp_pos * pos_err[0], -self.vmax, self.vmax),
                    clamp(self.kp_pos * pos_err[1], -self.vmax, self.vmax),
                    clamp(self.kp_pos * pos_err[2], -self.vmax, self.vmax),
                ],
                dtype=np.float64,
            )

            if target_rot is not None:
                current_rot = R.from_quat(cur_quat_xyzw)
                rot_err = target_rot * current_rot.inv()
                ang_err = float(rot_err.magnitude())
                rotvec_err = rot_err.as_rotvec()
                angular_cmd = self._clip_norm(self.kp_rot * rotvec_err, self.wmax)
            else:
                ang_err = 0.0
                angular_cmd = np.zeros(3, dtype=np.float64)

            self._publish_twist(linear_cmd, angular_cmd)

            pos_ok = np.linalg.norm(pos_err) < self.pos_tol
            ang_ok = (target_rot is None) or (ang_err < self.ang_tol)
            if pos_ok and ang_ok:
                if reached_since is None:
                    reached_since = time.time()
                if (time.time() - reached_since) >= self.settle_time:
                    self._stop(0.15)
                    return True
            else:
                reached_since = None

            rate.sleep()

        self._stop(0.25)
        return False

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
            rospy.logwarn("[real_ring_scan] cloud TF lookup failed: %s", str(exc))
            return None

        if HAS_TF2_SENSOR_MSGS:
            try:
                return tf2sm.do_transform_cloud(cloud_msg, tf_msg)
            except Exception as exc:
                rospy.logwarn("[real_ring_scan] tf2_sensor_msgs failed, falling back to point-wise transform: %s", str(exc))

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

    def _capture(self, idx, z_tag, after_stamp):
        try:
            cloud = self._wait_for_fresh_cloud(after_stamp)
        except Exception as exc:
            rospy.logwarn("[real_ring_scan] capture skipped: %s", str(exc))
            return False

        transformed = self._transform_cloud_to_base(cloud)
        if transformed is None:
            rospy.logwarn("[real_ring_scan] transformed cloud is empty or too small")
            return False

        pcd = self._cloud_to_o3d(transformed)
        if pcd is None:
            rospy.logwarn("[real_ring_scan] cropped/downsampled cloud has too few points")
            return False

        out_path = os.path.join(self.out_dir, f"scan_{idx:04d}_z{z_tag}.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        rospy.loginfo("[real_ring_scan] saved %s (points=%d)", out_path, len(pcd.points))
        return True

    def _theta_for_index(self, index):
        start = math.radians(self.start_angle_deg)
        arc = math.radians(self.scan_arc_deg)

        if self.n_poses <= 1:
            return start

        full_turn = abs(self.scan_arc_deg - 360.0) < 1e-6
        if full_turn:
            return start + arc * float(index) / float(self.n_poses)
        return start + arc * float(index) / float(self.n_poses - 1)

    def run_scan(self):
        cx, cy, cz = self.scan_center_xyz
        rate = rospy.Rate(self.rate_hz)

        for i in range(self.n_poses):
            theta = self._theta_for_index(i)
            target_xyz = np.array(
                [
                    cx + self.radius * math.cos(theta),
                    cy + self.radius * math.sin(theta),
                    cz,
                ],
                dtype=np.float64,
            )

            rospy.loginfo("[real_ring_scan] target %d/%d xyz=%s", i + 1, self.n_poses, np.round(target_xyz, 4))
            reached = self._drive_to_pose(target_xyz, self.scan_quat_xyzw)
            if not reached:
                rospy.logwarn("[real_ring_scan] target %d not reached within timeout; capturing anyway", i)

            self._stop(self.capture_wait_after_stop)
            capture_gate_stamp = rospy.Time.now()

            got = 0
            for k in range(max(1, self.take_multiple_frames)):
                capture_idx = i * max(1, self.take_multiple_frames) + k
                if self._capture(capture_idx, "+0", capture_gate_stamp):
                    got += 1
                capture_gate_stamp = self.last_cloud_stamp
                rospy.sleep(self.capture_spacing)

            if got == 0:
                rospy.logwarn("[real_ring_scan] no valid capture saved at waypoint %d", i)

            rate.sleep()

        self._stop(0.5)

        if self.return_to_start:
            rospy.loginfo("[real_ring_scan] returning to start pose %s", np.round(self.start_xyz, 4))
            if not self._drive_to_pose(self.start_xyz, self.start_quat_xyzw):
                rospy.logwarn("[real_ring_scan] failed to return to start pose")
            self._stop(0.3)

        rospy.loginfo("[real_ring_scan] scan complete, saved frames in %s", self.out_dir)


def main():
    scanner = RealAutoRingScanAndFuse()
    scanner.run_scan()

    do_fuse = bool(rospy.get_param("~do_fuse", True))
    if not do_fuse:
        rospy.loginfo("[real_ring_scan] do_fuse=false, skipping posegraph fusion.")
        return

    rospy.loginfo("[real_ring_scan] starting posegraph fusion ...")
    try:
        before_path, after_path = fuse_scans(
            ply_dir=scanner.out_dir,
            pattern=str(rospy.get_param("~fuse_pattern", "scan_*.ply")),
            out_before=str(rospy.get_param("~out_before", "merged_posegraph_before_drop.ply")),
            out_after=str(rospy.get_param("~out_after", "merged_posegraph_after_drop.ply")),
            voxel_reg=float(rospy.get_param("~voxel_reg", 0.005)),
            voxel_final=float(rospy.get_param("~voxel_final", 0.003)),
            fitness_gate=float(rospy.get_param("~fitness_gate", 0.30)),
            rmse_gate=float(rospy.get_param("~rmse_gate", 0.015)),
            loop_k=int(rospy.get_param("~loop_k", 6)),
            loop_fitness_gate=float(rospy.get_param("~loop_fitness_gate", 0.40)),
            loop_rmse_gate=float(rospy.get_param("~loop_rmse_gate", 0.012)),
            opt_max_corr=float(rospy.get_param("~opt_max_corr", 0.02)),
            drop_k=int(rospy.get_param("~drop_k", 1)),
            drop_fitness_th=float(rospy.get_param("~drop_fitness_th", 0.35)),
            drop_rmse_th=float(rospy.get_param("~drop_rmse_th", 0.012)),
            verbose=bool(rospy.get_param("~fuse_verbose", True)),
        )
        rospy.loginfo("[real_ring_scan] fusion done. before=%s after=%s", before_path, after_path)
    except Exception as exc:
        rospy.logerr("[real_ring_scan] fusion failed: %s", str(exc))
        raise


if __name__ == "__main__":
    rospy.init_node("auto_scan_ring_and_fuse_real")
    main()
