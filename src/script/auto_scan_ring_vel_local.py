#!/usr/bin/env python3
import os, math, time
import numpy as np
import rospy

import tf2_ros
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_geometry_msgs

from relaxed_ik_ros1.msg import EEVelGoals
import open3d as o3d

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class AutoRingScannerVelLocal:
    def __init__(self):
        self.out_dir     = rospy.get_param("~out_dir", "/home/heyang/scans")
        self.cloud_topic = rospy.get_param("~cloud_topic", "/realsense/depth/points")
        self.js_topic    = rospy.get_param("~js_topic", "/relaxed_ik/joint_angle_solutions")
        self.vel_topic   = rospy.get_param("~vel_topic", "/relaxed_ik/ee_vel_goals")

        self.base_frame  = rospy.get_param("~base_frame", "world")
        self.tip_frame   = rospy.get_param("~tip_frame", "right_l6")

        # local ring
        self.radius = float(rospy.get_param("~radius", 0.08))     # 8cm 
        self.n_poses = int(rospy.get_param("~n_poses", 16))
        self.z_offset = float(rospy.get_param("~z_offset", 0.0))  

        # controller
        self.rate_hz = float(rospy.get_param("~rate", 60.0))
        self.kp_pos  = float(rospy.get_param("~kp_pos", 2.0))
        self.vmax    = float(rospy.get_param("~vmax", 0.05))      # 5cm/s max speed
        self.pos_tol = float(rospy.get_param("~pos_tol", 0.006))  # 6mm
        self.settle_time = float(rospy.get_param("~settle_time", 0.25))
        self.timeout = float(rospy.get_param("~timeout", 8.0))

        os.makedirs(self.out_dir, exist_ok=True)

        self.pub_vel = rospy.Publisher(self.vel_topic, EEVelGoals, queue_size=1)

        self.last_cloud = None
        self.last_js = None
        rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=1)
        rospy.Subscriber(self.js_topic, JointState, self._js_cb, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo("[auto_scan_local] waiting for first cloud and jointstate...")
        while not rospy.is_shutdown() and (self.last_cloud is None or self.last_js is None):
            rospy.sleep(0.05)

        # capture initial EE position as center
        self.center_xyz = self._get_tip_xyz()
        self.center_xyz[2] += self.z_offset
        rospy.loginfo("[auto_scan_local] center_xyz=%s", np.round(self.center_xyz, 3))

        # ---- IMPORTANT: fix TF extrapolation by using ROS time from /use_sim_time ----
        # If simulation time is used, pointcloud stamps might be ahead/behind wall time.
        # We'll transform clouds at latest available TF time instead of cloud stamp.
        self.use_latest_tf_for_cloud = bool(rospy.get_param("~use_latest_tf_for_cloud", True))

    def _cloud_cb(self, msg): self.last_cloud = msg
    def _js_cb(self, msg): self.last_js = msg

    def _get_tip_xyz(self):
        tf = self.tf_buffer.lookup_transform(self.base_frame, self.tip_frame, rospy.Time(0), rospy.Duration(1.0))
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z], dtype=np.float64)

    def _publish_twist(self, vx, vy, vz):
        msg = EEVelGoals()
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.linear.z = float(vx), float(vy), float(vz)
        tw.angular.x, tw.angular.y, tw.angular.z = 0.0, 0.0, 0.0  # NO ROTATION, just translation for scanning
        msg.ee_vels.append(tw)
        msg.tolerances.append(Twist())
        self.pub_vel.publish(msg)

    def _stop(self, duration=0.2):
        t_end = time.time() + duration
        r = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and time.time() < t_end:
            self._publish_twist(0,0,0)
            r.sleep()

    def _drive_to_xyz(self, target_xyz):
        t0 = time.time()
        r = rospy.Rate(self.rate_hz)
        reached_since = None

        while not rospy.is_shutdown() and (time.time() - t0) < self.timeout:
            cur = self._get_tip_xyz()
            err = target_xyz - cur

            vx = clamp(self.kp_pos * err[0], -self.vmax, self.vmax)
            vy = clamp(self.kp_pos * err[1], -self.vmax, self.vmax)
            vz = clamp(self.kp_pos * err[2], -self.vmax, self.vmax)

            self._publish_twist(vx, vy, vz)

            if np.linalg.norm(err) < self.pos_tol:
                if reached_since is None:
                    reached_since = time.time()
                if time.time() - reached_since >= self.settle_time:
                    self._stop(0.15)
                    return True
            else:
                reached_since = None

            r.sleep()

        self._stop(0.2)
        return False

    def _capture_cloud_world(self, idx):
        cloud = self.last_cloud
        if cloud is None:
            rospy.logwarn("No cloud to capture")
            return False

        try:
            if self.use_latest_tf_for_cloud:
                tf = self.tf_buffer.lookup_transform(self.base_frame, cloud.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            else:
                tf = self.tf_buffer.lookup_transform(self.base_frame, cloud.header.frame_id, cloud.header.stamp, rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("TF lookup failed: %s", str(e))
            return False

        pts = []
        for p in pc2.read_points(cloud, skip_nans=True):
            pt = PointStamped()
            pt.header = cloud.header
            pt.point.x, pt.point.y, pt.point.z = p[0], p[1], p[2]
            pw = tf2_geometry_msgs.do_transform_point(pt, tf)
            pts.append([pw.point.x, pw.point.y, pw.point.z])

        if len(pts) < 200:
            rospy.logwarn("Too few points captured: %d", len(pts))
            return False

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        out = os.path.join(self.out_dir, f"scan_{idx:03d}.ply")
        o3d.io.write_point_cloud(out, pcd)
        rospy.loginfo("Saved %s (points=%d)", out, len(pts))
        return True

    def run(self):
        cx, cy, cz = self.center_xyz

        for i in range(self.n_poses):
            theta = 2.0 * math.pi * float(i) / float(self.n_poses)

            # local ring around initial pose (XY)
            x = cx + self.radius * math.cos(theta)
            y = cy + self.radius * math.sin(theta)
            z = cz

            target = np.array([x,y,z], dtype=np.float64)
            rospy.loginfo("[auto_scan_local] target %d/%d xyz=%s", i+1, self.n_poses, np.round(target,3))

            ok = self._drive_to_xyz(target)
            if not ok:
                rospy.logwarn("[auto_scan_local] failed to reach target %d, capturing anyway", i)

            self._capture_cloud_world(i)

        self._stop(0.5)
        rospy.loginfo("[auto_scan_local] done. scans in %s", self.out_dir)

if __name__ == "__main__":
    rospy.init_node("auto_scan_ring_vel_local")
    AutoRingScannerVelLocal().run()
