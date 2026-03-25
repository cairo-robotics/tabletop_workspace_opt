#!/usr/bin/env python3
import rospy
import numpy as np
import open3d as o3d
import cv2

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

import tf2_ros
import tf.transformations as tft

def tf_to_matrix(trans, rot):
    """geometry_msgs/TransformStamped -> 4x4"""
    T = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    T[0, 3] = trans.x
    T[1, 3] = trans.y
    T[2, 3] = trans.z
    return T

class KeyframeTSDF:
    def __init__(self):
        rospy.init_node("keyframe_tsdf_fusion", anonymous=True)

        # ---- Params  ----
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_frame   = rospy.get_param("~ee_frame", "right_gripper")  # 例如 sawyer 的末端frame名
        self.cam_frame  = rospy.get_param("~cam_frame", "camera_color_optical_frame")

        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.info_topic  = rospy.get_param("~info_topic",  "/camera/color/camera_info")

        # Stillness detection params
        self.still_window = rospy.get_param("~still_window", 0.5)  # seconds
        self.pos_eps = rospy.get_param("~pos_eps", 0.002)          # 2mm
        self.rot_eps_deg = rospy.get_param("~rot_eps_deg", 0.2)    # 0.2 degrees

        # Keyframe params
        self.min_interval = rospy.get_param("~min_interval", 0.8)  # seconds
        self.max_keyframes = rospy.get_param("~max_keyframes", 50)

        # Depth params
        self.depth_scale = rospy.get_param("~depth_scale", 1000.0) # uint16(mm) -> 1000
        self.depth_trunc = rospy.get_param("~depth_trunc", 2.0)

        # TSDF params
        voxel_len = rospy.get_param("~voxel_length", 0.005)
        sdf_trunc = rospy.get_param("~sdf_trunc", 0.04)

        # ee->cam (eye-to-hand params)
        # 4x4 row-major, now this is only a default identity matrix
        ee_T_cam_list = rospy.get_param("~T_ee_cam", [1,0,0,0,
                                                     0,1,0,0,
                                                     0,0,1,0,
                                                     0,0,0,1])
        self.T_ee_cam = np.array(ee_T_cam_list, dtype=np.float64).reshape(4,4)

        self.bridge = CvBridge()

        # TF
        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)

        # Camera intrinsics
        self.o3d_intr = None
        rospy.Subscriber(self.info_topic, CameraInfo, self._on_info, queue_size=1)

        # TSDF volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_len,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # Sync RGB+Depth
        color_sub = message_filters.Subscriber(self.color_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.05)
        sync.registerCallback(self._on_rgbd)

        # For stillness detection
        self.pose_hist = []  # list of (t, T_base_ee)
        self.last_keyframe_time = rospy.Time(0)
        self.keyframe_count = 0

        rospy.loginfo("Keyframe TSDF fusion node started.")

    def _on_info(self, msg: CameraInfo):
        if self.o3d_intr is not None:
            return
        fx, fy = msg.K[0], msg.K[4]
        cx, cy = msg.K[2], msg.K[5]
        self.o3d_intr = o3d.camera.PinholeCameraIntrinsic(msg.width, msg.height, fx, fy, cx, cy)
        rospy.loginfo("Got camera intrinsics: fx=%.3f fy=%.3f cx=%.3f cy=%.3f", fx, fy, cx, cy)

    def _lookup_T_base_ee(self, stamp):
        try:
            tfmsg = self.tfbuf.lookup_transform(self.base_frame, self.ee_frame, stamp, rospy.Duration(0.05))
        except Exception:
            tfmsg = self.tfbuf.lookup_transform(self.base_frame, self.ee_frame, rospy.Time(0), rospy.Duration(0.1))
        return tf_to_matrix(tfmsg.transform.translation, tfmsg.transform.rotation)

    def _update_pose_hist(self, t, T_base_ee):
        self.pose_hist.append((t, T_base_ee))
        cutoff = t - rospy.Duration(self.still_window)
        while self.pose_hist and self.pose_hist[0][0] < cutoff:
            self.pose_hist.pop(0)

    @staticmethod
    def _pose_delta(T1, T2):
        # position
        dp = np.linalg.norm(T1[0:3,3] - T2[0:3,3])
        # rotation angle
        R = T1[0:3,0:3].T @ T2[0:3,0:3]
        angle = np.arccos(np.clip((np.trace(R)-1)/2.0, -1.0, 1.0))
        return dp, angle

    def _is_still(self):
        if len(self.pose_hist) < 2:
            return False
        T_start = self.pose_hist[0][1]
        T_end   = self.pose_hist[-1][1]
        dp, ang = self._pose_delta(T_start, T_end)
        return (dp < self.pos_eps) and (np.degrees(ang) < self.rot_eps_deg)

    def _on_rgbd(self, color_msg: Image, depth_msg: Image):
        if self.o3d_intr is None:
            return

        stamp = color_msg.header.stamp
        now = rospy.Time.now()

        # update pose history
        try:
            T_base_ee = self._lookup_T_base_ee(stamp)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "TF lookup failed: %s", str(e))
            return

        self._update_pose_hist(now, T_base_ee)

        # keyframe decision: still + interval + max count
        if self.keyframe_count >= self.max_keyframes:
            return
        if (now - self.last_keyframe_time).to_sec() < self.min_interval:
            return
        if not self._is_still():
            return

        # capture 1 keyframe and integrate
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logwarn("cv_bridge convert failed: %s", str(e))
            return

        # depth may be 16UC1(mm) or 32FC1(m)
        if depth.dtype == np.float32:
            # Open3D needs depth in uint16 with scale
            # convert to mm
            depth_u16 = (depth * 1000.0).astype(np.uint16)
            depth_img = depth_u16
            depth_scale = 1000.0
        else:
            depth_img = depth
            depth_scale = self.depth_scale

        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = o3d.geometry.Image(depth_img)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )

        # Compute T_base_cam
        T_base_cam = T_base_ee @ self.T_ee_cam

        # Open3D integrate need extrinsic: (world-to-camera）
        extrinsic = np.linalg.inv(T_base_cam)

        self.volume.integrate(rgbd, self.o3d_intr, extrinsic)

        self.last_keyframe_time = now
        self.keyframe_count += 1
        rospy.loginfo("Integrated keyframe #%d (still).", self.keyframe_count)

    def save(self):
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh("tsdf_mesh.ply", mesh)
        pcd = self.volume.extract_point_cloud()
        o3d.io.write_point_cloud("tsdf_pcd.ply", pcd)
        rospy.loginfo("Saved tsdf_mesh.ply and tsdf_pcd.ply")

if __name__ == "__main__":
    node = KeyframeTSDF()
    try:
        rospy.spin()
    finally:
        node.save()
