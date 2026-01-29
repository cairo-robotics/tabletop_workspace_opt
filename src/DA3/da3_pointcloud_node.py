#!/usr/bin/env python3
"""
DA3 PointCloud Node (ROS1 / rospy)

Subscribes:
  - /camera/color/image_raw      (sensor_msgs/Image)
  - /camera/color/camera_info    (sensor_msgs/CameraInfo)

Publishes:
  - ~pointcloud                 (sensor_msgs/PointCloud2)  in base_frame

It uses Depth Anything 3 (DA3) with pose-conditioned scaling:
  prediction = model.inference(
      image=[...],
      intrinsics=K,
      extrinsics=E,  # world-to-camera (w2c)
      align_to_input_ext_scale=True,
  )

TF handling:
  - Prefer TF lookup base_frame -> camera_color_optical_frame
  - If missing, try base_frame -> camera_link (+ camera_link -> camera_color_optical_frame)
  - If right_hand -> camera_link is missing, use a hardcoded static transform:
      right_hand -> camera_link : (0, 0, -0.3) and RPY(deg) = (180, -90, 0)
    then chain base -> right_hand from TF.
"""

from __future__ import annotations

import threading
import collections
import struct
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import cv2
import rospy
import tf2_ros
import tf.transformations as tft
import message_filters

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2 as pc2


import torch
from depth_anything_3.api import DepthAnything3

@dataclass
class FrameDatum:
    stamp: rospy.Time
    img_rgb: np.ndarray  # H,W,3 uint8
    K: np.ndarray  # 3,3 float32 (at input image resolution)
    T_base_cam: np.ndarray  # 4,4 float64 transform: base <- camera


def _transform_to_mat44(t) -> np.ndarray:
    """Convert geometry_msgs/Transform to 4x4 matrix."""
    q = np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w], dtype=np.float64)
    M = tft.quaternion_matrix(q)  # 4x4
    M[0, 3] = float(t.translation.x)
    M[1, 3] = float(t.translation.y)
    M[2, 3] = float(t.translation.z)
    return M


def _mat44_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _rpy_deg_to_mat44(roll_deg: float, pitch_deg: float, yaw_deg: float, xyz: Tuple[float, float, float]) -> np.ndarray:
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    M = tft.euler_matrix(r, p, y, axes="sxyz")
    M[0, 3] = xyz[0]
    M[1, 3] = xyz[1]
    M[2, 3] = xyz[2]
    return M.astype(np.float64)


def _pack_rgb_float(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pack uint8 r,g,b into a float32 field compatible with PointCloud2 rgb.
    """
    rgb_uint32 = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    return rgb_uint32.view(np.float32)


class DA3PointCloudNode:
    def __init__(self):
        rospy.init_node("da3_pointcloud_node")
        # Params
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

        # Frames
        self.base_frame = rospy.get_param("~base_frame", "reference/base")
        self.hand_frame = rospy.get_param("~hand_frame", "reference/right_hand")
        self.camera_link_frame = rospy.get_param("~camera_link_frame", "camera_link")
        self.camera_optical_frame = rospy.get_param("~camera_optical_frame", "camera_color_optical_frame")

        # Model
        self.model_name = rospy.get_param("~model_name", "depth-anything/DA3-LARGE-1.1")
        self.process_res = int(rospy.get_param("~process_res", 504))
        self.process_res_method = rospy.get_param("~process_res_method", "upper_bound_resize")
        self.align_to_input_ext_scale = bool(rospy.get_param("~align_to_input_ext_scale", True))

        # Sliding window size for pose-scale alignment (>=3 recommended)
        self.window_size = int(rospy.get_param("~window_size", 3))
        if self.window_size < 1:
            self.window_size = 1

        # Point cloud density control
        self.downsample = int(rospy.get_param("~downsample", 4))
        if self.downsample < 1:
            self.downsample = 1

        self.max_depth_m = float(rospy.get_param("~max_depth_m", 5.0))
        self.min_depth_m = float(rospy.get_param("~min_depth_m", 0.05))

        self.publish_camera_frame = bool(rospy.get_param("~publish_camera_frame", False))

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Fallback static transform: right_hand -> camera_link
        self.T_hand_camlink_static = _rpy_deg_to_mat44(
            roll_deg=180.0, pitch_deg=-90.0, yaw_deg=0.0, xyz=(0.0, 0.0, -0.3)
        )

        # DA3 model
        if not torch.cuda.is_available():
            rospy.logerr("torch.cuda.is_available() is False. Please check your GPU configuration.")
            raise RuntimeError("torch.cuda.is_available() is False. CUDA is not available.")
        device = torch.device("cuda")
        rospy.loginfo("Loading DA3 model: %s on %s", self.model_name, str(device))
        self.model = DepthAnything3.from_pretrained(self.model_name).to(device=device)

        # ROS I/O
        self.bridge = CvBridge()
        self.pub_pc = rospy.Publisher("~pointcloud", PointCloud2, queue_size=1)

        img_sub = message_filters.Subscriber(self.image_topic, Image)
        info_sub = message_filters.Subscriber(self.info_topic, CameraInfo)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [img_sub, info_sub], queue_size=5, slop=0.05
        )
        self.ts.registerCallback(self._sync_cb)

        # State
        self._buf: Deque[FrameDatum] = collections.deque(maxlen=max(self.window_size, 1))
        self._lock = threading.Lock()
        self._busy = False

        rospy.loginfo("DA3 PointCloud Node ready.")
        rospy.loginfo("Subscribing: %s and %s", self.image_topic, self.info_topic)
        rospy.loginfo("Publishing: ~pointcloud in frame %s", self.base_frame)

    def _lookup_T_base_cam(self, stamp: rospy.Time) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Returns (T_base_cam, camera_frame_used).
        """
        # 1) Preferred: base <- optical
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_optical_frame, stamp, rospy.Duration(0.05)
            )
            return _transform_to_mat44(tfm.transform), self.camera_optical_frame
        except Exception:
            pass

        # 2) base <- camera_link then camera_link <- optical (if available)
        try:
            tfm_base_link = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_link_frame, stamp, rospy.Duration(0.05)
            )
            T_base_camlink = _transform_to_mat44(tfm_base_link.transform)

            try:
                tfm_link_opt = self.tf_buffer.lookup_transform(
                    self.camera_link_frame, self.camera_optical_frame, stamp, rospy.Duration(0.05)
                )
                T_camlink_opt = _transform_to_mat44(tfm_link_opt.transform)
                return T_base_camlink @ T_camlink_opt, self.camera_optical_frame
            except Exception:
                return T_base_camlink, self.camera_link_frame
        except Exception:
            pass

        # 3) fallback: base <- right_hand and right_hand <- camera_link (static), then optional link<-optical
        try:
            tfm_base_hand = self.tf_buffer.lookup_transform(
                self.base_frame, self.hand_frame, stamp, rospy.Duration(0.05)
            )
            T_base_hand = _transform_to_mat44(tfm_base_hand.transform)
            T_base_camlink = T_base_hand @ self.T_hand_camlink_static

            try:
                tfm_link_opt = self.tf_buffer.lookup_transform(
                    self.camera_link_frame, self.camera_optical_frame, stamp, rospy.Duration(0.05)
                )
                T_camlink_opt = _transform_to_mat44(tfm_link_opt.transform)
                return T_base_camlink @ T_camlink_opt, self.camera_optical_frame
            except Exception:
                return T_base_camlink, self.camera_link_frame
        except Exception:
            return None, None

    def _camera_info_to_K(self, info: CameraInfo) -> np.ndarray:
        fx = info.K[0]
        fy = info.K[4]
        cx = info.K[2]
        cy = info.K[5]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    def _sync_cb(self, img_msg: Image, info_msg: CameraInfo):
        # Avoid building up a backlog of expensive inferences.
        with self._lock:
            if self._busy:
                return
            self._busy = True

        try:
            # TF at message time
            T_base_cam, cam_frame_used = self._lookup_T_base_cam(img_msg.header.stamp)
            if T_base_cam is None:
                rospy.logwarn_throttle(2.0, "TF lookup failed for base->camera at stamp; skipping frame.")
                return

            # Convert ROS image -> RGB numpy
            try:
                bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            except Exception:
                # Some drivers publish rgb8; fallback and convert.
                rgb = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            K = self._camera_info_to_K(info_msg)

            datum = FrameDatum(
                stamp=img_msg.header.stamp,
                img_rgb=rgb,
                K=K,
                T_base_cam=T_base_cam,
            )

            with self._lock:
                self._buf.append(datum)
                buf_len = len(self._buf)

            if buf_len < self.window_size:
                return

            # Assemble window inputs
            frames = list(self._buf)[-self.window_size :]
            imgs = [f.img_rgb for f in frames]
            intrinsics = np.stack([f.K for f in frames], axis=0)  # (N,3,3)

            # DA3 expects extrinsics as world-to-camera (w2c).
            extrinsics = np.stack([_mat44_inv(f.T_base_cam) for f in frames], axis=0)  # (N,4,4)

            prediction = self.model.inference(
                image=imgs,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                align_to_input_ext_scale=self.align_to_input_ext_scale,
                process_res=self.process_res,
                process_res_method=self.process_res_method,
            )

            depth = prediction.depth[-1]
            Kp = prediction.intrinsics[-1]

            # Ensure numpy
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            if hasattr(Kp, "cpu"):
                Kp = Kp.cpu().numpy()
            depth = depth.astype(np.float32)
            Kp = Kp.astype(np.float32)

            # Color aligned to processed resolution
            color = prediction.processed_images[-1]
            if hasattr(color, "cpu"):
                color = color.cpu().numpy()
            color = color.astype(np.uint8)
            h, w = depth.shape
            if color.shape[:2] != (h, w):
                color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)

            # Unproject in camera frame
            fx, fy = float(Kp[0, 0]), float(Kp[1, 1])
            cx, cy = float(Kp[0, 2]), float(Kp[1, 2])

            xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            xx = xx[:: self.downsample, :: self.downsample].reshape(-1)
            yy = yy[:: self.downsample, :: self.downsample].reshape(-1)
            z = depth[:: self.downsample, :: self.downsample].reshape(-1)
            rgb_ds = color[:: self.downsample, :: self.downsample].reshape(-1, 3)

            valid = np.isfinite(z)
            valid &= (z >= self.min_depth_m) & (z <= self.max_depth_m)
            if not np.any(valid):
                return

            xx = xx[valid]
            yy = yy[valid]
            z = z[valid]
            rgb_ds = rgb_ds[valid]

            x = (xx - cx) * z / fx
            y = (yy - cy) * z / fy

            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)  # 4,M

            # Transform into base frame using the latest datum transform
            T_base_cam_last = frames[-1].T_base_cam
            pts_base = (T_base_cam_last @ pts_cam).T[:, :3].astype(np.float32)  # M,3

            # Pack RGB (color is already RGB)
            r = rgb_ds[:, 0].astype(np.uint8)
            g = rgb_ds[:, 1].astype(np.uint8)
            b = rgb_ds[:, 2].astype(np.uint8)
            rgb_f = _pack_rgb_float(r, g, b)

            cloud = np.zeros((pts_base.shape[0],), dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)])
            cloud["x"] = pts_base[:, 0]
            cloud["y"] = pts_base[:, 1]
            cloud["z"] = pts_base[:, 2]
            cloud["rgb"] = rgb_f

            frame_id = cam_frame_used if self.publish_camera_frame else self.base_frame
            header = Header(stamp=img_msg.header.stamp, frame_id=frame_id)

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            pc_msg = pc2.create_cloud(header, fields, cloud)
            self.pub_pc.publish(pc_msg)
        except Exception as e:
            rospy.logerr_throttle(2.0, "DA3 pointcloud failed: %s", repr(e))
        finally:
            with self._lock:
                self._busy = False


def main():
    DA3PointCloudNode()
    rospy.spin()


if __name__ == "__main__":
    main()

