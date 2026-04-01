#!/usr/bin/env python3
"""
Standalone test client for the GenerateGrasps service.

Uses the local PNGs in this folder:
- test_image_rgb.png
- test_image_depth.png

It:
- loads the images with OpenCV
- wraps them in sensor_msgs/Image + CameraInfo
- calls /generate_grasps_service_node/generate_grasps
"""

import os

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header

from tabletop_workspace_opt.srv import GenerateGrasps, GenerateGraspsRequest


def _make_header(frame_id: str = "camera_color_optical_frame") -> Header:
    h = Header()
    h.stamp = rospy.Time.now()
    h.frame_id = frame_id
    return h


def _np_to_image_msg(arr: np.ndarray, encoding: str, frame_id: str) -> Image:
    msg = Image()
    msg.header = _make_header(frame_id)
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.strides[0]
    msg.data = arr.tobytes()
    return msg


def _make_camera_info(width: int, height: int) -> CameraInfo:
    """
    Build a CameraInfo using the intrinsics from the original script.
    """
    K_matrix = np.array(
        [
            [611.55, 0.0, 315.97],
            [0.0, 611.51, 244.46],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    info = CameraInfo()
    info.header = _make_header("camera_color_optical_frame")
    info.width = width
    info.height = height
    info.K = K_matrix.flatten().tolist()
    info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.R = [1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0]
    # Simple pinhole P matrix: [K | 0]
    info.P = [
        K_matrix[0, 0], 0.0, K_matrix[0, 2], 0.0,
        0.0, K_matrix[1, 1], K_matrix[1, 2], 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    return info


def main():
    rospy.init_node("test_generate_grasps_client", anonymous=True)

    here = os.path.dirname(os.path.abspath(__file__))
    color_path = os.path.join(here, "test_image_rgb.png")
    depth_path = os.path.join(here, "test_image_depth.png")

    if not os.path.exists(color_path):
        rospy.logerr("Missing color image: %s", color_path)
        return
    if not os.path.exists(depth_path):
        rospy.logerr("Missing depth image: %s", depth_path)
        return

    color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if color_bgr is None:
        rospy.logerr("Failed to load color image from %s", color_path)
        return
    if depth_raw is None:
        rospy.logerr("Failed to load depth image from %s", depth_path)
        return

    if depth_raw.dtype != np.uint16:
        rospy.logwarn("Depth PNG dtype is %s, converting to uint16 for 16UC1 encoding", depth_raw.dtype)
        depth_raw = depth_raw.astype(np.uint16)

    if color_bgr.shape[:2] != depth_raw.shape[:2]:
        rospy.logwarn(
            "Color and depth shapes differ: color=%s depth=%s; resizing depth to match color",
            color_bgr.shape,
            depth_raw.shape,
        )
        depth_raw = cv2.resize(depth_raw, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Build ROS messages
    color_msg = _np_to_image_msg(color_bgr, "bgr8", "camera_color_optical_frame")
    depth_msg = _np_to_image_msg(depth_raw, "16UC1", "camera_color_optical_frame")
    cam_info = _make_camera_info(color_bgr.shape[1], color_bgr.shape[0])

    service_name = "/generate_grasps_service_node/generate_grasps"
    rospy.loginfo("Waiting for service %s", service_name)
    rospy.wait_for_service(service_name)

    client = rospy.ServiceProxy(service_name, GenerateGrasps)

    req = GenerateGraspsRequest()
    req.color_image = color_msg
    req.depth_image = depth_msg
    req.camera_info = cam_info
    req.top_k = 10

    rospy.loginfo("Calling GenerateGrasps with test images")
    resp = client(req)

    rospy.loginfo("Service success: %s", resp.success)
    rospy.loginfo("Message: %s", resp.message)
    rospy.loginfo("Returned %d grasps", len(resp.poses))
    if resp.poses:
        rospy.loginfo("First grasp pose: %s", resp.poses[0])
        rospy.loginfo("First score/width: %.3f  %.4f", resp.scores[0], resp.widths[0])


if __name__ == "__main__":
    main()


