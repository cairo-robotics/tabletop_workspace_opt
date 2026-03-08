#!/usr/bin/env python3
"""
ROS Noetic service server for grasp generation.

Service: `~generate_grasps` (type: tabletop_workspace_opt/GenerateGrasps)

Request:
- color_image: sensor_msgs/Image (expected BGR8)
- depth_image: sensor_msgs/Image (commonly 16UC1 mm or 32FC1 meters)
- camera_info: sensor_msgs/CameraInfo (intrinsics for the color stream)
- top_k: number of grasps to return (<=0 uses ~default_top_k)

Response:
- poses: geometry_msgs/Pose[] in the color camera optical frame
- scores: float32[] (aligned with poses)
- widths: float32[] (aligned with poses, meters)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
import rospy
import rospkg
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo

from graspnet_wrapper import GraspDetector
from image_tools import ImageTools
from tabletop_workspace_opt.srv import GenerateGrasps, GenerateGraspsResponse


def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w) using numpy only."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected (3,3) rotation matrix, got {R.shape}")

    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 0:
        q /= n
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _infer_factor_depth(depth_cv: np.ndarray, depth_encoding: str) -> float:
    if depth_cv.dtype == np.uint16:
        return 1000.0
    if depth_cv.dtype in (np.float32, np.float64):
        return 1.0
    rospy.logwarn(
        f"Unexpected depth dtype={depth_cv.dtype} encoding={depth_encoding}; defaulting factor_depth=1.0"
    )
    return 1.0


def _make_o3d_intrinsics(info: CameraInfo) -> o3d.camera.PinholeCameraIntrinsic:
    fx = info.K[0]
    fy = info.K[4]
    cx = info.K[2]
    cy = info.K[5]
    return o3d.camera.PinholeCameraIntrinsic(
        width=int(info.width),
        height=int(info.height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )


def _graspgroup_to_pose_score_width(grasp_group, top_k: int):
    """
    Best-effort extraction from graspnetAPI.GraspGroup.

    GraspNet baseline array format (common):
      [0]=score, [1]=width, [4:13]=rotation (9), [13:16]=translation (3)
    """
    arr = None
    if hasattr(grasp_group, "grasp_group_array"):
        try:
            arr = np.asarray(grasp_group.grasp_group_array)
        except Exception:
            arr = None

    if arr is None:
        rows = []
        try:
            for g in grasp_group[:top_k]:
                if hasattr(g, "grasp_array"):
                    rows.append(np.asarray(g.grasp_array))
        except Exception:
            rows = []
        if rows:
            arr = np.stack(rows, axis=0)

    if arr is None or arr.ndim != 2 or arr.shape[1] < 16:
        raise RuntimeError("Could not extract grasp arrays from GraspGroup (unexpected graspnetAPI format).")

    arr = arr[:top_k]
    scores = arr[:, 0].astype(np.float32).tolist()
    widths = arr[:, 1].astype(np.float32).tolist()
    rotations = arr[:, 4:13].reshape(-1, 3, 3)
    translations = arr[:, 13:16]

    poses = []
    for i in range(arr.shape[0]):
        R = rotations[i]
        t = translations[i]
        qx, qy, qz, qw = _rotmat_to_quat_xyzw(R)
        p = Pose()
        p.position.x = float(t[0])
        p.position.y = float(t[1])
        p.position.z = float(t[2])
        p.orientation.x = qx
        p.orientation.y = qy
        p.orientation.z = qz
        p.orientation.w = qw
        poses.append(p)

    return poses, scores, widths


@dataclass
class _Cfg:
    yolo_weight_path: str
    fastsam_weight_path: str
    yolo_conf_threshold: float
    fastsam_conf: float
    fastsam_iou: float
    graspnet_root: str
    checkpoint_path: str
    max_gripper_width: float
    collision_thresh: float
    voxel_size: float
    default_top_k: int


class GenerateGraspsServiceNode:
    def __init__(self):
        rospy.init_node("generate_grasps_service_node")
        self._bridge = CvBridge()

        pkg_path = rospkg.RosPack().get_path("tabletop_workspace_opt")
        models_dir = os.path.join(pkg_path, "assets", "models")

        self._cfg = _Cfg(
            yolo_weight_path=rospy.get_param("~yolo_weight_path", os.path.join(models_dir, "yolov8m.pt")),
            fastsam_weight_path=rospy.get_param("~fastsam_weight_path", os.path.join(models_dir, "FastSAM-s.pt")),
            yolo_conf_threshold=float(rospy.get_param("~yolo_conf_threshold", 0.5)),
            fastsam_conf=float(rospy.get_param("~fastsam_conf", 0.5)),
            fastsam_iou=float(rospy.get_param("~fastsam_iou", 0.7)),
            graspnet_root=str(rospy.get_param("~graspnet_root", "/home/yi-shiuan/sawyer_ws/src/graspnet-baseline")),
            checkpoint_path=str(rospy.get_param("~checkpoint_path", os.path.join(models_dir, "checkpoint.tar"))),
            max_gripper_width=float(rospy.get_param("~max_gripper_width", 0.05)),
            collision_thresh=float(rospy.get_param("~collision_thresh", 0.01)),
            voxel_size=float(rospy.get_param("~voxel_size", 0.01)),
            default_top_k=int(rospy.get_param("~default_top_k", 10)),
        )

        if not self._cfg.graspnet_root:
            raise RuntimeError("Parameter `~graspnet_root` is required (path to graspnet-baseline folder).")

        # Heavy models: load once.
        self._image_tools = ImageTools(
            yolo_weight_path=self._cfg.yolo_weight_path,
            fastsam_weight_path=self._cfg.fastsam_weight_path,
            yolo_conf_threshold=self._cfg.yolo_conf_threshold,
            fastsam_conf=self._cfg.fastsam_conf,
            fastsam_iou=self._cfg.fastsam_iou,
        )
        self._grasp_detector = GraspDetector(
            graspnet_root=self._cfg.graspnet_root,
            checkpoint_path=self._cfg.checkpoint_path,
            collision_thresh=self._cfg.collision_thresh,
            voxel_size=self._cfg.voxel_size,
            max_gripper_width=self._cfg.max_gripper_width,
        )

        self._srv = rospy.Service("~generate_grasps", GenerateGrasps, self._handle)
        rospy.loginfo("GenerateGrasps service ready: ~generate_grasps")

    def _handle(self, req: GenerateGrasps):
        resp = GenerateGraspsResponse()
        resp.header = req.color_image.header
        resp.success = False
        resp.message = ""

        try:
            color_bgr = self._bridge.imgmsg_to_cv2(req.color_image, desired_encoding="bgr8")
            depth_cv = self._bridge.imgmsg_to_cv2(req.depth_image, desired_encoding="passthrough")

            if color_bgr is None or color_bgr.size == 0:
                resp.message = "Empty color image"
                return resp
            if depth_cv is None or depth_cv.size == 0:
                resp.message = "Empty depth image"
                return resp

            intrinsics_o3d = _make_o3d_intrinsics(req.camera_info)
            factor_depth = _infer_factor_depth(depth_cv, req.depth_image.encoding)

            # Segmentation expects BGR; GraspNet typically expects RGB.
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

            _overlay, object_mask, _detections = self._image_tools.process_and_extract_objects(
                color_bgr, depth_cv, intrinsics_o3d
            )

            meta = {
                "intrinsic_matrix": np.array(req.camera_info.K, dtype=np.float32).reshape(3, 3),
                "factor_depth": float(factor_depth),
            }

            grasp_group, _cloud = self._grasp_detector.detect(
                color_rgb, depth_cv, object_mask, meta, visualize=False
            )

            if hasattr(grasp_group, "sort_by_score"):
                try:
                    grasp_group = grasp_group.sort_by_score()
                except Exception:
                    pass

            top_k = int(req.top_k) if int(req.top_k) > 0 else int(self._cfg.default_top_k)
            poses, scores, widths = _graspgroup_to_pose_score_width(grasp_group, top_k=top_k)

            resp.poses = poses
            resp.scores = scores
            resp.widths = widths
            resp.success = True
            resp.message = f"Returned {len(poses)} grasps"
            return resp

        except Exception as e:
            resp.message = str(e)
            return resp


if __name__ == "__main__":
    GenerateGraspsServiceNode()
    rospy.spin()


