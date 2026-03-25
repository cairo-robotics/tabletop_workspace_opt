#!/usr/bin/env python
# -*- coding: utf-8 -*-
# A ROS node that simulates a Realsense camera in MuJoCo and publishes RGB, depth, and pointcloud topics.
from __future__ import print_function

import os
import math
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

try:
    # ROS provides this in most setups
    import sensor_msgs.point_cloud2 as pc2
except Exception as e:
    pc2 = None

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

# MuJoCo (new python bindings)
import mujoco


def compute_intrinsics_from_fovy(fovy_deg, width, height):
    """Return fx, fy, cx, cy from vertical FOV."""
    fovy = math.radians(float(fovy_deg))
    fy = (height / 2.0) / math.tan(fovy / 2.0)
    fx = fy * (float(width) / float(height))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def depth_to_pointcloud(depth_m, fx, fy, cx, cy, stride=2, z_min=0.05, z_max=2.0):
    """
    depth_m: (H,W) float32, meters in camera frame
    returns Nx3 float32 points in camera frame
    """
    H, W = depth_m.shape
    # Downsample by stride to keep pointcloud lighter
    vv = np.arange(0, H, stride)
    uu = np.arange(0, W, stride)
    v, u = np.meshgrid(vv, uu, indexing='ij')  # shapes (H/stride, W/stride)

    z = depth_m[v, u]
    valid = np.isfinite(z) & (z > z_min) & (z < z_max)

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)  # (N,3)
    return pts


class RealsenseSimulationNode(object):
    def __init__(self):
        self.scene_xml = rospy.get_param("~scene_xml_path", "")
        self.scene_name = rospy.get_param("~scene_name", "")  # optional
        self.camera_name = rospy.get_param("~camera_name", "realsense_rgb")
        self.width = int(rospy.get_param("~width", 640))
        self.height = int(rospy.get_param("~height", 480))
        self.rate_hz = float(rospy.get_param("~rate", 10.0))

        self.frame_id = rospy.get_param("~frame_id", "realsense_link")
        self.optical_frame_id = rospy.get_param("~optical_frame_id", "realsense_color_optical_frame")

        self.joint_state_topic = rospy.get_param("~joint_state_topic", "/joint_states")

        # Depth -> pointcloud params
        self.pc_stride = int(rospy.get_param("~pc_stride", 2))
        self.z_min = float(rospy.get_param("~z_min", 0.05))
        self.z_max = float(rospy.get_param("~z_max", 2.0))

        # Camera intrinsics (if not provided, derive from fovy)
        self.fovy_deg = float(rospy.get_param("~fovy_deg", 58.0))
        self.fx = rospy.get_param("~fx", None)
        self.fy = rospy.get_param("~fy", None)
        self.cx = rospy.get_param("~cx", None)
        self.cy = rospy.get_param("~cy", None)

        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            fx, fy, cx, cy = compute_intrinsics_from_fovy(self.fovy_deg, self.width, self.height)
            self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        else:
            self.fx = float(self.fx); self.fy = float(self.fy)
            self.cx = float(self.cx); self.cy = float(self.cy)

        # Publishers
        self.pub_rgb = rospy.Publisher("/realsense/color/image_raw", Image, queue_size=1)
        self.pub_depth = rospy.Publisher("/realsense/depth/image_raw", Image, queue_size=1)
        self.pub_info = rospy.Publisher("/realsense/color/camera_info", CameraInfo, queue_size=1)
        self.pub_pc = rospy.Publisher("/realsense/depth/points", PointCloud2, queue_size=1)

        self.bridge = CvBridge() if CvBridge is not None else None

        # Load scene
        self.model = self._load_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
                # -------- Low-level offscreen rendering for depth --------
        # Create an OpenGL context for offscreen rendering
        self.gl = mujoco.GLContext(self.width, self.height)
        self.gl.make_current()

        # Visualization objects
        self.vopt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.cam = mujoco.MjvCamera()

        # Use the named MuJoCo camera in the model
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id < 0:
            raise RuntimeError("Camera name not found in model: %s" % self.camera_name)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

        # Render context
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        # Buffers for readPixels
        self._rgb_buf = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._depth_buf = np.zeros((self.height, self.width), dtype=np.float32)


        # Joint mapping: name -> qpos addr
        self.joint_name_to_qposadr = {}
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            qposadr = int(self.model.jnt_qposadr[j])
            self.joint_name_to_qposadr[name] = qposadr

        self.latest_js = None
        rospy.Subscriber(self.joint_state_topic, JointState, self._js_cb, queue_size=1)

        rospy.loginfo("[realsense_simulation_node] Loaded model with %d joints. Rendering from camera='%s'.",
                      self.model.njnt, self.camera_name)

    def _load_model(self):
        # Preferred: explicit path
        if self.scene_xml and os.path.isfile(self.scene_xml):
            return mujoco.MjModel.from_xml_path(self.scene_xml)

        # Fallback: try ROS param /mujoco_sim scene path pattern (you may adapt this)
        # If you have a known directory where scenes live, set ~scene_xml_path.
        raise RuntimeError(
            "scene_xml_path not set or file not found. "
            "Please pass ~scene_xml_path to the node (absolute path)."
        )

    def _js_cb(self, msg):
        self.latest_js = msg

    def _apply_joint_state_to_qpos(self, js):
        # Apply only joints that exist in MuJoCo model mapping
        if js is None:
            return
        for name, pos in zip(js.name, js.position):
            if name in self.joint_name_to_qposadr:
                adr = self.joint_name_to_qposadr[name]
                self.data.qpos[adr] = float(pos)

        mujoco.mj_forward(self.model, self.data)

    def _make_camerainfo(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.optical_frame_id
        msg.width = self.width
        msg.height = self.height

        msg.K = [self.fx, 0.0, self.cx,
                 0.0, self.fy, self.cy,
                 0.0, 0.0, 1.0]
        msg.P = [self.fx, 0.0, self.cx, 0.0,
                 0.0, self.fy, self.cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        msg.D = [0, 0, 0, 0, 0]
        msg.distortion_model = "plumb_bob"
        return msg
    
    def _render_rgb_and_depth_m(self):
        """
        Returns:
        rgb_u8: (H,W,3) uint8
        depth_m: (H,W) float32, meters, NaN where invalid
        """
        # Update scene for current state
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn
        )

        # Render offscreen
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        mujoco.mjr_render(self.viewport, self.scn, self.con)

        # Read pixels (OpenGL depth buffer in [0,1])
        mujoco.mjr_readPixels(self._rgb_buf, self._depth_buf, self.viewport, self.con)

        # Flip vertically (common OpenGL convention)
        rgb = np.flipud(self._rgb_buf)
        d = np.flipud(self._depth_buf)

        # Convert depth buffer -> linear depth (meters)
        n = float(self.model.vis.map.znear)
        f = float(self.model.vis.map.zfar)
        if f <= 0 or f <= n:
            # fallback if xml not set correctly
            n, f = 0.01, 3.0

        # OpenGL depth buffer to metric depth:
        # z = (2*n*f) / (f + n - (2*d - 1)*(f - n))
        z = (2.0 * n * f) / (f + n - (2.0 * d - 1.0) * (f - n))

        # Mark invalids
        z[(d <= 0.0) | (d >= 1.0) | ~np.isfinite(z)] = np.nan

        return rgb.astype(np.uint8), z.astype(np.float32)


    def spin(self):
        r = rospy.Rate(self.rate_hz)

        if pc2 is None:
            rospy.logwarn("sensor_msgs.point_cloud2 not available; PointCloud2 will not publish.")

        if self.bridge is None:
            rospy.logwarn("CvBridge not available; will publish raw Image buffers.")

        while not rospy.is_shutdown():
            # Sync MuJoCo state (optional)
            js = self.latest_js
            if js is not None:
                self._apply_joint_state_to_qpos(js)
            else:
                mujoco.mj_forward(self.model, self.data)

            # Render RGB + Depth (meters) from low-level offscreen buffer
            rgb_u8, depth_f32 = self._render_rgb_and_depth_m()

            stamp = rospy.Time.now()

            # Publish CameraInfo
            self.pub_info.publish(self._make_camerainfo(stamp))

            # Publish RGB
            if self.bridge is not None:
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb_u8, encoding="rgb8")
            else:
                rgb_msg = Image()
                rgb_msg.height = self.height
                rgb_msg.width = self.width
                rgb_msg.encoding = "rgb8"
                rgb_msg.step = self.width * 3
                rgb_msg.data = rgb_u8.tobytes()

            rgb_msg.header = Header(stamp=stamp, frame_id=self.optical_frame_id)
            self.pub_rgb.publish(rgb_msg)

            # Publish Depth (32FC1, meters)
            if self.bridge is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth_f32, encoding="32FC1")
            else:
                depth_msg = Image()
                depth_msg.height = self.height
                depth_msg.width = self.width
                depth_msg.encoding = "32FC1"
                depth_msg.step = self.width * 4
                depth_msg.data = depth_f32.tobytes()

            depth_msg.header = Header(stamp=stamp, frame_id=self.optical_frame_id)
            self.pub_depth.publish(depth_msg)

            # Publish PointCloud2 (only if someone subscribes)
            if pc2 is not None and self.pub_pc.get_num_connections() > 0:
                pts = depth_to_pointcloud(
                    depth_f32, self.fx, self.fy, self.cx, self.cy,
                    stride=self.pc_stride, z_min=self.z_min, z_max=self.z_max
                )
                header = Header(stamp=stamp, frame_id=self.optical_frame_id)
                cloud = pc2.create_cloud_xyz32(header, pts.tolist())
                self.pub_pc.publish(cloud)

            r.sleep()




if __name__ == "__main__":
    rospy.init_node("realsense_simulation_node", anonymous=False)
    try:
        node = RealsenseSimulationNode()
        node.spin()
    except Exception as e:
        rospy.logerr("realsense_simulation_node failed: %s", str(e))
        raise
