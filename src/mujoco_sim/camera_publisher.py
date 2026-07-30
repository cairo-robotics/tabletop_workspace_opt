#!/usr/bin/env python3
"""Publishes MuJoCo camera views as ROS sensor_msgs/Image streams.

Renders named MJCF cameras (e.g. ``static_camera``, ``wrist_camera`` added
for the VLM controller) offscreen and publishes them at a fixed rate.

Design notes:
  * The interactive viewer owns a GLFW window context on the main thread.
    This publisher renders in its OWN thread with an EGL context
    (``MUJOCO_GL=egl`` is set before the first ``mujoco.Renderer`` is
    created, unless the user already chose a backend), so the two GL
    contexts never touch.
  * The physics thread steps ``mjData`` continuously. Each frame is rendered
    from a preallocated snapshot copy (``mj_copyData``) so the renderer sees
    a consistent state; the copy itself is the only concurrent read, which
    matches the existing tolerance in simulation_server's publish loop.
  * Cameras missing from the loaded model are skipped with a warning, so
    scenes without VLM cameras run unchanged.
"""
import copy
import os
import threading

# Choose the offscreen GL backend before mujoco.Renderer first imports its
# GL context module. MUJOCO_CAMERA_GL (or a non-glfw MUJOCO_GL) is
# respected; 'glfw' is refused because an offscreen GLFW context in this
# background thread races the interactive GUI window's X11 error handler
# and aborts the process.
_camera_gl = (
    os.environ.get("MUJOCO_CAMERA_GL")
    or os.environ.get("MUJOCO_GL")
    or "egl"
)
if _camera_gl == "glfw":
    _camera_gl = "egl"
os.environ["MUJOCO_GL"] = _camera_gl

import mujoco
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class CameraStream:
    def __init__(self, camera_name, topic, frame_id):
        self.camera_name = camera_name
        self.topic = topic
        self.frame_id = frame_id
        self.publisher = rospy.Publisher(topic, Image, queue_size=1)


class MuJoCoCameraPublisher:
    """Renders named model cameras and publishes them at ``rate_hz``."""

    def __init__(self, model, data, step_lock=None, rate_hz=10.0,
                 width=640, height=360):
        self._model = model
        self._data = data
        self._step_lock = step_lock if step_lock is not None else threading.Lock()
        self._rate_hz = float(rate_hz)
        self._width = int(width)
        self._height = int(height)
        self._bridge = CvBridge()
        self._streams = []
        self._thread = None
        self._stop = threading.Event()
        self._gripper_mask = None

    def enable_gripper_mask(self, camera_name, topic, frame_id, body_names):
        """Publish a binary gripper silhouette (mono8) for one camera via
        MuJoCo segmentation rendering. Skipped (returns False) if the camera
        or the gripper geoms are not found."""
        cam_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            rospy.logwarn("gripper mask: camera %r not in model", camera_name)
            return False
        body_ids = set()
        for name in body_names:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                body_ids.add(bid)
        geom_ids = [g for g in range(self._model.ngeom)
                    if int(self._model.geom_bodyid[g]) in body_ids]
        if not geom_ids:
            rospy.logwarn("gripper mask: no geoms for bodies %s", body_names)
            return False
        self._gripper_mask = {
            "camera": camera_name,
            "frame_id": frame_id,
            "geom_ids": np.array(geom_ids, dtype=np.int32),
            "publisher": rospy.Publisher(topic, Image, queue_size=1),
        }
        rospy.loginfo("publishing gripper mask on %s (%d geoms from %s)",
                      topic, len(geom_ids), body_names)
        return True

    def add_camera(self, camera_name, topic, frame_id):
        """Register a camera; skipped with a warning if not in the model."""
        cam_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id < 0:
            rospy.logwarn(
                "camera %r not found in the MuJoCo model; not publishing "
                "(add a <camera> element to the scene XML)", camera_name,
            )
            return False
        self._streams.append(CameraStream(camera_name, topic, frame_id))
        rospy.loginfo(
            "publishing MuJoCo camera %r on %s (%dx%d @ %.1f Hz)",
            camera_name, topic, self._width, self._height, self._rate_hz,
        )
        return True

    def start(self):
        if not self._streams:
            rospy.logwarn("no MuJoCo cameras to publish")
            return
        self._thread = threading.Thread(
            target=self._render_loop, name="mujoco_camera_publisher",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _render_loop(self):
        # Renderer and the mjData snapshot are created inside the render
        # thread so the EGL context is current only here.
        try:
            renderer = mujoco.Renderer(
                self._model, height=self._height, width=self._width
            )
        except Exception as exc:
            rospy.logerr(
                "failed to create offscreen MuJoCo renderer (%s). Camera "
                "topics will not be published; try MUJOCO_GL=egl or osmesa.",
                exc,
            )
            return

        seg_renderer = None
        if self._gripper_mask is not None:
            try:
                seg_renderer = mujoco.Renderer(
                    self._model, height=self._height, width=self._width)
                seg_renderer.enable_segmentation_rendering()
            except Exception as exc:
                rospy.logerr("gripper-mask segmentation renderer failed "
                             "(%s); mask disabled", exc)
                self._gripper_mask = None

        rate = rospy.Rate(self._rate_hz)

        while not rospy.is_shutdown() and not self._stop.is_set():
            # Consistent snapshot: renderer never reads live-stepping data,
            # and the step lock keeps mj_step out of the copy (MuJoCo raises
            # FatalError if mjData is copied while its stack is in use).
            # mujoco 3.2 has no mj_copyData binding; MjData.__copy__ wraps
            # the same C call.
            try:
                with self._step_lock:
                    snapshot = copy.copy(self._data)
            except mujoco.FatalError as exc:
                rospy.logwarn_throttle(10.0, "mjData snapshot failed: %s", exc)
                rate.sleep()
                continue
            stamp = rospy.Time.now()
            for stream in self._streams:
                try:
                    renderer.update_scene(snapshot, camera=stream.camera_name)
                    rgb = renderer.render()
                except Exception as exc:
                    rospy.logwarn_throttle(
                        10.0, "render failed for %r: %s",
                        stream.camera_name, exc,
                    )
                    continue
                msg = self._bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
                msg.header.stamp = stamp
                msg.header.frame_id = stream.frame_id
                stream.publisher.publish(msg)

                if (seg_renderer is not None and self._gripper_mask is not None
                        and stream.camera_name
                        == self._gripper_mask["camera"]):
                    self._publish_gripper_mask(seg_renderer, snapshot, stamp)
            rate.sleep()

    def _publish_gripper_mask(self, seg_renderer, snapshot, stamp):
        cfg = self._gripper_mask
        try:
            seg_renderer.update_scene(snapshot, camera=cfg["camera"])
            # (H, W, 2) int32: channel 0 = object id, channel 1 = object type.
            seg = seg_renderer.render()
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "gripper mask render failed: %s", exc)
            return
        is_geom = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = np.isin(seg[:, :, 0], cfg["geom_ids"]) & is_geom
        mask = mask.astype(np.uint8) * 255
        try:
            mask_msg = self._bridge.cv2_to_imgmsg(mask, encoding="mono8")
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "gripper mask encode failed: %s", exc)
            return
        mask_msg.header.stamp = stamp
        mask_msg.header.frame_id = cfg["frame_id"]
        cfg["publisher"].publish(mask_msg)
