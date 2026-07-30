"""Static-camera projection and visual prompting (Set-of-Marks + arrow).

The static camera is defined identically in every scene MJCF:
    <camera name="static_camera" pos="1.35 0 1.9"
            xyaxes="0 1 0 -0.785 0 0.620" fovy="45"/>
rendered at 640x360. All math is in the WORLD frame (matching
/mujoco_sim/detections and /mujoco_sim/endpoint_state), which avoids the
world->base z offset entirely.
"""
import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 ships with ROS
    cv2 = None


class StaticCameraModel:
    def __init__(self,
                 pos=(1.35, 0.0, 1.9),
                 xyaxes=(0.0, 1.0, 0.0, -0.785, 0.0, 0.620),
                 fovy_deg: float = 45.0,
                 width: int = 640,
                 height: int = 360):
        self.pos = np.asarray(pos, dtype=float)
        x_cam = np.asarray(xyaxes[:3], dtype=float)
        y_cam = np.asarray(xyaxes[3:], dtype=float)
        x_cam /= np.linalg.norm(x_cam)
        y_cam /= np.linalg.norm(y_cam)
        z_cam = np.cross(x_cam, y_cam)  # camera looks along -z_cam
        self.rotation = np.stack([x_cam, y_cam, z_cam])  # rows = camera axes
        self.width = width
        self.height = height
        self.fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        self.fx = self.fy
        self.cx = width / 2.0
        self.cy = height / 2.0

    def project(self, p_world) -> Optional[Tuple[int, int]]:
        """World point -> (u, v) pixel, or None if behind the camera."""
        d = np.asarray(p_world, dtype=float) - self.pos
        xc, yc, zc = self.rotation @ d
        depth = -zc  # camera views along -z
        if depth <= 1e-6:
            return None
        u = self.cx + self.fx * xc / depth
        v = self.cy - self.fy * yc / depth
        return int(round(u)), int(round(v))

    def project_clamped(self, p_world) -> Tuple[Tuple[int, int], bool]:
        """Like project(), but clamps to the image border; returns
        ((u, v), in_view)."""
        uv = self.project(p_world)
        if uv is None:
            return (0, 0), False
        u, v = uv
        in_view = 0 <= u < self.width and 0 <= v < self.height
        return (min(max(u, 4), self.width - 5),
                min(max(v, 4), self.height - 5)), in_view

    def annotate(self, bgr_image, marks: List[dict],
                 arrow: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 gripper_mask=None, mask_color=(255, 255, 0),
                 mask_opacity: float = 0.5):
        """Draw the three CASPER annotations: gripper mask, Set-of-Marks
        labels, and the EE motion arrow.

        marks: [{"mark_id": "1", "position_world": [x,y,z]}, ...]
        arrow: (p_world_past, p_world_now) end-effector positions, or None.
        gripper_mask: mono (H, W) uint8 silhouette (>0 = gripper), or None.
            mask_color is BGR; kept distinct from the green marks/arrow.
        Returns a copy of the image.
        """
        if cv2 is None:
            return bgr_image
        img = bgr_image.copy()
        # Gripper mask first (underneath), so marks and arrow stay on top.
        if gripper_mask is not None and gripper_mask.shape[:2] == img.shape[:2]:
            sel = gripper_mask > 0
            if sel.any():
                overlay = img.copy()
                overlay[sel] = mask_color
                cv2.addWeighted(overlay, mask_opacity, img,
                                1.0 - mask_opacity, 0.0, img)
        for mark in marks:
            (u, v), in_view = self.project_clamped(mark["position_world"])
            color = (60, 220, 60) if in_view else (0, 165, 255)
            cv2.circle(img, (u, v), 13, color, 2)
            cv2.circle(img, (u, v), 13, (0, 0, 0), 1)
            label = str(mark["mark_id"])
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.putText(img, label, (u - size[0] // 2, v + size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, label, (u - size[0] // 2, v + size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if arrow is not None:
            tail, _ = self.project_clamped(arrow[0])
            tip, _ = self.project_clamped(arrow[1])
            if tail != tip:
                cv2.arrowedLine(img, tail, tip, (0, 255, 0), 4,
                                tipLength=0.35)
        return img
