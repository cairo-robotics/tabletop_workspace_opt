#!/usr/bin/env python3
"""YOLO 3D Pose Node

Subscribes to RGB, depth, and camera info topics. This version focuses on
manual multi-object tracking with semantic labels (Set A: 1-50, Set B: 51-100, Goals)
and features a pose-locking mechanism to stabilize 3D visualizations.
"""
import os
import numpy as np
import rospy
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import tf2_ros
import tf_conversions
import threading
import json

# Try to load YOLO
YOLO_OK = True
try:
    from ultralytics import YOLO
except Exception:
    YOLO_OK = False

class Yolo3DPoseNode:
    def __init__(self):
        if not YOLO_OK:
            rospy.logerr("Ultralytics YOLO not installed. Run: pip install ultralytics")
            raise RuntimeError("Missing YOLO package")

        rospy.init_node("yolo_3d_pose_node")
        self.bridge = CvBridge()

        # --- Topics and frames ---
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/camera/color/camera_info")
        self.color_frame = rospy.get_param("~color_optical_frame", "camera_color_optical_frame")
        self.base_frame  = rospy.get_param("~base_frame", "world")

        # YOLO config
        model_path = rospy.get_param("~model", "yolov8m.pt")
        self.show_gui   = rospy.get_param("~show_gui", True)

        self.model = YOLO(model_path)
        self.model.to("cpu")
        self.class_names = self.model.names

        self.fx = self.fy = self.cx = self.cy = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        if self.show_gui:
            cv.namedWindow("YOLO detections", cv.WINDOW_NORMAL)
            cv.setMouseCallback("YOLO detections", self._mouse_cb)

        # Publishers
        self.pub_dets    = rospy.Publisher("~detections", Detection2DArray, queue_size=10)
        self.pub_annotated = rospy.Publisher("~annotated_image", Image, queue_size=1)
        self.pub_markers   = rospy.Publisher("~object_markers", MarkerArray, queue_size=10)

        # --- State for manual annotation and tracking ---
        self._lock = threading.Lock()
        self.paused = False
        self.freeze_img = None
        self.freeze_header = None
        self.ann_boxes = []
        self._drag_start = None
        self._active_idx = None

        # State variables for multi-object tracking
        self.trackers = {}
        self.tracking_boxes = {}
        self.tracker_labels = {}
        self.tracker_id_counter = 0
        self.goal_toggle = True

        # State for pose locking
        self.poses_locked = False
        self.locked_poses = {}

        # Visualization colors
        self.set_a_color = (0.0, 0.0, 1.0, 0.6) # Blue
        self.set_b_color = (0.0, 1.0, 1.0, 0.6) # Cyan
        self.goal_color  = (1.0, 1.0, 0.0, 0.7) # Yellow

        self._vis_frame = None
        self._vis_lock = threading.Lock()
        self._last_header = None
        
        self._depth_msg = None
        self.sub_rgb   = rospy.Subscriber(self.image_topic, Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        self.sub_depth = rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1, buff_size=2**24)
        self.sub_info  = rospy.Subscriber(self.cam_info_topic, CameraInfo, self.info_cb, queue_size=1)

        rospy.loginfo("YOLO 3D Pose Node Ready")
        rospy.loginfo("GUI controls: [p]ause, [t]rack Set A (1-50), [b]rack Set B (51-100), [g]oal, [m] lock poses, [c]lear all, [del]ete box")

    def depth_cb(self, msg):
        with self._lock: self._depth_msg = msg

    def info_cb(self, info: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = info.K[0], info.K[4]
            self.cx, self.cy = info.K[2], info.K[5]

    def rgb_cb(self, img_msg: Image):
        if self.fx is None or self._depth_msg is None: return
        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        
        with self._lock: depth_msg_copy = self._depth_msg
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg_copy, "passthrough")
        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) / 1000.0

        lost_trackers = self.update_trackers(img)
        if lost_trackers: self.cleanup_lost_trackers(lost_trackers)

        annotated = img.copy()
        det_array = Detection2DArray(header=img_msg.header)
        marker_array = MarkerArray()
        
        with self._lock:
            current_tracking_boxes = self.tracking_boxes.copy()
            current_tracker_labels = self.tracker_labels.copy()

        for tracker_id, box in current_tracking_boxes.items():
            label = current_tracker_labels.get(tracker_id, "unknown")
            x, y, w, h = [int(v) for v in box]
            
            center_3d, corners_3d = None, None
            if self.poses_locked:
                if label in self.locked_poses:
                    pose_data = self.locked_poses[label]
                    center_3d, corners_3d = pose_data['center'], pose_data['corners']
                else: 
                    live_center, live_corners = self.get_3d_bbox_from_2d(x, y, x + w, y + h, depth_img)
                    if live_center is not None:
                        with self._lock:
                            self.locked_poses[label] = {'center': live_center, 'corners': live_corners}
                        center_3d, corners_3d = live_center, live_corners
            else: 
                center_3d, corners_3d = self.get_3d_bbox_from_2d(x, y, x + w, y + h, depth_img)

            if center_3d is None: continue
            
            ns, color = self.get_viz_properties(label)
            self.draw_tracked_bbox(annotated, box, label, ns)
            
            cube_marker = self.make_3d_bbox_marker(center_3d, corners_3d, img_msg.header, tracker_id, ns, color)
            text_marker = self.make_text_marker(center_3d, img_msg.header, tracker_id, ns, label)
            marker_array.markers.extend([cube_marker, text_marker])

            detection = self.create_detection_msg(img_msg.header, box, label, center_3d)
            if detection: det_array.detections.append(detection)

        if self.poses_locked:
            cv.putText(annotated, "POSES LOCKED", (annotated.shape[1] - 200, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.pub_markers.publish(marker_array)
        self.pub_dets.publish(det_array)
        self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))

        if self.show_gui and not self.paused:
            self._set_vis_frame(annotated, img_msg.header)

    def update_trackers(self, img):
        if self.poses_locked:
            return []
        
        lost_trackers = []
        with self._lock:
            if not self.trackers: return []
            for tracker_id, tracker in self.trackers.items():
                success, box = tracker.update(img)
                if success:
                    self.tracking_boxes[tracker_id] = box
                else:
                    rospy.logwarn(f"Tracking for {self.tracker_labels.get(tracker_id, tracker_id)} lost.")
                    lost_trackers.append(tracker_id)
        return lost_trackers

    def cleanup_lost_trackers(self, tracker_ids):
        with self._lock:
            for tracker_id in tracker_ids:
                label = self.tracker_labels.get(tracker_id)
                self.trackers.pop(tracker_id, None)
                self.tracking_boxes.pop(tracker_id, None)
                self.tracker_labels.pop(tracker_id, None)
                if label:
                    self.locked_poses.pop(label, None)

    def get_3d_bbox_from_2d(self, x1, y1, x2, y2, depth_img):
        points_3d = []
        step = max(1, (x2 - x1) // 5)
        for uu in range(x1, x2, step):
            for vv in range(y1, y2, step):
                p = self.pixel_to_3d_robot(uu, vv, depth_img)
                if p is not None: points_3d.append(p)
        if not points_3d: return None, None
        points_3d = np.array(points_3d)
        min_xyz, max_xyz = points_3d.min(axis=0), points_3d.max(axis=0)
        center = (min_xyz + max_xyz) / 2
        return center, np.vstack([min_xyz, max_xyz])

    def pixel_to_3d_robot(self, u, v, depth_img):
        if not (0 <= v < depth_img.shape[0] and 0 <= u < depth_img.shape[1]): return None
        z = depth_img[v, u]
        if np.isnan(z) or z <= 0.1: return None
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        point_cam = np.array([x, y, z, 1.0])
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.color_frame, rospy.Time(0), rospy.Duration(0.05))
            t, q = trans.transform.translation, trans.transform.rotation
            T = tf_conversions.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            T[:3, 3] = [t.x, t.y, t.z]
            return (T @ point_cam)[:3]
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, f"TF transform failed: {e}")
            return None

    def create_detection_msg(self, header, box, label, center_3d):
        x, y, w, h = box
        detection = Detection2D(header=header)
        
        detection.bbox.center.x, detection.bbox.center.y = x + w/2, y + h/2
        detection.bbox.size_x, detection.bbox.size_y = w, h
        
        hypothesis = ObjectHypothesisWithPose()
        numeric_id = self.label_to_numeric_id(label)
        if numeric_id is None: return None
        
        hypothesis.id = numeric_id
        hypothesis.score = 1.0
        hypothesis.pose.pose.position.x, hypothesis.pose.pose.position.y, hypothesis.pose.pose.position.z = center_3d
        
        detection.results.append(hypothesis)
        return detection

    def _create_tracker(self, box_dict, label, image):
        self.tracker_id_counter += 1
        tracker_id = self.tracker_id_counter
        x1, y1, x2, y2 = box_dict["x1"], box_dict["y1"], box_dict["x2"], box_dict["y2"]
        track_box_tuple = (x1, y1, x2 - x1, y2 - y1)
        tracker = cv.TrackerCSRT_create()
        tracker.init(image, track_box_tuple)
        self.trackers[tracker_id] = tracker
        self.tracking_boxes[tracker_id] = track_box_tuple
        self.tracker_labels[tracker_id] = label
        rospy.loginfo(f"Started tracking new object with label: {label}")

    def gui_tick(self):
        if not self.show_gui: return
        with self._vis_lock:
            live_frame = self._vis_frame.copy() if self._vis_frame is not None else None
        
        vis_frame = self._render_annotation_overlay(self.freeze_img) if self.paused and self.freeze_img is not None else live_frame
        if vis_frame is None:
            cv.waitKey(1)
            return

        cv.imshow("YOLO detections", vis_frame)
        key = cv.waitKey(1) & 0xFF
        
        if self.paused and self.freeze_img is not None:
            if key in (ord('p'), 27): self._toggle_pause(None, None)
            elif key == ord('t'): self.start_tracking_for_set('A')
            elif key == ord('b'): self.start_tracking_for_set('B')
            elif key == ord('g'): self.start_tracking_for_goal()
            elif key in (8, 127):
                if self.ann_boxes:
                    self.ann_boxes.pop()
                    self._active_idx = None
        else:
            if key == ord('p'): self._toggle_pause(live_frame, self._last_header)
            elif key == ord('c'): self.stop_all_tracking()
            elif key == ord('m'):
                self._toggle_pose_lock()
    
    def _toggle_pose_lock(self):
        with self._lock:
            self.poses_locked = not self.poses_locked
            if self.poses_locked:
                rospy.loginfo("--- Object poses LOCKED ---")
                self.locked_poses.clear()
            else:
                rospy.loginfo("--- Object poses UNLOCKED (Live) ---")
                self.locked_poses.clear()

    def start_tracking_for_set(self, set_char):
        if not self.ann_boxes:
            rospy.logwarn(f"Draw boxes before pressing '{set_char.lower()}' to track Set {set_char}.")
            return
        with self._lock:
            # Get all currently used numeric labels
            current_labels_as_int = {int(l) for l in self.tracker_labels.values() if l.isdigit()}

            if set_char == 'A':
                valid_range = range(1, 51)
            else: # set_char == 'B'
                valid_range = range(51, 101)

            for box in self.ann_boxes:
                # Find the first available ID in the specified range
                next_id = -1
                for i in valid_range:
                    if i not in current_labels_as_int:
                        next_id = i
                        break
                
                if next_id != -1:
                    label = str(next_id)
                    self._create_tracker(box, label, self.freeze_img)
                    # Reserve this ID so the next box in the same batch doesn't reuse it
                    current_labels_as_int.add(next_id)
                else:
                    rospy.logwarn(f"No available IDs in Set {set_char}. The set is full.")
                    break # Stop trying to add more boxes if the set is full
        self._toggle_pause(None, None)

    def start_tracking_for_goal(self):
        if len(self.ann_boxes) != 1:
            rospy.logwarn("Please draw exactly ONE box to define a goal.")
            return
        with self._lock:
            label = "G1" if self.goal_toggle else "G2"
            self._create_tracker(self.ann_boxes[0], label, self.freeze_img)
            self.goal_toggle = not self.goal_toggle
        self._toggle_pause(None, None)

    def stop_all_tracking(self):
        with self._lock:
            if not self.trackers: return
            rospy.loginfo("Stopping all manual tracking.")
            self.trackers.clear()
            self.tracking_boxes.clear()
            self.tracker_labels.clear()
            self.goal_toggle = True
            
            self.poses_locked = False
            self.locked_poses.clear()
            
            marker_array = MarkerArray()
            marker = Marker(action=Marker.DELETEALL)
            marker_array.markers.append(marker)
            self.pub_markers.publish(marker_array)

    def get_viz_properties(self, label):
        try:
            num = int(label)
            if 1 <= num <= 50:
                return "set_A", self.set_a_color
            elif 51 <= num <= 100:
                return "set_B", self.set_b_color
        except ValueError:
            pass  # Not a number, check for goal labels below
        
        if "G" in label:
            return "goals", self.goal_color
        
        return "unknown", (0.5, 0.5, 0.5, 0.5)

    def draw_tracked_bbox(self, img, box, label, ns):
        x,y,w,h = [int(v) for v in box]
        color_map = {"set_A": (255,0,0), "set_B": (255,255,0), "goals": (0,255,255)}
        color = color_map.get(ns, (128,128,128))
        cv.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv.putText(img, label, (x, max(0,y-5)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def make_3d_bbox_marker(self, center, corners, header, marker_id, ns, color_rgba):
        min_xyz, max_xyz = corners[0], corners[1]
        marker = Marker(header=header, ns=ns, id=marker_id, type=Marker.CUBE, action=Marker.ADD)
        marker.header.frame_id = self.base_frame
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(0.01, max_xyz[0] - min_xyz[0])
        marker.scale.y = max(0.01, max_xyz[1] - min_xyz[1])
        marker.scale.z = max(0.01, max_xyz[2] - min_xyz[2])
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_rgba
        return marker

    def make_text_marker(self, center, header, marker_id, ns, text):
        marker = Marker(header=header, ns=ns + "_text", id=marker_id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        marker.header.frame_id = self.base_frame
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center[0], center[1], center[2] + 0.08
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 1.0, 1.0, 1.0)
        marker.text = text
        return marker

    def label_to_numeric_id(self, label):
        try:
            # For Set A and B, the label (e.g., "42") is the numeric ID itself.
            num = int(label)
            if 1 <= num <= 100:
                return num
        except ValueError:
            # Not a number, handle goal labels
            if label == "G1": return 9001
            if label == "G2": return 9002
        
        rospy.logwarn_once(f"Could not convert label '{label}' to a numeric ID.")
        return None

    def _mouse_cb(self, event, x, y, flags, param=None):
        if not self.paused: return
        if event == cv.EVENT_LBUTTONDOWN:
            self.ann_boxes.append({"x1": x, "y1": y, "x2": x, "y2": y})
            self._active_idx = len(self.ann_boxes) - 1
            self._drag_start = (x, y)
        elif event == cv.EVENT_MOUSEMOVE and self._drag_start is not None:
            if self._active_idx is not None and self._active_idx < len(self.ann_boxes):
                self.ann_boxes[self._active_idx]['x2'] = x
                self.ann_boxes[self._active_idx]['y2'] = y
        elif event == cv.EVENT_LBUTTONUP and self._drag_start is not None:
            if self._active_idx is not None and self._active_idx < len(self.ann_boxes):
                box = self.ann_boxes[self._active_idx]
                x1_final, x2_final = sorted([self._drag_start[0], x])
                y1_final, y2_final = sorted([self._drag_start[1], y])
                box['x1'], box['x2'], box['y1'], box['y2'] = x1_final, x2_final, y1_final, y2_final
                if (box['x2'] - box['x1']) < 5 or (box['y2'] - box['y1']) < 5:
                    self.ann_boxes.pop(self._active_idx)
            self._drag_start = None
            self._active_idx = None

    def _render_annotation_overlay(self, base_img):
        overlay = base_img.copy()
        for i, bx in enumerate(self.ann_boxes):
            x1, x2 = sorted([bx["x1"], bx["x2"]])
            y1, y2 = sorted([bx["y1"], bx["y2"]])
            color = (0, 165, 255) if i == self._active_idx else (0, 0, 255)
            cv.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        return overlay

    def _toggle_pause(self, img_bgr, header):
        self.paused = not self.paused; self.ann_boxes = []
        if self.paused: 
            self.freeze_img, self.freeze_header = img_bgr.copy(), header
            rospy.loginfo("Paused for annotation.")
        else: 
            self.freeze_img, self.freeze_header = None, None
            rospy.loginfo("Resuming live detection.")

    def _set_vis_frame(self, frame_bgr, header=None):
        with self._vis_lock: self._vis_frame = None if frame_bgr is None else frame_bgr.copy()
        if header is not None: self._last_header = header

if __name__ == "__main__":
    node = Yolo3DPoseNode()
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            node.gui_tick()
            rate.sleep()
    finally:
        if node.show_gui:
            cv.destroyAllWindows()