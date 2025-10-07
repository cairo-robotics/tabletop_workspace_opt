#!/usr/bin/env python3
"""YOLO 3D Pose Node

Subscribes to RGB, depth, and camera info topics; runs YOLO object detection;
projects detections to 3D using depth and camera intrinsics; publishes
Detection2DArray, 3D markers, and an annotated image. Supports optional
manual multi-object tracking via GUI interactions.
"""
import os
import numpy as np
import rospy
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import tf2_ros
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

        # Topics and frames
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/camera/color/camera_info")
        self.color_frame = rospy.get_param("~color_optical_frame", "camera_color_optical_frame")
        self.base_frame  = rospy.get_param("~base_frame", "world")

        # YOLO config
        model_path = rospy.get_param("~model", "yolov8m.pt")
        self.conf_thres = rospy.get_param("~conf_thres", 0.4)
        self.iou_thres  = rospy.get_param("~iou_thres", 0.3)
        self.show_gui   = rospy.get_param("~show_gui", True)

        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to("cuda")
        self.model.fuse()
        self.class_names = self.model.names

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        if self.show_gui:
            cv.namedWindow("YOLO detections", cv.WINDOW_NORMAL)
            cv.setMouseCallback("YOLO detections", self._mouse_cb)

        # Subscribers
        self._lock = threading.Lock()
        self._depth_msg = None
        self.sub_rgb   = rospy.Subscriber(self.image_topic, Image, self.rgb_cb, queue_size=1, buff_size=2**24)
        self.sub_depth = rospy.Subscriber(self.depth_topic, Image, self.depth_cb, queue_size=1, buff_size=2**24)
        self.sub_info  = rospy.Subscriber(self.cam_info_topic, CameraInfo, self.info_cb, queue_size=1)

        # Publishers
        self.pub_dets    = rospy.Publisher("~detections", Detection2DArray, queue_size=10)
        self.pub_poses   = rospy.Publisher("~object_poses", PoseStamped, queue_size=10)
        self.pub_annotated = rospy.Publisher("~annotated_image", Image, queue_size=1)
        self.pub_markers   = rospy.Publisher("~object_markers", MarkerArray, queue_size=10)
        
        # --- Manual annotation state ---
        self.annot_dir = os.path.expanduser("~/yolo_manual_labels")
        os.makedirs(self.annot_dir, exist_ok=True)
        self.paused = False
        self.freeze_img = None
        self.freeze_header = None
        self.ann_boxes = []
        self._drag_start = None
        self._active_idx = None

        # State variables for multi-object tracking
        self.trackers = {} # Dict to hold tracker objects {tracker_id: tracker}
        self.tracking_boxes = {} # Dict to hold current bounding boxes {tracker_id: (x,y,w,h)}
        self.tracker_id_counter = 0

        self._vis_frame = None
        self._vis_lock = threading.Lock()
        self._last_header = None
        rospy.loginfo("YOLO 3D Pose Node Ready")
        rospy.loginfo("GUI controls: [p]ause/resume, [s]ave annotations, [t]rack selected boxes, [c]lear all tracking, [del]ete box")

    def depth_cb(self, msg):
        with self._lock:
            self._depth_msg = msg

    def info_cb(self, info: CameraInfo):
        if self.fx is None:
            self.fx = info.K[0]; self.fy = info.K[4]
            self.cx = info.K[2]; self.cy = info.K[5]

    def rgb_cb(self, img_msg: Image):
        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        if self.fx is None or self._depth_msg is None: return

        with self._lock:
            depth_img = self.bridge.imgmsg_to_cv2(self._depth_msg, "passthrough")
        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32)/1000.0

        if img is None or img.size == 0: return

        # Update loop for multiple trackers
        with self._lock:
            if self.trackers:
                lost_trackers = []
                for tracker_id, tracker in self.trackers.items():
                    success, box = tracker.update(img)
                    if success:
                        self.tracking_boxes[tracker_id] = box
                    else:
                        rospy.logwarn(f"Tracking for {tracker_id} lost.")
                        lost_trackers.append(tracker_id)
                
                # Remove lost trackers
                for tracker_id in lost_trackers:
                    self.trackers.pop(tracker_id, None)
                    self.tracking_boxes.pop(tracker_id, None)

        # YOLO inference
        results = self.model.predict(img, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        r = results[0]
        xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes else []
        confs = r.boxes.conf.cpu().numpy() if r.boxes else []
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else []

        annotated = img.copy()
        det_array = Detection2DArray(header=img_msg.header)
        
        # Create a single MarkerArray for all objects
        master_marker_array = MarkerArray()
        marker_id = 0

        # Draw and publish info for all tracked objects
        with self._lock:
            if self.tracking_boxes:
                for tracker_id, box in self.tracking_boxes.items():
                    x, y, w, h = [int(v) for v in box]
                    p1 = (x, y)
                    p2 = (x + w, y + h)
                    cv.rectangle(annotated, p1, p2, (255, 0, 0), 2, 1) # Blue for tracker
                    cv.putText(annotated, str(tracker_id), (p1[0], p1[1] - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    center, corners = self.get_3d_bbox_from_2d(x, y, x+w, y+h, depth_img)
                    if center is not None:
                        marker = self.make_3d_bbox_marker(center, corners, img_msg.header, marker_id, "tracked_objects", (0.0, 0.0, 1.0, 0.6))
                        master_marker_array.markers.append(marker)
                        marker_id += 1
                    
                    # Create Detection2D for tracked object
                    detection = Detection2D()
                    detection.header = img_msg.header
                    
                    # Set bounding box
                    detection.bbox.center.x = x + w/2
                    detection.bbox.center.y = y + h/2
                    detection.bbox.size_x = w
                    detection.bbox.size_y = h
                    
                    # Set object hypothesis (using tracker_id as label)
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.id = marker_id
                    hypothesis.pose.pose.position.x = marker.pose.position.x
                    hypothesis.pose.pose.position.y = marker.pose.position.y
                    hypothesis.pose.pose.position.z = marker.pose.position.z
                    detection.results.append(hypothesis)
                    
                    det_array.detections.append(detection)

        # Process YOLO detections
        for i in range(len(xyxy)):
            cls_name = self.class_names.get(cls[i], str(cls[i]))
            
            x1, y1, x2, y2 = xyxy[i].astype(int)
            score = float(confs[i])
            self.draw_bbox(annotated, xyxy[i], cls_name, score, color=(0, 255, 0)) # Green for YOLO

            # ... (Detection2D and other info publishing can be added here if needed) ...

            center, corners = self.get_3d_bbox_from_2d(x1, y1, x2, y2, depth_img)
            if center is None: continue

            marker = self.make_3d_bbox_marker(center, corners, img_msg.header, marker_id, "yolo_objects", (0.0, 1.0, 0.0, 0.5))
            master_marker_array.markers.append(marker)
            marker_id += 1

            # commented out to only use manually tracked boxes for now; uncomment to use YOLO detections
            # detection = Detection2D()
            # detection.header = img_msg.header
            # detection.bbox.center.x = x1 + (x2 - x1)/2
            # detection.bbox.center.y = y1 + (y2 - y1)/2

            # hypothesis = ObjectHypothesisWithPose()
            # hypothesis.id = marker_id
            # hypothesis.pose.pose.position.x = marker.pose.position.x
            # hypothesis.pose.pose.position.y = marker.pose.position.y
            # hypothesis.pose.pose.position.z = marker.pose.position.z
            # detection.results.append(hypothesis)
            # det_array.detections.append(detection)

        # Publish all markers at once
        self.pub_markers.publish(master_marker_array)
        self.pub_dets.publish(det_array)

        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        annotated_msg.header = img_msg.header
        self.pub_annotated.publish(annotated_msg)

        if self.show_gui and not self.paused:
            self._set_vis_frame(annotated, img_msg.header)

    def get_3d_bbox_from_2d(self, x1, y1, x2, y2, depth_img):
        points_3d = []
        step = max(1, (x2 - x1) // 5)
        for uu in range(x1, x2, step):
            for vv in range(y1, y2, step):
                p = self.pixel_to_3d_robot(uu, vv, depth_img)
                if p is not None:
                    points_3d.append(p)
        if not points_3d: return None, None
        
        points_3d = np.array(points_3d)
        min_xyz = points_3d.min(axis=0)
        max_xyz = points_3d.max(axis=0)
        center = (min_xyz + max_xyz) / 2
        return center, np.vstack([min_xyz, max_xyz])

    def make_3d_bbox_marker(self, center, corners, header, marker_id, namespace, color_rgba):
        min_xyz, max_xyz = corners[0], corners[1]
        
        # Initialize the marker with the image header to get the correct timestamp
        marker = Marker(header=header, ns=namespace, id=marker_id, type=Marker.CUBE, action=Marker.ADD)
        
        # *** THE FIX: Override the frame_id to match the coordinate frame of the pose ***
        marker.header.frame_id = self.base_frame
        
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(0.01, max_xyz[0] - min_xyz[0])
        marker.scale.y = max(0.01, max_xyz[1] - min_xyz[1])
        marker.scale.z = max(0.01, max_xyz[2] - min_xyz[2])
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_rgba
        return marker

    def pixel_to_3d_robot(self, u, v, depth_img):
        if u < 0 or u >= depth_img.shape[1] or v < 0 or v >= depth_img.shape[0]: return None
        z = depth_img[v, u]
        if np.isnan(z) or z <= 0.01: return None
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        point_cam = np.array([x, y, z, 1.0])
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.color_frame, rospy.Time(0), rospy.Duration(0.1))
            t = trans.transform.translation; q = trans.transform.rotation
            import tf_conversions
            T = tf_conversions.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
            T[:3,3] = [t.x, t.y, t.z]
            return (T @ point_cam)[:3]
        except Exception:
            # Fallback to camera frame if TF fails
            return point_cam[:3]

    def draw_bbox(self, img, xyxy, label, score, color=(0,255,0)):
        x1,y1,x2,y2 = [int(v) for v in xyxy]
        cv.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv.putText(img, f"{label} {score:.2f}", (x1, max(0,y1-5)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
    def _mouse_cb(self, event, x, y, flags, param=None):
        if not self.paused or self.freeze_img is None: return
        if event == cv.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self.ann_boxes.append({"x1": x, "y1": y, "x2": x, "y2": y, "label": "unknown"})
            self._active_idx = len(self.ann_boxes) - 1
        elif event == cv.EVENT_MOUSEMOVE and self._drag_start is not None:
            self.ann_boxes[self._active_idx].update({"x2": x, "y2": y})
        elif event == cv.EVENT_LBUTTONUP and self._drag_start is not None:
            bx = self.ann_boxes[self._active_idx]
            x1, y1 = self._drag_start; x2, y2 = x, y
            self._drag_start = None
            bx["x1"], bx["x2"] = sorted([x1, x2])
            bx["y1"], bx["y2"] = sorted([y1, y2])
            if bx["x2"] - bx["x1"] < 3 or bx["y2"] - bx["y1"] < 3:
                self.ann_boxes.pop(); self._active_idx = None

    def _render_annotation_overlay(self, base_img):
        overlay = base_img.copy()
        for i, bx in enumerate(self.ann_boxes):
            x1, y1, x2, y2 = bx["x1"], bx["y1"], bx["x2"], bx["y2"]
            color = (0, 165, 255) if i == self._active_idx else (0, 0, 255) # orange/red
            cv.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv.putText(overlay, f'{bx.get("label","?")}', (x1, max(0, y1-6)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return overlay

    def _toggle_pause(self, img_bgr, header):
        if not self.paused:
            print("Pausing for annotation/tracking selection.")
            self.paused = True
            self.freeze_img = img_bgr.copy()
            self.freeze_header = header
            self.ann_boxes = []
        else:
            print("Resuming live detection.")
            self.paused = False
            self.freeze_img = None
            self.freeze_header = None
            self.ann_boxes = []

    def start_multi_tracking(self, boxes_to_track, image):
        with self._lock:
            for box_dict in boxes_to_track:
                self.tracker_id_counter += 1
                tracker_id = self.tracker_id_counter

                x1, y1, x2, y2 = box_dict["x1"], box_dict["y1"], box_dict["x2"], box_dict["y2"]
                track_box_tuple = (x1, y1, x2 - x1, y2 - y1)

                tracker = cv.TrackerCSRT_create()
                tracker.init(image, track_box_tuple)
                
                self.trackers[tracker_id] = tracker
                self.tracking_boxes[tracker_id] = track_box_tuple
            
            rospy.loginfo(f"Started tracking {len(boxes_to_track)} new object(s).")

    def stop_all_tracking(self):
        with self._lock:
            if not self.trackers: return
            rospy.loginfo("Stopping all manual tracking.")
            self.trackers.clear()
            self.tracking_boxes.clear()
            # Also clear the markers in RViz
            marker_array = MarkerArray()
            marker = Marker(header=self._last_header, ns="tracked_objects", id=0, action=Marker.DELETEALL)
            marker.header.frame_id = self.base_frame # Important to specify frame here too
            marker_array.markers.append(marker)
            self.pub_markers.publish(marker_array)

    def _save_annotations(self):
        # This function can be implemented to save manual annotations if needed
        pass

    def _set_vis_frame(self, frame_bgr, header=None):
        with self._vis_lock:
            self._vis_frame = None if frame_bgr is None else frame_bgr.copy()
        if header is not None: self._last_header = header

    def gui_tick(self):
        if not self.show_gui: return
        with self._vis_lock:
            live = None if self._vis_frame is None else self._vis_frame.copy()
        vis = self._render_annotation_overlay(self.freeze_img) if self.paused and self.freeze_img is not None else live
        if vis is None:
            cv.waitKey(1)
            return

        cv.imshow("YOLO detections", vis)
        key = cv.waitKey(1) & 0xFF
        
        if self.paused and self.freeze_img is not None:
            if key == ord('p') or key == 27: self._toggle_pause(None, None) # p or Esc
            elif key == ord('s'): self._save_annotations()
            elif key == ord('t'):
                if self.ann_boxes:
                    self.start_multi_tracking(self.ann_boxes, self.freeze_img)
                    self._toggle_pause(None, None) # Resume after starting track
                else:
                    rospy.logwarn("Draw one or more boxes first before pressing 't' to track.")
            elif key in (8, 127): # backspace/delete
                if self.ann_boxes:
                    self.ann_boxes.pop()
                    self._active_idx = len(self.ann_boxes) - 1 if self.ann_boxes else None
            # Number key labeling logic can be added here if needed
        else: # Live mode
            if key == ord('p'): self._toggle_pause(live, self._last_header)
            elif key == ord('c'): self.stop_all_tracking()

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