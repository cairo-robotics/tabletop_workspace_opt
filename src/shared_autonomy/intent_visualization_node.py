#!/usr/bin/env python3
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import Float32MultiArray
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped, Point
from cv_bridge import CvBridge
import json
import math
import threading

from collections import deque


# ---- Visualization Parameters (Bar Chart) ----
BAR_W = 900
BAR_H = 500
MARGIN = 30

# ---- Visualization Parameters (2D Map) ----
MAP_W = 700  # Width of the 2D map window in pixels
MAP_H = 700  # Height of the 2D map window in pixels
MAP_SCALE_M_PER_PX = 0.005 # 1 pixel = 5 mm. Adjust this to cover your workspace.
MAP_ORIGIN_X_M = 0.0  # X-coordinate in meters of the center of the map in base_frame
MAP_ORIGIN_Y_M = 0.5  # Y-coordinate in meters of the center of the map in base_frame


class IntentViz:
    def __init__(self):
        """Initializes the visualization node."""
        rospy.init_node("intent_viz")
        self.bridge = CvBridge()

        # --- State Variables (Bar Chart) ---
        self.last_probs = None      # list[float] from the inference node
        self.last_det_labels = []   # list[str] from the detection node
        self.last_annot_img = None  # BGR image from the YOLO node for combined view

        # --- State Variables (2D Map) ---
        self.current_tracker_point = None # geometry_msgs.msg.PointStamped
        self.all_detected_objects = {}    # {label: (Point, prob)}
        self.top_goal_pose = None         # geometry_msgs.msg.PoseStamped
        self.tracker_path_history = deque(maxlen=200) # Store (x, y) for path

        # --- ROS Parameters ---
        self.class_names = rospy.get_param("~class_names",
                                           ["black tea", "chai", "cup", "milk", "meiji panda", "ritz"])
        if isinstance(self.class_names, str):
            self.class_names = [s.strip() for s in self.class_names.strip("[]").split(",") if s.strip()]
        self.id_to_class = {name: name for name in self.class_names} # Simplify map for string IDs

        self.det_topic = rospy.get_param("~det_topic", "/yolo_3d_pose/detections")
        self.prob_topic = rospy.get_param("~prob_topic", "/intent_inference/distribution")
        self.img_topic = rospy.get_param("~img_topic", "/yolo_3d_pose/annotated_image")
        self.tracker_point_topic = rospy.get_param("~tracker_point_topic", "/intent_inference/current_tracker_point")
        self.top_pose_topic = rospy.get_param("~top_pose_topic", "/intent_inference/top_pose")
        
        self.show_combined = bool(rospy.get_param("~show_combined", True))
        self.show_map = bool(rospy.get_param("~show_map", True))

        # --- Map Parameters ---
        self.map_w = rospy.get_param("~map_width", MAP_W)
        self.map_h = rospy.get_param("~map_height", MAP_H)
        self.map_scale = rospy.get_param("~map_scale_m_per_px", MAP_SCALE_M_PER_PX)
        self.map_origin_x_m = rospy.get_param("~map_origin_x_m", MAP_ORIGIN_X_M)
        self.map_origin_y_m = rospy.get_param("~map_origin_y_m", MAP_ORIGIN_Y_M)


        # --- Subscribers ---
        rospy.Subscriber(self.det_topic, Detection2DArray, self.det_cb, queue_size=1)
        rospy.Subscriber(self.prob_topic, Float32MultiArray, self.prob_cb, queue_size=1)
        rospy.Subscriber(self.img_topic, Image, self.img_cb, queue_size=1)
        rospy.Subscriber(self.tracker_point_topic, PointStamped, self.tracker_point_cb, queue_size=1)
        rospy.Subscriber(self.top_pose_topic, PoseStamped, self.top_pose_cb, queue_size=1)

        rospy.loginfo("IntentViz is ready.")
        rospy.loginfo(f"Listening for detections on: {self.det_topic}")
        rospy.loginfo(f"Listening for probabilities on: {self.prob_topic}")
        rospy.loginfo(f"Listening for tracker points on: {self.tracker_point_topic}")
        rospy.loginfo(f"Listening for top goal pose on: {self.top_pose_topic}")


    # -------------------------- Callbacks --------------------------

    def det_cb(self, msg: Detection2DArray):
        """
        Callback for detected objects. Extracts labels and 3D poses.
        This provides *all* detected objects for the map, not just the labels for the bar chart.
        """
        new_objects = {}
        labels_for_bar_chart = [] # This is what the prob_cb expects in order
        for d in msg.detections:
            if d.results:
                hypothesis = d.results[0]
                label = str(hypothesis.id) # Assuming ID is already the string label
                pos = hypothesis.pose.pose.position
                new_objects[label] = (pos, 0.0) # Store with a placeholder prob
                labels_for_bar_chart.append(label)
        
        with threading.Lock(): # Protect shared state for object poses and probabilities
            self.all_detected_objects = new_objects
            self.last_det_labels = labels_for_bar_chart


    def prob_cb(self, msg: Float32MultiArray):
        """
        Callback for the probability distribution.
        Updates the probabilities of the objects stored from det_cb.
        """
        if not self.last_det_labels:
            self.last_probs = list(msg.data) # Store raw probs if no labels yet
            return

        prob_data = list(msg.data)
        
        with threading.Lock():
            # Update probabilities for known objects
            num_items = min(len(prob_data), len(self.last_det_labels))
            for i in range(num_items):
                label = self.last_det_labels[i]
                prob = prob_data[i]
                if label in self.all_detected_objects:
                    pos, _ = self.all_detected_objects[label]
                    self.all_detected_objects[label] = (pos, prob)
            
            self.last_probs = prob_data # Also keep for bar chart


    def img_cb(self, msg: Image):
        """Callback for the annotated image from the detector."""
        try:
            self.last_annot_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting annotated image: {e}")

    def tracker_point_cb(self, msg: PointStamped):
        """Callback for the current tracker point (hand or end-effector)."""
        self.current_tracker_point = msg
        self.tracker_path_history.append((msg.point.x, msg.point.y))

    def top_pose_cb(self, msg: PoseStamped):
        """Callback for the top inferred goal pose."""
        self.top_goal_pose = msg

    # -------------------------- Bar Chart Logic --------------------------

    def get_full_distribution_for_bar_chart(self) -> dict:
        """
        Creates a dictionary mapping every possible class name to its inferred probability.
        Uses `self.last_det_labels` and `self.last_probs`.
        """
        prob_map = {}
        with threading.Lock():
            if self.last_probs and self.last_det_labels:
                num_items = min(len(self.last_probs), len(self.last_det_labels))
                prob_map = {self.last_det_labels[i]: self.last_probs[i] for i in range(num_items)}
        
        full_dist = {}
        for name in self.class_names:
            full_dist[name] = prob_map.get(name, 0.0)
        return full_dist

    def make_bar_canvas(self):
        """Creates the bar chart visualization as a NumPy image."""
        canvas = np.zeros((BAR_H + 2 * MARGIN, BAR_W + 2 * MARGIN, 3), np.uint8)
        canvas[:] = (30, 30, 30) # Dark grey background

        full_dist = self.get_full_distribution_for_bar_chart()
        
        names = self.class_names
        probs = np.array([full_dist.get(name, 0.0) for name in names])
        probs_sum = probs.sum()
        if probs_sum > 1e-8: # Avoid division by zero
            probs /= probs_sum
        else:
            probs = np.zeros_like(probs) # All zeros if no probs or sum is zero

        cv.putText(canvas, "Intent Probability", (MARGIN, MARGIN + 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

        if not names:
            cv.putText(canvas, "No classes defined in ~class_names.", (MARGIN, BAR_H // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            return canvas

        bar_gap = 10
        num_bars = len(names)
        bar_w = max(1, (BAR_W - bar_gap * (num_bars - 1)) // num_bars)
        baseline_y = BAR_H + MARGIN

        for i, (name, p) in enumerate(zip(names, probs)):
            h = int(np.clip(p, 0.0, 1.0) * BAR_H)
            x0 = MARGIN + i * (bar_w + bar_gap)
            y0 = baseline_y - h
            
            # Bar color: Green for top goal, blue otherwise
            bar_color = (80, 190, 250) # Light blue
            if self.top_goal_pose and self.top_goal_pose.header.frame_id == name: # assuming top_goal.header.frame_id stores the label
                 bar_color = (0, 255, 0) # Green for top goal
            
            cv.rectangle(canvas, (x0, y0), (x0 + bar_w, baseline_y), bar_color, -1)
            
            text_size, _ = cv.getTextSize(name, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_x = x0 + (bar_w - text_size[0]) // 2
            cv.putText(canvas, name, (text_x, baseline_y + 18),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

        return canvas

    # -------------------------- 2D Map Logic --------------------------

    def _project_to_map(self, x_m, y_m):
        """Converts meters (base_frame X, Y) to pixels (map_canvas u, v)."""
        u = int(self.map_w / 2 - (y_m - self.map_origin_y_m) / self.map_scale) # Y in ROS is typically left/right, becomes X on map
        v = int(self.map_h / 2 - (x_m - self.map_origin_x_m) / self.map_scale) # X in ROS is typically forward, becomes -Y on map
        return u, v

    def make_map_canvas(self):
        """Creates the 2D map visualization as a NumPy image."""
        canvas = np.zeros((self.map_h, self.map_w, 3), np.uint8)
        canvas[:] = (50, 50, 50) # Darker grey background

        # Draw grid lines for reference (optional)
        cv.line(canvas, self._project_to_map(self.map_origin_x_m, -100), self._project_to_map(self.map_origin_x_m, 100), (80, 80, 80), 1)
        cv.line(canvas, self._project_to_map(-100, self.map_origin_y_m), self._project_to_map(100, self.map_origin_y_m), (80, 80, 80), 1)
        cv.circle(canvas, self._project_to_map(self.map_origin_x_m, self.map_origin_y_m), 5, (0, 0, 255), -1) # Origin (red dot)

        # Draw all detected objects
        with threading.Lock():
            for label, (pos, prob) in self.all_detected_objects.items():
                u, v = self._project_to_map(pos.x, pos.y)
                
                # Color based on probability (brighter for higher prob)
                intensity = int(255 * (prob * 0.8 + 0.2)) # Min 20% brightness
                obj_color = (0, intensity, intensity) # Yellow-ish for objects
                if self.top_goal_pose and self.top_goal_pose.pose.position == pos and self.top_goal_pose.header.frame_id == label:
                    obj_color = (0, 255, 0) # Green if it's the top goal
                
                cv.circle(canvas, (u, v), 10, obj_color, -1)
                cv.putText(canvas, f"{label} ({prob:.2f})", (u + 15, v + 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, obj_color, 1)

        # Draw tracker path history
        if len(self.tracker_path_history) > 1:
            points_px = [self._project_to_map(p[0], p[1]) for p in self.tracker_path_history]
            for i in range(1, len(points_px)):
                cv.line(canvas, points_px[i-1], points_px[i], (255, 100, 0), 2) # Blue path

        # Draw current tracker point
        if self.current_tracker_point:
            x, y = self.current_tracker_point.point.x, self.current_tracker_point.point.y
            u, v = self._project_to_map(x, y)
            cv.circle(canvas, (u, v), 8, (255, 255, 255), -1) # White circle for current point
            cv.putText(canvas, f"Tracker ({x:.2f},{y:.2f})", (u + 15, v - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv.putText(canvas, "2D Workspace Map (X-Y Plane of Base Frame)", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

        return canvas

    def spin(self):
        """Main loop to generate and display the visualization."""
        rate = rospy.Rate(30)
        
        # Creates window names ahead of time
        cv.namedWindow("Intent Probability", cv.WINDOW_NORMAL)
        if self.show_combined:
            cv.namedWindow("Annotated Camera + Bar Chart", cv.WINDOW_NORMAL)
        if self.show_map:
            cv.namedWindow("2D Workspace Map", cv.WINDOW_NORMAL)

        while not rospy.is_shutdown():
            # 1. Always create and show the bar chart in its own window
            bar_canvas = self.make_bar_canvas()
            cv.imshow("Intent Probability", bar_canvas)


            # 3. Optionally show the 2D map in a third window
            if self.show_map:
                map_canvas = self.make_map_canvas()
                cv.imshow("2D Workspace Map", map_canvas)

            cv.waitKey(1)
            rate.sleep()
        
        cv.destroyAllWindows()


def main():
    try:
        IntentViz().spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()