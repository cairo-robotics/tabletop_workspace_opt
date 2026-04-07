#!/usr/bin/env python3
import os
import yaml
import rospy
import cv2 as cv
import numpy as np
from std_msgs.msg import Float32MultiArray, String 
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped, Point
from cv_bridge import CvBridge
import threading
from collections import deque


# ---- Visualization Parameters (Bar Chart) ----
BAR_W = 900
BAR_H = 500
MARGIN = 30
PROMPT_H = 70

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
        self.lock = threading.Lock() 

        # --- State Variables ---
        self.last_probs = None
        self.last_det_labels = []
        self.top_goal_label = None        
        self.current_tracker_point = None
        self.all_detected_objects = {}
        self.top_goal_pose = None
        self.confirmation_prompt = ""
        self.tracker_path_history = deque(maxlen=200)

        # --- ROS Parameters ---
        self.candidate_source = rospy.get_param("~candidate_source", "detections")
        self.fixed_grasp_stage = rospy.get_param("~fixed_grasp_stage", "pregrasp_pose")
        self.fixed_grasp_yaml = self._resolve_fixed_grasp_yaml()
        self.class_names = rospy.get_param("~class_names",
                                           ["black tea", "chai", "cup", "milk", "meiji panda", "ritz"])
        if isinstance(self.class_names, str):
            self.class_names = [s.strip() for s in self.class_names.strip("[]").split(",") if s.strip()]

        # --- Topic Names ---
        self.det_topic = rospy.get_param("~det_topic", "/yolo_3d_pose/detections")
        self.prob_topic = rospy.get_param("~prob_topic", "/intent_inference/distribution")
        self.top_goal_topic = rospy.get_param("~top_goal_topic", "/intent_inference/top_goal") 
        self.tracker_point_topic = rospy.get_param("~tracker_point_topic", "/intent_inference/current_tracker_point")
        self.top_pose_topic = rospy.get_param("~top_pose_topic", "/intent_inference/top_pose")
        self.confirmation_prompt_topic = rospy.get_param("~confirmation_prompt_topic", "/intent_inference/confirmation_prompt")

        # --- Subscribers ---
        if self.candidate_source == "detections":
            rospy.Subscriber(self.det_topic, Detection2DArray, self.det_cb, queue_size=1)
        elif self.candidate_source == "fixed_grasps":
            self._load_fixed_grasps()
        else:
            raise RuntimeError(f"Unsupported candidate_source '{self.candidate_source}'")
        rospy.Subscriber(self.prob_topic, Float32MultiArray, self.prob_cb, queue_size=1)
        rospy.Subscriber(self.tracker_point_topic, PointStamped, self.tracker_point_cb, queue_size=1)
        rospy.Subscriber(self.top_pose_topic, PoseStamped, self.top_pose_cb, queue_size=1)
        rospy.Subscriber(self.top_goal_topic, String, self.top_goal_cb, queue_size=1)
        rospy.Subscriber(self.confirmation_prompt_topic, String, self.confirmation_prompt_cb, queue_size=1)

        rospy.loginfo("IntentViz is ready.")
        rospy.loginfo(f"Candidate source: {self.candidate_source}")
        if self.candidate_source == "detections":
            rospy.loginfo(f"Listening for detections on: {self.det_topic}")
        else:
            rospy.loginfo(f"Loaded fixed grasps from: {self.fixed_grasp_yaml}")
        rospy.loginfo(f"Listening for probabilities on: {self.prob_topic}")
        rospy.loginfo(f"Listening for top goal label on: {self.top_goal_topic}")

    def _resolve_fixed_grasp_yaml(self):
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_yaml = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")
        return os.path.expanduser(rospy.get_param("~fixed_grasp_yaml", default_yaml))

    def _load_fixed_grasps(self):
        if not os.path.exists(self.fixed_grasp_yaml):
            raise RuntimeError(f"Fixed grasp YAML not found: {self.fixed_grasp_yaml}")

        with open(self.fixed_grasp_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        fixed_objects = {}
        fixed_labels = []
        for grasp in data.get("grasps", []):
            if not isinstance(grasp, dict):
                continue
            grasp_id = str(grasp.get("grasp_id", "")).strip()
            pose_dict = grasp.get(self.fixed_grasp_stage)
            if not grasp_id or not isinstance(pose_dict, dict):
                continue
            position = pose_dict.get("position", [])
            if len(position) != 3:
                continue
            point = Point()
            point.x = float(position[0])
            point.y = float(position[1])
            point.z = float(position[2])
            fixed_objects[grasp_id] = (point, 0.0)
            fixed_labels.append(grasp_id)

        with self.lock:
            self.all_detected_objects = fixed_objects
            self.last_det_labels = fixed_labels

    # -------------------------- Callbacks --------------------------
    def det_cb(self, msg: Detection2DArray):
        """
        Callback for detected objects. Extracts labels and 3D poses.
        This provides *all* detected objects for the map.
        """
        new_objects = {}
        labels_for_bar_chart = []
        for d in msg.detections:
            if d.results:
                hypothesis = d.results[0]
                label = str(hypothesis.id)
                pos = hypothesis.pose.pose.position
                new_objects[label] = (pos, 0.0)
                labels_for_bar_chart.append(label)

        with self.lock:
            self.all_detected_objects = new_objects
            self.last_det_labels = labels_for_bar_chart

    def prob_cb(self, msg: Float32MultiArray):
        """
        Callback for the probability distribution. Updates probabilities for stored objects.
        """
        prob_data = list(msg.data)
        with self.lock:
            if not self.last_det_labels:
                self.last_probs = prob_data
                return

            num_items = min(len(prob_data), len(self.last_det_labels))
            for i in range(num_items):
                label = self.last_det_labels[i]
                prob = prob_data[i]
                if label in self.all_detected_objects:
                    pos, _ = self.all_detected_objects[label]
                    self.all_detected_objects[label] = (pos, prob)

            self.last_probs = prob_data

    def top_goal_cb(self, msg: String):
        """
        Callback for the top inferred goal label string.
        """
        self.top_goal_label = msg.data

    def tracker_point_cb(self, msg: PointStamped):
        """Callback for the current tracker point (hand or end-effector)."""
        self.current_tracker_point = msg
        self.tracker_path_history.append((msg.point.x, msg.point.y))

    def top_pose_cb(self, msg: PoseStamped):
        """Callback for the top inferred goal pose."""
        self.top_goal_pose = msg

    def confirmation_prompt_cb(self, msg: String):
        self.confirmation_prompt = msg.data

    # -------------------------- Bar Chart Logic --------------------------
    def make_bar_canvas(self):
        """Creates the bar chart visualization as a NumPy image."""
        canvas_h = BAR_H + 2 * MARGIN + PROMPT_H
        canvas = np.zeros((canvas_h, BAR_W + 2 * MARGIN, 3), np.uint8)
        canvas[:] = (30, 30, 30)
        chart_top = MARGIN + PROMPT_H
        baseline_y = chart_top + BAR_H

        with self.lock:
            names = self.last_det_labels[:] if self.last_det_labels is not None else []
            probs = self.last_probs[:] if self.last_probs is not None else []

        if len(names) != len(probs):
            return canvas # Return early if data is inconsistent

        cv.putText(canvas, "Intent Probability", (MARGIN, MARGIN + 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

        if self.confirmation_prompt:
            cv.rectangle(canvas, (MARGIN, MARGIN + 28), (BAR_W + MARGIN, MARGIN + 28 + PROMPT_H - 16), (20, 70, 120), -1)
            cv.putText(canvas, self.confirmation_prompt, (MARGIN + 10, MARGIN + 28 + 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if not names:
            cv.putText(canvas, "Awaiting detections...", (MARGIN, chart_top + BAR_H // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            return canvas

        bar_gap = 10
        num_bars = len(names)
        bar_w = max(1, (BAR_W - bar_gap * (num_bars - 1)) // num_bars)

        for i, (name, p) in enumerate(zip(names, probs)):
            h = int(np.clip(p, 0.0, 1.0) * BAR_H)
            x0 = MARGIN + i * (bar_w + bar_gap)
            y0 = baseline_y - h
            bar_color = (80, 190, 250) # Light blue
            if self.top_goal_label and self.top_goal_label == name:
                 bar_color = (0, 255, 0) # Green for top goal

            cv.rectangle(canvas, (x0, y0), (x0 + bar_w, baseline_y), bar_color, -1)

            text_size, _ = cv.getTextSize(name, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_x = x0 + (bar_w - text_size[0]) // 2
            cv.putText(canvas, name, (text_x, baseline_y + 18),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cv.putText(canvas, f"{p:.2f}", (x0 + 4, max(MARGIN + 30, y0 - 8)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

        return canvas

    # -------------------------- 2D Map Logic --------------------------
    def _project_to_map(self, x_m, y_m):
        """Converts meters (base_frame X, Y) to pixels (map_canvas u, v)."""
        u = int(MAP_W / 2 - (y_m - MAP_ORIGIN_Y_M) / MAP_SCALE_M_PER_PX)
        v = int(MAP_H / 2 - (x_m - MAP_ORIGIN_X_M) / MAP_SCALE_M_PER_PX)
        return u, v

    def make_map_canvas(self):
        """Creates the 2D map visualization as a NumPy image."""
        canvas = np.zeros((MAP_H, MAP_W, 3), np.uint8)
        canvas[:] = (50, 50, 50)

        # Draw grid lines and origin
        cv.line(canvas, self._project_to_map(MAP_ORIGIN_X_M, -100), self._project_to_map(MAP_ORIGIN_X_M, 100), (80, 80, 80), 1)
        cv.line(canvas, self._project_to_map(-100, MAP_ORIGIN_Y_M), self._project_to_map(100, MAP_ORIGIN_Y_M), (80, 80, 80), 1)
        cv.circle(canvas, self._project_to_map(MAP_ORIGIN_X_M, MAP_ORIGIN_Y_M), 5, (0, 0, 255), -1)

        # Draw all detected objects
        with self.lock:
            objects_copy = self.all_detected_objects.copy()

        for label, (pos, prob) in objects_copy.items():
            u, v = self._project_to_map(pos.x, pos.y)
            intensity = int(255 * (prob * 0.8 + 0.2))
            obj_color = (0, intensity, intensity)

            if self.top_goal_label and self.top_goal_label == label:
                obj_color = (0, 255, 0) # Green if it's the top goal

            cv.circle(canvas, (u, v), 10, obj_color, -1)
            cv.putText(canvas, f"{label} ({prob:.2f})", (u + 15, v + 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, obj_color, 1)

        # Draw tracker path history
        if len(self.tracker_path_history) > 1:
            points_px = [self._project_to_map(p[0], p[1]) for p in self.tracker_path_history]
            for i in range(1, len(points_px)):
                cv.line(canvas, points_px[i-1], points_px[i], (255, 100, 0), 2)

        # Draw current tracker point
        if self.current_tracker_point:
            x, y = self.current_tracker_point.point.x, self.current_tracker_point.point.y
            u, v = self._project_to_map(x, y)
            cv.circle(canvas, (u, v), 8, (255, 255, 255), -1)
            cv.putText(canvas, f"Tracker ({x:.2f},{y:.2f})", (u + 15, v - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv.putText(canvas, "2D Workspace Map (X-Y Plane of Base Frame)", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

        return canvas

    def spin(self):
        """Main loop to generate and display the visualization."""
        rate = rospy.Rate(30)
        cv.namedWindow("Intent Probability", cv.WINDOW_NORMAL)
        cv.namedWindow("2D Workspace Map", cv.WINDOW_NORMAL)

        while not rospy.is_shutdown():
            bar_canvas = self.make_bar_canvas()
            map_canvas = self.make_map_canvas()

            cv.imshow("Intent Probability", bar_canvas)
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
