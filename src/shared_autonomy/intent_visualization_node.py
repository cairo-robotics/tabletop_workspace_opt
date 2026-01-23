#!/usr/bin/env python3
import rospy
import cv2 as cv
import numpy as np
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from collections import deque
import threading


# ---- Visualization Parameters (Bar Chart) ----
BAR_W = 900
BAR_H = 500
MARGIN = 30

# ---- Visualization Parameters (2D Map) ----
MAP_W = 700
MAP_H = 700
MAP_SCALE_M_PER_PX = 0.005
MAP_ORIGIN_X_M = 0.0
MAP_ORIGIN_Y_M = 0.5


class IntentViz:
    def __init__(self):
        rospy.init_node("intent_viz")
        self.lock = threading.Lock()

        # --- State Variables ---
        self.intent_objects = []  # Will store list of dicts: {'label': str, 'pos': Point, 'prob': float}
        self.current_tracker_point = None
        self.tracker_path_history = deque(maxlen=200)

        # --- Subscribers ---
        # THIS IS NOW THE SINGLE SOURCE OF TRUTH for intent data
        rospy.Subscriber("/intent_inference/distribution", Detection2DArray, self.dist_cb, queue_size=1)
        rospy.Subscriber("/intent_inference/current_tracker_point", PointStamped, self.tracker_point_cb, queue_size=1)
        
        rospy.loginfo("IntentViz is ready.")

    # -------------------------- Callbacks --------------------------

    def dist_cb(self, msg: Detection2DArray):
        """Callback for the full intent distribution."""
        new_objects = []
        for det in msg.detections:
            if not det.results: continue
            hyp = det.results[0]
            
            # The viz node also needs to decode the ID
            label = self._numeric_id_to_label(hyp.id)
            
            new_objects.append({
                'label': label,
                'pos': hyp.pose.pose.position,
                'prob': hyp.score
            })
        
        # Sort objects by label so the bar chart is consistent
        new_objects.sort(key=lambda x: x['label'])

        with self.lock:
            self.intent_objects = new_objects

    def tracker_point_cb(self, msg: PointStamped):
        """Callback for the current tracker point (hand or end-effector)."""
        self.current_tracker_point = msg
        self.tracker_path_history.append((msg.point.x, msg.point.y))

    # --- Helper to decode IDs, copied from intent_inference node ---
    def _numeric_id_to_label(self, num_id):
        if 1 <= num_id <= 100: return str(num_id)
        elif num_id == 9001: return "G1"
        elif num_id == 9002: return "G2"
        return f"unknown_{num_id}"

    # -------------------------- Bar Chart Logic --------------------------

    def make_bar_canvas(self):
        """Creates the bar chart visualization as a NumPy image."""
        canvas = np.zeros((BAR_H + 2 * MARGIN, BAR_W + 2 * MARGIN, 3), np.uint8)
        canvas[:] = (30, 30, 30)

        with self.lock:
            objects_copy = self.intent_objects[:]

        # Find the object with the highest probability
        top_object_label = None
        if objects_copy:
            top_object = max(objects_copy, key=lambda x: x['prob'])
            top_object_label = top_object['label']

        cv.putText(canvas, "Intent Probability", (MARGIN, MARGIN + 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

        if not objects_copy:
            cv.putText(canvas, "Awaiting intent distribution...", (MARGIN, BAR_H // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            return canvas

        bar_gap = 10
        num_bars = len(objects_copy)
        bar_w = max(1, (BAR_W - bar_gap * (num_bars - 1)) // num_bars)
        baseline_y = BAR_H + MARGIN

        for i, obj in enumerate(objects_copy):
            name, p = obj['label'], obj['prob']
            h = int(np.clip(p, 0.0, 1.0) * BAR_H)
            x0 = MARGIN + i * (bar_w + bar_gap)
            y0 = baseline_y - h

            bar_color = (80, 190, 250)
            if top_object_label and top_object_label == name:
                 bar_color = (0, 255, 0)

            cv.rectangle(canvas, (x0, y0), (x0 + bar_w, baseline_y), bar_color, -1)

            text_size, _ = cv.getTextSize(name, cv.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            text_x = x0 + (bar_w - text_size[0]) // 2
            cv.putText(canvas, name, (text_x, baseline_y + 18),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

        return canvas

    # -------------------------- 2D Map Logic --------------------------

    def _project_to_map(self, x_m, y_m):
        u = int(MAP_W / 2 - (y_m - MAP_ORIGIN_Y_M) / MAP_SCALE_M_PER_PX)
        v = int(MAP_H / 2 - (x_m - MAP_ORIGIN_X_M) / MAP_SCALE_M_PER_PX)
        return u, v

    def make_map_canvas(self):
        canvas = np.zeros((MAP_H, MAP_W, 3), np.uint8)
        canvas[:] = (50, 50, 50)

        cv.line(canvas, self._project_to_map(MAP_ORIGIN_X_M, -100), self._project_to_map(MAP_ORIGIN_X_M, 100), (80, 80, 80), 1)
        cv.line(canvas, self._project_to_map(-100, MAP_ORIGIN_Y_M), self._project_to_map(100, MAP_ORIGIN_Y_M), (80, 80, 80), 1)
        cv.circle(canvas, self._project_to_map(MAP_ORIGIN_X_M, MAP_ORIGIN_Y_M), 5, (0, 0, 255), -1)

        with self.lock:
            objects_copy = self.intent_objects[:]

        top_object_label = None
        if objects_copy:
            top_object = max(objects_copy, key=lambda x: x['prob'])
            top_object_label = top_object['label']

        for obj in objects_copy:
            label, pos, prob = obj['label'], obj['pos'], obj['prob']
            u, v = self._project_to_map(pos.x, pos.y)
            intensity = int(255 * (prob * 0.8 + 0.2))
            obj_color = (0, intensity, intensity)

            if top_object_label and top_object_label == label:
                obj_color = (0, 255, 0)

            cv.circle(canvas, (u, v), 10, obj_color, -1)
            cv.putText(canvas, f"{label} ({prob:.2f})", (u + 15, v + 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, obj_color, 1)

        if len(self.tracker_path_history) > 1:
            points_px = [self._project_to_map(p[0], p[1]) for p in self.tracker_path_history]
            for i in range(1, len(points_px)):
                cv.line(canvas, points_px[i-1], points_px[i], (255, 100, 0), 2)

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