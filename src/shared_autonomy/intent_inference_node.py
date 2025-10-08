#!/usr/bin/env python3
"""Intent Inference Node

Tracks a hand (via MediaPipe) or robot end-effector, fuses with 3D object
positions, and computes a softmax distribution over goals. Publishes the full
distribution, top goal label, and pose. Optionally renders annotated frames.

When intent for a specific goal exceeds a configurable threshold, it sends a
PoseStamped goal to the robot controller.
"""
import rospy
import numpy as np
import threading
import math
import os
import time
from collections import deque
# --- Vision & Image Processing Imports ---
import cv2 as cv
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
# Conditional MediaPipe Import
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
USE_MEDIAPIPE_DEFAULT = True
# --- ROS Message Imports ---
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from intera_core_msgs.msg import EndpointState
# NEW: Import the correct goal message type for Relaxed IK
from relaxed_ik_ros1.msg import EEPoseGoals
# --- TF2 Imports ---
import tf2_ros
import tf2_geometry_msgs


class IntentInferenceNode:
    """
    Infers user intent by tracking either a robot's end-effector or a human hand.
    The source is selectable via a ROS parameter `~tracker_type`.
    """
    def __init__(self):
        """
        Initializes the node, parameters, subscribers, and publishers.
        """
        rospy.init_node("intent_inference")

        # --- Core Parameters ---
        self.tracker_type = rospy.get_param("~tracker_type", "end_effector") # "end_effector" or "hand"
        self.base_frame   = rospy.get_param("~base_frame", "world")

        # --- Inference Algorithm Configuration ---
        self.beta       = float(rospy.get_param("~beta", 25.0))
        self.window_s   = float(rospy.get_param("~window_sec", 1.2))
        self.speed_eps  = float(rospy.get_param("~stationary_speed_mps", 0.03))
        self.reset_hold = float(rospy.get_param("~reset_hold_sec", 2.0))
        self.intent_action_threshold = float(rospy.get_param("~intent_action_threshold", 0.85)) # Default to 1.0 (disabled)

        # --- State Variables ---
        self.hist = deque()      # Stores (timestamp, (x, y, z)) for path length
        self.S = None            # Start point (Point msg) of the current reach
        self.last_move_t = None  # Timestamp of last detected movement
        self.objects = []        # Stores (label:str, position_tuple)
        self.lock = threading.Lock()
        self.annotated_frame = None
        self.frame_lock = threading.Lock()
        self.commanded_goal_label = None

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Publishers ---
        self.pub_dist    = rospy.Publisher("~distribution", Float32MultiArray, queue_size=1)
        self.pub_top     = rospy.Publisher("~top_goal", String, queue_size=1)
        self.pub_toppose = rospy.Publisher("~top_pose", PoseStamped, queue_size=1)
        self.pub_current_tracker_point = rospy.Publisher("~current_tracker_point", PointStamped, queue_size=1)
        # MODIFIED: Publisher for sending goals to Relaxed IK, using the correct message type
        self.pub_ee_goal = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)


        # --- Subscriber for Object Detections (Common to both modes) ---
        self.det_topic = rospy.get_param("~detections_topic", "/yolo_3d_pose/detections")
        rospy.Subscriber(self.det_topic, Detection2DArray, self.detections_cb, queue_size=5)


        # ==================================================================
        # ======= MODE-SPECIFIC INITIALIZATION (Hand vs. End-Effector) =======
        # ==================================================================
        if self.tracker_type == "hand":
            rospy.loginfo("Tracker Type: [hand]. Initializing camera and MediaPipe.")
            self._init_hand_tracker()
        elif self.tracker_type == "end_effector":
            rospy.loginfo("Tracker Type: [end_effector]. Initializing robot state subscriber.")
            self._init_end_effector_tracker()
        else:
            rospy.logerr(f"Invalid tracker_type '{self.tracker_type}'. Must be 'hand' or 'end_effector'. Shutting down.")
            rospy.signal_shutdown("Invalid tracker_type parameter.")
            return
            
        rospy.loginfo(f"Intent action threshold set to {self.intent_action_threshold:.2%}")
        rospy.loginfo("Intent inference node is ready.")

    def _init_end_effector_tracker(self):
        """Sets up subscriber for the robot's end-effector state."""
        self.ee_topic = rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        rospy.Subscriber(self.ee_topic, EndpointState, self._end_effector_cb, queue_size=10)
        rospy.loginfo(f"Subscribing to end-effector on: {self.ee_topic}")

    def _init_hand_tracker(self):
        """Sets up subscribers and resources for hand tracking."""
        # --- Hand Tracker Parameters ---
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/camera/color/camera_info")
        self.color_optical_frame = rospy.get_param("~color_optical_frame", "camera_color_optical_frame")
        self.show_gui = rospy.get_param("~show_gui", True)
        self.use_mediapipe = rospy.get_param("~use_mediapipe", False) and USE_MEDIAPIPE_DEFAULT

        # --- Hand Tracker State ---
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None  # Camera intrinsics
        self.fps_tracker = 0.0
        self.last_t_tracker = rospy.get_time()

        # --- Hand Tracker MediaPipe Context ---
        self.mp_ctx = None
        if self.use_mediapipe:
            self.mp_ctx = mp_hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            rospy.loginfo("HandTracker: using MediaPipe hands.")
        else:
            rospy.logwarn("HandTracker: MediaPipe disabled or unavailable. Hand tracking will not function.")

        # --- Hand Tracker Subscribers ---
        self.sub_rgb = Subscriber(self.image_topic, Image)
        self.sub_depth = Subscriber(self.depth_topic, Image)
        self.sub_info = Subscriber(self.cam_info_topic, CameraInfo)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_info], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self._hand_tracker_cb)

        # --- Hand Tracker Publishers (For visualization/debugging) ---
        self.pub_annot = rospy.Publisher("~annotated_image", Image, queue_size=1)
        self.pub_hand_point_base = rospy.Publisher("~hand_in_base", PointStamped, queue_size=10)


    # -------------------------- Callbacks --------------------------

    def _end_effector_cb(self, msg: EndpointState):
        """
        Callback for the end-effector. Converts message to a standard PointStamped
        and passes it to the core processing logic.
        """
        ps = PointStamped()
        ps.header.stamp = msg.header.stamp
        ps.header.frame_id = self.base_frame # Assume EE state is in the base frame
        ps.point = msg.pose.position

        self._process_tracker_point(ps)

    def _hand_tracker_cb(self, img_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        """
        Callback for synchronized image/depth data. Finds a hand and passes its
        3D point to the core processing logic.
        """
        if not self.use_mediapipe or self.mp_ctx is None: return

        if self.fx is None: self.set_intrinsics(info_msg)

        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        Z_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth.astype(np.float32)

        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = self.mp_ctx.process(rgb)

        if res.multi_hand_landmarks:
            hand_landmarks = res.multi_hand_landmarks[0]
            if self.show_gui or self.pub_annot.get_num_connections() > 0:
                 mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            palm = hand_landmarks.landmark[0]
            u, v = int(palm.x * w), int(palm.y * h)

            if 0 <= u < w and 0 <= v < h:
                Z = float(Z_m[v, u])
                if np.isfinite(Z) and Z > 0.1:
                    X = (u - self.cx) * Z / self.fx
                    Y = (v - self.cy) * Z / self.fy

                    ps_cam = PointStamped()
                    ps_cam.header.stamp = img_msg.header.stamp
                    ps_cam.header.frame_id = self.color_optical_frame
                    ps_cam.point.x, ps_cam.point.y, ps_cam.point.z = X, Y, Z
                    
                    try:
                        ps_base = self.tf_buffer.transform(ps_cam, self.base_frame, rospy.Duration(0.1))
                        self.pub_hand_point_base.publish(ps_base)
                        self._process_tracker_point(ps_base)

                        if self.show_gui or self.pub_annot.get_num_connections() > 0:
                            txt = f"({ps_base.point.x:.2f}, {ps_base.point.y:.2f}, {ps_base.point.z:.2f})m"
                            cv.putText(annotated, txt, (u, v - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except Exception as e:
                        rospy.logwarn_throttle(2.0, f"TF transform failed: {e}")

        self._update_and_draw_fps(annotated)

        # Use a lock to prevent race conditions between this callback thread and the main thread.
        with self.frame_lock:
            self.annotated_frame = annotated.copy()

        # Publish the annotated image if there are subscribers
        if self.pub_annot.get_num_connections() > 0:
            self.pub_annot.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

    def detections_cb(self, msg: Detection2DArray):
        """Callback for object detections. Stores labels and 3D positions."""
        new_objects = []
        for det in msg.detections:
            if not det.results: continue
            hypothesis = det.results[0]
            label = str(hypothesis.id)
            pos = hypothesis.pose.pose.position
            pos_tuple = (pos.x, pos.y, pos.z)
            new_objects.append((label, pos_tuple))

        with self.lock:
            self.objects = new_objects
            
    # ------------------- Inference Logic -------------------

    def _process_tracker_point(self, msg: PointStamped):
        """
        CENTRAL LOGIC. Takes a PointStamped, tracks movement, manages history,
        and triggers intent inference updates.
        """
        # Publish the incoming point for visualization
        self.pub_current_tracker_point.publish(msg)

        t = msg.header.stamp.to_sec()
        p_tuple = (msg.point.x, msg.point.y, msg.point.z)

        self.hist.append((t, p_tuple))
        t_min = t - self.window_s
        while self.hist and self.hist[0][0] < t_min: self.hist.popleft()

        speed = 0.0
        if len(self.hist) >= 2:
            (t0, p0), (t1, p1) = self.hist[-2], self.hist[-1]
            dt = max(1e-6, t1 - t0)
            speed = np.linalg.norm(np.subtract(p1, p0)) / dt

        if speed > self.speed_eps:
            self.last_move_t = t
            if self.S is None:
                rospy.loginfo("Reach detected. Starting inference.")
                self.S = msg.point
        else:
            if self.last_move_t is not None and (t - self.last_move_t) > self.reset_hold:
                if self.S is not None:
                    rospy.loginfo("Reach ended. Resetting.")
                    self.S = None
                    self.commanded_goal_label = None

        with self.lock:
            current_objects = list(self.objects)

        if self.S is not None and current_objects:
            self.update_distribution(p_now=msg.point, S=self.S, objects=current_objects, stamp=msg.header.stamp)

    def update_distribution(self, p_now: Point, S: Point, objects: list, stamp):
        """Calculates and publishes the probability distribution over goal objects."""
        L_obs = self.path_length_observed()
        start, current = (S.x, S.y, S.z), (p_now.x, p_now.y, p_now.z)

        scores = []
        for (label, g_pos) in objects:
            d_Sg = self.vec_dist(start, g_pos)
            d_Qg = self.vec_dist(current, g_pos)
            if d_Sg < 1e-3: continue

            score = -self.beta * (L_obs + d_Qg)/d_Sg
            scores.append((label, g_pos, score))

        if not scores: return

        max_score = max(s for _, _, s in scores)
        exp_scores = [math.exp(s - max_score) for _, _, s in scores]
        Z = sum(exp_scores)
        norm_probs = [p / Z for p in exp_scores]

        # --- Publish Results ---
        dist_msg = Float32MultiArray()
        dist_msg.layout.dim.append(MultiArrayDimension(label="objects", size=len(norm_probs), stride=len(norm_probs)))
        dist_msg.data = norm_probs
        self.pub_dist.publish(dist_msg)

        top_index = int(np.argmax(norm_probs))
        top_prob = norm_probs[top_index]
        top_label, top_g_pos, _ = scores[top_index]
        self.pub_top.publish(String(data=top_label))
        
        top_pose_stamped = PoseStamped()
        top_pose_stamped.header.frame_id = self.base_frame
        top_pose_stamped.header.stamp = stamp
        top_pose_stamped.pose.position.x, top_pose_stamped.pose.position.y, top_pose_stamped.pose.position.z = top_g_pos
        top_pose_stamped.pose.orientation.w = 1.0 # Keep orientation neutral
        self.pub_toppose.publish(top_pose_stamped)

        # --- ROBOT ACTION LOGIC ---
        if top_prob >= self.intent_action_threshold:
            if top_label != self.commanded_goal_label:
                rospy.loginfo(f"Intent for '{top_label}' ({top_prob:.2%}) passed threshold. Sending goal to robot.")
                
                # Construct the EEPoseGoals message
                goal_msg = EEPoseGoals()
                goal_msg.header = top_pose_stamped.header
                goal_msg.ee_poses.append(top_pose_stamped.pose) # Append the Pose, not PoseStamped
                # Tolerances can be left empty if not needed
                goal_msg.ee_poses[0].position.z += 0.15 # hover over the intent object for grasping
                self.pub_ee_goal.publish(goal_msg)
                self.commanded_goal_label = top_label


    # -------------------------- Helper Methods --------------------------

    def set_intrinsics(self, cam_info: CameraInfo):
        self.fx, self.fy = cam_info.K[0], cam_info.K[4]
        self.cx, self.cy = cam_info.K[2], cam_info.K[5]

    def path_length_observed(self) -> float:
        if len(self.hist) < 2: return 0.0
        points = [p for (_, p) in self.hist]
        return sum(np.linalg.norm(np.subtract(points[i], points[i-1])) for i in range(1, len(points)))

    def vec_dist(self, p1, p2) -> float:
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))

    def _update_and_draw_fps(self, image):
        now = rospy.get_time()
        dt = now - self.last_t_tracker
        if dt > 0: self.fps_tracker = 0.9 * self.fps_tracker + 0.1 * (1.0 / dt)
        self.last_t_tracker = now
        cv.putText(image, f"FPS: {self.fps_tracker:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


    def run(self):
        # Use a rate object to control the loop frequency
        rate = rospy.Rate(30) # Render at 30 Hz

        while not rospy.is_shutdown():
            # The hand tracker mode is the only one with a GUI window
            if self.tracker_type == "hand" and self.show_gui:
                
                # Check if there is a new frame to display
                local_frame = None
                with self.frame_lock:
                    if self.annotated_frame is not None:
                        local_frame = self.annotated_frame.copy()
                
                if local_frame is not None:
                    cv.imshow("Hand Tracker", local_frame)
                    cv.waitKey(1)
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                rospy.logwarn("ROS Time moved backwards, continuing.")


        if self.tracker_type == "hand" and self.show_gui:
            cv.destroyAllWindows()
            

def main():
    try:
        node = IntentInferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()