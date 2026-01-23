#!/usr/bin/env python3
"""Intent Inference Node

Publishes a filtered list of objects with their inferred intent probabilities.
"""
from warnings import filters
import rospy
import numpy as np
import threading
import math
from collections import deque
# --- ROS Message Imports ---
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point, PointStamped, Pose, Twist, Quaternion
from sensor_msgs.msg import Image, CameraInfo, Joy
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from intera_core_msgs.msg import EndpointState
# Import the correct goal message type for Relaxed IK
from chomp import generate_smooth_path
from relaxed_ik_ros1.msg import EEPoseGoals
# --- TF2 Imports ---
import tf2_ros

class IntentInferenceNode:
    def __init__(self):
        rospy.init_node("intent_inference")

        # --- Core Parameters ---
        self.tracker_type = rospy.get_param("~tracker_type", "end_effector")
        self.base_frame   = rospy.get_param("~base_frame", "world")
        self.beta       = float(rospy.get_param("~beta", 25.0))
        self.window_s   = float(rospy.get_param("~window_sec", 1.2))
        self.speed_eps  = float(rospy.get_param("~stationary_speed_mps", 0.03))
        self.reset_hold = float(rospy.get_param("~reset_hold_sec", 2.0))
        self.intent_action_threshold = float(rospy.get_param("~intent_action_threshold", 0.85))

        # --- State Variables ---
        self.hist = deque()
        self.S = None
        self.last_move_t = None
        self.objects = []
        self.lock = threading.Lock()
        self.current_ee_pos = None

        # --- Robot Action State Machine ---
        self.robot_action_state = "IDLE"
        self.held_object_label = None
        self.filtered_objects = []
        self.pickup_target_pose = None
        self.current_top_goal_pose = None
        self.last_joy_y_button = 0
        self.joy_place_button_idx = rospy.get_param("~joy_place_button", 3)

        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Publishers ---
        # NEW: This publisher now sends a complete picture of the intent distribution
        self.pub_dist    = rospy.Publisher("~distribution", Detection2DArray, queue_size=1)
        self.pub_top     = rospy.Publisher("~top_goal", String, queue_size=1)
        self.pub_toppose = rospy.Publisher("~top_pose", PoseStamped, queue_size=1)
        self.pub_current_tracker_point = rospy.Publisher("~current_tracker_point", PointStamped, queue_size=1)
        self.pub_ee_goal = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        
        # --- Subscribers ---
        rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, self._end_effector_state_cb, queue_size=10)
        rospy.Subscriber("/yolo_3d_pose/detections", Detection2DArray, self.detections_cb, queue_size=5)
        rospy.Subscriber("/joy", Joy, self._joy_cb, queue_size=1)
        
        # ... (rest of __init__ is the same, no need for hand tracker part for this example)
        rospy.loginfo("Intent inference node is ready. Robot state: IDLE")

    # ... (callbacks and other methods are the same up to update_distribution)
    def _joy_cb(self, msg: Joy):
        """Callback for joystick messages, used to trigger the placement action."""
        y_button_pressed = msg.buttons[self.joy_place_button_idx] == 1
        if y_button_pressed and not self.last_joy_y_button:
            if self.robot_action_state == "AWAITING_PLACEMENT_COMMAND":
                rospy.loginfo("Y button pressed. Initiating placement trajectory.")
                if self.current_top_goal_pose is None:
                    rospy.logwarn("Placement triggered, but no current goal is available!")
                    return

                self.robot_action_state = "MOVING_TO_PLACE"
                thread = threading.Thread(target=self._execute_placement_trajectory)
                thread.daemon = True
                thread.start()
            else:
                rospy.logwarn_throttle(5, f"Y button pressed, but robot is not in AWAITING_PLACEMENT_COMMAND state (current: {self.robot_action_state}). Ignoring.")
        
        self.last_joy_y_button = msg.buttons[self.joy_place_button_idx]

    def _end_effector_state_cb(self, msg: EndpointState):
        self.current_ee_pos = msg.pose.position
        if self.tracker_type == 'end_effector':
            ps = PointStamped(header=msg.header, point=msg.pose.position)
            ps.header.frame_id = self.base_frame
            self._process_tracker_point(ps)

    def detections_cb(self, msg: Detection2DArray):
        new_objects = []
        for det in msg.detections:
            if not det.results: continue
            hypothesis = det.results[0]
            numeric_id = hypothesis.id
            label = self._numeric_id_to_label(numeric_id)
            pos = hypothesis.pose.pose.position
            pos_tuple = (pos.x, pos.y, pos.z)
            new_objects.append((label, pos_tuple, numeric_id)) # Store numeric_id as well
        with self.lock:
            self.objects = new_objects

    def _process_tracker_point(self, msg: PointStamped):
        if self.robot_action_state in ["MOVING_TO_HOVER", "MOVING_TO_PLACE"]:
            return

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
        with self.lock:
            current_objects = list(self.objects)
        if self.S is not None and current_objects:
            self.update_distribution(p_now=msg.point, S=self.S, objects=current_objects, stamp=msg.header.stamp)

    def update_distribution(self, p_now: Point, S: Point, objects: list, stamp):
        L_obs = self.path_length_observed()
        start, current = (S.x, S.y, S.z), (p_now.x, p_now.y, p_now.z)
        
        # First, filter objects and calculate raw scores
        scored_objects = []
        for (label, g_pos, numeric_id) in objects:
            is_goal_object = label.startswith('G')
            if label in self.filtered_objects: continue

            if self.robot_action_state == "IDLE" and is_goal_object: continue
            if self.robot_action_state == "AWAITING_PLACEMENT_COMMAND" and not is_goal_object: continue
            if label == self.held_object_label: continue

            d_Sg = self.vec_dist(start, g_pos)
            if d_Sg < 1e-3: continue
            d_Qg = self.vec_dist(current, g_pos)
            
            raw_score = -self.beta * (L_obs + d_Qg)/d_Sg
            scored_objects.append({'label': label, 'pos': g_pos, 'numeric_id': numeric_id, 'raw_score': raw_score})
        
        if not scored_objects:
            self.pub_dist.publish(Detection2DArray(header=rospy.Header(stamp=stamp, frame_id=self.base_frame)))
            self.current_top_goal_pose = None
            return

        # Normalize scores (softmax)
        max_score = max(o['raw_score'] for o in scored_objects)
        exp_scores = [math.exp(o['raw_score'] - max_score) for o in scored_objects]
        Z = sum(exp_scores)
        norm_probs = [p / Z for p in exp_scores]

        # --- PUBLISH  ---
        dist_msg = Detection2DArray(header=rospy.Header(stamp=stamp, frame_id=self.base_frame))
        for i, obj in enumerate(scored_objects):
            det = Detection2D()
            hyp = ObjectHypothesisWithPose()
            hyp.id = obj['numeric_id']
            hyp.score = norm_probs[i] # <-- Probability is now the score
            hyp.pose.pose.position = Point(*obj['pos'])
            det.results.append(hyp)
            dist_msg.detections.append(det)
        
        self.pub_dist.publish(dist_msg)
        
        # Publish top goal info (as before)
        top_index = int(np.argmax(norm_probs))
        top_prob = norm_probs[top_index]
        top_object = scored_objects[top_index]
        top_label, top_g_pos = top_object['label'], top_object['pos']
        
        self.pub_top.publish(String(data=top_label))
        top_pose_stamped = PoseStamped(header=rospy.Header(frame_id=self.base_frame, stamp=stamp))
        top_pose_stamped.pose.position = Point(*top_g_pos)
        top_pose_stamped.pose.orientation.w = 1.0
        self.pub_toppose.publish(top_pose_stamped)
        
        self.current_top_goal_pose = top_g_pos

        # Trigger robot action (as before)
        if top_prob >= self.intent_action_threshold and self.robot_action_state == "IDLE":
            self.held_object_label = top_label
            self.filtered_objects.append(top_label)
            self.robot_action_state = "MOVING_TO_HOVER"
            self.pickup_target_pose = top_g_pos
            rospy.loginfo(f"Intent for '{top_label}' ({top_prob:.2%}) passed threshold. Executing hover.")
            thread = threading.Thread(target=self._execute_hover_trajectory)
            thread.daemon = True
            thread.start()

    def _execute_hover_trajectory(self):
        rospy.loginfo("Starting hover trajectory...")
        start_pos = [self.current_ee_pos.x, self.current_ee_pos.y, self.current_ee_pos.z]
        goal_pos = [self.pickup_target_pose[0], self.pickup_target_pose[1], self.pickup_target_pose[2] + 0.25]
        cartesian_path = generate_smooth_path(x_start=start_pos, x_goal=goal_pos, T=40, n_iter=150, weight_smooth=15.0)
        rate = rospy.Rate(20)
        for point in cartesian_path:
            if rospy.is_shutdown(): break
            goal_msg = EEPoseGoals(header=rospy.Header(frame_id=self.base_frame, stamp=rospy.Time.now()))
            pose = Pose(position=Point(*point), orientation=Quaternion(0, 1, 0, 0))
            goal_msg.ee_poses.append(pose)
            goal_msg.tolerances.append(Twist())
            self.pub_ee_goal.publish(goal_msg)
            rate.sleep()
        rospy.loginfo("Hover position reached. Ready for manual pickup.")
        rospy.loginfo("--> Press 'Y' button on joystick to move to placement goal. <--" )
        self.robot_action_state = "AWAITING_PLACEMENT_COMMAND"

    def _execute_placement_trajectory(self):
        rospy.loginfo("Starting placement trajectory...")
        start_pos = [self.current_ee_pos.x, self.current_ee_pos.y, self.current_ee_pos.z]
        goal_pos = [self.current_top_goal_pose[0], self.current_top_goal_pose[1], self.current_top_goal_pose[2] + 0.25]
        cartesian_path = generate_smooth_path(x_start=start_pos, x_goal=goal_pos, T=50, n_iter=150, weight_smooth=15.0)
        rate = rospy.Rate(20)
        for point in cartesian_path:
            if rospy.is_shutdown(): break
            goal_msg = EEPoseGoals(header=rospy.Header(frame_id=self.base_frame, stamp=rospy.Time.now()))
            pose = Pose(position=Point(*point), orientation=Quaternion(0, 1, 0, 0))
            goal_msg.ee_poses.append(pose)
            goal_msg.tolerances.append(Twist())
            self.pub_ee_goal.publish(goal_msg)
            rate.sleep()
        rospy.sleep(2.0)
        rospy.loginfo("Placement complete. Returning to IDLE state.")
        self.robot_action_state = "IDLE"
        self.held_object_label = None
        self.S = None

    def _numeric_id_to_label(self, num_id):
        if 1 <= num_id <= 100: return str(num_id)
        elif num_id == 9001: return "G1"
        elif num_id == 9002: return "G2"
        return f"unknown_{num_id}"
        
    def vec_dist(self, p1, p2) -> float:
        return math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
        
    def path_length_observed(self) -> float:
        if len(self.hist) < 2: return 0.0
        points = [p for (_, p) in self.hist]
        return sum(np.linalg.norm(np.subtract(points[i], points[i-1])) for i in range(1, len(points)))

    def run(self):
        rospy.spin()

def main():
    try:
        IntentInferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()