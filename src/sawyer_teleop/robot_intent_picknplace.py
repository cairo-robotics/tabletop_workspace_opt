#!/usr/bin/env python3
import rospy
import numpy as np
import intera_interface
from intera_interface import CHECK_VERSION
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
import threading
import json
from collections import OrderedDict


# <<< NEW IMPORT for standard ROS transformations
import tf.transformations as tr

from relaxed_ik_ros1.msg import EEPoseGoals

class PickAndPlaceController:
    def __init__(self):
        """
        Initializes the node, robot interface, subscribers, and new publisher.
        """
        rospy.init_node("pick_and_place_controller")

        # --- Robot Initialization ---
        self.limb = intera_interface.Limb('right') # For getting current pose
        self.gripper = intera_interface.Gripper('right_gripper')
        rs = intera_interface.RobotEnable(CHECK_VERSION)
        rs.enable()

        if not self.gripper.is_calibrated():
            rospy.loginfo("Calibrating gripper...")
            self.gripper.calibrate()
        self.gripper.open()
        
        # --- Configuration ---
        self.action_threshold = 0.1
        self.object_timeout_sec = 2.0
        self.is_busy = False
        self.lock = threading.Lock()
        self.base_frame = 'base'

        # --- Motion Control Parameters ---
        self.hover_height = 0.11
        self.grasp_offset_z = -0.03
        self.fast_duration = 2.5
        self.slow_duration = 4.0
        self.trajectory_steps = 75

        # --- State Variables ---
        self.scene_objects = OrderedDict()
        self.task_object_labels = []
        self.grasp_orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        self.neutral_pose = Pose(
            position=Point(0.6, 0.0, 0.3),
            orientation=self.grasp_orientation
        )

        # --- Publisher for RelaxedIK ---
        self.ee_pose_goals_pub = rospy.Publisher('/relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)

        # --- Subscribers ---
        rospy.Subscriber("/intent_inference/distribution", Float32MultiArray, self.intent_cb, queue_size=1)
        rospy.Subscriber("/stitch_object_detection/object_info", String, self.object_info_cb, queue_size=10)
        rospy.Timer(rospy.Duration(1.0), self.cleanup_objects_cb)

        rospy.loginfo("Pick and Place Controller is ready.")
        rospy.on_shutdown(self.clean_shutdown)

    def _get_current_pose(self) -> Pose:
        """Helper function to get the current end-effector pose."""
        endpoint_dict = self.limb.endpoint_pose()
        position = endpoint_dict['position']
        orientation = endpoint_dict['orientation']
        return Pose(position=position, orientation=orientation)

    def clean_shutdown(self):
        rospy.loginfo("Shutting down. Moving arm to neutral pose.")
        self.move_along_trajectory(self.neutral_pose, duration=3.0, num_steps=100)

    # --- Callbacks (object_info_cb, cleanup_objects_cb, intent_cb) remain the same ---
    def object_info_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            label = data["name"]
            pos_dict = data["pose"]
            pos = Point(x=pos_dict['x'], y=pos_dict['y'], z=pos_dict['z'])
            pose = Pose(position=pos, orientation=self.grasp_orientation)
            with self.lock:
                self.scene_objects[label] = {'pose': pose, 'last_seen': rospy.get_time()}
                self.task_object_labels = sorted([
                    k for k in self.scene_objects.keys() if "_A" in k or "_B" in k
                ])
        except (json.JSONDecodeError, KeyError) as e:
            rospy.logwarn_throttle(5.0, f"Could not parse object info JSON: {e}")
            
    def cleanup_objects_cb(self, event):
        with self.lock:
            now = rospy.get_time()
            stale_keys = [label for label, data in self.scene_objects.items()
                          if (now - data['last_seen']) > self.object_timeout_sec]
            if stale_keys:
                for key in stale_keys:
                    del self.scene_objects[key]
                self.task_object_labels = sorted([
                    k for k in self.scene_objects.keys() if "_A" in k or "_B" in k
                ])

    def intent_cb(self, msg: Float32MultiArray):
        with self.lock:
            if self.is_busy or not self.task_object_labels: return
            probabilities = msg.data
            if len(probabilities) != len(self.task_object_labels): return
            min_prob_idx = np.argmin(probabilities)
            if probabilities[min_prob_idx] > self.action_threshold: return

            self.is_busy = True
            target_label = self.task_object_labels[min_prob_idx]
            pick_pose = self.scene_objects[target_label]['pose']
            place_goal_label = "G1" if "_A" in target_label else "G2"
            
            if place_goal_label not in self.scene_objects:
                rospy.logwarn(f"Cannot perform action. Required goal '{place_goal_label}' is not tracked.")
                self.is_busy = False
                return

            place_pose = self.scene_objects[place_goal_label]['pose']
            rospy.loginfo(f"ACTION TRIGGERED: Picking up '{target_label}' to place at goal '{place_goal_label}'")
            threading.Thread(target=self.execute_pick_and_place, args=(pick_pose, place_pose)).start()

    # <<< MODIFIED: This function now uses tf.transformations for Slerp
    def move_along_trajectory(self, target_pose: Pose, duration: float, num_steps: int):
        """
        Generates and follows a linear trajectory to the target pose.
        """
        start_pose = self._get_current_pose()
        
        # Guard against zero duration
        if duration <= 0:
            duration = 0.1
        rate = rospy.Rate(num_steps / duration)

        # Position interpolation
        x_points = np.linspace(start_pose.position.x, target_pose.position.x, num_steps)
        y_points = np.linspace(start_pose.position.y, target_pose.position.y, num_steps)
        z_points = np.linspace(start_pose.position.z, target_pose.position.z, num_steps)

        # Orientation arrays for tf.transformations
        q_start = [start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w]
        q_target = [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]

        # Publishing loop
        for i in range(num_steps):
            if rospy.is_shutdown(): return

            pose_goal = Pose()
            pose_goal.position.x = x_points[i]
            pose_goal.position.y = y_points[i]
            pose_goal.position.z = z_points[i]
            
            # Calculate interpolated quaternion for the current step
            fraction = float(i) / (num_steps - 1) if num_steps > 1 else 1.0
            interp_quat = tr.quaternion_slerp(q_start, q_target, fraction)
            
            pose_goal.orientation.x = interp_quat[0]
            pose_goal.orientation.y = interp_quat[1]
            pose_goal.orientation.z = interp_quat[2]
            pose_goal.orientation.w = interp_quat[3]

            msg = EEPoseGoals()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.base_frame
            msg.ee_poses.append(pose_goal)
            msg.tolerances.append(Twist()) # Default tolerance
            self.ee_pose_goals_pub.publish(msg)
            rate.sleep()

    def execute_pick_and_place(self, pick_pose: Pose, place_pose: Pose):
        """Executes the full sequence using trajectories and variable speeds."""
        input("--> Press Enter to confirm the pick and place action, or Ctrl+C in terminal to cancel.")
        rospy.loginfo("--- User confirmed. Starting Pick and Place Sequence ---")
        
        # 1. Open gripper and move to a hover position above the pick location (fast)
        self.gripper.open()
        hover_pick_pose = Pose(
            position=Point(pick_pose.position.x, pick_pose.position.y, pick_pose.position.z + self.hover_height),
            orientation=self.grasp_orientation)
        rospy.loginfo(f"Moving to hover-pick pose (fast)...")
        self.move_along_trajectory(hover_pick_pose, self.fast_duration, self.trajectory_steps)

        # 2. Descend to the pick position (slow)
        grasp_pose = Pose(
            position=Point(pick_pose.position.x, pick_pose.position.y, pick_pose.position.z + self.grasp_offset_z),
            orientation=self.grasp_orientation)
        rospy.loginfo(f"Descending to grasp pose (slow)...")
        self.move_along_trajectory(grasp_pose, self.slow_duration, self.trajectory_steps)
        
        # 3. Grasp the object
        rospy.loginfo("Grasping object...")
        self.gripper.close()
        rospy.sleep(1.0)

        # 4. Ascend back to the hover position (slow)
        rospy.loginfo("Ascending with object (slow)...")
        self.move_along_trajectory(hover_pick_pose, self.slow_duration, self.trajectory_steps)
        
        # 5. Descend to the hover position(slow)
         
        # 5. Move to a hover position above the place location (fast)
        hover_place_pose = Pose(
            position=Point(place_pose.position.x, place_pose.position.y, place_pose.position.z + self.hover_height),
            orientation=self.grasp_orientation)
        rospy.loginfo("Moving to hover-place pose (fast)...")
        self.move_along_trajectory(hover_place_pose, self.fast_duration, self.trajectory_steps)

        # 6. Descend to the place position (slow)
        rospy.loginfo("Descending to place object (slow)...")
        self.move_along_trajectory(place_pose, self.slow_duration, self.trajectory_steps)
            
        # 7. Release the object
        rospy.loginfo("Releasing object...")
        self.gripper.open()
        rospy.sleep(1.0)
        
        # 8. Ascend back to the hover position (slow)
        rospy.loginfo("Ascending after place (slow)...")
        self.move_along_trajectory(hover_place_pose, self.slow_duration, self.trajectory_steps)
        
        # 9. Return to neutral pose (fast)
        rospy.loginfo("Returning to neutral pose (fast).")
        self.move_along_trajectory(self.neutral_pose, self.fast_duration, self.trajectory_steps)
        
        rospy.loginfo("--- Pick and Place Sequence Complete ---")
        self.reset_state()

    def reset_state(self):
        rospy.sleep(1.0)
        self.is_busy = False
        rospy.loginfo("Robot is ready for a new task.")

def main():
    try:
        controller = PickAndPlaceController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()