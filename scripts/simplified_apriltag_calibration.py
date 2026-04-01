#!/usr/bin/env python3
"""
Simplified AprilTag Calibration for Known Tag Position

This script calibrates the hand-camera transform (T_hand_cam) assuming:
1. The AprilTag position in base frame (T_base_tag) is KNOWN
2. Camera orientation relative to end-effector is FIXED
3. Only need to sample different (x,y,z) positions

Math:
  T_base_tag = T_base_hand * T_hand_cam * T_cam_tag
  
  Rearranging:
  T_hand_cam = inv(T_base_hand) * T_base_tag * inv(T_cam_tag)
  
  Since T_tag_cam is measured (camera sees tag), we have:
  T_hand_cam = inv(T_base_hand) * T_base_tag * T_tag_cam
"""

import rospy
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import os

from apriltag_camera_calibration import AprilTagCameraCalibration
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Pose
from intera_core_msgs.msg import DigitalIOState
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
from relaxed_ik_ros1.msg import EEVelGoals
from geometry_msgs.msg import Twist


def tf_to_matrix(tf_msg):
    """Convert TransformStamped to 4x4 matrix."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def matrix_to_xyz_quat(T):
    """Convert 4x4 matrix to position and quaternion."""
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3])
    quat = rot.as_quat()  # [x, y, z, w]
    return pos, quat


class SimplifiedCalibrationNode:
    """
    Simplified calibration node for known AprilTag position.
    """
    
    def __init__(self):
        rospy.init_node('simplified_apriltag_calibration', anonymous=False)
        
        # Tag parameters
        self.tag_size = rospy.get_param('~tag_size', 0.053)
        self.tag_family = rospy.get_param('~tag_family', 'tag36h11')
        
        # Camera namespace and frames
        self.camera_ns = rospy.get_param('~camera_ns', '/camera')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_color_optical_frame')
        self.base_frame = rospy.get_param('~base_frame', 'base')
        self.ee_frame = rospy.get_param('~ee_frame', 'right_hand')
        self.tf_prefix = rospy.get_param('~tf_prefix', '')
        
        # Known AprilTag position in base frame [x, y, z]
        tag_pos_param = rospy.get_param('~apriltag_position', [0.568, 0.07, 0.02])
        if isinstance(tag_pos_param, str):
            import ast
            tag_pos_param = ast.literal_eval(tag_pos_param)
        
        # Known AprilTag orientation in base frame [qx, qy, qz, qw]
        # Default: tag lying flat on table, facing up
        tag_quat_param = rospy.get_param('~apriltag_orientation', [0, 0, 0, 1])
        if isinstance(tag_quat_param, str):
            import ast
            tag_quat_param = ast.literal_eval(tag_quat_param)
        
        # Build T_base_tag
        self.T_base_tag = np.eye(4)
        self.T_base_tag[:3, :3] = Rotation.from_quat(tag_quat_param).as_matrix()
        self.T_base_tag[:3, 3] = tag_pos_param
        
        rospy.loginfo(f"Known T_base_tag position: {tag_pos_param}")
        rospy.loginfo(f"Known T_base_tag orientation: {tag_quat_param}")
        
        # Sampling parameters
        self.num_samples = rospy.get_param('~num_samples', 20)
        self.grid_size_x = rospy.get_param('~grid_size_x', 0.2)
        self.grid_size_y = rospy.get_param('~grid_size_y', 0.2)
        self.fixed_height = rospy.get_param('~fixed_height', 0.3)  # Height above tag
        
        # Fixed camera orientation (looking down at tag)
        # [qx, qy, qz, qw] - default is camera looking straight down
        self.fixed_orientation = rospy.get_param('~fixed_orientation', [0.707, 0.707, 0.0, 0.0])
        
        # Wait for button press
        self.wait_for_button = rospy.get_param('~wait_for_button', False)
        self.button_pressed = False
        
        # Output file
        self.output_file = rospy.get_param('~output_file',
                                           os.path.expanduser('~/camera_hand_transform.npz'))
        self.samples_file = rospy.get_param('~samples_file',
                                           os.path.expanduser('~/calib_samples_simple.npz'))
        
        # Initialize components
        self.bridge = CvBridge()
        self.calib = AprilTagCameraCalibration(self.tag_size, self.tag_family)
        
        # Camera intrinsics
        self.cam_matrix = None
        self.cam_dist = None
        
        # Latest image
        self.latest_image = None
        self.latest_image_stamp = None
        
        # TF components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Button subscriber
        if self.wait_for_button:
            rospy.Subscriber('/robot/digital_io/right_button_ok/state',
                           DigitalIOState, self.button_callback, queue_size=1)
        
        # Subscribers
        rospy.Subscriber(f'{self.camera_ns}/color/camera_info', CameraInfo,
                        self.camera_info_callback)
        rospy.Subscriber(f'{self.camera_ns}/color/image_raw', Image, self.image_callback)
        
        # Publisher for visualization
        self.image_viz_pub = rospy.Publisher('~camera_visualization', Image, queue_size=1)
        self.viz_timer = rospy.Timer(rospy.Duration(0.1), self.publish_visualization)
        
        # Services
        rospy.Service('~capture_observation', Trigger, self.capture_observation_callback)
        rospy.Service('~compute_calibration', Trigger, self.compute_calibration_callback)
        rospy.Service('~start_auto_calibration', Trigger, self.start_auto_calibration_callback)
        
        # Observations storage
        self.observations = []
        
        rospy.loginfo("=== Simplified AprilTag Calibration ===")
        rospy.loginfo(f"Tag size: {self.tag_size}m")
        rospy.loginfo(f"Camera: {self.camera_ns}")
        rospy.loginfo(f"Sampling: {self.num_samples} positions in {self.grid_size_x}x{self.grid_size_y}m grid")
        rospy.loginfo(f"Fixed height: {self.fixed_height}m above tag")
        rospy.loginfo("\nServices:")
        rospy.loginfo("  - ~capture_observation: Capture current observation")
        rospy.loginfo("  - ~compute_calibration: Compute T_hand_cam from observations")
        rospy.loginfo("  - ~start_auto_calibration: Auto-sample positions and calibrate")
        if self.wait_for_button:
            rospy.loginfo("  - Press OK button to capture each observation")
    
    def button_callback(self, msg):
        """Button press callback."""
        if msg.state == 1:
            self.button_pressed = True
            rospy.loginfo("🔘 Button pressed!")
    
    def camera_info_callback(self, msg):
        """Update camera intrinsics."""
        if self.cam_matrix is None:
            self.cam_matrix = np.array(msg.K).reshape(3, 3)
            self.cam_dist = np.array(msg.D)
            rospy.loginfo("✓ Received camera intrinsics")
    
    def image_callback(self, image_msg):
        """Store latest image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.latest_image_stamp = image_msg.header.stamp
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
    
    def capture_observation_callback(self, req):
        """Manually capture an observation."""
        # Check prerequisites
        if self.latest_image is None:
            return TriggerResponse(False, "No image received yet")
        if self.cam_matrix is None:
            return TriggerResponse(False, "No camera intrinsics received yet")
        
        # Detect AprilTag
        T_tag_cam, tag_id = self.calib.detect_apriltag(
            self.latest_image, self.cam_matrix, self.cam_dist
        )
        
        if T_tag_cam is None:
            return TriggerResponse(False, "Failed to detect AprilTag")
        
        # Get current end-effector pose
        stamp = self.latest_image_stamp if self.latest_image_stamp else rospy.Time(0)
        try:
            tf_base_hand = self.tf_buffer.lookup_transform(
                self.tf_prefix + self.base_frame,
                self.tf_prefix + self.ee_frame,
                stamp, rospy.Duration(1.0)
            )
            T_base_hand = tf_to_matrix(tf_base_hand)
        except Exception as e:
            return TriggerResponse(False, f"Failed to get EE pose: {e}")
        
        # Store observation
        self.observations.append({
            'T_base_hand': T_base_hand,
            'T_tag_cam': T_tag_cam,
            'tag_id': tag_id
        })
        
        # Compute T_hand_cam for this observation
        T_hand_cam_i = np.linalg.inv(T_base_hand) @ self.T_base_tag @ T_tag_cam
        
        msg = f"✓ Observation {len(self.observations)} captured (tag ID: {tag_id})"
        rospy.loginfo(msg)
        rospy.loginfo(f"  T_hand_cam (this obs): {T_hand_cam_i[:3, 3]}")
        
        # Save samples
        self.save_samples()
        
        return TriggerResponse(True, msg)
    
    def compute_calibration_callback(self, req):
        """Compute T_hand_cam from all observations."""
        if len(self.observations) < 3:
            return TriggerResponse(False, f"Need at least 3 observations (have {len(self.observations)})")
        
        rospy.loginfo(f"\n=== Computing Calibration from {len(self.observations)} observations ===")
        
        # Compute T_hand_cam for each observation
        T_hand_cam_list = []
        for i, obs in enumerate(self.observations):
            T_base_hand = obs['T_base_hand']
            T_tag_cam = obs['T_tag_cam']
            
            # T_hand_cam = inv(T_base_hand) * T_base_tag * T_tag_cam
            T_hand_cam_i = np.linalg.inv(T_base_hand) @ self.T_base_tag @ T_tag_cam
            T_hand_cam_list.append(T_hand_cam_i)
            
            rospy.loginfo(f"Obs {i+1}: T_hand_cam translation = {T_hand_cam_i[:3, 3]}")
        
        # Average the transforms
        # For rotation: convert to quaternions, average, normalize
        positions = np.array([T[:3, 3] for T in T_hand_cam_list])
        quaternions = np.array([Rotation.from_matrix(T[:3, :3]).as_quat() for T in T_hand_cam_list])
        
        # Average position
        pos_mean = positions.mean(axis=0)
        pos_std = positions.std(axis=0)
        
        # Average quaternion (simple average, can use more sophisticated methods)
        quat_mean = quaternions.mean(axis=0)
        quat_mean = quat_mean / np.linalg.norm(quat_mean)  # Normalize
        
        # Build final T_hand_cam
        T_hand_cam = np.eye(4)
        T_hand_cam[:3, :3] = Rotation.from_quat(quat_mean).as_matrix()
        T_hand_cam[:3, 3] = pos_mean
        
        # Compute consistency metric
        rospy.loginfo(f"\n=== Calibration Results ===")
        rospy.loginfo(f"Position mean: {pos_mean}")
        rospy.loginfo(f"Position std:  {pos_std}")
        rospy.loginfo(f"Position std norm: {np.linalg.norm(pos_std):.4f}m")
        
        euler = Rotation.from_matrix(T_hand_cam[:3, :3]).as_euler('xyz', degrees=True)
        rospy.loginfo(f"Orientation (XYZ Euler deg): {euler}")
        
        # Save to file
        np.savez(self.output_file,
                 T_hand_cam=T_hand_cam,
                 position_mean=pos_mean,
                 position_std=pos_std,
                 quaternion=quat_mean)
        
        rospy.loginfo(f"✓ Saved to {self.output_file}")
        
        # Publish as static TF
        self.publish_static_transform(T_hand_cam)
        
        message = (
            f"Calibration complete!\n"
            f"Observations: {len(self.observations)}\n"
            f"Position std: {np.linalg.norm(pos_std)*1000:.2f}mm\n"
            f"Saved to: {self.output_file}"
        )
        
        return TriggerResponse(True, message)
    
    def generate_sample_positions(self):
        """Generate grid of sample positions above the tag."""
        positions = []
        
        # Grid pattern
        n_side = int(np.sqrt(self.num_samples))
        x_vals = np.linspace(-self.grid_size_x/2, self.grid_size_x/2, n_side)
        y_vals = np.linspace(-self.grid_size_y/2, self.grid_size_y/2, n_side)
        
        tag_pos = self.T_base_tag[:3, 3]
        
        for x_offset in x_vals:
            for y_offset in y_vals:
                # Position above tag
                pos = tag_pos + np.array([x_offset, y_offset, self.fixed_height])
                positions.append(pos)
        
        rospy.loginfo(f"Generated {len(positions)} sample positions in {n_side}x{n_side} grid")
        return positions[:self.num_samples]  # Limit to num_samples
    
    def start_auto_calibration_callback(self, req):
        """Start automatic calibration by sampling positions."""
        if self.cam_matrix is None:
            return TriggerResponse(False, "No camera intrinsics available")
        
        rospy.loginfo("\n=== Starting Automatic Calibration ===")
        
        # Generate sample positions
        positions = self.generate_sample_positions()
        
        rospy.loginfo(f"Will sample {len(positions)} positions")
        rospy.loginfo("MANUAL MODE: Move robot to each position and press button to capture")
        
        successful = 0
        for i, pos in enumerate(positions):
            rospy.loginfo(f"\n--- Position {i+1}/{len(positions)} ---")
            rospy.loginfo(f"Target: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            rospy.loginfo(f"Orientation: {self.fixed_orientation}")
            rospy.loginfo("⏸️  Move robot to this position and press OK button...")
            
            # Wait for button
            if self.wait_for_button:
                self.button_pressed = False
                while not self.button_pressed and not rospy.is_shutdown():
                    rospy.sleep(0.1)
                if rospy.is_shutdown():
                    break
            
            # Capture observation
            response = self.capture_observation_callback(Trigger._request_class())
            if response.success:
                successful += 1
            else:
                rospy.logwarn(f"Failed to capture: {response.message}")
        
        rospy.loginfo(f"\n=== Sampling Complete: {successful}/{len(positions)} ===")
        
        # Auto-compute calibration if we have enough samples
        if successful >= 3:
            rospy.loginfo("Computing calibration...")
            return self.compute_calibration_callback(req)
        else:
            return TriggerResponse(False, f"Not enough samples ({successful} < 3)")
    
    def save_samples(self):
        """Save observations to file."""
        if len(self.observations) == 0:
            return
        
        T_base_hand_list = [obs['T_base_hand'] for obs in self.observations]
        T_tag_cam_list = [obs['T_tag_cam'] for obs in self.observations]
        
        np.savez(self.samples_file,
                 T_base_hand=np.array(T_base_hand_list),
                 T_tag_cam=np.array(T_tag_cam_list),
                 T_base_tag=self.T_base_tag)
        
        rospy.loginfo(f"💾 Saved {len(self.observations)} samples to {self.samples_file}")
    
    def publish_visualization(self, event):
        """Publish visualization image."""
        if self.latest_image is not None and self.cam_matrix is not None:
            vis_image = self.latest_image.copy()
            
            # Detect and visualize tag
            try:
                T_tag_cam, tag_id = self.calib.detect_apriltag(
                    self.latest_image, self.cam_matrix, self.cam_dist
                )
                
                if T_tag_cam is not None:
                    # Draw detection status
                    cv2.putText(vis_image, f"TAG DETECTED (ID: {tag_id})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Draw observation count
                    cv2.putText(vis_image, f"Observations: {len(self.observations)}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(vis_image, "NO TAG DETECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Visualization error: {e}")
            
            # Publish
            try:
                img_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
                self.image_viz_pub.publish(img_msg)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Error publishing viz: {e}")
    
    def publish_static_transform(self, T_hand_cam):
        """Publish T_hand_cam as static TF."""
        static_tf = TransformStamped()
        static_tf.header.stamp = rospy.Time.now()
        static_tf.header.frame_id = self.ee_frame
        static_tf.child_frame_id = self.camera_frame
        
        pos, quat = matrix_to_xyz_quat(T_hand_cam)
        static_tf.transform.translation.x = pos[0]
        static_tf.transform.translation.y = pos[1]
        static_tf.transform.translation.z = pos[2]
        static_tf.transform.rotation.x = quat[0]
        static_tf.transform.rotation.y = quat[1]
        static_tf.transform.rotation.z = quat[2]
        static_tf.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(static_tf)
        rospy.loginfo("✓ Published static TF: right_hand -> camera_color_optical_frame")


if __name__ == "__main__":
    try:
        node = SimplifiedCalibrationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
