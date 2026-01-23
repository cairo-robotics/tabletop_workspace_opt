#!/usr/bin/env python3
"""
Hand-Eye Calibration using OpenCV's calibrateHandEye() function.
Configured for:
- RealSense D435 camera mounted on Sawyer robot wrist (eye-in-hand)
- AprilTag detection for calibration target
- ROS integration for robot pose tracking

Usage:
1. Place an AprilTag in the workspace (fixed position)
2. Run this script
3. Move the robot to different poses so camera sees the tag from different angles
4. Press 'a' to add a sample when tag is detected
5. After collecting 8+ samples, press 's' to solve
6. Result: transformation from end-effector to camera (gripper -> camera_link)
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros
import tf.transformations as tft
from pupil_apriltags import Detector
from threading import Lock
import yaml

# ========== Configuration Parameters ==========
# AprilTag settings
APRILTAG_FAMILY = 'tag36h11'  # Options: tag36h11, tag25h9, tag16h5, tagCircle21h7, tagStandard41h12
APRILTAG_ID = 0               # ID of the tag to use for calibration
TAG_SIZE_M = 0.055            # Physical size of the tag in meters (measure outer black square)

# Frame names
BASE_FRAME = "base"           # Robot base frame
EEF_FRAME = "right_hand"      # End-effector frame (gripper)
CAMERA_LINK = "camera_link"   # Camera frame (parent of optical frame)

# ROS topics for RealSense D435
IMAGE_TOPIC = "/camera/color/image_raw"
INFO_TOPIC = "/camera/color/camera_info"

# Calibration parameters
MIN_SAMPLES = 8               # Minimum number of samples needed for calibration
CALIB_METHOD = cv2.CALIB_HAND_EYE_TSAI  # Options: TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS

# Output file for saving calibration result
OUTPUT_FILE = "handeye_calibration_result.yaml"
# ==============================================


def tf_to_matrix(tf_msg):
    """Convert TF TransformStamped message to 4x4 homogeneous transformation matrix."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def matrix_to_R_t(T):
    """Extract rotation matrix and translation vector from 4x4 matrix."""
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].reshape(3, 1).astype(np.float64)
    return R, t


def R_t_to_matrix(R, t):
    """Combine rotation matrix and translation vector into 4x4 matrix."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


class HandEyeCalibrator:
    """Hand-eye calibration using AprilTags and OpenCV."""
    
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = Lock()
        
        # Camera intrinsics (will be set from CameraInfo)
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # AprilTag detector
        self.detector = Detector(
            families=APRILTAG_FAMILY,
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Storage for calibration data
        # For eye-in-hand: we need gripper->base and target->camera
        self.R_gripper2base = []  # Rotation matrices (3x3)
        self.t_gripper2base = []  # Translation vectors (3x1)
        self.R_target2cam = []    # Rotation matrices (3x3)
        self.t_target2cam = []    # Translation vectors (3x1)
        self.n_samples = 0
        
        # ROS subscribers
        self.img_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback, queue_size=1)
        self.info_sub = rospy.Subscriber(INFO_TOPIC, CameraInfo, self.camera_info_callback, queue_size=1)
        
        # TF listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Current image for display
        self.current_image = None
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("Hand-Eye Calibration - AprilTag Method")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Configuration:")
        rospy.loginfo(f"  - AprilTag Family: {APRILTAG_FAMILY}")
        rospy.loginfo(f"  - Target Tag ID: {APRILTAG_ID}")
        rospy.loginfo(f"  - Tag Size: {TAG_SIZE_M} m")
        rospy.loginfo(f"  - Calibration Method: {self._get_method_name(CALIB_METHOD)}")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Instructions:")
        rospy.loginfo("  1. Place AprilTag ID {} in a FIXED position".format(APRILTAG_ID))
        rospy.loginfo("  2. Move robot to different poses (camera sees tag from different angles)")
        rospy.loginfo("  3. Press 'a' to add sample when tag is visible")
        rospy.loginfo("  4. Collect at least {} samples".format(MIN_SAMPLES))
        rospy.loginfo("  5. Press 's' to solve calibration")
        rospy.loginfo("  6. Press 'q' to quit")
        rospy.loginfo("=" * 60)
    
    def _get_method_name(self, method):
        """Get human-readable name for calibration method."""
        methods = {
            cv2.CALIB_HAND_EYE_TSAI: "Tsai",
            cv2.CALIB_HAND_EYE_PARK: "Park",
            cv2.CALIB_HAND_EYE_HORAUD: "Horaud",
            cv2.CALIB_HAND_EYE_ANDREFF: "Andreff",
            cv2.CALIB_HAND_EYE_DANIILIDIS: "Daniilidis"
        }
        return methods.get(method, "Unknown")
    
    def camera_info_callback(self, msg):
        """Extract camera intrinsics from CameraInfo message."""
        if self.fx is None:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Assume rectified image (dist_coeffs = 0)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
            
            rospy.loginfo("Camera intrinsics received:")
            rospy.loginfo(f"  fx={self.fx:.2f}, fy={self.fy:.2f}")
            rospy.loginfo(f"  cx={self.cx:.2f}, cy={self.cy:.2f}")
    
    def get_gripper_pose(self, timestamp):
        """Get gripper pose in base frame at given timestamp."""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                BASE_FRAME, EEF_FRAME, timestamp, rospy.Duration(1.0)
            )
            return tf_to_matrix(tf_msg)
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"Failed to get gripper pose: {e}")
            return None
    
    def detect_apriltag(self, image):
        """
        Detect AprilTag and estimate its pose relative to camera.
        Returns: (R, t, corners) or (None, None, None)
        """
        if self.camera_matrix is None:
            return None, None, None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.fx, self.fy, self.cx, self.cy),
            tag_size=TAG_SIZE_M
        )
        
        # Find the target tag
        for tag in tags:
            if tag.tag_id == APRILTAG_ID:
                # Get pose (rotation matrix and translation vector)
                R = tag.pose_R.astype(np.float64)  # 3x3 rotation matrix
                t = tag.pose_t.astype(np.float64).reshape(3, 1)  # 3x1 translation vector
                corners = tag.corners
                return R, t, corners
        
        return None, None, None
    
    def draw_detection(self, image, R, t, corners):
        """Draw detected tag on image with pose visualization."""
        if R is None or corners is None:
            return image
        
        display = image.copy()
        
        # Draw tag corners
        corners_int = corners.astype(int)
        for i in range(4):
            cv2.line(display, tuple(corners_int[i]), 
                    tuple(corners_int[(i+1)%4]), (0, 255, 0), 2)
        
        # Draw center
        center = corners.mean(axis=0).astype(int)
        cv2.circle(display, tuple(center), 5, (0, 0, 255), -1)
        
        # Draw coordinate frame axes
        rvec, _ = cv2.Rodrigues(R)
        cv2.drawFrameAxes(display, self.camera_matrix, self.dist_coeffs, 
                         rvec, t, TAG_SIZE_M * 0.5)
        
        # Display tag info
        cv2.putText(display, f"Tag ID: {APRILTAG_ID}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Distance: {np.linalg.norm(t):.3f}m", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return display
    
    def image_callback(self, msg):
        """Process incoming images: detect AprilTag and handle user input."""
        if self.camera_matrix is None:
            return
        
        # Convert ROS image to OpenCV
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return
        
        self.current_image = image.copy()
        
        # Detect AprilTag
        R_target2cam, t_target2cam, corners = self.detect_apriltag(image)
        
        # Draw detection
        display = self.draw_detection(image, R_target2cam, t_target2cam, corners)
        
        # Add sample count to display
        cv2.putText(display, f"Samples: {self.n_samples}/{MIN_SAMPLES}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Handle keyboard input
        cv2.imshow("Hand-Eye Calibration", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('a'):  # Add sample
            if R_target2cam is not None:
                self.add_sample(msg.header.stamp, R_target2cam, t_target2cam)
            else:
                rospy.logwarn("Cannot add sample: AprilTag not detected!")
        
        elif key == ord('s'):  # Solve calibration
            self.solve_calibration()
        
        elif key == ord('c'):  # Clear all samples
            with self.lock:
                self.R_gripper2base = []
                self.t_gripper2base = []
                self.R_target2cam = []
                self.t_target2cam = []
                self.n_samples = 0
            rospy.loginfo("All samples cleared!")
        
        elif key == ord('q'):  # Quit
            rospy.signal_shutdown("User requested shutdown")
    
    def add_sample(self, timestamp, R_target2cam, t_target2cam):
        """Add a calibration sample (robot pose + tag detection)."""
        # Get gripper pose at image timestamp
        T_base2gripper = self.get_gripper_pose(timestamp)
        
        if T_base2gripper is None:
            rospy.logwarn("Failed to get gripper pose, sample not added")
            return
        
        # For eye-in-hand calibration, OpenCV expects:
        # - gripper2base: transformation from gripper to base (inverse of base2gripper)
        # - target2cam: transformation from target to camera
        
        T_gripper2base = np.linalg.inv(T_base2gripper)
        R_gripper2base, t_gripper2base = matrix_to_R_t(T_gripper2base)
        
        with self.lock:
            self.R_gripper2base.append(R_gripper2base.copy())
            self.t_gripper2base.append(t_gripper2base.copy())
            self.R_target2cam.append(R_target2cam.copy())
            self.t_target2cam.append(t_target2cam.copy())
            self.n_samples += 1
        
        rospy.loginfo(f"Sample {self.n_samples} added successfully!")
        
        # Provide feedback about diversity
        if self.n_samples >= MIN_SAMPLES:
            rospy.loginfo(f"Ready to calibrate! Press 's' to solve.")
    
    def solve_calibration(self):
        """Solve hand-eye calibration using collected samples."""
        with self.lock:
            n = len(self.R_gripper2base)
            
            if n < MIN_SAMPLES:
                rospy.logwarn(f"Need at least {MIN_SAMPLES} samples, you have {n}")
                return
            
            rospy.loginfo("=" * 60)
            rospy.loginfo(f"Solving hand-eye calibration with {n} samples...")
            rospy.loginfo(f"Method: {self._get_method_name(CALIB_METHOD)}")
            
            try:
                # Call OpenCV's calibrateHandEye
                # For eye-in-hand: returns camera pose in gripper frame (gripper -> camera)
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    self.R_gripper2base,
                    self.t_gripper2base,
                    self.R_target2cam,
                    self.t_target2cam,
                    method=CALIB_METHOD
                )
                
                # Convert to 4x4 matrix
                T_gripper2cam = R_t_to_matrix(R_cam2gripper, t_cam2gripper)
                
                # Also compute inverse for convenience
                T_cam2gripper = np.linalg.inv(T_gripper2cam)
                
                rospy.loginfo("Calibration successful!")
                rospy.loginfo("=" * 60)
                
                # Print results
                self.print_results(T_gripper2cam, T_cam2gripper)
                
                # Save results
                self.save_results(T_gripper2cam, T_cam2gripper)
                
            except Exception as e:
                rospy.logerr(f"Calibration failed: {e}")
                rospy.logerr("Try collecting more diverse samples")
    
    def print_results(self, T_gripper2cam, T_cam2gripper):
        """Print calibration results in a readable format."""
        print("\n" + "=" * 60)
        print("CALIBRATION RESULTS")
        print("=" * 60)
        
        # Gripper to Camera (what we typically want)
        print("\n1. Gripper -> Camera ({} -> {})".format(EEF_FRAME, CAMERA_LINK))
        print("-" * 60)
        t = T_gripper2cam[:3, 3]
        quat = tft.quaternion_from_matrix(T_gripper2cam)
        rpy = tft.euler_from_matrix(T_gripper2cam[:3, :3], 'sxyz')
        
        print(f"Translation (m):  x={t[0]:7.4f}  y={t[1]:7.4f}  z={t[2]:7.4f}")
        print(f"Quaternion (xyzw): [{quat[0]:8.5f}, {quat[1]:8.5f}, {quat[2]:8.5f}, {quat[3]:8.5f}]")
        print(f"RPY (rad):         [{rpy[0]:7.4f}, {rpy[1]:7.4f}, {rpy[2]:7.4f}]")
        print(f"RPY (deg):         [{np.rad2deg(rpy[0]):7.2f}, {np.rad2deg(rpy[1]):7.2f}, {np.rad2deg(rpy[2]):7.2f}]")
        
        # Static transform publisher command
        print("\nStatic TF Publisher Command:")
        print("-" * 60)
        print(f"rosrun tf2_ros static_transform_publisher \\")
        print(f"  {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} \\")
        print(f"  {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} \\")
        print(f"  {EEF_FRAME} {CAMERA_LINK}")
        
        # Camera to Gripper (inverse, for reference)
        print("\n2. Camera -> Gripper (inverse)")
        print("-" * 60)
        t_inv = T_cam2gripper[:3, 3]
        quat_inv = tft.quaternion_from_matrix(T_cam2gripper)
        print(f"Translation (m):  x={t_inv[0]:7.4f}  y={t_inv[1]:7.4f}  z={t_inv[2]:7.4f}")
        print(f"Quaternion (xyzw): [{quat_inv[0]:8.5f}, {quat_inv[1]:8.5f}, {quat_inv[2]:8.5f}, {quat_inv[3]:8.5f}]")
        
        print("\n" + "=" * 60 + "\n")
    
    def save_results(self, T_gripper2cam, T_cam2gripper):
        """Save calibration results to YAML file."""
        t = T_gripper2cam[:3, 3]
        quat = tft.quaternion_from_matrix(T_gripper2cam)
        rpy = tft.euler_from_matrix(T_gripper2cam[:3, :3], 'sxyz')
        
        t_inv = T_cam2gripper[:3, 3]
        quat_inv = tft.quaternion_from_matrix(T_cam2gripper)
        rpy_inv = tft.euler_from_matrix(T_cam2gripper[:3, :3], 'sxyz')
        
        data = {
            'calibration_date': rospy.get_time(),
            'num_samples': self.n_samples,
            'method': self._get_method_name(CALIB_METHOD),
            'apriltag_id': int(APRILTAG_ID),
            'tag_size_m': float(TAG_SIZE_M),
            
            'gripper_to_camera': {
                'parent_frame': EEF_FRAME,
                'child_frame': CAMERA_LINK,
                'translation': {
                    'x': float(t[0]),
                    'y': float(t[1]),
                    'z': float(t[2])
                },
                'rotation_quaternion_xyzw': [
                    float(quat[0]), float(quat[1]), 
                    float(quat[2]), float(quat[3])
                ],
                'rotation_rpy_rad': [
                    float(rpy[0]), float(rpy[1]), float(rpy[2])
                ],
                'rotation_rpy_deg': [
                    float(np.rad2deg(rpy[0])), 
                    float(np.rad2deg(rpy[1])), 
                    float(np.rad2deg(rpy[2]))
                ],
                'transformation_matrix': T_gripper2cam.tolist()
            },
            
            'camera_to_gripper_inverse': {
                'parent_frame': CAMERA_LINK,
                'child_frame': EEF_FRAME,
                'translation': {
                    'x': float(t_inv[0]),
                    'y': float(t_inv[1]),
                    'z': float(t_inv[2])
                },
                'rotation_quaternion_xyzw': [
                    float(quat_inv[0]), float(quat_inv[1]), 
                    float(quat_inv[2]), float(quat_inv[3])
                ],
                'rotation_rpy_rad': [
                    float(rpy_inv[0]), float(rpy_inv[1]), float(rpy_inv[2])
                ],
                'transformation_matrix': T_cam2gripper.tolist()
            },
            
            'static_transform_publisher_command': (
                f"rosrun tf2_ros static_transform_publisher "
                f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} "
                f"{EEF_FRAME} {CAMERA_LINK}"
            )
        }
        
        try:
            with open(OUTPUT_FILE, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            rospy.loginfo(f"Results saved to: {OUTPUT_FILE}")
        except Exception as e:
            rospy.logerr(f"Failed to save results: {e}")


def main():
    """Main function."""
    rospy.init_node('handeye_calibration_apriltag')
    
    try:
        calibrator = HandEyeCalibrator()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
