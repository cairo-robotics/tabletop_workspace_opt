#!/usr/bin/env python3
"""
ROS node for AprilTag-based camera calibration.

"""
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Pose, Twist
from intera_core_msgs.msg import DigitalIOState
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
from relaxed_ik_ros1.msg import EEVelGoals
import threading
import message_filters
import os
from scipy.spatial.transform import Rotation
import intera_interface
from apriltag_camera_calibration import AprilTagCameraCalibration


def rt_to_T(self, Rm, tv):
    T = np.eye(4)
    T[:3,:3] = Rm
    T[:3, 3] = np.array(tv).reshape(3)
    return T

def T_to_rt(self, T):
    return T[:3,:3].copy(), T[:3,3].reshape(3,1).copy()

def invT(self, T):
    Rm = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = Rm.T
    Ti[:3,3] = -Rm.T @ t
    return Ti

def rot_angle_deg(self, R_a, R_b):
    # angle of R_b * R_a^{-1}
    Rerr = R_b @ R_a.T
    ang = Rotation.from_matrix(Rerr).magnitude()
    return float(np.rad2deg(ang))

def trans_stats(self, Ts, name=""):
    P = np.array([T[:3,3] for T in Ts])
    mean = P.mean(axis=0)
    std = P.std(axis=0)
    return mean, std

def perturb_pitch_yaw(R_nominal, max_angle_deg=5.0):
    max_angle = np.deg2rad(max_angle_deg)

    # Small pitch/yaw perturbation
    d_pitch = np.random.uniform(-max_angle, max_angle)
    d_yaw   = np.random.uniform(-max_angle, max_angle)

    R_delta = Rotation.from_euler("yx", [d_pitch, d_yaw])
    return R_delta * R_nominal


def pose_spread(T_list):
    Ts = np.array([T[:3,3] for T in T_list])
    return Ts.mean(axis=0), Ts.std(axis=0)


def clamp_norm(v, max_norm):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v * min(1.0, max_norm / n)

# limits (tune)
MAX_LIN = 0.05   # m/s
MAX_ANG = 0.3    # rad/s

DEFAULT_QUAT = np.array([0.707, 0.707, 0.0, 0.0])


class AprilTagCalibrationNode:
    def __init__(self):
        rospy.init_node('apriltag_camera_calibration', anonymous=False)
        
        # Parameters
        self.tag_size = rospy.get_param('~tag_size', 0.05)  # meters
        self.tag_family = rospy.get_param('~tag_family', 'tag36h11')
        
        # Single camera parameters
        self.camera_ns = rospy.get_param('~camera_ns', '/camera')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')
        
        self.output_file = rospy.get_param('~output_file', 
                                           os.path.expanduser('~/camera_transform.npz'))
        self.samples_file = rospy.get_param('~samples_file',
                                           os.path.expanduser('~/calibration_samples.npz'))
        self.load_existing_samples = rospy.get_param('~load_existing_samples', False)
        self.publish_tf = rospy.get_param('~publish_tf', True)
        
        # Automation parameters
        self.auto_calibrate = rospy.get_param('~auto_calibrate', False)
        self.num_samples = rospy.get_param('~num_samples', 60)
        self.cube_size_x = rospy.get_param('~cube_size_x', 0.2)
        self.cube_size_y = rospy.get_param('~cube_size_y', 0.2)
        self.cube_size_z = rospy.get_param('~cube_size_z', 0.2)
        self.min_height = rospy.get_param('~min_height', 0.2)
        
        # Parse apriltag_location (optional, for pose sampling)
        apriltag_location_param = rospy.get_param('~apriltag_location', None)
        if apriltag_location_param is not None:
            if isinstance(apriltag_location_param, str):
                import ast
                try:
                    self.apriltag_location = ast.literal_eval(apriltag_location_param)
                except (ValueError, SyntaxError) as e:
                    rospy.logwarn(f"Failed to parse apriltag_location: {e}")
                    self.apriltag_location = None
            else:
                self.apriltag_location = apriltag_location_param
        else:
            self.apriltag_location = None
        
        self.settle_time = rospy.get_param('~settle_time', 20.0)
        self.ee_vel_topic = rospy.get_param('~ee_vel_topic', '/relaxed_ik/ee_vel_goals')
        self.base_frame = rospy.get_param('~base_frame', 'base')
        self.ee_frame = rospy.get_param('~ee_frame', 'right_hand')  # End effector frame
        self.tf_prefix = rospy.get_param('~tf_prefix', '')  # Optional prefix for simulation (e.g., 'sim/')
        self.wait_for_button = rospy.get_param('~wait_for_button', True)  # Wait for button press before capturing
        
        # Velocity control parameters for linear interpolation
        self.vel_control_rate = rospy.get_param('~vel_control_rate', 30.0)  # Hz
        self.linear_velocity_scale = rospy.get_param('~linear_velocity_scale', 0.02)  # m/s
        self.angular_velocity_scale = rospy.get_param('~angular_velocity_scale', 0.1)  # rad/s
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.05)  # meters
        self.orientation_tolerance = rospy.get_param('~orientation_tolerance', 0.1)  # radians

        # apriltag gridboard
        self.tag_spacing_mm = rospy.get_param('~tag_spacing_mm', 10)
        self.tag_spacing = self.tag_spacing_mm / 1000.0
        self.rows = rospy.get_param('~rows', 3)
        self.cols = rospy.get_param('~cols', 3)
        
        # Initialize
        self.bridge = CvBridge()
        self.calib = AprilTagCameraCalibration(self.tag_size, self.tag_family)
        
        # Camera intrinsics (will be updated from CameraInfo message)
        self.cam_matrix = None
        self.cam_dist = None
        
        # TF listener for tracking end effector position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Button state for manual capture triggering
        self.button_pressed = False
        if self.wait_for_button:
            rospy.Subscriber('/robot/digital_io/right_button_ok/state', 
                           DigitalIOState, self.button_callback, queue_size=1)
        
        # TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        # Subscribers for camera info and images
        rospy.Subscriber(f'{self.camera_ns}/color/camera_info', CameraInfo, 
                        self.camera_info_callback)
        rospy.Subscriber(f'{self.camera_ns}/color/image_raw', Image, self.image_callback)
        
        # Service to capture observation
        self.capture_srv = rospy.Service('~capture_observation', Trigger, 
                                        self.capture_observation_callback)
        
        # Service to compute calibration
        self.compute_srv = rospy.Service('~compute_calibration', Trigger, 
                                        self.compute_calibration_callback)
        
        # Service to start automatic calibration
        self.auto_calib_srv = rospy.Service('~start_auto_calibration', Trigger,
                                           self.start_auto_calibration_callback)
        
        # Latest image
        self.latest_image = None
        self.latest_image_stamp = None
        
        # Automation state
        self.auto_calib_running = False
        self.auto_calib_thread = None
        
        # Publisher for robot control (if auto calibration enabled)
        if self.auto_calibrate:
            self.ee_vel_pub = rospy.Publisher(self.ee_vel_topic, EEVelGoals, queue_size=1)
            rospy.loginfo(f"Auto-calibration enabled, publishing to {self.ee_vel_topic}")
        
        # Publisher for visualization
        self.image_viz_pub = rospy.Publisher('~camera_visualization', Image, queue_size=1)
        
        # Timer for publishing visualization at lower rate
        self.viz_timer = rospy.Timer(rospy.Duration(0.1), self.publish_visualization)
        
        rospy.loginfo("AprilTag Camera Calibration Node started")
        rospy.loginfo("Mode: Single camera calibration (camera to base frame)")
        rospy.loginfo(f"Camera: {self.camera_ns}")
        rospy.loginfo(f"Camera frame: {self.camera_frame}")
        rospy.loginfo(f"Base frame: {self.base_frame}")
        rospy.loginfo(f"Tag size: {self.tag_size} m")
        rospy.loginfo("\nServices:")
        rospy.loginfo("  - Call '~capture_observation' to capture current observation")
        rospy.loginfo("  - Call '~compute_calibration' to compute final transform")
        if self.auto_calibrate:
            rospy.loginfo("  - Call '~start_auto_calibration' to start automatic calibration")
        if self.wait_for_button:
            rospy.loginfo("  - Press the OK button on the robot cuff to capture each observation")
        
        # Load existing samples if requested
        if self.load_existing_samples:
            loaded_count = self.load_samples()
            if loaded_count > 0:
                rospy.loginfo(f"✓ Starting with {loaded_count} pre-loaded samples")

        self.limb = intera_interface.Limb("right")
    
    def button_callback(self, msg: DigitalIOState):
        """Callback for robot button press."""
        # The button state message is a custom type, but we just need to detect any message
        # which indicates the button was pressed
        if msg.state == 1:
            self.button_pressed = True
            rospy.loginfo("🔘 Button pressed - ready to capture!")
        
    def camera_info_callback(self, msg):
        """Update camera intrinsics from CameraInfo message."""
        if self.cam_matrix is None:
            self.cam_matrix = np.array(msg.K).reshape(3, 3)
            self.cam_dist = np.array(msg.D)
            self.cam_info_w = msg.width
            self.cam_info_h = msg.height
            rospy.loginfo(f"CameraInfo: {msg.width}x{msg.height}, fx={self.cam_matrix[0,0]:.2f}, fy={self.cam_matrix[1,1]:.2f}")
    
    def image_callback(self, image_msg):
        """Store latest image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.latest_image_stamp = image_msg.header.stamp
            self.latest_image_frame = image_msg.header.frame_id
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")
    
    def tf_to_matrix(self, tf_msg):
        """geometry_msgs/TransformStamped -> 4x4 numpy"""
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def mat_to_rt(self, T):
        R = T[:3, :3].copy()
        t = T[:3, 3].reshape(3, 1).copy()
        return R, t
    
    def save_samples(self):
        """Save current observations to file."""
        if len(self.calib.observations) == 0:
            rospy.logwarn("No observations to save")
            return
        
        # Extract T_base_hand and T_tag_cam from observations
        T_base_hand_list = []
        T_tag_cam_list = []
        
        for obs_dict in self.calib.observations:
            T_base_hand_list.append(obs_dict['T_base_hand'])
            T_tag_cam_list.append(obs_dict['T_tag_cam'])
        
        np.savez(self.samples_file,
                 T_base_hand=np.array(T_base_hand_list),
                 T_tag_cam=np.array(T_tag_cam_list))
        
        rospy.loginfo(f"💾 Saved {len(self.calib.observations)} samples to {self.samples_file}")
    
    def load_samples(self):
        """Load observations from file if it exists."""
        if not os.path.exists(self.samples_file):
            rospy.loginfo(f"No existing samples file found at {self.samples_file}")
            return 0
        
        try:
            data = np.load(self.samples_file)
            T_base_hand_list = data['T_base_hand']
            T_tag_cam_list = data['T_tag_cam']
            
            # Add to observations
            for T_base_hand, T_tag_cam in zip(T_base_hand_list, T_tag_cam_list):
                self.calib.observations.append({
                    'T_base_hand': T_base_hand,
                    'T_tag_cam': T_tag_cam
                })
            
            rospy.loginfo(f"📂 Loaded {len(self.calib.observations)} existing samples from {self.samples_file}")
            return len(self.calib.observations)
        except Exception as e:
            rospy.logerr(f"Failed to load samples: {e}")
            return 0
    
    def capture_observation_callback(self, req):
        """Service callback to capture an observation (camera to base)."""
        # Check if we have everything we need
        if self.cam_matrix is None:
            return TriggerResponse(
                success=False,
                message="Camera intrinsics not yet received"
            )
        
        if self.latest_image is None:
            return TriggerResponse(
                success=False,
                message="No image received yet"
            )

        if hasattr(self, "latest_image_frame") and self.latest_image_frame:
            self.camera_frame = self.latest_image_frame
            rospy.loginfo(f"Camera frame updated to {self.camera_frame}")
        
        # Detect AprilTag in camera
        rospy.loginfo("Capturing observation...")
        T_tag_cam, tag_id = self.calib.detect_apriltag(
            self.latest_image,
            self.cam_matrix,
            self.cam_dist
        )

        t = T_tag_cam[:3,3]
        rospy.loginfo(f"T_tag_cam t = {t}, norm={np.linalg.norm(t):.3f}")
        
        if T_tag_cam is None:
            return TriggerResponse(
                success=False,
                message="Failed to detect AprilTag"
            )

        stamp = self.latest_image_stamp if self.latest_image_stamp is not None else rospy.Time(0)
        
        # Get current end effector pose (for wrist-mounted camera)
        try:
            tf_base_hand = self.tf_buffer.lookup_transform(
                self.tf_prefix + self.base_frame, 
                self.tf_prefix + self.ee_frame, 
                stamp, rospy.Duration(1.0)
            )
            T_base_hand = self.tf_to_matrix(tf_base_hand)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            return TriggerResponse(False, f"Failed to get end effector pose: {e}")
            
        # Store raw observations for calibrateHandEye
        self.calib.observations.append({
            "T_base_hand": T_base_hand,   # base -> hand
            "T_tag_cam": T_tag_cam,       # tag -> cam  (from OpenCV)
            "tag_id": tag_id
        })

        p = T_base_hand[:3,3]
        rospy.loginfo(f"EE in base: {p}")

        msg = f"Observation {len(self.calib.observations)} captured (tag ID: {tag_id})"
        rospy.loginfo(msg)
        
        # Save samples after each capture
        self.save_samples()
        
        return TriggerResponse(True, msg)

    def base_tag_std(self, T_hand_cam, assume_T_tag_cam_is_tag_to_cam: bool):
        Ts = []
        for obs in self.calib.observations:
            T_base_hand = obs["T_base_hand"]
            T_tag_cam   = obs["T_tag_cam"]

            if assume_T_tag_cam_is_tag_to_cam:
                T_cam_tag = np.linalg.inv(T_tag_cam)
            else:
                # assume stored was cam->tag already
                T_cam_tag = T_tag_cam

            T_base_tag = T_base_hand @ T_hand_cam @ T_cam_tag
            Ts.append(T_base_tag[:3,3])

        Ts = np.array(Ts)
        return Ts.mean(axis=0), Ts.std(axis=0)
    
    def compute_calibration_callback(self, req):
        if len(self.calib.observations) < 5:
            return TriggerResponse(
                success=False,
                message=f"Need more observations for calibrateHandEye (have {len(self.calib.observations)}, recommend >= 10)."
            )

        rospy.loginfo(f"Computing hand-eye with cv2.calibrateHandEye from {len(self.calib.observations)} observations...")

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for obs in self.calib.observations:
            T_base_hand = obs["T_base_hand"]
            T_tag_cam = obs["T_tag_cam"]

            # OpenCV wants gripper->base (NOT base->gripper)
            T_hand_base = np.linalg.inv(T_base_hand)

            Rg, tg = self.mat_to_rt(T_hand_base)  # gripper->base
            Rt, tt = self.mat_to_rt(T_tag_cam)    # target(tag)->cam

            R_gripper2base.append(Rg)
            t_gripper2base.append(tg)
            R_target2cam.append(Rt)
            t_target2cam.append(tt)

        # Choose a method; Tsai is a good default
        try:
            R_cam_hand, t_cam_hand = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
        except Exception as e:
            return TriggerResponse(False, f"cv2.calibrateHandEye failed: {e}")
        
        T_cam_hand = np.eye(4)
        T_cam_hand[:3, :3] = R_cam_hand
        T_cam_hand[:3, 3] = t_cam_hand.reshape(3)
        T_hand_cam = np.linalg.inv(T_cam_hand)

        from scipy.spatial.transform import Rotation
        translation = T_hand_cam[:3, 3]
        euler_angles = Rotation.from_matrix(T_hand_cam[:3, :3]).as_euler("xyz", degrees=True)

        rospy.loginfo("\n=== Hand-Eye Calibration Result (OpenCV calibrateHandEye) ===")
        rospy.loginfo(f"Transform {self.ee_frame} -> {self.camera_frame}:")
        rospy.loginfo(f"\n{T_hand_cam}")

        rospy.loginfo("\n=== Translation (meters) ===")
        rospy.loginfo(f"X: {translation[0]:.6f}  Y: {translation[1]:.6f}  Z: {translation[2]:.6f}")
        rospy.loginfo("\n=== Rotation (Euler XYZ, degrees) ===")
        rospy.loginfo(f"Roll: {euler_angles[0]:.2f}  Pitch: {euler_angles[1]:.2f}  Yaw: {euler_angles[2]:.2f}")
        rospy.loginfo(f"python3 set_cams_transforms.py {self.ee_frame} {self.camera_frame} {translation[0]} {translation[1]} {translation[2]} {euler_angles[0]} {euler_angles[1]} {euler_angles[2]}")

        if self.publish_tf:
            self.publish_static_transform(T_hand_cam)

        message = (
            f"Calibration complete with {len(self.calib.observations)} observations.\n"
            f"{self.ee_frame} -> {self.camera_frame}\n"
            f"t (m): [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]\n"
            f"rpy xyz (deg): [{euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}]\n"
        )

        T_base_tag_list = []
        for obs in self.calib.observations:
            T_base_hand = obs["T_base_hand"]
            T_tag_cam   = obs["T_tag_cam"]

            # base->tag = base->hand * hand->cam * cam->tag
            # cam->tag is inv(tag->cam)
            T_base_tag = T_base_hand @ T_hand_cam @ np.linalg.inv(T_tag_cam)
            T_base_tag_list.append(T_base_tag)

        mean_t, std_t = pose_spread(T_base_tag_list)
        rospy.loginfo(f"[CHECK] base->tag translation mean: {mean_t}, std: {std_t}")

        return TriggerResponse(True, message)

    def generate_calibration_poses(self):
        """
        Generate end effector poses sampled from a cube above the AprilTag.
        
        Returns:
            list of Pose: List of poses in base frame, camera oriented toward tag
        """
        # Use apriltag_location parameter if available, otherwise use default workspace center
        if self.apriltag_location is not None:
            tag_pos = np.array(self.apriltag_location)
        else:
            tag_pos = np.array([0.6, 0.0, 0.0])  # Default: 60cm in front of robot base
            rospy.logwarn("apriltag_location not set, using default sampling center [0.6, 0.0, 0.0]")
        
        rospy.loginfo(f"Generating {self.num_samples} poses in cube above AprilTag at {tag_pos}")
        rospy.loginfo(f"Cube size: {self.cube_size_x} x {self.cube_size_y} x {self.cube_size_z} m")
        rospy.loginfo(f"Min height above tag: {self.min_height} m")
        
        poses = []
        # Sample uniformly in a cube above the tag
        for i in range(self.num_samples):
            # Random position within cube
            # X: centered on tag +/- cube_size_x/2
            # Y: centered on tag +/- cube_size_y/2
            # Z: min_height to min_height + cube_size_z above tag
            x = tag_pos[0] + (np.random.rand() - 0.5) * self.cube_size_x
            y = tag_pos[1] + (np.random.rand() - 0.5) * self.cube_size_y
            z = tag_pos[2] + self.min_height + np.random.rand() * self.cube_size_z
            
            camera_pos = np.array([x, y, z])
            
            # Create Pose message
            pose = Pose()
            pose.position.x = camera_pos[0]
            pose.position.y = camera_pos[1]
            pose.position.z = camera_pos[2]
            pose.orientation.x = 0.707
            pose.orientation.y = 0.707
            pose.orientation.z = 0.0
            pose.orientation.w = 0.0
            
            poses.append(pose)
            
            rospy.logdebug(f"Pose {i+1}: pos=[{x:.3f}, {y:.3f}, {z:.3f}]")

        rospy.loginfo(f"Generated {len(poses)} calibration poses")
        return poses
    
    def get_current_ee_position(self):
        """Get current end effector position from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.tf_prefix + self.base_frame, 
                self.tf_prefix + self.ee_frame,
                rospy.Time(0), rospy.Duration(0.1)
            )
            return np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]), np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"Failed to get EE position: {e}")
            return None, None
    
    def move_to_pose(self, target_pos, target_quat):
        """
        Move end effector to target pose using unified velocity control.
        
        Controls both position and orientation simultaneously in a single loop.
        
        Args:
            target_pos: np.array([x, y, z]) target position
            target_quat: np.array([qx, qy, qz, qw]) target orientation
        
        Returns:
            bool: True if successfully reached target, False otherwise
        """
        rate = rospy.Rate(self.vel_control_rate)
        move_duration = self.settle_time * 0.7  # 70% of time for movement
        start_time = rospy.Time.now()
        
        rospy.loginfo("Moving to target pose...")
        
        while (rospy.Time.now() - start_time).to_sec() < move_duration:
            # Get current EE state
            current_pos, current_quat = self.get_current_ee_position()
            
            if current_pos is None or current_quat is None:
                rospy.logwarn_throttle(2.0, "No EE position from TF, waiting...")
                rate.sleep()
                continue
            
            # Ensure quaternion consistency (avoid 180° flips) WITHOUT mutating target_quat
            tgt = target_quat.copy()
            if np.dot(tgt, current_quat) < 0:
                tgt = -tgt
            R_target = Rotation.from_quat(tgt)
            
            # Compute position error
            position_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(position_error)

            # Orientation error
            R_current = Rotation.from_quat(current_quat)
            tgt = target_quat.copy()
            if np.dot(tgt, current_quat) < 0:
                tgt = -tgt
            R_target = Rotation.from_quat(tgt)

            R_error = R_target * R_current.inv()
            rotvec_error = R_error.as_rotvec()
            angle_error = np.linalg.norm(rotvec_error)
            
            # Check if we've reached target (both position and orientation)
            if pos_error_norm < self.position_tolerance and angle_error < self.orientation_tolerance:
                rospy.loginfo(f"✓ Reached target pose (pos: {pos_error_norm*1000:.1f}mm, orient: {np.rad2deg(angle_error):.1f}°)")
                break
            
            # --- Orientation gating ---
            # 0 when far, 1 when close
            pos_gate_start = 0.3   # 20 cm: start caring about orientation
            pos_gate_full  = 0.2   # 10 cm: fully care
            if pos_error_norm >= pos_gate_start:
                orient_gain = 0.0
            elif pos_error_norm <= pos_gate_full:
                orient_gain = 1.0
            else:
                # linear ramp
                orient_gain = (pos_gate_start - pos_error_norm) / (pos_gate_start - pos_gate_full)

            # Compute angular velocity command
            if angle_error < self.orientation_tolerance:
                angular_vel = np.zeros(3)
            else:
                angular_vel = (orient_gain * self.angular_velocity_scale) * rotvec_error
                angular_vel = clamp_norm(angular_vel, MAX_ANG)

            # Compute linear velocity command
            linear_vel = position_error * self.linear_velocity_scale
            linear_vel = clamp_norm(linear_vel, MAX_LIN)
            
            # Publish combined velocity command
            vel_msg = EEVelGoals()
            vel_msg.header.stamp = rospy.Time.now()
            vel_msg.header.frame_id = self.base_frame
            
            twist = Twist()
            twist.linear.x = linear_vel[0]
            twist.linear.y = linear_vel[1]
            twist.linear.z = linear_vel[2]
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = 0
            
            vel_msg.ee_vels.append(twist)
            vel_msg.tolerances.append(Twist())
            
            self.ee_vel_pub.publish(vel_msg)
            rate.sleep()
        
        # Send zero velocity to stop
        vel_msg = EEVelGoals()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.header.frame_id = self.base_frame
        twist = Twist()  # All zeros
        vel_msg.ee_vels.append(twist)
        vel_msg.tolerances.append(Twist())
        self.ee_vel_pub.publish(vel_msg)

        # Wait for settling
        settle_duration = self.settle_time * 0.3
        rospy.loginfo(f"Settling for {settle_duration:.1f}s...")
        rospy.sleep(settle_duration)
        
        return True

    def move_joint_angle(self, joint_name, delta):
        rospy.loginfo(f"Moving {joint_name} by {delta:.2f} radians...")
        current_position = self.limb.joint_angle(joint_name)
        joint_command = {joint_name: current_position + delta}
        self.limb.set_joint_positions(joint_command)
    
    def run_auto_calibration(self):
        """Run automatic calibration sequence."""
        rospy.loginfo("=== Starting Automatic Calibration ===")
        
        # Generate poses
        poses = self.generate_calibration_poses()
        
        if len(poses) == 0:
            rospy.logerr("Failed to generate calibration poses")
            self.auto_calib_running = False
            return
        
        # Give publisher time to connect
        rospy.sleep(1.0)
        
        successful_captures = 0
        
        for i, pose in enumerate(poses):
            if not self.auto_calib_running or rospy.is_shutdown():
                rospy.logwarn("Auto calibration stopped")
                break
            
            rospy.loginfo(f"\n--- Pose {i+1}/{len(poses)} ---")
            
            # Move to pose using helper function
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])

            rospy.loginfo(f"Moving to position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
            
            # self.move_to_pose(target_pos, DEFAULT_QUAT)
            # j6 = np.random.uniform(-0.1, 0.1)
            # self.move_joint_angle("right_j6", j6)
            # j5 = np.random.uniform(-0.1, 0.1)
            # self.move_joint_angle("right_j5", j5)
            
            # Wait for button press if enabled
            if self.wait_for_button:
                self.button_pressed = False  # Reset flag
                rospy.loginfo("⏸️  Waiting for button press... (Press OK button on cuff to capture)")
                while not self.button_pressed and not rospy.is_shutdown():
                    rospy.sleep(0.1)
                if rospy.is_shutdown():
                    break
                rospy.loginfo("✓ Button pressed, capturing now...")

            # Capture observation
            rospy.loginfo("Capturing observation...")
            
            # Check if we have everything we need
            if self.cam_matrix is None:
                rospy.logerr("Camera intrinsics not available, skipping capture")
                continue
            
            if self.latest_image is None:
                rospy.logerr("No image available, skipping capture")
                continue

            h, w = self.latest_image.shape[:2]
            fx = self.cam_matrix[0,0]
            rospy.loginfo_throttle(1.0, f"cv image {w}x{h}, fx={fx:.2f}")
            
            # Detect and add observation (using the same logic as capture_observation_callback)
            T_board_cam, meta = self.calib.detect_apriltag_gridboard(
                self.latest_image, self.cam_matrix, self.cam_dist,
                rows=self.rows, cols=self.cols,
                tag_size=self.tag_size,
                tag_spacing=self.tag_spacing,
                min_tags=2
            )
            if T_board_cam is None:
                rospy.logwarn("✗ Failed to detect AprilTag")
                continue
            
            # Get current end effector pose for hand-eye calibration
            try:
                tf_base_hand = self.tf_buffer.lookup_transform(
                    self.tf_prefix + self.base_frame, 
                    self.tf_prefix + self.ee_frame, 
                    rospy.Time(0), rospy.Duration(1.0)
                )
                T_base_hand = self.tf_to_matrix(tf_base_hand)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                return TriggerResponse(False, f"Failed to get end effector pose: {e}")
            stamp = self.latest_image_stamp if self.latest_image_stamp is not None else rospy.Time(0)
            age = (rospy.Time.now() - tf_base_hand.header.stamp).to_sec()
            rospy.loginfo(f"[sync] image_stamp={stamp.to_sec():.3f} tf_stamp={tf_base_hand.header.stamp.to_sec():.3f} age={age:.3f}s")

            # Store raw observations for calibrateHandEye
            self.calib.observations.append({
                "T_base_hand": T_base_hand,   # base -> hand
                "T_tag_cam": T_board_cam,       # tag -> cam  (from OpenCV)
                "tag_id": -1
            })

            # Move back to original position
            # self.move_joint_angle("right_j6", -j6)
            # self.move_joint_angle("right_j5", -j5)

            p = T_base_hand[:3,3]
            rospy.loginfo(f"EE in base: {p}")
            
            successful_captures += 1
            rospy.loginfo(f"✓ Observation {successful_captures} captured successfully")
            rospy.loginfo(tf_base_hand)

        # Compute final calibration
        rospy.loginfo(f"\n=== Auto Calibration Complete ===")
        rospy.loginfo(f"Successful captures: {successful_captures}/{len(poses)}")
        
        # Save all samples
        self.save_samples()
        
        if successful_captures >= 3:
            rospy.loginfo("Computing final calibration...")
            # Call the compute calibration method directly
            result = self.compute_calibration_callback(None)
            if result.success:
                rospy.loginfo("✓ Calibration computed successfully!")
            else:
                rospy.logerr(f"✗ Failed to compute calibration: {result.message}")
        else:
            rospy.logerr(f"Not enough successful captures ({successful_captures}), need at least 3")
        
        self.auto_calib_running = False
    
    def start_auto_calibration_callback(self, req):
        """Service callback to start automatic calibration."""
        if self.auto_calib_running:
            return TriggerResponse(
                success=False,
                message="Auto calibration already running"
            )
        
        if not self.auto_calibrate:
            return TriggerResponse(
                success=False,
                message="Auto calibration not enabled. Set ~auto_calibrate:=true"
            )
        
        # Start automation in separate thread
        self.auto_calib_running = True
        self.auto_calib_thread = threading.Thread(target=self.run_auto_calibration)
        self.auto_calib_thread.daemon = True
        self.auto_calib_thread.start()
        
        return TriggerResponse(
            success=True,
            message=f"Started automatic calibration with {self.num_samples} poses"
        )
    
    def visualize_apriltag_detection(self, image, camera_matrix, dist_coeffs, camera_name):
        """
        Visualize AprilTag detection on image.
        
        Args:
            image: Input image
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            camera_name: Name of camera for display
            
        Returns:
            Visualization image with AprilTag drawn
        """
        if image is None or camera_matrix is None:
            return image
        
        # Make a copy for visualization
        vis_image = image.copy()
        
        # Try to detect AprilTag
        try:
            T_tag_cam, meta = self.calib.detect_apriltag_gridboard(image, camera_matrix, dist_coeffs, rows=self.rows, cols=self.cols, tag_size=self.tag_size, tag_spacing=self.tag_spacing)
            
            if T_tag_cam is not None:

                for tag_id, corners in meta['corners_by_id'].items():
                    corners = corners.astype(int)

                    # Draw the tag outline
                    cv2.polylines(vis_image, [corners], True, (0, 255, 0), 3)
                    
                    # Draw corners
                    for i, corner in enumerate(corners):
                        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][i]
                        cv2.circle(vis_image, tuple(corner), 8, color, -1)
                    
                    # Draw tag ID
                    center = corners.mean(axis=0).astype(int)
                    cv2.putText(vis_image, f"ID: {tag_id}", 
                            (center[0] - 40, center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    
                    # Draw coordinate axes
                    axis_length = self.tag_size / 2
                    axis_points = np.float32([
                        [0, 0, 0],
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length]
                    ])
                    
                    # Project 3D points to image
                    rvec = cv2.Rodrigues(T_tag_cam[:3, :3])[0]
                    tvec = T_tag_cam[:3, 3].reshape(3, 1)
                    image_points, _ = cv2.projectPoints(axis_points, rvec, tvec, 
                                                    camera_matrix, dist_coeffs)
                    image_points = image_points.reshape(-1, 2).astype(int)
                    
                    # Draw axes (X: red, Y: green, Z: blue)
                    origin = tuple(image_points[0])
                    cv2.line(vis_image, origin, tuple(image_points[1]), (0, 0, 255), 3)  # X - red
                    cv2.line(vis_image, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y - green
                    cv2.line(vis_image, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z - blue
                    
                    # Add text showing detection success
                    cv2.putText(vis_image, "DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                # No detection
                cv2.putText(vis_image, "NO TAG DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        except Exception as e:
            rospy.logwarn(f"Error in visualization: {e}")
        
        # Add camera name
        cv2.putText(vis_image, camera_name, (10, vis_image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add observation count
        cv2.putText(vis_image, f"Observations: {len(self.calib.observations)}", 
                   (vis_image.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image
    
    def publish_visualization(self, event):
        """Timer callback to publish visualization images."""
        if self.latest_image is not None and self.cam_matrix is not None:
            vis_image = self.visualize_apriltag_detection(
                self.latest_image, self.cam_matrix, self.cam_dist, "Camera"
            )
            try:
                img_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
                self.image_viz_pub.publish(img_msg)
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Error publishing camera visualization: {e}")
    
    def publish_static_transform(self, T_hand_cam):
        """Publish the calibrated transform as a static TF."""
        from scipy.spatial.transform import Rotation
        
        static_transform = TransformStamped()
        static_transform.header.stamp = rospy.Time.now()
        static_transform.header.frame_id = self.ee_frame
        static_transform.child_frame_id = self.camera_frame
        
        # Set translation
        static_transform.transform.translation.x = T_hand_cam[0, 3]
        static_transform.transform.translation.y = T_hand_cam[1, 3]
        static_transform.transform.translation.z = T_hand_cam[2, 3]
        
        # Set rotation (convert to quaternion)
        rotation = Rotation.from_matrix(T_hand_cam[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        static_transform.transform.rotation.x = quat[0]
        static_transform.transform.rotation.y = quat[1]
        static_transform.transform.rotation.z = quat[2]
        static_transform.transform.rotation.w = quat[3]
        
        # Broadcast
        self.tf_broadcaster.sendTransform(static_transform)
        rospy.loginfo(f"Published static transform: {self.ee_frame} → {self.camera_frame}")
    
    def spin(self):
        """Main loop."""
        rospy.spin()


if __name__ == '__main__':
    try:
        node = AprilTagCalibrationNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
