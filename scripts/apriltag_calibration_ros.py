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
import select
import sys
import termios
import threading
import time
import tty
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
        # Capture trigger: "key" (keyboard, default), "button" (robot cuff), or "none" (no wait)
        self.capture_trigger = rospy.get_param('~capture_trigger', 'button')
        # Keep ~wait_for_button for backwards compat: if explicitly False, override to "none"
        legacy_wait = rospy.get_param('~wait_for_button', None)
        if legacy_wait is not None and not legacy_wait:
            self.capture_trigger = 'none'
        
        # Velocity control parameters for linear interpolation
        self.vel_control_rate = rospy.get_param('~vel_control_rate', 30.0)  # Hz
        self.linear_velocity_scale = rospy.get_param('~linear_velocity_scale', 0.02)  # m/s
        self.angular_velocity_scale = rospy.get_param('~angular_velocity_scale', 0.1)  # rad/s
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.05)  # meters
        self.orientation_tolerance = rospy.get_param('~orientation_tolerance', 0.1)  # radians

        # apriltag gridboard
        self.tag_spacing_mm = rospy.get_param('~tag_spacing_mm', 9)
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
        
        # Capture triggering state
        self.button_pressed = False
        self._last_button_time = 0.0
        self._key_pressed = False
        if self.capture_trigger == 'button':
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
        if self.capture_trigger == 'button':
            rospy.loginfo("  - Press the OK button on the robot cuff to capture each observation")
        elif self.capture_trigger == 'key':
            rospy.loginfo("  - Press SPACE/ENTER in terminal to capture each observation")
        
        # Load existing samples if requested
        if self.load_existing_samples:
            loaded_count = self.load_samples()
            if loaded_count > 0:
                rospy.loginfo(f"✓ Starting with {loaded_count} pre-loaded samples")

        self.limb = intera_interface.Limb("right")
    
    def button_callback(self, msg: DigitalIOState):
        """Callback for robot button press with debounce."""
        if msg.state == 1:
            now = time.time()
            if now - self._last_button_time < 0.5:
                return  # debounce: ignore presses within 500ms
            self._last_button_time = now
            self.button_pressed = True
            rospy.loginfo("Button pressed - ready to capture!")

    @staticmethod
    def _read_key(timeout_sec=0.1):
        """Non-blocking single key read from stdin (requires a TTY)."""
        if not sys.stdin.isatty():
            time.sleep(timeout_sec)
            return None
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
            if ready:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def camera_info_callback(self, msg):
        """Update camera intrinsics from CameraInfo message."""
        if self.cam_matrix is None:
            self.cam_matrix = np.array(msg.K).reshape(3, 3)
            self.cam_dist = np.array(msg.D)
            self.cam_info_w = msg.width
            self.cam_info_h = msg.height
            rospy.loginfo(f"CameraInfo: {msg.width}x{msg.height}, fx={self.cam_matrix[0,0]:.2f}, fy={self.cam_matrix[1,1]:.2f}")

    def _get_scaled_intrinsics(self, image):
        """Return (K, dist) scaled to match the actual image resolution.

        If the CameraInfo resolution differs from the image, we scale fx, fy,
        cx, cy proportionally so that solvePnP uses the correct intrinsics.
        """
        h, w = image.shape[:2]
        K = self.cam_matrix.copy()
        if w != self.cam_info_w or h != self.cam_info_h:
            sx = w / self.cam_info_w
            sy = h / self.cam_info_h
            K[0, 0] *= sx  # fx
            K[1, 1] *= sy  # fy
            K[0, 2] *= sx  # cx
            K[1, 2] *= sy  # cy
            rospy.loginfo_throttle(5.0,
                f"[intrinsics] Scaling CameraInfo {self.cam_info_w}x{self.cam_info_h} "
                f"-> image {w}x{h} (sx={sx:.3f}, sy={sy:.3f})")
        return K, self.cam_dist
    
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
        
        # Extract data from observations
        T_base_hand_list = []
        T_tag_cam_list = []
        all_corners = []

        for obs_dict in self.calib.observations:
            T_base_hand_list.append(obs_dict['T_base_hand'])
            T_tag_cam_list.append(obs_dict['T_tag_cam'])
            # Save raw corners for offline re-processing if needed
            corners = obs_dict.get('corners_by_id', {})
            all_corners.append(corners)

        save_dict = dict(
            T_base_hand=np.array(T_base_hand_list),
            T_tag_cam=np.array(T_tag_cam_list),
            corners_by_id=np.array(all_corners, dtype=object),
            tag_size=np.array(self.tag_size),
            tag_spacing=np.array(self.tag_spacing),
            grid_rows=np.array(self.rows),
            grid_cols=np.array(self.cols),
        )
        if self.cam_matrix is not None:
            save_dict['camera_matrix'] = self.cam_matrix
            save_dict['cam_info_resolution'] = np.array([self.cam_info_w, self.cam_info_h])
            save_dict['cam_dist'] = self.cam_dist
        np.savez(self.samples_file, **save_dict)
        
        rospy.loginfo(f"💾 Saved {len(self.calib.observations)} samples to {self.samples_file}")
    
    def load_samples(self):
        """Load observations from file if it exists."""
        if not os.path.exists(self.samples_file):
            rospy.loginfo(f"No existing samples file found at {self.samples_file}")
            return 0

        try:
            data = np.load(self.samples_file, allow_pickle=True)
            T_base_hand_list = data['T_base_hand']
            T_tag_cam_list = data['T_tag_cam']
            corners_list = data['corners_by_id'] if 'corners_by_id' in data else [{}] * len(T_base_hand_list)

            # Add to observations
            for i, (T_base_hand, T_tag_cam) in enumerate(zip(T_base_hand_list, T_tag_cam_list)):
                corners = corners_list[i] if i < len(corners_list) else {}
                self.calib.observations.append({
                    'T_base_hand': T_base_hand,
                    'T_tag_cam': T_tag_cam,
                    'corners_by_id': corners if isinstance(corners, dict) else {},
                })

            rospy.loginfo(f"Loaded {len(self.calib.observations)} existing samples from {self.samples_file}")
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
        
        # Detect AprilTag gridboard in camera (uses IPPE + cheirality check)
        rospy.loginfo("Capturing observation...")
        K_scaled, dist_scaled = self._get_scaled_intrinsics(self.latest_image)
        T_tag_cam, meta = self.calib.detect_apriltag_gridboard(
            self.latest_image, K_scaled, dist_scaled,
            rows=self.rows, cols=self.cols,
            tag_size=self.tag_size, tag_spacing=self.tag_spacing,
            min_tags=4
        )

        if T_tag_cam is None:
            return TriggerResponse(
                success=False,
                message="Failed to detect AprilTag gridboard (need >= 4 tags)"
            )

        t = T_tag_cam[:3,3]
        tag_id = meta['tag_ids'][0] if meta and 'tag_ids' in meta else -1
        num_tags = meta.get('num_tags', 0)
        reproj_rms = meta.get('reproj_rms', -1.0)
        rospy.loginfo(f"T_tag_cam t = {t}, norm={np.linalg.norm(t):.3f}, "
                      f"tags={num_tags}, reproj_rms={reproj_rms:.2f}px")

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
        corners_dict = meta.get('corners_by_id', {}) if meta else {}
        num_tags = meta.get('num_tags', 0) if meta else 0
        reproj_rms = meta.get('reproj_rms', -1.0) if meta else -1.0
        self.calib.observations.append({
            "T_base_hand": T_base_hand,   # base -> hand
            "T_tag_cam": T_tag_cam,       # tag -> cam  (from OpenCV)
            "tag_id": tag_id,
            "corners_by_id": corners_dict,
            "num_tags": num_tags,
            "reproj_rms": reproj_rms,
        })

        p = T_base_hand[:3,3]
        rospy.loginfo(f"EE in base: {p}")

        msg = f"Observation {len(self.calib.observations)} captured (tag ID: {tag_id})"
        rospy.loginfo(msg)
        
        # Save samples after each capture
        self.save_samples()
        
        return TriggerResponse(True, msg)

    def base_tag_std(self, T_hand_cam):
        """Compute tag-in-base statistics.

        T_hand_cam is T_{hand<-cam} (OpenCV cam2gripper output).
        Chain: T_base_tag = T_base_hand @ T_hand_cam @ T_tag_cam
        """
        Ts = []
        for obs in self.calib.observations:
            T_base_hand = obs["T_base_hand"]
            T_tag_cam   = obs["T_tag_cam"]
            T_base_tag = T_base_hand @ T_hand_cam @ T_tag_cam
            Ts.append(T_base_tag[:3,3])

        Ts = np.array(Ts)
        return Ts.mean(axis=0), Ts.std(axis=0)
    
    def _run_hand_eye(self, observations):
        """Run cv2.calibrateHandEye on a list of observations.

        Uses the CORRECT OpenCV convention:
          R_gripper2base / t_gripper2base = rotation/translation of gripper
              expressed in the base frame, i.e. T_base_hand.
          R_target2cam / t_target2cam = rotation/translation of the calibration
              target expressed in the camera frame, i.e. T_tag_cam.

        OpenCV returns cam2gripper = T_{hand<-cam}, which we store directly
        as T_hand_cam (transforms points FROM camera frame TO hand frame).

        Returns (T_hand_cam, method_name) or raises on failure.
        """
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for obs in observations:
            T_base_hand = obs["T_base_hand"]  # T_{base<-hand}: maps hand pts to base
            T_tag_cam = obs["T_tag_cam"]      # T_{cam<-tag}: maps tag pts to camera

            Rg, tg = self.mat_to_rt(T_base_hand)
            Rt, tt = self.mat_to_rt(T_tag_cam)

            R_gripper2base.append(Rg)
            t_gripper2base.append(tg)
            R_target2cam.append(Rt)
            t_target2cam.append(tt)

        methods = [
            ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
            ("PARK", cv2.CALIB_HAND_EYE_PARK),
            ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
            ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
        ]

        best_result = None
        best_std_norm = np.inf
        best_method_name = None

        for method_name, method_flag in methods:
            try:
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=method_flag
                )
            except Exception as e:
                rospy.logwarn(f"  {method_name} failed: {e}")
                continue

            # OpenCV output = cam2gripper = T_{hand<-cam}
            T_hand_cam = np.eye(4)
            T_hand_cam[:3, :3] = R_cam2gripper
            T_hand_cam[:3, 3] = t_cam2gripper.reshape(3)

            # Consistency: T_base_tag = T_base_hand @ T_hand_cam @ T_tag_cam
            # (chain: base <- hand <- cam <- tag)
            tag_positions = []
            for obs in observations:
                T_base_tag = obs["T_base_hand"] @ T_hand_cam @ obs["T_tag_cam"]
                tag_positions.append(T_base_tag[:3, 3])
            tag_positions = np.array(tag_positions)
            std = tag_positions.std(axis=0)
            std_norm = np.linalg.norm(std)

            t = T_hand_cam[:3, 3]
            rospy.loginfo(f"  {method_name}: t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] "
                          f"|t|={np.linalg.norm(t):.4f}  tag_std_norm={std_norm:.4f}")

            if std_norm < best_std_norm:
                best_std_norm = std_norm
                best_result = T_hand_cam
                best_method_name = method_name

        if best_result is None:
            raise RuntimeError("All calibrateHandEye methods failed")

        return best_result, best_method_name

    def _tag_in_base_errors(self, T_hand_cam, observations):
        """Compute per-sample tag-in-base position and distance from median.

        T_hand_cam is T_{hand<-cam} (cam2gripper from OpenCV).
        Chain: T_base_tag = T_base_hand @ T_hand_cam @ T_tag_cam

        Returns (tag_positions (N,3), errors (N,), median (3,)).
        """
        tag_positions = []
        for obs in observations:
            T_base_tag = obs["T_base_hand"] @ T_hand_cam @ obs["T_tag_cam"]
            tag_positions.append(T_base_tag[:3, 3])
        tag_positions = np.array(tag_positions)
        median = np.median(tag_positions, axis=0)
        errors = np.linalg.norm(tag_positions - median, axis=1)
        return tag_positions, errors, median

    def compute_calibration_callback(self, req):
        if len(self.calib.observations) < 5:
            return TriggerResponse(
                success=False,
                message=f"Need more observations for calibrateHandEye (have {len(self.calib.observations)}, recommend >= 10)."
            )

        observations = list(self.calib.observations)
        n_total = len(observations)
        rospy.loginfo(f"Computing hand-eye calibration from {n_total} observations...")

        # Log per-sample quality info
        for i, obs in enumerate(observations):
            num_tags = obs.get('num_tags', '?')
            reproj = obs.get('reproj_rms', -1.0)
            rospy.loginfo(f"  sample {i}: tags={num_tags}, reproj_rms={reproj:.2f}px")

        # --- Initial calibration with all samples ---
        rospy.loginfo("\n--- Initial calibration (all samples) ---")
        try:
            T_hand_cam, method_name = self._run_hand_eye(observations)
        except RuntimeError as e:
            return TriggerResponse(False, str(e))

        # --- Outlier rejection via tag-in-base consistency ---
        # Each sample should map the (stationary) tag to the same base position.
        # Two-phase rejection:
        #   1) Absolute threshold (5 cm) — good when data is clean
        #   2) Adaptive MAD-based — trims worst outliers when all data is noisy
        ABSOLUTE_THRESHOLD_M = 0.05
        MAD_MULTIPLIER = 2.0  # reject samples beyond 2x MAD from median
        MAX_ROUNDS = 3

        for round_idx in range(MAX_ROUNDS):
            tag_positions, errors, median = self._tag_in_base_errors(T_hand_cam, observations)

            # Try absolute threshold first
            abs_inlier_mask = errors < ABSOLUTE_THRESHOLD_M
            n_abs_inliers = int(abs_inlier_mask.sum())

            if n_abs_inliers >= 5:
                # Absolute threshold works — use it
                inlier_mask = abs_inlier_mask
                threshold_used = ABSOLUTE_THRESHOLD_M
                threshold_type = "absolute"
            else:
                # Fall back to adaptive MAD-based threshold
                mad = np.median(np.abs(errors - np.median(errors)))
                adaptive_threshold = np.median(errors) + MAD_MULTIPLIER * max(mad, 0.01)
                inlier_mask = errors < adaptive_threshold
                threshold_used = adaptive_threshold
                threshold_type = f"adaptive (MAD={mad:.4f})"

            n_inliers = int(inlier_mask.sum())
            n_outliers = len(observations) - n_inliers

            rospy.loginfo(f"\n--- Outlier rejection round {round_idx + 1} ({threshold_type}, threshold={threshold_used:.4f}m) ---")
            rospy.loginfo(f"  Median tag position in base: [{median[0]:.4f}, {median[1]:.4f}, {median[2]:.4f}]")
            rospy.loginfo(f"  Per-sample errors (m): {np.array2string(errors, precision=4)}")
            rospy.loginfo(f"  Inliers: {n_inliers}/{len(observations)}")

            if n_outliers == 0:
                rospy.loginfo("  No outliers found, done.")
                break

            if n_inliers < 5:
                rospy.logwarn(f"  Only {n_inliers} inliers remain — keeping all samples to avoid under-constrained solve.")
                break

            # Log which samples are rejected
            for i, (err, is_in) in enumerate(zip(errors, inlier_mask)):
                if not is_in:
                    rospy.logwarn(f"  REJECTED sample {i}: error={err:.4f}m")

            observations = [obs for obs, keep in zip(observations, inlier_mask) if keep]

            # Re-run calibration on inliers
            try:
                T_hand_cam, method_name = self._run_hand_eye(observations)
            except RuntimeError as e:
                return TriggerResponse(False, f"Re-calibration after outlier rejection failed: {e}")

        # --- Final quality report ---
        tag_positions, errors, median = self._tag_in_base_errors(T_hand_cam, observations)
        final_std = tag_positions.std(axis=0)
        final_std_norm = np.linalg.norm(final_std)

        translation = T_hand_cam[:3, 3]
        # Extrinsic XYZ decomposition: matches set_cams_transforms.py which uses
        # tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
        rpy = Rotation.from_matrix(T_hand_cam[:3, :3]).as_euler("XYZ", degrees=True)
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

        rospy.loginfo(f"\n=== Hand-Eye Calibration Result (method: {method_name}) ===")
        rospy.loginfo(f"Used {len(observations)}/{n_total} samples (rejected {n_total - len(observations)} outliers)")
        rospy.loginfo(f"Transform {self.ee_frame} -> {self.camera_frame}:")
        rospy.loginfo(f"\n{T_hand_cam}")
        rospy.loginfo(f"\n=== Translation (meters) ===")
        rospy.loginfo(f"X: {translation[0]:.6f}  Y: {translation[1]:.6f}  Z: {translation[2]:.6f}")
        rospy.loginfo(f"\n=== Rotation (extrinsic XYZ / sxyz, degrees) ===")
        rospy.loginfo(f"Roll (X): {roll:.2f}  Pitch (Y): {pitch:.2f}  Yaw (Z): {yaw:.2f}")
        rospy.loginfo(f"\n=== Quality ===")
        rospy.loginfo(f"Tag-in-base std:  [{final_std[0]:.4f}, {final_std[1]:.4f}, {final_std[2]:.4f}]  norm={final_std_norm:.4f}m")
        rospy.loginfo(f"Tag-in-base mean: [{median[0]:.4f}, {median[1]:.4f}, {median[2]:.4f}]")
        rospy.loginfo(f"Per-sample errors (m): mean={errors.mean():.4f} max={errors.max():.4f}")
        # set_cams_transforms.py expects: x y z yaw pitch roll
        rospy.loginfo(f"\npython3 set_cams_transforms.py {self.ee_frame} {self.camera_frame} "
                      f"{translation[0]} {translation[1]} {translation[2]} {yaw} {pitch} {roll}")

        if self.publish_tf:
            self.publish_static_transform(T_hand_cam)

        quality_note = ""
        if final_std_norm > 0.005:
            quality_note = (f"\nWARNING: tag position std ({final_std_norm*1000:.1f}mm) is high (>5mm). "
                           f"Consider collecting more samples with >= 4 tags visible.")

        message = (
            f"Calibration complete: {len(observations)}/{n_total} samples, method={method_name}.\n"
            f"{self.ee_frame} -> {self.camera_frame}\n"
            f"t (m): [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]\n"
            f"rpy (deg): roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f}\n"
            f"tag std norm: {final_std_norm:.4f}m, per-sample error: mean={errors.mean():.4f} max={errors.max():.4f}m"
            f"{quality_note}"
        )

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
            
            # Wait for capture trigger
            if self.capture_trigger == 'button':
                self.button_pressed = False
                rospy.loginfo("Waiting for button press... (Press OK button on cuff to capture)")
                while not self.button_pressed and not rospy.is_shutdown():
                    rospy.sleep(0.1)
                if rospy.is_shutdown():
                    break
                rospy.loginfo("Button pressed, capturing now...")
            elif self.capture_trigger == 'key':
                self._key_pressed = False
                rospy.loginfo("Press SPACE or ENTER to capture (q to quit calibration)...")
                while not self._key_pressed and not rospy.is_shutdown():
                    key = self._read_key(timeout_sec=0.1)
                    if key in (' ', '\n', '\r'):
                        self._key_pressed = True
                    elif key == 'q':
                        rospy.loginfo("Quit requested, stopping calibration.")
                        self.auto_calib_running = False
                        break
                if not self.auto_calib_running or rospy.is_shutdown():
                    break
                rospy.loginfo("Key pressed, capturing now...")

            # Capture observation
            rospy.loginfo("Capturing observation...")
            
            # Check if we have everything we need
            if self.cam_matrix is None:
                rospy.logerr("Camera intrinsics not available, skipping capture")
                continue
            
            if self.latest_image is None:
                rospy.logerr("No image available, skipping capture")
                continue

            K_scaled, dist_scaled = self._get_scaled_intrinsics(self.latest_image)
            h, w = self.latest_image.shape[:2]
            rospy.loginfo_throttle(1.0, f"cv image {w}x{h}, fx={K_scaled[0,0]:.2f} (CameraInfo {self.cam_info_w}x{self.cam_info_h})")

            # Detect and add observation (using the same logic as capture_observation_callback)
            T_board_cam, meta = self.calib.detect_apriltag_gridboard(
                self.latest_image, K_scaled, dist_scaled,
                rows=self.rows, cols=self.cols,
                tag_size=self.tag_size,
                tag_spacing=self.tag_spacing,
                min_tags=4
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

            # Validate board pose: must be in front of camera with plausible distance
            board_z = T_board_cam[2, 3]
            board_dist = np.linalg.norm(T_board_cam[:3, 3])
            if board_z <= 0.05:
                rospy.logwarn(f"✗ Board behind camera (z={board_z:.4f}m), skipping sample")
                continue
            if board_dist < 0.10 or board_dist > 2.0:
                rospy.logwarn(f"✗ Implausible board distance ({board_dist:.4f}m), skipping sample")
                continue
            rospy.loginfo(f"  Board distance: {board_dist:.3f}m (z={board_z:.3f}m) ✓")

            # Store raw observations for calibrateHandEye
            # Include raw corner data so we can re-run solvePnP offline if needed
            corners_dict = meta.get('corners_by_id', {}) if meta else {}
            self.calib.observations.append({
                "T_base_hand": T_base_hand,   # base -> hand
                "T_tag_cam": T_board_cam,       # tag -> cam  (from OpenCV)
                "tag_id": -1,
                "corners_by_id": corners_dict,  # raw image corners per tag
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
    
    def visualize_apriltag_detection(self, image, camera_matrix=None, dist_coeffs=None, camera_name="Camera"):
        """
        Visualize AprilTag detection on image.

        Args:
            image: Input image
            camera_matrix: Camera intrinsic matrix (if None, uses scaled intrinsics)
            dist_coeffs: Distortion coefficients
            camera_name: Name of camera for display

        Returns:
            Visualization image with AprilTag drawn
        """
        if image is None or self.cam_matrix is None:
            return image
        if camera_matrix is None:
            camera_matrix, dist_coeffs = self._get_scaled_intrinsics(image)
        
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
                self.latest_image, camera_name="Camera"
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
