#!/usr/bin/env python3
""" 
ROS Node for GraspNet grasp detection using RealSense D435 camera.
Modified from demo_realsense.py to work as a ROS node in tabletop_workspace_opt.
"""

import os
import sys
import numpy as np
import open3d as o3d
import cv2

import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3
import tf2_ros
import tf2_geometry_msgs
from relaxed_ik_ros1.msg import EEPoseGoals
from tabletop_workspace_opt.msg import GraspCandidate, GraspCandidateArray

import torch

try:
    import pyrealsense2 as rs  # noqa: F401
except ImportError:
    rs = None


GraspGroup = None
GraspNet = None
pred_decode = None
ModelFreeCollisionDetector = None
GraspNetCameraInfo = None
create_point_cloud_from_depth_image = None


def setup_graspnet_imports(graspnet_root):
    """Add graspnet-baseline checkout to sys.path and import runtime modules."""
    global GraspGroup
    global GraspNet
    global pred_decode
    global ModelFreeCollisionDetector
    global GraspNetCameraInfo
    global create_point_cloud_from_depth_image

    # Remove FastSAM from sys.path since it conflicts with graspnet-baseline's utils module.
    sys.path[:] = [p for p in sys.path if 'FastSAM' not in p]

    paths_to_add = [
        os.path.join(graspnet_root, 'pointnet2'),
        os.path.join(graspnet_root, 'utils'),
        os.path.join(graspnet_root, 'dataset'),
        os.path.join(graspnet_root, 'models'),
        graspnet_root,
    ]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

    from graspnetAPI import GraspGroup as ImportedGraspGroup
    from graspnet import GraspNet as ImportedGraspNet, pred_decode as imported_pred_decode
    from collision_detector import ModelFreeCollisionDetector as ImportedCollisionDetector
    from data_utils import (
        CameraInfo as ImportedCameraInfo,
        create_point_cloud_from_depth_image as imported_create_point_cloud,
    )

    GraspGroup = ImportedGraspGroup
    GraspNet = ImportedGraspNet
    pred_decode = imported_pred_decode
    ModelFreeCollisionDetector = ImportedCollisionDetector
    GraspNetCameraInfo = ImportedCameraInfo
    create_point_cloud_from_depth_image = imported_create_point_cloud


class Config:
    """Configuration object that reads from ROS parameter server."""
    def __init__(self):
        # Initialize ROS node first to access parameters
        rospy.init_node('graspnet_realsense', anonymous=True)

        default_graspnet_root = os.environ.get('GRASPNET_ROOT', '/home/yi-shiuan/sawyer_ws/src/graspnet-baseline')

        # Read parameters from ROS parameter server with default values
        self.graspnet_root = rospy.get_param('~graspnet_root', default_graspnet_root)
        self.checkpoint_path = rospy.get_param('~checkpoint_path', '')
        self.num_point = rospy.get_param('~num_point', 20000)
        self.num_view = rospy.get_param('~num_view', 300)
        self.collision_thresh = rospy.get_param('~collision_thresh', 0.01)
        self.voxel_size = rospy.get_param('~voxel_size', 0.01)
        self.depth_scale = rospy.get_param('~depth_scale', 1000.0)
        self.workspace_min_depth_m = float(rospy.get_param('~workspace_min_depth_m', 0.05))
        self.workspace_max_depth_m = float(rospy.get_param('~workspace_max_depth_m', 1.5))
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')
        self.base_frame = rospy.get_param('~base_frame', 'base')
        self.candidate_z_min = float(rospy.get_param('~candidate_z_min', 'nan'))
        self.candidate_z_max = float(rospy.get_param('~candidate_z_max', 'nan'))
        self.color_topic = rospy.get_param('~color_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')
        self.publish_to_relaxedik = rospy.get_param('~publish_to_relaxedik', False)
        self.visualize_in_rviz = rospy.get_param('~visualize_in_rviz', False)
        self.continuous = rospy.get_param('~continuous', False)
        self.interactive = rospy.get_param('~interactive', False)
        self.process_rate_hz = float(rospy.get_param('~process_rate_hz', 1.0))
        self.publish_candidate_grasps = rospy.get_param('~publish_candidate_grasps', True)
        self.candidate_grasp_topic = rospy.get_param('~candidate_grasp_topic', '/candidate_grasps')
        self.candidate_grasp_top_k = rospy.get_param('~candidate_grasp_top_k', 10)
        self.random_seed = int(rospy.get_param('~random_seed', 0))
        self.enable_nms = rospy.get_param('~enable_nms', True)
        self.camera_startup_delay_sec = float(rospy.get_param('~camera_startup_delay_sec', 0.0))
        self.camera_wait_timeout_sec = float(rospy.get_param('~camera_wait_timeout_sec', 30.0))

        # Validate required parameters
        if not self.graspnet_root:
            rospy.logfatal("graspnet_root parameter is required!")
            rospy.signal_shutdown("Missing required parameter: graspnet_root")
            sys.exit(1)
        if not os.path.isdir(self.graspnet_root):
            rospy.logfatal("graspnet_root does not exist: %s", self.graspnet_root)
            rospy.signal_shutdown("Invalid graspnet_root")
            sys.exit(1)
        if not self.checkpoint_path:
            rospy.logfatal("checkpoint_path parameter is required!")
            rospy.signal_shutdown("Missing required parameter: checkpoint_path")
            sys.exit(1)

        rospy.loginfo(f"Configuration loaded:")
        rospy.loginfo(f"  graspnet_root: {self.graspnet_root}")
        rospy.loginfo(f"  checkpoint_path: {self.checkpoint_path}")
        rospy.loginfo(f"  camera_frame: {self.camera_frame}, base_frame: {self.base_frame}")
        rospy.loginfo(
            f"  workspace depth range: [{self.workspace_min_depth_m:.3f}, {self.workspace_max_depth_m:.3f}] m"
        )
        rospy.loginfo(
            "  candidate z gate: min=%s max=%s",
            "disabled" if np.isnan(self.candidate_z_min) else f"{self.candidate_z_min:.3f}",
            "disabled" if np.isnan(self.candidate_z_max) else f"{self.candidate_z_max:.3f}",
        )
        rospy.loginfo(f"  color_topic: {self.color_topic}")
        rospy.loginfo(f"  depth_topic: {self.depth_topic}")
        rospy.loginfo(f"  camera_info_topic: {self.camera_info_topic}")
        rospy.loginfo(f"  publish_to_relaxedik: {self.publish_to_relaxedik}")
        rospy.loginfo(f"  visualize_in_rviz: {self.visualize_in_rviz}")
        rospy.loginfo(f"  continuous: {self.continuous}, interactive: {self.interactive}")
        rospy.loginfo(f"  publish_candidate_grasps: {self.publish_candidate_grasps}")
        rospy.loginfo(
            "  camera startup wait: delay=%.2fs timeout=%.2fs",
            self.camera_startup_delay_sec,
            self.camera_wait_timeout_sec,
        )


cfgs = Config()
setup_graspnet_imports(cfgs.graspnet_root)


def get_net():
    """Initialize and load the GraspNet model."""
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    rospy.loginfo("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


# Global variables to store latest camera data from ROS topics
latest_color_image = None
latest_depth_image = None
latest_camera_info = None
latest_color_stamp = None
latest_depth_stamp = None
latest_camera_info_stamp = None
bridge = CvBridge()


def color_callback(msg):
    """Callback for color image topic."""
    global latest_color_image
    global latest_color_stamp
    try:
        latest_color_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        latest_color_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
    except Exception as e:
        rospy.logerr(f"Failed to convert color image: {e}")


def depth_callback(msg):
    """Callback for depth image topic."""
    global latest_depth_image
    global latest_depth_stamp
    try:
        latest_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        latest_depth_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
    except Exception as e:
        rospy.logerr(f"Failed to convert depth image: {e}")


def camera_info_callback(msg):
    """Callback for camera info topic."""
    global latest_camera_info
    global latest_camera_info_stamp
    # Extract intrinsics from CameraInfo message
    width = msg.width
    height = msg.height
    fx = msg.K[0]  # K is a 3x3 matrix stored row-major
    fy = msg.K[4]
    cx = msg.K[2]
    cy = msg.K[5]
    
    # Create GraspNet CameraInfo object
    latest_camera_info = GraspNetCameraInfo(width, height, fx, fy, cx, cy, cfgs.depth_scale)
    latest_camera_info_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()


def wait_for_camera_data(timeout=10.0):
    """Wait for camera data to be available from ROS topics."""
    rospy.loginfo("Waiting for camera data...")
    start_time = rospy.Time.now()
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if latest_color_image is not None and latest_depth_image is not None and latest_camera_info is not None:
            rospy.loginfo("Camera data received!")
            return True
        
        if (rospy.Time.now() - start_time).to_sec() > timeout:
            rospy.logerr(f"Timeout waiting for camera data after {timeout} seconds")
            return False
        
        rate.sleep()
    
    return False


def create_workspace_mask(depth_image, min_depth=0.05, max_depth=1.5):
    """Create a simple workspace mask based on depth range."""
    depth_units_scale = float(cfgs.depth_scale) if float(cfgs.depth_scale) > 0 else 1.0
    min_depth_units = min_depth * depth_units_scale
    max_depth_units = max_depth * depth_units_scale

    mask = np.isfinite(depth_image) & (depth_image > min_depth_units) & (depth_image < max_depth_units)
    return mask.astype(np.uint8) * 255


def get_and_process_data(color_image, depth_image, camera_info):
    """Process captured camera data for grasp detection."""
    # Convert color from BGR to RGB and normalize
    color = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = depth_image.astype(np.float32)
    
    # Create workspace mask
    workspace_mask = create_workspace_mask(
        depth,
        min_depth=cfgs.workspace_min_depth_m,
        max_depth=cfgs.workspace_max_depth_m,
    )

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    # get valid points
    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) == 0:
        raise RuntimeError("No valid points left after workspace masking.")

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        rng = np.random.default_rng(cfgs.random_seed)
        idxs = rng.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        rng = np.random.default_rng(cfgs.random_seed)
        idxs2 = rng.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d


def get_grasps(net, end_points):
    """Run grasp detection."""
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    if cfgs.enable_nms:
        gg = gg.nms()
    gg = gg.sort_by_score()
    rospy.loginfo(f"Top 10 grasp scores: {[g.score for g in gg[:10]]}")
    return gg


def collision_detection(gg, cloud):
    """Filter grasps based on collision detection."""
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg


def vis_grasps(gg, cloud):
    """Visualize top grasps."""
    gg.nms()
    gg.sort_by_score()
    gg = gg[:10]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def transform_grasps_to_base_frame(
    gg,
    tf_buffer,
    camera_frame='camera_link',
    base_frame='base',
    top_k=10,
    stamp=None,
):
    """Transform grasps from camera frame to robot base frame using TF.
    
    Args:
        gg: GraspGroup object containing grasps in camera frame
        tf_buffer: TF2 buffer for looking up transforms
        camera_frame: Name of the camera frame (default: 'camera_link')
        base_frame: Name of the robot base frame (default: 'base')
    
    Returns:
        list: List of dictionaries with pose, score, approach direction, and feasibility.
    """
    try:
        # Get transform from camera to base
        lookup_stamp = stamp if stamp is not None else rospy.Time(0)
        transform = tf_buffer.lookup_transform(
            base_frame,
            camera_frame,
            lookup_stamp,
            rospy.Duration(1.0)  # Wait up to 1 second
        )
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"TF lookup failed: {e}")
        return []
    
    # Sort grasps by score
    gg = gg.sort_by_score()
    
    transformed_grasps = []
    
    # Transform top grasps
    for i, g in enumerate(gg[:top_k]):
        # Get grasp pose in camera frame
        t_CG = g.translation  # (3,) grasp center in camera frame
        R_CG = g.rotation_matrix.reshape(3, 3)  # (3,3) rotation in camera frame
        
        # Create transformation matrix T_CG (camera to grasp)
        T_CG = np.eye(4)
        T_CG[:3, :3] = R_CG
        T_CG[:3, 3] = t_CG
        
        # Create PoseStamped in camera frame
        pose_camera = PoseStamped()
        pose_camera.header.frame_id = camera_frame
        pose_camera.header.stamp = stamp if stamp is not None else rospy.Time.now()
        pose_camera.pose.position.x = t_CG[0]
        pose_camera.pose.position.y = t_CG[1]
        pose_camera.pose.position.z = t_CG[2]
        
        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(R_CG).as_quat()  # [x, y, z, w]
        pose_camera.pose.orientation.x = quat[0]
        pose_camera.pose.orientation.y = quat[1]
        pose_camera.pose.orientation.z = quat[2]
        pose_camera.pose.orientation.w = quat[3]
        
        # Transform to base frame
        try:
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_camera, transform)
            pose_base.header.frame_id = base_frame

            # GraspNet rotation matrices use the first column as the approach axis.
            rotation_base = np.array([
                pose_base.pose.orientation.x,
                pose_base.pose.orientation.y,
                pose_base.pose.orientation.z,
                pose_base.pose.orientation.w,
            ])
            quat_norm = np.linalg.norm(rotation_base)
            if quat_norm > 0:
                rotation_base = rotation_base / quat_norm
            from scipy.spatial.transform import Rotation
            approach_base = Rotation.from_quat(rotation_base).as_matrix()[:, 0]
            z_world = float(pose_base.pose.position.z)

            if not np.isnan(cfgs.candidate_z_min) and z_world < cfgs.candidate_z_min:
                rospy.loginfo(
                    "Skipping grasp %d below candidate_z_min: z=%.3f < %.3f",
                    i,
                    z_world,
                    cfgs.candidate_z_min,
                )
                continue
            if not np.isnan(cfgs.candidate_z_max) and z_world > cfgs.candidate_z_max:
                rospy.loginfo(
                    "Skipping grasp %d above candidate_z_max: z=%.3f > %.3f",
                    i,
                    z_world,
                    cfgs.candidate_z_max,
                )
                continue

            transformed_grasps.append({
                'index': i,
                'pose': pose_base,
                'score': float(g.score),
                'approach_direction': approach_base,
                'feasible': True,
            })
            
            rospy.loginfo(f"Grasp {i}: score={g.score:.3f}, "
                         f"camera=[{t_CG[0]:.3f}, {t_CG[1]:.3f}, {t_CG[2]:.3f}], "
                         f"base=[{pose_base.pose.position.x:.3f}, {pose_base.pose.position.y:.3f}, {pose_base.pose.position.z:.3f}]")
        except Exception as e:
            rospy.logerr(f"Failed to transform grasp {i}: {e}")
            continue
    
    return transformed_grasps


def transform_to_base_frame(gg, tf_buffer, camera_frame='camera_link', base_frame='base'):
    """Backward-compatible helper that returns only transformed poses."""
    transformed_grasps = transform_grasps_to_base_frame(
        gg,
        tf_buffer,
        camera_frame=camera_frame,
        base_frame=base_frame,
        top_k=10,
    )
    return [item['pose'] for item in transformed_grasps]


def publish_candidate_grasps(transformed_grasps, candidate_pub, stamp, frame_id):
    """Publish transformed grasp candidates for downstream shared autonomy nodes."""
    if candidate_pub is None:
        return

    msg = GraspCandidateArray()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    for grasp in transformed_grasps:
        candidate = GraspCandidate()
        candidate.grasp_id = f"grasp_{grasp['index']}"
        candidate.pose = grasp['pose'].pose
        candidate.approach_direction = Vector3(
            x=float(grasp['approach_direction'][0]),
            y=float(grasp['approach_direction'][1]),
            z=float(grasp['approach_direction'][2]),
        )
        candidate.grasp_score = float(grasp['score'])
        candidate.feasible = bool(grasp['feasible'])
        msg.grasps.append(candidate)

    candidate_pub.publish(msg)
    rospy.loginfo(
        "Published %d candidate grasps on %s",
        len(msg.grasps),
        getattr(candidate_pub, 'resolved_name', 'candidate_grasps'),
    )


def publish_empty_candidate_grasps(candidate_pub, stamp, frame_id):
    """Clear candidate grasp markers when detection fails or produces no grasps."""
    if candidate_pub is None:
        return

    msg = GraspCandidateArray()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    candidate_pub.publish(msg)


# Global variable to store the latest joint solution from RelaxedIK
latest_joint_solution = None


def joint_solution_callback(msg, js_preview_pub):
    """Callback for RelaxedIK joint angle solutions.
    
    Args:
        msg: JointState message from RelaxedIK
        js_preview_pub: Publisher for joint state preview (or None)
    """
    global latest_joint_solution
    latest_joint_solution = msg
    
    # Publish for RViz visualization if publisher is provided
    if js_preview_pub is not None:
        preview_msg = JointState()
        preview_msg.header.stamp = rospy.Time.now()
        preview_msg.name = msg.name
        preview_msg.position = msg.position
        js_preview_pub.publish(preview_msg)
        
    rospy.loginfo(f"Received IK solution with {len(msg.name)} joints")


def publish_grasps_to_relaxedik(grasps_base_frame, ee_pose_pub, num_grasps=1):
    """Publish grasps to RelaxedIK for IK computation.
    
    Args:
        grasps_base_frame: List of PoseStamped objects in base frame
        ee_pose_pub: Publisher for EEPoseGoals
        num_grasps: Number of top grasps to publish (default: 1)
    """
    if len(grasps_base_frame) == 0:
        rospy.logwarn("No grasps to publish to RelaxedIK")
        return
    
    # Publish the top grasps one at a time
    for i in range(min(num_grasps, len(grasps_base_frame))):
        ee_pose_goals = EEPoseGoals()
        ee_pose_goals.header.stamp = rospy.Time.now()
        ee_pose_goals.header.frame_id = grasps_base_frame[i].header.frame_id
        
        # Add the grasp pose to the array
        ee_pose_goals.ee_poses = [grasps_base_frame[i].pose]
        
        # Publish
        ee_pose_pub.publish(ee_pose_goals)
        rospy.loginfo(f"Published grasp {i} to RelaxedIK: "
                     f"pos=[{grasps_base_frame[i].pose.position.x:.3f}, "
                     f"{grasps_base_frame[i].pose.position.y:.3f}, "
                     f"{grasps_base_frame[i].pose.position.z:.3f}]")
        
        # Wait briefly for IK solution
        if i < num_grasps - 1:
            rospy.sleep(0.5)


def demo_realsense():
    """Main demo function."""
    # ROS node already initialized in Config class

    # Subscribe to camera topics
    rospy.loginfo("Subscribing to camera topics...")
    rospy.Subscriber(cfgs.color_topic, Image, color_callback, queue_size=1)
    rospy.Subscriber(cfgs.depth_topic, Image, depth_callback, queue_size=1)
    rospy.Subscriber(cfgs.camera_info_topic, CameraInfo, camera_info_callback, queue_size=1)

    if cfgs.camera_startup_delay_sec > 0.0:
        rospy.loginfo(
            "Waiting %.2f seconds before checking for camera data...",
            cfgs.camera_startup_delay_sec,
        )
        rospy.sleep(cfgs.camera_startup_delay_sec)
    
    # Wait for camera data to be available
    if not wait_for_camera_data(timeout=cfgs.camera_wait_timeout_sec):
        rospy.logfatal("Failed to receive camera data. Make sure the RealSense node is running.")
        return
    
    rospy.loginfo(f"Camera info: {latest_camera_info.width}x{latest_camera_info.height}, "
                 f"fx={latest_camera_info.fx:.2f}, fy={latest_camera_info.fy:.2f}")
    
    # Initialize TF2 for coordinate transforms
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)  # Give TF listener time to fill buffer
    
    # Initialize network
    net = get_net()
    
    # Initialize RelaxedIK publishers and subscribers if requested
    ee_pose_pub = None
    js_preview_pub = None
    candidate_pub = None
    if cfgs.publish_to_relaxedik:
        rospy.loginfo("Initializing RelaxedIK publisher...")
        ee_pose_pub = rospy.Publisher('/relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
        
        if cfgs.visualize_in_rviz:
            rospy.loginfo("Initializing RViz joint state preview publisher...")
            js_preview_pub = rospy.Publisher('/joint_states_preview', JointState, queue_size=1)
        
        # Subscribe to IK solutions
        rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointState, 
                        lambda msg: joint_solution_callback(msg, js_preview_pub))
        rospy.loginfo("Subscribed to RelaxedIK joint angle solutions")
        rospy.sleep(0.5)  # Give time for connections to establish

    if cfgs.publish_candidate_grasps:
        candidate_pub = rospy.Publisher(cfgs.candidate_grasp_topic, GraspCandidateArray, queue_size=1)
        rospy.loginfo("Publishing shared autonomy grasp candidates on %s", cfgs.candidate_grasp_topic)
    
    def process_latest_frame():
        if latest_color_image is None or latest_depth_image is None or latest_camera_info is None:
            rospy.logwarn_throttle(2.0, "No camera data available yet for grasp detection.")
            return

        color_image = latest_color_image.copy()
        depth_image = latest_depth_image.copy()
        frame_stamp = latest_depth_stamp or latest_color_stamp or latest_camera_info_stamp or rospy.Time.now()

        rospy.loginfo("Processing grasp detection...")
        end_points, cloud = get_and_process_data(color_image, depth_image, latest_camera_info)

        gg = get_grasps(net, end_points)

        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))

        rospy.loginfo("Transforming grasps to base frame...")
        transformed_grasps = transform_grasps_to_base_frame(
            gg,
            tf_buffer,
            camera_frame=cfgs.camera_frame,
            base_frame=cfgs.base_frame,
            top_k=cfgs.candidate_grasp_top_k,
            stamp=frame_stamp,
        )
        grasps_base_frame = [item['pose'] for item in transformed_grasps]
        rospy.loginfo(f"Transformed {len(grasps_base_frame)} grasps to base frame")

        if candidate_pub is not None:
            publish_candidate_grasps(
                transformed_grasps,
                candidate_pub,
                stamp=frame_stamp,
                frame_id=cfgs.base_frame,
            )

        if cfgs.publish_to_relaxedik and ee_pose_pub is not None:
            rospy.loginfo("Publishing grasp to RelaxedIK...")
            publish_grasps_to_relaxedik(grasps_base_frame, ee_pose_pub, num_grasps=3)
            rospy.sleep(0.5)

            if latest_joint_solution is not None:
                rospy.loginfo(f"Latest IK solution has {len(latest_joint_solution.name)} joints")

        if cfgs.visualize_in_rviz:
            vis_grasps(gg, cloud)

    try:
        if cfgs.continuous and cfgs.interactive:
            rospy.loginfo("Running in interactive continuous mode. Press 'q' to quit, 'space' to capture and process grasp.")
            while not rospy.is_shutdown():
                if latest_color_image is None or latest_depth_image is None:
                    rospy.sleep(0.1)
                    continue

                color_image = latest_color_image.copy()
                depth_image = latest_depth_image.copy()

                cv2.imshow('RealSense Color', color_image)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.imshow('RealSense Depth', depth_colormap)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord(' '):
                    process_latest_frame()
        elif cfgs.continuous:
            rospy.loginfo("Running in automatic continuous mode at %.2f Hz.", cfgs.process_rate_hz)
            rate = rospy.Rate(max(cfgs.process_rate_hz, 0.1))
            while not rospy.is_shutdown():
                try:
                    process_latest_frame()
                except Exception as exc:
                    failure_stamp = latest_depth_stamp or latest_color_stamp or latest_camera_info_stamp or rospy.Time.now()
                    publish_empty_candidate_grasps(candidate_pub, failure_stamp, cfgs.base_frame)
                    rospy.logerr_throttle(2.0, "Automatic grasp detection failed: %s", exc)
                rate.sleep()
        else:
            rospy.loginfo("Single-shot mode: waiting for camera data...")
            rospy.sleep(1.0)
            if latest_color_image is not None and latest_depth_image is not None and latest_camera_info is not None:
                process_latest_frame()
            else:
                rospy.logerr("No camera data available")
    
    finally:
        # Clean up
        cv2.destroyAllWindows()
        rospy.loginfo("Node shutting down.")


if __name__=='__main__':
    try:
        demo_realsense()
    except rospy.ROSInterruptException:
        pass
