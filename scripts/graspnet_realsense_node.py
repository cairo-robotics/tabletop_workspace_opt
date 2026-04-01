#!/usr/bin/env python3
""" 
ROS Node for GraspNet grasp detection using RealSense D435 camera.
Modified from demo_realsense.py to work as a ROS node in tabletop_workspace_opt.
"""

import os
import sys
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import cv2

import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, JointState
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import tf2_geometry_msgs
from relaxed_ik_ros1.msg import EEPoseGoals

import torch
from graspnetAPI import GraspGroup

# Remove FastSAM from sys.path since it conflicts with graspnet-baseline's utils module
# This script doesn't use FastSAM, so it's safe to remove
sys.path = [p for p in sys.path if 'FastSAM' not in p]

# Add graspnet-baseline and its subdirectories to Python path for imports
GRASPNET_ROOT = '/home/yi-shiuan/sawyer_ws/src/graspnet-baseline'
paths_to_add = [
    os.path.join(GRASPNET_ROOT, 'pointnet2'),
    os.path.join(GRASPNET_ROOT, 'utils'),
    os.path.join(GRASPNET_ROOT, 'dataset'),
    os.path.join(GRASPNET_ROOT, 'models'),
    GRASPNET_ROOT,
]
# Insert in reverse order so GRASPNET_ROOT ends up first
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import from graspnet-baseline package
from models import GraspNet, pred_decode
from dataset import GraspNetDataset
from utils import ModelFreeCollisionDetector, CameraInfo as GraspNetCameraInfo, create_point_cloud_from_depth_image


class Config:
    """Configuration object that reads from ROS parameter server."""
    def __init__(self):
        # Initialize ROS node first to access parameters
        rospy.init_node('graspnet_realsense', anonymous=True)
        
        # Read parameters from ROS parameter server with default values
        self.checkpoint_path = rospy.get_param('~checkpoint_path', '')
        self.num_point = rospy.get_param('~num_point', 20000)
        self.num_view = rospy.get_param('~num_view', 300)
        self.collision_thresh = rospy.get_param('~collision_thresh', 0.01)
        self.voxel_size = rospy.get_param('~voxel_size', 0.01)
        self.depth_scale = rospy.get_param('~depth_scale', 1000.0)
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')
        self.base_frame = rospy.get_param('~base_frame', 'base')
        self.publish_to_relaxedik = rospy.get_param('~publish_to_relaxedik', False)
        self.visualize_in_rviz = rospy.get_param('~visualize_in_rviz', False)
        self.continuous = rospy.get_param('~continuous', False)
        
        # Validate required parameters
        if not self.checkpoint_path:
            rospy.logfatal("checkpoint_path parameter is required!")
            rospy.signal_shutdown("Missing required parameter: checkpoint_path")
            sys.exit(1)
        
        rospy.loginfo(f"Configuration loaded:")
        rospy.loginfo(f"  checkpoint_path: {self.checkpoint_path}")
        rospy.loginfo(f"  camera_frame: {self.camera_frame}, base_frame: {self.base_frame}")
        rospy.loginfo(f"  publish_to_relaxedik: {self.publish_to_relaxedik}")
        rospy.loginfo(f"  visualize_in_rviz: {self.visualize_in_rviz}")


cfgs = Config()


def get_net():
    """Initialize and load the GraspNet model."""
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
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
bridge = CvBridge()


def color_callback(msg):
    """Callback for color image topic."""
    global latest_color_image
    try:
        latest_color_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"Failed to convert color image: {e}")


def depth_callback(msg):
    """Callback for depth image topic."""
    global latest_depth_image
    try:
        # Depth images are typically in 16UC1 format (millimeters)
        latest_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
    except Exception as e:
        rospy.logerr(f"Failed to convert depth image: {e}")


def camera_info_callback(msg):
    """Callback for camera info topic."""
    global latest_camera_info
    # Extract intrinsics from CameraInfo message
    width = msg.width
    height = msg.height
    fx = msg.K[0]  # K is a 3x3 matrix stored row-major
    fy = msg.K[4]
    cx = msg.K[2]
    cy = msg.K[5]
    
    # Create GraspNet CameraInfo object
    latest_camera_info = GraspNetCameraInfo(width, height, fx, fy, cx, cy, cfgs.depth_scale)


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


def create_workspace_mask(depth_image, min_depth=0.3, max_depth=1.5):
    """Create a simple workspace mask based on depth range."""
    min_depth_mm = min_depth * 1000
    max_depth_mm = max_depth * 1000
    
    mask = (depth_image > min_depth_mm) & (depth_image < max_depth_mm)
    return mask.astype(np.uint8) * 255


def get_and_process_data(color_image, depth_image, camera_info):
    """Process captured camera data for grasp detection."""
    # Convert color from BGR to RGB and normalize
    color = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    depth = depth_image.astype(np.float32)
    
    # Create workspace mask
    workspace_mask = create_workspace_mask(depth)

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    # get valid points
    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
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


def transform_to_base_frame(gg, tf_buffer, camera_frame='camera_link', base_frame='base'):
    """Transform grasps from camera frame to robot base frame using TF.
    
    Args:
        gg: GraspGroup object containing grasps in camera frame
        tf_buffer: TF2 buffer for looking up transforms
        camera_frame: Name of the camera frame (default: 'camera_link')
        base_frame: Name of the robot base frame (default: 'base')
    
    Returns:
        list: List of PoseStamped objects in base frame
    """
    try:
        # Get transform from camera to base
        transform = tf_buffer.lookup_transform(
            base_frame,
            camera_frame,
            rospy.Time(0),  # Get latest available transform
            rospy.Duration(1.0)  # Wait up to 1 second
        )
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr(f"TF lookup failed: {e}")
        return []
    
    # Sort grasps by score
    gg = gg.sort_by_score()
    
    transformed_poses = []
    
    # Transform top grasps
    for i, g in enumerate(gg[:10]):  # Transform top 10 grasps
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
        pose_camera.header.stamp = rospy.Time.now()
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
            transformed_poses.append(pose_base)
            
            rospy.loginfo(f"Grasp {i}: score={g.score:.3f}, "
                         f"camera=[{t_CG[0]:.3f}, {t_CG[1]:.3f}, {t_CG[2]:.3f}], "
                         f"base=[{pose_base.pose.position.x:.3f}, {pose_base.pose.position.y:.3f}, {pose_base.pose.position.z:.3f}]")
        except Exception as e:
            rospy.logerr(f"Failed to transform grasp {i}: {e}")
            continue
    
    return transformed_poses


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
    rospy.Subscriber('/camera/color/image_raw', Image, color_callback, queue_size=1)
    rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback, queue_size=1)
    rospy.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_callback, queue_size=1)
    
    # Wait for camera data to be available
    if not wait_for_camera_data(timeout=10.0):
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
    
    try:
        if cfgs.continuous:
            rospy.loginfo("Running in continuous mode. Press 'q' to quit, 'space' to capture and process grasp.")
            while not rospy.is_shutdown():
                # Get latest images from callbacks
                if latest_color_image is None or latest_depth_image is None:
                    rospy.sleep(0.1)
                    continue
                
                # Make copies to avoid data changing during processing
                color_image = latest_color_image.copy()
                depth_image = latest_depth_image.copy()
                
                # Display color image
                cv2.imshow('RealSense Color', color_image)
                
                # Display depth image
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                cv2.imshow('RealSense Depth', depth_colormap)
                
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord(' '):
                    rospy.loginfo("Processing grasp detection...")
                    # Process data
                    end_points, cloud = get_and_process_data(color_image, depth_image, latest_camera_info)
                    
                    # Get grasps
                    gg = get_grasps(net, end_points)
                    
                    # Collision detection
                    if cfgs.collision_thresh > 0:
                        gg = collision_detection(gg, np.array(cloud.points))
                    
                    # Transform to base frame
                    rospy.loginfo("Transforming grasps to base frame...")
                    grasps_base_frame = transform_to_base_frame(gg, tf_buffer, camera_frame=cfgs.camera_frame, base_frame=cfgs.base_frame)
                    rospy.loginfo(f"Transformed {len(grasps_base_frame)} grasps to base frame")
                    
                    # Publish to RelaxedIK if enabled
                    if cfgs.publish_to_relaxedik and ee_pose_pub is not None:
                        rospy.loginfo("Publishing grasp to RelaxedIK...")
                        publish_grasps_to_relaxedik(grasps_base_frame, ee_pose_pub, num_grasps=3)
                        rospy.sleep(0.5)  # Wait for IK solution
                        
                        if latest_joint_solution is not None:
                            rospy.loginfo(f"Latest IK solution has {len(latest_joint_solution.name)} joints")
                    
                    # Visualize
                    vis_grasps(gg, cloud)
        else:
            rospy.loginfo("Single-shot mode: waiting for camera data...")
            # Wait a bit for fresh data
            rospy.sleep(1.0)
            
            # Get latest images
            if latest_color_image is not None and latest_depth_image is not None:
                # Make copies
                color_image = latest_color_image.copy()
                depth_image = latest_depth_image.copy()
                
                rospy.loginfo("Processing grasp detection...")
                # Process data
                end_points, cloud = get_and_process_data(color_image, depth_image, latest_camera_info)
                
                # Get grasps
                gg = get_grasps(net, end_points)
                
                # Collision detection
                if cfgs.collision_thresh > 0:
                    gg = collision_detection(gg, np.array(cloud.points))
                
                # Transform to base frame
                rospy.loginfo("Transforming grasps to base frame...")
                grasps_base_frame = transform_to_base_frame(gg, tf_buffer, camera_frame=cfgs.camera_frame, base_frame=cfgs.base_frame)
                rospy.loginfo(f"Transformed {len(grasps_base_frame)} grasps to base frame")
                
                # Publish to RelaxedIK if enabled
                if cfgs.publish_to_relaxedik and ee_pose_pub is not None:
                    rospy.loginfo("Publishing grasp to RelaxedIK...")
                    publish_grasps_to_relaxedik(grasps_base_frame, ee_pose_pub, num_grasps=3)
                    rospy.sleep(0.5)  # Wait for IK solution
                    
                    if latest_joint_solution is not None:
                        rospy.loginfo(f"Latest IK solution has {len(latest_joint_solution.name)} joints")
                
                # Visualize
                vis_grasps(gg, cloud)
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
