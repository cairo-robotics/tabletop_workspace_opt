import os
import sys
import torch
from graspnetAPI import GraspGroup, RectGraspGroup
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree

class GraspDetector:
    def __init__(self, graspnet_root, checkpoint_path='checkpoint.tar', num_point=20000, num_view=300, 
                 collision_thresh=0.01, voxel_size=0.01, max_gripper_width=0.1):
        """Initialize GraspDetector
        
        Args:
            graspnet_root: Path to the directory containing graspnet-baseline folder
            checkpoint_path: Path to model checkpoint
            num_point: Number of points to sample
            num_view: Number of views
            collision_thresh: Collision threshold
            voxel_size: Voxel size for collision detection
            max_gripper_width: Maximum gripper opening width (meters). Used to clamp predicted grasp widths.
        """
        self.graspnet_root = graspnet_root
        self._setup_paths()
        # Clamp to a sane range (GraspNet baseline is trained with GRASP_MAX_WIDTH=0.1m).
        self.max_gripper_width = float(max(0.0, max_gripper_width))
        self._import_graspnet_modules()
        self.checkpoint_path = checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.net = self._init_network()
    
    def _setup_paths(self):
        """Setup necessary paths for graspnet-baseline"""
        paths = [
            'models',
            'dataset',
            'utils',
            ''  # for the base directory
        ]
        
        for path in paths:
            full_path = os.path.join(self.graspnet_root, path)
            if not os.path.exists(full_path):
                raise ValueError(f"Required path does not exist: {full_path}")
            if full_path not in sys.path:
                sys.path.append(full_path)

    def _import_graspnet_modules(self):
        """Import GraspNet modules after paths are set up"""
        global GraspNet, pred_decode, GraspNetDataset, ModelFreeCollisionDetector
        global CameraInfo, create_point_cloud_from_depth_image
        global graspnet_module
        
        import graspnet as graspnet_module
        # Patch the clamp used by graspnet.pred_decode() to match requested max opening width.
        graspnet_module.GRASP_MAX_WIDTH = self.max_gripper_width
        GraspNet = graspnet_module.GraspNet
        pred_decode = graspnet_module.pred_decode
        from graspnet_dataset import GraspNetDataset
        from collision_detector import ModelFreeCollisionDetector
        from data_utils import CameraInfo, create_point_cloud_from_depth_image
    
    def _init_network(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"-> loaded checkpoint {self.checkpoint_path} (epoch: {start_epoch})")
        # set model to eval mode
        net.eval()
        return net
    
    def process_data(self, color, depth, workspace_mask, meta):
        # Extract data from meta
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # Generate point cloud
        camera = CameraInfo(
            width=depth.shape[1],
            height=depth.shape[0],
            fx=intrinsic[0][0],
            fy=intrinsic[1][1],
            cx=intrinsic[0][2],
            cy=intrinsic[1][2],
            scale=factor_depth
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # Get valid points
        mask = workspace_mask & (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # Sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # Convert data to Open3D format
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        # Prepare data for neural network
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = torch.from_numpy(color_sampled.astype(np.float32)).to(device)

        return end_points, cloud_o3d
    
    def predict_grasps(self, end_points):
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        return GraspGroup(gg_array)
    
    def filter_collisions(self, grasp_group, cloud):
        if self.collision_thresh <= 0:
            return grasp_group
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.05, 
                                          collision_thresh=self.collision_thresh)
        return grasp_group[~collision_mask]
    
    def visualize_grasps(self, grasp_group, cloud, top_k=50):
        grasp_group.nms()
        grasp_group.sort_by_score()
        grasp_group = grasp_group[:top_k]
        grippers = grasp_group.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    
    def detect(self, color_image, depth_image, workspace_mask, meta, visualize=True):
        """Main method to detect grasps"""
        end_points, cloud = self.process_data(color_image, depth_image, workspace_mask, meta)
        grasp_group = self.predict_grasps(end_points)
        grasp_group = self.filter_collisions(grasp_group, np.array(cloud.points))
        
        if visualize:
            self.visualize_grasps(grasp_group, cloud)
            
        return grasp_group, cloud
        
    
                
    
    
