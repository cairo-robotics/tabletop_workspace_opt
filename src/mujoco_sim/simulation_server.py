#!/usr/bin/python3
import sys
# Ensure the system ROS Python path comes before any stale catkin_ws devel builds
# that may contain empty/broken moveit_msgs packages.
_ROS_PYTHON_PATH = '/opt/ros/noetic/lib/python3/dist-packages'
if _ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROS_PYTHON_PATH)
else:
    sys.path.insert(0, sys.path.pop(sys.path.index(_ROS_PYTHON_PATH)))

import rospy
import rospkg
import numpy as np
import os
import mujoco
from mujoco_visualizer import MuJoCoVisualizer
from sensor_msgs.msg import JointState, Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from intera_core_msgs.msg import EndpointState
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import threading
import time



class SimulationServer():
    """ ROS Node that manages simulation. It has a service that accepts joint commands and sends them to the 
        simulator to be executed.
    """

    def __init__(self):
        """ Initialize the simulation server node and create the service that accepts joint commands.
            Initialize the visualizer that will be used to simulate the robot.
        """
        rospkg_instance = rospkg.RosPack()
        # Get the path to the mujoco_sim package
        package_path = rospkg_instance.get_path('tabletop_workspace_opt')
        # Build the full path to the scene.xml file
        scene_name = rospy.get_param("~scene_name", "scene_breakfast")
        scene_path = os.path.join(package_path, 'src', 'assets', f'{scene_name}.xml')

        self.visualizer = MuJoCoVisualizer(scene_path)
        rospy.Subscriber("relaxed_ik/joint_angle_solutions", JointState, self.joint_solution_cb)

        # default joint positions for the Sawyer robot
        starting_config = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]
        self.visualizer.add_target_to_trajectory(starting_config)

        # Publisher for object detections
        self.det_pub = rospy.Publisher("/mujoco_sim/detections", Detection2DArray, queue_size=1)
        self.ee_pub = rospy.Publisher("/mujoco_sim/endpoint_state", EndpointState, queue_size=1)
        self.bridge = CvBridge()

        # Joint state publisher – feeds robot_state_publisher so TF and RViz stay in sync.
        self.joint_state_pub = rospy.Publisher("/robot/joint_states", JointState, queue_size=1)
        self._sawyer_joint_names = [
            "right_j0", "right_j1", "right_j2", "right_j3",
            "right_j4", "right_j5", "right_j6", "head_pan",
        ]

        # MoveIt planning scene publisher
        self.collision_pub = rospy.Publisher("/collision_object", CollisionObject, queue_size=10)
        self.dynamic_body_info = self._init_dynamic_bodies()
        self.object_names = list(self.dynamic_body_info.keys())
        rospy.loginfo(f"Tracking dynamic objects: {self.object_names}")
        self._publish_static_collision_objects()

        # Start publishing thread
        self.pub_thread = threading.Thread(target=self.publish_loop)
        self.pub_thread.daemon = True
        self.pub_thread.start()

        rospy.loginfo("Simulation server ready")
    
    def publish_loop(self):
        rate = rospy.Rate(10) # 10 Hz
        z_offset = -0.916 # offset compared to real robot
        while not rospy.is_shutdown():
            msg = Detection2DArray()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            # Publish Object Poses
            for i, obj_name in enumerate(self.object_names):
                pos = self.visualizer.get_object_position(obj_name)
                if np.isnan(pos).any(): continue
                det = Detection2D()
                det.header = msg.header

                hyp = ObjectHypothesisWithPose()
                hyp.id = i # Use index as ID
                hyp.score = 1.0
                hyp.pose.pose.position.x = pos[0]
                hyp.pose.pose.position.y = pos[1]
                hyp.pose.pose.position.z = pos[2]
                hyp.pose.pose.orientation.w = 1.0
                
                det.results.append(hyp)
                msg.detections.append(det)
                
            self.det_pub.publish(msg)

            # Publish EndpointState
            pos, quat = self.visualizer.get_pose()
            ee_msg = EndpointState()
            ee_msg.header.stamp = rospy.Time.now()
            ee_msg.header.frame_id = "world"
            ee_msg.pose.position.x = pos[0]
            ee_msg.pose.position.y = pos[1]
            ee_msg.pose.position.z = pos[2]
            ee_msg.pose.orientation.x = quat[0]
            ee_msg.pose.orientation.y = quat[1]
            ee_msg.pose.orientation.z = quat[2]
            ee_msg.pose.orientation.w = quat[3]
            self.ee_pub.publish(ee_msg)

            # Publish joint states so robot_state_publisher can broadcast TF.
            # head_pan is fixed at 0 (Sawyer head; not simulated in MuJoCo).
            js_msg = JointState()
            js_msg.header.stamp = rospy.Time.now()
            js_msg.name = self._sawyer_joint_names
            js_msg.position = list(self.visualizer.data.qpos[:7]) + [0.0]  # head_pan = 0
            self.joint_state_pub.publish(js_msg)

            # Publish dynamic objects to MoveIt planning scene
            for body_name, info in self.dynamic_body_info.items():
                obj_pos = self.visualizer.get_object_position(body_name)
                if np.isnan(obj_pos).any():
                    continue
                co = self._make_collision_object(
                    body_name, obj_pos,
                    info['half_extents'], info['center_offset']
                )
                self.collision_pub.publish(co)

            rate.sleep()

    def _init_dynamic_bodies(self) -> dict:
        """Find all free-joint bodies and precompute their bounding boxes from geom data."""
        model = self.visualizer.model
        body_info = {}
        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None:
                continue
            jnt_start = model.body_jntadr[body_id]
            jnt_num = model.body_jntnum[body_id]
            has_free_joint = any(
                model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                for j in range(jnt_start, jnt_start + jnt_num)
            )
            if not has_free_joint:
                continue
            center_offset, half_extents = self._compute_body_aabb(body_id)
            body_info[body_name] = {
                'body_id': body_id,
                'half_extents': half_extents,
                'center_offset': center_offset,
            }
        return body_info

    def _compute_body_aabb(self, body_id: int):
        """Compute AABB of a body in body-local frame from its geoms.

        Returns (center_offset, half_extents) both as np.ndarray shape (3,).
        Uses geom_aabb (precomputed by MuJoCo) and transforms each geom's AABB
        corners through the geom's local rotation to build a conservative AABB.
        """
        model = self.visualizer.model
        all_min = np.full(3, np.inf)
        all_max = np.full(3, -np.inf)
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] != body_id:
                continue
            # geom_aabb: [cx, cy, cz, hx, hy, hz] in geom-local frame
            aabb = model.geom_aabb[geom_id]
            center_local = aabb[:3]
            half = aabb[3:6]
            if np.all(half == 0):
                continue
            # Build 8 corners in geom-local frame
            signs = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                               [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=float)
            corners = signs * half + center_local  # (8,3)
            # Rotate corners into body frame (MuJoCo quat is [w,x,y,z])
            q = model.geom_quat[geom_id]  # [w,x,y,z]
            rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy: [x,y,z,w]
            corners_body = rot.apply(corners) + model.geom_pos[geom_id]
            all_min = np.minimum(all_min, corners_body.min(axis=0))
            all_max = np.maximum(all_max, corners_body.max(axis=0))
        if np.any(np.isinf(all_min)):
            return np.zeros(3), np.array([0.05, 0.05, 0.05])
        return (all_min + all_max) / 2.0, (all_max - all_min) / 2.0

    def _make_collision_object(self, name: str, position: np.ndarray,
                               half_extents: np.ndarray,
                               center_offset: np.ndarray = None,
                               operation: int = CollisionObject.ADD) -> CollisionObject:
        co = CollisionObject()
        co.header.stamp = rospy.Time.now()
        co.header.frame_id = "world"
        co.id = name
        co.operation = operation
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = (half_extents * 2.0).tolist()
        co.primitives = [prim]
        pose = Pose()
        offset = center_offset if center_offset is not None else np.zeros(3)
        pose.position.x = float(position[0] + offset[0])
        pose.position.y = float(position[1] + offset[1])
        pose.position.z = float(position[2] + offset[2])
        pose.orientation.w = 1.0
        co.primitive_poses = [pose]
        return co

    def _publish_static_collision_objects(self) -> None:
        """Publish static scene geometry (table) to the MoveIt planning scene once."""
        rospy.sleep(0.5)  # give planning scene publisher time to connect
        # Table top: body pos="0.6 0 0.915", geom box half-extents = [0.7, 0.4, 0.027]
        table_pos = np.array([0.6, 0.0, 0.915 - 0.027])
        table_half = np.array([0.7, 0.4, 0.027])
        co = self._make_collision_object("table", table_pos, table_half)
        self.collision_pub.publish(co)
        rospy.loginfo("Published static table collision object")

    def start_simulator(self) -> None:
        self.visualizer.simulate()

    def joint_solution_cb(self, joint_state: JointState) -> None:
        """ Callback function for the joint angle solutions. It receives a JointState message and sends it to the visualizer
            to be displayed.
        """
        joint_positions = np.array(joint_state.position)
        self.visualizer.add_target_to_trajectory(joint_positions)


if __name__ == "__main__":
    # set numpy print precision to 3 decimal places, no scientific notation
    np.set_printoptions(precision=3, suppress=True)

    rospy.init_node("simulation_server")
    server = SimulationServer()
    server.start_simulator()  # TODO: this should probably run in its own thread bc it blocks
    rospy.spin()
