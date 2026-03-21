#!/usr/bin/env python3
import sys
import os
import glob

# catkin_make_isolated creates separate devel spaces per package.
# Add all Python paths so we can import intera_core_msgs etc.
_ws = os.path.expanduser("~/sawyer_ws/devel_isolated")
for _d in glob.glob(os.path.join(_ws, "*/lib/python3/dist-packages")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import rospy
import rospkg
import numpy as np
from mujoco_visualizer import MuJoCoVisualizer
from sensor_msgs.msg import JointState, Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from intera_core_msgs.msg import EndpointState
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from std_msgs.msg import String as std_msgs_String
from std_srvs.srv import Trigger, TriggerResponse
from tabletop_workspace_opt.srv import OperateGripper, OperateGripperResponse
import threading
import time
import yaml



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
        scene_name = rospy.get_param("~scene_name", "simple_scene")
        scene_path = os.path.join(package_path, 'src', 'assets', f'{scene_name}.xml')

        self.visualizer = MuJoCoVisualizer(scene_path)
        rospy.Subscriber("relaxed_ik/joint_angle_solutions", JointState, self.joint_solution_cb)

        # default joint positions for the Sawyer robot
        starting_config = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]
        self.visualizer.add_target_to_trajectory(starting_config)
        
        # Publisher for object detections
        self.det_pub = rospy.Publisher("/mujoco_sim/detections", Detection2DArray, queue_size=1)
        self.ee_pub = rospy.Publisher("/mujoco_sim/endpoint_state", EndpointState, queue_size=1)
        self.js_pub = rospy.Publisher("/robot/joint_states", JointState, queue_size=1)
        self.js_pub2 = rospy.Publisher("/joint_states", JointState, queue_size=1)
        self.planning_scene_pub = rospy.Publisher("/planning_scene", PlanningScene, queue_size=1, latch=True)
        self.bridge = CvBridge()

        # Joint names matching the Sawyer URDF (head + 7 arm + optionally 2 gripper).
        # Gripper joints are added after checking robot_description (deferred to
        # publish_loop to avoid race with planning_context.launch).
        self.joint_names = [
            "head_pan",
            "right_j0", "right_j1", "right_j2", "right_j3",
            "right_j4", "right_j5", "right_j6",
        ]
        self.has_gripper_joints = False
        self._gripper_checked = False

        # Load object configuration from scene YAML (or fall back to hardcoded)
        self.base_z_offset = 0.92
        scene_config_path = os.path.join(package_path, 'config', 'scenes', f'{scene_name}.yaml')
        if os.path.exists(scene_config_path):
            with open(scene_config_path, 'r') as f:
                scene_cfg = yaml.safe_load(f)['scene']
            self.object_names = []
            self.object_sizes = {}
            self.object_short_names = {}
            for obj in scene_cfg['objects']:
                mname = obj['mujoco_name']
                self.object_names.append(mname)
                self.object_sizes[mname] = obj['half_extents']
                self.object_short_names[mname] = obj['short_name']
            table_cfg = scene_cfg['table']
            self.table_pos_world = table_cfg['pos_world']
            self.table_half_extents = table_cfg['half_extents']
            rospy.loginfo("Loaded scene config from %s (%d objects)",
                          scene_config_path, len(self.object_names))
        else:
            rospy.logwarn("No scene config at %s, using hardcoded breakfast defaults",
                          scene_config_path)
            self.object_names = [
                "bowl_0_a430d997_0564_4fae_801b_c01693feeee6",
                "cereal_0_1cd24fb39eb340b28b7a0ce00e6d3c6a",
                "napkin_0_d147f73d5f2249a7ba169a1cf0c21e95",
                "spoon_0_98c9fc6a50414f68979318c693fe25f8",
                "banana_0_cfd0a5186de3408cbb6fbde0cd6144ce",
                "milk_carton_0_8ce748cf2f4a410181650550275650b1",
            ]
            self.object_sizes = {
                "bowl_0_a430d997_0564_4fae_801b_c01693feeee6":      [0.08, 0.08, 0.04],
                "cereal_0_1cd24fb39eb340b28b7a0ce00e6d3c6a":        [0.035, 0.023, 0.08],
                "napkin_0_d147f73d5f2249a7ba169a1cf0c21e95":        [0.08, 0.08, 0.005],
                "spoon_0_98c9fc6a50414f68979318c693fe25f8":          [0.075, 0.015, 0.01],
                "banana_0_cfd0a5186de3408cbb6fbde0cd6144ce":        [0.09, 0.02, 0.025],
                "milk_carton_0_8ce748cf2f4a410181650550275650b1":    [0.04, 0.04, 0.11],
            }
            self.object_short_names = {}
            self.table_pos_world = [0.6, 0.0, 0.915]
            self.table_half_extents = [0.4, 0.7, 0.027]

        # Objects to exclude from the MoveIt planning scene (for grasping)
        self.excluded_from_scene = set()

        # Services for gripper, sim reset, and scene exclusion
        rospy.Service("/operate_gripper", OperateGripper, self.handle_operate_gripper)
        rospy.Service("/reset_sim", Trigger, self.handle_reset_sim)
        rospy.Subscriber("/sim/exclude_from_scene", std_msgs_String, self._exclude_cb)

        # Start publishing thread
        self.pub_thread = threading.Thread(target=self.publish_loop)
        self.pub_thread.daemon = True
        self.pub_thread.start()

        rospy.loginfo("Simulation server ready")
    
    def publish_loop(self):
        rate = rospy.Rate(10) # 10 Hz
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            det_msg = Detection2DArray()
            det_msg.header.stamp = stamp
            det_msg.header.frame_id = "world"

            scene_msg = PlanningScene()
            scene_msg.is_diff = True

            # Publish Object Poses and collision objects
            for i, obj_name in enumerate(self.object_names):
                pos = self.visualizer.get_object_position(obj_name)
                if np.isnan(pos).any():
                    continue

                # Detection message
                det = Detection2D()
                det.header = det_msg.header
                hyp = ObjectHypothesisWithPose()
                hyp.id = i
                hyp.score = 1.0
                hyp.pose.pose.position.x = pos[0]
                hyp.pose.pose.position.y = pos[1]
                hyp.pose.pose.position.z = pos[2]
                hyp.pose.pose.orientation.w = 1.0
                det.results.append(hyp)
                det_msg.detections.append(det)

                # Collision object for MoveIt / RViz (in base frame)
                co = CollisionObject()
                co.header.stamp = stamp
                co.header.frame_id = "base"
                co.id = obj_name
                # Remove excluded objects from the planning scene (for grasping)
                if obj_name in self.excluded_from_scene:
                    co.operation = CollisionObject.REMOVE
                    scene_msg.world.collision_objects.append(co)
                    continue
                co.operation = CollisionObject.ADD

                box = SolidPrimitive()
                box.type = SolidPrimitive.BOX
                half = self.object_sizes[obj_name]
                box.dimensions = [half[0] * 2, half[1] * 2, half[2] * 2]

                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = pos[2] - self.base_z_offset
                pose.orientation.w = 1.0

                co.primitives.append(box)
                co.primitive_poses.append(pose)
                scene_msg.world.collision_objects.append(co)

            # Add or remove table collision object (in base frame)
            table_co = CollisionObject()
            table_co.header.stamp = stamp
            table_co.header.frame_id = "base"
            table_co.id = "table"
            if "table" in self.excluded_from_scene:
                table_co.operation = CollisionObject.REMOVE
            else:
                table_co.operation = CollisionObject.ADD
                table_box = SolidPrimitive()
                table_box.type = SolidPrimitive.BOX
                table_box.dimensions = [
                    self.table_half_extents[0] * 2,
                    self.table_half_extents[1] * 2,
                    self.table_half_extents[2] * 2,
                ]
                table_pose = Pose()
                table_pose.position.x = self.table_pos_world[0]
                table_pose.position.y = self.table_pos_world[1]
                table_pose.position.z = self.table_pos_world[2] - self.base_z_offset
                table_pose.orientation.w = 1.0
                table_co.primitives.append(table_box)
                table_co.primitive_poses.append(table_pose)
            scene_msg.world.collision_objects.append(table_co)

            self.det_pub.publish(det_msg)
            self.planning_scene_pub.publish(scene_msg)

            # Publish EndpointState
            pos, quat = self.visualizer.get_pose()
            ee_msg = EndpointState()
            ee_msg.header.stamp = stamp
            ee_msg.header.frame_id = "world"
            ee_msg.pose.position.x = pos[0]
            ee_msg.pose.position.y = pos[1]
            ee_msg.pose.position.z = pos[2]
            ee_msg.pose.orientation.x = quat[0]
            ee_msg.pose.orientation.y = quat[1]
            ee_msg.pose.orientation.z = quat[2]
            ee_msg.pose.orientation.w = quat[3]
            self.ee_pub.publish(ee_msg)

            # Deferred check: detect if URDF has gripper joints (once, after
            # planning_context.launch has had time to set robot_description).
            if not self._gripper_checked:
                self._gripper_checked = True
                try:
                    urdf_str = rospy.get_param("robot_description", "")
                    if "right_gripper_l_finger_joint" in urdf_str:
                        self.has_gripper_joints = True
                        self.joint_names += [
                            "right_gripper_l_finger_joint",
                            "right_gripper_r_finger_joint",
                        ]
                        rospy.loginfo("Electric gripper detected in URDF")
                except Exception:
                    pass

            # Publish joint states so robot_state_publisher and MoveIt know
            # the current robot configuration.
            js_msg = JointState()
            js_msg.header.stamp = stamp
            js_msg.name = self.joint_names
            qpos = self.visualizer.data.qpos
            # head_pan=0 (static), then 7 arm joints
            positions = [0.0] + list(qpos[:7])
            velocities = [0.0] + list(self.visualizer.data.qvel[:7])

            if self.has_gripper_joints:
                # MuJoCo gripper: qpos[7]=right_close, qpos[8]=left_close, range [0, 0.035]
                # URDF: left_finger [0, 0.020833], right_finger [-0.020833, 0]
                mj_grip = max(qpos[7] if len(qpos) > 7 else 0.0,
                              qpos[8] if len(qpos) > 8 else 0.0)
                urdf_l = 0.020833 * (1.0 - mj_grip / 0.035)
                positions += [urdf_l, -urdf_l]
                velocities += [0.0, 0.0]

            js_msg.position = positions
            js_msg.velocity = velocities
            self.js_pub.publish(js_msg)
            self.js_pub2.publish(js_msg)

            rate.sleep()

    def _exclude_cb(self, msg):
        """Add or remove object names from the planning scene exclusion set.
        Prefix with '-' to re-include (e.g. '-banana_0_...')."""
        name = msg.data.strip()
        if name.startswith("-"):
            self.excluded_from_scene.discard(name[1:])
            rospy.loginfo("Re-included '%s' in planning scene", name[1:])
        elif name == "":
            self.excluded_from_scene.clear()
            rospy.loginfo("Cleared all planning scene exclusions")
        else:
            self.excluded_from_scene.add(name)
            rospy.loginfo("Excluded '%s' from planning scene", name)

    def handle_operate_gripper(self, req):
        self.visualizer.operate_gripper(req.open)
        state = "open" if req.open else "closed"
        rospy.loginfo("Gripper %s", state)
        return OperateGripperResponse(message=state)

    def handle_reset_sim(self, req):
        self.visualizer.reset()
        self.excluded_from_scene.clear()
        rospy.loginfo("Simulation reset to keyframe")
        return TriggerResponse(success=True, message="Reset to keyframe")

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
