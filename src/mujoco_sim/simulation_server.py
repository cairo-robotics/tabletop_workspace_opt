#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
import os
from mujoco_visualizer import MuJoCoVisualizer
from sensor_msgs.msg import JointState, Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from intera_core_msgs.msg import EndpointState
from cv_bridge import CvBridge
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
        self.bridge = CvBridge()
        self.object_names = ["block1", "block2", "block3"]

        # Start publishing thread
        self.pub_thread = threading.Thread(target=self.publish_loop)
        self.pub_thread.daemon = True
        self.pub_thread.start()

        rospy.loginfo("Simulation server ready")
    
    def publish_loop(self):
        rate = rospy.Rate(10) # 10 Hz
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

            rate.sleep()

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
