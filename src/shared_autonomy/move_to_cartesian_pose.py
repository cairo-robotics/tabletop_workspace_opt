#!/usr/bin/python3
"""MoveIt-based Cartesian pose motion planner for Sawyer.

Plans collision-free trajectories with MoveIt and bridges execution to the
MuJoCo simulator by publishing waypoints to relaxed_ik/joint_angle_solutions.
"""
import sys
# Ensure the system ROS Python path comes before any stale catkin_ws devel builds
# that may contain empty/broken moveit_commander or moveit_msgs packages.
_ROS_PYTHON_PATH = '/opt/ros/noetic/lib/python3/dist-packages'
if _ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROS_PYTHON_PATH)
else:
    sys.path.insert(0, sys.path.pop(sys.path.index(_ROS_PYTHON_PATH)))

import rospy
import moveit_commander
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from tabletop_workspace_opt.srv import (
    MoveToCartesianPose, MoveToCartesianPoseResponse,
    OperateGripper, OperateGripperResponse,
)

# Seconds to wait per second of trajectory time before sending the next waypoint.
# Increase if MuJoCo can't keep up; decrease for faster sim playback.
TRAJECTORY_TIME_SCALE = 1.0
# Minimum sleep between waypoints (seconds)
MIN_WAYPOINT_SLEEP = 0.05


class MoveItPlanner:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("right_arm")

        # Tune planning behaviour
        self.group.set_planning_time(5.0)
        self.group.set_num_planning_attempts(10)
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)

        # Publisher that simulation_server subscribes to for joint targets
        self.joint_pub = rospy.Publisher(
            "relaxed_ik/joint_angle_solutions", JointState, queue_size=10
        )

        rospy.loginfo("MoveIt planner ready. Planning frame: %s, EE link: %s",
                      self.group.get_planning_frame(),
                      self.group.get_end_effector_link())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def move_to_pose(self, position, orientation):
        """Plan and execute a move to a Cartesian pose.

        Args:
            position: (x, y, z)
            orientation: (qx, qy, qz, qw)

        Returns:
            (success: bool, message: str)
        """
        target = Pose()
        target.position.x, target.position.y, target.position.z = position
        target.orientation.x, target.orientation.y = orientation[0], orientation[1]
        target.orientation.z, target.orientation.w = orientation[2], orientation[3]

        self.group.set_pose_target(target)
        success, plan, planning_time, error_code = self.group.plan()
        self.group.clear_pose_targets()

        if not success or not plan.joint_trajectory.points:
            msg = f"Planning failed (error_code={error_code.val}, time={planning_time:.2f}s)"
            rospy.logwarn(msg)
            return False, msg

        rospy.loginfo("Plan found in %.2fs with %d waypoints. Executing...",
                      planning_time, len(plan.joint_trajectory.points))
        self._execute_on_mujoco(plan.joint_trajectory)
        return True, "OK"

    def move_to_joint_angles(self, joint_angles):
        """Plan and execute a move to explicit joint angles.

        Args:
            joint_angles: list of 7 joint angles [right_j0 .. right_j6]

        Returns:
            (success: bool, message: str)
        """
        self.group.set_joint_value_target(joint_angles)
        success, plan, planning_time, error_code = self.group.plan()
        self.group.clear_pose_targets()

        if not success or not plan.joint_trajectory.points:
            msg = f"Planning failed (error_code={error_code.val})"
            rospy.logwarn(msg)
            return False, msg

        self._execute_on_mujoco(plan.joint_trajectory)
        return True, "OK"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_on_mujoco(self, joint_trajectory):
        """Stream trajectory waypoints to the MuJoCo simulation server.

        Publishes each waypoint as a JointState to relaxed_ik/joint_angle_solutions,
        sleeping between waypoints to match the planned timing.
        """
        points = joint_trajectory.points
        joint_names = joint_trajectory.joint_names

        for i, point in enumerate(points):
            js = JointState()
            js.header.stamp = rospy.Time.now()
            js.name = joint_names
            js.position = list(point.positions)
            self.joint_pub.publish(js)

            # Sleep until the next waypoint's scheduled time
            if i < len(points) - 1:
                current_t = point.time_from_start.to_sec()
                next_t = points[i + 1].time_from_start.to_sec()
                sleep_dur = max((next_t - current_t) * TRAJECTORY_TIME_SCALE,
                                MIN_WAYPOINT_SLEEP)
                rospy.sleep(sleep_dur)

        rospy.loginfo("Trajectory execution complete (%d waypoints)", len(points))


# ------------------------------------------------------------------
# ROS service handlers
# ------------------------------------------------------------------

def handle_move_to_cartesian_pose(req):
    position = (req.x, req.y, req.z)
    orientation = (req.qx, req.qy, req.qz, req.qw)
    success, message = planner.move_to_pose(position, orientation)
    return MoveToCartesianPoseResponse(success=success, message=message)


def handle_operate_gripper(req):
    # In simulation the gripper is driven via the visualizer directly.
    # This is a no-op stub; wire to your gripper controller as needed.
    rospy.loginfo("Gripper %s requested (sim stub)", "open" if req.open else "close")
    return OperateGripperResponse(message="ok")


if __name__ == '__main__':
    rospy.init_node('move_to_cartesian_pose')
    planner = MoveItPlanner()

    rospy.Service('move_to_cartesian_pose', MoveToCartesianPose,
                  handle_move_to_cartesian_pose)
    rospy.Service('operate_gripper', OperateGripper,
                  handle_operate_gripper)

    rospy.loginfo("move_to_cartesian_pose service ready")
    rospy.spin()
