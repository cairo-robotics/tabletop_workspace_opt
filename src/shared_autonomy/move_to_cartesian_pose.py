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
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tabletop_workspace_opt.srv import (
    MoveToCartesianPose, MoveToCartesianPoseResponse,
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
        self.group.set_planning_time(10.0)
        self.group.set_num_planning_attempts(25)
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)
        # Allow ~5° orientation tolerance for more IK solutions
        self.group.set_goal_orientation_tolerance(0.09)  # radians (~5°)
        self.group.set_goal_position_tolerance(0.005)  # 5mm

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

    def _compute_ik_with_seeds(self, position, orientation, num_seeds=20):
        """Try computing IK with multiple random seed states.

        The KDL solver often fails to find solutions for certain orientations
        from the default seed. This method tries random seeds to explore
        different solution branches.

        Returns joint values on success, None on failure.
        """
        try:
            ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
            ik_srv.wait_for_service(timeout=2.0)
        except (rospy.ROSException, rospy.ServiceException):
            return None

        joint_names = self.group.get_active_joints()
        joint_limits = []
        for name in joint_names:
            # Get joint limits from the robot model
            joint = self.robot.get_joint(name)
            bounds = joint.bounds()
            joint_limits.append(bounds)

        qx, qy, qz, qw = orientation

        for i in range(num_seeds):
            req = GetPositionIKRequest()
            req.ik_request.group_name = "right_arm"
            req.ik_request.pose_stamped.header.frame_id = self.group.get_planning_frame()
            req.ik_request.pose_stamped.pose.position.x = position[0]
            req.ik_request.pose_stamped.pose.position.y = position[1]
            req.ik_request.pose_stamped.pose.position.z = position[2]
            req.ik_request.pose_stamped.pose.orientation.x = qx
            req.ik_request.pose_stamped.pose.orientation.y = qy
            req.ik_request.pose_stamped.pose.orientation.z = qz
            req.ik_request.pose_stamped.pose.orientation.w = qw
            req.ik_request.timeout = rospy.Duration(1.0)

            # Use random seed state
            seed = RobotState()
            seed.joint_state.name = joint_names
            if i == 0:
                seed.joint_state.position = self.group.get_current_joint_values()
            else:
                seed.joint_state.position = [
                    np.random.uniform(lo, hi) for (lo, hi) in joint_limits
                ]
            req.ik_request.robot_state = seed

            try:
                resp = ik_srv(req)
                if resp.error_code.val == 1:  # SUCCESS
                    # Extract just the arm joint values
                    result = []
                    for name in joint_names:
                        idx = list(resp.solution.joint_state.name).index(name)
                        result.append(resp.solution.joint_state.position[idx])
                    rospy.loginfo("IK found on seed %d/%d", i + 1, num_seeds)
                    return result
            except rospy.ServiceException:
                continue

        return None

    def move_to_pose(self, position, orientation):
        """Plan and execute a move to a Cartesian pose.

        If orientation is near-zero (all components < 0.01), plans to the
        position only and lets MoveIt choose a feasible orientation.

        If the default planner fails (often due to KDL IK limitations),
        falls back to computing IK with random seeds and planning in
        joint space.

        Args:
            position: (x, y, z)
            orientation: (qx, qy, qz, qw)

        Returns:
            (success: bool, message: str)
        """
        qx, qy, qz, qw = orientation
        use_position_only = all(abs(v) < 0.01 for v in (qx, qy, qz, qw))

        if use_position_only:
            self.group.set_position_target(list(position))
        else:
            target = Pose()
            target.position.x, target.position.y, target.position.z = position
            target.orientation.x, target.orientation.y = qx, qy
            target.orientation.z, target.orientation.w = qz, qw
            self.group.set_pose_target(target)

        success, plan, planning_time, error_code = self.group.plan()
        self.group.clear_pose_targets()

        if success and plan.joint_trajectory.points:
            rospy.loginfo("Plan found in %.2fs with %d waypoints. Executing...",
                          planning_time, len(plan.joint_trajectory.points))
            self._execute_on_mujoco(plan.joint_trajectory)
            return True, "OK"

        # If pose planning failed quickly (IK issue), try random seed IK
        if not use_position_only and planning_time < 1.0:
            rospy.loginfo("Pose planning failed (IK), trying random seed IK...")
            ik_joints = self._compute_ik_with_seeds(position, orientation)
            if ik_joints is not None:
                # Plan in joint space (avoids the IK issue)
                self.group.set_joint_value_target(ik_joints)
                success, plan, planning_time, error_code = self.group.plan()
                self.group.clear_pose_targets()
                if success and plan.joint_trajectory.points:
                    rospy.loginfo("Joint-space plan found in %.2fs with %d waypoints.",
                                  planning_time, len(plan.joint_trajectory.points))
                    self._execute_on_mujoco(plan.joint_trajectory)
                    return True, "OK"

        msg = f"Planning failed (error_code={error_code.val}, time={planning_time:.2f}s)"
        rospy.logwarn(msg)
        return False, msg

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


if __name__ == '__main__':
    rospy.init_node('move_to_cartesian_pose')
    planner = MoveItPlanner()

    rospy.Service('move_to_cartesian_pose', MoveToCartesianPose,
                  handle_move_to_cartesian_pose)

    rospy.loginfo("move_to_cartesian_pose service ready")
    rospy.spin()
