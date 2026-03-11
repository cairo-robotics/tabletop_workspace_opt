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
        self.group.set_planning_time(15.0)
        self.group.set_num_planning_attempts(50)
        self.group.set_max_velocity_scaling_factor(0.3)
        self.group.set_max_acceleration_scaling_factor(0.3)
        # Allow ~15° orientation tolerance for more planning flexibility
        self.group.set_goal_orientation_tolerance(0.26)  # radians (~15°)
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

    def _compute_multiple_ik(self, position, orientation, max_solutions=5,
                             num_seeds=30):
        """Compute multiple distinct IK solutions for a target pose.

        Returns a list of joint-value lists, each representing a different
        arm configuration that achieves the target pose.
        """
        try:
            ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
            ik_srv.wait_for_service(timeout=2.0)
        except (rospy.ROSException, rospy.ServiceException):
            return []

        joint_names = self.group.get_active_joints()
        joint_limits = []
        for name in joint_names:
            joint = self.robot.get_joint(name)
            joint_limits.append(joint.bounds())

        qx, qy, qz, qw = orientation
        solutions = []

        for i in range(num_seeds):
            req = GetPositionIKRequest()
            req.ik_request.group_name = "right_arm"
            req.ik_request.pose_stamped.header.frame_id = \
                self.group.get_planning_frame()
            req.ik_request.pose_stamped.pose.position.x = position[0]
            req.ik_request.pose_stamped.pose.position.y = position[1]
            req.ik_request.pose_stamped.pose.position.z = position[2]
            req.ik_request.pose_stamped.pose.orientation.x = qx
            req.ik_request.pose_stamped.pose.orientation.y = qy
            req.ik_request.pose_stamped.pose.orientation.z = qz
            req.ik_request.pose_stamped.pose.orientation.w = qw
            req.ik_request.timeout = rospy.Duration(1.0)

            seed = RobotState()
            seed.joint_state.name = joint_names
            if i == 0:
                seed.joint_state.position = \
                    self.group.get_current_joint_values()
            else:
                seed.joint_state.position = [
                    np.random.uniform(lo, hi) for (lo, hi) in joint_limits
                ]
            req.ik_request.robot_state = seed

            try:
                resp = ik_srv(req)
                if resp.error_code.val == 1:  # SUCCESS
                    result = []
                    for name in joint_names:
                        idx = list(resp.solution.joint_state.name).index(name)
                        result.append(resp.solution.joint_state.position[idx])

                    # Check if this is sufficiently different from existing
                    is_distinct = True
                    for existing in solutions:
                        diff = np.linalg.norm(
                            np.array(result) - np.array(existing))
                        if diff < 0.3:  # radians
                            is_distinct = False
                            break
                    if is_distinct:
                        solutions.append(result)
                        rospy.loginfo("IK solution %d found (seed %d/%d)",
                                      len(solutions), i + 1, num_seeds)
                        if len(solutions) >= max_solutions:
                            break
            except rospy.ServiceException:
                continue

        return solutions

    def move_cartesian_line(self, position, orientation):
        """Move the EE in a straight Cartesian line to the target pose.

        Uses MoveIt's compute_cartesian_path() to produce a trajectory where
        the EE follows a straight line from current pose to target. This avoids
        the lateral swings that joint-space planning produces.

        Args:
            position: (x, y, z)
            orientation: (qx, qy, qz, qw)

        Returns:
            (success: bool, message: str)
        """
        qx, qy, qz, qw = orientation
        use_position_only = all(abs(v) < 0.01 for v in (qx, qy, qz, qw))

        target = Pose()
        target.position.x, target.position.y, target.position.z = position

        if use_position_only:
            # Use current orientation
            current_pose = self.group.get_current_pose().pose
            target.orientation = current_pose.orientation
        else:
            target.orientation.x, target.orientation.y = qx, qy
            target.orientation.z, target.orientation.w = qz, qw

        waypoints = [target]

        # compute_cartesian_path(waypoints, eef_step, avoid_collisions)
        # eef_step: max step between interpolated waypoints (1mm for precision)
        # avoid_collisions: False since we already excluded relevant objects
        plan, fraction = self.group.compute_cartesian_path(
            waypoints, 0.001, False
        )

        if fraction < 0.95:
            msg = (f"Cartesian path only {fraction*100:.0f}% achieved "
                   f"(need >=95%)")
            rospy.logwarn(msg)
            return False, msg

        if not plan.joint_trajectory.points:
            return False, "Cartesian path produced empty trajectory"

        rospy.loginfo("Cartesian path: %.0f%% achieved, %d waypoints. Executing...",
                      fraction * 100, len(plan.joint_trajectory.points))
        # Use slower execution for Cartesian paths (2x normal time)
        self._execute_on_mujoco(plan.joint_trajectory, time_scale=2.0)
        return True, "OK"

    def move_to_pose(self, position, orientation):
        """Plan and execute a move to a Cartesian pose.

        If orientation is near-zero (all components < 0.01), plans to the
        position only and lets MoveIt choose a feasible orientation.

        For oriented poses, the primary approach is:
          1. Compute IK with multiple random seeds to find joint-space goal
          2. Plan with joint-space RRT (collision-aware)
        This is more reliable than MoveIt's built-in pose target (KDL IK)
        and produces collision-free trajectories via RRT.

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
            success, plan, planning_time, error_code = self.group.plan()
            self.group.clear_pose_targets()
            if success and plan.joint_trajectory.points:
                rospy.loginfo("Position-only plan found in %.2fs (%d waypoints)",
                              planning_time, len(plan.joint_trajectory.points))
                self._execute_on_mujoco(plan.joint_trajectory)
                return True, "OK"
            msg = f"Position-only planning failed (error={error_code.val})"
            rospy.logwarn(msg)
            return False, msg

        # Primary: compute IK with random seeds, then joint-space RRT.
        # Try multiple IK solutions — different arm configurations may
        # have easier collision-free paths through the planning scene.
        rospy.loginfo("Computing IK for target [%.3f, %.3f, %.3f]...",
                      *position)
        ik_solutions = self._compute_multiple_ik(position, orientation,
                                                  max_solutions=5)
        for i, ik_joints in enumerate(ik_solutions):
            rospy.loginfo("Trying IK solution %d/%d...", i + 1,
                          len(ik_solutions))
            self.group.set_joint_value_target(ik_joints)
            success, plan, planning_time, error_code = self.group.plan()
            self.group.clear_pose_targets()
            if success and plan.joint_trajectory.points:
                rospy.loginfo("IK+RRT plan found in %.2fs (%d waypoints)",
                              planning_time, len(plan.joint_trajectory.points))
                self._execute_on_mujoco(plan.joint_trajectory)
                return True, "OK"
            rospy.logwarn("RRT failed for IK solution %d (error=%d)",
                          i + 1, error_code.val)

        # Fallback: MoveIt's built-in pose target (uses KDL IK internally)
        rospy.loginfo("Falling back to MoveIt pose target planning...")
        target = Pose()
        target.position.x, target.position.y, target.position.z = position
        target.orientation.x, target.orientation.y = qx, qy
        target.orientation.z, target.orientation.w = qz, qw
        self.group.set_pose_target(target)
        success, plan, planning_time, error_code = self.group.plan()
        self.group.clear_pose_targets()

        if success and plan.joint_trajectory.points:
            rospy.loginfo("MoveIt pose plan found in %.2fs (%d waypoints)",
                          planning_time, len(plan.joint_trajectory.points))
            self._execute_on_mujoco(plan.joint_trajectory)
            return True, "OK"

        msg = f"All planning attempts failed (last error={error_code.val})"
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

    def _execute_on_mujoco(self, joint_trajectory, time_scale=None):
        """Stream trajectory waypoints to the MuJoCo simulation server.

        Publishes each waypoint as a JointState to relaxed_ik/joint_angle_solutions,
        sleeping between waypoints to match the planned timing.
        """
        if time_scale is None:
            time_scale = TRAJECTORY_TIME_SCALE

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
                sleep_dur = max((next_t - current_t) * time_scale,
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
