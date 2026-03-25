#!/usr/bin/env python3
"""Plan pregrasp/grasp motions in MoveIt from shared autonomy grasp selection."""

import copy
import sys
import threading
from typing import Optional, Tuple

import moveit_commander
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState
from tabletop_workspace_opt.msg import GraspCandidate
from tf2_geometry_msgs import do_transform_pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _as_numpy_xyz(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=np.float64)


def _normalize(vec: np.ndarray) -> Tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64), 0.0
    return vec / norm, norm


def _pose_distance(a: PoseStamped, b: PoseStamped) -> float:
    pa = _as_numpy_xyz(a.pose.position.x, a.pose.position.y, a.pose.position.z)
    pb = _as_numpy_xyz(b.pose.position.x, b.pose.position.y, b.pose.position.z)
    return float(np.linalg.norm(pa - pb))


class SharedAutonomyMoveItExecutor:
    def __init__(self):
        rospy.init_node("shared_autonomy_moveit_executor")
        moveit_commander.roscpp_initialize(sys.argv)

        default_ready_joint_names = [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]
        default_ready_joint_positions = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]

        self.group_name = rospy.get_param("~group_name", "right_arm")
        self.ee_link = rospy.get_param("~ee_link", "right_gripper_tip")
        self.base_frame = rospy.get_param("~base_frame", "world")
        self.selected_grasp_topic = rospy.get_param("~selected_grasp_topic", "/shared_autonomy/selected_grasp")
        self.adapted_pregrasp_pose_input_topic = rospy.get_param(
            "~adapted_pregrasp_pose_input_topic", ""
        )
        self.adapted_grasp_pose_input_topic = rospy.get_param(
            "~adapted_grasp_pose_input_topic", ""
        )
        self.pregrasp_pose_topic = rospy.get_param("~pregrasp_pose_topic", "/shared_autonomy/pregrasp_pose")
        self.grasp_pose_topic = rospy.get_param("~grasp_pose_topic", "/shared_autonomy/executor_grasp_pose")
        self.display_trajectory_topic = rospy.get_param("~display_trajectory_topic", "/move_group/display_planned_path")
        self.joint_command_topic = rospy.get_param("~joint_command_topic", "/relaxed_ik/joint_angle_solutions")
        self.sim_joint_trajectory_topic = rospy.get_param("~sim_joint_trajectory_topic", "/mujoco_sim/joint_trajectory")

        self.pregrasp_offset_m = float(rospy.get_param("~pregrasp_offset_m", 0.10))
        self.grasp_center_offset_m = float(rospy.get_param("~grasp_center_offset_m", 0.0))
        self.stable_selection_sec = float(rospy.get_param("~stable_selection_sec", 0.75))
        self.min_replan_interval_sec = float(rospy.get_param("~min_replan_interval_sec", 1.0))
        self.min_goal_position_delta_m = float(rospy.get_param("~min_goal_position_delta_m", 0.01))
        self.plan_only = bool(rospy.get_param("~plan_only", True))
        self.plan_grasp_from_pregrasp = bool(rospy.get_param("~plan_grasp_from_pregrasp", True))
        self.execute_pregrasp_only_on_grasp_failure = bool(
            rospy.get_param("~execute_pregrasp_only_on_grasp_failure", True)
        )
        self.use_ready_start_pose = bool(rospy.get_param("~use_ready_start_pose", True))
        self.allow_approximate_ik = bool(rospy.get_param("~allow_approximate_ik", True))
        self.allow_service_ik_fallback = bool(rospy.get_param("~allow_service_ik_fallback", True))
        self.ik_timeout_sec = float(rospy.get_param("~ik_timeout_sec", 0.15))
        self.publish_display_trajectory = bool(rospy.get_param("~publish_display_trajectory", True))
        self.execute_in_sim = bool(rospy.get_param("~execute_in_sim", True))
        self.replay_joint_states = bool(rospy.get_param("~replay_joint_states", True))
        self.robot_model_id = rospy.get_param("~robot_model_id", "sawyer")
        self.ready_transition_duration_sec = float(rospy.get_param("~ready_transition_duration_sec", 2.0))
        self.ready_transition_steps = int(rospy.get_param("~ready_transition_steps", 25))
        self.ready_joint_names = list(rospy.get_param("~ready_joint_names", default_ready_joint_names))
        self.ready_joint_positions = [
            float(value) for value in rospy.get_param("~ready_joint_positions", default_ready_joint_positions)
        ]

        if len(self.ready_joint_names) != len(self.ready_joint_positions):
            raise ValueError(
                "ready_joint_names and ready_joint_positions must have the same length "
                f"(got {len(self.ready_joint_names)} names and {len(self.ready_joint_positions)} positions)."
            )
        self.ready_joint_map = {
            joint_name: joint_position
            for joint_name, joint_position in zip(self.ready_joint_names, self.ready_joint_positions)
        }

        planning_time = float(rospy.get_param("~planning_time", 5.0))
        planning_attempts = int(rospy.get_param("~num_planning_attempts", 5))
        goal_position_tolerance = float(rospy.get_param("~goal_position_tolerance", 0.01))
        goal_orientation_tolerance = float(rospy.get_param("~goal_orientation_tolerance", 0.08))
        velocity_scaling = float(rospy.get_param("~max_velocity_scaling_factor", 0.2))
        acceleration_scaling = float(rospy.get_param("~max_acceleration_scaling_factor", 0.2))

        self.robot = moveit_commander.RobotCommander()
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        self.planning_frame = self.group.get_planning_frame()
        self.group.set_pose_reference_frame(self.base_frame)
        self.group.set_end_effector_link(self.ee_link)
        self.group.set_planning_time(planning_time)
        self.group.set_num_planning_attempts(planning_attempts)
        self.group.set_goal_position_tolerance(goal_position_tolerance)
        self.group.set_goal_orientation_tolerance(goal_orientation_tolerance)
        self.group.set_max_velocity_scaling_factor(velocity_scaling)
        self.group.set_max_acceleration_scaling_factor(acceleration_scaling)

        self.pregrasp_pub = rospy.Publisher(self.pregrasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.grasp_pub = rospy.Publisher(self.grasp_pose_topic, PoseStamped, queue_size=1, latch=True)
        self.display_pub = rospy.Publisher(self.display_trajectory_topic, DisplayTrajectory, queue_size=1, latch=True)
        self.joint_command_pub = rospy.Publisher(self.joint_command_topic, JointState, queue_size=1)
        self.sim_trajectory_pub = rospy.Publisher(self.sim_joint_trajectory_topic, JointTrajectory, queue_size=1, latch=True)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.compute_ik_srv = None
        self.state_validity_srv = None
        if self.allow_service_ik_fallback:
            rospy.loginfo("Waiting for MoveIt IK/state validity services for executor fallback...")
            rospy.wait_for_service("/compute_ik")
            rospy.wait_for_service("/check_state_validity")
            self.compute_ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
            self.state_validity_srv = rospy.ServiceProxy("/check_state_validity", GetStateValidity)

        self.latest_selected_grasp: Optional[GraspCandidate] = None
        self.latest_selected_pose: Optional[PoseStamped] = None
        self.latest_adapted_pregrasp_pose: Optional[PoseStamped] = None
        self.latest_adapted_grasp_pose: Optional[PoseStamped] = None
        self.selection_changed_at: Optional[rospy.Time] = None
        self.last_planned_at: Optional[rospy.Time] = None
        self.last_planned_pose: Optional[PoseStamped] = None
        self.last_planned_grasp_id: Optional[str] = None
        self.is_executing = False
        self.execution_thread: Optional[threading.Thread] = None

        rospy.Subscriber(self.selected_grasp_topic, GraspCandidate, self.selected_grasp_cb, queue_size=1)
        if self.adapted_pregrasp_pose_input_topic:
            rospy.Subscriber(
                self.adapted_pregrasp_pose_input_topic,
                PoseStamped,
                self.adapted_pregrasp_pose_cb,
                queue_size=1,
            )
        if self.adapted_grasp_pose_input_topic:
            rospy.Subscriber(
                self.adapted_grasp_pose_input_topic,
                PoseStamped,
                self.adapted_grasp_pose_cb,
                queue_size=1,
            )
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_cb)

        rospy.loginfo(
            "Shared autonomy MoveIt executor ready. group=%s ee_link=%s selected_grasp=%s "
            "plan_only=%s execute_pregrasp_only_on_grasp_failure=%s use_ready_start_pose=%s allow_approximate_ik=%s allow_service_ik_fallback=%s planning_frame=%s "
            "adapted_pose_inputs=(%s, %s)",
            self.group_name,
            self.ee_link,
            self.selected_grasp_topic,
            self.plan_only,
            self.execute_pregrasp_only_on_grasp_failure,
            self.use_ready_start_pose,
            self.allow_approximate_ik,
            self.allow_service_ik_fallback,
            self.planning_frame,
            self.adapted_pregrasp_pose_input_topic or "disabled",
            self.adapted_grasp_pose_input_topic or "disabled",
        )

    def selected_grasp_cb(self, msg: GraspCandidate):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.base_frame
        pose.pose = copy.deepcopy(msg.pose)

        changed = False
        if self.latest_selected_grasp is None:
            changed = True
        elif msg.grasp_id != self.latest_selected_grasp.grasp_id:
            changed = True
        elif self.latest_selected_pose is not None and _pose_distance(pose, self.latest_selected_pose) > self.min_goal_position_delta_m:
            changed = True

        self.latest_selected_grasp = copy.deepcopy(msg)
        self.latest_selected_pose = pose
        if changed or self.selection_changed_at is None:
            self.selection_changed_at = rospy.Time.now()

    def adapted_pregrasp_pose_cb(self, msg: PoseStamped):
        self.latest_adapted_pregrasp_pose = copy.deepcopy(msg)

    def adapted_grasp_pose_cb(self, msg: PoseStamped):
        self.latest_adapted_grasp_pose = copy.deepcopy(msg)

    def timer_cb(self, _event):
        if self.latest_selected_grasp is None or self.latest_selected_pose is None or self.selection_changed_at is None:
            return
        if self.is_executing:
            return

        now = rospy.Time.now()
        if (now - self.selection_changed_at).to_sec() < self.stable_selection_sec:
            return

        if self.last_planned_at is not None and (now - self.last_planned_at).to_sec() < self.min_replan_interval_sec:
            return

        if (
            self.last_planned_grasp_id == self.latest_selected_grasp.grasp_id
            and self.last_planned_pose is not None
            and _pose_distance(self.last_planned_pose, self.latest_selected_pose) < self.min_goal_position_delta_m
        ):
            return

        self.plan_for_selected_grasp()

    def _build_target_poses(self, candidate: GraspCandidate) -> Tuple[PoseStamped, PoseStamped]:
        approach = _as_numpy_xyz(
            candidate.approach_direction.x,
            candidate.approach_direction.y,
            candidate.approach_direction.z,
        )
        approach_dir, approach_norm = _normalize(approach)
        if approach_norm < 1e-6:
            rospy.logwarn("Selected grasp has near-zero approach direction; using +Z fallback.")
            approach_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        grasp_pose = PoseStamped()
        grasp_pose.header.stamp = rospy.Time.now()
        grasp_pose.header.frame_id = self.base_frame
        grasp_pose.pose = copy.deepcopy(candidate.pose)
        grasp_pose.pose.position.x += float(self.grasp_center_offset_m * approach_dir[0])
        grasp_pose.pose.position.y += float(self.grasp_center_offset_m * approach_dir[1])
        grasp_pose.pose.position.z += float(self.grasp_center_offset_m * approach_dir[2])

        pregrasp_pose = PoseStamped()
        pregrasp_pose.header = grasp_pose.header
        pregrasp_pose.pose = copy.deepcopy(grasp_pose.pose)
        pregrasp_pose.pose.position.x += float(self.pregrasp_offset_m * approach_dir[0])
        pregrasp_pose.pose.position.y += float(self.pregrasp_offset_m * approach_dir[1])
        pregrasp_pose.pose.position.z += float(self.pregrasp_offset_m * approach_dir[2])

        return pregrasp_pose, grasp_pose

    def _get_planning_target_poses(self, candidate: GraspCandidate) -> Tuple[PoseStamped, PoseStamped]:
        if (
            self.latest_adapted_pregrasp_pose is not None
            and self.latest_adapted_grasp_pose is not None
            and self.selection_changed_at is not None
            and self.latest_adapted_pregrasp_pose.header.stamp >= self.selection_changed_at
            and self.latest_adapted_grasp_pose.header.stamp >= self.selection_changed_at
        ):
            return (
                copy.deepcopy(self.latest_adapted_pregrasp_pose),
                copy.deepcopy(self.latest_adapted_grasp_pose),
            )
        return self._build_target_poses(candidate)

    def _plan_to_pose(self, pose: PoseStamped, label: str, start_state: Optional[RobotState] = None):
        if start_state is None:
            self.group.set_start_state_to_current_state()
            start_state = self.robot.get_current_state()
        else:
            self.group.set_start_state(start_state)

        self.group.clear_pose_targets()
        self.group.set_pose_target(pose, end_effector_link=self.ee_link)
        plan_result = self.group.plan()
        self.group.clear_pose_targets()

        success = False
        plan = None
        if isinstance(plan_result, tuple):
            if len(plan_result) >= 2:
                success = bool(plan_result[0])
                plan = plan_result[1]
        else:
            plan = plan_result
            success = hasattr(plan, "joint_trajectory") and bool(plan.joint_trajectory.points)

        if plan is not None and hasattr(plan, "joint_trajectory") and not plan.joint_trajectory.points:
            success = False

        if not success and self.allow_approximate_ik:
            rospy.loginfo("MoveIt %s exact pose planning failed; retrying with approximate IK target.", label)
            self.group.set_start_state(start_state)
            self.group.clear_pose_targets()
            try:
                approximate_pose = self._transform_pose_to_frame(pose, self.planning_frame)
                previous_reference_frame = self.group.get_pose_reference_frame()
                self.group.set_pose_reference_frame(self.planning_frame)
                self.group.set_joint_value_target(approximate_pose.pose, self.ee_link, True)
                self.group.set_pose_reference_frame(previous_reference_frame)
                plan_result = self.group.plan()
            except Exception as exc:
                rospy.logwarn("MoveIt %s approximate IK target setup failed: %s", label, exc)
                plan_result = None
                self.group.set_pose_reference_frame(self.base_frame)

            success = False
            plan = None
            if isinstance(plan_result, tuple):
                if len(plan_result) >= 2:
                    success = bool(plan_result[0])
                    plan = plan_result[1]
            else:
                plan = plan_result
                success = hasattr(plan, "joint_trajectory") and bool(plan.joint_trajectory.points)

            if plan is not None and hasattr(plan, "joint_trajectory") and not plan.joint_trajectory.points:
                success = False

        if not success and self.allow_service_ik_fallback and self.compute_ik_srv is not None:
            rospy.loginfo("MoveIt %s pose planning failed; retrying with service IK joint target.", label)
            try:
                ik_solution = self._compute_collision_free_ik_solution(pose, start_state)
                if ik_solution is not None:
                    joint_goal = self._joint_goal_map_from_state(ik_solution)
                    self.group.set_start_state(start_state)
                    self.group.clear_pose_targets()
                    self.group.set_joint_value_target(joint_goal)
                    plan_result = self.group.plan()
                else:
                    plan_result = None
            except Exception as exc:
                rospy.logwarn("MoveIt %s service IK fallback failed: %s", label, exc)
                plan_result = None

            success = False
            plan = None
            if isinstance(plan_result, tuple):
                if len(plan_result) >= 2:
                    success = bool(plan_result[0])
                    plan = plan_result[1]
            else:
                plan = plan_result
                success = hasattr(plan, "joint_trajectory") and bool(plan.joint_trajectory.points)

            if plan is not None and hasattr(plan, "joint_trajectory") and not plan.joint_trajectory.points:
                success = False

        if success and self.publish_display_trajectory and plan is not None:
            display = DisplayTrajectory()
            display.model_id = self.robot_model_id
            display.trajectory_start = start_state
            display.trajectory.append(plan)
            self.display_pub.publish(display)

        point_count = 0
        if plan is not None and hasattr(plan, "joint_trajectory"):
            point_count = len(plan.joint_trajectory.points)

        rospy.loginfo(
            "MoveIt %s plan %s with %d trajectory points.",
            label,
            "succeeded" if success else "failed",
            point_count,
        )
        return success, plan, start_state

    def _transform_pose_to_frame(self, pose: PoseStamped, target_frame: str) -> PoseStamped:
        if pose.header.frame_id == target_frame:
            return copy.deepcopy(pose)

        source_stamp = pose.header.stamp if pose.header.stamp != rospy.Time() else rospy.Time(0)
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            pose.header.frame_id,
            source_stamp,
            rospy.Duration(0.5),
        )
        transformed_pose = do_transform_pose(pose, transform)
        transformed_pose.header.frame_id = target_frame
        return transformed_pose

    def _current_joint_position_map(self):
        current_state = self.robot.get_current_state()
        return {
            joint_name: joint_position
            for joint_name, joint_position in zip(current_state.joint_state.name, current_state.joint_state.position)
        }

    def _build_robot_state_from_joint_map(self, joint_map) -> RobotState:
        state = copy.deepcopy(self.robot.get_current_state())
        state.joint_state.name = list(state.joint_state.name)
        state.joint_state.position = list(state.joint_state.position)
        state.joint_state.velocity = list(state.joint_state.velocity)
        state.joint_state.effort = list(state.joint_state.effort)
        name_to_index = {name: idx for idx, name in enumerate(state.joint_state.name)}
        for joint_name, joint_position in joint_map.items():
            if joint_name in name_to_index:
                state.joint_state.position[name_to_index[joint_name]] = float(joint_position)
        state.joint_state.header.stamp = rospy.Time.now()
        return state

    def _joint_goal_map_from_state(self, state: RobotState):
        active_joints = set(self.group.get_active_joints())
        return {
            joint_name: float(joint_position)
            for joint_name, joint_position in zip(state.joint_state.name, state.joint_state.position)
            if joint_name in active_joints
        }

    def _compute_collision_free_ik_solution(self, pose: PoseStamped, seed_state: RobotState):
        request = GetPositionIKRequest()
        request.ik_request.group_name = self.group_name
        request.ik_request.robot_state = copy.deepcopy(seed_state)
        request.ik_request.avoid_collisions = True
        request.ik_request.ik_link_name = self.ee_link
        request.ik_request.pose_stamped = copy.deepcopy(self._transform_pose_to_frame(pose, self.planning_frame))
        request.ik_request.timeout = rospy.Duration.from_sec(self.ik_timeout_sec)

        response = self.compute_ik_srv(request)
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        validity_request = GetStateValidityRequest()
        validity_request.robot_state = response.solution
        validity_request.group_name = self.group_name
        validity_response = self.state_validity_srv(validity_request)
        if not validity_response.valid:
            return None
        return response.solution

    def _build_ready_start_state(self) -> Optional[RobotState]:
        if not self.use_ready_start_pose:
            return None
        return self._build_robot_state_from_joint_map(self.ready_joint_map)

    def _build_ready_transition_trajectory(self, joint_names) -> Optional[JointTrajectory]:
        if not self.use_ready_start_pose:
            return None

        current_joint_map = self._current_joint_position_map()
        if not current_joint_map:
            rospy.logwarn("Unable to read current robot state for ready-pose transition.")
            return None

        start_positions = []
        goal_positions = []
        for joint_name in joint_names:
            if joint_name not in current_joint_map:
                rospy.logwarn("Current robot state is missing joint '%s'; skipping ready transition.", joint_name)
                return None
            start_position = float(current_joint_map[joint_name])
            goal_position = float(self.ready_joint_map.get(joint_name, start_position))
            start_positions.append(start_position)
            goal_positions.append(goal_position)

        start_positions_np = np.array(start_positions, dtype=np.float64)
        goal_positions_np = np.array(goal_positions, dtype=np.float64)
        if np.linalg.norm(goal_positions_np - start_positions_np) < 1e-5:
            return None

        step_count = max(2, self.ready_transition_steps)
        duration_sec = max(0.2, self.ready_transition_duration_sec)

        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.joint_names = list(joint_names)

        start_point = JointTrajectoryPoint()
        start_point.positions = start_positions_np.tolist()
        start_point.time_from_start = rospy.Duration(0.0)
        trajectory.points.append(start_point)

        for step_index in range(1, step_count + 1):
            alpha = float(step_index) / float(step_count)
            positions = ((1.0 - alpha) * start_positions_np + alpha * goal_positions_np).tolist()
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = rospy.Duration.from_sec(alpha * duration_sec)
            trajectory.points.append(point)

        rospy.loginfo(
            "Prepared ready-pose transition with %d points over %.2fs.",
            len(trajectory.points),
            duration_sec,
        )
        return trajectory

    def _robot_state_from_plan_end(self, plan) -> Optional[RobotState]:
        if plan is None or not hasattr(plan, "joint_trajectory") or not plan.joint_trajectory.points:
            return None

        state = copy.deepcopy(self.robot.get_current_state())
        state.joint_state.name = list(plan.joint_trajectory.joint_names)
        state.joint_state.position = list(plan.joint_trajectory.points[-1].positions)
        if plan.joint_trajectory.points[-1].velocities:
            state.joint_state.velocity = list(plan.joint_trajectory.points[-1].velocities)
        else:
            state.joint_state.velocity = []
        state.joint_state.effort = []
        state.joint_state.header.stamp = rospy.Time.now()
        return state

    def _merge_joint_trajectories(self, pregrasp_plan, grasp_plan) -> Optional[JointTrajectory]:
        trajectories = []
        if pregrasp_plan is not None and hasattr(pregrasp_plan, "joint_trajectory"):
            trajectories.append(pregrasp_plan.joint_trajectory)
        if grasp_plan is not None and hasattr(grasp_plan, "joint_trajectory"):
            trajectories.append(grasp_plan.joint_trajectory)
        return self._merge_trajectory_sequence(trajectories)

    def _merge_trajectory_sequence(self, trajectories) -> Optional[JointTrajectory]:
        merged = None
        for trajectory in trajectories:
            if trajectory is None or not trajectory.points:
                continue

            if merged is None:
                merged = JointTrajectory()
                merged.header.stamp = rospy.Time.now()
                merged.joint_names = list(trajectory.joint_names)
                merged.points = [copy.deepcopy(point) for point in trajectory.points]
                continue

            if list(trajectory.joint_names) != merged.joint_names:
                rospy.logwarn("Cannot merge trajectories: joint names differ.")
                return merged if merged.points else None

            offset = merged.points[-1].time_from_start if merged.points else rospy.Duration(0.0)
            start_index = 1 if len(trajectory.points) > 1 else 0
            for point in trajectory.points[start_index:]:
                merged_point = copy.deepcopy(point)
                merged_point.time_from_start = offset + point.time_from_start
                merged.points.append(merged_point)

        return merged if merged and merged.points else None

    def _execute_joint_trajectory_thread(self, trajectory: JointTrajectory):
        try:
            start_time = rospy.Time.now()
            previous_time = 0.0
            for point in trajectory.points:
                target_time = point.time_from_start.to_sec()
                sleep_dt = max(0.0, target_time - previous_time)
                if sleep_dt > 0.0:
                    rospy.sleep(sleep_dt)
                previous_time = target_time

                joint_state = JointState()
                joint_state.header.stamp = start_time + point.time_from_start
                joint_state.name = list(trajectory.joint_names)
                joint_state.position = list(point.positions)
                joint_state.velocity = list(point.velocities) if point.velocities else []
                joint_state.effort = []
                self.joint_command_pub.publish(joint_state)
        finally:
            self.is_executing = False

    def _execute_joint_trajectory(self, trajectory: JointTrajectory):
        if self.execute_in_sim:
            self.sim_trajectory_pub.publish(trajectory)
            rospy.loginfo(
                "Published trajectory with %d points to %s.",
                len(trajectory.points),
                self.sim_joint_trajectory_topic,
            )

        if self.replay_joint_states:
            self.is_executing = True
            self.execution_thread = threading.Thread(
                target=self._execute_joint_trajectory_thread,
                args=(trajectory,),
                daemon=True,
            )
            self.execution_thread.start()

    def plan_for_selected_grasp(self):
        assert self.latest_selected_grasp is not None
        pregrasp_pose, grasp_pose = self._get_planning_target_poses(self.latest_selected_grasp)
        pregrasp_start_state = self._build_ready_start_state()

        self.pregrasp_pub.publish(pregrasp_pose)
        self.grasp_pub.publish(grasp_pose)

        rospy.loginfo(
            "Planning for selected grasp '%s': pregrasp=(%.3f, %.3f, %.3f) grasp=(%.3f, %.3f, %.3f)",
            self.latest_selected_grasp.grasp_id,
            pregrasp_pose.pose.position.x,
            pregrasp_pose.pose.position.y,
            pregrasp_pose.pose.position.z,
            grasp_pose.pose.position.x,
            grasp_pose.pose.position.y,
            grasp_pose.pose.position.z,
        )
        if pregrasp_start_state is not None:
            rospy.loginfo(
                "Planning grasp from configured ready pose instead of the current robot state."
            )

        pregrasp_success, pregrasp_plan, pregrasp_start = self._plan_to_pose(
            pregrasp_pose,
            "pregrasp",
            start_state=pregrasp_start_state,
        )

        if not pregrasp_success:
            self.last_planned_at = rospy.Time.now()
            return

        grasp_start = pregrasp_start
        if self.plan_grasp_from_pregrasp:
            planned_state = self._robot_state_from_plan_end(pregrasp_plan)
            if planned_state is not None:
                grasp_start = planned_state

        grasp_success, grasp_plan, _ = self._plan_to_pose(grasp_pose, "grasp", start_state=grasp_start)

        should_execute_pregrasp_only = (
            (not grasp_success)
            and self.execute_pregrasp_only_on_grasp_failure
            and pregrasp_plan is not None
            and hasattr(pregrasp_plan, "joint_trajectory")
            and bool(pregrasp_plan.joint_trajectory.points)
        )

        if should_execute_pregrasp_only:
            rospy.logwarn(
                "Grasp plan failed for '%s'; executing ready/pregrasp segment only.",
                self.latest_selected_grasp.grasp_id,
            )

        if (grasp_success or should_execute_pregrasp_only) and not self.plan_only:
            ready_transition = None
            if hasattr(pregrasp_plan, "joint_trajectory"):
                ready_transition = self._build_ready_transition_trajectory(pregrasp_plan.joint_trajectory.joint_names)

            merged_trajectory = self._merge_trajectory_sequence(
                [
                    ready_transition,
                    pregrasp_plan.joint_trajectory if hasattr(pregrasp_plan, "joint_trajectory") else None,
                    (
                        grasp_plan.joint_trajectory
                        if (grasp_success and hasattr(grasp_plan, "joint_trajectory"))
                        else None
                    ),
                ]
            )
            if merged_trajectory is None:
                rospy.logwarn("Failed to merge planned trajectories for execution.")
            else:
                rospy.loginfo(
                    "Executing %s trajectory through simulation/joint-state replay.",
                    "ready/pregrasp/grasp" if grasp_success else "ready/pregrasp",
                )
                self._execute_joint_trajectory(merged_trajectory)

        self.last_planned_at = rospy.Time.now()
        self.last_planned_pose = copy.deepcopy(self.latest_selected_pose)
        self.last_planned_grasp_id = self.latest_selected_grasp.grasp_id

    def shutdown(self):
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    node = SharedAutonomyMoveItExecutor()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()
