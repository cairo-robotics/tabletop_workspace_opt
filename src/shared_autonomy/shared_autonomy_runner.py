#!/usr/bin/env python3
"""Shared autonomy system for the breakfast task.

Combines: simulated joystick, intent inference, task state machine,
and auto-completion into a single node.

Usage:
    python3 shared_autonomy_runner.py [config.yaml] [--noise 0.01] [--threshold 0.8]
"""
import sys
import os
import glob
import argparse

# Ensure ROS Python paths are available
_ws = os.path.expanduser("~/sawyer_ws/devel_isolated")
for _d in glob.glob(os.path.join(_ws, "*/lib/python3/dist-packages")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
_ROS_PYTHON_PATH = '/opt/ros/noetic/lib/python3/dist-packages'
if _ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROS_PYTHON_PATH)
else:
    sys.path.insert(0, sys.path.pop(sys.path.index(_ROS_PYTHON_PATH)))

import time
import yaml
import numpy as np
import rospy
import mujoco
from sensor_msgs.msg import JointState
from intera_core_msgs.msg import EndpointState
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import Trigger
from tabletop_workspace_opt.srv import OperateGripper, TeleportObject
from geometry_msgs.msg import Point
from tabletop_workspace_opt.msg import ValidGoal, ValidGoals, AutoCompleteTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_Z_OFFSET = 0.92


def get_object_position(det_msg, det_id):
    """Extract object position from a Detection2DArray message."""
    for det in det_msg.detections:
        for res in det.results:
            if res.id == det_id:
                p = res.pose.pose.position
                return np.array([p.x, p.y, p.z])
    return None


def compute_goal_position(goal_spec, det_msg, object_det_ids):
    """Compute the world-frame target position for a goal.

    For pick: returns the object's current position.
    For place/pour: returns reference position + offset.
    """
    action = goal_spec["action"]

    if action == "pick":
        obj = goal_spec["object"]
        det_id = object_det_ids.get(obj)
        if det_id is None:
            return None
        return get_object_position(det_msg, det_id)

    elif action in ("place", "pour"):
        dest = goal_spec["destination"]
        # Support absolute positions (no reference object)
        if "absolute" in dest:
            ab = dest["absolute"]
            return np.array([ab["x"], ab["y"], ab["z"]])
        ref = dest.get("reference")
        if ref is None:
            return None
        det_id = object_det_ids.get(ref)
        if det_id is None:
            return None
        ref_pos = get_object_position(det_msg, det_id)
        if ref_pos is None:
            return None
        off = dest.get("offset", {})
        return ref_pos + np.array([
            off.get("x", 0), off.get("y", 0), off.get("z", 0)])

    return None


# ---------------------------------------------------------------------------
# Intent Inference
# ---------------------------------------------------------------------------

class IntentInferenceEngine:
    """Bayesian intent inference using path efficiency."""

    def __init__(self, beta=5.0, threshold=0.8):
        self.beta = beta
        self.threshold = threshold
        self.path_length = 0.0
        self.start_pos = None
        self.prev_pos = None
        self.distribution = {}

    def reset(self, ee_pos):
        self.start_pos = ee_pos.copy()
        self.prev_pos = ee_pos.copy()
        self.path_length = 0.0
        self.distribution = {}

    def update(self, ee_pos, goal_positions):
        """Update the probability distribution over goals.

        Uses path efficiency + proximity bonus. If the EE is very close
        to a goal (< proximity_radius), that goal gets a strong boost.

        Returns (distribution_dict, top_goal_id, top_prob).
        """
        if self.prev_pos is not None:
            step = np.linalg.norm(ee_pos - self.prev_pos)
            self.path_length += step
        self.prev_pos = ee_pos.copy()

        if self.start_pos is None or len(goal_positions) == 0:
            return {}, None, 0.0

        proximity_radius = 0.15  # 15cm — strong proximity boost

        scores = {}
        for gid, gpos in goal_positions.items():
            d_sg = np.linalg.norm(gpos - self.start_pos)
            d_qg = np.linalg.norm(gpos - ee_pos)
            if d_sg < 0.01:
                d_sg = 0.01

            # Path efficiency score
            cost = (self.path_length + d_qg) / d_sg
            score = -self.beta * cost

            # Proximity bonus: strongly favor goals the EE is very close to
            if d_qg < proximity_radius:
                score += self.beta * (1.0 - d_qg / proximity_radius)

            scores[gid] = score

        # Softmax
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        self.distribution = {k: v / total for k, v in exp_scores.items()}

        top_goal = max(self.distribution, key=self.distribution.get)
        top_prob = self.distribution[top_goal]
        return self.distribution, top_goal, top_prob


# ---------------------------------------------------------------------------
# Simulated Joystick
# ---------------------------------------------------------------------------

class SimulatedJoystick:
    """Generates noisy velocity commands toward a target goal."""

    def __init__(self, noise_sigma=0.01, max_speed=0.15, gain=2.0):
        self.noise_sigma = noise_sigma
        self.max_speed = max_speed
        self.gain = gain

    def compute_velocity(self, ee_pos, target_pos):
        """Compute Cartesian velocity toward target with Gaussian noise.

        Returns 3D velocity vector.
        """
        direction = target_pos - ee_pos
        dist = np.linalg.norm(direction)

        if dist < 0.005:
            # At target — just noise
            return np.random.normal(0, self.noise_sigma, 3)

        # Scale to max_speed, with proportional slowdown near target
        speed = min(self.max_speed, self.gain * dist)
        vel_ideal = (direction / dist) * speed

        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_sigma, 3)
        return vel_ideal + noise


# ---------------------------------------------------------------------------
# Task State Machine
# ---------------------------------------------------------------------------

class TaskStateMachine:
    """Manages task state transitions from the YAML config."""

    def __init__(self, config, object_det_ids):
        self.states = config["states"]
        self.current_state = "initial"
        self.object_det_ids = object_det_ids
        self.holding = None  # object name being held

    def get_valid_goals(self, det_msg):
        """Return list of (goal_spec, target_position) for current state."""
        state = self.states.get(self.current_state, {})
        goals = state.get("valid_goals", [])
        result = []
        for g in goals:
            pos = compute_goal_position(g, det_msg, self.object_det_ids)
            if pos is not None:
                result.append((g, pos))
        return result

    def transition(self, goal_spec):
        """Execute state transition for a completed goal."""
        next_state = goal_spec.get("next_state", self.current_state)
        action = goal_spec["action"]
        if action == "pick":
            self.holding = goal_spec["object"]
        elif action == "place":
            self.holding = None
        elif action == "pour":
            pass  # still holding
        rospy.loginfo("  State: %s -> %s", self.current_state, next_state)
        self.current_state = next_state

    def is_done(self):
        state = self.states.get(self.current_state, {})
        return len(state.get("valid_goals", [])) == 0


# ---------------------------------------------------------------------------
# Auto-Completion
# ---------------------------------------------------------------------------

class AutoCompleter:
    """Completes tasks by teleporting objects and snapping EE to home.

    Since we're evaluating goal inference (not task execution), we just
    update the scene state so inference has correct context for the
    next goal.
    """

    HOME_JOINTS = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]

    def __init__(self):
        rospy.wait_for_service("/sim/teleport_object", timeout=10)
        self.teleport_srv = rospy.ServiceProxy("/sim/teleport_object", TeleportObject)
        self.joint_pub = rospy.Publisher(
            "relaxed_ik/joint_angle_solutions", JointState, queue_size=10)

    def _teleport(self, mujoco_name, pos):
        """Teleport an object to a position."""
        try:
            self.teleport_srv(
                object_name=mujoco_name,
                x=pos[0], y=pos[1], z=pos[2],
                qw=1.0, qx=0.0, qy=0.0, qz=0.0)
        except Exception as e:
            rospy.logwarn("  Teleport '%s' failed: %s", mujoco_name, e)

    def _go_home(self):
        """Snap EE to home position."""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = self.HOME_JOINTS
        self.joint_pub.publish(msg)
        rospy.sleep(1.0)

    def complete(self, action, object_name, dest_pos, mujoco_name):
        """Complete any action: teleport object if needed, snap EE home."""
        if action == "pick":
            rospy.loginfo("  Auto-complete PICK: %s", object_name)
        elif action == "place":
            rospy.loginfo("  Auto-complete PLACE: %s -> [%.3f, %.3f, %.3f]",
                          object_name, *dest_pos)
            self._teleport(mujoco_name, dest_pos)
        elif action == "pour":
            rospy.loginfo("  Auto-complete POUR: %s", object_name)
            self._teleport(mujoco_name, dest_pos)

        self._go_home()


# ---------------------------------------------------------------------------
# Velocity Controller (Jacobian-based)
# ---------------------------------------------------------------------------

class JacobianVelocityController:
    """Converts Cartesian velocities to joint velocities using MuJoCo Jacobian.

    Handles joint limits by repelling joints away from their limits and
    clamping the output to stay within bounds.
    """

    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = self.model.site("end_effector").id

        # Find arm joint DOF indices and limits
        self.arm_dofs = []
        self.arm_qpos_addrs = []
        self.joint_lower = []
        self.joint_upper = []
        for i in range(7):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_j{i}")
            self.arm_dofs.append(self.model.jnt_dofadr[jid])
            self.arm_qpos_addrs.append(self.model.jnt_qposadr[jid])
            self.joint_lower.append(self.model.jnt_range[jid, 0])
            self.joint_upper.append(self.model.jnt_range[jid, 1])
        self.joint_lower = np.array(self.joint_lower)
        self.joint_upper = np.array(self.joint_upper)

    def compute_new_joints(self, joint_positions, cart_vel, dt):
        """Convert Cartesian velocity to new joint positions.

        Includes joint limit avoidance and clamping.

        Args:
            joint_positions: current 7 arm joint positions
            cart_vel: desired [vx, vy, vz] Cartesian velocity
            dt: timestep

        Returns:
            7-element array of new joint positions (clamped to limits)
        """
        # Set the model to the current joint state
        for i, addr in enumerate(self.arm_qpos_addrs):
            self.data.qpos[addr] = joint_positions[i]
        mujoco.mj_forward(self.model, self.data)

        # Compute position Jacobian (3x nv)
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        # Extract arm columns
        J = jacp[:, self.arm_dofs]  # 3x7

        # Damped pseudo-inverse for stability near singularities
        damping = 0.05
        JJT = J @ J.T + damping * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)  # 7x3

        joint_vel = J_pinv @ cart_vel

        # Joint limit avoidance: add repulsive velocity away from limits
        margin = 0.1  # radians — start repelling within this margin
        repulsion_gain = 2.0
        for i in range(7):
            dist_to_lower = joint_positions[i] - self.joint_lower[i]
            dist_to_upper = self.joint_upper[i] - joint_positions[i]
            if dist_to_lower < margin:
                joint_vel[i] += repulsion_gain * (margin - dist_to_lower)
            if dist_to_upper < margin:
                joint_vel[i] -= repulsion_gain * (margin - dist_to_upper)

        # Integrate
        new_joints = joint_positions + joint_vel * dt

        # Clamp to joint limits (with small margin to avoid hitting hard stops)
        safety = 0.02
        new_joints = np.clip(new_joints,
                             self.joint_lower + safety,
                             self.joint_upper - safety)

        return new_joints


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shared autonomy runner")
    parser.add_argument("task_config", nargs="?",
                        default="config/tasks/full_breakfast_sa.yaml",
                        help="Path to SA task YAML")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Gaussian noise sigma (m/s)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Intent confidence threshold")
    parser.add_argument("--beta", type=float, default=5.0,
                        help="Rationality coefficient for inference")
    parser.add_argument("--max-speed", type=float, default=0.15,
                        help="Max EE speed (m/s)")
    parser.add_argument("--control-rate", type=float, default=20.0,
                        help="Control loop rate (Hz)")
    parser.add_argument("--user-intent", type=str, default=None,
                        help="Force simulated user to always pick this goal ID")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name (overrides task config)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("shared_autonomy_runner", anonymous=True)

    # Resolve config path
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if not os.path.isabs(args.task_config):
        args.task_config = os.path.join(pkg_root, args.task_config)

    with open(args.task_config) as f:
        task_config = yaml.safe_load(f)["task"]

    # Load scene config: CLI --scene overrides task config
    scene_name = args.scene if args.scene else task_config.get("scene", "scene_breakfast")
    scene_path = os.path.join(pkg_root, "config", "scenes", f"{scene_name}.yaml")
    with open(scene_path) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    object_det_ids = {}
    object_mujoco_names = {}
    for i, obj in enumerate(scene_cfg["objects"]):
        object_det_ids[obj["short_name"]] = i
        object_mujoco_names[obj["short_name"]] = obj["mujoco_name"]

    # MuJoCo model path for Jacobian computation
    model_path = os.path.join(
        pkg_root, "src", "assets", f"{scene_name}.xml")

    print(f"\n{'=' * 60}")
    print(f"  SHARED AUTONOMY: {task_config['name']}")
    print(f"  Noise sigma: {args.noise} m/s")
    print(f"  Threshold: {args.threshold}")
    print(f"  Beta: {args.beta}")
    print(f"  Max speed: {args.max_speed} m/s")
    print(f"  Control rate: {args.control_rate} Hz")
    print(f"{'=' * 60}\n")

    # Initialize components
    joystick = SimulatedJoystick(
        noise_sigma=args.noise, max_speed=args.max_speed)
    inference = IntentInferenceEngine(
        beta=args.beta, threshold=args.threshold)
    state_machine = TaskStateMachine(task_config, object_det_ids)
    auto_completer = AutoCompleter()
    jac_controller = JacobianVelocityController(model_path)

    # Publishers
    goals_pub = rospy.Publisher(
        "/shared_autonomy/valid_goals", ValidGoals, queue_size=1)
    trigger_pub = rospy.Publisher(
        "/shared_autonomy/auto_complete_trigger", AutoCompleteTrigger, queue_size=1)

    # Wait for sim
    rospy.loginfo("Waiting for simulation...")
    rospy.wait_for_service("/operate_gripper", timeout=30)

    # Reset sim
    reset_srv = rospy.ServiceProxy("/reset_sim", Trigger)
    reset_srv()
    rospy.sleep(3.0)

    # Get initial EE pose
    ee_msg = rospy.wait_for_message(
        "/mujoco_sim/endpoint_state", EndpointState, timeout=5)
    ee_pos = np.array([ee_msg.pose.position.x,
                       ee_msg.pose.position.y,
                       ee_msg.pose.position.z])

    rate = rospy.Rate(args.control_rate)
    dt = 1.0 / args.control_rate
    user_goal_id = args.user_intent  # override or None

    steps_completed = 0
    max_steps = 50  # safety limit

    rospy.loginfo("Starting shared autonomy loop...")

    while not rospy.is_shutdown() and not state_machine.is_done():
        if steps_completed >= max_steps:
            rospy.logwarn("Max steps reached, stopping.")
            break

        # Get current detections
        try:
            det_msg = rospy.wait_for_message(
                "/mujoco_sim/detections", Detection2DArray, timeout=5)
        except rospy.ROSException:
            rospy.logwarn("No detections received")
            continue

        # Get valid goals for current state
        valid_goals = state_machine.get_valid_goals(det_msg)
        if not valid_goals:
            rospy.loginfo("No valid goals in state '%s' — done.",
                          state_machine.current_state)
            break

        # Publish valid goals
        vg_msg = ValidGoals()
        vg_msg.header.stamp = rospy.Time.now()
        vg_msg.current_state = state_machine.current_state
        goal_positions = {}
        for g, pos in valid_goals:
            vg = ValidGoal()
            vg.goal_id = g["id"]
            vg.action_type = g["action"]
            vg.target_position = Point(x=pos[0], y=pos[1], z=pos[2])
            vg.object_name = g.get("object", "")
            vg_msg.goals.append(vg)
            goal_positions[g["id"]] = pos
        goals_pub.publish(vg_msg)

        # Choose which goal the simulated user is going for
        if user_goal_id and user_goal_id in goal_positions:
            target_id = user_goal_id
        else:
            # Default: pick the first valid goal
            target_id = valid_goals[0][0]["id"]
        target_pos = goal_positions[target_id]

        rospy.loginfo("=" * 50)
        rospy.loginfo("State: %s | User intent: %s | Goals: %s",
                      state_machine.current_state, target_id,
                      [g["id"] for g, _ in valid_goals])

        # Reset inference for this goal
        inference.reset(ee_pos)
        goal_start_time = time.time()

        # Control loop: move toward goal until intent is inferred
        goal_reached = False
        loop_count = 0
        max_loops = int(30.0 * args.control_rate)  # 30 seconds max

        while not rospy.is_shutdown() and not goal_reached:
            loop_count += 1
            if loop_count > max_loops:
                rospy.logwarn("  Timeout reaching goal '%s'", target_id)
                break

            # Get current state
            try:
                ee_msg = rospy.wait_for_message(
                    "/mujoco_sim/endpoint_state", EndpointState, timeout=2)
                js_msg = rospy.wait_for_message(
                    "/joint_states", JointState, timeout=2)
            except rospy.ROSException:
                continue

            ee_pos = np.array([ee_msg.pose.position.x,
                               ee_msg.pose.position.y,
                               ee_msg.pose.position.z])

            # Extract arm joint positions (skip head_pan at index 0)
            joint_names = list(js_msg.name)
            joint_positions = []
            for i in range(7):
                idx = joint_names.index(f"right_j{i}")
                joint_positions.append(js_msg.position[idx])
            joint_positions = np.array(joint_positions)

            # Simulated joystick: compute velocity toward target
            cart_vel = joystick.compute_velocity(ee_pos, target_pos)

            # Convert to new joint positions (with joint limit avoidance)
            new_joints = jac_controller.compute_new_joints(
                joint_positions, cart_vel, dt)

            # Publish joint targets
            js_out = JointState()
            js_out.header.stamp = rospy.Time.now()
            js_out.position = new_joints.tolist()
            auto_completer.joint_pub.publish(js_out)

            # Update intent inference
            dist, top_goal, top_prob = inference.update(ee_pos, goal_positions)

            # Log periodically
            if loop_count % int(args.control_rate) == 0:
                dist_to_target = np.linalg.norm(target_pos - ee_pos)
                dist_str = " ".join(
                    f"{k}={v:.2f}" for k, v in sorted(dist.items()))
                rospy.loginfo("  d=%.0fmm | %s", dist_to_target * 1000,
                              dist_str)

            # Check if intent is confident enough
            if top_prob >= args.threshold and top_goal == target_id:
                inference_time = time.time() - goal_start_time
                rospy.loginfo("  INTENT INFERRED: %s (p=%.3f, t=%.1fs)",
                              top_goal, top_prob, inference_time)

                # Publish trigger
                trig = AutoCompleteTrigger()
                trig.goal_id = top_goal
                trig.confidence = top_prob
                trigger_pub.publish(trig)

                # Find the goal spec
                goal_spec = None
                for g, _ in valid_goals:
                    if g["id"] == top_goal:
                        goal_spec = g
                        break

                if goal_spec is None:
                    rospy.logerr("Goal spec not found for '%s'", top_goal)
                    break

                # Auto-complete the action
                action = goal_spec["action"]
                obj_name = goal_spec.get("object",
                                         state_machine.holding or "")
                mujoco_name = object_mujoco_names.get(obj_name, "")

                auto_completer.complete(
                    action, obj_name, target_pos, mujoco_name)

                # Transition state
                state_machine.transition(goal_spec)
                steps_completed += 1
                goal_reached = True

                # Reset user intent for next goal
                user_goal_id = None

            rate.sleep()

    # Final summary
    rospy.loginfo("")
    rospy.loginfo("=" * 50)
    rospy.loginfo("SHARED AUTONOMY COMPLETE")
    rospy.loginfo("  Steps completed: %d", steps_completed)
    rospy.loginfo("  Final state: %s", state_machine.current_state)
    rospy.loginfo("=" * 50)


if __name__ == "__main__":
    main()
