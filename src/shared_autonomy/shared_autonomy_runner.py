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

# Make the project's pure-Python envopt modules importable (src/envopt).
_PKG_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_SRC not in sys.path:
    sys.path.insert(0, _PKG_SRC)

import time
import yaml
import numpy as np
import rospy
import mujoco
from sensor_msgs.msg import JointState
from intera_core_msgs.msg import EndpointState
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import Trigger
from tabletop_workspace_opt.srv import (
    MoveToCartesianPose, OperateGripper, TeleportObject)
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
# SE(3) grasp-goal helper (M8)
# ---------------------------------------------------------------------------

SE3_INTENT_MODES = ("2d-center", "3d-center", "3d-grasp-pos", "se3-grasp")


def build_se3_goal(goal_spec, pos, object_yaws, grasp_library):
    """Wrap a goal in the SE(3) dict the se3_observers expect.

    For pick actions with a library entry, resolves the grasp pose in the
    world frame using the live object position and the scene's optimized
    yaw. The inference *target* is the pre-grasp standoff pose; the
    executable grasp tip is stored separately for the auto-completer.

    Returns a dict with keys:
        pos      — target position for inference (3,) np.ndarray
        R        — target rotation for inference (3,3) or None
        is_grasp — True when a library entry was resolved
        grasp_pos, grasp_quat       — world-frame grasp tip (is_grasp only)
        pregrasp_pos, pregrasp_quat — world-frame pre-grasp (is_grasp only)
    """
    action = goal_spec.get("action")
    if action == "pick" and grasp_library is not None:
        obj_name = goal_spec.get("object")
        entry = grasp_library.get(obj_name)
        if entry is not None:
            yaw = object_yaws.get(obj_name, 0.0)
            poses = entry.resolve(pos, yaw)
            return {
                "pos": poses["pregrasp_pos"],
                "R": poses["grasp_R"],
                "is_grasp": True,
                "grasp_pos": poses["grasp_pos"],
                "grasp_quat": poses["grasp_quat"],
                "pregrasp_pos": poses["pregrasp_pos"],
                "pregrasp_quat": poses["pregrasp_quat"],
            }
    return {"pos": np.asarray(pos, dtype=float), "R": None, "is_grasp": False}


# ---------------------------------------------------------------------------
# Intent Inference
# ---------------------------------------------------------------------------

class GaussianDirectionInference:
    """Bayesian intent inference using Gaussian direction model.

    At each timestep, compares the observed velocity direction against the
    expected direction toward each goal. Accumulates Gaussian log-likelihoods.
    This matches the model used by the intent separability optimizer.

    Log-score update:
        log S_t(g) -= 0.5 * (u_t - mu_g)^T * sigma_inv * (u_t - mu_g)

    where:
        u_t = observed velocity direction (unit vector + noise)
        mu_g = expected direction toward goal g (unit vector)
    """

    def __init__(self, sigma=0.1, threshold=0.8):
        self.sigma = sigma
        self.sigma_inv = np.eye(3) / (sigma ** 2)
        self.threshold = threshold
        self.log_scores = {}
        self.distribution = {}

    def reset(self, ee_pos):
        self.log_scores = {}
        self.distribution = {}

    def update(self, ee_pos, goal_positions, observed_velocity=None):
        """Update posterior using the observed velocity command.

        Args:
            ee_pos: current EE position (3,)
            goal_positions: {goal_id: position (3,)}
            observed_velocity: the noisy velocity command from the joystick (3,)

        Returns (distribution_dict, top_goal_id, top_prob).
        """
        if observed_velocity is None or len(goal_positions) == 0:
            return {}, None, 0.0

        # Normalize observed velocity to get direction
        speed = np.linalg.norm(observed_velocity)
        if speed < 1e-6:
            # No movement — skip update, return current distribution
            if self.distribution:
                top_goal = max(self.distribution, key=self.distribution.get)
                return self.distribution, top_goal, self.distribution[top_goal]
            return {}, None, 0.0
        u_t = observed_velocity / speed

        # Initialize log-scores on first call
        if not self.log_scores:
            for gid in goal_positions:
                self.log_scores[gid] = 0.0  # uniform prior (log 1/M cancels)

        # Update log-score for each goal
        for gid, gpos in goal_positions.items():
            direction = gpos - ee_pos
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                mu_g = np.zeros(3)
            else:
                mu_g = direction / dist  # expected unit direction toward goal g

            diff = u_t - mu_g
            self.log_scores[gid] -= 0.5 * diff @ self.sigma_inv @ diff

        # Softmax to get posterior
        scores = np.array(list(self.log_scores.values()))
        scores -= np.max(scores)  # numerical stability
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()

        self.distribution = {gid: float(p)
                            for gid, p in zip(self.log_scores.keys(), probs)}

        top_goal = max(self.distribution, key=self.distribution.get)
        top_prob = self.distribution[top_goal]
        return self.distribution, top_goal, top_prob


class PathEfficiencyInference:
    """Bayesian intent inference using path efficiency (Dragan-style).

    Computes cost as path-efficiency ratio and applies Boltzmann softmax.
    """

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

    def update(self, ee_pos, goal_positions, observed_velocity=None):
        """Update the probability distribution over goals.

        Args:
            ee_pos: current EE position (3,)
            goal_positions: {goal_id: position (3,)}
            observed_velocity: ignored (path efficiency uses positions only)

        Returns (distribution_dict, top_goal_id, top_prob).
        """
        if self.prev_pos is not None:
            step = np.linalg.norm(ee_pos - self.prev_pos)
            self.path_length += step
        self.prev_pos = ee_pos.copy()

        if self.start_pos is None or len(goal_positions) == 0:
            return {}, None, 0.0

        scores = {}
        for gid, gpos in goal_positions.items():
            d_sg = np.linalg.norm(gpos - self.start_pos)
            d_qg = np.linalg.norm(gpos - ee_pos)
            if d_sg < 0.01:
                d_sg = 0.01

            cost = (self.path_length + d_qg) / d_sg
            scores[gid] = -self.beta * cost

        # Softmax
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        self.distribution = {k: v / total for k, v in exp_scores.items()}

        top_goal = max(self.distribution, key=self.distribution.get)
        top_prob = self.distribution[top_goal]
        return self.distribution, top_goal, top_prob


def create_inference_engine(model, **kwargs):
    """Factory for inference engines.

    Args:
        model: "gaussian" or "path_efficiency"
        **kwargs: passed to the constructor
    """
    if model == "gaussian":
        return GaussianDirectionInference(
            sigma=kwargs.get("sigma", 0.1),
            threshold=kwargs.get("threshold", 0.8),
        )
    elif model == "path_efficiency":
        return PathEfficiencyInference(
            beta=kwargs.get("beta", 5.0),
            threshold=kwargs.get("threshold", 0.8),
        )
    else:
        raise ValueError(f"Unknown inference model: {model}")


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


class PickAndReturnStateMachine:
    """Dynamic state machine for pick-and-return tasks.

    Each object is picked once, used, then returned to its original
    position. ALL objects remain as candidates at every pick state
    (returned objects are available again). Task ends when every
    object has been picked exactly once.
    """

    def __init__(self, config, object_det_ids):
        self.pick_objects = list(config["pick_objects"])
        self.object_det_ids = object_det_ids
        self.picked = set()
        self.holding = None
        self.held_origin = None
        self.current_state = "pick"

    def get_valid_goals(self, det_msg):
        if self.current_state == "pick":
            goals = []
            for obj in self.pick_objects:
                det_id = self.object_det_ids.get(obj)
                if det_id is None:
                    continue
                pos = get_object_position(det_msg, det_id)
                if pos is not None:
                    goals.append(({"id": f"pick_{obj}",
                                   "action": "pick",
                                   "object": obj}, pos))
            return goals
        elif self.current_state == "return":
            if self.holding and self.held_origin is not None:
                return [({"id": f"return_{self.holding}",
                          "action": "place",
                          "object": self.holding},
                         self.held_origin)]
            return []
        return []

    def transition(self, goal_spec):
        action = goal_spec["action"]
        if action == "pick":
            self.holding = goal_spec["object"]
            self.current_state = "return"
            rospy.loginfo("  State: pick -> return (%s)", self.holding)
        elif action == "place":
            self.picked.add(self.holding)
            rospy.loginfo("  State: return -> pick (%s returned, %d/%d done)",
                         self.holding, len(self.picked), len(self.pick_objects))
            self.holding = None
            self.held_origin = None
            self.current_state = "pick"

    def save_origin(self, pos):
        self.held_origin = pos.copy()

    def is_done(self):
        return (self.current_state == "pick" and
                len(self.picked) >= len(self.pick_objects))


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

    def __init__(self, enable_move_service=False):
        rospy.wait_for_service("/sim/teleport_object", timeout=10)
        self.teleport_srv = rospy.ServiceProxy("/sim/teleport_object", TeleportObject)
        self.joint_pub = rospy.Publisher(
            "relaxed_ik/joint_angle_solutions", JointState, queue_size=10)

        self.move_srv = None
        if enable_move_service:
            try:
                rospy.wait_for_service("/move_to_cartesian_pose", timeout=5)
                self.move_srv = rospy.ServiceProxy(
                    "/move_to_cartesian_pose", MoveToCartesianPose)
                rospy.loginfo("AutoCompleter: /move_to_cartesian_pose ready")
            except Exception as e:
                rospy.logwarn(
                    "AutoCompleter: /move_to_cartesian_pose unavailable (%s); "
                    "SE(3) drive will be a no-op", e)

    def drive_to_pose(self, pos, quat_wxyz):
        """Drive the EE to an SE(3) pose via /move_to_cartesian_pose.

        quat_wxyz is (w, x, y, z). Returns True on success.
        """
        if self.move_srv is None:
            rospy.logwarn("  drive_to_pose: move service not connected")
            return False
        w, x, y, z = quat_wxyz
        try:
            resp = self.move_srv(
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                qx=float(x), qy=float(y), qz=float(z), qw=float(w),
                cartesian=False)
            if not resp.success:
                rospy.logwarn("  drive_to_pose failed: %s", resp.message)
            return bool(resp.success)
        except Exception as e:
            rospy.logwarn("  drive_to_pose exception: %s", e)
            return False

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
# Live Probability Visualizer
# ---------------------------------------------------------------------------

class PosteriorVisualizer:
    """Live matplotlib bar chart of goal posterior probabilities."""

    def __init__(self, threshold=0.95):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.threshold = threshold
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.canvas.manager.set_window_title("Intent Inference — Posterior")
        self.bars = None
        self.goal_ids = []
        self.target_id = None
        plt.ion()
        self.fig.show()

    def reset(self, goal_ids, target_id):
        """Reset for a new goal inference episode."""
        self.goal_ids = list(goal_ids)
        self.target_id = target_id
        self.ax.clear()
        self.bars = None

    def update(self, distribution, top_goal, top_prob, elapsed_time):
        """Update the bar chart with the current posterior distribution."""
        if not self.goal_ids:
            return
        probs = [distribution.get(gid, 0.0) for gid in self.goal_ids]
        colors = ["#2ecc71" if gid == self.target_id else "#3498db"
                  for gid in self.goal_ids]

        self.ax.clear()
        y_pos = range(len(self.goal_ids))
        self.ax.barh(y_pos, probs, color=colors, height=0.6)
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(self.goal_ids, fontsize=10)
        self.ax.set_xlim(0, 1.05)
        self.ax.axvline(x=self.threshold, color="#e74c3c", linestyle="--",
                        linewidth=1.5, label=f"threshold={self.threshold}")

        # Annotate probabilities on bars
        for i, p in enumerate(probs):
            self.ax.text(min(p + 0.02, 0.98), i, f"{p:.3f}",
                        va="center", fontsize=9)

        inferred_str = f"{top_goal} (p={top_prob:.3f})" if top_goal else "—"
        self.ax.set_title(
            f"Target: {self.target_id}  |  Top: {inferred_str}  |  "
            f"t={elapsed_time:.2f}s",
            fontsize=11)
        self.ax.legend(loc="lower right", fontsize=9)
        self.ax.set_xlabel("P(goal | observations)")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        self.plt.close(self.fig)


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shared autonomy runner")
    parser.add_argument("task_config", nargs="?",
                        default="config/tasks/full_breakfast_sa.yaml",
                        help="Path to SA task YAML")
    parser.add_argument("--noise", type=float, default=0.03,
                        help="Gaussian noise sigma (m/s)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Intent confidence threshold")
    parser.add_argument("--inference-model", type=str, default="gaussian",
                        choices=["gaussian", "path_efficiency"],
                        help="Intent inference model (default: gaussian)")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Gaussian direction model noise sigma (default: 0.5)")
    parser.add_argument("--beta", type=float, default=5.0,
                        help="Path efficiency model rationality coefficient")
    parser.add_argument("--max-speed", type=float, default=0.15,
                        help="Max EE speed (m/s)")
    parser.add_argument("--control-rate", type=float, default=20.0,
                        help="Control loop rate (Hz)")
    parser.add_argument("--user-intent", type=str, default=None,
                        help="Force simulated user to always pick this goal ID")
    parser.add_argument("--user-sequence", type=str, default=None,
                        help="Comma-separated pick order, e.g. 'mug,stapler,pen_cup'. "
                             "At each pick state, pursue the next object in this list.")
    parser.add_argument("--scene", type=str, default=None,
                        help="Scene name (overrides task config)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: pause for keyboard input between goals")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live matplotlib bar chart of goal posteriors")
    parser.add_argument(
        "--intent-mode", type=str, default="2d-center",
        choices=list(SE3_INTENT_MODES),
        help="Goal representation: 2d-center (legacy), 3d-center, "
             "3d-grasp-pos, or se3-grasp (adds rotation channel). "
             "Modes other than 2d-center switch to the se3_observers "
             "and require --grasp-library.")
    parser.add_argument(
        "--grasp-library", type=str, default=None,
        help="Path to grasp_poses_3d.yaml. Defaults to "
             "<project>/config/grasp_poses_3d.yaml when intent-mode is "
             "not 2d-center.")
    parser.add_argument(
        "--lambda-R", dest="lambda_R", type=float, default=0.04,
        help="SE(3) metric rotation weight (m^2/rad^2). Only used when "
             "--intent-mode=se3-grasp.")
    parser.add_argument(
        "--sigma-v", dest="sigma_v", type=float, default=0.5,
        help="SE(3) Gaussian observer translation noise.")
    parser.add_argument(
        "--sigma-w", dest="sigma_w", type=float, default=0.5,
        help="SE(3) Gaussian observer rotation noise.")
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
    scene_name = args.scene if (args.scene and args.scene.strip()) else task_config.get("scene", "scene_breakfast")
    scene_path = os.path.join(pkg_root, "config", "scenes", f"{scene_name}.yaml")
    with open(scene_path) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    object_det_ids = {}
    object_mujoco_names = {}
    object_yaws = {}
    for i, obj in enumerate(scene_cfg["objects"]):
        object_det_ids[obj["short_name"]] = i
        object_mujoco_names[obj["short_name"]] = obj["mujoco_name"]
        if "yaw" in obj:
            object_yaws[obj["short_name"]] = float(obj["yaw"])

    # SE(3) plumbing (M8). Load the 3D grasp library and SE(3) observers
    # only when the user asks for a non-legacy intent mode.
    intent_mode = args.intent_mode
    is_se3_mode = intent_mode != "2d-center"
    use_rotation = intent_mode == "se3-grasp"
    grasp_library = None
    quat_to_rot = None
    if is_se3_mode:
        from envopt.grasp_library import GraspLibrary
        from envopt.se3_utils import quat_to_rot as _quat_to_rot
        quat_to_rot = _quat_to_rot
        grasp_lib_path = args.grasp_library or os.path.join(
            pkg_root, "config", "grasp_poses_3d.yaml")
        grasp_library = GraspLibrary.load(grasp_lib_path)
        rospy.loginfo("SE(3) mode: %s  lib: %s  lambda_R: %.3f",
                      intent_mode, grasp_lib_path,
                      args.lambda_R if use_rotation else 0.0)

    # MuJoCo model path for Jacobian computation. The maintained optimized
    # scene variants only exist as SE(3) MAP-Elites YAML layouts under
    # config/scenes/; the MuJoCo XML is the base scene. Strip the suffix to
    # find it.
    base_scene_name = scene_name
    for _suffix in ("_se3_me_optimized",):
        if base_scene_name.endswith(_suffix):
            base_scene_name = base_scene_name[: -len(_suffix)]
            break
    model_path = os.path.join(
        pkg_root, "src", "assets", f"{base_scene_name}.xml")

    print(f"\n{'=' * 60}")
    print(f"  SHARED AUTONOMY: {task_config['name']}")
    print(f"  Intent mode: {args.intent_mode}")
    print(f"  Inference model: {args.inference_model}")
    print(f"  Noise sigma: {args.noise} m/s")
    print(f"  Threshold: {args.threshold}")
    if args.inference_model == "gaussian":
        print(f"  Gaussian sigma: {args.sigma}")
    else:
        print(f"  Beta: {args.beta}")
    print(f"  Max speed: {args.max_speed} m/s")
    print(f"  Control rate: {args.control_rate} Hz")
    if args.debug:
        print(f"  DEBUG MODE: will pause between goals")
    if args.visualize:
        print(f"  VISUALIZE: live posterior chart enabled")
    print(f"{'=' * 60}\n")

    # Initialize components
    joystick = SimulatedJoystick(
        noise_sigma=args.noise, max_speed=args.max_speed)
    if is_se3_mode:
        from envopt.se3_observers import (
            GaussianDirectionInferenceSE3, PathEfficiencySE3Inference)
        effective_lambda_R = args.lambda_R if use_rotation else 0.0
        if args.inference_model == "gaussian":
            inference = GaussianDirectionInferenceSE3(
                sigma_v=args.sigma_v, sigma_w=args.sigma_w,
                lambda_R=effective_lambda_R, threshold=args.threshold)
        else:
            inference = PathEfficiencySE3Inference(
                beta=args.beta, lambda_R=effective_lambda_R,
                threshold=args.threshold)
    else:
        inference = create_inference_engine(
            args.inference_model,
            sigma=args.sigma, beta=args.beta, threshold=args.threshold)
    task_mode = task_config.get("mode", "state_machine")
    if task_mode == "pick_and_return":
        state_machine = PickAndReturnStateMachine(task_config, object_det_ids)
    else:
        state_machine = TaskStateMachine(task_config, object_det_ids)
    auto_completer = AutoCompleter(enable_move_service=is_se3_mode)
    jac_controller = JacobianVelocityController(model_path)
    visualizer = PosteriorVisualizer(threshold=args.threshold) if args.visualize else None

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

    def _ee_pose_from_msg(msg):
        """Return (pos[3], R[3,3] or None) from an EndpointState message."""
        p = np.array([msg.pose.position.x,
                      msg.pose.position.y,
                      msg.pose.position.z])
        if not is_se3_mode or quat_to_rot is None:
            return p, None
        o = msg.pose.orientation
        q = np.array([o.w, o.x, o.y, o.z], dtype=float)
        return p, quat_to_rot(q)

    # Get initial EE pose
    ee_msg = rospy.wait_for_message(
        "/mujoco_sim/endpoint_state", EndpointState, timeout=5)
    ee_pos, ee_R = _ee_pose_from_msg(ee_msg)

    rate = rospy.Rate(args.control_rate)
    dt = 1.0 / args.control_rate
    user_goal_id = args.user_intent  # override or None

    # Parse user sequence: list of object names to pick in order
    user_sequence = None
    user_seq_idx = 0
    if args.user_sequence and args.user_sequence.strip():
        user_sequence = [s.strip() for s in args.user_sequence.split(",")
                        if s.strip()]
        if user_sequence:
            rospy.loginfo("User sequence: %s", user_sequence)
        else:
            user_sequence = None

    steps_completed = 0
    max_steps = 50  # safety limit
    step_times = []  # track inference time per step (pick only)
    pick_origins = {}  # object_name -> position when picked (for return)

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
        se3_goals = {} if is_se3_mode else None
        for g, pos in valid_goals:
            vg = ValidGoal()
            vg.goal_id = g["id"]
            vg.action_type = g["action"]
            vg.target_position = Point(x=pos[0], y=pos[1], z=pos[2])
            vg.object_name = g.get("object", "")
            vg_msg.goals.append(vg)
            goal_positions[g["id"]] = pos
            if is_se3_mode:
                se3_goals[g["id"]] = build_se3_goal(
                    g, pos, object_yaws, grasp_library)
        goals_pub.publish(vg_msg)

        # Choose which goal the simulated user is going for
        if user_goal_id and user_goal_id in goal_positions:
            target_id = user_goal_id
        elif user_sequence and user_seq_idx < len(user_sequence):
            # Find the pick goal matching the next object in the sequence
            next_obj = user_sequence[user_seq_idx]
            matched = False
            for g, _ in valid_goals:
                if g["action"] == "pick" and g.get("object") == next_obj:
                    target_id = g["id"]
                    matched = True
                    break
            if not matched:
                # Non-pick state (place/pour) or object not available: use first goal
                target_id = valid_goals[0][0]["id"]
        elif isinstance(state_machine, PickAndReturnStateMachine):
            # Default for pick_and_return: pick next unpicked object
            target_id = None
            for g, _ in valid_goals:
                if g["action"] == "pick" and g.get("object") not in state_machine.picked:
                    target_id = g["id"]
                    break
            if target_id is None:
                target_id = valid_goals[0][0]["id"]
        else:
            # Default: pick the first valid goal not yet picked
            target_id = None
            for g, _ in valid_goals:
                if g["action"] == "pick" and g.get("object") not in pick_origins:
                    target_id = g["id"]
                    break
            if target_id is None:
                target_id = valid_goals[0][0]["id"]
        target_pos = goal_positions[target_id]
        # In SE(3) modes, drive the joystick toward the pre-grasp standoff
        # rather than the object center. Inference still uses the same
        # target via se3_goals[target_id]["pos"]. Falls back to the plain
        # position for non-pick or library-missing goals.
        target_se3 = se3_goals[target_id] if is_se3_mode else None
        if target_se3 is not None and target_se3.get("is_grasp"):
            target_pos = np.asarray(target_se3["pos"], dtype=float)

        # Look up goal_spec early so we can check the action type
        goal_spec = None
        for g, _ in valid_goals:
            if g["id"] == target_id:
                goal_spec = g
                break
        if goal_spec is None:
            rospy.logerr("Goal spec not found for '%s'", target_id)
            break

        action = goal_spec["action"]
        obj_name = goal_spec.get("object", state_machine.holding or "")
        mujoco_name = object_mujoco_names.get(obj_name, "")

        rospy.loginfo("=" * 50)
        rospy.loginfo("State: %s | User intent: %s | Goals: %s",
                      state_machine.current_state, target_id,
                      [g["id"] for g, _ in valid_goals])

        # ----- Place actions: skip inference, return to pick origin -----
        if action == "place":
            # Use saved pick origin as destination
            return_pos = pick_origins.get(obj_name, target_pos)
            rospy.loginfo("  PLACE (no inference): %s -> [%.3f, %.3f, %.3f]",
                          obj_name, *return_pos)

            auto_completer.complete(action, obj_name, return_pos, mujoco_name)
            state_machine.transition(goal_spec)
            # For pick_and_return: also update held_origin so the state
            # machine uses the correct position
            if isinstance(state_machine, PickAndReturnStateMachine):
                state_machine.save_origin(return_pos)

            if args.debug:
                print(f"\n{'─' * 50}")
                print(f"  PLACE (auto): {obj_name} returned to origin")
                print(f"{'─' * 50}")
                try:
                    with open("/dev/tty") as tty:
                        print("  Press Enter to continue "
                              "(or 'q' + Enter to quit)...", end=" ",
                              flush=True)
                        resp = tty.readline().strip()
                        if resp.lower() == "q":
                            rospy.loginfo("User quit from debug pause.")
                            rospy.signal_shutdown("debug quit")
                            break
                except (IOError, OSError):
                    input("  Press Enter to continue... ")
            continue  # next goal — place is not recorded in results

        # ----- Pick actions: save origin, run inference -----
        pick_origins[obj_name] = target_pos.copy()

        # For pick_and_return: also update the state machine's origin
        if isinstance(state_machine, PickAndReturnStateMachine):
            state_machine.save_origin(target_pos)

        # Reset inference for this goal
        if is_se3_mode:
            inference.reset(ee_pos, ee_R)
        else:
            inference.reset(ee_pos)
        goal_start_time = time.time()

        if visualizer:
            visualizer.reset(list(goal_positions.keys()), target_id)

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

            ee_pos, ee_R = _ee_pose_from_msg(ee_msg)

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

            # Update intent inference. SE(3) observers take (ee_pos, ee_R,
            # se3_goals, observed_twist_6d); 2D observers take the legacy
            # (ee_pos, goal_positions, observed_velocity=cart_vel) signature.
            if is_se3_mode:
                # ω=0 MVP: the simulated joystick is translation-only for
                # now. The Boltzmann path-efficiency observer still sees
                # the rotation channel through the EE pose trajectory.
                observed_twist = np.concatenate([cart_vel, np.zeros(3)])
                dist, top_goal, top_prob = inference.update(
                    ee_pos, ee_R, se3_goals, observed_twist)
            else:
                dist, top_goal, top_prob = inference.update(
                    ee_pos, goal_positions, observed_velocity=cart_vel)

            # Update visualizer
            if visualizer:
                visualizer.update(dist, top_goal, top_prob,
                                  time.time() - goal_start_time)

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

                # SE(3) mode: actually execute the inferred grasp by
                # commanding MoveIt to the final grasp tip pose. This is
                # the M8 criterion — "auto-completion executes the
                # intended SE(3) grasp". Legacy 2d-center mode keeps the
                # old behavior (log + snap home).
                if is_se3_mode and action == "pick" and target_se3 is not None \
                        and target_se3.get("is_grasp"):
                    rospy.loginfo(
                        "  SE(3) drive to grasp tip [%.3f, %.3f, %.3f]",
                        *target_se3["grasp_pos"])
                    auto_completer.drive_to_pose(
                        target_se3["grasp_pos"], target_se3["grasp_quat"])

                auto_completer.complete(
                    action, obj_name, target_pos, mujoco_name)

                # Transition state
                state_machine.transition(goal_spec)
                steps_completed += 1
                goal_reached = True
                step_times.append({
                    "step": steps_completed,
                    "goal_id": top_goal,
                    "action": action,
                    "object": obj_name,
                    "inference_time": inference_time,
                    "confidence": float(top_prob),
                })

                # Advance user sequence if this was a pick
                if user_sequence and action == "pick":
                    user_seq_idx += 1

                # Reset user intent for next goal
                user_goal_id = None

                # Debug pause: show summary and wait for user
                if args.debug:
                    print(f"\n{'─' * 50}")
                    print(f"  DEBUG PAUSE — Step {steps_completed} complete")
                    print(f"  Action: {action} {obj_name}")
                    print(f"  Inference time: {inference_time:.3f}s")
                    print(f"  Confidence: {top_prob:.4f}")
                    print(f"  Final posterior:")
                    for gid in sorted(dist.keys()):
                        marker = " ◄ TRUE" if gid == target_id else ""
                        print(f"    {gid:30s}  {dist[gid]:.4f}{marker}")
                    print(f"{'─' * 50}")
                    try:
                        with open("/dev/tty") as tty:
                            print("  Press Enter to continue "
                                  "(or 'q' + Enter to quit)...", end=" ",
                                  flush=True)
                            resp = tty.readline().strip()
                            if resp.lower() == "q":
                                rospy.loginfo("User quit from debug pause.")
                                rospy.signal_shutdown("debug quit")
                                break
                    except (IOError, OSError):
                        input("  Press Enter to continue... ")

            rate.sleep()

    # Close visualizer
    if visualizer:
        visualizer.close()

    # Final summary
    rospy.loginfo("")
    rospy.loginfo("=" * 50)
    rospy.loginfo("SHARED AUTONOMY COMPLETE")
    rospy.loginfo("  Steps completed: %d", steps_completed)
    rospy.loginfo("  Final state: %s", state_machine.current_state)
    if step_times:
        # Only count pick steps with >1 competing goal for inference time
        pick_infer_times = [s["inference_time"] for s in step_times
                           if s["action"] == "pick"]
        total_infer = sum(s["inference_time"] for s in step_times)
        rospy.loginfo("  Per-step breakdown:")
        for s in step_times:
            rospy.loginfo("    Step %d: %s %s — %.1fs (p=%.3f)",
                         s["step"], s["action"], s["object"],
                         s["inference_time"], s["confidence"])
        rospy.loginfo("  Total inference time: %.1fs", total_infer)
        if pick_infer_times:
            rospy.loginfo("  Mean pick inference time: %.1fs",
                         np.mean(pick_infer_times))
    if user_sequence:
        rospy.loginfo("  User sequence: %s", user_sequence)
    rospy.loginfo("=" * 50)

    # Save results JSON
    import json
    results_out = {
        "task": task_config["name"],
        "scene": scene_name,
        "intent_mode": args.intent_mode,
        "inference_model": args.inference_model,
        "user_sequence": user_sequence,
        "noise": args.noise,
        "threshold": args.threshold,
        "sigma": args.sigma if args.inference_model == "gaussian" else None,
        "beta": args.beta if args.inference_model == "path_efficiency" else None,
        "lambda_R": args.lambda_R if args.intent_mode == "se3-grasp" else None,
        "sigma_v": args.sigma_v if is_se3_mode else None,
        "sigma_w": args.sigma_w if use_rotation else None,
        "steps_completed": steps_completed,
        "step_times": step_times,
        "total_inference_time": sum(s["inference_time"] for s in step_times) if step_times else 0,
    }
    out_dir = os.path.join(pkg_root, "results", "sa_runs")
    os.makedirs(out_dir, exist_ok=True)
    seq_str = "_".join(user_sequence) if user_sequence else "default"
    out_path = os.path.join(out_dir, f"{scene_name}_{seq_str}.json")
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    rospy.loginfo("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
