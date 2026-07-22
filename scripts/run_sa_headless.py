#!/usr/bin/env python3
"""Headless shared autonomy runner — no ROS, no GUI.

Replicates the shared_autonomy_runner.py logic using MuJoCo directly:
- Loads scene XML and steps physics
- Computes EE position via forward kinematics (mj_forward)
- Uses the same SimulatedJoystick, Inference, and Jacobian code
- Teleports objects by setting qpos directly
- Outputs the same metrics (per-step inference time, total time)

Differences from the ROS version:
- No ROS communication overhead (~1-5ms per message wait)
- No RelaxedIK intermediary (joint targets applied directly)
- Physics steps are synchronous (no real-time clock)
- Object positions from qpos, not detection messages
- Timing is measured in simulated control steps, not wall clock

These differences mean inference times should be very similar in terms
of number of control steps, but wall-clock times will differ (headless
is faster). The inference algorithm and decisions are identical.

Usage:
    python3 scripts/run_sa_headless.py config/tasks/desk_organize_v2_sa.yaml
    python3 scripts/run_sa_headless.py config/tasks/desk_organize_v2_sa.yaml --scene scene_desk_se3_me_optimized
    python3 scripts/run_sa_headless.py config/tasks/desk_organize_v2_sa.yaml --user-sequence stapler,pen_cup,mug
"""
import sys
import os
import re
import json
import argparse
import time
import numpy as np
import yaml
import mujoco

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make envopt importable when this script is run directly.
_SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from envopt.layout_sampling import (
    MIN_DIST,
    ROBOT_EXCLUSION_RADIUS,
    START_POS_2D,
    TABLE_BOUNDS_X,
    TABLE_BOUNDS_Y,
    generate_random_layouts,
    generate_random_se3_layouts,
)
from shared_autonomy.headless_core import (
    GaussianDirectionInference,
    PathEfficiencyInference,
    SimulatedJoystick,
    select_transition_spec,
)
from shared_autonomy.headless_state import (
    HeadlessTaskStateMachine,
    PickAndReturnStateMachine,
)


def apply_layout(sim, scene_cfg, layout_xy):
    """Teleport movable objects to the given 2D positions.

    Preserves z from the current sim state.
    """
    mujoco_names = {o["short_name"]: o["mujoco_name"]
                    for o in scene_cfg["objects"]}
    for short_name, xy in layout_xy.items():
        mname = mujoco_names.get(short_name)
        if mname is None:
            continue
        current_pos = sim.get_object_pos(mname)
        new_pos = np.array([xy[0], xy[1], current_pos[2]])
        sim.teleport_object(mname, new_pos)


# ---------------------------------------------------------------------------
# Headless MuJoCo simulation
# ---------------------------------------------------------------------------

class HeadlessSim:
    """MuJoCo simulation without ROS."""

    HOME_JOINTS = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]

    def __init__(self, scene_xml_path):
        self.model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = self.model.site("end_effector").id

        # Arm joint indices
        self.arm_qpos_addrs = []
        self.arm_dof_addrs = []
        self.joint_lower = []
        self.joint_upper = []
        for i in range(7):
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_j{i}")
            self.arm_qpos_addrs.append(self.model.jnt_qposadr[jid])
            self.arm_dof_addrs.append(self.model.jnt_dofadr[jid])
            self.joint_lower.append(self.model.jnt_range[jid, 0])
            self.joint_upper.append(self.model.jnt_range[jid, 1])
        self.joint_lower = np.array(self.joint_lower)
        self.joint_upper = np.array(self.joint_upper)

        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_rot(self):
        """Return the current 3x3 EE rotation matrix (world frame)."""
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def get_joint_positions(self):
        return np.array([self.data.qpos[a] for a in self.arm_qpos_addrs])

    def set_joint_positions(self, positions):
        for i, addr in enumerate(self.arm_qpos_addrs):
            self.data.qpos[addr] = positions[i]
        mujoco.mj_forward(self.model, self.data)

    def get_object_pos(self, mujoco_name):
        """Get object position from body pos."""
        bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
        return self.data.xpos[bid].copy()

    def teleport_object(self, mujoco_name, pos):
        """Move an object by setting its freejoint qpos."""
        bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
        jid = self.model.body_jntadr[bid]
        if jid < 0:
            return  # no joint (fixed body)
        qadr = self.model.jnt_qposadr[jid]
        self.data.qpos[qadr:qadr+3] = pos
        self.data.qpos[qadr+3:qadr+7] = [1, 0, 0, 0]  # identity quaternion
        # Zero velocities
        vadr = self.model.jnt_dofadr[jid]
        self.data.qvel[vadr:vadr+6] = 0
        mujoco.mj_forward(self.model, self.data)

    def compute_new_joints(self, joint_positions, cart_vel, dt):
        """Jacobian-based IK identical to JacobianVelocityController."""
        self.set_joint_positions(joint_positions)

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        J = jacp[:, self.arm_dof_addrs]
        damping = 0.05
        JJT = J @ J.T + damping * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)
        joint_vel = J_pinv @ cart_vel

        # Joint limit avoidance
        margin = 0.1
        repulsion_gain = 2.0
        for i in range(7):
            dist_lo = joint_positions[i] - self.joint_lower[i]
            dist_hi = self.joint_upper[i] - joint_positions[i]
            if dist_lo < margin:
                joint_vel[i] += repulsion_gain * (margin - dist_lo)
            if dist_hi < margin:
                joint_vel[i] -= repulsion_gain * (margin - dist_hi)

        new_joints = joint_positions + joint_vel * dt
        safety = 0.02
        return np.clip(new_joints,
                      self.joint_lower + safety,
                      self.joint_upper - safety)

    def compute_new_joints_6d(self, joint_positions, cart_twist, dt):
        """6-DoF Jacobian-based velocity controller.

        Args:
            joint_positions: (7,) current arm joint positions.
            cart_twist: (6,) desired twist (vx, vy, vz, wx, wy, wz) in world.
            dt: timestep in seconds.

        Returns new (7,) joint positions after one dt, with joint limit
        avoidance and clamping identical to compute_new_joints().
        """
        self.set_joint_positions(joint_positions)

        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

        Jp = jacp[:, self.arm_dof_addrs]   # 3x7 translation
        Jr = jacr[:, self.arm_dof_addrs]   # 3x7 rotation
        J = np.vstack([Jp, Jr])            # 6x7

        damping = 0.05
        JJT = J @ J.T + damping * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJT)  # 7x6
        joint_vel = J_pinv @ cart_twist

        # Joint limit avoidance (same as 3D path)
        margin = 0.1
        repulsion_gain = 2.0
        for i in range(7):
            dist_lo = joint_positions[i] - self.joint_lower[i]
            dist_hi = self.joint_upper[i] - joint_positions[i]
            if dist_lo < margin:
                joint_vel[i] += repulsion_gain * (margin - dist_lo)
            if dist_hi < margin:
                joint_vel[i] -= repulsion_gain * (margin - dist_hi)

        new_joints = joint_positions + joint_vel * dt
        safety = 0.02
        return np.clip(new_joints,
                       self.joint_lower + safety,
                       self.joint_upper - safety)

    def go_home(self):
        self.set_joint_positions(self.HOME_JOINTS)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

# Thresholds recorded in every run when bypass_threshold is enabled.
# Chosen to densely cover the Boltzmann-saturation region [0.8, 0.95].
TAU_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def _select_transition_spec(target_spec, inferred_spec, benchmark_mode):
    """Backward-compatible alias for tests and older imports."""
    return select_transition_spec(target_spec, inferred_spec, benchmark_mode)


def run_headless_sa(task_config_path, scene_name=None,
                    inference_model="path_efficiency", noise=0.03,
                    threshold=0.9, sigma=0.5, beta=5.0,
                    max_speed=0.05, control_rate=20.0,
                    user_sequence=None, seed=42,
                    goal_timeout=30.0,
                    layout_override=None, quiet=False,
                    motion_2d=True,
                    bypass_threshold=False):
    """Run shared autonomy headlessly. Returns results dict.

    Args:
        layout_override: optional {short_name: np.array([x, y])} to
            teleport movable objects before running. Used for random
            baseline evaluation without generating separate scene XMLs.
        quiet: if True, suppress per-step log output.
        motion_2d: if True, constrain EE to move only in xy plane (no
            z descent), and feed 2D positions to the path-efficiency
            observer. This matches the optimizer's 2D-integrator
            assumption.
        bypass_threshold: if True, do not terminate the control loop
            when the posterior crosses `threshold`. Instead, run until
            stall or timeout and record the first-crossing step/goal
            for every threshold in TAU_SWEEP. Enables post-hoc
            threshold sensitivity analysis from a single run.
    """
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)["task"]

    task_mode = task_config.get("mode", "state_machine")

    if not scene_name:
        scene_name = task_config.get("scene", "scene_breakfast")

    scene_xml = os.path.join(PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    scene_yaml = os.path.join(PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")

    with open(scene_yaml) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    # Initialize
    sim = HeadlessSim(scene_xml)

    # Apply layout override if provided (teleport objects)
    if layout_override:
        apply_layout(sim, scene_cfg, layout_override)

    joystick = SimulatedJoystick(noise_sigma=noise, max_speed=max_speed)
    joystick.rng = np.random.default_rng(seed)

    if inference_model == "gaussian":
        inference = GaussianDirectionInference(sigma=sigma, threshold=threshold)
    else:
        inference = PathEfficiencyInference(beta=beta, threshold=threshold)

    if task_mode == "pick_and_return":
        state_machine = PickAndReturnStateMachine(task_config, scene_cfg)
    else:
        state_machine = HeadlessTaskStateMachine(task_config, scene_cfg)
    dt = 1.0 / control_rate

    # Parse user sequence
    seq = None
    seq_idx = 0
    if user_sequence:
        seq = [s.strip() for s in user_sequence.split(",") if s.strip()]

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  HEADLESS SA: {task_config['name']}")
        print(f"  Scene: {scene_name}")
        print(f"  Inference: {inference_model}, threshold={threshold}")
        print(f"  Noise: {noise} m/s, max_speed: {max_speed} m/s")
        if seq:
            print(f"  User sequence: {seq}")
        if layout_override:
            print(f"  Layout override: {list(layout_override.keys())}")
        print(f"{'='*60}\n")

    steps_completed = 0
    step_times = []
    max_steps = 50

    while not state_machine.is_done() and steps_completed < max_steps:
        valid_goals = state_machine.get_valid_goals(sim)
        if not valid_goals:
            break

        # Choose target
        goal_positions = {}
        for g, pos in valid_goals:
            goal_positions[g["id"]] = pos

        target_id = None
        if seq and seq_idx < len(seq):
            # Follow explicit user sequence
            for g, _ in valid_goals:
                if g["action"] == "pick" and g.get("object") == seq[seq_idx]:
                    target_id = g["id"]
                    break
        if target_id is None and isinstance(state_machine, PickAndReturnStateMachine):
            # Default for pick_and_return: pick next unpicked object
            for g, _ in valid_goals:
                if g["action"] == "pick" and g.get("object") not in state_machine.picked:
                    target_id = g["id"]
                    break
        if target_id is None:
            target_id = valid_goals[0][0]["id"]
        target_pos = goal_positions[target_id]

        if not quiet:
            print(f"  State: {state_machine.current_state} | "
                  f"Intent: {target_id} | "
                  f"Goals: {list(goal_positions.keys())}")

        # Reset inference (use 2D positions when motion_2d to keep
        # the path-efficiency observer consistent with the optimizer)
        ee_pos = sim.get_ee_pos()
        if motion_2d and not isinstance(inference, GaussianDirectionInference):
            inference.reset(ee_pos[:2])
        else:
            inference.reset(ee_pos)

        # Control loop — run until threshold crossed, EE has stopped
        # advancing (loitering), or timeout. Loitering termination
        # avoids corrupting the cost numerator after the user has
        # effectively arrived at the goal.
        goal_reached = False
        loop_count = 0
        max_loops = int(goal_timeout * control_rate)
        max_prob_seen = 0.0
        step_at_max_prob = 0
        last_top_goal = None
        last_top_prob = 0.0
        last_dist = {}
        # Loiter detection: terminate when EE has stopped advancing
        # toward ANY goal. Use a windowed net-displacement check
        # rather than per-step delta because the joystick noise keeps
        # the EE jittering at ~1.5mm/step even after arrival.
        STALL_WINDOW = 20  # 1 second at 20Hz
        STALL_NET_DELTA = 0.005  # 5mm net displacement over the window
        ee_window = []
        stall_terminated = False
        # Threshold-sweep recording: for each tau in TAU_SWEEP, record
        # (step, top_goal) at the moment top_prob first reaches tau.
        tau_crossings = {tau: None for tau in TAU_SWEEP}

        while not goal_reached and loop_count < max_loops:
            loop_count += 1

            ee_pos = sim.get_ee_pos()
            joint_pos = sim.get_joint_positions()

            # In 2D mode, fix target z to current EE z so the joystick
            # produces no z velocity, and zero out any residual z drift.
            if motion_2d:
                effective_target = np.array(
                    [target_pos[0], target_pos[1], ee_pos[2]])
            else:
                effective_target = target_pos
            cart_vel = joystick.compute_velocity(ee_pos, effective_target)
            if motion_2d:
                cart_vel[2] = 0.0

            # Move the arm
            new_joints = sim.compute_new_joints(joint_pos, cart_vel, dt)
            sim.set_joint_positions(new_joints)

            # Build observation for inference
            ee_pos_2d = ee_pos[:2]
            goal_pos_2d = {gid: gpos[:2] for gid, gpos in goal_positions.items()}

            if isinstance(inference, GaussianDirectionInference):
                target_2d = target_pos[:2]
                dir_2d = target_2d - ee_pos_2d
                dir_norm = np.linalg.norm(dir_2d)
                if dir_norm > 1e-6:
                    unit_dir = dir_2d / dir_norm
                else:
                    unit_dir = np.zeros(2)
                obs_noise = joystick.rng.normal(0, inference.sigma, 2)
                u_t = unit_dir + obs_noise
                dist, top_goal, top_prob = inference.update(
                    ee_pos_2d, goal_pos_2d, observed_velocity=u_t)
            elif motion_2d:
                # 2D path-efficiency observer matches the optimizer
                # exactly: positions and distances live in xy.
                dist, top_goal, top_prob = inference.update(
                    ee_pos_2d, goal_pos_2d)
            else:
                dist, top_goal, top_prob = inference.update(
                    ee_pos, goal_positions)

            last_top_goal = top_goal
            last_top_prob = top_prob
            last_dist = dist

            # Track the max probability the true goal reached (for
            # threshold sensitivity: "would threshold X have been met?")
            p_true = dist.get(target_id, 0.0) if dist else 0.0
            if p_true > max_prob_seen:
                max_prob_seen = p_true
                step_at_max_prob = loop_count

            # Record first-crossing step/goal for each threshold in
            # the sweep. Only fills unseen thresholds once per pick.
            for tau in TAU_SWEEP:
                if tau_crossings[tau] is None and top_prob >= tau:
                    tau_crossings[tau] = (loop_count, top_goal)

            # Stall detection: maintain a sliding window of EE positions
            # and trigger when the NET displacement across the window
            # is below STALL_NET_DELTA. This catches "wandering in
            # place" that per-step deltas miss.
            ee_pos_after = sim.get_ee_pos()
            ee_window.append(ee_pos_after[:2].copy())
            if len(ee_window) > STALL_WINDOW:
                ee_window.pop(0)
            stalled_now = False
            if len(ee_window) == STALL_WINDOW:
                net_window = np.linalg.norm(
                    ee_window[-1] - ee_window[0])
                if net_window < STALL_NET_DELTA:
                    stalled_now = True

            # Log every second of sim time
            if not quiet and loop_count % int(control_rate) == 0:
                d = np.linalg.norm(target_pos - ee_pos)
                dist_str = " ".join(f"{k}={v:.2f}" for k, v in
                                   sorted(dist.items()))
                print(f"    d={d*1000:.0f}mm | {dist_str}")

            # Stall-based termination: EE has effectively arrived.
            # Break out of the loop and let the post-loop handler
            # record this as a stall outcome (separate from threshold
            # convergence and timeout). In bypass-threshold mode,
            # stall terminates regardless of current top_prob.
            stalled = stalled_now and (bypass_threshold or
                                       top_prob < threshold)
            if stalled:
                if not quiet:
                    print(f"    STALLED at step {loop_count}, "
                          f"argmax={top_goal} (p={top_prob:.3f})")
                stall_terminated = True
                break

            if not bypass_threshold and top_prob >= threshold:
                inference_time = loop_count / control_rate
                correct = (top_goal == target_id)
                status = "CORRECT" if correct else "WRONG"
                if not quiet:
                    print(f"    INFERRED: {top_goal} (p={top_prob:.3f}, "
                          f"t={inference_time:.1f}s, "
                          f"steps={loop_count}, {status})")
                    if not correct:
                        print(f"    ** Expected {target_id}, got {top_goal} **")

                # Auto-complete with the inferred goal
                goal_spec = None
                for g, _ in valid_goals:
                    if g["id"] == top_goal:
                        goal_spec = g
                        break
                if goal_spec is None:
                    for g, _ in valid_goals:
                        if g["id"] == target_id:
                            goal_spec = g
                            break

                action = goal_spec["action"]
                obj_name = goal_spec.get("object",
                                        state_machine.holding or "")
                mname = state_machine.mujoco_names.get(obj_name, "")

                if (action == "pick" and mname and
                        isinstance(state_machine, PickAndReturnStateMachine)):
                    origin = sim.get_object_pos(mname)
                    state_machine.save_origin(origin)

                dest = goal_positions.get(top_goal, target_pos)
                if action in ("place", "pour") and mname:
                    sim.teleport_object(mname, dest)
                    if not quiet:
                        print(f"    Auto-complete {action.upper()}: {obj_name}")

                sim.go_home()
                state_machine.transition(goal_spec)
                steps_completed += 1
                step_times.append({
                    "step": steps_completed,
                    "goal_id": top_goal,
                    "target_id": target_id,
                    "correct": correct,
                    "argmax_correct": correct,
                    "action": action,
                    "object": obj_name,
                    "inference_time_s": inference_time,
                    "inference_steps": loop_count,
                    "confidence": float(top_prob),
                    "max_prob_true": float(max_prob_seen),
                    "step_at_max_prob": step_at_max_prob,
                    "timeout": False,
                    "tau_crossings": {
                        str(tau): (c if c is None else
                                   [int(c[0]), str(c[1])])
                        for tau, c in tau_crossings.items()
                    },
                })

                if action == "pick" and seq:
                    seq_idx += 1

                goal_reached = True

        # Handle timeout or stall — threshold was not crossed.
        # Record argmax correctness separately from threshold correctness.
        if not goal_reached:
            stalled_out = stall_terminated
            elapsed_time = loop_count / control_rate
            argmax_goal = last_top_goal
            argmax_correct = (argmax_goal == target_id)
            if not quiet:
                status = "CORRECT" if argmax_correct else "WRONG"
                kind = "STALL" if stalled_out else "TIMEOUT"
                print(f"    {kind} after {elapsed_time:.1f}s on {target_id} "
                      f"(argmax={argmax_goal}, p={last_top_prob:.3f}, "
                      f"{status})")

            # Force-complete the TRUE goal to advance the task
            goal_spec = None
            for g, _ in valid_goals:
                if g["id"] == target_id:
                    goal_spec = g
                    break
            if goal_spec is None:
                goal_spec = valid_goals[0][0]
            action = goal_spec["action"]
            obj_name = goal_spec.get("object",
                                    state_machine.holding or "")
            mname = state_machine.mujoco_names.get(obj_name, "")
            if (action == "pick" and mname and
                    isinstance(state_machine, PickAndReturnStateMachine)):
                origin = sim.get_object_pos(mname)
                state_machine.save_origin(origin)
            if action in ("place", "pour") and mname:
                dest = goal_positions.get(target_id, target_pos)
                sim.teleport_object(mname, dest)
            sim.go_home()
            state_machine.transition(goal_spec)
            steps_completed += 1
            step_times.append({
                "step": steps_completed,
                "goal_id": argmax_goal,
                "target_id": target_id,
                "correct": False,  # threshold not crossed
                "argmax_correct": argmax_correct,
                "action": action,
                "object": obj_name,
                "inference_time_s": elapsed_time,
                "inference_steps": loop_count,
                "confidence": float(last_top_prob),
                "max_prob_true": float(max_prob_seen),
                "step_at_max_prob": step_at_max_prob,
                "timeout": not stalled_out,
                "stalled": stalled_out,
                "tau_crossings": {
                    str(tau): (c if c is None else
                               [int(c[0]), str(c[1])])
                    for tau, c in tau_crossings.items()
                },
            })
            if action == "pick" and seq:
                seq_idx += 1

    # Summary
    if not quiet:
        print(f"\n{'='*60}")
        print(f"  COMPLETE: {steps_completed} steps")
        print(f"  Final state: {state_machine.current_state}")
        if step_times:
            pick_times = [s["inference_time_s"] for s in step_times
                         if s["action"] == "pick"
                         and s["inference_steps"] > 1]
            total_time = sum(s["inference_time_s"] for s in step_times)
            print(f"  Per-step:")
            for s in step_times:
                print(f"    {s['step']:2d}. {s['action']:5s} {s['object']:15s} "
                      f"t={s['inference_time_s']:5.1f}s  p={s['confidence']:.3f}")
            print(f"  Total inference time: {total_time:.1f}s")
            if pick_times:
                print(f"  Mean pick inference (disambig only): "
                      f"{np.mean(pick_times):.1f}s")
        print(f"{'='*60}")

    results = {
        "task": task_config["name"],
        "scene": scene_name,
        "inference_model": inference_model,
        "user_sequence": seq,
        "noise": noise,
        "threshold": threshold,
        "sigma": sigma if inference_model == "gaussian" else None,
        "beta": beta if inference_model == "path_efficiency" else None,
        "steps_completed": steps_completed,
        "step_times": step_times,
        "total_inference_time": sum(s["inference_time_s"]
                                   for s in step_times) if step_times else 0,
        "done": state_machine.is_done(),
    }

    # Save (skip for quiet/batch runs)
    if not quiet:
        out_dir = os.path.join(PROJECT_ROOT, "results", "sa_headless")
        os.makedirs(out_dir, exist_ok=True)
        seq_str = "_".join(seq) if seq else "default"
        out_path = os.path.join(out_dir, f"{scene_name}_{seq_str}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# SE(3) / 3D headless runner
# ---------------------------------------------------------------------------

def _build_se3_goal(goal_spec, target_pos_3d, intent_mode,
                    grasp_library, state_machine, sim, object_yaws):
    """Compute an SE(3) goal (position + rotation) for the observer.

    Args:
        goal_spec: the goal dict from the state machine.
        target_pos_3d: the (3,) object/place position from the SM.
        intent_mode: "3d-center", "3d-grasp-pos", or "se3-grasp".
        grasp_library: loaded GraspLibrary or None for center modes.
        state_machine: the headless state machine.
        sim: HeadlessSim instance (for looking up current object pose).
        object_yaws: {short_name: yaw_rad} for grasp pose resolution.

    Returns dict {'pos': (3,), 'R': (3,3) or None, 'grasp_entry': entry
    or None}.
    """
    from envopt.se3_utils import quat_to_rot
    action = goal_spec.get("action")
    if intent_mode == "3d-center":
        return {"pos": np.asarray(target_pos_3d, dtype=float),
                "R": None, "grasp_entry": None}

    # Grasp-aware modes: for pick, resolve grasp pose from library.
    if action == "pick" and grasp_library is not None:
        obj_name = goal_spec.get("object")
        entry = grasp_library.get(obj_name)
        if entry is None:
            # Fall back to center position if the library has no entry.
            return {"pos": np.asarray(target_pos_3d, dtype=float),
                    "R": None, "grasp_entry": None}
        yaw = object_yaws.get(obj_name, 0.0)
        poses = entry.resolve(np.asarray(target_pos_3d), yaw)
        # Use the PRE-GRASP pose as the inference target: the user reaches
        # for the standoff pose, not straight into the object.
        pos = poses["pregrasp_pos"]
        R = poses["grasp_R"] if intent_mode == "se3-grasp" else None
        return {"pos": pos, "R": R, "grasp_entry": entry}

    # Place or move: no grasp — use the raw target with current EE
    # orientation (no rotation incentive).
    return {"pos": np.asarray(target_pos_3d, dtype=float),
            "R": None, "grasp_entry": None}


def run_headless_sa_se3(task_config_path, scene_name=None,
                        inference_model="path_efficiency",
                        intent_mode="3d-center",
                        grasp_library_path=None,
                        noise=0.03, threshold=0.9,
                        sigma_v=0.5, sigma_w=0.5, beta=5.0,
                        lambda_R=0.04,
                        max_speed=0.05, max_omega=1.0,
                        control_rate=20.0,
                        user_sequence=None, seed=42,
                        goal_timeout=30.0,
                        arrival_dist=0.02,
                        layout_override=None,
                        layout_yaws=None,
                        quiet=False,
                        bypass_threshold=False,
                        benchmark_mode=False):
    """3D / SE(3) headless shared autonomy runner.

    Parallel to run_headless_sa() but operates in full 3D with SE(3)
    observers. Used by M1 onward. Does NOT touch the 2D code path.

    Args:
        intent_mode: one of "3d-center", "3d-grasp-pos", "se3-grasp".
            "3d-center":    goal = object center in 3D, no rotation
                            observations.
            "3d-grasp-pos": goal = grasp-tip / pre-grasp position from
                            the library, no rotation observations.
            "se3-grasp":    goal = full SE(3) pre-grasp pose, rotation
                            observations on.
        grasp_library_path: path to grasp_poses_3d.yaml (required for
            non-center modes).
        lambda_R: rotation weight in the SE(3) metric (only used when
            intent_mode == "se3-grasp").
        layout_yaws: {short_name: yaw_rad} assignment of object yaws,
            applied after layout_override.
        benchmark_mode: record the inferred goal, but advance the task using
            the prescribed target goal. This keeps benchmark trials
            independent after an inference error. Normal interactive behavior
            continues to transition on the inferred goal.

    Returns the same results dict shape as run_headless_sa().
    """
    from envopt.se3_observers import (
        GaussianDirectionInferenceSE3, PathEfficiencySE3Inference)
    from envopt.grasp_library import GraspLibrary

    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)["task"]

    task_mode = task_config.get("mode", "state_machine")
    if not scene_name:
        scene_name = task_config.get("scene", "scene_breakfast")

    # Task-file overrides: let the YAML specify intent_mode / grasp_library
    # so invocations don't have to pass them explicitly. CLI args still win.
    if intent_mode == "3d-center" and "intent_mode" in task_config:
        intent_mode = task_config["intent_mode"]
    task_grasp_lib = task_config.get("grasp_library")
    if grasp_library_path is None and task_grasp_lib is not None:
        grasp_library_path = task_grasp_lib
    if grasp_library_path is not None and not os.path.isabs(grasp_library_path):
        grasp_library_path = os.path.join(PROJECT_ROOT, grasp_library_path)

    scene_xml = os.path.join(PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    scene_yaml = os.path.join(PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    with open(scene_yaml) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    # Load grasp library if needed
    grasp_library = None
    if intent_mode in ("3d-grasp-pos", "se3-grasp"):
        path = grasp_library_path or os.path.join(
            PROJECT_ROOT, "config", "grasp_poses_3d.yaml")
        grasp_library = GraspLibrary.load(path)

    sim = HeadlessSim(scene_xml)
    if layout_override:
        apply_layout(sim, scene_cfg, layout_override)

    object_yaws = dict(layout_yaws or {})

    joystick = SimulatedJoystick(noise_sigma=noise, max_speed=max_speed)
    joystick.rng = np.random.default_rng(seed)
    rng = joystick.rng

    # Build observer
    if inference_model == "gaussian":
        obs_lambda = lambda_R if intent_mode == "se3-grasp" else 0.0
        inference = GaussianDirectionInferenceSE3(
            sigma_v=sigma_v, sigma_w=sigma_w,
            lambda_R=obs_lambda, threshold=threshold)
    else:
        obs_lambda = lambda_R if intent_mode == "se3-grasp" else 0.0
        inference = PathEfficiencySE3Inference(
            beta=beta, lambda_R=obs_lambda, threshold=threshold)

    if task_mode == "pick_and_return":
        state_machine = PickAndReturnStateMachine(task_config, scene_cfg)
    else:
        state_machine = HeadlessTaskStateMachine(task_config, scene_cfg)

    dt = 1.0 / control_rate

    # Parse user sequence
    seq = None
    seq_idx = 0
    if user_sequence:
        seq = [s.strip() for s in user_sequence.split(",") if s.strip()]

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  HEADLESS SA (SE(3)): {task_config['name']}")
        print(f"  Scene: {scene_name}")
        print(f"  Intent mode: {intent_mode}")
        print(f"  Inference: {inference_model}, lambda_R={obs_lambda}")
        print(f"  Noise: v={noise} m/s, max_speed={max_speed} m/s")
        if seq:
            print(f"  User sequence: {seq}")
        print(f"{'='*60}\n")

    steps_completed = 0
    step_times = []
    max_steps = 50

    while not state_machine.is_done() and steps_completed < max_steps:
        valid_goals = state_machine.get_valid_goals(sim)
        if not valid_goals:
            break

        # Build SE(3) goal dict for every valid goal
        se3_goals = {}
        goal_spec_by_id = {}
        for g, pos in valid_goals:
            gid = g["id"]
            se3_goals[gid] = _build_se3_goal(
                g, pos, intent_mode, grasp_library,
                state_machine, sim, object_yaws)
            goal_spec_by_id[gid] = (g, pos)

        # Choose target
        target_id = None
        if seq and seq_idx < len(seq):
            for g, _ in valid_goals:
                if g["action"] == "pick" and g.get("object") == seq[seq_idx]:
                    target_id = g["id"]
                    break
        if target_id is None and isinstance(state_machine, PickAndReturnStateMachine):
            for g, _ in valid_goals:
                if g["action"] == "pick" and g.get("object") not in state_machine.picked:
                    target_id = g["id"]
                    break
        if target_id is None:
            target_id = valid_goals[0][0]["id"]

        target_pose = se3_goals[target_id]

        if not quiet:
            print(f"  State: {state_machine.current_state} | "
                  f"Intent: {target_id} | goals: {list(se3_goals.keys())}")

        # Reset inference
        ee_pos = sim.get_ee_pos()
        ee_R = sim.get_ee_rot()
        inference.reset(ee_pos, ee_R)

        goal_reached = False
        loop_count = 0
        max_loops = int(goal_timeout * control_rate)
        max_prob_seen = 0.0
        step_at_max_prob = 0
        last_top_goal = None
        last_top_prob = 0.0
        last_dist = {}

        # Stall detection in 3D: track net EE displacement window.
        STALL_WINDOW = 20
        STALL_NET_DELTA = 0.005
        ee_window = []
        stall_terminated = False
        # Threshold sweep recording (same as 2D runner)
        tau_crossings = {tau: None for tau in TAU_SWEEP}

        while not goal_reached and loop_count < max_loops:
            loop_count += 1

            ee_pos = sim.get_ee_pos()
            ee_R = sim.get_ee_rot()
            joint_pos = sim.get_joint_positions()

            # Build the ideal 6-twist toward the target pose
            tgt_pos = target_pose["pos"]
            tgt_R = target_pose.get("R")

            v_dir = tgt_pos - ee_pos
            vn = np.linalg.norm(v_dir)
            if vn > 1e-6:
                v_unit = v_dir / vn
            else:
                v_unit = np.zeros(3)
            v_cmd_world = min(max_speed, 2.0 * vn) * v_unit

            if tgt_R is not None:
                from envopt.se3_utils import rot_log
                R_err = tgt_R @ ee_R.T
                w_log = rot_log(R_err)
                wn = np.linalg.norm(w_log)
                if wn > 1e-6:
                    w_unit = w_log / wn
                else:
                    w_unit = np.zeros(3)
                w_cmd_world = min(max_omega, 2.0 * wn) * w_unit
            else:
                w_cmd_world = np.zeros(3)

            # Add noise to the commanded twist (Gaussian joystick noise)
            noise_v = rng.normal(0, noise, 3)
            noise_w = rng.normal(0, noise, 3) if tgt_R is not None else np.zeros(3)
            cart_twist = np.concatenate([v_cmd_world + noise_v,
                                         w_cmd_world + noise_w])

            # Step the arm with the 6-DoF Jacobian controller.
            new_joints = sim.compute_new_joints_6d(joint_pos, cart_twist, dt)
            sim.set_joint_positions(new_joints)

            # Observation: feed the same 6-twist (as unit direction) to the
            # Gaussian observer, and the current pose to Boltzmann.
            if isinstance(inference, GaussianDirectionInferenceSE3):
                # Use unit direction + gaussian noise in the observation
                # space, matching the 2D Gaussian path's normalization.
                obs_v_noise = rng.normal(0, inference.sigma_v, 3)
                obs_twist_v = v_unit + obs_v_noise
                if tgt_R is not None and inference.use_rotation:
                    obs_w_noise = rng.normal(0, inference.sigma_w, 3)
                    obs_twist_w = w_unit + obs_w_noise
                else:
                    obs_twist_w = np.zeros(3)
                obs_twist = np.concatenate([obs_twist_v, obs_twist_w])
                dist, top_goal, top_prob = inference.update(
                    ee_pos, ee_R, se3_goals, obs_twist)
            else:
                dist, top_goal, top_prob = inference.update(
                    ee_pos, ee_R, se3_goals)

            last_top_goal = top_goal
            last_top_prob = top_prob
            last_dist = dist

            p_true = dist.get(target_id, 0.0) if dist else 0.0
            if p_true > max_prob_seen:
                max_prob_seen = p_true
                step_at_max_prob = loop_count

            # Record first-crossing step for each threshold in the sweep
            for tau in TAU_SWEEP:
                if tau_crossings[tau] is None and top_prob >= tau:
                    tau_crossings[tau] = (loop_count, top_goal)

            # Stall detection
            ee_after = sim.get_ee_pos()
            ee_window.append(ee_after.copy())
            if len(ee_window) > STALL_WINDOW:
                ee_window.pop(0)
            stalled_now = False
            if len(ee_window) == STALL_WINDOW:
                net = np.linalg.norm(ee_window[-1] - ee_window[0])
                if net < STALL_NET_DELTA:
                    stalled_now = True

            if not quiet and loop_count % int(control_rate) == 0:
                d = np.linalg.norm(tgt_pos - ee_pos)
                dist_str = " ".join(f"{k}={v:.2f}" for k, v in sorted(dist.items()))
                print(f"    d={d*1000:.0f}mm | {dist_str}")

            # Arrival-based termination: EE has reached the true target.
            # Commit on the current argmax — mirrors a deployed system
            # where the user physically arrives and the loop stops.
            d_target = np.linalg.norm(tgt_pos - ee_pos)
            arrived = (arrival_dist is not None and
                       not bypass_threshold and
                       d_target < arrival_dist)

            # In bypass mode, stall terminates regardless of top_prob
            stalled = stalled_now and (bypass_threshold or
                                       top_prob < threshold)
            if stalled:
                if not quiet:
                    print(f"    STALLED at step {loop_count}, "
                          f"argmax={top_goal} (p={top_prob:.3f})")
                stall_terminated = True
                break

            if (not bypass_threshold and top_prob >= threshold) or arrived:
                inference_time = loop_count / control_rate
                correct = (top_goal == target_id)
                status = "CORRECT" if correct else "WRONG"
                reason = "ARRIVED" if arrived else "INFERRED"
                if not quiet:
                    print(f"    {reason}: {top_goal} (p={top_prob:.3f}, "
                          f"t={inference_time:.1f}s, "
                          f"steps={loop_count}, "
                          f"d={d_target*1000:.0f}mm, {status})")

                inferred_spec, _ = goal_spec_by_id.get(
                    top_goal, goal_spec_by_id[target_id])
                target_spec, _ = goal_spec_by_id[target_id]
                transition_spec = _select_transition_spec(
                    target_spec, inferred_spec, benchmark_mode)
                action = transition_spec["action"]
                obj_name = transition_spec.get(
                    "object", state_machine.holding or "")
                inferred_obj_name = inferred_spec.get("object", "")
                target_obj_name = target_spec.get("object", "")
                mname = state_machine.mujoco_names.get(obj_name, "")

                if (action == "pick" and mname and
                        isinstance(state_machine, PickAndReturnStateMachine)):
                    origin = sim.get_object_pos(mname)
                    state_machine.save_origin(origin)

                # Auto-complete: teleport for place/pour, snap home after.
                if action in ("place", "pour") and mname:
                    transition_id = (target_id if benchmark_mode else top_goal)
                    dest = goal_spec_by_id[transition_id][1] \
                        if transition_id in goal_spec_by_id else tgt_pos
                    sim.teleport_object(mname, dest)

                sim.go_home()
                state_machine.transition(transition_spec)
                steps_completed += 1
                step_times.append({
                    "step": steps_completed,
                    "goal_id": top_goal,
                    "target_id": target_id,
                    "inferred_id": top_goal,
                    "target_object": target_obj_name,
                    "inferred_object": inferred_obj_name,
                    "correct": correct,
                    "argmax_correct": correct,
                    "action": action,
                    "object": obj_name,
                    "inference_time_s": inference_time,
                    "inference_steps": loop_count,
                    "confidence": float(top_prob),
                    "max_prob_true": float(max_prob_seen),
                    "step_at_max_prob": step_at_max_prob,
                    "timeout": False,
                    "tau_crossings": {
                        str(tau): (c if c is None else
                                   [int(c[0]), str(c[1])])
                        for tau, c in tau_crossings.items()
                    },
                })
                if action == "pick" and seq:
                    seq_idx += 1
                goal_reached = True

        if not goal_reached:
            stalled_out = stall_terminated
            elapsed = loop_count / control_rate
            argmax_goal = last_top_goal
            argmax_correct = (argmax_goal == target_id)
            if not quiet:
                kind = "STALL" if stalled_out else "TIMEOUT"
                print(f"    {kind} after {elapsed:.1f}s on {target_id} "
                      f"(argmax={argmax_goal}, p={last_top_prob:.3f})")

            goal_spec, _ = goal_spec_by_id[target_id]
            action = goal_spec["action"]
            obj_name = goal_spec.get("object", state_machine.holding or "")
            mname = state_machine.mujoco_names.get(obj_name, "")
            if (action == "pick" and mname and
                    isinstance(state_machine, PickAndReturnStateMachine)):
                origin = sim.get_object_pos(mname)
                state_machine.save_origin(origin)
            if action in ("place", "pour") and mname:
                sim.teleport_object(mname, target_pose["pos"])
            sim.go_home()
            state_machine.transition(goal_spec)
            steps_completed += 1
            step_times.append({
                "step": steps_completed,
                "goal_id": argmax_goal,
                "target_id": target_id,
                "inferred_id": argmax_goal,
                "target_object": goal_spec.get("object", ""),
                "inferred_object": goal_spec_by_id.get(
                    argmax_goal, ({}, None))[0].get("object", ""),
                "correct": False,
                "argmax_correct": argmax_correct,
                "action": action,
                "object": obj_name,
                "inference_time_s": elapsed,
                "inference_steps": loop_count,
                "confidence": float(last_top_prob),
                "max_prob_true": float(max_prob_seen),
                "step_at_max_prob": step_at_max_prob,
                "timeout": not stalled_out,
                "stalled": stalled_out,
                "tau_crossings": {
                    str(tau): (c if c is None else
                               [int(c[0]), str(c[1])])
                    for tau, c in tau_crossings.items()
                },
            })
            if action == "pick" and seq:
                seq_idx += 1

    if not quiet:
        print(f"\n{'='*60}")
        print(f"  COMPLETE (SE3): {steps_completed} steps")
        if step_times:
            pick_times = [s["inference_time_s"] for s in step_times
                          if s["action"] == "pick" and s["inference_steps"] > 1]
            print(f"  Per-step:")
            for s in step_times:
                print(f"    {s['step']:2d}. {s['action']:5s} {s['object']:15s} "
                      f"t={s['inference_time_s']:5.1f}s  p={s['confidence']:.3f}")
            if pick_times:
                print(f"  Mean pick: {np.mean(pick_times):.2f}s")
        print(f"{'='*60}")

    return {
        "task": task_config["name"],
        "scene": scene_name,
        "intent_mode": intent_mode,
        "inference_model": inference_model,
        "lambda_R": obs_lambda,
        "user_sequence": seq,
        "noise": noise,
        "threshold": threshold,
        "benchmark_mode": benchmark_mode,
        "steps_completed": steps_completed,
        "step_times": step_times,
        "total_inference_time": sum(s["inference_time_s"] for s in step_times)
            if step_times else 0,
        "done": state_machine.is_done(),
    }


def main():
    parser = argparse.ArgumentParser(description="Headless shared autonomy")
    parser.add_argument("task_config",
                        help="Path to SA task YAML")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--inference-model", type=str, default="path_efficiency",
                        choices=["gaussian", "path_efficiency"])
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--max-speed", type=float, default=0.05)
    parser.add_argument("--control-rate", type=float, default=5.0)
    parser.add_argument("--user-sequence", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step logs and result-file writes.")
    parser.add_argument("--intent-mode", type=str, default="2d-center",
                        choices=["2d-center", "3d-center",
                                 "3d-grasp-pos", "se3-grasp"],
                        help="Observer intent mode (default 2d-center; "
                             "non-2d routes to the SE(3) runner).")
    parser.add_argument("--grasp-library", type=str, default=None,
                        help="Path to grasp_poses_3d.yaml (3D modes only).")
    parser.add_argument("--lambda-R", type=float, default=0.04,
                        help="Rotation weight in SE(3) metric.")
    parser.add_argument("--sigma-v", type=float, default=0.5,
                        help="SE(3) Gaussian observer translation sigma.")
    parser.add_argument("--sigma-w", type=float, default=0.5,
                        help="SE(3) Gaussian observer rotation sigma.")
    parser.add_argument("--benchmark-mode", action="store_true",
                        help="SE(3) only: record inferred goals but advance "
                             "the task using the intended target object.")
    args = parser.parse_args()

    if not os.path.isabs(args.task_config):
        args.task_config = os.path.join(PROJECT_ROOT, args.task_config)

    if args.intent_mode == "2d-center":
        run_headless_sa(
            args.task_config, scene_name=args.scene,
            inference_model=args.inference_model, noise=args.noise,
            threshold=args.threshold, sigma=args.sigma, beta=args.beta,
            max_speed=args.max_speed, control_rate=args.control_rate,
            user_sequence=args.user_sequence, seed=args.seed,
            quiet=args.quiet,
        )
    else:
        run_headless_sa_se3(
            args.task_config, scene_name=args.scene,
            inference_model=args.inference_model,
            intent_mode=args.intent_mode,
            grasp_library_path=args.grasp_library,
            noise=args.noise, threshold=args.threshold,
            sigma_v=args.sigma_v, sigma_w=args.sigma_w, beta=args.beta,
            lambda_R=args.lambda_R,
            max_speed=args.max_speed, control_rate=args.control_rate,
            user_sequence=args.user_sequence, seed=args.seed,
            quiet=args.quiet, benchmark_mode=args.benchmark_mode,
        )


if __name__ == "__main__":
    main()
