#!/usr/bin/env python3
"""MAP-Elites Workspace Optimization: Tiered Evaluation Harness

Runs CMA-ME optimization and evaluates baseline vs optimized layouts
across Easy/Medium/Hard environments for a paper submission.

Usage:
    python3 scripts/eval/eval_map_elites_tiers.py
    python3 scripts/eval/eval_map_elites_tiers.py --scenes desk kitchen_prep
    python3 scripts/eval/eval_map_elites_tiers.py --seeds 3 --n-trials 50  # quick test
"""
import sys
import os
import re
import json
import time
import argparse
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from scipy.stats import wilcoxon
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "envopt"))

from intent_separability import (
    intent_separability_objective,
    simulate_goal_inference,
    simulate_multistep_goal_inference,
    parse_task_targets,
)
from task_aware import extract_pick_states, deduplicate_pick_states

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "scene_2d"))
import cv2

# ---------------------------------------------------------------------------
# Environment definitions
# ---------------------------------------------------------------------------

ENVIRONMENTS = OrderedDict([
    ("breakfast_easy", {
        "tier": "Easy",
        "scene": "scene_breakfast_easy",
        "task": "set_table_easy_sa",
        "movable_objects": ["cereal", "banana"],
        "fixed_objects": ["bowl"],
        "iterations": 200,
    }),
    ("six_objects", {
        "tier": "Custom",
        "scene": "scene_six_objects",
        "task": "six_objects_two_groups",
        "movable_objects": [
            "honey_cereal", "frootloop_cereal", "chocolate",
            "milk_carton", "oat_carton", "soy_carton",
        ],
        "fixed_objects": [],
        "iterations": 300,
    }),
    ("desk", {
        "tier": "Medium-A",
        "scene": "scene_desk",
        "task": "desk_organize_v2_sa",
        "movable_objects": ["mug", "stapler", "pen_cup"],
        "fixed_objects": ["book", "phone"],
        "iterations": 500,
    }),
    ("breakfast", {
        "tier": "Medium-B",
        "scene": "scene_breakfast",
        "task": "full_breakfast_sa",
        "movable_objects": ["cereal", "banana", "milk_carton"],
        "fixed_objects": ["bowl", "napkin", "spoon"],
        "iterations": 500,
    }),
    ("kitchen_prep", {
        "tier": "Medium-C",
        "scene": "scene_kitchen_prep",
        "task": "sort_by_zone_sa",
        "movable_objects": ["apple", "can", "bottle"],
        "fixed_objects": ["cutting_board", "sponge"],
        "iterations": 500,
    }),
    ("meal_assembly", {
        "tier": "Hard-A",
        "scene": "scene_meal_assembly",
        "task": "meal_assembly_sa",
        "movable_objects": ["cereal", "banana", "apple", "can", "bottle"],
        "fixed_objects": ["bowl", "cutting_board"],
        "iterations": 200,
    }),
    ("cluttered", {
        "tier": "Hard-B",
        "scene": "scene_cluttered",
        "task": "cluttered_sort_sa",
        "movable_objects": ["red_block", "blue_block", "green_cylinder",
                           "yellow_block", "orange_cylinder", "purple_block",
                           "white_cylinder", "pink_block"],
        "fixed_objects": ["tray_left", "tray_right", "bin_center"],
        "iterations": 200,
    }),
])

# Constants
# EE home position computed from MuJoCo FK with home joint config
# Old value (0.479, -0.060) was a rough 2D approximation that was 222mm off
START_POS = np.array([0.452, 0.160])
SIGMA_BAR = 0.1 * np.eye(2)
SIGMA = 0.08 * np.eye(2)
SIGMA_BAR_INV = np.linalg.inv(SIGMA_BAR)
SIGMA_INV = np.linalg.inv(SIGMA)
TABLE_BOUNDS_X = (0.30, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)
MIN_DIST = 0.12               # legacy fallback when half_extents are absent
MIN_PAIR_CLEARANCE = 0.01     # 1 cm gap added on top of size-based separation
ROBOT_EXCLUSION_RADIUS = 0.15


def size_aware_min_dist(name_a: str, name_b: str,
                        half_extents: Dict[str, list]) -> float:
    """Minimum center-to-center distance for two objects given footprints.

    Uses the mean of (hx, hy) per object so elongated objects (e.g. cereal
    box 90mm x 22mm half-extents) aren't treated as if they were 90mm x 90mm.
    Falls back to the global ``MIN_DIST`` if either object has no entry.
    """
    he_a = half_extents.get(name_a)
    he_b = half_extents.get(name_b)
    if he_a is None or he_b is None:
        return MIN_DIST
    r_a = (float(he_a[0]) + float(he_a[1])) * 0.5
    r_b = (float(he_b[0]) + float(he_b[1])) * 0.5
    return r_a + r_b + MIN_PAIR_CLEARANCE
CONTROL_RATE_HZ = 10.0  # assumed control frequency for time conversion
SECONDS_PER_STEP = 1.0 / CONTROL_RATE_HZ

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scene_positions(scene_name: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Load object positions from scene XML keyframe and body pos attributes."""
    config_path = os.path.join(PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    xml_path = os.path.join(PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["scene"]
    with open(xml_path) as f:
        xml = f.read()

    match = re.search(r'<key\s+name="home"\s+qpos="([^"]+)"', xml)
    qpos = [float(x) for x in match.group(1).strip().split()]

    names = [obj["short_name"] for obj in cfg["objects"]]
    half_extents = {obj["short_name"]: obj["half_extents"] for obj in cfg["objects"]}

    # Count freejoint bodies to correctly index qpos
    # Objects with freejoints get 7 qpos entries; fixed bodies get 0
    positions = {}
    freejoint_idx = 0
    for obj in cfg["objects"]:
        short = obj["short_name"]
        mujoco_name = obj["mujoco_name"]

        # Check if this body has a freejoint in the XML
        body_match = re.search(
            rf'<body\s+name="{re.escape(mujoco_name)}"[^>]*>.*?</body>',
            xml, re.DOTALL)
        has_freejoint = body_match and "freejoint" in body_match.group(0) if body_match else False

        if has_freejoint:
            idx = 9 + freejoint_idx * 7
            if idx + 2 <= len(qpos):
                positions[short] = np.array(qpos[idx:idx+2])
            else:
                positions[short] = np.array([0.5, 0.0])
            freejoint_idx += 1
        else:
            # Fixed body: parse pos from body tag
            pos_match = re.search(
                rf'<body\s+name="{re.escape(mujoco_name)}"\s+pos="([^"]+)"', xml)
            if pos_match:
                pos = [float(x) for x in pos_match.group(1).split()]
                positions[short] = np.array(pos[:2])
            else:
                positions[short] = np.array([0.5, 0.0])

    return names, positions, half_extents


def load_task_config(task_name: str) -> dict:
    """Load task YAML config."""
    path = os.path.join(PROJECT_ROOT, "config", "tasks", f"{task_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def extract_linear_path_from_state_machine(task_config: dict,
                                           object_positions: Dict[str, np.ndarray]) -> List[dict]:
    """Extract a single linear path through a state-machine task.

    Follows the first valid goal at each state to produce a sequential
    list of pick/place targets for multi-step evaluation.
    """
    task = task_config["task"]

    # If task has explicit steps, use them
    if "steps" in task:
        return parse_task_targets(task_config, object_positions)

    # Otherwise, traverse the state machine
    states = task["states"]
    targets = []
    current_state = "initial"
    visited = set()

    while current_state and current_state != "done" and current_state not in visited:
        visited.add(current_state)
        state_data = states.get(current_state, {})
        goals = state_data.get("valid_goals", [])
        if not goals:
            break

        goal = goals[0]  # Follow first branch
        action = goal["action"]

        if action == "pick":
            obj_name = goal["object"]
            if obj_name in object_positions:
                pos = object_positions[obj_name][:2] if len(object_positions[obj_name]) > 2 else object_positions[obj_name]
                targets.append({
                    "action": "pick",
                    "target_name": obj_name,
                    "target_pos": np.array(pos),
                })
        elif action in ("place", "pour"):
            dest = goal.get("destination", {})
            if "reference" in dest:
                ref = dest["reference"]
                ref_pos = object_positions.get(ref, np.array([0.5, 0.0]))
                if len(ref_pos) > 2:
                    ref_pos = ref_pos[:2]
                offset = np.array([dest["offset"]["x"], dest["offset"]["y"]])
                pos = ref_pos + offset
            elif "absolute" in dest:
                pos = np.array([dest["absolute"]["x"], dest["absolute"]["y"]])
            else:
                pos = np.array([0.5, 0.0])
            targets.append({
                "action": action,
                "target_name": f"{action}_near_{dest.get('reference', 'target')}",
                "target_pos": pos,
            })

        current_state = goal.get("next_state", None)

    return targets


# ---------------------------------------------------------------------------
# Random layout generation
# ---------------------------------------------------------------------------

def generate_random_layouts(n_layouts: int, movable_names: List[str],
                           fixed_positions: Dict[str, np.ndarray],
                           half_extents: Dict[str, list],
                           seed: int = 0) -> List[Dict[str, np.ndarray]]:
    """Generate collision-free random layouts avoiding robot position."""
    rng = np.random.default_rng(seed)
    layouts = []

    for _ in range(n_layouts * 10):  # try many times
        if len(layouts) >= n_layouts:
            break
        positions = dict(fixed_positions)
        placed = list(fixed_positions.values())
        valid = True

        # Track names alongside placed positions so size-aware separation
        # can look up half-extents per pair.
        placed_names = list(fixed_positions.keys())

        for name in movable_names:
            success = False
            for _ in range(200):
                x = rng.uniform(TABLE_BOUNDS_X[0] + 0.05, TABLE_BOUNDS_X[1] - 0.05)
                y = rng.uniform(TABLE_BOUNDS_Y[0] + 0.05, TABLE_BOUNDS_Y[1] - 0.05)
                pos = np.array([x, y])

                # Check robot exclusion
                if np.linalg.norm(pos - START_POS) < ROBOT_EXCLUSION_RADIUS:
                    continue

                # Check size-aware min distance to all placed objects
                too_close = False
                for p, pname in zip(placed, placed_names):
                    min_d = size_aware_min_dist(name, pname, half_extents)
                    if np.linalg.norm(pos - p) < min_d:
                        too_close = True
                        break
                if too_close:
                    continue

                positions[name] = pos
                placed.append(pos)
                placed_names.append(name)
                success = True
                break

            if not success:
                valid = False
                break

        if valid:
            layouts.append(positions)

    return layouts[:n_layouts]


# ---------------------------------------------------------------------------
# Feature 2 candidates
# ---------------------------------------------------------------------------

def feature_nn_var(movable_pos: np.ndarray) -> float:
    """Nearest-neighbor distance variance."""
    M = len(movable_pos)
    if M < 2:
        return 0.0
    nn_dists = []
    for i in range(M):
        min_d = min(np.linalg.norm(movable_pos[i] - movable_pos[j])
                    for j in range(M) if j != i)
        nn_dists.append(min_d)
    return float(np.var(nn_dists))


def feature_hull_area(movable_pos: np.ndarray) -> float:
    """Convex hull area of movable object positions."""
    M = len(movable_pos)
    if M < 3:
        # Degenerate: return distance between 2 points as proxy
        if M == 2:
            return float(np.linalg.norm(movable_pos[0] - movable_pos[1]))
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(movable_pos)
        return float(hull.volume)  # 2D: volume = area
    except Exception:
        return 0.0


def feature_centroid_offset(movable_pos: np.ndarray) -> float:
    """Distance from layout centroid to table center (0.6, 0.0)."""
    centroid = movable_pos.mean(axis=0)
    return float(np.linalg.norm(centroid - np.array([0.6, 0.0])))


def feature_bbox_aspect(movable_pos: np.ndarray) -> float:
    """Aspect ratio of axis-aligned bounding box."""
    ptp = np.ptp(movable_pos, axis=0)  # [width, height]
    return float(max(ptp) / max(min(ptp), 1e-6))


FEATURE2_FUNCTIONS = {
    "nn_var": feature_nn_var,
    "hull_area": feature_hull_area,
    "centroid_offset": feature_centroid_offset,
    "bbox_aspect": feature_bbox_aspect,
}


def compute_features(movable_pos: np.ndarray, feature2_name: str) -> Tuple[float, float]:
    """Compute (mean_pairwise_dist, feature2) for a layout."""
    M = len(movable_pos)
    dists = []
    for i in range(M):
        for j in range(i+1, M):
            dists.append(np.linalg.norm(movable_pos[i] - movable_pos[j]))
    mean_dist = float(np.mean(dists)) if dists else 0.0
    f2 = FEATURE2_FUNCTIONS[feature2_name](movable_pos)
    return mean_dist, f2


# ---------------------------------------------------------------------------
# Feature range calibration
# ---------------------------------------------------------------------------

def calibrate_feature_ranges(movable_names: List[str],
                             fixed_positions: Dict[str, np.ndarray],
                             half_extents: Dict[str, list],
                             feature2_name: str,
                             n_samples: int = 1000,
                             seed: int = 0) -> List[Tuple[float, float]]:
    """Sample random layouts and compute [p5, p95] for each feature."""
    layouts = generate_random_layouts(
        n_samples, movable_names, fixed_positions, half_extents, seed=seed)

    f1_vals = []
    f2_vals = []
    for layout in layouts:
        pos = np.array([layout[n] for n in movable_names])
        f1, f2 = compute_features(pos, feature2_name)
        f1_vals.append(f1)
        f2_vals.append(f2)

    f1_vals = np.array(f1_vals)
    f2_vals = np.array(f2_vals)

    f1_range = (float(np.percentile(f1_vals, 5)), float(np.percentile(f1_vals, 95)))
    f2_range = (float(np.percentile(f2_vals, 5)), float(np.percentile(f2_vals, 95)))

    # Ensure non-zero range
    if f1_range[1] - f1_range[0] < 0.01:
        f1_range = (f1_range[0] - 0.05, f1_range[1] + 0.05)
    if f2_range[1] - f2_range[0] < 1e-4:
        f2_range = (f2_range[0] - 0.01, f2_range[1] + 0.01)

    return [f1_range, f2_range]


# ---------------------------------------------------------------------------
# Legibility objective (Dragan's path efficiency model)
# ---------------------------------------------------------------------------

def trajectory_margin_objective(object_positions: np.ndarray,
                                start_pos: np.ndarray,
                                beta: float = 5.0,
                                sigma_u: float = 0.03,
                                dt: float = 0.05,
                                step_size: float = 0.01,
                                alpha: float = 0.05) -> float:
    """Compute worst-case linearized Boltzmann margin slack.

    Combines Gaussian joystick noise with Boltzmann path efficiency:
    1. Simulate nominal trajectory toward each goal
    2. Compute cost difference between true and alternative goals
    3. Compute gradient of cost difference w.r.t. trajectory (via finite diff)
    4. Compute trajectory noise covariance from joystick noise
    5. Return slack = m_g - Phi_inv(1-alpha_g) * sqrt(v_g)

    Higher slack = more robust goal identification.
    """
    from scipy.stats import norm as sp_norm

    M = object_positions.shape[0]
    if M < 2:
        return 0.0

    alpha_g = alpha / (M - 1)
    quantile = sp_norm.ppf(1 - alpha_g)
    worst_slack = float('inf')

    for star_idx in range(M):
        true_goal = object_positions[star_idx]
        d_start_true = np.linalg.norm(true_goal - start_pos)
        if d_start_true < 0.02:
            worst_slack = min(worst_slack, -100.0)
            continue

        # Simulate nominal straight-line trajectory
        n_steps = min(int(np.ceil(d_start_true / step_size)), 200)
        direction = (true_goal - start_pos) / d_start_true
        nominal_traj = np.array([start_pos + i * step_size * direction
                                 for i in range(n_steps + 1)])
        # Clamp final point to goal
        nominal_traj[-1] = true_goal

        # Path length of nominal trajectory
        L_nom = sum(np.linalg.norm(nominal_traj[i+1] - nominal_traj[i])
                    for i in range(len(nominal_traj) - 1))

        # Compute cost for each goal on nominal trajectory
        def path_cost(traj, goal):
            L = sum(np.linalg.norm(traj[i+1] - traj[i])
                    for i in range(len(traj) - 1))
            d_rem = np.linalg.norm(traj[-1] - goal)
            d_st = max(np.linalg.norm(traj[0] - goal), 0.02)
            return (L + d_rem) / d_st

        c_star = path_cost(nominal_traj, true_goal)

        for alt_idx in range(M):
            if alt_idx == star_idx:
                continue
            alt_goal = object_positions[alt_idx]

            c_alt = path_cost(nominal_traj, alt_goal)
            delta_c = c_alt - c_star

            # Mean margin
            m_g = beta * delta_c  # uniform prior: log(P(g*)/P(g)) = 0

            # Compute gradient via finite differences on the final position
            # (the most informative point of the trajectory)
            eps = 1e-4
            grad = np.zeros(2)
            for dim in range(2):
                traj_plus = nominal_traj.copy()
                traj_plus[-1, dim] += eps
                traj_minus = nominal_traj.copy()
                traj_minus[-1, dim] -= eps
                dc_plus = path_cost(traj_plus, alt_goal) - path_cost(traj_plus, true_goal)
                dc_minus = path_cost(traj_minus, alt_goal) - path_cost(traj_minus, true_goal)
                grad[dim] = (dc_plus - dc_minus) / (2 * eps)

            # Trajectory noise variance at final position
            # For integrator dynamics: var(x_T) = T * sigma_u^2 * dt^2
            traj_var = n_steps * (sigma_u ** 2) * (dt ** 2)

            # Margin variance: beta^2 * grad^T * Sigma_xi * grad
            # Sigma_xi at final point is traj_var * I
            v_g = (beta ** 2) * traj_var * np.dot(grad, grad)

            # Slack
            if v_g > 0:
                slack = m_g - quantile * np.sqrt(v_g)
            else:
                slack = m_g

            worst_slack = min(worst_slack, slack)

    return worst_slack


def legibility_objective(object_positions: np.ndarray,
                         start_pos: np.ndarray,
                         beta: float = 5.0,
                         T: int = 200,
                         step_size: float = 0.01) -> float:
    """Compute negative worst-case legibility cost across all goals.

    For each true goal g*, simulates a straight-line trajectory and
    computes the Boltzmann path-efficiency posterior at each step.
    Returns the negative max cost (= negative worst legibility loss)
    so that higher fitness = more legible layout.

    Uses Dragan's model:
        P(g | traj) ∝ exp(-β · (L + d_remaining) / d_start)
    """
    M = object_positions.shape[0]
    if M < 2:
        return 0.0

    total_loss = 0.0

    for star_idx in range(M):
        true_goal = object_positions[star_idx]
        d_start_true = np.linalg.norm(true_goal - start_pos)
        if d_start_true < 1e-6:
            continue

        # Simulate trajectory toward true goal
        pos = start_pos.copy()
        path_length = 0.0

        # Evaluate at 25%, 50%, 75% of estimated trajectory length
        n_steps = min(int(np.ceil(d_start_true / step_size)), T)
        checkpoints = [max(1, int(p * n_steps)) for p in [0.25, 0.5, 0.75]]

        for t in range(1, n_steps + 1):
            direction = true_goal - pos
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                break
            step = min(step_size, dist)
            pos = pos + (direction / dist) * step
            path_length += step

            if t not in checkpoints:
                continue

            # Compute Boltzmann posterior at this checkpoint
            scores = np.zeros(M)
            for g in range(M):
                d_start_g = np.linalg.norm(object_positions[g] - start_pos)
                if d_start_g < 1e-6:
                    scores[g] = 0.0
                    continue
                d_remaining_g = np.linalg.norm(object_positions[g] - pos)
                cost = (path_length + d_remaining_g) / d_start_g
                scores[g] = -beta * cost

            # Softmax
            scores -= np.max(scores)
            exp_scores = np.exp(scores)
            posterior = exp_scores / exp_scores.sum()

            # Cross-entropy loss for this goal at this checkpoint
            p_true = max(posterior[star_idx], 1e-8)
            percent = checkpoints.index(t)
            weight = [0.25, 0.5, 0.75][percent]
            total_loss += -np.log(p_true) * weight

    # Return negative loss (higher = better legibility)
    return -total_loss


# ---------------------------------------------------------------------------
# CMA-ME Optimization
# ---------------------------------------------------------------------------

def run_cma_me(movable_names: List[str],
               fixed_positions: Dict[str, np.ndarray],
               pick_states: List[Tuple[str, List[str]]],
               iterations: int,
               seed: int,
               log_interval: int = 50,
               feature2_name: str = "nn_var",
               measure_ranges: Optional[List[Tuple[float, float]]] = None,
               objective: str = "separability",
               half_extents: Optional[Dict[str, list]] = None) -> dict:
    """Run CMA-ME MAP-Elites optimization for intent separability.

    Returns dict with best_positions, top5_positions, archive_stats, history.
    """
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler

    M = len(movable_names)
    solution_dim = M * 2  # (x, y) per movable object
    if half_extents is None:
        half_extents = {}

    # Initial positions: spread evenly
    rng = np.random.default_rng(seed)
    x0 = np.zeros(solution_dim)
    for i in range(M):
        x0[i*2] = rng.uniform(TABLE_BOUNDS_X[0] + 0.1, TABLE_BOUNDS_X[1] - 0.1)
        x0[i*2+1] = rng.uniform(TABLE_BOUNDS_Y[0] + 0.1, TABLE_BOUNDS_Y[1] - 0.1)

    # Bounds
    lower = np.array([TABLE_BOUNDS_X[0], TABLE_BOUNDS_Y[0]] * M)
    upper = np.array([TABLE_BOUNDS_X[1], TABLE_BOUNDS_Y[1]] * M)

    # Feature ranges: use provided calibrated ranges or fallback
    if measure_ranges is None:
        if M <= 2:
            measure_ranges = [(0.05, 0.55), (0.0, 0.6)]
        elif M <= 4:
            measure_ranges = [(0.05, 0.50), (0.0, 0.5)]
        else:
            measure_ranges = [(0.05, 0.40), (0.0, 0.4)]

    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[20, 20],
        ranges=measure_ranges,
    )

    emitters = [
        EvolutionStrategyEmitter(
            archive, x0=x0, sigma0=0.08,
            ranker="2imp",
            bounds=list(zip(lower, upper)),
            batch_size=30,
        )
        for _ in range(3)
    ]

    scheduler = Scheduler(archive, emitters)

    history = {"iterations": [], "best_fitness": [], "coverage": [], "qd_score": []}
    name_to_idx = {n: i for i, n in enumerate(movable_names)}

    for itr in range(1, iterations + 1):
        sols = scheduler.ask()
        objs = []
        measures = []

        for sol in sols:
            # Reconstruct positions dict
            positions = {}
            for i, name in enumerate(movable_names):
                positions[name] = sol[i*2:(i+1)*2]

            # Evaluate fitness based on objective type
            if objective == "separability":
                worst_slack = float('inf')
                for _, competing in pick_states:
                    indices = [name_to_idx[o] for o in competing if o in name_to_idx]
                    if len(indices) < 2:
                        continue
                    state_pos = np.array([sol[i*2:(i+1)*2] for i in indices])
                    slack, _ = intent_separability_objective(
                        state_pos, START_POS, SIGMA_BAR_INV,
                        T=50, step_size=0.01, alpha=0.05,
                    )
                    worst_slack = min(worst_slack, slack)
                raw_fitness = worst_slack
            elif objective == "legibility":
                worst_leg = float('inf')
                for _, competing in pick_states:
                    indices = [name_to_idx[o] for o in competing if o in name_to_idx]
                    if len(indices) < 2:
                        continue
                    state_pos = np.array([sol[i*2:(i+1)*2] for i in indices])
                    leg = legibility_objective(
                        state_pos, START_POS, beta=5.0,
                        T=200, step_size=0.01,
                    )
                    worst_leg = min(worst_leg, leg)
                raw_fitness = worst_leg
            elif objective == "trajectory_margin":
                worst_tm = float('inf')
                for _, competing in pick_states:
                    indices = [name_to_idx[o] for o in competing if o in name_to_idx]
                    if len(indices) < 2:
                        continue
                    state_pos = np.array([sol[i*2:(i+1)*2] for i in indices])
                    tm = trajectory_margin_objective(
                        state_pos, START_POS, beta=5.0,
                        sigma_u=0.03, dt=0.05, step_size=0.01,
                        alpha=0.05,
                    )
                    worst_tm = min(worst_tm, tm)
                raw_fitness = worst_tm
            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Size-aware overlap penalty (object-object + object-fixed).
            # Per-pair minimum distance scales with each object's footprint
            # so elongated boxes don't get treated as squares.
            penalty = 0.0
            all_names_pen = list(positions.keys()) + list(fixed_positions.keys())
            all_pos = list(positions.values()) + list(fixed_positions.values())
            for i in range(len(all_pos)):
                for j in range(i + 1, len(all_pos)):
                    min_d = size_aware_min_dist(
                        all_names_pen[i], all_names_pen[j], half_extents)
                    d = np.linalg.norm(all_pos[i] - all_pos[j])
                    if d < min_d:
                        penalty += (min_d - d) ** 2

            # Robot exclusion: strongly penalize objects near EE home
            for pos_val in positions.values():
                d_ee = np.linalg.norm(pos_val - START_POS)
                if d_ee < ROBOT_EXCLUSION_RADIUS:
                    penalty += 10.0 * (ROBOT_EXCLUSION_RADIUS - d_ee) ** 2

            fitness = raw_fitness - 100.0 * penalty

            # Features
            movable_pos = np.array([sol[i*2:(i+1)*2] for i in range(M)])
            f1, f2 = compute_features(movable_pos, feature2_name)

            objs.append(fitness)
            measures.append([f1, f2])

        scheduler.tell(objs, measures)

        if itr % log_interval == 0 or itr == iterations:
            history["iterations"].append(itr)
            history["best_fitness"].append(float(archive.stats.obj_max))
            history["coverage"].append(float(archive.stats.coverage))
            history["qd_score"].append(float(archive.stats.qd_score))

    # Extract best and top-5
    best_elite = archive.best_elite
    best_sol = best_elite["solution"]
    best_positions = {}
    for i, name in enumerate(movable_names):
        best_positions[name] = best_sol[i*2:(i+1)*2].copy()

    # Top-5: get all elites sorted by fitness, pick top 5 from different cells
    all_elites = list(archive)
    all_elites.sort(key=lambda e: e["objective"], reverse=True)
    top5_positions = []
    seen_cells = set()
    for elite in all_elites:
        cell = int(elite["index"])
        if cell not in seen_cells:
            seen_cells.add(cell)
            pos = {}
            for i, name in enumerate(movable_names):
                pos[name] = elite["solution"][i*2:(i+1)*2].copy()
            top5_positions.append(pos)
            if len(top5_positions) >= 5:
                break

    archive_stats = {
        "num_elites": int(archive.stats.num_elites),
        "coverage": float(archive.stats.coverage),
        "qd_score": float(archive.stats.qd_score),
        "best_fitness": float(archive.stats.obj_max),
        "mean_fitness": float(archive.stats.obj_mean),
    }

    return {
        "best_positions": best_positions,
        "top5_positions": top5_positions,
        "archive_stats": archive_stats,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_single_step(object_positions: np.ndarray,
                         n_trials: int = 200,
                         seed: int = 0) -> dict:
    """Single-step goal ID evaluation."""
    M = object_positions.shape[0]
    rng = np.random.default_rng(seed)
    per_goal_correct = np.zeros(M)
    per_goal_steps = np.zeros(M)
    all_steps = []

    for g_star in range(M):
        trial_steps = []
        correct = 0
        for _ in range(n_trials):
            pred, steps = simulate_goal_inference(
                object_positions, START_POS, g_star,
                SIGMA_BAR_INV, SIGMA,
                T=200, step_size=0.01,
                confidence_threshold=0.95, rng=rng,
            )
            if pred == g_star:
                correct += 1
            trial_steps.append(steps)
        per_goal_correct[g_star] = correct / n_trials
        per_goal_steps[g_star] = np.mean(trial_steps)
        all_steps.extend(trial_steps)

    mean_steps = float(np.mean(per_goal_steps))
    worst_steps = float(np.percentile(all_steps, 95))
    return {
        "accuracy": float(np.mean(per_goal_correct)),
        "mean_steps": mean_steps,
        "worst_case_steps": worst_steps,
        "mean_time_s": mean_steps * SECONDS_PER_STEP,
        "worst_case_time_s": worst_steps * SECONDS_PER_STEP,
        "per_goal_accuracy": per_goal_correct.tolist(),
        "per_goal_mean_steps": per_goal_steps.tolist(),
        "per_goal_mean_time_s": (per_goal_steps * SECONDS_PER_STEP).tolist(),
    }


def evaluate_multi_step(task_targets: List[dict],
                        all_object_positions: np.ndarray,
                        n_trials: int = 200,
                        seed: int = 0) -> dict:
    """Multi-step task evaluation."""
    rng = np.random.default_rng(seed)
    all_total_steps = []
    all_accuracies = []
    n_steps = len(task_targets)
    per_step_steps_acc = [[] for _ in range(n_steps)]
    per_step_correct_acc = [[] for _ in range(n_steps)]

    for _ in range(n_trials):
        result = simulate_multistep_goal_inference(
            task_targets, all_object_positions, START_POS,
            SIGMA_BAR_INV, SIGMA,
            T_per_step=200, step_size=0.01,
            confidence_threshold=0.95, rng=rng,
        )
        all_total_steps.append(result["total_steps"])
        all_accuracies.append(result["accuracy"])
        for s in range(min(n_steps, len(result["per_step_steps"]))):
            per_step_steps_acc[s].append(result["per_step_steps"][s])
            per_step_correct_acc[s].append(result["per_step_correct"][s])

    total_steps = float(np.mean(all_total_steps))
    per_step_ms = [float(np.mean(s)) if s else 0.0 for s in per_step_steps_acc]
    return {
        "total_steps": total_steps,
        "total_time_s": total_steps * SECONDS_PER_STEP,
        "accuracy": float(np.mean(all_accuracies)),
        "per_step_mean_steps": per_step_ms,
        "per_step_mean_time_s": [s * SECONDS_PER_STEP for s in per_step_ms],
        "per_step_accuracy": [float(np.mean(c)) if c else 0.0 for c in per_step_correct_acc],
        "n_steps": n_steps,
        "step_targets": [t["target_name"] for t in task_targets],
    }


def compute_separability_slack(object_positions: np.ndarray) -> float:
    """Compute analytical worst-case separability slack."""
    if object_positions.shape[0] < 2:
        return float('inf')
    slack, _ = intent_separability_objective(
        object_positions, START_POS, SIGMA_BAR_INV,
        T=50, step_size=0.01, alpha=0.05,
    )
    return float(slack)


def compute_layout_metrics(positions: Dict[str, np.ndarray]) -> dict:
    """Compute secondary layout quality metrics."""
    pos_arr = np.array(list(positions.values()))
    M = len(pos_arr)

    # Pairwise distances
    dists = []
    for i in range(M):
        for j in range(i+1, M):
            dists.append(np.linalg.norm(pos_arr[i] - pos_arr[j]))

    mean_dist = float(np.mean(dists)) if dists else 0.0
    min_dist = float(np.min(dists)) if dists else 0.0

    # Angular coverage from start position
    deltas = pos_arr - START_POS
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    sorted_angles = np.sort(angles)
    if len(sorted_angles) > 1:
        diffs = np.diff(sorted_angles)
        wrap = (sorted_angles[0] + 2*np.pi) - sorted_angles[-1]
        diffs = np.append(diffs, wrap)
        angular_std = float(np.std(diffs))
    else:
        angular_std = 0.0

    return {
        "mean_pairwise_dist": mean_dist,
        "min_pairwise_dist": min_dist,
        "angular_spread_std": angular_std,
    }


def evaluate_layout(positions: Dict[str, np.ndarray],
                    movable_names: List[str],
                    pick_states: List[Tuple[str, List[str]]],
                    task_targets: List[dict],
                    all_names: List[str],
                    n_trials: int = 200,
                    eval_seed: int = 42) -> dict:
    """Full evaluation of a single layout."""
    # Build arrays for pick objects (for single-step eval)
    pick_objects = set()
    for _, competing in pick_states:
        pick_objects.update(competing)
    pick_names = sorted(pick_objects & set(positions.keys()))

    if len(pick_names) >= 2:
        pick_pos = np.array([positions[n] for n in pick_names])
        single_step = evaluate_single_step(pick_pos, n_trials=n_trials, seed=eval_seed)
        single_step["object_names"] = pick_names
        sep_slack = compute_separability_slack(pick_pos)
    else:
        single_step = {"accuracy": 1.0, "mean_steps": 0, "worst_case_steps": 0,
                       "per_goal_accuracy": [], "per_goal_mean_steps": []}
        sep_slack = float('inf')

    # Multi-step evaluation
    all_pos = np.array([positions[n] for n in all_names if n in positions])
    if task_targets:
        multi_step = evaluate_multi_step(task_targets, all_pos,
                                        n_trials=n_trials, seed=eval_seed)
    else:
        multi_step = {"total_steps": 0, "accuracy": 1.0, "n_steps": 0,
                      "per_step_mean_steps": [], "per_step_accuracy": [],
                      "step_targets": []}

    layout_metrics = compute_layout_metrics(positions)

    return {
        "single_step": single_step,
        "multi_step": multi_step,
        "separability_slack": sep_slack,
        "layout_metrics": layout_metrics,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(env_key: str, env_config: dict,
                   n_seeds: int = 5, n_trials: int = 200,
                   n_random: int = 10,
                   feature2_name: str = "nn_var",
                   calibrate: bool = True,
                   objective: str = "separability") -> dict:
    """Run full experiment for one environment."""
    print(f"\n{'='*70}")
    print(f"Environment: {env_key} ({env_config['tier']})")
    print(f"{'='*70}")

    # Load scene and task
    scene_name = env_config["scene"]
    task_name = env_config["task"]
    movable_names = env_config["movable_objects"]
    fixed_names = env_config["fixed_objects"]
    iterations = env_config["iterations"]

    all_names, all_positions, half_extents = load_scene_positions(scene_name)
    task_config = load_task_config(task_name)

    # Extract pick states from state machine
    task_data = task_config["task"]
    if "states" in task_data:
        pick_states_raw = extract_pick_states(task_data)
        pick_states = deduplicate_pick_states(pick_states_raw)
    else:
        pick_states = []

    print(f"  Scene: {scene_name}, Task: {task_name}")
    print(f"  Objects: {all_names}")
    print(f"  Movable: {movable_names}, Fixed: {fixed_names}")
    print(f"  Pick states: {len(pick_states)}")
    for sn, objs in pick_states:
        print(f"    {sn}: {objs}")

    # Build fixed positions
    fixed_positions = {n: all_positions[n] for n in fixed_names if n in all_positions}

    # Extract task targets for multi-step eval
    task_targets = extract_linear_path_from_state_machine(task_config, all_positions)
    print(f"  Task path: {len(task_targets)} steps")
    for t in task_targets:
        print(f"    {t['action']}: {t['target_name']} -> {t['target_pos']}")

    results = {
        "env_key": env_key,
        "tier": env_config["tier"],
        "scene": scene_name,
        "task": task_name,
        "n_objects": len(all_names),
        "n_movable": len(movable_names),
        "n_picks": max((len(objs) for _, objs in pick_states), default=0),
        "iterations": iterations,
    }

    # --- Baseline evaluation ---
    print(f"\n  Evaluating baseline...")
    baseline_eval = evaluate_layout(
        all_positions, movable_names, pick_states, task_targets, all_names,
        n_trials=n_trials, eval_seed=42,
    )
    results["baseline"] = baseline_eval
    print(f"    Accuracy: {baseline_eval['single_step']['accuracy']:.2%}")
    print(f"    Time: {baseline_eval['single_step']['mean_time_s']:.2f}s "
          f"({baseline_eval['single_step']['mean_steps']:.1f} steps)")
    print(f"    Slack: {baseline_eval['separability_slack']:.2f}")

    # --- Random baselines ---
    print(f"\n  Generating {n_random} random layouts...")
    random_layouts = generate_random_layouts(
        n_random, movable_names, fixed_positions, half_extents, seed=123,
    )
    print(f"    Generated {len(random_layouts)} valid layouts")

    random_evals = []
    for i, layout in enumerate(random_layouts):
        # Merge with fixed positions
        full_pos = dict(fixed_positions)
        full_pos.update(layout)
        ev = evaluate_layout(
            full_pos, movable_names, pick_states, task_targets, all_names,
            n_trials=n_trials, eval_seed=42+i,
        )
        random_evals.append(ev)

    if random_evals:
        rnd_steps = [e["single_step"]["mean_steps"] for e in random_evals]
        results["random"] = {
            "n_layouts": len(random_evals),
            "mean_accuracy": float(np.mean([e["single_step"]["accuracy"] for e in random_evals])),
            "mean_steps": float(np.mean(rnd_steps)),
            "mean_time_s": float(np.mean(rnd_steps)) * SECONDS_PER_STEP,
            "mean_slack": float(np.mean([e["separability_slack"] for e in random_evals])),
            "mean_total_task_steps": float(np.mean([e["multi_step"]["total_steps"] for e in random_evals])),
            "std_accuracy": float(np.std([e["single_step"]["accuracy"] for e in random_evals])),
            "std_steps": float(np.std(rnd_steps)),
        }
        print(f"    Random mean acc: {results['random']['mean_accuracy']:.2%}")
        print(f"    Random mean time: {results['random']['mean_time_s']:.2f}s "
              f"({results['random']['mean_steps']:.1f} steps)")

    # --- Calibrate feature ranges ---
    cal_ranges = None
    if calibrate:
        print(f"\n  Calibrating feature ranges (feature2={feature2_name})...")
        cal_ranges = calibrate_feature_ranges(
            movable_names, fixed_positions, half_extents,
            feature2_name=feature2_name, n_samples=1000, seed=99,
        )
        print(f"    Feature 1 (mean pairwise dist): [{cal_ranges[0][0]:.4f}, {cal_ranges[0][1]:.4f}]")
        print(f"    Feature 2 ({feature2_name}): [{cal_ranges[1][0]:.4f}, {cal_ranges[1][1]:.4f}]")
        results["calibrated_ranges"] = {
            "feature1": list(cal_ranges[0]),
            "feature2": list(cal_ranges[1]),
            "feature2_name": feature2_name,
        }

    # --- MAP-Elites optimization (multiple seeds) ---
    me_results_per_seed = []
    for seed_idx in range(n_seeds):
        seed = seed_idx * 100 + 42
        print(f"\n  MAP-Elites seed {seed_idx+1}/{n_seeds} (seed={seed})...")
        t0 = time.time()

        me_result = run_cma_me(
            movable_names, fixed_positions, pick_states,
            iterations=iterations, seed=seed, log_interval=50,
            feature2_name=feature2_name, measure_ranges=cal_ranges,
            objective=objective,
            half_extents=half_extents,
        )
        opt_time = time.time() - t0
        print(f"    Time: {opt_time:.1f}s")
        print(f"    Archive: {me_result['archive_stats']['num_elites']} elites, "
              f"coverage={me_result['archive_stats']['coverage']:.3f}")
        print(f"    Best fitness: {me_result['archive_stats']['best_fitness']:.2f}")

        # Evaluate best elite
        best_full = dict(fixed_positions)
        best_full.update(me_result["best_positions"])
        # Recompute task targets with optimized positions
        opt_task_targets = extract_linear_path_from_state_machine(task_config, best_full)
        best_eval = evaluate_layout(
            best_full, movable_names, pick_states, opt_task_targets, all_names,
            n_trials=n_trials, eval_seed=42,
        )
        print(f"    Best acc: {best_eval['single_step']['accuracy']:.2%}, "
              f"time: {best_eval['single_step']['mean_time_s']:.2f}s, "
              f"slack: {best_eval['separability_slack']:.2f}")

        # Evaluate top-5
        top5_evals = []
        for t5_pos in me_result["top5_positions"]:
            t5_full = dict(fixed_positions)
            t5_full.update(t5_pos)
            t5_targets = extract_linear_path_from_state_machine(task_config, t5_full)
            t5_eval = evaluate_layout(
                t5_full, movable_names, pick_states, t5_targets, all_names,
                n_trials=n_trials, eval_seed=42,
            )
            top5_evals.append(t5_eval)

        me_results_per_seed.append({
            "seed": seed,
            "opt_time": opt_time,
            "archive_stats": me_result["archive_stats"],
            "history": me_result["history"],
            "best_eval": best_eval,
            "best_positions": {k: v.tolist() for k, v in me_result["best_positions"].items()},
            "top5_mean_accuracy": float(np.mean([e["single_step"]["accuracy"] for e in top5_evals])),
            "top5_mean_steps": float(np.mean([e["single_step"]["mean_steps"] for e in top5_evals])),
            "top5_mean_slack": float(np.mean([e["separability_slack"] for e in top5_evals])),
        })

    # Aggregate ME results across seeds
    best_accs = [r["best_eval"]["single_step"]["accuracy"] for r in me_results_per_seed]
    best_steps = [r["best_eval"]["single_step"]["mean_steps"] for r in me_results_per_seed]
    best_slacks = [r["best_eval"]["separability_slack"] for r in me_results_per_seed]
    best_task_steps = [r["best_eval"]["multi_step"]["total_steps"] for r in me_results_per_seed]

    best_times = [s * SECONDS_PER_STEP for s in best_steps]
    results["map_elites"] = {
        "n_seeds": n_seeds,
        "per_seed": me_results_per_seed,
        "best_mean_accuracy": float(np.mean(best_accs)),
        "best_std_accuracy": float(np.std(best_accs)),
        "best_mean_steps": float(np.mean(best_steps)),
        "best_std_steps": float(np.std(best_steps)),
        "best_mean_time_s": float(np.mean(best_times)),
        "best_std_time_s": float(np.std(best_times)),
        "best_mean_slack": float(np.mean(best_slacks)),
        "best_std_slack": float(np.std(best_slacks)),
        "best_mean_task_steps": float(np.mean(best_task_steps)),
        "best_std_task_steps": float(np.std(best_task_steps)),
        "best_mean_task_time_s": float(np.mean(best_task_steps)) * SECONDS_PER_STEP,
        "mean_opt_time": float(np.mean([r["opt_time"] for r in me_results_per_seed])),
        "mean_coverage": float(np.mean([r["archive_stats"]["coverage"] for r in me_results_per_seed])),
        "mean_num_elites": float(np.mean([r["archive_stats"]["num_elites"] for r in me_results_per_seed])),
    }

    # Speedup
    baseline_steps = results["baseline"]["single_step"]["mean_steps"]
    baseline_task = results["baseline"]["multi_step"]["total_steps"]
    me_steps = results["map_elites"]["best_mean_steps"]
    me_task = results["map_elites"]["best_mean_task_steps"]

    results["speedup_single"] = baseline_steps / max(me_steps, 0.1)
    results["speedup_task"] = baseline_task / max(me_task, 0.1) if baseline_task > 0 else 1.0

    # Statistical test (Wilcoxon) — baseline vs ME best across seeds
    if n_seeds >= 5:
        baseline_rep = [baseline_eval["single_step"]["mean_steps"]] * n_seeds
        try:
            stat, p_val = wilcoxon(baseline_rep, best_steps, alternative='greater')
            results["wilcoxon_p"] = float(p_val)
        except Exception:
            results["wilcoxon_p"] = None
    else:
        results["wilcoxon_p"] = None

    print(f"\n  --- Summary for {env_key} ---")
    bl_time = baseline_eval['single_step']['mean_time_s']
    me_time = results['map_elites']['best_mean_time_s']
    print(f"  Baseline: acc={baseline_eval['single_step']['accuracy']:.2%}, "
          f"time={bl_time:.2f}s, slack={baseline_eval['separability_slack']:.2f}")
    print(f"  ME Best:  acc={results['map_elites']['best_mean_accuracy']:.2%}±{results['map_elites']['best_std_accuracy']:.2%}, "
          f"time={me_time:.2f}s±{results['map_elites']['best_std_time_s']:.2f}s")
    print(f"  Speedup (single): {results['speedup_single']:.2f}x")
    print(f"  Speedup (task):   {results['speedup_task']:.2f}x")

    return results


# ---------------------------------------------------------------------------
# Layout rendering
# ---------------------------------------------------------------------------

def _ensure_colors():
    """Register colors for all object types used across scenes."""
    from top_down_scene import CLASS_COLORS
    extra = {
        "mug": (180, 140, 60),
        "book": (140, 180, 220),
        "phone": (80, 80, 160),
        "pen_cup": (200, 160, 100),
        "stapler": (120, 120, 180),
        "sponge": (100, 200, 200),
        "bottle": (180, 100, 60),
        "milk_carton": (200, 200, 200),
        "red_block": (60, 60, 220),
        "blue_block": (220, 140, 60),
        "green_cylinder": (60, 200, 60),
        "yellow_block": (30, 220, 240),
        "orange_cylinder": (50, 140, 240),
        "purple_block": (180, 80, 180),
        "white_cylinder": (220, 220, 220),
        "pink_block": (180, 140, 220),
        "tray_left": (100, 120, 160),
        "tray_right": (100, 160, 120),
        "bin_center": (140, 140, 140),
    }
    for k, v in extra.items():
        if k not in CLASS_COLORS:
            CLASS_COLORS[k] = v


def render_layout(positions: Dict[str, np.ndarray],
                  half_extents: Dict[str, list],
                  title: str = "",
                  optimized_positions: Optional[Dict[str, np.ndarray]] = None,
                  ee_pos: Optional[Tuple[float, float]] = None,
                  baseline_slack: Optional[float] = None,
                  optimized_slack: Optional[float] = None,
                  ) -> np.ndarray:
    """Render a top-down layout image using TopDownScene.

    Returns BGR numpy image (800x800).
    """
    from top_down_scene import TopDownScene, SceneObject
    _ensure_colors()

    scene = TopDownScene(canvas_size=(800, 800))
    if ee_pos:
        scene.set_ee_position(*ee_pos)
    else:
        scene.set_ee_position(START_POS[0], START_POS[1])

    objects = []
    for name, pos in positions.items():
        he = half_extents.get(name, [0.03, 0.03, 0.03])
        objects.append(SceneObject(
            label=name, x=float(pos[0]), y=float(pos[1]),
            half_extents_xy=(float(he[0]), float(he[1])),
        ))
    scene.set_objects(objects)

    if optimized_positions:
        opt_objects = []
        for name, pos in optimized_positions.items():
            he = half_extents.get(name, [0.03, 0.03, 0.03])
            opt_objects.append(SceneObject(
                label=name, x=float(pos[0]), y=float(pos[1]),
                half_extents_xy=(float(he[0]), float(he[1])),
            ))
        scene.set_optimized_objects(opt_objects)

    if baseline_slack is not None and optimized_slack is not None:
        scene.set_slack_scores(baseline_slack, optimized_slack)

    img = scene.render()

    # Add title text (ASCII only — OpenCV can't render Unicode)
    if title:
        cv2.putText(img, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def save_layout_screenshots(env_key: str,
                            all_positions: Dict[str, np.ndarray],
                            half_extents: Dict[str, list],
                            me_best_positions: Optional[Dict[str, np.ndarray]],
                            random_layout: Optional[Dict[str, np.ndarray]],
                            output_dir: str,
                            baseline_slack: Optional[float] = None,
                            optimized_slack: Optional[float] = None):
    """Save layout screenshots for one environment."""
    os.makedirs(output_dir, exist_ok=True)

    # Baseline
    img = render_layout(all_positions, half_extents,
                       title=f"{env_key} - Baseline")
    cv2.imwrite(os.path.join(output_dir, f"{env_key}_baseline.png"), img)

    # Random (one representative)
    if random_layout:
        img = render_layout(random_layout, half_extents,
                           title=f"{env_key} - Random")
        cv2.imwrite(os.path.join(output_dir, f"{env_key}_random.png"), img)

    # ME Best
    if me_best_positions:
        img = render_layout(me_best_positions, half_extents,
                           title=f"{env_key} - MAP-Elites Best")
        cv2.imwrite(os.path.join(output_dir, f"{env_key}_me_best.png"), img)

        # Comparison overlay: baseline (solid) + optimized (green dashed)
        # with displacement arrows and slack scores in legend
        img = render_layout(all_positions, half_extents,
                           title=f"{env_key} - Baseline vs Optimized",
                           optimized_positions=me_best_positions,
                           baseline_slack=baseline_slack,
                           optimized_slack=optimized_slack)
        cv2.imwrite(os.path.join(output_dir, f"{env_key}_comparison.png"), img)


# ---------------------------------------------------------------------------
# Inference video recording
# ---------------------------------------------------------------------------

def simulate_goal_inference_stepwise(
    object_positions: np.ndarray,
    start_pos: np.ndarray,
    true_goal_idx: int,
    sigma_bar_inv: np.ndarray,
    sigma: np.ndarray,
    T: int = 200,
    step_size: float = 0.01,
    confidence_threshold: float = 0.95,
    rng=None,
):
    """Generator version of simulate_goal_inference that yields per-step state.

    Yields: (timestep, ee_pos, posterior, predicted_goal, identified)
    """
    if rng is None:
        rng = np.random.default_rng()

    M, d_dim = object_positions.shape
    prior = np.ones(M) / M
    true_goal = object_positions[true_goal_idx]
    pos = start_pos.copy()
    log_scores = np.log(prior).copy()
    sigma_inv = np.linalg.inv(sigma)
    identified = False

    for t in range(T):
        mu_true = goal_conditioned_policy(pos, true_goal)
        noise = rng.multivariate_normal(np.zeros(d_dim), sigma)
        u_t = mu_true + noise

        for g in range(M):
            mu_g = goal_conditioned_policy(pos, object_positions[g])
            diff = u_t - mu_g
            log_scores[g] -= 0.5 * diff @ sigma_inv @ diff

        predicted = np.argmax(log_scores)
        log_shifted = log_scores - np.max(log_scores)
        posterior = np.exp(log_shifted)
        posterior /= posterior.sum()

        if predicted == true_goal_idx and posterior[predicted] >= confidence_threshold:
            identified = True

        yield (t, pos.copy(), posterior.copy(), int(predicted), identified)

        if identified:
            return

        pos = pos + step_size * mu_true
        if np.linalg.norm(pos - start_pos) > np.linalg.norm(true_goal - start_pos):
            pos = true_goal.copy()


def goal_conditioned_policy(ee_pos, goal_pos):
    """Normalized direction toward goal (imported inline to avoid circular)."""
    diff = goal_pos - ee_pos
    dist = np.linalg.norm(diff)
    if dist < 1e-8:
        return np.zeros_like(diff)
    return diff / dist


def _draw_posterior_bar_chart(posterior: np.ndarray,
                              object_names: List[str],
                              true_idx: int,
                              predicted_idx: int,
                              identified: bool,
                              width: int = 300,
                              height: int = 200) -> np.ndarray:
    """Draw a posterior bar chart as a BGR image."""
    img = np.full((height, width, 3), 40, dtype=np.uint8)
    M = len(posterior)
    if M == 0:
        return img

    bar_w = max(1, (width - 40) // M)
    bar_max_h = height - 50
    x_start = 20

    for i in range(M):
        x = x_start + i * bar_w
        bar_h = int(posterior[i] * bar_max_h)

        # Color: green for true, red for predicted (if wrong), blue otherwise
        if i == true_idx:
            color = (80, 200, 80)
        elif i == predicted_idx and predicted_idx != true_idx:
            color = (80, 80, 200)
        else:
            color = (180, 140, 60)

        cv2.rectangle(img, (x, height - 30 - bar_h), (x + bar_w - 2, height - 30),
                     color, -1)

        # Label
        label = object_names[i][:6] if i < len(object_names) else str(i)
        cv2.putText(img, label, (x, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    # Confidence threshold line
    thresh_y = height - 30 - int(0.95 * bar_max_h)
    cv2.line(img, (x_start, thresh_y), (x_start + M * bar_w, thresh_y),
            (0, 255, 255), 1)
    cv2.putText(img, "95%", (x_start + M * bar_w + 2, thresh_y + 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

    # Status text
    status = "IDENTIFIED" if identified else "inferring..."
    color = (80, 255, 80) if identified else (200, 200, 200)
    cv2.putText(img, status, (width // 2 - 40, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return img


def record_inference_video(
    object_positions: np.ndarray,
    object_names: List[str],
    true_goal_idx: int,
    half_extents: Dict[str, list],
    all_positions: Dict[str, np.ndarray],
    output_path: str,
    seed: int = 42,
    fps: int = 10,
):
    """Record an inference video showing EE trajectory + posterior evolution."""
    from top_down_scene import TopDownScene, SceneObject

    rng = np.random.default_rng(seed)
    frame_w, frame_h = 1100, 800  # layout (800) + bar chart (300)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    # Pre-build the base scene objects
    base_objects = []
    for name, pos in all_positions.items():
        he = half_extents.get(name, [0.03, 0.03, 0.03])
        base_objects.append(SceneObject(
            label=name, x=float(pos[0]), y=float(pos[1]),
            half_extents_xy=(float(he[0]), float(he[1])),
        ))

    trajectory = []

    for step_data in simulate_goal_inference_stepwise(
        object_positions, START_POS, true_goal_idx,
        SIGMA_BAR_INV, SIGMA, T=200, step_size=0.01,
        confidence_threshold=0.95, rng=rng,
    ):
        t, ee_pos, posterior, predicted, identified = step_data
        trajectory.append(ee_pos.copy())

        # Render layout
        scene = TopDownScene(canvas_size=(800, 800))
        scene.set_objects(base_objects)
        scene.set_ee_position(float(ee_pos[0]), float(ee_pos[1]))
        layout_img = scene.render()

        # Draw trajectory trace on layout
        for i in range(1, len(trajectory)):
            p1 = scene._world_to_pixel(trajectory[i-1][0], trajectory[i-1][1])
            p2 = scene._world_to_pixel(trajectory[i][0], trajectory[i][1])
            cv2.line(layout_img, p1, p2, (0, 255, 0), 2)

        # Draw title
        true_name = object_names[true_goal_idx] if true_goal_idx < len(object_names) else f"goal_{true_goal_idx}"
        cv2.putText(layout_img, f"Goal: {true_name}  t={t}  ({t*SECONDS_PER_STEP:.1f}s)",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Render posterior bar chart
        bar_img = _draw_posterior_bar_chart(
            posterior, object_names, true_goal_idx, predicted, identified,
            width=300, height=800,
        )

        # Composite
        frame = np.full((frame_h, frame_w, 3), 40, dtype=np.uint8)
        frame[:800, :800] = layout_img
        frame[:800, 800:1100] = bar_img

        writer.write(frame)

        if identified:
            # Hold final frame for 2 seconds
            for _ in range(fps * 2):
                writer.write(frame)
            break

    writer.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAP-Elites tiered evaluation")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Subset of environments to run (keys from ENVIRONMENTS)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of MAP-Elites seeds per scene")
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Monte Carlo trials per goal/task")
    parser.add_argument("--n-random", type=int, default=10,
                        help="Number of random baseline layouts")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--feature2", type=str, default="nn_var",
                        choices=list(FEATURE2_FUNCTIONS.keys()),
                        help="Feature 2 for MAP-Elites archive")
    parser.add_argument("--objective", type=str, default="separability",
                        choices=["separability", "legibility", "trajectory_margin"],
                        help="Optimization objective function")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip feature range calibration")
    args = parser.parse_args()

    envs_to_run = OrderedDict()
    if args.scenes:
        for key in args.scenes:
            if key in ENVIRONMENTS:
                envs_to_run[key] = ENVIRONMENTS[key]
            else:
                print(f"Warning: unknown environment '{key}', skipping")
    else:
        envs_to_run = ENVIRONMENTS

    all_results = {}
    total_start = time.time()

    for env_key, env_config in envs_to_run.items():
        result = run_experiment(
            env_key, env_config,
            n_seeds=args.seeds,
            n_trials=args.n_trials,
            n_random=args.n_random,
            feature2_name=args.feature2,
            calibrate=not args.no_calibrate,
            objective=args.objective,
        )
        all_results[env_key] = result

    total_time = time.time() - total_start

    # Save results
    output_path = args.output or os.path.join(
        PROJECT_ROOT, "results", "map_elites_tier_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy arrays to lists for JSON
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print("CROSS-TIER SUMMARY")
    print(f"{'='*100}")
    print(f"{'Tier':<10} {'Scene':<16} {'#Obj':>5} {'Base Time':>10} {'ME Time':>8} "
          f"{'Speedup':>8} {'Base Acc':>9} {'ME Acc':>7} {'Base Slack':>11} {'ME Slack':>9}")
    print("-" * 100)
    for key, r in all_results.items():
        print(f"{r['tier']:<10} {key:<16} {r['n_objects']:>5} "
              f"{r['baseline']['single_step']['mean_time_s']:>9.2f}s "
              f"{r['map_elites']['best_mean_time_s']:>7.2f}s "
              f"{r['speedup_single']:>7.2f}x "
              f"{r['baseline']['single_step']['accuracy']:>8.2%} "
              f"{r['map_elites']['best_mean_accuracy']:>6.2%} "
              f"{r['baseline']['separability_slack']:>11.1f} "
              f"{r['map_elites']['best_mean_slack']:>9.1f}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Control rate: {CONTROL_RATE_HZ} Hz ({SECONDS_PER_STEP}s per step)")


if __name__ == "__main__":
    main()
