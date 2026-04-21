"""Layout optimizer for SE(3) grasp-aware shared autonomy.

Outer loop: differential evolution over 2D object positions (x, y).
Inner loop (per candidate): torch Adam on the differentiable SE(3)
separability objective over object yaws.

Hard feasibility (TracIK + approach collision) is applied as an
infinity penalty on the outer fitness. Soft penalties (overlap,
min-distance, bounds) guide DE.

Entry point: `optimize_scene(scene_cfg, task_objects, ...)`.

Outputs: dict with best (x, y, yaw) per object, slack, feasibility flags.
"""
from typing import Dict, List, Optional  # noqa: F401

import numpy as np
from scipy.optimize import differential_evolution

from envopt.grasp_feasibility import (
    check_layout_feasibility, object_footprints,
    pairwise_overlap_penalty, minimum_pair_distance)
from envopt.yaw_optimizer import optimize_yaw, slack_with_yaws


# Shared defaults for the Sawyer breakfast-table workspace
TABLE_BOUNDS_X = (0.35, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)
ROBOT_HOME_2D = np.array([0.452, 0.160])
ROBOT_EXCLUSION = 0.15   # keep objects at least this far from home
MIN_OBJ_DIST = 0.12


def _params_to_layout(params: np.ndarray, order: List[str],
                      z_map: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Unflatten a DE parameter vector into {name: (x, y, z)}."""
    layout = {}
    for i, name in enumerate(order):
        layout[name] = np.array([
            params[2 * i], params[2 * i + 1], z_map[name]])
    return layout


def _soft_layout_penalty(layout_xy, half_extents, min_clearance=0.03,
                         exclusion_radius=ROBOT_EXCLUSION,
                         fixed_positions=None):
    """Penalty terms that keep DE in the feasible region.

    - Robot exclusion: distance of each object center to the home pos.
    - Size-aware pairwise separation between ALL objects (movable + fixed).
      The minimum center-to-center distance for each pair is
      max(he_i) + max(he_j) + min_clearance, so objects with large
      footprints need proportionally more separation.

    ``fixed_positions`` is an optional {name: (x, y [,z])} dict of
    non-task objects (e.g. bowl, cutting_board) that are not decision
    variables but must not be overlapped.
    """
    penalty = 0.0
    names = list(layout_xy.keys())
    for name in names:
        xy = layout_xy[name][:2]
        d_home = np.linalg.norm(xy - ROBOT_HOME_2D)
        if d_home < exclusion_radius:
            penalty += (exclusion_radius - d_home) ** 2 * 100.0

    # Build the full set of positions for pairwise separation checks.
    all_positions = dict(layout_xy)
    if fixed_positions:
        all_positions.update(fixed_positions)
    all_names = list(all_positions.keys())

    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            d = np.linalg.norm(all_positions[all_names[i]][:2]
                               - all_positions[all_names[j]][:2])
            # Size-aware minimum: sum of mean half-extents + clearance.
            # Using the mean (not max) so elongated objects like the
            # banana (90mm × 20mm) aren't treated as 90mm × 90mm.
            he_i = half_extents.get(all_names[i], (0.03, 0.03))
            he_j = half_extents.get(all_names[j], (0.03, 0.03))
            r_i = (he_i[0] + he_i[1]) * 0.5
            r_j = (he_j[0] + he_j[1]) * 0.5
            min_d = r_i + r_j + min_clearance
            if d < min_d:
                penalty += (min_d - d) ** 2 * 100.0
    return penalty


def optimize_scene(task_objects: List[str],
                   half_extents: Dict[str, tuple],
                   z_map: Dict[str, float],
                   grasp_library,
                   oracle,
                   start_pos: np.ndarray,
                   bounds_x=TABLE_BOUNDS_X,
                   bounds_y=TABLE_BOUNDS_Y,
                   sigma_v: float = 0.5,
                   sigma_w: float = 0.5,
                   lambda_R: float = 0.04,
                   maxiter: int = 40,
                   popsize: int = 12,
                   seed: int = 0,
                   yaw_steps: int = 20,
                   verbose: bool = True,
                   mujoco_names: Optional[Dict[str, str]] = None,
                   check_arm_collisions: bool = True,
                   fixed_positions: Optional[Dict[str, np.ndarray]] = None,
                   objective: str = "separability"):
    """Run DE + nested yaw GD to optimize a layout for SE(3) separability.

    Args:
        task_objects: ordered list of short_names appearing on the table.
        half_extents: {name: (hx, hy)} footprint half-extents.
        z_map: {name: z} frozen z-heights (object center world z).
        grasp_library: loaded GraspLibrary.
        oracle: ReachabilityOracle instance.
        start_pos: (3,) EE home position.
        maxiter, popsize: differential_evolution params.
        fixed_positions: optional {name: (x, y [,z])} for non-task objects
            that must not be overlapped. Included in pairwise overlap and
            separation checks but not optimized.

    Returns dict with:
        positions: {name: (x, y, z)}
        yaws: {name: yaw_rad}
        slack: final SE(3) separability slack
        feasibility: full check_layout_feasibility output
    """
    N = len(task_objects)
    bounds = []
    for _ in range(N):
        bounds.append(bounds_x)
        bounds.append(bounds_y)

    eval_count = {"n": 0, "feas": 0, "infeas": 0, "best": -np.inf}

    def fitness(params):
        eval_count["n"] += 1
        layout_xy = _params_to_layout(params, task_objects, z_map)
        pen = _soft_layout_penalty(layout_xy, half_extents,
                                   fixed_positions=fixed_positions)
        if pen > 1e-3:
            return 1000.0 + pen   # keep DE moving toward feasible

        # Quick overlap check before calling IK.
        # Include fixed objects in the footprint map so movable objects
        # can't land on top of bowls, cutting boards, etc.
        all_positions = dict(layout_xy)
        all_yaws = {n: 0.0 for n in task_objects}
        all_he = dict(half_extents)
        if fixed_positions:
            for fn, fxy in fixed_positions.items():
                all_positions[fn] = fxy
                all_yaws[fn] = 0.0
                if fn not in all_he:
                    all_he[fn] = half_extents.get(fn, (0.03, 0.03))
        fps = object_footprints(all_positions, all_yaws, all_he)
        overlap = pairwise_overlap_penalty(fps)
        if overlap > 1e-6:
            return 500.0 + overlap * 10000.0

        # Inner loop: optimize yaws for this (x, y) layout
        yaws, slack = optimize_yaw(
            layout_xy, task_objects, grasp_library, start_pos,
            n_steps=yaw_steps, sigma_v=sigma_v, sigma_w=sigma_w,
            lambda_R=lambda_R, objective=objective)

        # Hard feasibility check with the optimized yaws
        yaw_map = {n: float(y) for n, y in zip(task_objects, yaws)}
        feas = check_layout_feasibility(
            layout_xy, yaw_map, half_extents, grasp_library, oracle,
            task_objects=task_objects,
            min_approach_clearance=0.005,
            min_object_separation=0.005,
            mujoco_names=mujoco_names,
            check_arm_collisions=check_arm_collisions,
            fixed_positions=fixed_positions)

        if not feas["feasible"]:
            eval_count["infeas"] += 1
            return 200.0   # hard infeasibility penalty
        eval_count["feas"] += 1
        if slack > eval_count["best"]:
            eval_count["best"] = slack
            if verbose:
                print(f"  eval {eval_count['n']:4d}: slack={slack:.3f}"
                      f" feas={feas['feasible']} (best)")
        return -slack

    if verbose:
        print(f"DE: N={N} objs, {N*2}D, maxiter={maxiter}, pop={popsize}")

    res = differential_evolution(
        fitness, bounds,
        maxiter=maxiter, popsize=popsize,
        seed=seed, tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        disp=verbose,
        init='sobol',
    )

    layout_xy = _params_to_layout(res.x, task_objects, z_map)

    # Build a feasibility callback for the polish step so the yaw
    # optimizer only returns yaws that pass the hard feasibility check.
    # This prevents the polish step from overshooting into infeasible
    # configurations while chasing higher slack.
    def _feas_check(yaw_dict):
        feas_result = check_layout_feasibility(
            layout_xy, yaw_dict, half_extents, grasp_library, oracle,
            task_objects=task_objects,
            min_approach_clearance=0.005,
            min_object_separation=0.005,
            mujoco_names=mujoco_names,
            check_arm_collisions=check_arm_collisions,
            fixed_positions=fixed_positions)
        return feas_result["feasible"]

    yaws, final_slack = optimize_yaw(
        layout_xy, task_objects, grasp_library, start_pos,
        n_steps=80,   # polish step
        sigma_v=sigma_v, sigma_w=sigma_w, lambda_R=lambda_R,
        feasibility_fn=_feas_check, objective=objective)
    yaw_map = {n: float(y) for n, y in zip(task_objects, yaws)}
    feas = check_layout_feasibility(
        layout_xy, yaw_map, half_extents, grasp_library, oracle,
        task_objects=task_objects,
        min_approach_clearance=0.005,
        min_object_separation=0.005,
        mujoco_names=mujoco_names,
        check_arm_collisions=check_arm_collisions,
        fixed_positions=fixed_positions)
    if verbose:
        print(f"Final slack: {final_slack:.3f}  feasible: {feas['feasible']}")
        print(f"Eval stats: {eval_count['n']} total, "
              f"{eval_count['feas']} feasible, "
              f"{eval_count['infeas']} infeasible")

    return {
        "positions": layout_xy,
        "yaws": yaw_map,
        "slack": final_slack,
        "feasibility": feas,
        "eval_count": eval_count,
    }
