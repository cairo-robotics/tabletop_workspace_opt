"""Grasp-aware feasibility checks for 3D workspace optimization.

Used by the MAP-Elites inner loop (and the nested yaw GD) to decide
whether a candidate layout is *actually* reachable: for every object in
the task, does the assigned grasp pose have an IK solution AND is the
pre-grasp → grasp approach segment free of collisions with the other
objects' rotated footprints?

The soft proxies (distance-based) are used during yaw GD (gradient
flow); the hard binary checks are used by the outer MAP-Elites fitness.

All 2D footprint math assumes axis-aligned or yaw-rotated rectangles
from the scene YAML's `half_extents` field.
"""
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon, LineString


def _rotated_rect(center_xy, half_extents_xy, yaw):
    """Return a shapely Polygon for a yaw-rotated 2D rectangle."""
    hx, hy = half_extents_xy
    corners_local = np.array([
        [-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_world = (R @ corners_local.T).T + np.asarray(center_xy)
    return Polygon(corners_world)


def object_footprints(layout_xy: Dict[str, np.ndarray],
                      layout_yaws: Dict[str, float],
                      half_extents: Dict[str, Tuple[float, float]]):
    """Build a dict {name: shapely Polygon} for every object in the layout."""
    footprints = {}
    for name, xy in layout_xy.items():
        he = half_extents.get(name)
        if he is None:
            continue
        yaw = layout_yaws.get(name, 0.0)
        footprints[name] = _rotated_rect(xy[:2], he[:2], yaw)
    return footprints


def pairwise_overlap_penalty(footprints: Dict[str, Polygon],
                             separation: float = 0.0) -> float:
    """Sum of intersection areas between all pairs (zero if non-overlapping).

    If `separation > 0`, each polygon is inflated by that radius before
    intersecting, so "near-touching" rectangles also get penalized.
    """
    names = list(footprints.keys())
    penalty = 0.0
    for i in range(len(names)):
        pi = footprints[names[i]]
        if separation > 0:
            pi = pi.buffer(separation * 0.5)
        for j in range(i + 1, len(names)):
            pj = footprints[names[j]]
            if separation > 0:
                pj = pj.buffer(separation * 0.5)
            if pi.intersects(pj):
                penalty += pi.intersection(pj).area
    return penalty


def minimum_pair_distance(footprints: Dict[str, Polygon]) -> float:
    """Return the smallest edge-to-edge distance between any two footprints.

    Positive = non-overlapping. Zero or negative (via buffer hack) = overlap.
    """
    names = list(footprints.keys())
    if len(names) < 2:
        return np.inf
    best = np.inf
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pi = footprints[names[i]]
            pj = footprints[names[j]]
            if pi.intersects(pj):
                return -pi.intersection(pj).area   # negative = overlap
            best = min(best, pi.distance(pj))
    return best


def approach_collision(approach_seg: Tuple[np.ndarray, np.ndarray],
                       moving_name: str,
                       footprints: Dict[str, Polygon]) -> float:
    """Distance from an approach segment to the nearest OTHER object footprint.

    Approach is specified as (pregrasp_xy, grasp_xy) in 2D. Returns
    minimum distance (positive = clear). Zero or negative on collision.
    """
    seg = LineString([approach_seg[0][:2], approach_seg[1][:2]])
    best = np.inf
    for name, poly in footprints.items():
        if name == moving_name:
            continue
        if seg.intersects(poly):
            return 0.0
        best = min(best, seg.distance(poly))
    return best


def check_layout_feasibility(layout_xy: Dict[str, np.ndarray],
                             layout_yaws: Dict[str, float],
                             half_extents: Dict[str, Tuple[float, float]],
                             grasp_library,
                             oracle,
                             task_objects: List[str],
                             min_approach_clearance: float = 0.005,
                             min_object_separation: float = 0.02,
                             mujoco_names: Dict[str, str] = None,
                             check_arm_collisions: bool = True,
                             fixed_positions: Dict[str, np.ndarray] = None):
    """Check every task object's grasp feasibility for a layout.

    Args:
        layout_xy: {short_name: (x, y)} object positions.
        layout_yaws: {short_name: yaw} object yaws (radians).
        half_extents: {short_name: (hx, hy)} footprint half-extents.
        grasp_library: loaded GraspLibrary.
        oracle: ReachabilityOracle instance.
        task_objects: list of short_names whose grasps must be feasible.
        min_approach_clearance: meters; approach segment must stay at
            least this far from other object footprints.
        min_object_separation: meters; objects must be separated by at
            least this edge-to-edge distance.
        mujoco_names: optional {short_name: mujoco_body_name} map.
            Required when ``check_arm_collisions=True`` so the oracle
            can teleport objects in its internal model and run real
            arm-vs-other-object collision queries at the IK solution.
        check_arm_collisions: if True, after each successful IK solve
            the oracle's MuJoCo model is updated to the candidate
            layout (objects teleported), the arm is set to the IK
            joints, and contacts are walked. Any arm/gripper geom in
            contact with an object body OTHER than the target is
            counted as a failure. Self-collisions and arm-vs-table
            contacts are NOT included here (they would need a separate
            check). Requires ``mujoco_names`` to be supplied.

    Returns dict:
        feasible: bool — all hard checks pass.
        failures: list of str — names of failing objects (IK,
            approach-collision, or arm-collision). Empty if feasible.
        overlap_penalty: float — summed overlap area (0 if clean).
        min_approach_clearance_seen: float — worst clearance observed.
        min_object_separation_seen: float — worst pair separation.
        arm_collisions: list of (obj_name, [(arm_body, other_body), ...])
            non-empty entries when arm-other-object collisions were
            detected. Always present (possibly empty) when arm
            collision checking ran.
    """
    # Build footprints for ALL objects (movable + fixed) so pairwise
    # overlap and separation checks catch movable-vs-fixed collisions.
    all_positions = dict(layout_xy)
    all_yaws = dict(layout_yaws)
    all_he = dict(half_extents)
    if fixed_positions:
        for fn, fxy in fixed_positions.items():
            if fn not in all_positions:
                all_positions[fn] = fxy
                all_yaws.setdefault(fn, 0.0)
                # half_extents should already have fixed objects if
                # the caller populated it from the scene YAML
    footprints = object_footprints(all_positions, all_yaws, all_he)
    overlap = pairwise_overlap_penalty(footprints, separation=0.0)
    min_sep = minimum_pair_distance(footprints)

    failures = []
    worst_clear = np.inf
    arm_collisions = []

    # If we're going to do arm collision checks, teleport every object
    # in the oracle's model to the candidate layout positions ONCE up
    # front. The oracle solves IK against this scene, and the contact
    # check uses the same scene state.
    do_arm_collision = (check_arm_collisions and oracle is not None
                        and mujoco_names is not None)
    if do_arm_collision:
        for obj_name, xy in layout_xy.items():
            mname = mujoco_names.get(obj_name)
            if mname is None:
                continue
            if len(xy) < 3:
                pos3 = np.array([xy[0], xy[1], 0.95])
            else:
                pos3 = np.asarray(xy)
            oracle.teleport_object(
                mname, pos3, layout_yaws.get(obj_name, 0.0))

    for obj_name in task_objects:
        entry = grasp_library.get(obj_name)
        if entry is None:
            failures.append(f"{obj_name}:no_grasp_entry")
            continue

        obj_xy = layout_xy.get(obj_name)
        if obj_xy is None:
            failures.append(f"{obj_name}:no_position")
            continue

        if len(obj_xy) < 3:
            obj_pos3 = np.array([obj_xy[0], obj_xy[1], 0.95])
        else:
            obj_pos3 = np.asarray(obj_xy)

        yaw = layout_yaws.get(obj_name, 0.0)
        poses = entry.resolve(obj_pos3, yaw)

        # IK reachability of the pre-grasp pose (what the observer sees)
        # and the grasp pose itself (auto-completion target).
        q_pregrasp = None
        q_grasp = None
        if oracle is not None:
            q_pregrasp = oracle.solve_ik(poses["pregrasp_pos"], poses["grasp_R"])
            if q_pregrasp is None:
                failures.append(f"{obj_name}:pregrasp_ik")
                continue
            q_grasp = oracle.solve_ik(poses["grasp_pos"], poses["grasp_R"])
            if q_grasp is None:
                failures.append(f"{obj_name}:grasp_ik")
                continue

        # Approach segment collision with other objects' footprints (2D).
        seg = (poses["pregrasp_pos"], poses["grasp_pos"])
        clear = approach_collision(seg, obj_name, footprints)
        worst_clear = min(worst_clear, clear)
        if clear < min_approach_clearance:
            failures.append(f"{obj_name}:approach_collide({clear:.3f})")

        # Real 3D arm-vs-other-object collision check at both poses.
        # The target object is excluded from "other" — touching the
        # target is the point of grasping.
        if do_arm_collision and q_grasp is not None and q_pregrasp is not None:
            target_body = mujoco_names.get(obj_name)
            for label, q in (("grasp", q_grasp), ("pregrasp", q_pregrasp)):
                contacts = oracle.arm_object_collisions(q, target_body)
                if contacts:
                    arm_collisions.append({
                        "object": obj_name,
                        "phase": label,
                        "contacts": contacts,
                    })
                    failures.append(
                        f"{obj_name}:{label}_arm_collide("
                        f"{len(contacts)} pairs)")
                    break  # one failure per object is enough

    feasible = (len(failures) == 0
                and overlap < 1e-6
                and min_sep >= min_object_separation)
    return {
        "feasible": feasible,
        "failures": failures,
        "overlap_penalty": float(overlap),
        "min_approach_clearance_seen": float(worst_clear),
        "min_object_separation_seen": float(min_sep),
        "arm_collisions": arm_collisions,
    }
