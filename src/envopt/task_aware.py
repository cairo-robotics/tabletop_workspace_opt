"""Task-aware workspace optimization functions.

Reusable module for task-aware optimization. This replaced the older
standalone task-aware optimizer script so shared logic can be imported by
current tools and ROS nodes.

Supports two optimization methods:
  - "differential_evolution": scipy global optimizer (default)
  - "map_elites": Quality-Diversity optimization via pyribs (CMA-ME)
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from intent_separability import intent_separability_for_optimizer

# Table bounds and constraints (same as workspace_optimizer.py)
TABLE_BOUNDS_X = (0.30, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)
MIN_DIST = 0.12


def extract_pick_states(task_config: dict) -> List[Tuple[str, List[str]]]:
    """Extract states where pick disambiguation is needed.

    Args:
        task_config: The 'task' dict from a task YAML config.

    Returns:
        List of (state_name, [object_names]) for states with 2+ pick goals.
    """
    states = task_config["states"]
    pick_states = []
    for sname, sdata in states.items():
        goals = sdata.get("valid_goals", [])
        pick_goals = [g for g in goals if g["action"] == "pick"]
        if len(pick_goals) >= 2:
            obj_names = [g["object"] for g in pick_goals]
            pick_states.append((sname, obj_names))
    return pick_states


def deduplicate_pick_states(
    pick_states: List[Tuple[str, List[str]]]
) -> List[Tuple[str, List[str]]]:
    """Remove duplicate pick states that have the same set of competing objects."""
    seen = set()
    unique = []
    for state_name, competing_objects in pick_states:
        key = tuple(sorted(competing_objects))
        if key not in seen:
            seen.add(key)
            unique.append((state_name, competing_objects))
    return unique


def task_aware_objective(
    flat_positions: np.ndarray,
    object_names: List[str],
    pick_states: List[Tuple[str, List[str]]],
    start_pos: np.ndarray,
    sigma_bar_inv: np.ndarray,
    T: int = 50,
    step_size: float = 0.01,
    alpha: float = 0.05,
) -> float:
    """Compute worst-case separability across unique task states.

    Only considers objects that compete at each state, and deduplicates
    states with identical competing object sets.

    Returns:
        Negative (worst_slack - penalty) for scipy minimization.
    """
    d = 2
    positions = flat_positions.reshape(-1, d)
    name_to_idx = {n: i for i, n in enumerate(object_names)}

    worst_slack = float('inf')
    for state_name, competing_objects in pick_states:
        indices = [name_to_idx[obj] for obj in competing_objects
                   if obj in name_to_idx]
        if len(indices) < 2:
            continue
        state_positions = positions[indices]
        neg_slack = intent_separability_for_optimizer(
            state_positions.flatten(), start_pos, sigma_bar_inv,
            d=d, T=T, step_size=step_size, alpha=alpha)
        slack = -neg_slack
        worst_slack = min(worst_slack, slack)

    # Overlap penalty
    penalty = 0.0
    M = len(positions)
    for i in range(M):
        for j in range(i + 1, M):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < MIN_DIST:
                penalty += (MIN_DIST - dist) ** 2

    return -(worst_slack - 100.0 * penalty)


def _compute_task_me_features(flat_positions: np.ndarray,
                              start_pos: np.ndarray,
                              d: int = 2) -> Tuple[float, float]:
    """Compute MAP-Elites feature descriptors for a layout.

    Same features as workspace_optimizer._compute_me_features:
        1. Mean pairwise distance between objects
        2. Angular spread std — std of angular gaps from start_pos
    """
    M = len(flat_positions) // d
    positions = flat_positions.reshape(M, d)

    # Feature 1: mean pairwise distance
    dists = []
    for i in range(M):
        for j in range(i + 1, M):
            dists.append(np.linalg.norm(positions[i] - positions[j]))
    mean_dist = np.mean(dists) if dists else 0.0

    # Feature 2: angular spread std
    deltas = positions - start_pos[:d]
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])
    sorted_angles = np.sort(angles)
    angle_diffs = np.diff(sorted_angles)
    wrap_diff = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
    angle_diffs = np.append(angle_diffs, wrap_diff)
    angular_std = np.std(angle_diffs)

    return mean_dist, angular_std


def run_task_aware_map_elites(
    object_names: List[str],
    initial_positions: np.ndarray,
    pick_states: List[Tuple[str, List[str]]],
    start_pos: np.ndarray,
    sigma_bar_inv: np.ndarray,
    bounds_x: Tuple[float, float] = TABLE_BOUNDS_X,
    bounds_y: Tuple[float, float] = TABLE_BOUNDS_Y,
    T: int = 50,
    step_size: float = 0.01,
    alpha: float = 0.05,
    min_dist: float = MIN_DIST,
    maxiter: int = 200,
    seed: int = 42,
    verbose: bool = True,
    batch_size: int = 30,
) -> np.ndarray:
    """Run MAP-Elites (CMA-ME) with task-aware objective.

    Returns optimized flat positions array.
    """
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler

    M = len(object_names)
    d = 2
    solution_dim = M * d
    x0 = initial_positions.flatten()

    lower_bounds = np.array([[bounds_x[0], bounds_y[0]]] * M).flatten()
    upper_bounds = np.array([[bounds_x[1], bounds_y[1]]] * M).flatten()

    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[25, 25],
        ranges=[(0.1, 1.0), (0.0, 1.5)],
    )

    np.random.seed(seed)
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=x0,
            sigma0=0.08,
            ranker="2imp",
            bounds=list(zip(lower_bounds, upper_bounds)),
            batch_size=batch_size,
        )
        for _ in range(3)
    ]

    scheduler = Scheduler(archive, emitters)

    for itr in range(1, maxiter + 1):
        sols = scheduler.ask()

        objs = []
        measures = []
        for sol in sols:
            neg_obj = task_aware_objective(
                sol, object_names, pick_states, start_pos, sigma_bar_inv,
                T=T, step_size=step_size, alpha=alpha,
            )
            obj = -neg_obj  # MAP-Elites maximizes

            feat1, feat2 = _compute_task_me_features(sol, start_pos, d=d)
            objs.append(obj)
            measures.append([feat1, feat2])

        scheduler.tell(objs, measures)

        if verbose and itr % 25 == 0:
            print(f"  MAP-Elites iter {itr}/{maxiter}: "
                  f"elites={archive.stats.num_elites}, "
                  f"max_obj={archive.stats.obj_max:.2f}, "
                  f"mean_obj={archive.stats.obj_mean:.2f}")

    best = archive.best_elite
    if verbose:
        print(f"  MAP-Elites final: {archive.stats.num_elites} elites, "
              f"max_obj={archive.stats.obj_max:.2f}")
    return best["solution"]
