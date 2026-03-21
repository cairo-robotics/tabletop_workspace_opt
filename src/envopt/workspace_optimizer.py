"""Workspace Optimizer for Intent Separability

Optimizes object placements on the table to maximize the worst-case
intent separability slack (Section 1.6 of Shared Autonomy Notes).

Supports two optimization methods:
  - "differential_evolution": scipy global optimizer (default)
  - "map_elites": Quality-Diversity optimization via pyribs (CMA-ME)
"""

import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Tuple, Optional, Dict
from intent_separability import (
    intent_separability_for_optimizer,
    intent_separability_objective,
    simulate_goal_inference,
    parse_task_targets,
    simulate_multistep_goal_inference,
)


# Breakfast scene defaults (world frame)
DEFAULT_OBJECT_POSITIONS = {
    "bowl": np.array([0.600, 0.300]),
    "cereal": np.array([0.480, -0.250]),
    "banana": np.array([0.500, 0.050]),
    "milk": np.array([0.780, 0.300]),
}

# Robot EE starting position (world frame, approx home)
DEFAULT_START_POS = np.array([0.479, -0.060])

# Table bounds (world frame) — conservative margins from edges
TABLE_BOUNDS_X = (0.30, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)

# Minimum distance between object centers to avoid overlap
MIN_OBJECT_DISTANCE = 0.12


def overlap_penalty(flat_positions: np.ndarray, d: int = 2,
                    min_dist: float = MIN_OBJECT_DISTANCE) -> float:
    """Penalty for objects being too close together.

    Returns sum of squared violations of minimum distance constraint.
    """
    M = len(flat_positions) // d
    positions = flat_positions.reshape(M, d)
    penalty = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    return penalty


def penalized_objective(flat_positions: np.ndarray,
                        start_pos: np.ndarray,
                        sigma_bar_inv: np.ndarray,
                        d: int = 2,
                        T: int = 50,
                        step_size: float = 0.01,
                        alpha: float = 0.05,
                        prior: Optional[np.ndarray] = None,
                        overlap_weight: float = 100.0,
                        min_dist: float = MIN_OBJECT_DISTANCE) -> float:
    """Objective for scipy minimizers: negative slack + overlap penalty."""
    neg_slack = intent_separability_for_optimizer(
        flat_positions, start_pos, sigma_bar_inv,
        d=d, T=T, step_size=step_size, alpha=alpha, prior=prior,
    )
    penalty = overlap_weight * overlap_penalty(flat_positions, d=d, min_dist=min_dist)
    return neg_slack + penalty


def _compute_me_features(flat_positions: np.ndarray, start_pos: np.ndarray,
                         d: int = 2) -> Tuple[float, float]:
    """Compute MAP-Elites feature descriptors for a layout.

    Features:
        1. Mean pairwise distance between objects
        2. Angular spread std — std of angular gaps from start_pos

    Returns:
        (mean_pairwise_dist, angular_spread_std)
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


def _run_differential_evolution(
    M: int, d: int, start_pos: np.ndarray, sigma_bar_inv: np.ndarray,
    bounds_x: Tuple, bounds_y: Tuple, T: int, step_size: float,
    alpha: float, min_dist: float, maxiter: int, seed: int, verbose: bool,
) -> np.ndarray:
    """Run differential evolution optimizer. Returns optimized flat positions."""
    bounds = []
    for _ in range(M):
        bounds.append(bounds_x)
        bounds.append(bounds_y)

    result = differential_evolution(
        penalized_objective,
        bounds=bounds,
        args=(start_pos, sigma_bar_inv, d, T, step_size, alpha, None, 100.0, min_dist),
        maxiter=maxiter,
        seed=seed,
        tol=1e-6,
        polish=True,
        disp=verbose,
    )
    return result.x


def _run_map_elites(
    M: int, d: int, start_pos: np.ndarray, sigma_bar_inv: np.ndarray,
    initial_positions: np.ndarray,
    bounds_x: Tuple, bounds_y: Tuple, T: int, step_size: float,
    alpha: float, min_dist: float, maxiter: int, seed: int, verbose: bool,
    batch_size: int = 30,
) -> np.ndarray:
    """Run MAP-Elites (CMA-ME) optimizer. Returns optimized flat positions."""
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler

    solution_dim = M * d
    x0 = initial_positions.flatten()

    # Per-dimension bounds
    lower_bounds = np.array([[bounds_x[0], bounds_y[0]]] * M).flatten()
    upper_bounds = np.array([[bounds_x[1], bounds_y[1]]] * M).flatten()

    # Archive: 2D feature space (mean pairwise dist, angular spread std)
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[25, 25],
        ranges=[(0.1, 1.0), (0.0, 1.5)],
    )

    # Emitters: multiple CMA-ME emitters for diversity
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

        # Evaluate each solution
        objs = []
        measures = []
        for sol in sols:
            # Objective: intent separability slack (with overlap penalty)
            neg_slack = penalized_objective(
                sol, start_pos, sigma_bar_inv,
                d=d, T=T, step_size=step_size, alpha=alpha,
                min_dist=min_dist,
            )
            obj = -neg_slack  # MAP-Elites maximizes

            # Features
            feat1, feat2 = _compute_me_features(sol, start_pos, d=d)
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


def optimize_workspace(
    object_names: Optional[List[str]] = None,
    initial_positions: Optional[np.ndarray] = None,
    start_pos: Optional[np.ndarray] = None,
    sigma_bar: Optional[np.ndarray] = None,
    T: int = 50,
    step_size: float = 0.01,
    alpha: float = 0.05,
    bounds_x: Tuple[float, float] = TABLE_BOUNDS_X,
    bounds_y: Tuple[float, float] = TABLE_BOUNDS_Y,
    min_dist: float = MIN_OBJECT_DISTANCE,
    maxiter: int = 200,
    seed: int = 42,
    verbose: bool = True,
    method: str = "differential_evolution",
    batch_size: int = 30,
) -> Dict:
    """Run workspace optimization.

    Args:
        object_names: Names of objects to optimize. Defaults to breakfast objects.
        initial_positions: Initial (M, 2) positions. Defaults to scene positions.
        start_pos: EE start position (2,). Defaults to home position.
        sigma_bar: Noise covariance upper bound (2, 2). Defaults to σ²I.
        T: Trajectory timesteps for objective evaluation
        step_size: Step size per timestep
        alpha: Total failure probability budget
        bounds_x: (min, max) x bounds on table
        bounds_y: (min, max) y bounds on table
        min_dist: Minimum distance between object centers
        maxiter: Maximum optimization iterations
        seed: Random seed
        verbose: Print progress
        method: "differential_evolution" or "map_elites"
        batch_size: Batch size for MAP-Elites emitters

    Returns:
        Dict with keys: 'optimized_positions', 'initial_positions',
        'initial_slack', 'optimized_slack', 'object_names', 'method'
    """
    if method not in ("differential_evolution", "map_elites"):
        raise ValueError(f"Unknown method: {method}. "
                         f"Use 'differential_evolution' or 'map_elites'.")

    if object_names is None:
        object_names = list(DEFAULT_OBJECT_POSITIONS.keys())
    if initial_positions is None:
        initial_positions = np.array([DEFAULT_OBJECT_POSITIONS[n] for n in object_names])
    if start_pos is None:
        start_pos = DEFAULT_START_POS.copy()
    if sigma_bar is None:
        sigma_bar = 0.1 * np.eye(2)

    sigma_bar_inv = np.linalg.inv(sigma_bar)
    M = len(object_names)
    d = 2

    # Compute initial slack
    init_slack, init_per_goal = intent_separability_objective(
        initial_positions, start_pos, sigma_bar_inv,
        T=T, step_size=step_size, alpha=alpha,
    )
    if verbose:
        print(f"Initial worst-case slack: {init_slack:.4f}")
        for i, name in enumerate(object_names):
            print(f"  {name}: per-goal slack = {init_per_goal[i]:.4f}")

    # Run optimizer
    if method == "differential_evolution":
        opt_flat = _run_differential_evolution(
            M, d, start_pos, sigma_bar_inv,
            bounds_x, bounds_y, T, step_size, alpha, min_dist,
            maxiter, seed, verbose,
        )
    else:
        opt_flat = _run_map_elites(
            M, d, start_pos, sigma_bar_inv, initial_positions,
            bounds_x, bounds_y, T, step_size, alpha, min_dist,
            maxiter, seed, verbose, batch_size=batch_size,
        )

    optimized_positions = opt_flat.reshape(M, d)

    # Compute optimized slack
    opt_slack, opt_per_goal = intent_separability_objective(
        optimized_positions, start_pos, sigma_bar_inv,
        T=T, step_size=step_size, alpha=alpha,
    )
    if verbose:
        print(f"\nOptimized worst-case slack ({method}): {opt_slack:.4f}")
        for i, name in enumerate(object_names):
            print(f"  {name}: per-goal slack = {opt_per_goal[i]:.4f}")
        print(f"\nImprovement: {opt_slack - init_slack:.4f}")
        print(f"\nOptimized positions:")
        for i, name in enumerate(object_names):
            print(f"  {name}: ({optimized_positions[i, 0]:.3f}, {optimized_positions[i, 1]:.3f})")

    return {
        "object_names": object_names,
        "initial_positions": initial_positions,
        "optimized_positions": optimized_positions,
        "initial_slack": init_slack,
        "initial_per_goal_slacks": init_per_goal,
        "optimized_slack": opt_slack,
        "optimized_per_goal_slacks": opt_per_goal,
        "method": method,
    }


def evaluate_goal_prediction(
    object_positions: np.ndarray,
    start_pos: np.ndarray,
    sigma_bar: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    T: int = 200,
    step_size: float = 0.01,
    n_trials: int = 50,
    confidence_threshold: float = 0.95,
    seed: int = 0,
) -> Dict:
    """Evaluate goal prediction performance via Monte Carlo simulation.

    Args:
        object_positions: Goal positions (M, 2)
        start_pos: EE start position (2,)
        sigma_bar: Noise covariance upper bound (2, 2)
        sigma: Actual noise covariance (2, 2). Defaults to sigma_bar.
        T: Max timesteps per trial
        step_size: Step size per timestep
        n_trials: Number of trials per goal
        confidence_threshold: Posterior threshold for identification
        seed: Random seed

    Returns:
        Dict with 'accuracy', 'mean_steps', 'per_goal_accuracy',
        'per_goal_mean_steps'
    """
    if sigma is None:
        sigma = sigma_bar.copy()

    sigma_bar_inv = np.linalg.inv(sigma_bar)
    M = object_positions.shape[0]
    rng = np.random.default_rng(seed)

    per_goal_correct = np.zeros(M)
    per_goal_steps = np.zeros(M)

    for g_star in range(M):
        correct = 0
        total_steps = 0
        for _ in range(n_trials):
            pred, steps = simulate_goal_inference(
                object_positions, start_pos, g_star,
                sigma_bar_inv, sigma,
                T=T, step_size=step_size,
                confidence_threshold=confidence_threshold,
                rng=rng,
            )
            if pred == g_star:
                correct += 1
            total_steps += steps

        per_goal_correct[g_star] = correct / n_trials
        per_goal_steps[g_star] = total_steps / n_trials

    return {
        "accuracy": np.mean(per_goal_correct),
        "mean_steps": np.mean(per_goal_steps),
        "per_goal_accuracy": per_goal_correct,
        "per_goal_mean_steps": per_goal_steps,
    }


def evaluate_multistep_task(
    task_config: dict,
    object_positions_dict: dict,
    all_object_positions: np.ndarray,
    start_pos: np.ndarray,
    sigma_bar: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    T_per_step: int = 200,
    step_size: float = 0.01,
    n_trials: int = 50,
    confidence_threshold: float = 0.95,
    seed: int = 0,
) -> Dict:
    """Evaluate intent identification over a multi-step task.

    Args:
        task_config: Parsed YAML task dict
        object_positions_dict: {name: (x, y)} for resolving targets
        all_object_positions: All candidate positions (M, 2)
        start_pos: EE start position (2,)
        sigma_bar: Noise covariance upper bound (2, 2)
        sigma: Actual noise covariance (2, 2)
        T_per_step: Max timesteps per task step
        step_size: Step size per timestep
        n_trials: Number of Monte Carlo trials
        confidence_threshold: Posterior threshold
        seed: Random seed

    Returns:
        Dict with 'mean_total_steps', 'mean_accuracy',
        'per_step_mean_steps', 'per_step_accuracy', 'task_name'
    """
    if sigma is None:
        sigma = sigma_bar.copy()

    sigma_bar_inv = np.linalg.inv(sigma_bar)
    task_targets = parse_task_targets(task_config, object_positions_dict)
    n_steps = len(task_targets)

    rng = np.random.default_rng(seed)

    all_total_steps = []
    all_accuracies = []
    per_step_steps_acc = [[] for _ in range(n_steps)]
    per_step_correct_acc = [[] for _ in range(n_steps)]

    for _ in range(n_trials):
        result = simulate_multistep_goal_inference(
            task_targets, all_object_positions, start_pos,
            sigma_bar_inv, sigma,
            T_per_step=T_per_step, step_size=step_size,
            confidence_threshold=confidence_threshold, rng=rng,
        )
        all_total_steps.append(result["total_steps"])
        all_accuracies.append(result["accuracy"])
        for s in range(n_steps):
            per_step_steps_acc[s].append(result["per_step_steps"][s])
            per_step_correct_acc[s].append(result["per_step_correct"][s])

    task_name = task_config.get("task", {}).get("name", "unknown")
    return {
        "task_name": task_name,
        "n_task_steps": n_steps,
        "mean_total_steps": np.mean(all_total_steps),
        "mean_accuracy": np.mean(all_accuracies),
        "per_step_mean_steps": [np.mean(s) for s in per_step_steps_acc],
        "per_step_accuracy": [np.mean(c) for c in per_step_correct_acc],
        "step_targets": [t["target_name"] for t in task_targets],
    }


# Aliases: task YAMLs use "milk_carton", optimizer uses "milk"
_NAME_ALIASES = {
    "milk": "milk_carton",
    "milk_carton": "milk",
}


def _build_object_positions_dict(object_names: List[str],
                                 positions: np.ndarray) -> dict:
    """Build {name: (x, y)} dict from parallel lists.

    Also adds aliases (e.g., milk <-> milk_carton) so task configs
    that reference either name will resolve correctly.
    """
    d = {name: positions[i] for i, name in enumerate(object_names)}
    for name, alias in _NAME_ALIASES.items():
        if name in d and alias not in d:
            d[alias] = d[name]
    return d


def compare_methods(
    T: int = 50,
    step_size: float = 0.01,
    maxiter: int = 200,
    n_eval_trials: int = 100,
    seed: int = 42,
    task_configs: Optional[List[dict]] = None,
    task_dir: Optional[str] = None,
):
    """Run both optimizers and compare against the un-optimized baseline.

    Args:
        T, step_size, maxiter, n_eval_trials, seed: optimization params
        task_configs: List of parsed YAML task dicts for multi-step eval.
            If None and task_dir is provided, loads all YAMLs from task_dir.
        task_dir: Directory to load task YAML files from.
    """
    import time
    import os
    import yaml

    sigma_bar = 0.1 * np.eye(2)
    sigma = 0.08 * np.eye(2)
    start_pos = DEFAULT_START_POS

    # Load task configs if needed
    if task_configs is None and task_dir is not None:
        task_configs = []
        for fname in sorted(os.listdir(task_dir)):
            if fname.endswith(".yaml"):
                with open(os.path.join(task_dir, fname)) as f:
                    task_configs.append(yaml.safe_load(f))

    results = {}

    # --- Differential Evolution ---
    print("=" * 60)
    print("Method: Differential Evolution")
    print("=" * 60)
    t0 = time.time()
    de_result = optimize_workspace(
        T=T, step_size=step_size, maxiter=maxiter, seed=seed,
        method="differential_evolution", verbose=True,
    )
    de_time = time.time() - t0
    results["differential_evolution"] = de_result
    print(f"Time: {de_time:.1f}s")

    # --- MAP-Elites ---
    print("\n" + "=" * 60)
    print("Method: MAP-Elites (CMA-ME)")
    print("=" * 60)
    t0 = time.time()
    me_result = optimize_workspace(
        T=T, step_size=step_size, maxiter=maxiter, seed=seed,
        method="map_elites", verbose=True,
    )
    me_time = time.time() - t0
    results["map_elites"] = me_result
    print(f"Time: {me_time:.1f}s")

    # --- Single-step evaluation ---
    print("\n" + "=" * 60)
    print("Single-Step Goal Prediction (Monte Carlo)")
    print("=" * 60)

    initial_positions = de_result["initial_positions"]
    object_names = de_result["object_names"]

    configs = {
        "Original (un-optimized)": initial_positions,
        "Differential Evolution": de_result["optimized_positions"],
        "MAP-Elites (CMA-ME)": me_result["optimized_positions"],
    }

    eval_results = {}
    for label, positions in configs.items():
        print(f"\n--- {label} ---")
        ev = evaluate_goal_prediction(
            positions, start_pos, sigma_bar, sigma,
            n_trials=n_eval_trials, seed=42,
        )
        eval_results[label] = ev
        print(f"Accuracy: {ev['accuracy']:.2%}")
        print(f"Mean steps to identify: {ev['mean_steps']:.1f}")
        for i, name in enumerate(object_names):
            print(f"  {name}: accuracy={ev['per_goal_accuracy'][i]:.2%}, "
                  f"steps={ev['per_goal_mean_steps'][i]:.1f}")

    # --- Single-step summary ---
    orig_steps = eval_results["Original (un-optimized)"]["mean_steps"]
    print("\n" + "=" * 60)
    print("Single-Step Summary")
    print("=" * 60)
    print(f"{'Method':<30} {'Slack':>10} {'Accuracy':>10} {'Steps':>8} {'Speedup':>10}")
    print("-" * 70)

    rows = [
        ("Original (un-optimized)", de_result["initial_slack"],
         eval_results["Original (un-optimized)"]),
        ("Differential Evolution", de_result["optimized_slack"],
         eval_results["Differential Evolution"]),
        ("MAP-Elites (CMA-ME)", me_result["optimized_slack"],
         eval_results["MAP-Elites (CMA-ME)"]),
    ]
    for label, slack, ev in rows:
        speedup = orig_steps / max(ev["mean_steps"], 0.1)
        print(f"{label:<30} {slack:>10.2f} {ev['accuracy']:>9.2%} "
              f"{ev['mean_steps']:>8.1f} {speedup:>9.2f}x")

    # --- Multi-step task evaluation ---
    if task_configs:
        print("\n" + "=" * 60)
        print("Multi-Step Task Evaluation (Monte Carlo)")
        print("=" * 60)

        for tc in task_configs:
            task_name = tc.get("task", {}).get("name", "unknown")
            n_steps = len(tc.get("task", {}).get("steps", []))
            print(f"\n{'─' * 60}")
            print(f"Task: {task_name} ({n_steps} steps)")
            print(f"{'─' * 60}")

            task_eval_results = {}
            for label, positions in configs.items():
                obj_dict = _build_object_positions_dict(object_names, positions)
                ev = evaluate_multistep_task(
                    tc, obj_dict, positions, start_pos,
                    sigma_bar, sigma,
                    T_per_step=200, step_size=0.01,
                    n_trials=n_eval_trials // 2,  # fewer trials (more steps)
                    seed=42,
                )
                task_eval_results[label] = ev

                print(f"\n  {label}:")
                print(f"    Total steps: {ev['mean_total_steps']:.1f}  "
                      f"Accuracy: {ev['mean_accuracy']:.2%}")
                for s_idx in range(ev["n_task_steps"]):
                    target = ev["step_targets"][s_idx]
                    s_steps = ev["per_step_mean_steps"][s_idx]
                    s_acc = ev["per_step_accuracy"][s_idx]
                    print(f"      Step {s_idx+1} ({target}): "
                          f"steps={s_steps:.1f}, acc={s_acc:.2%}")

            # Task summary
            orig_total = task_eval_results["Original (un-optimized)"]["mean_total_steps"]
            print(f"\n  {'Method':<30} {'Total Steps':>12} {'Accuracy':>10} {'Speedup':>10}")
            print(f"  {'-' * 64}")
            for label in configs:
                ev = task_eval_results[label]
                speedup = orig_total / max(ev["mean_total_steps"], 0.1)
                print(f"  {label:<30} {ev['mean_total_steps']:>12.1f} "
                      f"{ev['mean_accuracy']:>9.2%} {speedup:>9.2f}x")

    return results, eval_results


if __name__ == "__main__":
    import os
    task_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "config", "tasks",
    )
    compare_methods(task_dir=task_dir)
