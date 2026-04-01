"""Intent Separability Objective for Workspace Optimization

Implements the workspace design objective from Section 1.6 of the Shared
Autonomy Notes. Optimizes object placements θ to maximize worst-case
"intent separability slack" under bounded Gaussian joystick noise.

The objective:
    max_θ  min_{g≠g*}  [ m̃(g;θ) - √Ṽ(g;θ) · Φ⁻¹(1-α_g) ]

where:
    Ṽ(g;θ) = Σ_t Δ_t(g;θ)^T Σ̄⁻¹ Δ_t(g;θ)   (cumulative separation)
    m̃(g;θ) = ½Ṽ(g;θ) - log(p(g)/p(g*))       (separation margin)
    Δ_t(g;θ) = μ_t(g;θ) - μ_t(g*;θ)           (mean command difference)
    μ_t(g;θ) = (goal_pos - x_t) / ||goal_pos - x_t||  (point-toward-goal policy)
"""

import numpy as np
from scipy.stats import norm
from typing import List, Tuple, Optional


def goal_conditioned_policy(ee_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """Compute the mean joystick command: normalized direction toward goal.

    μ_t(g) = π_h(x_t, g) = (g - x_t) / ||g - x_t||

    Args:
        ee_pos: End-effector position (d,)
        goal_pos: Goal position (d,)

    Returns:
        Mean command vector (d,). Zero vector if already at goal.
    """
    diff = goal_pos - ee_pos
    dist = np.linalg.norm(diff)
    if dist < 1e-8:
        return np.zeros_like(diff)
    return diff / dist


def simulate_trajectory(start_pos: np.ndarray, goal_pos: np.ndarray,
                        T: int, step_size: float) -> np.ndarray:
    """Simulate T timesteps of moving toward a goal with fixed step size.

    Args:
        start_pos: Starting EE position (d,)
        goal_pos: True goal position (d,)
        T: Number of timesteps
        step_size: Distance moved per timestep

    Returns:
        Trajectory array of shape (T, d) — EE positions at each timestep.
    """
    d = start_pos.shape[0]
    trajectory = np.zeros((T, d))
    pos = start_pos.copy()
    for t in range(T):
        trajectory[t] = pos
        direction = goal_pos - pos
        dist = np.linalg.norm(direction)
        if dist < 1e-8:
            # Already at goal, stay put for remaining steps
            trajectory[t:] = pos
            break
        direction = direction / dist
        pos = pos + step_size * direction
        # Don't overshoot
        if np.linalg.norm(pos - start_pos) > np.linalg.norm(goal_pos - start_pos):
            pos = goal_pos.copy()
    return trajectory


def compute_pairwise_separation(trajectory: np.ndarray,
                                true_goal: np.ndarray,
                                alt_goal: np.ndarray,
                                sigma_bar_inv: np.ndarray) -> float:
    """Compute Ṽ(g;θ) — the cumulative Mahalanobis separation.

    Ṽ(g;θ) = Σ_t Δ_t(g;θ)^T Σ̄⁻¹ Δ_t(g;θ)

    where Δ_t(g;θ) = μ_t(g;θ) - μ_t(g*;θ)

    Args:
        trajectory: EE positions (T, d) under the true goal trajectory
        true_goal: True goal position (d,)
        alt_goal: Alternative goal position (d,)
        sigma_bar_inv: Inverse of noise covariance upper bound (d, d)

    Returns:
        Cumulative separation scalar Ṽ(g;θ)
    """
    T = trajectory.shape[0]
    V_tilde = 0.0
    for t in range(T):
        mu_alt = goal_conditioned_policy(trajectory[t], alt_goal)
        mu_true = goal_conditioned_policy(trajectory[t], true_goal)
        delta = mu_alt - mu_true
        V_tilde += delta @ sigma_bar_inv @ delta
    return V_tilde


def compute_separation_margin(V_tilde: float,
                              log_prior_ratio: float) -> float:
    """Compute m̃(g;θ) — the separation margin.

    m̃(g;θ) = ½Ṽ(g;θ) - log(p(g)/p(g*))

    Args:
        V_tilde: Cumulative separation Ṽ(g;θ)
        log_prior_ratio: log(p(g) / p(g*))

    Returns:
        Separation margin scalar
    """
    return 0.5 * V_tilde - log_prior_ratio


def compute_separability_slack(V_tilde: float,
                               m_tilde: float,
                               alpha_g: float) -> float:
    """Compute the intent separability slack for one goal pair.

    slack = m̃(g;θ) - √Ṽ(g;θ) · Φ⁻¹(1-α_g)

    Args:
        V_tilde: Cumulative separation
        m_tilde: Separation margin
        alpha_g: Per-goal failure probability budget

    Returns:
        Separability slack scalar
    """
    quantile = norm.ppf(1 - alpha_g)
    return m_tilde - np.sqrt(max(V_tilde, 0.0)) * quantile


def intent_separability_objective(
    object_positions: np.ndarray,
    start_pos: np.ndarray,
    sigma_bar_inv: np.ndarray,
    T: int = 50,
    step_size: float = 0.01,
    alpha: float = 0.05,
    prior: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Compute the workspace design objective from Section 1.6.

    For each possible true goal g*, computes the worst-case separability
    slack across all alternatives g ≠ g*, then returns the overall
    worst case (min over all g*).

    Objective: max_θ  min_{g*}  min_{g≠g*} slack(g, g*; θ)

    Args:
        object_positions: Goal positions (M, d) in world frame
        start_pos: Starting EE position (d,)
        sigma_bar_inv: Inverse covariance upper bound (d, d)
        T: Number of trajectory timesteps
        step_size: Step size for trajectory simulation
        alpha: Total failure probability budget
        prior: Goal prior probabilities (M,). Uniform if None.

    Returns:
        (worst_case_slack, per_goal_slacks) where per_goal_slacks[i] is the
        worst-case slack when g* = goal i.
    """
    M = object_positions.shape[0]
    if prior is None:
        prior = np.ones(M) / M

    # Distribute alpha equally across alternatives (union bound)
    alpha_g = alpha / (M - 1) if M > 1 else alpha

    per_goal_slacks = np.zeros(M)

    for star_idx in range(M):
        true_goal = object_positions[star_idx]
        # Simulate trajectory toward the true goal
        trajectory = simulate_trajectory(start_pos, true_goal, T, step_size)

        min_slack = np.inf
        for alt_idx in range(M):
            if alt_idx == star_idx:
                continue
            alt_goal = object_positions[alt_idx]
            log_prior_ratio = np.log(prior[alt_idx] / prior[star_idx])

            V_tilde = compute_pairwise_separation(
                trajectory, true_goal, alt_goal, sigma_bar_inv
            )
            m_tilde = compute_separation_margin(V_tilde, log_prior_ratio)
            slack = compute_separability_slack(V_tilde, m_tilde, alpha_g)
            min_slack = min(min_slack, slack)

        per_goal_slacks[star_idx] = min_slack

    worst_case_slack = np.min(per_goal_slacks)
    return worst_case_slack, per_goal_slacks


def intent_separability_for_optimizer(
    flat_positions: np.ndarray,
    start_pos: np.ndarray,
    sigma_bar_inv: np.ndarray,
    d: int = 2,
    T: int = 50,
    step_size: float = 0.01,
    alpha: float = 0.05,
    prior: Optional[np.ndarray] = None,
) -> float:
    """Wrapper that takes a flat position vector for use with scipy optimizers.

    Args:
        flat_positions: Flattened (M*d,) array of goal positions
        start_pos: Starting EE position (d,)
        sigma_bar_inv: Inverse covariance upper bound (d, d)
        d: Dimensionality of positions
        T, step_size, alpha, prior: Passed through to objective.

    Returns:
        Negative worst-case slack (for minimization).
    """
    M = len(flat_positions) // d
    object_positions = flat_positions.reshape(M, d)
    worst_slack, _ = intent_separability_objective(
        object_positions, start_pos, sigma_bar_inv,
        T=T, step_size=step_size, alpha=alpha, prior=prior,
    )
    # Negate because scipy minimizes
    return -worst_slack


def simulate_goal_inference(
    object_positions: np.ndarray,
    start_pos: np.ndarray,
    true_goal_idx: int,
    sigma_bar_inv: np.ndarray,
    sigma: np.ndarray,
    T: int = 200,
    step_size: float = 0.01,
    prior: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, int]:
    """Simulate noisy joystick inputs and measure how many steps until
    the MAP estimator correctly identifies the true goal with high confidence.

    Args:
        object_positions: Goal positions (M, d)
        start_pos: Starting EE position (d,)
        true_goal_idx: Index of the true goal
        sigma_bar_inv: Inverse covariance upper bound (d, d)
        sigma: Actual noise covariance (d, d) for generating noise
        T: Maximum number of timesteps
        step_size: Step size per timestep
        prior: Goal prior (M,). Uniform if None.
        confidence_threshold: Posterior threshold for "identified"
        rng: Random number generator

    Returns:
        (predicted_goal_idx, steps_to_identify) where steps_to_identify
        is the first timestep the MAP estimate is correct with posterior
        >= confidence_threshold, or T if never reached.
    """
    if rng is None:
        rng = np.random.default_rng()

    M, d_dim = object_positions.shape
    if prior is None:
        prior = np.ones(M) / M

    true_goal = object_positions[true_goal_idx]
    pos = start_pos.copy()

    # Accumulate log-likelihoods (score function S(g))
    log_scores = np.log(prior).copy()
    sigma_inv = np.linalg.inv(sigma)

    steps_to_identify = T
    final_prediction = np.argmax(log_scores)

    for t in range(T):
        # True mean command (toward true goal)
        mu_true = goal_conditioned_policy(pos, true_goal)

        # Noisy observation
        noise = rng.multivariate_normal(np.zeros(d_dim), sigma)
        u_t = mu_true + noise

        # Update score for each goal
        for g in range(M):
            mu_g = goal_conditioned_policy(pos, object_positions[g])
            diff = u_t - mu_g
            log_scores[g] -= 0.5 * diff @ sigma_inv @ diff

        # MAP estimate
        predicted = np.argmax(log_scores)

        # Compute posterior (softmax of log_scores)
        log_scores_shifted = log_scores - np.max(log_scores)
        posterior = np.exp(log_scores_shifted)
        posterior /= posterior.sum()

        if predicted == true_goal_idx and posterior[predicted] >= confidence_threshold:
            if steps_to_identify == T:
                steps_to_identify = t + 1

        # Move EE toward true goal
        pos = pos + step_size * mu_true
        if np.linalg.norm(pos - start_pos) > np.linalg.norm(true_goal - start_pos):
            pos = true_goal.copy()

    final_prediction = np.argmax(log_scores)
    return final_prediction, steps_to_identify


# ---------------------------------------------------------------------------
# Multi-step task evaluation
# ---------------------------------------------------------------------------

def parse_task_targets(task_config: dict, object_positions: dict) -> List[dict]:
    """Extract the sequence of reach targets from a task YAML config.

    For each step, determines where the EE needs to go:
      - pick: reach the named object
      - place: reach the destination (reference + offset or absolute position)

    Args:
        task_config: Parsed YAML dict with 'task.steps'
        object_positions: Dict mapping object name -> (x, y) position

    Returns:
        List of dicts, each with:
          'action': 'pick' or 'place'
          'target_name': human-readable label
          'target_pos': np.ndarray (2,) position
    """
    steps = task_config["task"]["steps"]
    targets = []

    for step in steps:
        action = step["action"]
        if action == "pick":
            obj_name = step["object"]
            pos = np.array(object_positions[obj_name][:2])
            targets.append({
                "action": "pick",
                "target_name": obj_name,
                "target_pos": pos,
            })
        elif action == "place":
            dest = step["destination"]
            if "reference" in dest:
                ref_name = dest["reference"]
                ref_pos = np.array(object_positions[ref_name][:2])
                offset = np.array([dest["offset"]["x"], dest["offset"]["y"]])
                pos = ref_pos + offset
                label = f"place_near_{ref_name}"
            else:
                pos = np.array([dest["position"]["x"], dest["position"]["y"]])
                label = f"place_at_{pos[0]:.2f}_{pos[1]:.2f}"
            targets.append({
                "action": "place",
                "target_name": label,
                "target_pos": pos,
            })
        elif action == "move_to":
            pos_dict = step.get("position", step.get("destination", {}).get("position", {}))
            pos = np.array([pos_dict["x"], pos_dict["y"]])
            targets.append({
                "action": "move_to",
                "target_name": "move_to",
                "target_pos": pos,
            })

    return targets


def simulate_multistep_goal_inference(
    task_targets: List[dict],
    all_object_positions: np.ndarray,
    start_pos: np.ndarray,
    sigma_bar_inv: np.ndarray,
    sigma: np.ndarray,
    T_per_step: int = 200,
    step_size: float = 0.01,
    confidence_threshold: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Simulate intent identification across a multi-step task sequence.

    At each task step, the user reaches toward the step's target. The robot
    must identify which of the candidate goals the user intends. Candidates
    include all object positions plus any place destinations.

    The EE starts at start_pos, and after each step completes, the EE is
    assumed to be at the step's target position (simulating successful
    execution).

    Args:
        task_targets: List from parse_task_targets()
        all_object_positions: All goal candidate positions (M, 2)
        start_pos: Initial EE position (2,)
        sigma_bar_inv: Inverse covariance bound (2, 2)
        sigma: Actual noise covariance (2, 2)
        T_per_step: Max timesteps per step for identification
        step_size: EE movement per timestep
        confidence_threshold: Posterior threshold
        rng: Random generator

    Returns:
        Dict with 'total_steps', 'per_step_steps', 'per_step_correct',
        'accuracy', 'step_details'
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build candidate set: all objects + unique place destinations
    candidate_positions = list(all_object_positions)
    candidate_names = [f"obj_{i}" for i in range(len(all_object_positions))]

    # Add unique place/move targets that don't coincide with existing objects
    for target in task_targets:
        pos = target["target_pos"]
        is_dup = any(np.linalg.norm(pos - c) < 0.05 for c in candidate_positions)
        if not is_dup:
            candidate_positions.append(pos)
            candidate_names.append(target["target_name"])

    candidates = np.array(candidate_positions)
    M = len(candidates)
    sigma_inv = np.linalg.inv(sigma)

    current_pos = start_pos.copy()
    per_step_steps = []
    per_step_correct = []
    step_details = []

    for step_idx, target in enumerate(task_targets):
        target_pos = target["target_pos"]

        # Find which candidate index matches this target
        dists = np.linalg.norm(candidates - target_pos, axis=1)
        true_idx = np.argmin(dists)

        # Run inference for this step
        log_scores = np.zeros(M)  # uniform prior (reset each step)
        pos = current_pos.copy()
        steps_to_id = T_per_step
        correct = False

        for t in range(T_per_step):
            mu_true = goal_conditioned_policy(pos, target_pos)
            if np.linalg.norm(mu_true) < 1e-8:
                break  # already at target

            noise = rng.multivariate_normal(np.zeros(2), sigma)
            u_t = mu_true + noise

            for g in range(M):
                mu_g = goal_conditioned_policy(pos, candidates[g])
                diff = u_t - mu_g
                log_scores[g] -= 0.5 * diff @ sigma_inv @ diff

            predicted = np.argmax(log_scores)
            log_shifted = log_scores - np.max(log_scores)
            posterior = np.exp(log_shifted)
            posterior /= posterior.sum()

            if predicted == true_idx and posterior[predicted] >= confidence_threshold:
                if steps_to_id == T_per_step:
                    steps_to_id = t + 1
                    correct = True

            pos = pos + step_size * mu_true
            total_dist = np.linalg.norm(target_pos - current_pos)
            if total_dist > 0 and np.linalg.norm(pos - current_pos) > total_dist:
                pos = target_pos.copy()

        final_pred = np.argmax(log_scores)
        if final_pred == true_idx:
            correct = True

        per_step_steps.append(steps_to_id)
        per_step_correct.append(correct)
        step_details.append({
            "step": step_idx,
            "action": target["action"],
            "target": target["target_name"],
            "steps_to_identify": steps_to_id,
            "correct": correct,
        })

        # After step completes, EE is at target position
        current_pos = target_pos.copy()

    return {
        "total_steps": sum(per_step_steps),
        "per_step_steps": per_step_steps,
        "per_step_correct": per_step_correct,
        "accuracy": np.mean(per_step_correct),
        "step_details": step_details,
        "n_task_steps": len(task_targets),
    }
