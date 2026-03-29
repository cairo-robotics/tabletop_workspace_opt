#!/usr/bin/env python3
"""Optimize object AND place/pour destination positions for intent separability.

Extends the workspace optimizer to also optimize where objects get placed,
ensuring pour vs place goals are well-separated for intent inference.

The optimizer jointly optimizes:
  - Initial object positions (for pick disambiguation)
  - Place destination positions (for pour vs place disambiguation)

Constraints:
  - All positions within table bounds
  - Minimum distance between any two positions (no collisions)
  - Place positions within robot reach
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "envopt"))

import numpy as np
from scipy.optimize import differential_evolution
from intent_separability import intent_separability_for_optimizer

# Table bounds (world frame)
TABLE_BOUNDS_X = (0.30, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)
MIN_DIST = 0.12

# Robot EE home position
START_POS = np.array([0.479, -0.060])


def overlap_penalty(positions, min_dist=MIN_DIST):
    """Penalty for any two positions being too close."""
    M = len(positions)
    penalty = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return penalty


def multi_state_objective(flat_vars, state_goal_indices, start_pos,
                          sigma_bar_inv, T=50, step_size=0.01, alpha=0.05):
    """Compute worst-case separability across all task states.

    flat_vars: flat array of ALL target positions (objects + destinations)
    state_goal_indices: list of lists — for each state, which indices
                        in flat_vars correspond to the valid goals
    """
    d = 2
    all_positions = flat_vars.reshape(-1, d)

    # Compute separability for each state with multiple goals
    worst_slack = float('inf')
    for goal_indices in state_goal_indices:
        if len(goal_indices) <= 1:
            continue
        state_positions = all_positions[goal_indices]
        neg_slack = intent_separability_for_optimizer(
            state_positions.flatten(), start_pos, sigma_bar_inv,
            d=d, T=T, step_size=step_size, alpha=alpha)
        slack = -neg_slack
        worst_slack = min(worst_slack, slack)

    # Overlap penalty across ALL positions
    penalty = 100.0 * overlap_penalty(all_positions, MIN_DIST)

    return -(worst_slack - penalty)


def optimize_breakfast_layout():
    """Optimize breakfast scene: object positions + place destinations."""
    sigma_bar = 0.1 * np.eye(2)
    sigma_bar_inv = np.linalg.inv(sigma_bar)

    # Define all target positions to optimize:
    # Index 0: bowl (pick target + pour reference)
    # Index 1: cereal (pick target)
    # Index 2: banana (pick target)
    # Index 3: milk (pick target)
    # Index 4: cereal_place (place destination for cereal)
    # Index 5: milk_place (place destination for milk)
    # Pour targets are at bowl position (projected 2D), so they share index 0

    target_names = [
        "bowl", "cereal", "banana", "milk",
        "cereal_place", "milk_place",
    ]

    initial_positions = np.array([
        [0.600, 0.300],   # bowl
        [0.480, -0.250],  # cereal
        [0.500, 0.050],   # banana
        [0.780, 0.300],   # milk
        [0.450, 0.150],   # cereal_place (bowl + (-0.15, -0.15))
        [0.750, 0.300],   # milk_place (bowl + (0.15, 0))
    ])

    # States with multiple goals and their target indices:
    # initial: pick_cereal(1), pick_banana(2), pick_milk(3)
    # holding_cereal: pour_cereal(0=bowl), place_cereal(4)
    # cereal_done: pick_banana(2), pick_milk(3)
    # holding_milk: pour_milk(0=bowl), place_milk(5)
    state_goal_indices = [
        [1, 2, 3],    # initial: pick cereal, banana, milk
        [0, 4],        # holding_cereal: pour(bowl) vs place_cereal
        [2, 3],        # cereal_done: pick banana vs milk
        [0, 5],        # holding_milk: pour(bowl) vs place_milk
    ]

    M = len(target_names)
    d = 2

    # Bounds
    bounds = [(TABLE_BOUNDS_X[0], TABLE_BOUNDS_X[1]),
              (TABLE_BOUNDS_Y[0], TABLE_BOUNDS_Y[1])] * M

    # Compute initial slack
    init_obj = multi_state_objective(
        initial_positions.flatten(), state_goal_indices,
        START_POS, sigma_bar_inv)
    print(f"Initial worst-case slack: {-init_obj:.2f}")

    # Optimize
    print("\nRunning differential evolution...")
    result = differential_evolution(
        multi_state_objective,
        bounds=bounds,
        args=(state_goal_indices, START_POS, sigma_bar_inv),
        maxiter=200,
        seed=42,
        tol=1e-6,
        polish=True,
        disp=True,
    )

    opt_positions = result.x.reshape(-1, d)
    opt_obj = multi_state_objective(
        result.x, state_goal_indices,
        START_POS, sigma_bar_inv)

    print(f"\nOptimized worst-case slack: {-opt_obj:.2f}")
    print(f"Improvement: {-opt_obj - (-init_obj):.2f}")

    print("\nOptimized positions:")
    for i, name in enumerate(target_names):
        pos = opt_positions[i]
        print(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f})")

    # Compute place offsets relative to bowl
    bowl_pos = opt_positions[0]
    cereal_place = opt_positions[4]
    milk_place = opt_positions[5]
    print(f"\nPlace offsets (relative to bowl at ({bowl_pos[0]:.3f}, {bowl_pos[1]:.3f})):")
    cereal_off = cereal_place - bowl_pos
    milk_off = milk_place - bowl_pos
    print(f"  cereal_place offset: x={cereal_off[0]:.3f}, y={cereal_off[1]:.3f}")
    print(f"  milk_place offset: x={milk_off[0]:.3f}, y={milk_off[1]:.3f}")

    # Per-state analysis
    print("\nPer-state separability:")
    for i, indices in enumerate(state_goal_indices):
        if len(indices) <= 1:
            continue
        state_pos = opt_positions[indices]
        neg_slack = intent_separability_for_optimizer(
            state_pos.flatten(), START_POS, sigma_bar_inv)
        names = [target_names[j] for j in indices]
        print(f"  State {i} ({', '.join(names)}): slack = {-neg_slack:.2f}")

    return {
        "target_names": target_names,
        "initial_positions": initial_positions,
        "optimized_positions": opt_positions,
        "state_goal_indices": state_goal_indices,
    }


if __name__ == "__main__":
    result = optimize_breakfast_layout()
