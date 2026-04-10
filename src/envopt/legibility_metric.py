"""Legibility metric based on Dragan et al. (2013).

Computes trajectory legibility using the Boltzmann-rational path
efficiency model:

    P(g | traj) ∝ exp(-β · C(traj, g))
    C(traj, g) = (L(traj) + d(ee, g)) / d(start, g)

where L is the cumulative path length, d(ee, g) is the remaining
distance to goal g, and d(start, g) is the initial distance.
"""
import numpy as np
from typing import List, Tuple


class Layout:
    def __init__(self, starting_pos: Tuple, object_positions: List[Tuple], object_labels: List[int]):
        self.starting_pos = starting_pos
        self.object_positions = object_positions
        self.object_labels = object_labels


class Task:
    def __init__(self, object_labels: List[int]):
        self.object_labels = object_labels


def get_closest_goal(layout: Layout, starting_position: tuple, goal: int):
    """Return the index of the closest object with the given label."""
    object_labels = np.array(layout.object_labels)
    goal_indices = np.where(object_labels == goal)[0]
    closest_goal_idx = None
    start_to_goal_dist = np.inf
    for goal_idx in goal_indices:
        dist = np.linalg.norm(np.array(starting_position) - layout.object_positions[goal_idx][:2])
        if dist < start_to_goal_dist:
            start_to_goal_dist = dist
            closest_goal_idx = goal_idx
    return closest_goal_idx


def prob_traj_given_goal(layout: Layout, trajectory: List,
                         goal_pos: Tuple[float, float], beta: float = 5.0):
    """Compute P(traj | goal) using Dragan's path efficiency model.

    P(traj | g) ∝ exp(-β · (L + d_remaining) / d_start)

    Args:
        layout: Layout with starting position
        trajectory: list of (x, y) positions
        goal_pos: goal position (x, y)
        beta: rationality coefficient (higher = more rational user)

    Returns:
        Unnormalized probability (to be normalized over all goals)
    """
    goal_pos = np.array(goal_pos)[:2]
    start_pos = np.array(layout.starting_pos)[:2]

    d_start = np.linalg.norm(start_pos - goal_pos)
    if d_start < 1e-6:
        # Goal is at start position — assign high probability
        return 1.0

    if len(trajectory) == 0:
        return np.exp(-beta)  # cost = d_start / d_start = 1

    final_pos = np.array(trajectory[-1])[:2]
    d_remaining = np.linalg.norm(final_pos - goal_pos)

    # Cumulative path length
    path_length = 0.0
    for i in range(1, len(trajectory)):
        path_length += np.linalg.norm(
            np.array(trajectory[i])[:2] - np.array(trajectory[i - 1])[:2])

    cost = (path_length + d_remaining) / d_start
    return np.exp(-beta * cost)


def compute_trajectory(start, goal, step_size):
    """Compute a straight-line trajectory from start to goal."""
    start = np.array(start)[:2]
    goal = np.array(goal)[:2]

    direction = goal - start
    distance = np.linalg.norm(direction)

    if distance == 0:
        return [start]

    direction = direction / distance
    num_steps = int(np.ceil(distance / step_size))
    num_steps = min(num_steps, 500)
    trajectory = [start + i * step_size * direction for i in range(num_steps)]
    trajectory.append(goal)
    return np.array(trajectory)


def cross_entropy_loss(preds, targets):
    """Cross-entropy loss between predicted and true distributions."""
    preds = np.clip(preds, 1e-8, 1.0)
    return -np.sum(targets * np.log(preds))


def legibility_loss(layout: Layout, task_list: List[Task], beta: float = 5.0):
    """Compute legibility loss of a layout using Dragan's model.

    For each goal in each task, simulates a straight-line trajectory
    and computes cross-entropy between the Boltzmann posterior and the
    true goal at 25%, 50%, 75% along the path.

    Lower loss = more legible layout.

    Args:
        layout: Layout with starting position and object positions
        task_list: list of Task objects
        beta: rationality coefficient for the path efficiency model

    Returns:
        Total cross-entropy loss (lower is better)
    """
    loss = 0
    for task in task_list:
        for goal_label_idx in range(len(task.object_labels)):
            goal_label = task.object_labels[goal_label_idx]
            goal_layout_indices = [i for i, label in enumerate(layout.object_labels)
                                   if label == goal_label]

            for goal_layout_idx in goal_layout_indices:
                goal_position = layout.object_positions[goal_layout_idx]
                trajectory = compute_trajectory(
                    layout.starting_pos,
                    layout.object_positions[goal_layout_idx],
                    step_size=0.02)
                trajectory = trajectory[:100]

                percentages = [0.25, 0.5, 0.75]
                for percent in percentages:
                    traj = trajectory[:int(percent * len(trajectory))]

                    # Compute Boltzmann posterior over all goals
                    prob_dist = np.zeros(len(task.object_labels))
                    for i, candidate_goal in enumerate(task.object_labels):
                        if candidate_goal != goal_label:
                            closest_goal_idx = get_closest_goal(
                                layout, layout.starting_pos, candidate_goal)
                            prob_dist[i] = prob_traj_given_goal(
                                layout, traj,
                                layout.object_positions[closest_goal_idx],
                                beta=beta)
                        else:
                            prob_dist[i] = prob_traj_given_goal(
                                layout, traj, goal_position, beta=beta)

                    # Normalize to get posterior
                    total = np.sum(prob_dist)
                    if total > 0:
                        prob_dist /= total
                    else:
                        prob_dist = np.ones_like(prob_dist) / len(prob_dist)

                    # Cross-entropy with true distribution
                    true_dist = np.zeros(len(task.object_labels))
                    true_dist[goal_label_idx] = 1
                    loss += cross_entropy_loss(prob_dist, true_dist) * percent

                    if np.isnan(loss) or np.isinf(loss):
                        print(layout.starting_pos,
                              layout.object_positions[goal_layout_idx])
                        print(prob_dist, true_dist)
                        raise ValueError("Loss is NaN or Inf")
    return loss
