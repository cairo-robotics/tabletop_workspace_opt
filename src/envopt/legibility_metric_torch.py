"""Torch version of legibility metric (Dragan et al. 2013).

Differentiable implementation for gradient-based optimization (DQD).

    P(g | traj) ∝ exp(-β · (L + d_remaining) / d_start)
"""
import torch
from typing import List, Tuple


class Layout:
    def __init__(self, starting_pos: torch.Tensor, object_positions: torch.Tensor,
                 object_labels: torch.Tensor):
        self.starting_pos = starting_pos
        self.object_positions = object_positions
        self.object_labels = object_labels


class Task:
    def __init__(self, object_labels: List[int]):
        self.object_labels = object_labels


def torch_cross_entropy(preds, targets):
    """Cross-entropy loss.

    preds: [B, K] predicted probs
    targets: [B, K] one-hot ground truth
    """
    return -(targets * torch.log(preds + 1e-8)).sum(dim=-1)


def compute_trajectory_torch(start, goal, step_size=0.02):
    """Compute straight-line trajectory from start to goal (2D)."""
    goal = goal[:2]
    direction = goal - start
    dist = torch.norm(direction)

    if dist.item() == 0:
        return start.unsqueeze(0)

    direction = direction / dist
    steps = int(torch.ceil(dist / step_size).item())
    steps = min(steps, 500)

    trajectory = torch.stack([start + i * step_size * direction for i in range(steps)])
    return torch.cat([trajectory, goal.unsqueeze(0)], dim=0)


def prob_traj_given_goal_torch(layout: Layout, traj: torch.Tensor,
                                goal_idx: int, beta: float = 5.0):
    """Compute P(traj | goal) using Dragan's path efficiency model.

    P(traj | g) ∝ exp(-β · (L + d_remaining) / d_start)
    """
    goal_pos = layout.object_positions[goal_idx][:2]
    start_pos = layout.starting_pos

    d_start = torch.norm(start_pos - goal_pos)
    if d_start.item() < 1e-6:
        return torch.tensor(1.0, requires_grad=True)

    final_pos = traj[-1]
    d_remaining = torch.norm(final_pos - goal_pos)
    path_length = torch.sum(torch.norm(traj[1:] - traj[:-1], dim=1))

    cost = (path_length + d_remaining) / d_start
    return torch.exp(-beta * cost)


def legibility_loss_torch(layout: Layout, task_list: List[Task],
                          beta: float = 5.0, verbose: bool = False):
    """Compute legibility loss using Dragan's path efficiency model.

    For each goal, simulates a straight-line trajectory and computes
    cross-entropy between the Boltzmann posterior and the true goal
    at 25%, 50%, 75% along the path.

    Lower loss = more legible layout.
    """
    loss = torch.tensor(0.0, requires_grad=True)

    for task in task_list:
        for goal_label_idx, goal_label in enumerate(task.object_labels):
            goal_layout_indices = [i for i, label in enumerate(layout.object_labels)
                                   if label == goal_label]

            for goal_layout_idx in goal_layout_indices:
                trajectory = compute_trajectory_torch(
                    layout.starting_pos, layout.object_positions[goal_layout_idx])
                trajectory = trajectory[:100]

                for percent in [0.25, 0.5, 0.75]:
                    cutoff = max(1, int(len(trajectory) * percent))
                    partial_traj = trajectory[:cutoff]

                    probs = []
                    for candidate_label in task.object_labels:
                        if candidate_label == goal_label:
                            goal_idx = goal_layout_idx
                        else:
                            candidate_indices = [i for i, l in enumerate(layout.object_labels)
                                                if l == candidate_label]
                            dists = [torch.norm(layout.starting_pos - layout.object_positions[i][:2])
                                    for i in candidate_indices]
                            closest_idx = candidate_indices[torch.argmin(torch.stack(dists)).item()]
                            goal_idx = closest_idx
                        p = prob_traj_given_goal_torch(
                            layout, partial_traj, goal_idx, beta=beta)
                        probs.append(p)

                    prob_dist = torch.stack(probs)
                    prob_dist = prob_dist / (prob_dist.sum() + 1e-8)

                    true_dist = torch.zeros_like(prob_dist)
                    true_dist[goal_label_idx] = 1.0

                    if verbose:
                        print(f"Goal: {goal_label}, percent: {percent}, "
                              f"prob_dist: {prob_dist.detach()}, "
                              f"true_dist: {true_dist}")

                    ce = torch_cross_entropy(
                        prob_dist.unsqueeze(0), true_dist.unsqueeze(0))
                    loss = loss + ce * percent

    return loss


def batched_legibility_loss(positions_batch, starting_pos, object_labels,
                            task_list, beta=5.0):
    """Batched legibility loss for diffusion/batch optimization.

    Args:
        positions_batch: [B, N, 2] tensor
        starting_pos: [2] tensor
        object_labels: list[int] (length N)
        task_list: list[Task]
        beta: rationality coefficient

    Returns:
        scalar loss averaged over the batch
    """
    B = positions_batch.shape[0]
    total_loss = torch.zeros(B, device=positions_batch.device)

    for b in range(B):
        pos = positions_batch[b]
        layout = Layout(
            starting_pos=starting_pos,
            object_positions=pos,
            object_labels=object_labels,
        )
        total_loss[b] = legibility_loss_torch(layout, task_list, beta=beta)

    return total_loss.mean()
