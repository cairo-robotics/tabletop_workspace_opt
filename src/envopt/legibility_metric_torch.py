import torch
from typing import List, Tuple


class Layout:
    def __init__(self, starting_pos: torch.Tensor, object_positions: torch.Tensor, object_labels: torch.Tensor):
        self.starting_pos = starting_pos
        self.object_positions = object_positions
        self.object_labels = object_labels


class Task:
    def __init__(self, object_labels: List[int]):
        self.object_labels = object_labels


def torch_cross_entropy(preds, targets):
    """
    preds: [B, K] predicted probs
    targets: [B, K] one-hot ground truth
    """
    return -(targets * torch.log(preds + 1e-8)).sum(dim=-1)  # avoid log(0)


def compute_trajectory_torch(start, goal, step_size=0.02):
    goal = goal[:2]
    direction = goal - start
    dist = torch.norm(direction)

    if dist.item() == 0:
        return start.unsqueeze(0)

    direction = direction / dist
    steps = int(torch.ceil(dist / step_size).item())
    steps = min(steps, 500)  # Limit to 500 steps for performance

    trajectory = torch.stack([start + i * step_size * direction for i in range(steps)])
    return torch.cat([trajectory, goal.unsqueeze(0)], dim=0)


def prob_traj_given_goal_torch(layout: Layout, traj: torch.Tensor, goal_idx: int):
    goal_pos = layout.object_positions[goal_idx][:2]
    start_pos = layout.starting_pos
    final_pos = traj[-1]

    start_to_goal = torch.norm(start_pos - goal_pos)
    end_to_goal = torch.norm(final_pos - goal_pos)
    start_to_end = torch.sum(torch.norm(traj[1:] - traj[:-1], dim=1))

    return torch.exp(start_to_end - end_to_goal) / torch.exp(-start_to_goal)


def legibility_loss_torch(layout: Layout, task_list: List[Task], verbose: bool = False):
    loss = 0.0
    for task in task_list:
        for goal_label_idx, goal_label in enumerate(task.object_labels):
            # Find all objects with the goal label in the layout and get their indices
            goal_layout_indices = [i for i, label in enumerate(layout.object_labels) if label == goal_label]
            
            for goal_layout_idx in goal_layout_indices:

                # Trajectory to the closest goal
                trajectory = compute_trajectory_torch(layout.starting_pos, layout.object_positions[goal_layout_idx])

                for percent in [0.25, 0.5, 0.75]:
                    cutoff = max(1, int(len(trajectory) * percent))
                    partial_traj = trajectory[:cutoff]

                    probs = []
                    for candidate_label in task.object_labels:
                        if candidate_label == goal_label:
                            goal_idx = goal_layout_idx
                        else:
                            candidate_indices = [i for i, l in enumerate(layout.object_labels) if l == candidate_label]
                            dists = [torch.norm(layout.starting_pos - layout.object_positions[i][:2]) for i in candidate_indices]
                            closest_idx = candidate_indices[torch.argmin(torch.stack(dists)).item()]
                            goal_idx = closest_idx
                        p = prob_traj_given_goal_torch(layout, partial_traj, goal_idx)
                        probs.append(p)

                    prob_dist = torch.stack(probs)
                    prob_dist = prob_dist / (prob_dist.sum() + 1e-8)

                    true_dist = torch.zeros_like(prob_dist)
                    true_dist[goal_label_idx] = 1.0

                    if verbose:
                        print("Object labels:", task.object_labels)
                        print(f"Goal label: {goal_label}, percent: {percent}, prob_dist: {prob_dist}, true_dist: {true_dist}")

                    ce = torch_cross_entropy(prob_dist.unsqueeze(0), true_dist.unsqueeze(0))
                    loss = loss + ce * percent

    return loss


def batched_legibility_loss(positions_batch, starting_pos, object_labels, task_list):
    """
    positions_batch: [B, N, 2] tensor (from diffusion sample)
    starting_pos: [2] tensor
    object_labels: list[int] (length N)
    task_list: list[Task]

    Returns:
        scalar loss averaged over the batch
    """
    B = positions_batch.shape[0]
    total_loss = torch.zeros(B, device=positions_batch.device)

    for b in range(B):
        pos = positions_batch[b]  # [N, 2]
        layout = Layout(
            starting_pos=starting_pos,
            object_positions=pos,
            object_labels=object_labels
        )

        leg_loss = legibility_loss_torch(layout, task_list)
        total_loss[b] = leg_loss

    return total_loss.mean()
