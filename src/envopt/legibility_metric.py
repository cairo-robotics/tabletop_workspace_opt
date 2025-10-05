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
    """
    :param layout: Layout object representing the environment layout
    :param starting_position: Tuple representing the agent's starting position
    :param goal: The goal object's label
    :return: The position of the goal object that is closest to the starting position
    """
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


def prob_traj_given_goal(layout: Layout, trajectory: List, goal_pos: Tuple[float, float]):
    """
    :param layout: Layout object representing the environment layout
    :param trajectory: List of tuples representing the agent's positions through time
    :param goal_pos: The goal object's position
    :return: The probability of the agent reaching the goal object given the trajectory
    """
    goal_pos = np.array(goal_pos)[:2]  # Ignore orientation
    # Calculate the Euclidian distance from the starting position to the goal object
    start_to_goal_dist = np.linalg.norm(np.array(layout.starting_pos) - goal_pos)

    if len(trajectory) == 0:
        return 1e-6

    # Calculate the Euclidian distance from the agent's final position to the goal object
    final_pos = trajectory[-1]
    end_to_goal_dist = np.linalg.norm(np.array(final_pos) - goal_pos)

    # Calculate the Euclidian distance from the starting position to the agent's final position
    start_to_end_dist = 0
    for i in range(1, len(trajectory)):
        start_to_end_dist += np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i - 1]))
    
    if np.exp(-start_to_goal_dist) == 0:
        return 1e-6
    # Calculate the probability of reaching the goal object given the trajectory
    return np.exp(start_to_end_dist-end_to_goal_dist) / np.exp(-start_to_goal_dist)


def compute_trajectory(start, goal, step_size):
    # Compute a trajectory from start to goal with a fixed step size
    start = np.array(start)[:2] # ignore orientation
    goal = np.array(goal)[:2]
    
    direction = goal - start
    distance = np.linalg.norm(direction)
    
    if distance == 0:
        return [start]  # Start and goal are the same
    
    direction = direction / distance  # Normalize
    
    num_steps = int(np.ceil(distance / step_size))  # Ensure each step is at most step_size
    num_steps = min(num_steps, 500)  # Limit to 500 steps for performance
    trajectory = [start + i * step_size * direction for i in range(num_steps)]
    trajectory.append(goal)  # Ensure the goal is included
    return np.array(trajectory)


def cross_entropy_loss(preds, targets):
    """
    :param preds: Predicted probabilities
    :param targets: True labels
    :return: Cross-entropy loss
    """
    return -np.sum(targets * np.log(preds))


def legibility_loss(layout: Layout, task_list: List[Task]):
    """
    :param layout: Layout object representing the environment layout
    :param task_list: List of Task objects representing the objects to pick up to complete the task
    :return: The legibility loss of the layout based on the tasks
    """
    loss = 0
    for task in task_list:
        for goal_label_idx in range(len(task.object_labels)):
            goal_label = task.object_labels[goal_label_idx]
            # assume goal label is the object to pick up
            # simulate trajectory to reach goal label
            # Find all objects with the goal label in the layout and get their indices
            goal_layout_indices = [i for i, label in enumerate(layout.object_labels) if label == goal_label]
            
            for goal_layout_idx in goal_layout_indices:
                goal_position = layout.object_positions[goal_layout_idx]
                trajectory = compute_trajectory(layout.starting_pos, layout.object_positions[goal_layout_idx], step_size=0.02)
                # set maximum trajectory length
                trajectory = trajectory[:100]
                # compute legibility at different points along the trajectory
                percentages = [0.25, 0.5, 0.75]
                for percent in percentages:
                    traj = trajectory[:int(percent * len(trajectory))]

                    # calculate distribution of goals given the trajectory
                    prob_dist = np.zeros(len(task.object_labels))
                    for i, candidate_goal in enumerate(task.object_labels):
                        if candidate_goal != goal_label:
                            closest_goal_idx = get_closest_goal(layout, layout.starting_pos, candidate_goal)
                            prob_dist[i] = prob_traj_given_goal(layout, traj, layout.object_positions[closest_goal_idx])
                        else:
                            prob_dist[i] = prob_traj_given_goal(layout, traj, goal_position)
                    # normalize distribution
                    prob_dist /= np.sum(prob_dist)
                    # calculate cross-entropy loss between predicted distribution and true distribution
                    true_dist = np.zeros(len(task.object_labels))
                    true_dist[goal_label_idx] = 1
                    loss += cross_entropy_loss(prob_dist, true_dist) * percent
                    if np.isnan(loss) or np.isinf(loss):
                        print(layout.starting_pos, layout.object_positions[goal_layout_idx])
                        print(prob_dist, true_dist)
                        print(len(trajectory))
                        raise ValueError("Loss is NaN or Inf")
    return loss
