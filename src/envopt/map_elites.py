import sys
import time
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import pandas as pd
from typing import List, Tuple
import torch
from legibility_metric import Layout, Task, legibility_loss
from legibility_metric_torch import legibility_loss_torch, Layout as LayoutTorch, Task as TaskTorch

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter, GradientArborescenceEmitter, GeneticAlgorithmEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import grid_archive_heatmap
from shapely.geometry import box
from shapely.affinity import rotate, translate
import pickle


class CustomEmitter(EvolutionStrategyEmitter):
    """
    Custom emitter that also swaps object positions
    """

    def __init__(self, archive, x0, sigma0=0.01, batch_size=30, ranker="2imp"):
        super().__init__(archive=archive, x0=x0, sigma0=sigma0, batch_size=batch_size, ranker=ranker)

    @property
    def parent_type(self):
        """int: Parent Type to be used by selector."""
        return 1

    def ask(self):
        solutions = super().ask()
        # Create a writable copy of the solutions array
        solutions = solutions.copy()
        # Swap two random object positions in half of the solutions
        for i in range(len(solutions)//2):
            obj1, obj2 = np.random.choice(num_objects, size=2, replace=False)
            idx1 = obj1 * 3
            idx2 = obj2 * 3
            # Swap (x, y, theta)
            solutions[i][idx1:idx1+3], solutions[i][idx2:idx2+3] = solutions[i][idx2:idx2+3].copy(), solutions[i][idx1:idx1+3].copy()
        return solutions


def has_overlap(object_positions: np.ndarray, box_sizes: List[Tuple[float, float]]) -> bool:
    """
    Checks if any rotated rectangles (given as center (x, y) and angle theta) overlap.
    
    Args:
        object_positions: np.ndarray of shape (num_objects, 3), each row is (x, y, theta).
        box_sizes: List of (width, height) tuples for each object.

    Returns:
        True if any two boxes overlap, False otherwise.
    """
    num_objects = len(object_positions)
    polygons = []

    for (x, y, theta), (w, h) in zip(object_positions, box_sizes):
        rect = box(-w / 2, -h / 2, w / 2, h / 2)
        rotated_rect = rotate(rect, theta, use_radians=False)
        translated_rect = translate(rotated_rect, xoff=x, yoff=y)
        polygons.append(translated_rect)

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            if polygons[i].intersects(polygons[j]):
                return True

    return False

def compute_overlap_area(object_positions: np.ndarray, box_sizes: List[Tuple[float, float]]) -> float:
    """
    Computes the area of intersection between list of objects
    
    Args:
        object_positions: np.ndarray of shape (num_objects, 3), each row is (x, y, theta).
        box_sizes: List of (width, height) tuples for each object.

    Returns:
        Area of overlap
    """
    # Compute total overlap area
    total_overlap = 0.0
    polygons = []

    for (x, y, theta), (w, h) in zip(object_positions, box_sizes):
        # Create rectangle centered at origin
        rect = box(-w / 2, -h / 2, w / 2, h / 2)
        
        # Rotate by theta (degrees)
        rotated_rect = rotate(rect, theta, use_radians=False)
        
        # Translate to (x, y)
        translated_rect = translate(rotated_rect, xoff=x, yoff=y)

        polygons.append(translated_rect)

    # Compute pairwise intersection area
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            intersection = polygons[i].intersection(polygons[j])
            if not intersection.is_empty:
                total_overlap += intersection.area
    return total_overlap


def generate_non_overlapping_positions(num_samples, box_sizes, workspace_bounds, movable_object_mask, object_positions, max_attempts=1000):
    """
    Generate (x, y, theta) positions for multiple objects with varying bounding box sizes,
    ensuring no two rotated boxes overlap.
    
    Returns:
        - positions: numpy array of shape (num_samples, num_objects, 3)
    """
    num_objects = len(box_sizes)
    all_configs = []

    for sample_idx in range(num_samples):
        for _ in range(max_attempts):
            positions = []
            placed_polygons = []

            # place fixed objects first
            for i in range(num_objects):
                if movable_object_mask[i] == 0:
                    positions.append(object_positions[i])
                    # Create polygon for overlap checking
                    x, y, theta = object_positions[i]
                    w, h = box_sizes[i]
                    rect = box(-w/2, -h/2, w/2, h/2)
                    rotated_rect = rotate(rect, theta, use_radians=False)
                    translated_rect = translate(rotated_rect, xoff=x, yoff=y)
                    placed_polygons.append(translated_rect)

            for i in range(num_objects):
                if movable_object_mask[i] == 0:
                    continue  # already placed
                w, h = box_sizes[i]
                success = False

                for _ in range(100):  # inner placement attempts
                    x = np.random.uniform(workspace_bounds[0][0] + w/2, workspace_bounds[0][1] - w/2)
                    y = np.random.uniform(workspace_bounds[1][0] + h/2, workspace_bounds[1][1] - h/2)
                    theta = np.random.uniform(-180, 180)

                    # Create rotated polygon using shapely
                    rect = box(-w/2, -h/2, w/2, h/2)
                    rotated_rect = rotate(rect, theta, use_radians=False)
                    translated_rect = translate(rotated_rect, xoff=x, yoff=y)

                    # Check overlap with existing polygons
                    overlap = any(translated_rect.intersects(p) for p in placed_polygons)

                    if not overlap:
                        positions.append([x, y, theta])
                        placed_polygons.append(translated_rect)
                        success = True
                        break

                if not success:
                    break  # failed to place

            if len(positions) == num_objects:
                all_configs.append(positions)
                break  # success

        else:
            raise RuntimeError(f"Failed to generate sample {sample_idx+1} after {max_attempts} attempts.")

    return np.array(all_configs, dtype=np.float32)


def visualize_layout(positions, box_sizes, workspace_bounds=(-1, 1), title="Object Layout", filename=None):
    """
    Visualize a single layout of (x, y) object positions with bounding boxes.
    
    Parameters:
        - positions: (num_objects, 2) array of (x, y) positions
        - box_sizes: list of (width, height) tuples per object
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(workspace_bounds)
    ax.set_ylim(workspace_bounds)
    ax.set_aspect('equal')
    ax.set_title(title)

    # Draw bounding boxes
    for i, ((x, y, theta), (w, h)) in enumerate(zip(positions, box_sizes)):
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=1, color=colors[i], alpha=0.6
        )
        # Rotate around center
        t = transforms.Affine2D().rotate_deg_around(x, y, theta) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        ax.plot(x, y, 'ko')  # center point

    plt.grid(True)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def std_angular_separation(object_positions, start_position):
    """
    Computes the standard deviation of angular separations between objects from a start position.

    Args:
        object_positions: (n_objects, 3) array (x, y, theta)
        start_position: (2,) array (x, y)

    Returns:
        std_angular_sep: scalar (standard deviation of angular separation in radians)
    """
    centers = object_positions[:, :2]  # Use only (x, y)

    # Compute angles from start to each object
    delta = centers - np.array(start_position)  # (n_objects, 2)
    angles = np.arctan2(delta[:, 1], delta[:, 0])  # angles in radians

    # Sort angles
    sorted_angles = np.sort(angles)

    # Compute differences between consecutive angles
    angle_diffs = np.diff(sorted_angles)

    # Handle wrap-around (last to first)
    wrap_diff = (sorted_angles[0] + 2*np.pi) - sorted_angles[-1]
    angle_diffs = np.append(angle_diffs, wrap_diff)

    # Standard deviation of angular separations
    std_angular_sep = np.std(angle_diffs)
    return std_angular_sep

def std_angular_separation_and_jacobian(object_positions, start_position):
    """
    Computes the standard deviation of angular separations and its Jacobian w.r.t. object positions (x, y only).

    Args:
        object_positions: (n_objects, 3) torch tensor (x, y, theta)
        start_position: (2,) torch tensor (x, y)

    Returns:
        std_sep: scalar torch tensor (standard deviation of angular separation)
        jacobian: (n_objects, 3) torch tensor (only x, y gradients filled, theta gradients are zero)
    """
    centers = object_positions[:, :2]  # (n_objects, 2)
    delta = centers - start_position.unsqueeze(0)  # (n_objects, 2)
    x = delta[:, 0]
    y = delta[:, 1]

    # Compute angles
    angles = torch.atan2(y, x)  # (n_objects,)

    # Sort angles
    sorted_angles, sort_indices = torch.sort(angles)

    # Differences between sorted angles
    angle_diffs = sorted_angles[1:] - sorted_angles[:-1]
    wrap_diff = (sorted_angles[0] + 2 * torch.pi) - sorted_angles[-1]
    angle_diffs = torch.cat([angle_diffs, wrap_diff.unsqueeze(0)], dim=0)

    # Compute mean and std
    mean_angle_diff = angle_diffs.mean()
    std_angle_diff = torch.sqrt(torch.mean((angle_diffs - mean_angle_diff) ** 2))

    # Now compute Jacobian
    n = object_positions.shape[0]
    inv_n = 1.0 / n

    denom = (x**2 + y**2)  # (n,)
    dtheta_dx = -y / denom  # (n,)
    dtheta_dy = x / denom   # (n,)

    # Initialize full Jacobian
    jacobian = torch.zeros_like(object_positions)  # (n_objects, 3)

    # First, map sort_indices -> reverse indices
    reverse_sort_indices = torch.argsort(sort_indices)

    # Build per-angle-diff gradients
    # Each gap is affected by two angles: previous and next
    gap_grad_dx = torch.zeros_like(angles)
    gap_grad_dy = torch.zeros_like(angles)

    for idx in range(n):
        i = sort_indices[idx]
        next_idx = sort_indices[(idx + 1) % n]

        gap_grad_dx[i] -= dtheta_dx[i]
        gap_grad_dx[next_idx] += dtheta_dx[next_idx]

        gap_grad_dy[i] -= dtheta_dy[i]
        gap_grad_dy[next_idx] += dtheta_dy[next_idx]

    # Now compute gradient of std
    centered = angle_diffs - mean_angle_diff  # (n,)

    # Derivative of std w.r.t. each angle_diff
    d_std_d_diff = centered / (n * std_angle_diff + 1e-8)  # (n,)

    # d(angle_diff) / dtheta_i
    # Each angle_diff depends on two angles: diff = theta_next - theta_current
    d_diff_dtheta = torch.zeros((n, n), device=object_positions.device)  # (n_diffs, n_angles)

    for idx in range(n-1):
        d_diff_dtheta[idx, sort_indices[idx]] = -1
        d_diff_dtheta[idx, sort_indices[idx+1]] = 1
    # Wrap-around diff
    d_diff_dtheta[-1, sort_indices[-1]] = -1
    d_diff_dtheta[-1, sort_indices[0]] = 1

    # Chain rule: d(std) / dtheta
    d_std_dtheta = torch.matmul(d_std_d_diff.to(d_diff_dtheta.dtype), d_diff_dtheta)  # (n,)

    # Now backpropagate to x, y
    for i in range(n):
        jacobian[i, 0] = d_std_dtheta[i] * dtheta_dx[i]  # d(std)/dx
        jacobian[i, 1] = d_std_dtheta[i] * dtheta_dy[i]  # d(std)/dy
        jacobian[i, 2] = 0.0  # d(std)/dtheta = 0, no dependency

    return std_angle_diff, jacobian

def get_features(layout: Layout):
    """
    :param layout: Layout object representing the environment layout
    :return: A numpy array of features representing the layout
    """
    features = []

    # Calculate the mean pairwise distance between objects (encourages clustering or dispersion)
    pairwise_dists = []
    for i in range(len(layout.object_positions)):
        for j in range(i + 1, len(layout.object_positions)):
            dist = np.linalg.norm(layout.object_positions[i] - layout.object_positions[j])
            pairwise_dists.append(dist)
    features.append(np.mean(pairwise_dists))

    # Quantify how symmetric the layout is (encourages symmetry)
    # y_coords = layout.object_positions[:, 1]
    # symmetry_y = 1.0 - np.abs(np.mean(y_coords - (y_coords.max() + y_coords.min()) / 2))
    # features.append(symmetry_y)
    std_angular_separation_value = std_angular_separation(layout.object_positions, layout.starting_pos)
    features.append(std_angular_separation_value)
    return np.array(features)


def approximate_overlap_loss(positions, box_sizes):
    """
    Differentiable approximate overlap loss.

    Args:
        positions: Tensor of shape (num_objects, 3) with (x, y, theta)
        box_sizes: Tensor of shape (num_objects, 2) with (w, h)
    Returns:
        overlap_loss: Scalar Tensor
    """
    loss = 0.0
    num_objects = positions.shape[0]
    
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            xi, yi, _ = positions[i]
            xj, yj, _ = positions[j]
            wi, hi = box_sizes[i]
            wj, hj = box_sizes[j]

            # Distance between centers
            dx = xi - xj
            dy = yi - yj
            center_dist = torch.sqrt(dx**2 + dy**2 + 1e-6)  # add epsilon to prevent sqrt(0)

            # Sum of half-diagonals (rough approx for bounding box radius)
            diag_i = torch.sqrt((wi/2)**2 + (hi/2)**2)
            diag_j = torch.sqrt((wj/2)**2 + (hj/2)**2)
            min_dist = diag_i + diag_j

            # If centers are too close, penalize
            overlap_amount = torch.relu(min_dist - center_dist)
            loss += overlap_amount**2  # can tune power if needed

    return loss


def simulate(model: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
    sol = model.reshape(num_objects, 3)
    layout = Layout(starting_pos, sol, object_labels)
        
    # Penalize score by overlap area
    overlap_penalty = compute_overlap_area(sol, box_sizes)

    # Compute efficiency of task
    dist = 0
    for task in task_list:
        for i in range(1, len(task.object_labels)):
            dist += np.linalg.norm(sol[task.object_labels[i]]-sol[task.object_labels[i-1]])

    lower_bounds = torch.tensor(lower_bounds)
    upper_bounds = torch.tensor(upper_bounds)
    sol_tensor = torch.tensor(sol).reshape(-1)
    bound_violation, _ = compute_bounds_violation(sol_tensor, lower_bounds, upper_bounds)
    
    # Compute legibility score
    score = legibility_w * -legibility_loss(layout, task_list) \
            - efficiency_w * dist \
            - bounds_violation_w * bound_violation \
            - overlap_w * overlap_penalty
    features = get_features(layout)
    
    return score, features[0], features[1]


def compute_bounds_violation(sol: torch.Tensor, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bounds violation penalty and its gradient.
    
    Args:
        sol: Solution tensor
        lower_bounds: Lower bounds tensor
        upper_bounds: Upper bounds tensor
        
    Returns:
        Tuple of (penalty, gradient)
    """
    # Compute violations (positive values indicate violation)
    lower_violation = torch.relu(lower_bounds - sol)
    upper_violation = torch.relu(sol - upper_bounds)
    
    # Total violation
    total_violation = torch.sum(lower_violation + upper_violation)
    
    # Gradient of violation
    grad = torch.zeros_like(sol)
    grad = grad - torch.sign(lower_violation)  # Gradient for lower bound violations
    grad = grad + torch.sign(upper_violation)  # Gradient for upper bound violations
    
    return total_violation, grad


def compute_efficiency(sol: torch.Tensor):
    # compute distance between objects required for the task
    sol = sol.reshape(-1, 2)
    dist = 0
    for task in task_list_torch:
        for i in range(1, len(task.object_labels)):
            dist += torch.linalg.norm(sol[task.object_labels[i]]-sol[task.object_labels[i-1]])
    return dist
            

def compute_objective(sols: np.ndarray, object_labels: List[int], lower_bounds: np.ndarray, upper_bounds: np.ndarray):
    """
    :param sols: (num_solutions, solution_dim) array of solutions
    :return: (num_solutions, gradient) array of objective values and gradients
    """
    objs = []
    jacobians = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert bounds to tensors
    lower_bounds = torch.tensor(lower_bounds, device=device)
    upper_bounds = torch.tensor(upper_bounds, device=device)
    
    # use torch tensors
    starting_pos = torch.tensor([0, 0], device=device)
    object_labels = torch.tensor(object_labels).to(device)
    
    # compute legibility score and gradient for each solution
    for sol in sols:
        sol = torch.tensor(sol).to(device)
        sol = sol.detach().requires_grad_(True)
        
        # Compute legibility score and gradient
        layout = LayoutTorch(starting_pos, sol.reshape(num_objects, 3), object_labels)
        # We want to minimize legibility loss
        legibility_score = -legibility_loss_torch(layout, task_list_torch)
        legibility_grad = torch.autograd.grad(legibility_score, sol)[0]
        
        # Compute efficiency for tasks
        efficiency_loss = compute_efficiency(sol)
        efficiency_grad = torch.autograd.grad(efficiency_loss, sol)[0]

        # Compute bounds violation penalty and gradient
        bounds_penalty, bounds_grad = compute_bounds_violation(sol, lower_bounds, upper_bounds)
        
        overlap_loss = approximate_overlap_loss(sol.reshape(num_objects, 3), torch.tensor(box_sizes).to(device))
        overlap_grad = torch.autograd.grad(overlap_loss, sol)[0]

        # Combine scores and gradients
        total_score = legibility_w * legibility_score \
                    - efficiency_w * efficiency_loss \
                    - bounds_violation_w * bounds_penalty \
                    - overlap_w * overlap_loss
        total_grad = legibility_w * legibility_grad \
                    - efficiency_w * efficiency_grad \
                    - bounds_violation_w * bounds_grad \
                    - overlap_w * overlap_grad
        
        objs.append(total_score.cpu().detach().numpy()[0])
        jacobians.append(-total_grad.cpu().detach().numpy())

    return np.array(objs), np.array(jacobians)


def compute_measure(sols: np.ndarray, object_labels: List[int]):
    """
    Computes features for each solution and its gradient

    :param sols: (num_solutions, solution_dim) array of solutions
    :return: (num_solutions, num_measures), (num_solutions, num_measure) array of features and gradients
    """
    measures = []
    gradients = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use torch tensors
    starting_pos = torch.tensor([0, 0], device=device)
    object_labels = torch.tensor(object_labels).to(device)

    def get_feature_torch(layout: LayoutTorch):
        features = []
        grads = []
        pairwise_dists = 0
        for i in range(len(layout.object_positions)):
            for j in range(i + 1, len(layout.object_positions)):
                dist = torch.linalg.norm(layout.object_positions[i] - layout.object_positions[j])
                pairwise_dists += dist
        avg_pairwise_dist = pairwise_dists / (num_objects * (num_objects + 1) / 2)
        features.append(avg_pairwise_dist)
        grads.append(torch.autograd.grad(avg_pairwise_dist, layout.object_positions)[0])

        # Quantify how symmetric the layout is (encourages symmetry)
        # y_coords = layout.object_positions[:, 1]
        # symmetry_y = 1.0 - torch.abs(torch.mean(y_coords - (y_coords.max() + y_coords.min()) / 2))
        # features.append(symmetry_y)
        # grads.append(torch.autograd.grad(symmetry_y, layout.object_positions)[0])
        ang_sep, ang_sep_grad = std_angular_separation_and_jacobian(layout.object_positions, layout.starting_pos)
        features.append(ang_sep)
        grads.append(ang_sep_grad)
        return torch.stack(features), torch.stack(grads)

    for sol in sols:
        sol = torch.tensor(sol).to(device)
        sol = sol.clone().detach().requires_grad_(True)
        layout = LayoutTorch(starting_pos, sol.reshape(num_objects, 3), object_labels)
        features, feature_grads = get_feature_torch(layout)

        measures.append(features.cpu().detach().numpy())
        gradients.append(feature_grads.cpu().detach().numpy())
    return np.array(measures), np.array(gradients).reshape(1, 2, -1)


if __name__ == "__main__":
    box_sizes = [
        [0.22, 0.14], # black tea
        [0.15, 0.076], # chai
        [0.1, 0.1], # cup
        [0.15, 0.15], # milk
        [0.18, 0.12], # panda
        [0.15, 0.15], # ritz
    ]
    object_positions = [()]
    movable_object_mask = [0, 1, 1, 1, 1, 1]

    objects = ['black tea', 'chai', 'cup', 'milk', 'panda', 'ritz']
    object_labels = list(range(len(objects)))
    colors = ['brown', 'orange', 'red', 'purple', 'black', 'blue']
    num_objects = len(box_sizes)
    workspace_bounds = ((0.2, 0.8), (-0.5, 0.5), (-np.pi/2.0, np.pi/2.0))
    starting_pos = (0, 0)
    task_list = [Task([0, 1]), Task([4, 5])]
    task_list_torch = [TaskTorch([0, 1]), TaskTorch([4, 5])]

    sampled_layouts = generate_non_overlapping_positions(100, box_sizes, workspace_bounds, movable_object_mask, object_positions)
    leg_loss = []
    for position in sampled_layouts:
        leg_loss.append(legibility_loss(Layout(starting_pos, tuple(position), object_labels), task_list))
    print(np.min(leg_loss), np.max(leg_loss))

    initial_model = generate_non_overlapping_positions(1, box_sizes, workspace_bounds, movable_object_mask, object_positions)[0]

    # Workspace bounds: ((x_min, x_max), (y_min, y_max))
    x_min, x_max = workspace_bounds[0]
    y_min, y_max = workspace_bounds[1]
    theta_min, theta_max = workspace_bounds[2]

    # Repeat for each object, then flatten
    lower_bounds = np.array([[x_min, y_min, theta_min]] * num_objects).flatten()
    upper_bounds = np.array([[x_max, y_max, theta_max]] * num_objects).flatten()

    archive_dims = [30, 30]  # 30 cells along each dimension.
    measure_ranges = [(0.5, 3.0), (1.0, 3.0)]  # Ranges for the two features.

    # options for method: "cma-me", "cma-mae", "me", "dqd"
    MAP_ELITES_METHOD = "cma-me"
    train_mode = True

    # weights for objectives
    legibility_w = 2.0
    efficiency_w = 1.0
    bounds_violation_w = 2.0
    overlap_w = 5.0

    if MAP_ELITES_METHOD == "cma-me":

        archive = GridArchive(
            solution_dim=initial_model.size,  # Dimensionality of solutions in the archive.
            dims=archive_dims,
            ranges=measure_ranges,
        )

        # Relax bounds for better sampling
        emitters = [
            CustomEmitter(
                archive=archive,
                x0=initial_model.flatten(),
                sigma0=0.01,  # Initial step size.
                ranker="2imp",
                batch_size=30,  # If we do not specify a batch size, the emitter will
                                # automatically use a batch size equal to the default
                                # population size of CMA-ES.
            )
        ]

        scheduler = Scheduler(archive, emitters)
    
    elif MAP_ELITES_METHOD == "cma-mae":
        archive = GridArchive(
            solution_dim=initial_model.size,  # Dimensionality of solutions in the archive.
            dims=archive_dims,
            ranges=measure_ranges,
            learning_rate=0.05,
            threshold_min=-101,  # prevent negative scores
        )

        result_archive = GridArchive(
            solution_dim=initial_model.size,  # Dimensionality of solutions in the archive.
            dims=archive_dims,
            ranges=measure_ranges,
        )

        emitters = [
            EvolutionStrategyEmitter(
                archive,
                x0=initial_model.flatten(),
                sigma0=0.1,
                ranker="imp",
                selection_rule="mu",
                restart_rule="basic",
                bounds=tuple(zip(lower_bounds, upper_bounds)),
            ) for _ in range(5)
        ]

        scheduler = Scheduler(archive, emitters, result_archive=result_archive)

    elif MAP_ELITES_METHOD == "me":
        archive = GridArchive(
            solution_dim=initial_model.size,  # Dimensionality of solutions in the archive.
            dims=archive_dims,
            ranges=measure_ranges,
            qd_score_offset=-100,  # prevent negative scores
        )

        emitters = [
            GaussianEmitter(
                archive=archive,
                x0=initial_model.flatten(),
                sigma0=0.1,  # Initial step size.
                bounds=tuple(zip(lower_bounds, upper_bounds)),
            ) for _ in range(5)  # Create 5 separate emitters.
        ]

        scheduler = Scheduler(archive, emitters)

    elif MAP_ELITES_METHOD == "dqd":
        archive = GridArchive(
            solution_dim=initial_model.size,  # Dimensionality of solutions in the archive.
            dims=archive_dims,
            ranges=measure_ranges,
            learning_rate=0.05,
            threshold_min=-10000,  # prevent negative scores
        )

        result_archive = GridArchive(
            solution_dim=initial_model.size,  # Dimensionality of solutions in the archive.
            dims=archive_dims,
            ranges=measure_ranges,
        )
    
        emitters = [
            GradientArborescenceEmitter(
                archive=archive,
                x0=initial_model.flatten(),
                sigma0=0.02,  # Initial standard deviation for the coefficient distribution.
                lr=0.05,  # Learning rate for updating theta with gradient ascent.
                ranker="imp",
                selection_rule="mu",
                restart_rule='basic',
            ) 
        ]
        scheduler = Scheduler(archive, emitters, result_archive=result_archive)

    start_time = time.time()
    total_itrs = 1000

    if train_mode:

        min_features = np.array([np.inf for r in measure_ranges])
        max_features = np.array([-np.inf for r in measure_ranges])
        for itr in trange(1, total_itrs + 1, file=sys.stdout, desc='Iterations'):
            if MAP_ELITES_METHOD == "dqd" or MAP_ELITES_METHOD == "diffusion":
                # Gradient phase
                sols = scheduler.ask_dqd()

                # clip solutions to be within bounds
                sols = np.clip(sols, lower_bounds, upper_bounds)

                # sols has shape (num_emitters, solution_dim)
                obj, jacobian_obj = compute_objective(sols, object_labels, lower_bounds, upper_bounds)
                measure, jacobian_measure = compute_measure(sols, object_labels)
                jacobian_obj = jacobian_obj.reshape(jacobian_obj.shape[0], 1, jacobian_obj.shape[1])
                jacobian = np.concatenate((jacobian_obj, jacobian_measure), axis=1)
                scheduler.tell_dqd(obj, measure, jacobian)
            
            # Request models from the scheduler.
            sols = scheduler.ask()
            # clip solutions to be within bounds
            sols = np.clip(sols, lower_bounds, upper_bounds)
            results = [simulate(sol, lower_bounds, upper_bounds) for sol in sols]

            objs, meas = [], []
            for obj, feature1, feature2 in results:
                objs.append(obj)
                meas.append([feature1, feature2])

            meas = np.array(meas)
            for b in range(meas.shape[1]):
                min_features[b] = min(min_features[b], np.min(meas[:, b]))
                max_features[b] = max(max_features[b], np.max(meas[:, b]))

            # Send the results back to the scheduler.
            scheduler.tell(objs, meas)

            # Logging.
            if itr % 25 == 0:
                tqdm.write(f"> {itr} itrs completed after {time.time() - start_time:.2f}s")
                tqdm.write(f"  - Size: {archive.stats.num_elites}")    # Number of elites in the archive. len(archive) also provides this info.
                tqdm.write(f"  - Coverage: {archive.stats.coverage}")  # Proportion of archive cells which have an elite.
                tqdm.write(f"  - QD Score: {archive.stats.qd_score}")  # QD score, i.e. sum of objective values of all elites in the archive.
                                                                    # Accounts for qd_score_offset as described in the GridArchive section.
                tqdm.write(f"  - Max Obj: {archive.stats.obj_max}")    # Maximum objective value in the archive.
                tqdm.write(f"  - Mean Obj: {archive.stats.obj_mean}")  # Mean objective value of elites in the archive.

        with open(f"wo_archive_{MAP_ELITES_METHOD}.pkl", "wb") as f:
            pickle.dump(archive, f)
    else:
        with open(f"wo_archive_{MAP_ELITES_METHOD}.pkl", "rb") as f:
            archive = pickle.load(f)
        
        print(f"Archive size: {archive.stats.num_elites}")
        print(f"Coverage: {archive.stats.coverage}")
        print(f"QD Score: {archive.stats.qd_score}")
        print(f"Max Obj: {archive.stats.obj_max}")
        print(f"Mean Obj: {archive.stats.obj_mean}")
    
    best_sol = archive.best_elite
    sol = best_sol['solution']
    object_positions = sol.reshape(num_objects, 3)

    # Reconstruct layout
    layout = Layout(starting_pos, object_positions, object_labels)
    visualize_layout(layout.object_positions, [box_sizes[i] for i in object_labels], filename=f"wo_layout_{MAP_ELITES_METHOD}.png")
    
    print("object positions: ", object_positions)
    print("Legibility loss: ", legibility_loss(layout, task_list))
    print("Efficiency: ", compute_efficiency(torch.tensor(object_positions)))

    print("Feature ranges observed:")
    for i, (min_f, max_f) in enumerate(zip(min_features, max_features)):
        print(f"  - Feature {i}: [{min_f}, {max_f}]")

    # Visualize the archive
    fig, ax = plt.subplots(figsize=(8, 8))
    grid_archive_heatmap(archive, ax=ax)
    plt.title(f"Archive Heatmap {MAP_ELITES_METHOD}")
    plt.savefig(f"wo_archive_heatmap_{MAP_ELITES_METHOD}.png")
    plt.close()
