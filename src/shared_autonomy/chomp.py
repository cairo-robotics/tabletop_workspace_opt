import torch
import numpy as np

def smoothness_penalty(traj):
    """Calculates the smoothness cost of a trajectory."""
    return torch.sum(torch.norm(traj[2:] - 2*traj[1:-1] + traj[:-2], dim=1))

def generate_smooth_path(x_start, x_goal, T=50, n_iter=150, lr=0.01, weight_smooth=1.0, weight_goal=100.0):
    """
    Generates a smooth trajectory with tunable weights for smoothness and goal achievement.
    
    :param x_start: The starting 3D position.
    :param x_goal: The goal 3D position.
    :param T: The number of waypoints in the trajectory.
    :param n_iter: The number of optimization iterations.
    :param lr: The learning rate for the optimizer.
    :param weight_smooth: The weight for the smoothness cost. A higher value creates a smoother path.
    :param weight_goal: The weight for the goal penalty. A higher value ensures the path ends closer to the goal.
    :return: An array of 3D points representing the smooth path.
    """
    x_start = torch.tensor(x_start, dtype=torch.float32)
    x_goal = torch.tensor(x_goal, dtype=torch.float32)

    # Initialize a straight-line trajectory
    traj = torch.linspace(0, 1, T + 1).unsqueeze(1) * (x_goal - x_start) + x_start
    traj.requires_grad_(True)
    
    optimizer = torch.optim.Adam([traj], lr=lr)

    for i in range(n_iter):
        optimizer.zero_grad()

        # Calculate smoothness and goal costs
        smooth_cost = smoothness_penalty(traj)
        goal_cost = torch.nn.functional.mse_loss(traj[-1], x_goal)
        
        # Total cost is a weighted sum of the two penalties
        total_cost = (weight_smooth * smooth_cost) + (weight_goal * goal_cost)
        
        total_cost.backward()
        optimizer.step()

        # Enforce that the path always starts at the beginning point
        with torch.no_grad():
            traj[0] = x_start
            
    print(f"Optimization finished with final cost: {total_cost.item():.4f}")
    return traj.detach().numpy()