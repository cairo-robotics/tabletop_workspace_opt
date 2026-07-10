"""Inner-loop nested yaw gradient descent for SE(3) layout optimization.

Given a frozen set of object (x, y) positions, optimize their yaws via
Adam on the differentiable SE(3) separability objective from
`intent_separability_torch.py`. The feasibility of the resulting
grasps is checked afterwards with the hard TracIK / collision oracle;
on failure, the returned loss is +infinity so the outer MAP-Elites
naturally evolves away from that cell.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from envopt.intent_separability_torch import (
    separability_slack_se3, trajectory_margin_slack_se3,
    layout_tensors_from_scene)


def optimize_yaw(layout_xy: Dict[str, np.ndarray],
                 object_order: List[str],
                 grasp_library,
                 start_pos: np.ndarray,
                 n_steps: int = 20,
                 lr: float = 0.1,
                 sigma_v: float = 0.5,
                 sigma_w: float = 0.5,
                 lambda_R: float = 0.04,
                 init_yaws: Optional[np.ndarray] = None,
                 traj_steps: int = 75,
                 dtype=torch.float64,
                 feasibility_fn=None,
                 objective: str = "separability",
                 beta: float = 5.0,
                 sigma_u: float = 0.03,
                 ) -> Tuple[np.ndarray, float]:
    """Run Adam on yaws to maximize the SE(3) separability slack.

    Args:
        layout_xy: {short_name: (x, y [,z])} frozen positions.
        object_order: list defining tensor index order (must match
            every per-object dict).
        grasp_library: loaded GraspLibrary.
        start_pos: (3,) numpy start position of the EE.
        n_steps: Adam iterations (default 20).
        lr: Adam learning rate.
        init_yaws: (N,) init guess; if None, use zeros.
        sigma_v, sigma_w, lambda_R: observer parameters.
        traj_steps: number of trajectory samples for the slack.
        feasibility_fn: optional callable(yaw_dict) -> bool. When
            supplied, only yaw configurations that pass feasibility
            are considered for the best-so-far tracker. The returned
            yaws are guaranteed to be the best *feasible* configuration
            seen during the optimization, rather than the raw best
            (which may have overshot into an infeasible region).

    Returns:
        (yaws, slack) — optimized yaws (N,) in [-pi, pi] and final slack.
        If feasibility_fn is supplied and no feasible configuration was
        found during the run, returns the initial yaws with slack=-inf.
    """
    t = layout_tensors_from_scene(
        layout_xy, object_order, grasp_library, start_pos, dtype=dtype)

    if init_yaws is not None:
        init = torch.tensor(np.asarray(init_yaws), dtype=dtype)
        yaws = init.clone().detach().requires_grad_(True)
    else:
        yaws = t["yaws"]

    opt = torch.optim.Adam([yaws], lr=lr)

    # Track two bests: the raw best (any yaw) and the best feasible.
    best_slack = -np.inf
    best_yaws = yaws.detach().numpy().copy()
    best_feasible_slack = -np.inf
    best_feasible_yaws = None

    # Check initial yaws for feasibility
    if feasibility_fn is not None:
        yaw_dict = {n: float(yaws.data[i])
                    for i, n in enumerate(object_order)}
        if feasibility_fn(yaw_dict):
            best_feasible_yaws = yaws.detach().numpy().copy()
            # Leave best_feasible_slack at -inf; the loop's first iteration
            # will overwrite it with the real slack at these initial yaws.
            # Using 0.0 as a placeholder is wrong when the achievable slack
            # is negative (e.g., Hard-B), because s > 0.0 is never true and
            # the placeholder leaks into the returned value.

    for step in range(n_steps):
        opt.zero_grad()
        if objective == "trajectory_margin":
            slack, _ = trajectory_margin_slack_se3(
                t["positions"], yaws, t["local_offs"], t["local_quats"],
                t["start_pos"],
                beta=beta, sigma_u=sigma_u,
                n_steps=traj_steps)
        else:
            slack, _ = separability_slack_se3(
                t["positions"], yaws, t["local_offs"], t["local_quats"],
                t["start_pos"],
                sigma_v=sigma_v, sigma_w=sigma_w, lambda_R=lambda_R,
                n_steps=traj_steps)
        loss = -slack
        loss.backward()
        opt.step()
        # Wrap yaws to [-pi, pi] to prevent runaway during long GD
        with torch.no_grad():
            yaws.data = torch.atan2(torch.sin(yaws.data), torch.cos(yaws.data))
        s = slack.item()

        if s > best_slack:
            best_slack = s
            best_yaws = yaws.detach().numpy().copy()

        # Track best feasible configuration
        if feasibility_fn is not None and s > best_feasible_slack:
            yaw_dict = {n: float(yaws.data[i])
                        for i, n in enumerate(object_order)}
            if feasibility_fn(yaw_dict):
                best_feasible_slack = s
                best_feasible_yaws = yaws.detach().numpy().copy()

    # If a feasibility function was provided, always return the best
    # feasible yaws (not the raw best which may be infeasible).
    if feasibility_fn is not None:
        if best_feasible_yaws is not None:
            # Re-evaluate slack at the actual yaws we are returning.
            # The cached best_feasible_slack is misaligned: the loop stores
            # the pre-step slack alongside post-step yaws.
            with torch.no_grad():
                final_y = torch.tensor(best_feasible_yaws, dtype=dtype)
                if objective == "trajectory_margin":
                    final_s, _ = trajectory_margin_slack_se3(
                        t["positions"], final_y,
                        t["local_offs"], t["local_quats"],
                        t["start_pos"],
                        beta=beta, sigma_u=sigma_u,
                        n_steps=traj_steps)
                else:
                    final_s, _ = separability_slack_se3(
                        t["positions"], final_y,
                        t["local_offs"], t["local_quats"],
                        t["start_pos"],
                        sigma_v=sigma_v, sigma_w=sigma_w,
                        lambda_R=lambda_R, n_steps=traj_steps)
            return best_feasible_yaws, float(final_s)
        else:
            # No feasible yaw found — return initial with -inf slack
            # so the caller knows this layout has no feasible yaws.
            return best_yaws, float('-inf')

    return best_yaws, float(best_slack)


def slack_with_yaws(layout_xy: Dict[str, np.ndarray],
                    object_order: List[str],
                    yaws: np.ndarray,
                    grasp_library,
                    start_pos: np.ndarray,
                    sigma_v: float = 0.5,
                    sigma_w: float = 0.5,
                    lambda_R: float = 0.04,
                    traj_steps: int = 30,
                    dtype=torch.float64,
                    objective: str = "separability",
                    beta: float = 5.0,
                    sigma_u: float = 0.03) -> float:
    """Evaluate the SE(3) slack at a fixed (x, y, yaw) layout (no grad)."""
    t = layout_tensors_from_scene(
        layout_xy, object_order, grasp_library, start_pos, dtype=dtype)
    with torch.no_grad():
        y = torch.tensor(np.asarray(yaws), dtype=dtype)
        if objective == "trajectory_margin":
            slack, _ = trajectory_margin_slack_se3(
                t["positions"], y, t["local_offs"], t["local_quats"],
                t["start_pos"], beta=beta, sigma_u=sigma_u,
                n_steps=traj_steps)
            return float(slack)
        slack, _ = separability_slack_se3(
            t["positions"], y, t["local_offs"], t["local_quats"],
            t["start_pos"],
            sigma_v=sigma_v, sigma_w=sigma_w, lambda_R=lambda_R,
            n_steps=traj_steps)
    return float(slack.item())
