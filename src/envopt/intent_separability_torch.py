"""Differentiable SE(3) intent separability / trajectory-margin objective.

Torch version of `intent_separability.py` and the trajectory-margin
slack, extended to 3D translation and (optionally) SE(3) rotation.

Used by the nested yaw GD loop in the MAP-Elites optimizer (see
`envopt/yaw_optimizer.py`). The outer loop picks object (x, y); this
module, given a fixed (x, y), optimizes object yaws via gradient
descent on a differentiable objective.

Key tensors (for a layout with N objects):
    positions  : (N, 3)   world positions of object centers (frozen)
    yaws       : (N,)     per-object yaws (the DoF being optimized)
    local_offs : (N, 3)   grasp offsets in object frame (from library)
    local_quats: (N, 4)   grasp orientations in object frame (w, x, y, z)

The objective is the negated SE(3) separability slack:

    loss(yaws) = - min_i min_{j != i}
                 [ m(i, j; yaws) - quantile * sqrt(V(i, j; yaws)) ]
"""
import numpy as np
import torch

# Robust quantile (1 - alpha/(N-1)) for alpha = 0.05 and up to 8 objects
_Z_05 = 1.6449


# ---------------------------------------------------------------------------
# Torch quaternion / rotation helpers
# ---------------------------------------------------------------------------

def yaw_to_quat_torch(yaws):
    """Convert (N,) yaw tensor to (N, 4) (w, x, y, z) quaternions."""
    half = 0.5 * yaws
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(yaws)
    return torch.stack([w, zeros, zeros, z], dim=-1)


def quat_mul_torch(q1, q2):
    """Hamilton product of (*, 4) quaternions in (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_to_rot_torch(q):
    """Convert (*, 4) quat to (*, 3, 3) rotation matrix."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R00 = 1 - 2 * (y * y + z * z)
    R01 = 2 * (x * y - z * w)
    R02 = 2 * (x * z + y * w)
    R10 = 2 * (x * y + z * w)
    R11 = 1 - 2 * (x * x + z * z)
    R12 = 2 * (y * z - x * w)
    R20 = 2 * (x * z - y * w)
    R21 = 2 * (y * z + x * w)
    R22 = 1 - 2 * (x * x + y * y)
    row0 = torch.stack([R00, R01, R02], dim=-1)
    row1 = torch.stack([R10, R11, R12], dim=-1)
    row2 = torch.stack([R20, R21, R22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quat_rotate_torch(q, v):
    """Rotate a (*, 3) vector by a (*, 4) quaternion."""
    R = quat_to_rot_torch(q)
    return torch.einsum("...ij,...j->...i", R, v)


def rot_log_torch(R):
    """Batched matrix log of (*, 3, 3) rotations. Returns (*, 3) axis-angle.

    Uses the stable trace-based formula with a small-angle series near
    identity. Not designed for angles near pi — the objective never
    pushes rotations that far.
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((tr - 1.0) * 0.5, -0.99999, 0.99999)
    theta = torch.acos(cos_theta)

    # Small-angle: log(R) ≈ 0.5 * skew(R - R^T)
    # General: log(R) = theta / (2 sin theta) * skew(R - R^T)
    sin_theta = torch.sin(theta)
    factor = torch.where(
        sin_theta.abs() < 1e-4,
        0.5 + theta * theta / 12.0,  # first-order Taylor
        theta / (2.0 * sin_theta + 1e-12),
    )
    skew = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    return factor.unsqueeze(-1) * skew


# ---------------------------------------------------------------------------
# Grasp pose resolution (differentiable in yaws)
# ---------------------------------------------------------------------------

def resolve_grasp_poses_torch(positions, yaws, local_offs, local_quats):
    """Compose world-frame grasp poses from object poses + library.

    Args:
        positions : (N, 3) frozen object positions.
        yaws      : (N,) per-object yaw (the gradient DoF).
        local_offs: (N, 3) grasp offsets in object frame.
        local_quats: (N, 4) grasp orientations (w, x, y, z) in object frame.

    Returns:
        grasp_pos : (N, 3) world positions.
        grasp_R   : (N, 3, 3) world rotation matrices.
        grasp_quat: (N, 4) world quaternions.
    """
    q_yaw = yaw_to_quat_torch(yaws)
    world_offs = quat_rotate_torch(q_yaw, local_offs)
    grasp_pos = positions + world_offs
    grasp_quat = quat_mul_torch(q_yaw, local_quats)
    grasp_quat = grasp_quat / (grasp_quat.norm(dim=-1, keepdim=True) + 1e-12)
    grasp_R = quat_to_rot_torch(grasp_quat)
    return grasp_pos, grasp_R, grasp_quat


# ---------------------------------------------------------------------------
# Separability objective
# ---------------------------------------------------------------------------

def _trajectory_to_goal_torch(start_pos, goal_pos, n_steps=30):
    """Straight-line trajectory from start to goal (translation only).

    Returns (n_steps, 3) tensor. Interpolation is linear; no clamping
    because we use it for observation simulation, not execution.
    """
    t = torch.linspace(0.0, 1.0, n_steps, device=start_pos.device)
    return start_pos.unsqueeze(0) + t.unsqueeze(-1) * (goal_pos - start_pos)


def separability_slack_se3(positions, yaws, local_offs, local_quats,
                           start_pos, start_R=None,
                           sigma_v=0.5, sigma_w=0.5, lambda_R=0.04,
                           alpha=0.05, n_steps=30):
    """SE(3) intent separability slack — scalar to MAXIMIZE.

    Implements the Gaussian-direction observer's closed-form slack,
    lifted to SE(3) via the weighted product metric:

        slack = min_i min_{j != i}
                [ 0.5 * V(i, j) - z * sqrt(V(i, j)) ]

    where V(i, j) is the cumulative Mahalanobis separation of the ideal
    twists toward grasps i and j along the straight-line trajectory
    from `start_pos` to grasp i. Twist has 3 translation + 3 rotation
    components; rotation contributes iff lambda_R > 0.

    Returns:
        slack : scalar tensor. Differentiable w.r.t. `yaws`.
        V     : (N, N) tensor of per-pair cumulative separations.
    """
    N = positions.shape[0]
    grasp_pos, grasp_R, _ = resolve_grasp_poses_torch(
        positions, yaws, local_offs, local_quats)

    start_pos = start_pos.to(positions)
    if start_R is None:
        start_R = torch.eye(3, device=positions.device,
                            dtype=positions.dtype)

    inv_v = 1.0 / (sigma_v ** 2)
    inv_w = lambda_R / (sigma_w ** 2) if lambda_R > 0 else 0.0

    # For each true goal i, simulate the straight-line trajectory then
    # compute V(i, j) for all j != i.
    V = torch.zeros(N, N, device=positions.device, dtype=positions.dtype)

    for i in range(N):
        traj = _trajectory_to_goal_torch(start_pos, grasp_pos[i], n_steps)
        # At each step, compute the ideal direction toward each goal.
        # Shape: (n_steps, N, 3)
        diffs = grasp_pos.unsqueeze(0) - traj.unsqueeze(1)
        dists = diffs.norm(dim=-1, keepdim=True) + 1e-8
        mu_v = diffs / dists

        if inv_w > 0:
            # Ideal rotation vec at each step: log(goal_R @ R_cur^T).
            # For straight-line translation we keep R_cur = start_R so
            # the ideal ω is constant per goal — precompute once.
            R_err = torch.einsum("gij,jk->gik", grasp_R, start_R.T)
            mu_w = rot_log_torch(R_err)     # (N, 3)
            mu_w = mu_w.unsqueeze(0).expand(n_steps, N, 3)
        else:
            mu_w = torch.zeros_like(mu_v)

        # Compare goal i's twist to each goal j's twist at every step.
        mu_v_true = mu_v[:, i, :].unsqueeze(1)    # (n_steps, 1, 3)
        dv = mu_v_true - mu_v                     # (n_steps, N, 3)
        v_sep = inv_v * (dv * dv).sum(dim=-1)     # (n_steps, N)

        if inv_w > 0:
            mu_w_true = mu_w[:, i, :].unsqueeze(1)
            dw = mu_w_true - mu_w
            w_sep = inv_w * (dw * dw).sum(dim=-1)
            V[i] = (v_sep + w_sep).sum(dim=0)
        else:
            V[i] = v_sep.sum(dim=0)

    # Slack per pair: s(i, j) = 0.5 * V(i, j) - z * sqrt(V(i, j))
    # We take min over j != i, then min over i.
    z = _Z_05
    sqrtV = torch.sqrt(torch.clamp(V, min=1e-8))
    slack = 0.5 * V - z * sqrtV

    # Mask out diagonals so they don't win the min
    big = torch.full_like(slack, 1e6)
    mask = torch.eye(N, dtype=torch.bool, device=positions.device)
    slack_masked = torch.where(mask, big, slack)

    worst_per_goal = slack_masked.min(dim=1).values   # (N,)
    overall = worst_per_goal.min()
    return overall, V


def separability_loss_se3(positions, yaws, local_offs, local_quats,
                          start_pos, start_R=None,
                          sigma_v=0.5, sigma_w=0.5, lambda_R=0.04,
                          alpha=0.05, n_steps=30):
    """Minimization loss = negative slack."""
    slack, V = separability_slack_se3(
        positions, yaws, local_offs, local_quats,
        start_pos, start_R, sigma_v, sigma_w, lambda_R, alpha, n_steps)
    return -slack, V


# ---------------------------------------------------------------------------
# Trajectory-margin objective (Boltzmann path-efficiency, SE(3))
# ---------------------------------------------------------------------------

def trajectory_margin_slack_se3(positions, yaws, local_offs, local_quats,
                                 start_pos, beta=5.0, sigma_u=0.03,
                                 dt=0.20, alpha=0.05, n_steps=75):
    """Trajectory-margin slack using Boltzmann path-efficiency in 3D.

    This is the SE(3) version of the 2D ``trajectory_margin_objective``
    from ``eval_map_elites_tiers.py``. It computes the linearized margin
    between each pair of grasp goals under the Boltzmann path-efficiency
    cost C(ξ, g) = (L(ξ) + d_rem(g)) / d_start(g), and returns the
    worst-case slack.

    Unlike ``separability_slack_se3`` (which uses the Gaussian direction
    model and is distance-invariant), this function includes ``d_start``
    in the denominator, correctly penalizing layouts where a grasp tip
    is placed too close to the EE start.

    Returns:
        slack : scalar tensor. Differentiable w.r.t. ``yaws``.
        margins : (N, N) tensor of per-pair mean margins (for debugging).
    """
    from scipy.stats import norm as sp_norm

    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    grasp_pos, grasp_R, _ = resolve_grasp_poses_torch(
        positions, yaws, local_offs, local_quats)
    start = start_pos.to(positions)

    alpha_g = alpha / max(N - 1, 1)
    z = float(sp_norm.ppf(1.0 - alpha_g))

    margins = torch.zeros(N, N, device=device, dtype=dtype)
    variances = torch.zeros(N, N, device=device, dtype=dtype)

    for i in range(N):
        goal_i = grasp_pos[i]
        d_start_i = (goal_i - start).norm() + 1e-6

        # Straight-line trajectory from start to goal i
        traj = _trajectory_to_goal_torch(start, goal_i, n_steps)

        # Path length of the nominal trajectory (straight line = d_start_i)
        steps = traj[1:] - traj[:-1]
        L = steps.norm(dim=-1).sum()

        # d_remaining for goal i at the end of trajectory (≈ 0)
        d_rem_i = (traj[-1] - goal_i).norm() + 1e-6
        C_i = (L + d_rem_i) / d_start_i

        for j in range(N):
            if j == i:
                continue
            goal_j = grasp_pos[j]
            d_start_j = (goal_j - start).norm() + 1e-6
            d_rem_j = (traj[-1] - goal_j).norm()
            C_j = (L + d_rem_j) / d_start_j

            delta_c = C_j - C_i
            m_g = beta * delta_c
            margins[i, j] = m_g

            # Variance: linearize delta_c around the final traj point.
            # grad = d(delta_c) / d(traj[-1])
            # Use autograd to compute this cleanly.
            # d_rem_j / d(traj[-1]) = (traj[-1] - goal_j) / d_rem_j / d_start_j
            # d_rem_i / d(traj[-1]) = (traj[-1] - goal_i) / d_rem_i / d_start_i
            # But path length L also depends on traj[-1]:
            # dL / d(traj[-1]) = (traj[-1] - traj[-2]) / ||traj[-1] - traj[-2]||
            # delta_c gradient combines all three.
            #
            # Rather than derive manually, compute via finite differences
            # (detached) for robustness — same approach as the 2D version.
            eps = 1e-4
            grad_sq = torch.tensor(0.0, device=device, dtype=dtype)
            traj_end = traj[-1].detach()
            traj_base = traj[:-1].detach()
            for dim in range(3):
                e = torch.zeros(3, device=device, dtype=dtype)
                e[dim] = eps
                # Plus perturbation
                end_p = traj_end + e
                traj_p = torch.cat([traj_base, end_p.unsqueeze(0)], dim=0)
                L_p = (traj_p[1:] - traj_p[:-1]).norm(dim=-1).sum()
                dr_i_p = (end_p - goal_i.detach()).norm() + 1e-6
                dr_j_p = (end_p - goal_j.detach()).norm()
                dc_p = ((L_p + dr_j_p) / d_start_j.detach()
                        - (L_p + dr_i_p) / d_start_i.detach())
                # Minus perturbation
                end_m = traj_end - e
                traj_m = torch.cat([traj_base, end_m.unsqueeze(0)], dim=0)
                L_m = (traj_m[1:] - traj_m[:-1]).norm(dim=-1).sum()
                dr_i_m = (end_m - goal_i.detach()).norm() + 1e-6
                dr_j_m = (end_m - goal_j.detach()).norm()
                dc_m = ((L_m + dr_j_m) / d_start_j.detach()
                        - (L_m + dr_i_m) / d_start_i.detach())
                grad_dim = (dc_p - dc_m) / (2 * eps)
                grad_sq = grad_sq + grad_dim ** 2

            traj_var = n_steps * (sigma_u ** 2) * (dt ** 2)
            v_g = (beta ** 2) * traj_var * grad_sq
            variances[i, j] = v_g

    # Slack = m_g - z * sqrt(v_g), worst case over all pairs
    sqrt_v = torch.sqrt(torch.clamp(variances, min=1e-10))
    slack_matrix = margins - z * sqrt_v

    # Mask diagonal
    big = torch.full_like(slack_matrix, 1e6)
    mask = torch.eye(N, dtype=torch.bool, device=device)
    slack_masked = torch.where(mask, big, slack_matrix)

    worst_per_goal = slack_masked.min(dim=1).values
    overall = worst_per_goal.min()
    return overall, margins


# ---------------------------------------------------------------------------
# Numpy → torch bridge helpers
# ---------------------------------------------------------------------------

def layout_tensors_from_scene(layout_xy, object_order, grasp_library,
                              start_pos, dtype=torch.float64):
    """Package numpy scene state into torch tensors for the objective.

    Args:
        layout_xy: {short_name: np.array([x, y, z])} (3-vec) object pos.
        object_order: list of short_names that defines N-tensor indexing.
        grasp_library: loaded GraspLibrary.
        start_pos: (3,) numpy — EE home position.

    Returns dict with keys positions, yaws (zeros), local_offs,
    local_quats, start_pos. All torch tensors.
    """
    positions = []
    local_offs = []
    local_quats = []
    for name in object_order:
        p = np.asarray(layout_xy[name], dtype=float)
        if len(p) == 2:
            p = np.array([p[0], p[1], 0.95])
        positions.append(p)
        entry = grasp_library[name]
        local_offs.append(entry.local_offset)
        local_quats.append(entry.local_quat)
    positions = torch.tensor(np.array(positions), dtype=dtype)
    local_offs = torch.tensor(np.array(local_offs), dtype=dtype)
    local_quats = torch.tensor(np.array(local_quats), dtype=dtype)
    yaws = torch.zeros(len(object_order), dtype=dtype, requires_grad=True)
    start_t = torch.tensor(np.asarray(start_pos, dtype=float), dtype=dtype)
    return {
        "positions": positions,
        "yaws": yaws,
        "local_offs": local_offs,
        "local_quats": local_quats,
        "start_pos": start_t,
    }
