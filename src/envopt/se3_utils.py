"""SE(3) math utilities used by the 3D shared-autonomy extension.

Numpy-only helpers for quaternions, rotation matrices, and the weighted
product metric used by the SE(3) observers and the trajectory-margin
optimizer.

Convention: quaternions are stored as (w, x, y, z) tuples/arrays. World
frame is ROS / MuJoCo convention (x forward, y left, z up).
"""
import numpy as np


def quat_to_rot(q):
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    R = np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])
    return R


def rot_to_quat(R):
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quat_mul(q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    """Inverse of a unit (w, x, y, z) quaternion."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def yaw_to_quat(yaw):
    """Quaternion for a rotation of `yaw` radians about z."""
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def rot_log(R):
    """Matrix log of a 3x3 rotation — returns axis-angle 3-vector.

    For R close to identity returns a small 3-vector. For R = exp([w]_x),
    returns w. Numerically stable for angles up to pi.
    """
    R = np.asarray(R, dtype=float)
    tr = np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(tr)
    if theta < 1e-6:
        # Small-angle: log(R) ≈ 0.5 * (R - R^T) skew
        return 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
    if np.pi - theta < 1e-4:
        # Near-pi: find dominant diagonal
        d = np.array([R[0, 0], R[1, 1], R[2, 2]])
        k = int(np.argmax(d))
        ax = np.zeros(3)
        ax[k] = np.sqrt(max((d[k] + 1.0) * 0.5, 0.0))
        ax[(k + 1) % 3] = R[k, (k + 1) % 3] / (2.0 * ax[k] + 1e-12)
        ax[(k + 2) % 3] = R[k, (k + 2) % 3] / (2.0 * ax[k] + 1e-12)
        return theta * ax
    return theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def quat_angle_between(q1, q2):
    """Shortest-angle rotation magnitude between two quaternions (radians).

    Uses the invariant |<q1, q2>| = cos(theta/2), so handles the
    double-cover identification correctly.
    """
    dot = abs(np.dot(q1, q2))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def orientation_error_vec(R_current, R_target):
    """Angular error vector to drive R_current toward R_target.

    Returns ω ∈ R^3 such that exp([ω]_x) R_current = R_target.
    Use as the ideal rotational velocity command for a proportional
    orientation controller.
    """
    R_err = R_target @ R_current.T
    return rot_log(R_err)


def se3_distance(p1, R1, p2, R2, lambda_R):
    """Weighted product SE(3) distance.

        d^2 = ||p1 - p2||^2 + lambda_R * ||log(R1^T R2)||^2

    Returns distance (not squared).
    """
    dp = np.asarray(p1) - np.asarray(p2)
    dpos = float(np.dot(dp, dp))
    if lambda_R > 0:
        R_rel = np.asarray(R1).T @ np.asarray(R2)
        w = rot_log(R_rel)
        drot = float(np.dot(w, w))
    else:
        drot = 0.0
    return float(np.sqrt(dpos + lambda_R * drot))


def compose_grasp_pose(obj_pos, obj_yaw, local_offset, local_quat):
    """Compose a grasp pose in the world frame.

    Args:
        obj_pos: (3,) object world position.
        obj_yaw: float, object yaw about world z.
        local_offset: (3,) grasp offset in object body frame.
        local_quat: (4,) grasp orientation as (w, x, y, z) in object body.

    Returns:
        (world_pos, world_quat): world-frame SE(3) pose as a (3,) and
        (4,) pair. Quaternion is (w, x, y, z).
    """
    q_yaw = yaw_to_quat(obj_yaw)
    R_yaw = quat_to_rot(q_yaw)
    world_offset = R_yaw @ np.asarray(local_offset, dtype=float)
    world_pos = np.asarray(obj_pos, dtype=float) + world_offset
    world_quat = quat_mul(q_yaw, np.asarray(local_quat, dtype=float))
    # Renormalize to guard against drift from accumulated products
    world_quat = world_quat / (np.linalg.norm(world_quat) + 1e-12)
    return world_pos, world_quat


def pregrasp_pose(grasp_pos, grasp_quat, standoff, approach_axis_local):
    """Compute the pre-grasp standoff pose.

    The standoff pose is `standoff` meters back along the approach axis
    (expressed in the grasp frame). If approach is [0, 0, -1] and
    standoff is 0.10, the standoff is 10 cm above the grasp along the
    grasp frame's -z axis (so 10 cm above in world if the grasp is
    top-down).

    Args:
        grasp_pos, grasp_quat: the grasp pose in world frame.
        standoff: backoff distance in meters.
        approach_axis_local: (3,) unit vector in the grasp frame that
            points FROM the pre-grasp TO the grasp. So the pre-grasp is
            at grasp_pos - standoff * R_grasp @ approach_axis_local.

    Returns:
        (pregrasp_pos, pregrasp_quat) — same orientation as the grasp.
    """
    R = quat_to_rot(grasp_quat)
    ax_world = R @ np.asarray(approach_axis_local, dtype=float)
    ax_world = ax_world / (np.linalg.norm(ax_world) + 1e-12)
    pregrasp_pos = np.asarray(grasp_pos, dtype=float) - standoff * ax_world
    return pregrasp_pos, np.asarray(grasp_quat, dtype=float).copy()
