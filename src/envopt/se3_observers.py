"""SE(3) Gaussian and Boltzmann observers for 3D shared autonomy.

These observers extend the 2D equivalents in `run_sa_headless.py` to
operate on full SE(3) poses. The design mirrors the 2D code so the
same runner can switch between them with an `intent_mode` flag.

Goal representation:
    Each goal is a dict {'pos': (3,) np.ndarray, 'R': (3,3) np.ndarray}
    giving the target EE pose. Rotation may be absent (None / identity)
    when the intent mode is translation-only.

Observation representation:
    `observed_twist` is a 6-vector (v_x, v_y, v_z, w_x, w_y, w_z) in the
    world frame — the user's noisy commanded EE twist. The observers
    normalize it to a unit direction for the Gaussian path.

Both observers support the plain 3D translation-only case by setting
`lambda_R = 0.0` (Boltzmann) or `use_rotation = False` (Gaussian).
"""
import numpy as np

from envopt.se3_utils import rot_log


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ideal_twist(ee_pos, ee_R, goal_pos, goal_R, use_rotation):
    """Compute the ideal commanded twist toward a goal.

    Returns a 6-vector (v, w). v is the normalized direction toward
    goal_pos; w is the log-map of R_err = goal_R @ ee_R.T (the
    proportional orientation controller).

    When use_rotation is False or goal_R is None, w = 0.
    """
    v = goal_pos - ee_pos
    vn = np.linalg.norm(v)
    if vn > 1e-6:
        v_unit = v / vn
    else:
        v_unit = np.zeros(3)

    if use_rotation and goal_R is not None and ee_R is not None:
        R_err = goal_R @ ee_R.T
        w = rot_log(R_err)
    else:
        w = np.zeros(3)

    return v_unit, w


# ---------------------------------------------------------------------------
# Gaussian direction observer (SE(3))
# ---------------------------------------------------------------------------

class GaussianDirectionInferenceSE3:
    """Gaussian observer over 6-DoF twist commands.

    Log-score update per step:

        log S(g) -= 0.5 * [(v - v*_g).T / sigma_v^2 (v - v*_g)
                         + lambda_R * (w - w*_g).T / sigma_w^2 (w - w*_g)]

    where (v, w) is the observed noisy unit direction and (v*_g, w*_g)
    is the ideal twist direction toward goal g. This mirrors the 2D
    Gaussian observer in `run_sa_headless.py`.
    """

    def __init__(self, sigma_v=0.5, sigma_w=0.5, lambda_R=0.0,
                 threshold=0.9):
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        self.lambda_R = lambda_R
        self.use_rotation = lambda_R > 0.0
        self.threshold = threshold
        self.log_scores = {}
        self.distribution = {}

    def reset(self, ee_pos, ee_R=None):
        self.log_scores = {}
        self.distribution = {}

    def update(self, ee_pos, ee_R, goals, observed_twist):
        """Update posterior given a 6-vector twist observation.

        Args:
            ee_pos: (3,) current EE position.
            ee_R: (3, 3) current EE rotation, or None.
            goals: dict {goal_id: {'pos': (3,), 'R': (3,3) or None}}.
            observed_twist: (6,) observed unit twist with noise.

        Returns (distribution, top_goal_id, top_prob).
        """
        if observed_twist is None or len(goals) == 0:
            return {}, None, 0.0

        v_obs = np.asarray(observed_twist[:3], dtype=float)
        w_obs = np.asarray(observed_twist[3:], dtype=float) \
            if len(observed_twist) >= 6 else np.zeros(3)

        if np.linalg.norm(v_obs) < 1e-8 and not self.use_rotation:
            if self.distribution:
                top_goal = max(self.distribution, key=self.distribution.get)
                return self.distribution, top_goal, self.distribution[top_goal]
            return {}, None, 0.0

        if not self.log_scores:
            for gid in goals:
                self.log_scores[gid] = 0.0

        inv_v = 1.0 / (self.sigma_v ** 2)
        inv_w = 1.0 / (self.sigma_w ** 2) if self.use_rotation else 0.0

        for gid, g in goals.items():
            g_pos = g["pos"]
            g_R = g.get("R")
            v_star, w_star = _ideal_twist(
                ee_pos, ee_R, g_pos, g_R, self.use_rotation)
            dv = v_obs - v_star
            score = 0.5 * inv_v * float(np.dot(dv, dv))
            if self.use_rotation:
                dw = w_obs - w_star
                score += 0.5 * self.lambda_R * inv_w * float(np.dot(dw, dw))
            self.log_scores[gid] -= score

        scores = np.array(list(self.log_scores.values()))
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        self.distribution = {gid: float(p)
                             for gid, p in zip(self.log_scores.keys(), probs)}
        top_goal = max(self.distribution, key=self.distribution.get)
        top_prob = self.distribution[top_goal]
        return self.distribution, top_goal, top_prob


# ---------------------------------------------------------------------------
# Boltzmann path-efficiency observer (SE(3))
# ---------------------------------------------------------------------------

class PathEfficiencySE3Inference:
    """Boltzmann path-efficiency observer with an SE(3) metric.

    Cost C(traj, g) = (L + d_remaining) / d_start, where L, d_remaining
    and d_start are measured under the weighted product metric

        d(T_a, T_b)^2 = ||p_a - p_b||^2 + lambda_R * ||log(R_a^T R_b)||^2

    Path length L accumulates both translational and (weighted)
    rotational displacements between successive EE poses. Posterior is
    softmax(-beta * C).

    When lambda_R = 0 this reduces to translation-only path efficiency
    in 3D.
    """

    def __init__(self, beta=5.0, lambda_R=0.0, threshold=0.9,
                 min_step_for_path=1e-3):
        self.beta = beta
        self.lambda_R = lambda_R
        self.threshold = threshold
        self.min_step_for_path = min_step_for_path
        self.path_length = 0.0
        self.start_pos = None
        self.start_R = None
        self.prev_pos = None
        self.prev_R = None
        self.distribution = {}

    def reset(self, ee_pos, ee_R=None):
        self.start_pos = np.asarray(ee_pos, dtype=float).copy()
        self.start_R = np.asarray(ee_R, dtype=float).copy() \
            if ee_R is not None else None
        self.prev_pos = self.start_pos.copy()
        self.prev_R = None if self.start_R is None else self.start_R.copy()
        self.path_length = 0.0
        self.distribution = {}

    def _d(self, p1, R1, p2, R2):
        """Weighted SE(3) distance between two poses."""
        dp = p1 - p2
        dsq = float(np.dot(dp, dp))
        if self.lambda_R > 0 and R1 is not None and R2 is not None:
            R_rel = R1.T @ R2
            w = rot_log(R_rel)
            dsq += self.lambda_R * float(np.dot(w, w))
        return float(np.sqrt(max(dsq, 0.0)))

    def update(self, ee_pos, ee_R, goals, observed_twist=None):
        """Update posterior given current EE pose.

        Args:
            ee_pos: (3,) current EE position.
            ee_R: (3,3) current EE rotation or None.
            goals: dict {goal_id: {'pos': (3,), 'R': (3,3) or None}}.
            observed_twist: ignored — path efficiency uses EE pose only.
        """
        if self.prev_pos is not None:
            delta = self._d(self.prev_pos, self.prev_R,
                            np.asarray(ee_pos), ee_R)
            if delta >= self.min_step_for_path:
                self.path_length += delta
        self.prev_pos = np.asarray(ee_pos, dtype=float).copy()
        self.prev_R = None if ee_R is None else np.asarray(ee_R).copy()

        if self.start_pos is None or len(goals) == 0:
            return {}, None, 0.0

        scores = {}
        for gid, g in goals.items():
            g_pos = np.asarray(g["pos"], dtype=float)
            g_R = g.get("R")
            d_sg = max(self._d(self.start_pos, self.start_R, g_pos, g_R), 0.01)
            d_qg = self._d(np.asarray(ee_pos), ee_R, g_pos, g_R)
            cost = (self.path_length + d_qg) / d_sg
            scores[gid] = -self.beta * cost

        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        self.distribution = {k: v / total for k, v in exp_scores.items()}
        top_goal = max(self.distribution, key=self.distribution.get)
        top_prob = self.distribution[top_goal]
        return self.distribution, top_goal, top_prob
