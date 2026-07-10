"""ROS-free shared-autonomy primitives used by headless simulation."""
from __future__ import annotations

import numpy as np


class GaussianDirectionInference:
    """Bayesian intent inference using Gaussian direction model (2D)."""

    def __init__(self, sigma=0.5, threshold=0.9):
        self.sigma = sigma
        self.sigma_inv = np.eye(2) / (sigma ** 2)
        self.threshold = threshold
        self.log_scores = {}
        self.distribution = {}

    def reset(self, ee_pos):
        self.log_scores = {}
        self.distribution = {}

    def update(self, ee_pos, goal_positions, observed_velocity=None):
        if observed_velocity is None or len(goal_positions) == 0:
            return {}, None, 0.0

        u_t = observed_velocity
        if np.linalg.norm(u_t) < 1e-6:
            if self.distribution:
                top_goal = max(self.distribution, key=self.distribution.get)
                return self.distribution, top_goal, self.distribution[top_goal]
            return {}, None, 0.0

        if not self.log_scores:
            for gid in goal_positions:
                self.log_scores[gid] = 0.0

        for gid, gpos in goal_positions.items():
            direction = gpos - ee_pos
            dist = np.linalg.norm(direction)
            mu_g = direction / dist if dist > 1e-6 else np.zeros(2)
            diff = u_t - mu_g
            self.log_scores[gid] -= 0.5 * diff @ self.sigma_inv @ diff

        scores = np.array(list(self.log_scores.values()))
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum()
        self.distribution = {gid: float(p)
                             for gid, p in zip(self.log_scores.keys(), probs)}

        top_goal = max(self.distribution, key=self.distribution.get)
        top_prob = self.distribution[top_goal]
        return self.distribution, top_goal, top_prob


class PathEfficiencyInference:
    """Bayesian intent inference using path efficiency."""

    def __init__(self, beta=5.0, threshold=0.9, min_step_for_path=0.001):
        self.beta = beta
        self.threshold = threshold
        self.min_step_for_path = min_step_for_path
        self.path_length = 0.0
        self.start_pos = None
        self.prev_pos = None
        self.distribution = {}

    def reset(self, ee_pos):
        self.start_pos = ee_pos.copy()
        self.prev_pos = ee_pos.copy()
        self.path_length = 0.0
        self.distribution = {}

    def update(self, ee_pos, goal_positions, observed_velocity=None):
        if self.prev_pos is not None:
            delta = np.linalg.norm(ee_pos - self.prev_pos)
            if delta >= self.min_step_for_path:
                self.path_length += delta
        self.prev_pos = ee_pos.copy()

        if self.start_pos is None or len(goal_positions) == 0:
            return {}, None, 0.0

        scores = {}
        for gid, gpos in goal_positions.items():
            d_sg = max(np.linalg.norm(gpos - self.start_pos), 0.01)
            d_qg = np.linalg.norm(gpos - ee_pos)
            scores[gid] = -self.beta * (self.path_length + d_qg) / d_sg

        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        self.distribution = {k: v / total for k, v in exp_scores.items()}

        top_goal = max(self.distribution, key=self.distribution.get)
        top_prob = self.distribution[top_goal]
        return self.distribution, top_goal, top_prob


class SimulatedJoystick:
    """Noisy Cartesian velocity command generator for synthetic users."""

    def __init__(self, noise_sigma=0.03, max_speed=0.05, gain=2.0):
        self.noise_sigma = noise_sigma
        self.max_speed = max_speed
        self.gain = gain
        self.rng = None

    def compute_velocity(self, ee_pos, target_pos):
        direction = target_pos - ee_pos
        dist = np.linalg.norm(direction)
        if dist < 0.005:
            return self.rng.normal(0, self.noise_sigma, 3)
        speed = min(self.max_speed, self.gain * dist)
        vel_ideal = (direction / dist) * speed
        noise = self.rng.normal(0, self.noise_sigma, 3)
        return vel_ideal + noise


def select_transition_spec(target_spec, inferred_spec, benchmark_mode):
    """Choose task progression semantics independently of result recording."""
    return target_spec if benchmark_mode else inferred_spec
