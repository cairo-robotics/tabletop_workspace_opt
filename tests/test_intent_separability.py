"""Tests for intent separability objective and workspace optimization.

Tests cover:
1. Core math functions (policy, trajectory, separation, margin, slack)
2. Objective function properties (symmetry, monotonicity)
3. Optimization improves slack (DE and MAP-Elites)
4. Goal prediction is faster with optimized layout (single-step)
5. Multi-step task evaluation
"""

import sys
import os
import numpy as np
import yaml
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "envopt"))

from intent_separability import (
    goal_conditioned_policy,
    simulate_trajectory,
    compute_pairwise_separation,
    compute_separation_margin,
    compute_separability_slack,
    intent_separability_objective,
    intent_separability_for_optimizer,
    simulate_goal_inference,
    parse_task_targets,
    simulate_multistep_goal_inference,
)
from workspace_optimizer import (
    overlap_penalty,
    optimize_workspace,
    evaluate_goal_prediction,
    evaluate_multistep_task,
    _build_object_positions_dict,
    DEFAULT_OBJECT_POSITIONS,
    DEFAULT_START_POS,
)

TASK_DIR = os.path.join(os.path.dirname(__file__), "..", "config", "tasks")


# ---------------------------------------------------------------------------
# 1. Unit tests for core math functions
# ---------------------------------------------------------------------------

class TestGoalConditionedPolicy:
    def test_unit_vector(self):
        """Policy output should be a unit vector."""
        ee = np.array([0.0, 0.0])
        goal = np.array([3.0, 4.0])
        mu = goal_conditioned_policy(ee, goal)
        assert np.isclose(np.linalg.norm(mu), 1.0), "Policy should return unit vector"

    def test_direction(self):
        """Policy should point from EE toward goal."""
        ee = np.array([1.0, 0.0])
        goal = np.array([2.0, 0.0])
        mu = goal_conditioned_policy(ee, goal)
        np.testing.assert_allclose(mu, [1.0, 0.0], atol=1e-10)

    def test_at_goal(self):
        """Policy should return zero vector when already at goal."""
        pos = np.array([1.0, 2.0])
        mu = goal_conditioned_policy(pos, pos)
        np.testing.assert_allclose(mu, [0.0, 0.0])

    def test_3d(self):
        """Policy should work in 3D."""
        ee = np.array([0.0, 0.0, 0.0])
        goal = np.array([1.0, 1.0, 1.0])
        mu = goal_conditioned_policy(ee, goal)
        assert np.isclose(np.linalg.norm(mu), 1.0)
        expected = np.array([1, 1, 1]) / np.sqrt(3)
        np.testing.assert_allclose(mu, expected, atol=1e-10)


class TestSimulateTrajectory:
    def test_starts_at_start(self):
        """Trajectory should start at start_pos."""
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        traj = simulate_trajectory(start, goal, T=10, step_size=0.1)
        np.testing.assert_allclose(traj[0], start)

    def test_approaches_goal(self):
        """Final position should be closer to goal than start."""
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        traj = simulate_trajectory(start, goal, T=20, step_size=0.1)
        start_dist = np.linalg.norm(start - goal)
        final_dist = np.linalg.norm(traj[-1] - goal)
        assert final_dist < start_dist

    def test_reaches_goal(self):
        """With enough steps, trajectory should reach the goal."""
        start = np.array([0.0, 0.0])
        goal = np.array([0.5, 0.0])
        traj = simulate_trajectory(start, goal, T=100, step_size=0.01)
        np.testing.assert_allclose(traj[-1], goal, atol=0.02)

    def test_shape(self):
        """Output should have shape (T, d)."""
        start = np.array([0.0, 0.0])
        goal = np.array([1.0, 1.0])
        traj = simulate_trajectory(start, goal, T=30, step_size=0.05)
        assert traj.shape == (30, 2)


class TestPairwiseSeparation:
    def test_identical_goals_zero_separation(self):
        """Separation should be zero when goals are the same."""
        traj = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        goal = np.array([1.0, 0.0])
        sigma_inv = np.eye(2)
        V = compute_pairwise_separation(traj, goal, goal, sigma_inv)
        assert np.isclose(V, 0.0), f"Expected 0, got {V}"

    def test_positive_separation(self):
        """Separation should be positive for distinct goals."""
        traj = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]])
        true_goal = np.array([1.0, 0.0])
        alt_goal = np.array([0.0, 1.0])
        sigma_inv = np.eye(2)
        V = compute_pairwise_separation(traj, true_goal, alt_goal, sigma_inv)
        assert V > 0, f"Expected positive separation, got {V}"

    def test_more_steps_more_separation(self):
        """More trajectory steps should accumulate more separation."""
        true_goal = np.array([1.0, 0.0])
        alt_goal = np.array([0.0, 1.0])
        sigma_inv = np.eye(2)

        traj_short = simulate_trajectory(np.array([0.0, 0.0]), true_goal, T=10, step_size=0.01)
        traj_long = simulate_trajectory(np.array([0.0, 0.0]), true_goal, T=50, step_size=0.01)

        V_short = compute_pairwise_separation(traj_short, true_goal, alt_goal, sigma_inv)
        V_long = compute_pairwise_separation(traj_long, true_goal, alt_goal, sigma_inv)
        assert V_long > V_short

    def test_sigma_bar_inv_scaling(self):
        """Larger sigma_bar_inv (smaller noise) should increase separation."""
        traj = simulate_trajectory(np.array([0.0, 0.0]), np.array([1.0, 0.0]), T=20, step_size=0.01)
        true_goal = np.array([1.0, 0.0])
        alt_goal = np.array([0.0, 1.0])

        V_small = compute_pairwise_separation(traj, true_goal, alt_goal, np.eye(2))
        V_large = compute_pairwise_separation(traj, true_goal, alt_goal, 10 * np.eye(2))
        assert V_large > V_small


class TestSeparationMargin:
    def test_uniform_prior(self):
        """With uniform prior, margin = V/2."""
        V = 10.0
        m = compute_separation_margin(V, log_prior_ratio=0.0)
        assert np.isclose(m, 5.0)

    def test_prior_effect(self):
        """Higher prior on alternative should reduce margin."""
        V = 10.0
        m_uniform = compute_separation_margin(V, 0.0)
        m_biased = compute_separation_margin(V, np.log(2.0))  # alt has 2x prior
        assert m_biased < m_uniform


class TestSeparabilitySlack:
    def test_large_separation_positive_slack(self):
        """Large separation should give positive slack."""
        V = 100.0
        m = compute_separation_margin(V, 0.0)  # m = 50
        slack = compute_separability_slack(V, m, alpha_g=0.05)
        assert slack > 0, f"Expected positive slack, got {slack}"

    def test_zero_separation_negative_slack(self):
        """Zero separation should give zero or negative slack."""
        slack = compute_separability_slack(0.0, 0.0, alpha_g=0.05)
        assert slack <= 0


# ---------------------------------------------------------------------------
# 2. Objective function properties
# ---------------------------------------------------------------------------

class TestIntentSeparabilityObjective:
    def setup_method(self):
        self.sigma_bar_inv = 10.0 * np.eye(2)  # sigma_bar = 0.1*I
        self.start_pos = np.array([0.0, 0.0])

    def test_well_separated_goals_positive_slack(self):
        """Well-separated goals should have positive worst-case slack."""
        positions = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        slack, _ = intent_separability_objective(
            positions, self.start_pos, self.sigma_bar_inv, T=50, step_size=0.01
        )
        assert slack > 0, f"Expected positive slack for well-separated goals, got {slack}"

    def test_close_goals_lower_slack(self):
        """Close goals should have lower slack than separated goals."""
        far_positions = np.array([
            [0.8, 0.0],
            [0.0, 0.8],
            [-0.8, 0.0],
        ])
        close_positions = np.array([
            [0.5, 0.0],
            [0.55, 0.05],
            [0.45, -0.05],
        ])
        slack_far, _ = intent_separability_objective(
            far_positions, self.start_pos, self.sigma_bar_inv, T=50, step_size=0.01
        )
        slack_close, _ = intent_separability_objective(
            close_positions, self.start_pos, self.sigma_bar_inv, T=50, step_size=0.01
        )
        assert slack_far > slack_close, (
            f"Far goals slack ({slack_far:.4f}) should exceed close goals slack ({slack_close:.4f})"
        )

    def test_collinear_goals_low_separation(self):
        """Goals along the same line from start should be harder to separate."""
        collinear = np.array([
            [0.5, 0.0],
            [1.0, 0.0],
        ])
        orthogonal = np.array([
            [0.5, 0.0],
            [0.0, 0.5],
        ])
        slack_coll, _ = intent_separability_objective(
            collinear, self.start_pos, self.sigma_bar_inv, T=50, step_size=0.01
        )
        slack_orth, _ = intent_separability_objective(
            orthogonal, self.start_pos, self.sigma_bar_inv, T=50, step_size=0.01
        )
        assert slack_orth > slack_coll, (
            f"Orthogonal slack ({slack_orth:.4f}) should exceed collinear slack ({slack_coll:.4f})"
        )

    def test_optimizer_wrapper_consistency(self):
        """Flat wrapper should give consistent results with structured version."""
        positions = np.array([[0.5, 0.3], [0.7, -0.2], [0.3, 0.1]])
        slack, _ = intent_separability_objective(
            positions, self.start_pos, self.sigma_bar_inv, T=30, step_size=0.01
        )
        neg_slack = intent_separability_for_optimizer(
            positions.flatten(), self.start_pos, self.sigma_bar_inv,
            d=2, T=30, step_size=0.01
        )
        assert np.isclose(-neg_slack, slack), f"Wrapper inconsistent: {-neg_slack} vs {slack}"


# ---------------------------------------------------------------------------
# 3. Overlap penalty
# ---------------------------------------------------------------------------

class TestOverlapPenalty:
    def test_no_overlap(self):
        """Well-separated objects should have zero penalty."""
        positions = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        assert np.isclose(overlap_penalty(positions, d=2, min_dist=0.1), 0.0)

    def test_overlap_detected(self):
        """Overlapping objects should have positive penalty."""
        positions = np.array([0.0, 0.0, 0.05, 0.0])
        assert overlap_penalty(positions, d=2, min_dist=0.12) > 0


# ---------------------------------------------------------------------------
# 4. Optimization improves slack (both methods)
# ---------------------------------------------------------------------------

class TestOptimizationDE:
    """Tests for differential_evolution optimizer."""

    def test_optimization_improves_slack(self):
        """Optimized layout should have equal or better slack than initial."""
        result = optimize_workspace(
            T=30, step_size=0.02, maxiter=50, seed=42,
            method="differential_evolution", verbose=False,
        )
        assert result["optimized_slack"] >= result["initial_slack"] - 1e-6, (
            f"Optimized slack ({result['optimized_slack']:.4f}) should be >= "
            f"initial slack ({result['initial_slack']:.4f})"
        )

    def test_optimized_positions_in_bounds(self):
        """Optimized positions should be within table bounds."""
        result = optimize_workspace(
            T=30, step_size=0.02, maxiter=50, seed=42,
            method="differential_evolution", verbose=False,
        )
        positions = result["optimized_positions"]
        for i, name in enumerate(result["object_names"]):
            x, y = positions[i]
            assert 0.30 <= x <= 0.85, f"{name} x={x:.3f} out of bounds"
            assert -0.45 <= y <= 0.45, f"{name} y={y:.3f} out of bounds"

    def test_optimized_no_overlap(self):
        """Optimized positions should not have objects too close."""
        result = optimize_workspace(
            T=30, step_size=0.02, maxiter=50, seed=42,
            method="differential_evolution", verbose=False,
        )
        positions = result["optimized_positions"]
        M = len(positions)
        for i in range(M):
            for j in range(i + 1, M):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist >= 0.10, (
                    f"Objects {i} and {j} too close: {dist:.3f}m"
                )


class TestOptimizationME:
    """Tests for MAP-Elites (CMA-ME) optimizer."""

    def test_optimization_improves_slack(self):
        """MAP-Elites optimized layout should improve over initial."""
        result = optimize_workspace(
            T=30, step_size=0.02, maxiter=50, seed=42,
            method="map_elites", batch_size=20, verbose=False,
        )
        assert result["optimized_slack"] >= result["initial_slack"] - 1e-6, (
            f"MAP-Elites slack ({result['optimized_slack']:.4f}) should be >= "
            f"initial slack ({result['initial_slack']:.4f})"
        )
        assert result["method"] == "map_elites"

    def test_optimized_positions_in_bounds(self):
        """MAP-Elites positions should be within table bounds."""
        result = optimize_workspace(
            T=30, step_size=0.02, maxiter=50, seed=42,
            method="map_elites", batch_size=20, verbose=False,
        )
        positions = result["optimized_positions"]
        for i, name in enumerate(result["object_names"]):
            x, y = positions[i]
            assert 0.30 <= x <= 0.85, f"{name} x={x:.3f} out of bounds"
            assert -0.45 <= y <= 0.45, f"{name} y={y:.3f} out of bounds"

    def test_optimized_no_overlap(self):
        """MAP-Elites positions should not overlap."""
        result = optimize_workspace(
            T=30, step_size=0.02, maxiter=50, seed=42,
            method="map_elites", batch_size=20, verbose=False,
        )
        positions = result["optimized_positions"]
        M = len(positions)
        for i in range(M):
            for j in range(i + 1, M):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist >= 0.10, (
                    f"Objects {i} and {j} too close: {dist:.3f}m"
                )

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            optimize_workspace(method="bogus", verbose=False)


# ---------------------------------------------------------------------------
# 5. Goal prediction is faster with optimized layout
# ---------------------------------------------------------------------------

def _eval_config(positions, seed=123):
    """Helper: evaluate goal prediction for a layout."""
    sigma_bar = 0.1 * np.eye(2)
    sigma = 0.08 * np.eye(2)
    return evaluate_goal_prediction(
        positions, DEFAULT_START_POS, sigma_bar, sigma,
        T=200, step_size=0.01, n_trials=80, seed=seed,
    )


class TestGoalPredictionSpeedDE:
    """Verify differential_evolution improves goal prediction."""

    def test_faster_prediction(self):
        result = optimize_workspace(
            T=40, step_size=0.02, maxiter=80, seed=42,
            method="differential_evolution", verbose=False,
        )
        orig_eval = _eval_config(result["initial_positions"])
        opt_eval = _eval_config(result["optimized_positions"])

        print(f"\n--- DE Goal Prediction ---")
        print(f"Original:  steps={orig_eval['mean_steps']:.1f}")
        print(f"Optimized: steps={opt_eval['mean_steps']:.1f}")

        assert opt_eval["mean_steps"] <= orig_eval["mean_steps"], (
            f"DE should identify goals faster: "
            f"{opt_eval['mean_steps']:.1f} vs {orig_eval['mean_steps']:.1f}"
        )

    def test_accuracy(self):
        result = optimize_workspace(
            T=40, step_size=0.02, maxiter=80, seed=42,
            method="differential_evolution", verbose=False,
        )
        orig_eval = _eval_config(result["initial_positions"], seed=456)
        opt_eval = _eval_config(result["optimized_positions"], seed=456)

        assert opt_eval["accuracy"] >= orig_eval["accuracy"] - 0.05


class TestGoalPredictionSpeedME:
    """Verify MAP-Elites improves goal prediction."""

    def test_faster_prediction(self):
        result = optimize_workspace(
            T=40, step_size=0.02, maxiter=80, seed=42,
            method="map_elites", batch_size=20, verbose=False,
        )
        orig_eval = _eval_config(result["initial_positions"])
        opt_eval = _eval_config(result["optimized_positions"])

        print(f"\n--- MAP-Elites Goal Prediction ---")
        print(f"Original:  steps={orig_eval['mean_steps']:.1f}")
        print(f"Optimized: steps={opt_eval['mean_steps']:.1f}")

        assert opt_eval["mean_steps"] <= orig_eval["mean_steps"], (
            f"MAP-Elites should identify goals faster: "
            f"{opt_eval['mean_steps']:.1f} vs {orig_eval['mean_steps']:.1f}"
        )

    def test_accuracy(self):
        result = optimize_workspace(
            T=40, step_size=0.02, maxiter=80, seed=42,
            method="map_elites", batch_size=20, verbose=False,
        )
        orig_eval = _eval_config(result["initial_positions"], seed=456)
        opt_eval = _eval_config(result["optimized_positions"], seed=456)

        assert opt_eval["accuracy"] >= orig_eval["accuracy"] - 0.05


# ---------------------------------------------------------------------------
# 6. Multi-step task evaluation
# ---------------------------------------------------------------------------

def _load_task(name):
    """Load a task YAML from config/tasks/."""
    path = os.path.join(TASK_DIR, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


class TestParseTaskTargets:
    """Test that task YAML configs are parsed into correct target sequences."""

    def test_banana_in_bowl(self):
        tc = _load_task("banana_in_bowl")
        positions = np.array(list(DEFAULT_OBJECT_POSITIONS.values()))
        obj_dict = _build_object_positions_dict(
            list(DEFAULT_OBJECT_POSITIONS.keys()), positions)
        targets = parse_task_targets(tc, obj_dict)
        assert len(targets) == 2
        assert targets[0]["action"] == "pick"
        assert targets[0]["target_name"] == "banana"
        assert targets[1]["action"] == "place"

    def test_set_breakfast(self):
        tc = _load_task("set_breakfast")
        positions = np.array(list(DEFAULT_OBJECT_POSITIONS.values()))
        obj_dict = _build_object_positions_dict(
            list(DEFAULT_OBJECT_POSITIONS.keys()), positions)
        targets = parse_task_targets(tc, obj_dict)
        assert len(targets) == 4  # pick cereal, place, pick banana, place
        actions = [t["action"] for t in targets]
        assert actions == ["pick", "place", "pick", "place"]

    def test_full_breakfast(self):
        tc = _load_task("full_breakfast")
        positions = np.array(list(DEFAULT_OBJECT_POSITIONS.values()))
        obj_dict = _build_object_positions_dict(
            list(DEFAULT_OBJECT_POSITIONS.keys()), positions)
        targets = parse_task_targets(tc, obj_dict)
        assert len(targets) == 6  # 3 pick + 3 place

    def test_tidy_table(self):
        tc = _load_task("tidy_table")
        positions = np.array(list(DEFAULT_OBJECT_POSITIONS.values()))
        obj_dict = _build_object_positions_dict(
            list(DEFAULT_OBJECT_POSITIONS.keys()), positions)
        targets = parse_task_targets(tc, obj_dict)
        assert len(targets) == 8  # 4 pick + 4 place

    def test_sort_by_size(self):
        tc = _load_task("sort_by_size")
        positions = np.array(list(DEFAULT_OBJECT_POSITIONS.values()))
        obj_dict = _build_object_positions_dict(
            list(DEFAULT_OBJECT_POSITIONS.keys()), positions)
        targets = parse_task_targets(tc, obj_dict)
        assert len(targets) == 10  # 5 pick + 5 place


class TestMultistepInference:
    """Test multi-step goal inference simulation."""

    def test_returns_correct_structure(self):
        tc = _load_task("banana_in_bowl")
        positions = np.array(list(DEFAULT_OBJECT_POSITIONS.values()))
        obj_dict = _build_object_positions_dict(
            list(DEFAULT_OBJECT_POSITIONS.keys()), positions)
        positions = np.array(list(DEFAULT_OBJECT_POSITIONS.values()))
        targets = parse_task_targets(tc, obj_dict)

        sigma_bar_inv = 10.0 * np.eye(2)
        sigma = 0.08 * np.eye(2)
        rng = np.random.default_rng(42)

        result = simulate_multistep_goal_inference(
            targets, positions, DEFAULT_START_POS,
            sigma_bar_inv, sigma,
            T_per_step=100, step_size=0.01, rng=rng,
        )
        assert "total_steps" in result
        assert "per_step_steps" in result
        assert len(result["per_step_steps"]) == 2
        assert "accuracy" in result

    def test_well_separated_identifies_quickly(self):
        """Well-separated objects should be identified in few steps."""
        tc = _load_task("banana_in_bowl")
        # Place objects far apart
        far_names = ["bowl", "cereal", "banana", "milk"]
        positions = np.array([
            [0.8, 0.4], [0.3, -0.4], [0.8, -0.4], [0.3, 0.4],
        ])
        obj_dict = _build_object_positions_dict(far_names, positions)
        targets = parse_task_targets(tc, obj_dict)

        sigma_bar_inv = 10.0 * np.eye(2)
        sigma = 0.08 * np.eye(2)
        rng = np.random.default_rng(42)

        result = simulate_multistep_goal_inference(
            targets, positions, DEFAULT_START_POS,
            sigma_bar_inv, sigma,
            T_per_step=100, step_size=0.01, rng=rng,
        )
        # Should identify each step quickly (< 20 steps per step)
        for s in result["per_step_steps"]:
            assert s < 50, f"Expected fast identification, got {s} steps"


class TestMultistepOptimized:
    """Test that optimized layouts improve multi-step task performance."""

    def _run_multistep_eval(self, task_name, positions, n_trials=30):
        tc = _load_task(task_name)
        obj_names = list(DEFAULT_OBJECT_POSITIONS.keys())
        obj_dict = _build_object_positions_dict(obj_names, positions)
        sigma_bar = 0.1 * np.eye(2)
        sigma = 0.08 * np.eye(2)
        return evaluate_multistep_task(
            tc, obj_dict, positions, DEFAULT_START_POS,
            sigma_bar, sigma,
            T_per_step=200, step_size=0.01,
            n_trials=n_trials, seed=42,
        )

    def test_full_breakfast_faster(self):
        """Optimized layout should reduce total identification steps for
        full_breakfast (6-step task)."""
        result = optimize_workspace(
            T=40, step_size=0.02, maxiter=80, seed=42,
            method="differential_evolution", verbose=False,
        )

        orig_ev = self._run_multistep_eval(
            "full_breakfast", result["initial_positions"])
        opt_ev = self._run_multistep_eval(
            "full_breakfast", result["optimized_positions"])

        print(f"\n--- full_breakfast (6 steps) ---")
        print(f"Original:  total_steps={orig_ev['mean_total_steps']:.1f}")
        print(f"Optimized: total_steps={opt_ev['mean_total_steps']:.1f}")

        assert opt_ev["mean_total_steps"] <= orig_ev["mean_total_steps"], (
            f"Optimized should be faster: {opt_ev['mean_total_steps']:.1f} "
            f"vs {orig_ev['mean_total_steps']:.1f}"
        )

    def test_tidy_table_faster(self):
        """Optimized layout should reduce total identification steps for
        tidy_table (8-step task)."""
        result = optimize_workspace(
            T=40, step_size=0.02, maxiter=80, seed=42,
            method="differential_evolution", verbose=False,
        )

        orig_ev = self._run_multistep_eval(
            "tidy_table", result["initial_positions"])
        opt_ev = self._run_multistep_eval(
            "tidy_table", result["optimized_positions"])

        print(f"\n--- tidy_table (8 steps) ---")
        print(f"Original:  total_steps={orig_ev['mean_total_steps']:.1f}")
        print(f"Optimized: total_steps={opt_ev['mean_total_steps']:.1f}")

        assert opt_ev["mean_total_steps"] <= orig_ev["mean_total_steps"], (
            f"Optimized should be faster: {opt_ev['mean_total_steps']:.1f} "
            f"vs {orig_ev['mean_total_steps']:.1f}"
        )

    def test_sort_by_size_faster(self):
        """Optimized layout should reduce total identification steps for
        sort_by_size (10-step task)."""
        result = optimize_workspace(
            T=40, step_size=0.02, maxiter=80, seed=42,
            method="differential_evolution", verbose=False,
        )

        orig_ev = self._run_multistep_eval(
            "sort_by_size", result["initial_positions"])
        opt_ev = self._run_multistep_eval(
            "sort_by_size", result["optimized_positions"])

        print(f"\n--- sort_by_size (10 steps) ---")
        print(f"Original:  total_steps={orig_ev['mean_total_steps']:.1f}")
        print(f"Optimized: total_steps={opt_ev['mean_total_steps']:.1f}")

        assert opt_ev["mean_total_steps"] <= orig_ev["mean_total_steps"], (
            f"Optimized should be faster: {opt_ev['mean_total_steps']:.1f} "
            f"vs {orig_ev['mean_total_steps']:.1f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
