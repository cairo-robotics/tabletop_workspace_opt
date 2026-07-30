"""Regression tests for the SE(3) shared-autonomy benchmark harness."""
import importlib.util
import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
EVAL_SCRIPTS = os.path.join(SCRIPTS, "eval")
SRC = os.path.join(ROOT, "src")
for path in (EVAL_SCRIPTS, SCRIPTS, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_sa_headless as runner


def _load_compare_module():
    path = os.path.join(EVAL_SCRIPTS, "compare_se3_sa_3d.py")
    spec = importlib.util.spec_from_file_location("compare_se3_sa_3d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load_compare_module()


def test_benchmark_transition_uses_target_without_changing_normal_semantics():
    target = {"id": "pick_a", "object": "a", "action": "pick"}
    inferred = {"id": "pick_b", "object": "b", "action": "pick"}
    assert runner._select_transition_spec(target, inferred, True) is target
    assert runner._select_transition_spec(target, inferred, False) is inferred


def test_target_object_controls_feasibility_filter(monkeypatch, tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("task:\n  pick_objects: [a, b]\n")

    def fake_run(*args, user_sequence, **kwargs):
        objects = user_sequence.split(",")
        steps = []
        for target in objects:
            inferred = "b" if target == "a" else "a"
            steps.append({
                "action": "pick", "inference_steps": 2,
                "target_object": target, "object": inferred,
                "inference_time_s": 1.0,
                # Only target a is correct. Since a is infeasible, accuracy
                # over feasible target trials must be zero.
                "correct": target == "a", "argmax_correct": target == "a",
            })
        return {"step_times": steps}

    monkeypatch.setattr(compare, "run_headless_sa_se3", fake_run)
    result = compare.run_sa_for_layout(
        str(task), "scene", {}, {}, {"a"}, ["a", "b"], {},
        max_orderings=2)
    assert result["n_total_picks"] == 4
    assert result["expected_picks"] == 4
    assert result["n_infeasible_picks"] == 2
    assert result["argmax_accuracy"] == 0.0


def test_missing_pick_fails_count_invariant(monkeypatch, tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text("task:\n  pick_objects: [a, b]\n")
    monkeypatch.setattr(
        compare, "run_headless_sa_se3",
        lambda *args, **kwargs: {"step_times": [{
            "action": "pick", "inference_steps": 2,
            "target_object": "a", "object": "a",
            "inference_time_s": 1.0, "correct": True,
            "argmax_correct": True,
        }]})
    with pytest.raises(RuntimeError, match="Benchmark invariant failed"):
        compare.run_sa_for_layout(
            str(task), "scene", {}, {}, set(), ["a", "b"], {},
            max_orderings=1)


def test_random_se3_layouts_are_deterministic_and_footprint_valid():
    names = ["a", "b", "c"]
    half_extents = {
        "a": (0.04, 0.02), "b": (0.03, 0.05), "c": (0.02, 0.02),
        "fixed": (0.08, 0.08),
    }
    fixed = {"fixed": np.array([0.75, 0.30])}
    first = runner.generate_random_se3_layouts(
        4, names, fixed, half_extents, seed=42)
    second = runner.generate_random_se3_layouts(
        4, names, fixed, half_extents, seed=42)
    for (pos_a, yaw_a), (pos_b, yaw_b) in zip(first, second):
        for name in names:
            np.testing.assert_allclose(pos_a[name], pos_b[name])
            assert yaw_a[name] == yaw_b[name]

    from envopt.grasp_feasibility import _rotated_rect
    from shapely.geometry import Point, box
    table = box(runner.TABLE_BOUNDS_X[0], runner.TABLE_BOUNDS_Y[0],
                runner.TABLE_BOUNDS_X[1], runner.TABLE_BOUNDS_Y[1])
    robot = Point(*runner.START_POS_2D).buffer(runner.ROBOT_EXCLUSION_RADIUS)
    fixed_poly = _rotated_rect(fixed["fixed"], half_extents["fixed"], 0.0)
    for positions, yaws in first:
        polys = []
        for name in names:
            poly = _rotated_rect(positions[name], half_extents[name], yaws[name])
            assert table.covers(poly)
            assert not poly.intersects(robot)
            assert poly.distance(fixed_poly) >= 0.03 - 1e-9
            assert all(poly.distance(other) >= 0.03 - 1e-9
                       for other in polys)
            polys.append(poly)


def test_metadata_merge_rejects_parameter_and_input_changes():
    base = {
        "schema_version": 2,
        "experiment": {
            "git_commit": "abc", "input_hashes": {"a": "1"},
            "parameters": {
                "beta": 5.0, "threshold": 0.9, "noise": 0.03,
                "control_rate": 5.0, "lambda_R": 0.04,
                "arrival_dist": 0.02, "goal_timeout": 30.0,
                "max_speed": 0.05, "seed": 42,
                "n_random": 10, "random_max_orderings": 30,
            },
        },
    }
    sampling_change = {
        "schema_version": 2,
        "experiment": {**base["experiment"], "parameters": {
            **base["experiment"]["parameters"], "n_random": 30}},
    }
    assert compare.metadata_compatible(base, sampling_change)
    runtime_change = {
        "schema_version": 2,
        "experiment": {**base["experiment"], "parameters": {
            **base["experiment"]["parameters"], "noise": 0.01}},
    }
    assert not compare.metadata_compatible(base, runtime_change)
    input_change = {
        "schema_version": 2,
        "experiment": {**base["experiment"], "input_hashes": {"a": "2"}},
    }
    assert not compare.metadata_compatible(base, input_change)
