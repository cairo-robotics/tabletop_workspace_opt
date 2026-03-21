#!/usr/bin/env python3
"""Evaluate intent identification for all scenes and tasks.

Compares original vs optimized layouts using Monte Carlo simulation
of noisy joystick-based goal inference (no ROS/sim required).
"""
import sys
import os
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "envopt"))

from intent_separability import parse_task_targets, simulate_multistep_goal_inference
from workspace_optimizer import (
    evaluate_goal_prediction,
    evaluate_multistep_task,
    _build_object_positions_dict,
    DEFAULT_START_POS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES_DIR = os.path.join(PROJECT_ROOT, "config", "scenes")
TASKS_DIR = os.path.join(PROJECT_ROOT, "config", "tasks")

# Map scenes to their applicable tasks
SCENE_TASKS = {
    "scene_breakfast": [
        "banana_in_bowl", "cereal_next_to_bowl", "set_breakfast", "full_breakfast",
    ],
    "scene_desk": ["desk_organize"],
    "scene_kitchen_prep": ["kitchen_prep_full"],
}

SIGMA_BAR = 0.1 * np.eye(2)
SIGMA = 0.08 * np.eye(2)
N_TRIALS = 50


def load_scene_positions(scene_name):
    """Load object positions from scene config YAML."""
    import re
    xml_path = os.path.join(PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    config_path = os.path.join(SCENES_DIR, f"{scene_name}.yaml")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["scene"]

    # Parse positions from XML keyframe
    with open(xml_path) as f:
        xml = f.read()
    match = re.search(r'<key\s+name="home"\s+qpos="([^"]+)"', xml)
    qpos = [float(x) for x in match.group(1).strip().split()]

    names = [obj["short_name"] for obj in cfg["objects"]]
    mujoco_names = [obj["mujoco_name"] for obj in cfg["objects"]]

    positions = {}
    for i, short in enumerate(names):
        idx = 9 + i * 7  # 9 robot DOF, then 7 per object
        positions[short] = np.array(qpos[idx:idx+2])

    return names, positions


def run_evaluation():
    results = {}

    for scene, tasks in SCENE_TASKS.items():
        print(f"\n{'=' * 70}")
        print(f"Scene: {scene}")
        print(f"{'=' * 70}")

        # Load original and optimized positions
        orig_names, orig_pos = load_scene_positions(scene)
        opt_names, opt_pos = load_scene_positions(f"{scene}_optimized")

        orig_arr = np.array([orig_pos[n] for n in orig_names])
        opt_arr = np.array([opt_pos[n] for n in opt_names])

        # Single-step evaluation
        print(f"\n--- Single-step goal identification ---")
        for label, positions in [("Original", orig_arr), ("Optimized", opt_arr)]:
            ev = evaluate_goal_prediction(
                positions, DEFAULT_START_POS, SIGMA_BAR, SIGMA,
                n_trials=N_TRIALS, seed=42,
            )
            print(f"  {label:12s}: accuracy={ev['accuracy']:.2%}, "
                  f"steps={ev['mean_steps']:.1f}")
            for i, name in enumerate(orig_names):
                print(f"    {name:15s}: acc={ev['per_goal_accuracy'][i]:.2%}, "
                      f"steps={ev['per_goal_mean_steps'][i]:.1f}")

        # Multi-step task evaluation
        for task_name in tasks:
            task_path = os.path.join(TASKS_DIR, f"{task_name}.yaml")
            with open(task_path) as f:
                tc = yaml.safe_load(f)

            n_steps = len(tc["task"]["steps"])
            print(f"\n--- Task: {task_name} ({n_steps} steps) ---")

            task_results = {}
            for label, positions, pos_dict_raw in [
                ("Original", orig_arr, orig_pos),
                ("Optimized", opt_arr, opt_pos),
            ]:
                obj_dict = _build_object_positions_dict(orig_names, positions)
                # Also add the raw positions for any referenced object
                for k, v in pos_dict_raw.items():
                    if k not in obj_dict:
                        obj_dict[k] = v

                ev = evaluate_multistep_task(
                    tc, obj_dict, positions, DEFAULT_START_POS,
                    SIGMA_BAR, SIGMA,
                    T_per_step=200, step_size=0.01,
                    n_trials=N_TRIALS, seed=42,
                )
                task_results[label] = ev

                print(f"  {label:12s}: total_steps={ev['mean_total_steps']:.1f}, "
                      f"accuracy={ev['mean_accuracy']:.2%}")
                for s in range(ev["n_task_steps"]):
                    target = ev["step_targets"][s]
                    print(f"    Step {s+1:2d} ({target:25s}): "
                          f"steps={ev['per_step_mean_steps'][s]:.1f}, "
                          f"acc={ev['per_step_accuracy'][s]:.2%}")

            # Speedup
            orig_total = task_results["Original"]["mean_total_steps"]
            opt_total = task_results["Optimized"]["mean_total_steps"]
            speedup = orig_total / max(opt_total, 0.1)
            print(f"  Speedup: {speedup:.2f}x")

            results[(scene, task_name)] = task_results

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Scene':<20} {'Task':<25} {'Orig Steps':>12} {'Opt Steps':>11} "
          f"{'Speedup':>9} {'Orig Acc':>9} {'Opt Acc':>8}")
    print("-" * 96)

    for (scene, task), res in results.items():
        orig = res["Original"]
        opt = res["Optimized"]
        speedup = orig["mean_total_steps"] / max(opt["mean_total_steps"], 0.1)
        print(f"{scene:<20} {task:<25} {orig['mean_total_steps']:>12.1f} "
              f"{opt['mean_total_steps']:>11.1f} {speedup:>8.2f}x "
              f"{orig['mean_accuracy']:>8.2%} {opt['mean_accuracy']:>7.2%}")


if __name__ == "__main__":
    run_evaluation()
