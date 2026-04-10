#!/usr/bin/env python3
"""Task-aware workspace optimizer.

Optimizes object positions considering the task structure — only objects
that compete at the same state need to be separable. This is more
efficient than optimizing all objects against each other.

Usage:
    python3 scripts/task_aware_optimizer.py config/tasks/kitchen_prep_sa.yaml
    python3 scripts/task_aware_optimizer.py config/tasks/meal_assembly_sa.yaml
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "envopt"))

import numpy as np
import yaml
import argparse
import json
import time
from scipy.optimize import differential_evolution
from intent_separability import intent_separability_for_optimizer
from task_aware import (
    extract_pick_states,
    deduplicate_pick_states,
    task_aware_objective,
    TABLE_BOUNDS_X,
    TABLE_BOUNDS_Y,
    MIN_DIST,
)

START_POS = np.array([0.479, -0.060])


def optimize_for_task(task_config_path, scene_config_path=None,
                      maxiter=200, seed=42, verbose=True):
    """Run task-aware optimization for a given task."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)["task"]

    # Load scene config
    scene_name = task_config.get("scene", "scene_breakfast")
    if scene_config_path is None:
        scene_config_path = os.path.join(
            pkg_root, "config", "scenes", f"{scene_name}.yaml")
    with open(scene_config_path) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    # Get object positions from scene XML
    scene_xml = os.path.join(pkg_root, "src", "assets", f"{scene_name}.xml")
    object_names = []
    object_positions = []
    for obj in scene_cfg["objects"]:
        sn = obj["short_name"]
        object_names.append(sn)
        # Parse position from XML
        import re
        with open(scene_xml) as f:
            xml = f.read()
        match = re.search(
            rf'<body name="{re.escape(obj["mujoco_name"])}" pos="([^"]+)"', xml)
        if match:
            pos = [float(x) for x in match.group(1).split()]
            object_positions.append(pos[:2])  # x, y only
        else:
            object_positions.append([0.5, 0.0])

    object_positions = np.array(object_positions)

    # Extract pick states from task and deduplicate
    pick_states_raw = extract_pick_states(task_config)
    pick_states = deduplicate_pick_states(pick_states_raw)

    if verbose:
        print(f"Task: {task_config['name']}")
        print(f"Scene: {scene_name}")
        print(f"Objects: {object_names}")
        print(f"Pick states: {len(pick_states_raw)} total, {len(pick_states)} unique")
        print(f"Pick disambiguation states:")
        for sname, objs in pick_states:
            print(f"  {sname}: {objs} ({len(objs)} choices)")

    sigma_bar_inv = np.linalg.inv(0.1 * np.eye(2))

    # Only optimize objects that appear in pick states
    pick_objects = set()
    for _, objs in pick_states:
        pick_objects.update(objs)

    # Separate: objects to optimize vs fixed landmarks
    opt_indices = [i for i, n in enumerate(object_names) if n in pick_objects]
    fixed_indices = [i for i, n in enumerate(object_names) if n not in pick_objects]

    opt_names = [object_names[i] for i in opt_indices]
    fixed_names = [object_names[i] for i in fixed_indices]

    if verbose:
        print(f"\nOptimizing: {opt_names}")
        print(f"Fixed (landmarks): {fixed_names}")

    # Initial positions for optimizable objects
    init_opt = object_positions[opt_indices]

    # Compute initial slack
    init_obj = task_aware_objective(
        object_positions[opt_indices].flatten(),
        opt_names, pick_states, START_POS, sigma_bar_inv)
    init_slack = -init_obj

    if verbose:
        print(f"\nInitial worst-case slack: {init_slack:.2f}")

    # Optimize
    M = len(opt_names)
    bounds = [(TABLE_BOUNDS_X[0], TABLE_BOUNDS_X[1]),
              (TABLE_BOUNDS_Y[0], TABLE_BOUNDS_Y[1])] * M

    if verbose:
        print(f"\nRunning differential evolution ({M * 2}D, {maxiter} iters)...")

    t0 = time.time()
    result = differential_evolution(
        task_aware_objective,
        bounds=bounds,
        args=(opt_names, pick_states, START_POS, sigma_bar_inv),
        maxiter=maxiter,
        seed=seed,
        tol=1e-6,
        polish=True,
        disp=verbose,
    )
    opt_time = time.time() - t0

    opt_positions = result.x.reshape(-1, 2)
    opt_obj = task_aware_objective(
        result.x, opt_names, pick_states, START_POS, sigma_bar_inv)
    opt_slack = -opt_obj

    if verbose:
        print(f"\nOptimized worst-case slack: {opt_slack:.2f}")
        print(f"Improvement: {opt_slack - init_slack:.2f} ({opt_slack/init_slack:.1f}x)")
        print(f"Time: {opt_time:.1f}s")
        print(f"\nOptimized positions:")
        for i, name in enumerate(opt_names):
            pos = opt_positions[i]
            old = init_opt[i]
            print(f"  {name}: ({old[0]:.3f}, {old[1]:.3f}) -> ({pos[0]:.3f}, {pos[1]:.3f})")

    # Build full position map
    full_positions = object_positions.copy()
    for i, idx in enumerate(opt_indices):
        full_positions[idx] = opt_positions[i]

    return {
        "task_name": task_config["name"],
        "scene_name": scene_name,
        "object_names": object_names,
        "initial_positions": object_positions,
        "optimized_positions": full_positions,
        "opt_object_names": opt_names,
        "opt_initial": init_opt,
        "opt_final": opt_positions,
        "initial_slack": init_slack,
        "optimized_slack": opt_slack,
        "pick_states": pick_states,
        "opt_time": opt_time,
    }


def update_scene_xml(scene_xml_path, output_path, object_names,
                     positions, scene_cfg):
    """Create optimized scene XML with new object positions."""
    with open(scene_xml_path) as f:
        xml = f.read()

    import re
    for obj_cfg in scene_cfg["objects"]:
        sn = obj_cfg["short_name"]
        if sn not in object_names:
            continue
        idx = object_names.index(sn)
        new_x, new_y = positions[idx]

        # Get original z from the XML
        mname = obj_cfg["mujoco_name"]
        match = re.search(
            rf'(<body name="{re.escape(mname)}" pos=")([^"]+)(")', xml)
        if match:
            old_pos = match.group(2).split()
            old_z = old_pos[2]
            new_pos = f"{new_x:.3f} {new_y:.3f} {old_z}"
            xml = xml[:match.start(2)] + new_pos + xml[match.end(2):]

    # Update keyframe if present
    # This is tricky — just write the XML and note that keyframe needs manual update
    with open(output_path, 'w') as f:
        f.write(xml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_configs", nargs="+",
                        help="Task YAML config files")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Dir to save optimized scene XMLs")
    args = parser.parse_args()

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_results = []

    for config_path in args.task_configs:
        if not os.path.isabs(config_path):
            config_path = os.path.join(pkg_root, config_path)

        print("\n" + "=" * 60)
        result = optimize_for_task(config_path, maxiter=args.maxiter,
                                    seed=args.seed, verbose=True)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Task':<25} {'Init Slack':>12} {'Opt Slack':>12} {'Improve':>10} {'Time':>8}")
    print("-" * 70)
    for r in all_results:
        improve = r["optimized_slack"] / r["initial_slack"]
        print(f"  {r['task_name']:<23} {r['initial_slack']:>12.1f} "
              f"{r['optimized_slack']:>12.1f} {improve:>9.1f}x {r['opt_time']:>7.1f}s")

    # Save results
    out_path = os.path.join(pkg_root, "results", "task_aware_optimization.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    serializable = []
    for r in all_results:
        s = {k: v for k, v in r.items()
             if not isinstance(v, np.ndarray)}
        s["initial_positions"] = r["initial_positions"].tolist()
        s["optimized_positions"] = r["optimized_positions"].tolist()
        s["opt_initial"] = r["opt_initial"].tolist()
        s["opt_final"] = r["opt_final"].tolist()
        serializable.append(s)
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
