#!/usr/bin/env python3
"""Compare grasp feasibility between random and SE(3)-optimized layouts.

For each scene tier:
  1. Sample N random layouts (positions in xy with the same constraints
     used elsewhere: min pairwise distance, robot exclusion, table bounds).
  2. Optimize yaws for each random layout via the same nested yaw GD
     the optimizer uses (so the random baseline isn't crippled by a
     bad yaw choice).
  3. Run check_layout_feasibility (with the new arm-vs-other-object
     collision check enabled) on every random layout and on the single
     SE(3)-optimized layout from config/scenes/scene_*_se3_optimized.yaml.
  4. Count feasibility verdicts and the breakdown of failure types.
  5. Print a comparison table.

Outputs:
    results/se3_audit/grasp_feasibility_comparison.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, SCRIPT_DIR)

from envopt.grasp_library import GraspLibrary
from envopt.grasp_feasibility import check_layout_feasibility
from envopt.reachability import ReachabilityOracle
from envopt.yaw_optimizer import optimize_yaw

# Reuse the same random sampler the headless SA evaluator uses
from run_sa_headless import generate_random_layouts


SCENE_TIERS = [
    {"name": "scene_breakfast_easy",
     "objects": ["cereal", "banana"]},
    {"name": "scene_breakfast",
     "objects": ["cereal", "banana", "milk_carton"]},
    {"name": "scene_desk",
     "objects": ["mug", "stapler", "pen_cup"]},
    {"name": "scene_kitchen_prep",
     "objects": ["apple", "can", "bottle"]},
    {"name": "scene_meal_assembly",
     "objects": ["cereal", "banana", "apple", "can", "bottle"]},
    {"name": "scene_cluttered",
     "objects": ["red_block", "blue_block", "green_cylinder",
                 "yellow_block", "orange_cylinder", "purple_block",
                 "white_cylinder", "pink_block"]},
]
START_POS_3D = np.array([0.452, 0.160, 1.05])


def load_scene_meta(scene_name: str):
    """Return (mujoco_names, half_extents_xy, z_map, fixed_xy)."""
    yaml_path = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)["scene"]
    mujoco_names = {o["short_name"]: o["mujoco_name"] for o in cfg["objects"]}
    half_extents_xy = {}
    for o in cfg["objects"]:
        he = o.get("half_extents", [0.03, 0.03, 0.03])
        half_extents_xy[o["short_name"]] = (he[0], he[1])

    # World z from MuJoCo XML for each free-jointed body
    import mujoco
    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    z_map = {}
    fixed_xy = {}
    for short, mname in mujoco_names.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mname)
        if bid < 0:
            continue
        z_map[short] = float(d.xpos[bid, 2])
        fixed_xy[short] = np.array([
            float(d.xpos[bid, 0]), float(d.xpos[bid, 1])])
    return mujoco_names, half_extents_xy, z_map, fixed_xy


def load_optimized_layout(scene_name: str, task_objects: List[str]):
    """Return (layout_xy3, yaw_map) from the SE(3)-optimized YAML, or None."""
    opt_path = os.path.join(
        PROJECT_ROOT, "config", "scenes",
        f"{scene_name}_se3_optimized.yaml")
    if not os.path.exists(opt_path):
        return None
    with open(opt_path) as f:
        cfg = yaml.safe_load(f)["scene"]
    layout_xy = {}
    yaw_map = {}
    for o in cfg["objects"]:
        short = o["short_name"]
        if "position" in o and short in task_objects:
            pos = np.asarray(o["position"], dtype=float)
            if len(pos) == 2:
                pos = np.array([pos[0], pos[1], 0.94])
            layout_xy[short] = pos
            yaw_map[short] = float(o.get("yaw", 0.0))
    return layout_xy, yaw_map


def categorize_failures(failures: List[str]) -> Dict[str, int]:
    """Bucket the failure strings emitted by check_layout_feasibility."""
    cats = Counter()
    for f in failures:
        if "pregrasp_ik" in f:
            cats["pregrasp_ik"] += 1
        elif "grasp_ik" in f:
            cats["grasp_ik"] += 1
        elif "approach_collide" in f:
            cats["approach_segment_2d"] += 1
        elif "arm_collide" in f:
            cats["arm_other_object_3d"] += 1
        else:
            cats["other"] += 1
    return dict(cats)


def evaluate_one_layout(layout_xy_dict, yaw_map, half_extents_xy, z_map,
                       grasp_lib, oracle, task_objects, mujoco_names):
    """Run check_layout_feasibility on a single layout. Returns the dict."""
    # Inject z values
    layout_with_z = {}
    for short, xy in layout_xy_dict.items():
        if len(xy) >= 3:
            layout_with_z[short] = np.asarray(xy, dtype=float)
        else:
            layout_with_z[short] = np.array(
                [xy[0], xy[1], z_map.get(short, 0.95)])
    return check_layout_feasibility(
        layout_with_z, yaw_map, half_extents_xy, grasp_lib, oracle,
        task_objects=task_objects,
        min_approach_clearance=0.005,
        min_object_separation=0.02,
        mujoco_names=mujoco_names,
        check_arm_collisions=True)


def evaluate_scene(scene_name, task_objects, n_random, seed, grasp_lib):
    print(f"\n--- {scene_name} ---")
    mujoco_names, half_extents_xy, z_map, fixed_xy = load_scene_meta(scene_name)
    base_xml = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    oracle = ReachabilityOracle(
        base_xml, pos_tol=0.02, rot_tol_deg=20.0, max_iters=30, n_seeds=4)

    # Fixed objects = those NOT in task_objects
    fixed_xy_for_random = {n: xy for n, xy in fixed_xy.items()
                           if n not in task_objects}

    layouts = generate_random_layouts(
        n_random, task_objects, fixed_xy_for_random, seed=seed)
    print(f"  Generated {len(layouts)}/{n_random} valid random layouts")

    random_results = []
    for i, layout_2d in enumerate(layouts):
        # Optimize yaw for this random layout (same as the optimizer does)
        layout_3d = {n: np.array([xy[0], xy[1], z_map.get(n, 0.95)])
                     for n, xy in layout_2d.items()}
        try:
            yaws, _ = optimize_yaw(
                layout_3d, task_objects, grasp_lib, START_POS_3D,
                n_steps=20, sigma_v=0.5, sigma_w=0.5, lambda_R=0.04)
            yaw_map = {n: float(y) for n, y in zip(task_objects, yaws)}
        except Exception:
            yaw_map = {n: 0.0 for n in task_objects}

        feas = evaluate_one_layout(
            layout_3d, yaw_map, half_extents_xy, z_map,
            grasp_lib, oracle, task_objects, mujoco_names)
        random_results.append(feas)

    # Optimized layout
    opt = load_optimized_layout(scene_name, task_objects)
    opt_result = None
    if opt is None:
        print(f"  WARNING: no SE(3)-optimized layout file for {scene_name}")
    else:
        layout_3d, yaw_map = opt
        opt_result = evaluate_one_layout(
            layout_3d, yaw_map, half_extents_xy, z_map,
            grasp_lib, oracle, task_objects, mujoco_names)

    return {
        "scene": scene_name,
        "task_objects": task_objects,
        "n_random_layouts": len(random_results),
        "random_results": random_results,
        "optimized_result": opt_result,
    }


def summarize(scene_record):
    """Aggregate per-scene metrics for the comparison table."""
    rand = scene_record["random_results"]
    opt = scene_record["optimized_result"]
    n_obj = len(scene_record["task_objects"])

    def per_layout_summary(results_list):
        n_total = len(results_list)
        if n_total == 0:
            return None
        feasible = sum(1 for r in results_list if r["feasible"])
        # Per-pick stats: how many picks would actually succeed if we
        # tried? Each layout has up to N picks; count how many failed.
        fail_cats = Counter()
        n_pick_attempts = 0
        n_pick_failures = 0
        for r in results_list:
            cats = categorize_failures(r["failures"])
            for k, v in cats.items():
                fail_cats[k] += v
            n_pick_attempts += n_obj
            n_pick_failures += len(r["failures"])
        return {
            "n_layouts": n_total,
            "n_feasible": feasible,
            "frac_feasible": feasible / n_total,
            "n_pick_attempts": n_pick_attempts,
            "n_pick_failures": n_pick_failures,
            "frac_picks_failing": n_pick_failures / max(n_pick_attempts, 1),
            "failure_breakdown": dict(fail_cats),
        }

    return {
        "scene": scene_record["scene"],
        "n_objects": n_obj,
        "random": per_layout_summary(rand),
        "optimized": per_layout_summary([opt] if opt else []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str,
                        default=os.path.join(
                            PROJECT_ROOT, "results", "se3_audit",
                            "grasp_feasibility_comparison.json"))
    args = parser.parse_args()

    grasp_lib = GraspLibrary.load(
        os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml"))

    all_records = []
    for tier in SCENE_TIERS:
        try:
            rec = evaluate_scene(
                tier["name"], tier["objects"], args.n_random, args.seed,
                grasp_lib)
            all_records.append(rec)
        except Exception as e:
            print(f"  ERROR in {tier['name']}: {e}")
            import traceback
            traceback.print_exc()

    summaries = [summarize(r) for r in all_records]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summaries": summaries,
                   "config": {"n_random": args.n_random, "seed": args.seed}},
                  f, indent=2, default=str)
    print(f"\nSaved: {args.out}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"GRASP FEASIBILITY: random vs SE(3)-optimized layouts")
    print(f"  arm-vs-other-object collision check: ENABLED")
    print(f"  n_random={args.n_random} per scene")
    print(f"{'='*100}")
    print(f"  {'scene':>22}  {'M':>3}  {'random feasible':>16}  "
          f"{'random pick OK':>16}  {'opt feas':>9}  {'opt pick OK':>12}")
    print("  " + "-" * 95)
    for s in summaries:
        rnd = s.get("random") or {}
        opt = s.get("optimized") or {}
        rand_layout_feas = (
            f"{rnd.get('n_feasible', 0)}/{rnd.get('n_layouts', 0)} "
            f"({rnd.get('frac_feasible', 0):.0%})"
            if rnd else "—")
        rand_pick_ok = (
            f"{(1 - rnd.get('frac_picks_failing', 0))*100:.0f}%"
            if rnd else "—")
        opt_feas = (
            f"{opt.get('n_feasible', 0)}/{opt.get('n_layouts', 0)}"
            if opt else "—")
        opt_pick_ok = (
            f"{(1 - opt.get('frac_picks_failing', 0))*100:.0f}%"
            if opt else "—")
        print(f"  {s['scene']:>22}  {s['n_objects']:>3}  "
              f"{rand_layout_feas:>16}  "
              f"{rand_pick_ok:>16}  "
              f"{opt_feas:>9}  "
              f"{opt_pick_ok:>12}")

    print(f"\n{'='*100}")
    print("FAILURE BREAKDOWN (sums over all picks across all layouts)")
    print(f"{'='*100}")
    for s in summaries:
        print(f"\n  {s['scene']} (M={s['n_objects']}):")
        for label, key in [("random", "random"), ("optimized", "optimized")]:
            d = s.get(key) or {}
            fb = d.get("failure_breakdown", {})
            n_attempts = d.get("n_pick_attempts", 0)
            print(f"    {label:>9}: {dict(fb)} of {n_attempts} pick attempts")


if __name__ == "__main__":
    main()
