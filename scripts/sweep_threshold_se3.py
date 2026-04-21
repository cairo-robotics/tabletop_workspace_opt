#!/usr/bin/env python3
"""SE(3) threshold sweep evaluation.

Runs headless SA with bypass_threshold=True to record first-crossing
steps for each threshold in TAU_SWEEP. Computes per-threshold metrics
for ME-best, DE-best, and N random layouts.

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/sweep_threshold_se3.py
    PYTHONUNBUFFERED=1 python3 scripts/sweep_threshold_se3.py --n-random 5  # quick
"""
import argparse
import json
import os
import sys
from itertools import permutations

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from run_sa_headless import run_headless_sa_se3, generate_random_layouts, TAU_SWEEP

GRASP_LIB_PATH = os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml")
MAX_ORDERINGS = 120

SCENE_TIERS = [
    ("config/tasks/breakfast_easy_pick_and_return_sa.yaml",
     "scene_breakfast_easy", "Easy",
     ["cereal", "banana"]),
    ("config/tasks/desk_pick_and_return_sa.yaml",
     "scene_desk", "Med-A",
     ["mug", "stapler", "pen_cup"]),
    ("config/tasks/breakfast_pick_and_return_sa.yaml",
     "scene_breakfast", "Med-B",
     ["cereal", "banana", "milk_carton"]),
    ("config/tasks/kitchen_pick_and_return_sa.yaml",
     "scene_kitchen_prep", "Med-C",
     ["apple", "can", "bottle"]),
    ("config/tasks/meal_pick_and_return_sa.yaml",
     "scene_meal_assembly", "Hard-A",
     ["cereal", "banana", "apple", "can", "bottle"]),
    ("config/tasks/cluttered_pick_and_return_sa.yaml",
     "scene_cluttered", "Hard-B",
     ["red_block", "blue_block", "green_cylinder", "yellow_block",
      "orange_cylinder", "purple_block", "white_cylinder", "pink_block"]),
]


def collect_tau_crossings(task_path, scene_name, layout_xy, layout_yaws,
                          sa_kwargs, seed=42):
    """Run all orderings with bypass_threshold and collect tau_crossings."""
    with open(task_path) as f:
        task_config = yaml.safe_load(f)["task"]

    pick_objects = task_config.get("pick_objects", [])
    all_perms = list(permutations(pick_objects))
    if len(all_perms) > MAX_ORDERINGS:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(all_perms), size=MAX_ORDERINGS, replace=False)
        perms = [all_perms[i] for i in indices]
    else:
        perms = all_perms

    picks = []
    for perm in perms:
        seq = ",".join(perm)
        r = run_headless_sa_se3(
            task_path, scene_name=scene_name,
            intent_mode="se3-grasp",
            grasp_library_path=GRASP_LIB_PATH,
            user_sequence=seq, seed=seed,
            layout_override=layout_xy,
            layout_yaws=layout_yaws,
            bypass_threshold=True, quiet=True,
            **sa_kwargs)

        for st in r.get("step_times", []):
            if st["action"] != "pick" or st["inference_steps"] <= 1:
                continue
            picks.append({
                "target_id": st["target_id"],
                "argmax_correct": st.get("argmax_correct", False),
                "tau_crossings": st.get("tau_crossings", {}),
            })

    return picks


def compute_metrics_at_tau(picks, tau, control_rate=20.0):
    """Compute per-threshold metrics from recorded crossings."""
    if not picks:
        return None
    tau_str = str(tau)
    n_correct = 0
    n_wrong = 0
    n_no_cross = 0
    converged_times = []

    for p in picks:
        c = p["tau_crossings"].get(tau_str)
        if c is None:
            n_no_cross += 1
            continue
        step, goal = c[0], c[1]
        time_s = step / control_rate
        if goal == p["target_id"]:
            n_correct += 1
        else:
            n_wrong += 1
        converged_times.append(time_s)

    n = len(picks)
    return {
        "threshold_accuracy": n_correct / n,
        "convergence_rate": (n_correct + n_wrong) / n,
        "premature_wrong_rate": n_wrong / n,
        "mean_time_s": float(np.mean(converged_times)) if converged_times else 0,
    }


def load_layout_from_yaml(yaml_path):
    """Return (layout_xy, yaw_map) from a scene YAML."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)["scene"]
    layout = {}
    yaws = {}
    for o in cfg["objects"]:
        if "position" in o:
            pos = o["position"]
            layout[o["short_name"]] = np.array(
                [float(pos[0]), float(pos[1])])
            yaws[o["short_name"]] = float(o.get("yaw", 0.0))
    return layout, yaws


def load_scene_fixed(scene_name, task_objects):
    """Get fixed object positions for random layout generation."""
    import mujoco
    yaml_path = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)["scene"]
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    mujoco_names = {o["short_name"]: o["mujoco_name"] for o in cfg["objects"]}
    fixed = {}
    for o in cfg["objects"]:
        if o["short_name"] not in task_objects:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o["mujoco_name"])
            if bid >= 0:
                fixed[o["short_name"]] = m.body(o["mujoco_name"]).id
                fixed[o["short_name"]] = d.xpos[bid, :2].copy()
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-random", type=int, default=10)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--lambda-R", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sa_kwargs = dict(
        inference_model="path_efficiency",
        beta=args.beta, noise=args.noise, threshold=0.5,
        lambda_R=args.lambda_R, max_speed=0.05, control_rate=20.0,
    )

    all_results = {"meta": {"tau_sweep": TAU_SWEEP, "n_random": args.n_random},
                   "tiers": {}}

    for task_rel, scene_name, tier, task_objects in SCENE_TIERS:
        task_path = os.path.join(PROJECT_ROOT, task_rel)
        print(f"\n{'='*70}\n  {tier} — {scene_name}\n{'='*70}")

        tier_data = {}

        # ME-best
        me_yaml = os.path.join(
            PROJECT_ROOT, "config", "scenes",
            f"{scene_name}_se3_me_optimized.yaml")
        if os.path.exists(me_yaml):
            layout, yaws = load_layout_from_yaml(me_yaml)
            print(f"  ME-best...", end="", flush=True)
            picks = collect_tau_crossings(
                task_path, scene_name, layout, yaws, sa_kwargs, seed=args.seed)
            tier_data["ME"] = {
                str(tau): compute_metrics_at_tau(picks, tau)
                for tau in TAU_SWEEP}
            tier_data["ME"]["n_picks"] = len(picks)
            argmax = sum(1 for p in picks if p["argmax_correct"]) / max(len(picks), 1)
            print(f" {len(picks)} picks, argmax={argmax:.0%}")

        # DE-best
        de_yaml = os.path.join(
            PROJECT_ROOT, "config", "scenes",
            f"{scene_name}_se3_optimized.yaml")
        if os.path.exists(de_yaml):
            layout, yaws = load_layout_from_yaml(de_yaml)
            print(f"  DE-best...", end="", flush=True)
            picks = collect_tau_crossings(
                task_path, scene_name, layout, yaws, sa_kwargs, seed=args.seed)
            tier_data["DE"] = {
                str(tau): compute_metrics_at_tau(picks, tau)
                for tau in TAU_SWEEP}
            tier_data["DE"]["n_picks"] = len(picks)
            argmax = sum(1 for p in picks if p["argmax_correct"]) / max(len(picks), 1)
            print(f" {len(picks)} picks, argmax={argmax:.0%}")

        # Random
        fixed = load_scene_fixed(scene_name, task_objects)
        layouts = generate_random_layouts(
            args.n_random, task_objects, fixed, seed=args.seed)
        print(f"  Random ({len(layouts)} layouts)...")
        all_random_picks = []
        rng = np.random.default_rng(args.seed + 2000)
        for i, layout_2d in enumerate(layouts):
            yaw_map = {n: float(rng.uniform(-np.pi, np.pi))
                       for n in task_objects}
            picks = collect_tau_crossings(
                task_path, scene_name, layout_2d, yaw_map,
                sa_kwargs, seed=args.seed + i)
            all_random_picks.extend(picks)
            print(f"    {i+1}/{len(layouts)}: {len(picks)} picks")

        tier_data["Random"] = {
            str(tau): compute_metrics_at_tau(all_random_picks, tau)
            for tau in TAU_SWEEP}
        tier_data["Random"]["n_picks"] = len(all_random_picks)
        argmax = sum(1 for p in all_random_picks if p["argmax_correct"]) / max(len(all_random_picks), 1)
        print(f"  Random total: {len(all_random_picks)} picks, argmax={argmax:.0%}")

        all_results["tiers"][tier] = tier_data

    # Save
    out = os.path.join(
        PROJECT_ROOT, "results", "sa_headless", "se3_threshold_sweep.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Summary tables
    for layout_type in ["ME", "DE", "Random"]:
        print(f"\n{'='*90}")
        print(f"{layout_type} — Threshold accuracy")
        print(f"{'='*90}")
        header = f"  {'Tier':>7}"
        for tau in TAU_SWEEP:
            header += f"  {'t='+str(tau):>8}"
        print(header)
        for _, _, tier, _ in SCENE_TIERS:
            td = all_results["tiers"].get(tier, {}).get(layout_type, {})
            row = f"  {tier:>7}"
            for tau in TAU_SWEEP:
                m = td.get(str(tau))
                if m:
                    row += f"  {m['threshold_accuracy']:>7.0%}"
                else:
                    row += f"  {'—':>8}"
            print(row)


if __name__ == "__main__":
    main()
