#!/usr/bin/env python3
"""Evaluate trajectory-margin optimized layouts vs random baselines.

Uses headless shared autonomy with the Boltzmann path-efficiency
observer (the inference model matched to the trajectory-margin
optimizer).

For each environment:
  1. Generate N random valid layouts (non-overlapping, robot-excluded)
  2. Run all pick orderings on each random layout → random baseline stats
  3. Run all pick orderings on the trajectory-margin optimized layout
  4. Report mean ± std and speedup

Usage:
    python3 scripts/run_sa_all_orderings_headless.py --all
    python3 scripts/run_sa_all_orderings_headless.py --all --n-random 30
    python3 scripts/run_sa_all_orderings_headless.py config/tasks/desk_pick_and_return_sa.yaml --scene scene_desk
"""
import sys
import os
import json
import argparse
import numpy as np
from itertools import permutations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from run_sa_headless import (
    run_headless_sa, generate_random_layouts, apply_layout,
)
import yaml


MAX_ORDERINGS = 120  # sample if more than this many permutations


def run_all_orderings(task_config_path, scene_name=None, seed=42,
                      max_orderings=MAX_ORDERINGS, layout_override=None,
                      quiet=False, **kwargs):
    """Run SA for every permutation of pick objects (or sample if too many).

    Returns summary dict.
    """
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)["task"]

    pick_objects = task_config.get("pick_objects", [])
    if not pick_objects:
        print(f"  No pick_objects in task config, skipping")
        return None

    all_perms = list(permutations(pick_objects))
    n_total = len(all_perms)

    if n_total <= max_orderings:
        perms = all_perms
        sampled = False
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=max_orderings, replace=False)
        perms = [all_perms[i] for i in indices]
        sampled = True

    n_perms = len(perms)
    if not quiet:
        if sampled:
            print(f"  Objects: {pick_objects} -> {n_total} total, sampling {n_perms}")
        else:
            print(f"  Objects: {pick_objects} -> {n_perms} orderings (exhaustive)")

    all_pick_times = []        # time to threshold (converged picks only)
    all_accuracies = []        # threshold-based: correct if threshold crossed correctly
    all_argmax_correct = []    # argmax-based: correct if argmax == true at end
    all_max_prob_true = []     # max P(true) reached (for threshold sensitivity)
    all_totals = []
    per_ordering = []

    for i, perm in enumerate(perms):
        seq = ",".join(perm)
        r = run_headless_sa(
            task_config_path, scene_name=scene_name,
            user_sequence=seq, seed=seed,
            layout_override=layout_override, quiet=quiet,
            **kwargs)

        picks = [s for s in r.get("step_times", [])
                 if s["action"] == "pick" and s["inference_steps"] > 1]
        pick_times = [s["inference_time_s"] for s in picks]
        corrects = [s.get("correct", True) for s in picks]
        argmax_cs = [s.get("argmax_correct", s.get("correct", True))
                     for s in picks]
        max_probs = [s.get("max_prob_true", 0.0) for s in picks]

        all_pick_times.extend(pick_times)
        all_accuracies.extend(corrects)
        all_argmax_correct.extend(argmax_cs)
        all_max_prob_true.extend(max_probs)
        all_totals.append(r.get("total_inference_time", 0))

        per_ordering.append({
            "ordering": list(perm),
            "mean_pick": float(np.mean(pick_times)) if pick_times else 0,
            "accuracy": float(np.mean(corrects)) if corrects else 1.0,
            "argmax_accuracy": float(np.mean(argmax_cs)) if argmax_cs else 1.0,
            "total_time": r.get("total_inference_time", 0),
        })

    # Converged picks: threshold was crossed (not timed out)
    converged_times = [t for t, c in zip(all_pick_times, all_accuracies)
                       if c]  # correct threshold crossing
    # Also count wrong threshold crossings as converged
    n_converged = sum(1 for s in all_accuracies
                      if True)  # all get a time; converged = not timeout
    # Better: use the raw step data
    # converged = threshold crossed (correct or wrong); timed out = not

    summary = {
        "task": task_config["name"],
        "scene": scene_name or task_config.get("scene"),
        "n_orderings": n_perms,
        "n_total_orderings": n_total,
        "sampled": sampled,
        # Threshold-based metrics
        "mean_pick_time": float(np.mean(all_pick_times)) if all_pick_times else 0,
        "std_pick_time": float(np.std(all_pick_times)) if all_pick_times else 0,
        "accuracy": float(np.mean(all_accuracies)) if all_accuracies else 1.0,
        "n_wrong": sum(1 for a in all_accuracies if not a),
        # Argmax-based metrics (threshold-free)
        "argmax_accuracy": float(np.mean(all_argmax_correct)) if all_argmax_correct else 1.0,
        "n_argmax_wrong": sum(1 for a in all_argmax_correct if not a),
        # For threshold sensitivity
        "max_prob_true_values": [float(v) for v in all_max_prob_true],
        # Totals
        "mean_total_time": float(np.mean(all_totals)),
        "std_total_time": float(np.std(all_totals)),
        "per_ordering": per_ordering,
    }
    return summary


ALL_TASKS = [
    ("config/tasks/breakfast_easy_pick_and_return_sa.yaml",
     "scene_breakfast_easy", "Easy"),
    ("config/tasks/desk_pick_and_return_sa.yaml",
     "scene_desk", "Med-A"),
    ("config/tasks/breakfast_pick_and_return_sa.yaml",
     "scene_breakfast", "Med-B"),
    ("config/tasks/kitchen_pick_and_return_sa.yaml",
     "scene_kitchen_prep", "Med-C"),
    ("config/tasks/meal_pick_and_return_sa.yaml",
     "scene_meal_assembly", "Hard-A"),
    ("config/tasks/cluttered_pick_and_return_sa.yaml",
     "scene_cluttered", "Hard-B"),
]


def get_movable_and_fixed(task_config_path, scene_name):
    """Extract movable (pick) and fixed object names from task + scene."""
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)["task"]
    scene_yaml = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    with open(scene_yaml) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    movable = set(task_config.get("pick_objects", []))
    all_names = [o["short_name"] for o in scene_cfg["objects"]]

    # Fixed positions: read from MuJoCo XML keyframe
    import re, mujoco
    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    mujoco_map = {o["short_name"]: o["mujoco_name"]
                  for o in scene_cfg["objects"]}
    fixed_positions = {}
    for name in all_names:
        if name not in movable:
            mname = mujoco_map.get(name)
            if mname:
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, mname)
                pos = data.xpos[bid][:2].copy()
                fixed_positions[name] = pos

    return list(movable), fixed_positions


def run_random_baselines(task_config_path, scene_name, n_random, seed,
                         max_orderings, sa_kwargs):
    """Run SA on N random layouts, averaging over all orderings each.

    Returns dict with aggregate stats across all random layouts.
    """
    task_path = os.path.join(PROJECT_ROOT, task_config_path)
    movable, fixed = get_movable_and_fixed(task_path, scene_name)
    layouts = generate_random_layouts(n_random, movable, fixed, seed=seed)

    if len(layouts) < n_random:
        print(f"  WARNING: only generated {len(layouts)}/{n_random} valid layouts")

    per_layout_pick = []
    per_layout_acc = []
    per_layout_argmax_acc = []
    per_layout_total = []
    all_max_probs = []  # for threshold sensitivity

    for i, layout in enumerate(layouts):
        print(f"    Random layout {i+1}/{len(layouts)}...", end="", flush=True)
        s = run_all_orderings(
            task_path, scene_name=scene_name, seed=seed + i,
            max_orderings=max_orderings, layout_override=layout,
            quiet=True, **sa_kwargs)
        if s:
            per_layout_pick.append(s["mean_pick_time"])
            per_layout_acc.append(s["accuracy"])
            per_layout_argmax_acc.append(s["argmax_accuracy"])
            per_layout_total.append(s["mean_total_time"])
            all_max_probs.extend(s.get("max_prob_true_values", []))
            print(f" pick={s['mean_pick_time']:.2f}s, "
                  f"acc={s['accuracy']:.0%}, "
                  f"argmax={s['argmax_accuracy']:.0%}")
        else:
            print(" skipped")

    if not per_layout_pick:
        return None

    return {
        "n_layouts": len(per_layout_pick),
        "mean_pick_time": float(np.mean(per_layout_pick)),
        "std_pick_time": float(np.std(per_layout_pick)),
        "mean_accuracy": float(np.mean(per_layout_acc)),
        "std_accuracy": float(np.std(per_layout_acc)),
        "mean_argmax_accuracy": float(np.mean(per_layout_argmax_acc)),
        "std_argmax_accuracy": float(np.std(per_layout_argmax_acc)),
        "mean_total_time": float(np.mean(per_layout_total)),
        "std_total_time": float(np.std(per_layout_total)),
        "per_layout_pick": per_layout_pick,
        "per_layout_acc": per_layout_acc,
        "per_layout_argmax_acc": per_layout_argmax_acc,
        "max_prob_true_values": all_max_probs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate optimized vs random layouts with Boltzmann observer")
    parser.add_argument("task_config", nargs="?", default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Run all environments")
    parser.add_argument("--n-random", type=int, default=30,
                        help="Number of random baseline layouts (default: 30)")
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sa_kwargs = dict(
        inference_model="path_efficiency",
        beta=args.beta, noise=args.noise, threshold=args.threshold,
        max_speed=0.05, control_rate=20.0,
        motion_2d=True,
    )

    if args.all:
        all_results = {}

        for task, bl_scene, tier in ALL_TASKS:
            task_path = os.path.join(PROJECT_ROOT, task)
            opt_scene = f"{bl_scene}_trajectory_optimized"
            opt_xml = os.path.join(
                PROJECT_ROOT, "src", "assets", f"{opt_scene}.xml")
            if not os.path.exists(opt_xml):
                print(f"\n  SKIP {tier}: {opt_scene}.xml not found")
                continue

            print(f"\n{'='*70}")
            print(f"  {tier} — {bl_scene}")
            print(f"  Inference: path_efficiency, beta={args.beta}, "
                  f"threshold={args.threshold}")
            print(f"{'='*70}")

            # --- Random baselines ---
            print(f"\n  Random baselines ({args.n_random} layouts):")
            rnd = run_random_baselines(
                task, bl_scene, n_random=args.n_random, seed=args.seed,
                max_orderings=MAX_ORDERINGS, sa_kwargs=sa_kwargs)
            if rnd:
                all_results[f"{tier}_random"] = rnd
                print(f"  -> mean pick: {rnd['mean_pick_time']:.2f}s "
                      f"± {rnd['std_pick_time']:.2f}, "
                      f"acc: {rnd['mean_accuracy']:.1%} "
                      f"± {rnd['std_accuracy']:.1%}")

            # --- Trajectory-margin optimized ---
            print(f"\n  Trajectory-margin optimized ({opt_scene}):")
            opt = run_all_orderings(
                task_path, scene_name=opt_scene, seed=args.seed,
                **sa_kwargs)
            if opt:
                all_results[f"{tier}_optimized"] = opt
                print(f"  -> mean pick: {opt['mean_pick_time']:.2f}s "
                      f"± {opt['std_pick_time']:.2f}, "
                      f"acc: {opt['accuracy']:.1%}")

        # Save
        out_dir = os.path.join(PROJECT_ROOT, "results", "sa_headless")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "random_vs_optimized.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {out_path}")

        # Summary table
        print(f"\n{'='*115}")
        print(f"RANDOM BASELINES vs TRAJECTORY-MARGIN OPTIMIZED")
        print(f"Inference: Boltzmann path-efficiency "
              f"(beta={args.beta}, threshold={args.threshold})")
        print(f"{'='*115}")
        print(f"{'Tier':>7} {'Layout':>12} {'Converge Time':>16} "
              f"{'Thresh Acc':>12} {'Argmax Acc':>12} "
              f"{'Total':>14} {'Speedup':>8}")
        print("-" * 115)

        for task, bl_scene, tier in ALL_TASKS:
            rnd = all_results.get(f"{tier}_random")
            opt = all_results.get(f"{tier}_optimized")
            if not rnd or not opt:
                continue

            sp = rnd["mean_pick_time"] / max(opt["mean_pick_time"], 0.01)

            # Random
            rnd_argmax = rnd.get("mean_argmax_accuracy",
                                 rnd.get("mean_accuracy", 0))
            rnd_argmax_std = rnd.get("std_argmax_accuracy", 0)
            print(f"{tier:>7} {'random':>12} "
                  f"{rnd['mean_pick_time']:>6.1f}s ± {rnd['std_pick_time']:>4.1f} "
                  f"{rnd['mean_accuracy']:>6.0%} ± {rnd['std_accuracy']:>4.0%} "
                  f"{rnd_argmax:>6.0%} ± {rnd_argmax_std:>4.0%} "
                  f"{rnd['mean_total_time']:>6.0f}s ± {rnd['std_total_time']:>4.0f} ")

            # Optimized
            opt_argmax = opt.get("argmax_accuracy", opt.get("accuracy", 0))
            print(f"{'':>7} {'optimized':>12} "
                  f"{opt['mean_pick_time']:>6.1f}s ± {opt['std_pick_time']:>4.1f} "
                  f"{opt['accuracy']:>6.0%}{'':>8} "
                  f"{opt_argmax:>6.0%}{'':>8} "
                  f"{opt['mean_total_time']:>6.0f}s ± {opt['std_total_time']:>4.0f} "
                  f"{sp:>7.1f}x")
            print()

        # Threshold sensitivity from recorded max_prob_true
        print(f"\n{'='*80}")
        print(f"THRESHOLD SENSITIVITY (convergence rate at different thresholds)")
        print(f"{'='*80}")
        thresholds = [0.8, 0.85, 0.9, 0.95]
        header = f"{'Tier':>7} {'Layout':>12}"
        for t in thresholds:
            header += f" {'τ='+str(t):>8}"
        print(header)
        print("-" * 80)

        for task, bl_scene, tier in ALL_TASKS:
            for label, key in [("random", f"{tier}_random"),
                               ("optimized", f"{tier}_optimized")]:
                r = all_results.get(key)
                if not r:
                    continue
                max_probs = r.get("max_prob_true_values", [])
                if not max_probs:
                    continue
                row = f"{tier:>7} {label:>12}"
                for t in thresholds:
                    rate = np.mean([p >= t for p in max_probs])
                    row += f" {rate:>7.0%}"
                print(row)
            print()

    else:
        if not args.task_config:
            parser.error("Provide a task config or use --all")
        task_path = args.task_config
        if not os.path.isabs(task_path):
            task_path = os.path.join(PROJECT_ROOT, task_path)
        s = run_all_orderings(task_path, scene_name=args.scene,
                             seed=args.seed, **sa_kwargs)
        if s:
            print(f"\nMean pick time: {s['mean_pick_time']:.2f}s "
                  f"± {s['std_pick_time']:.2f}")
            print(f"Accuracy: {s['accuracy']:.1%} ({s['n_wrong']} wrong)")
            print(f"Mean total time: {s['mean_total_time']:.1f}s "
                  f"± {s['std_total_time']:.1f}")
            print(f"Orderings tested: {s['n_orderings']}")


if __name__ == "__main__":
    main()
