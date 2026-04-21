#!/usr/bin/env python3
"""Compare time-to-goal-prediction between random and SE(3)-optimized layouts.

For each scene tier:
  1. Load the SE(3)-optimized layout from
     config/scenes/scene_*_se3_optimized.yaml.
  2. Run the headless shared-autonomy evaluator on (a) N random layouts
     and (b) the optimized layout, using teleport via layout_override.
  3. Aggregate per-pick metrics: time-to-commit, threshold accuracy,
     argmax accuracy.

Outputs:
    results/sa_headless/se3_random_vs_optimized.json
    plus a printed summary table.

Note: this uses the *same* SA evaluator as the trajectory-margin
experiments (path-efficiency Boltzmann observer with motion_2d=True
and stall termination). The SE(3) optimizer's slack objective is for
the 3D direction observer, but the evaluation harness is shared so
we can compare time-to-goal-prediction on the same scale.
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

from run_sa_headless import run_headless_sa, generate_random_layouts
from run_sa_all_orderings_headless import (
    run_all_orderings, get_movable_and_fixed, ALL_TASKS, MAX_ORDERINGS,
)


def load_se3_optimized_layout(scene_name: str):
    """Return {short_name: np.array([x, y])} for the optimized layout, or None.

    The SA evaluator only takes (x, y); z is preserved from the base
    scene XML during teleport.
    """
    path = os.path.join(
        PROJECT_ROOT, "config", "scenes",
        f"{scene_name}_se3_optimized.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cfg = yaml.safe_load(f)["scene"]
    layout = {}
    for o in cfg["objects"]:
        if "position" in o:
            pos = o["position"]
            layout[o["short_name"]] = np.array(
                [float(pos[0]), float(pos[1])])
    return layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sa_kwargs = dict(
        inference_model="path_efficiency",
        beta=args.beta, noise=args.noise, threshold=args.threshold,
        max_speed=0.05, control_rate=20.0, motion_2d=True,
    )

    all_results = {}

    for task_path, base_scene, tier in ALL_TASKS:
        full_task = os.path.join(PROJECT_ROOT, task_path)
        opt_layout = load_se3_optimized_layout(base_scene)
        if opt_layout is None:
            print(f"\n  SKIP {tier}: no SE(3)-optimized YAML for {base_scene}")
            continue

        print(f"\n{'='*70}")
        print(f"  {tier} — {base_scene}")
        print(f"{'='*70}")

        # --- Random baselines ---
        movable, fixed = get_movable_and_fixed(full_task, base_scene)
        layouts = generate_random_layouts(
            args.n_random, movable, fixed, seed=args.seed)
        print(f"  Random baselines ({len(layouts)} layouts)...")

        per_layout_pick = []
        per_layout_acc = []
        per_layout_argmax = []
        per_layout_total = []
        for i, layout in enumerate(layouts):
            s = run_all_orderings(
                full_task, scene_name=base_scene, seed=args.seed + i,
                max_orderings=MAX_ORDERINGS, layout_override=layout,
                quiet=True, **sa_kwargs)
            if s:
                per_layout_pick.append(s["mean_pick_time"])
                per_layout_acc.append(s["accuracy"])
                per_layout_argmax.append(s.get("argmax_accuracy", 0.0))
                per_layout_total.append(s["mean_total_time"])
                print(f"    {i+1}/{len(layouts)}: pick={s['mean_pick_time']:.2f}s "
                      f"acc={s['accuracy']:.0%} argmax={s.get('argmax_accuracy', 0):.0%}")

        rnd = {
            "n_layouts": len(per_layout_pick),
            "mean_pick_time": float(np.mean(per_layout_pick)) if per_layout_pick else 0,
            "std_pick_time": float(np.std(per_layout_pick)) if per_layout_pick else 0,
            "mean_accuracy": float(np.mean(per_layout_acc)) if per_layout_acc else 0,
            "std_accuracy": float(np.std(per_layout_acc)) if per_layout_acc else 0,
            "mean_argmax_accuracy": float(np.mean(per_layout_argmax)) if per_layout_argmax else 0,
            "std_argmax_accuracy": float(np.std(per_layout_argmax)) if per_layout_argmax else 0,
            "mean_total_time": float(np.mean(per_layout_total)) if per_layout_total else 0,
        }
        all_results[f"{tier}_random"] = rnd

        # --- SE(3) optimized layout (via teleport) ---
        print(f"  SE(3)-optimized layout...")
        opt = run_all_orderings(
            full_task, scene_name=base_scene, seed=args.seed,
            max_orderings=MAX_ORDERINGS, layout_override=opt_layout,
            quiet=True, **sa_kwargs)
        if opt:
            print(f"    pick={opt['mean_pick_time']:.2f}s "
                  f"acc={opt['accuracy']:.0%} "
                  f"argmax={opt.get('argmax_accuracy', 0):.0%}")
            all_results[f"{tier}_optimized"] = {
                "mean_pick_time": opt["mean_pick_time"],
                "std_pick_time": opt["std_pick_time"],
                "accuracy": opt["accuracy"],
                "argmax_accuracy": opt.get("argmax_accuracy", 0),
                "mean_total_time": opt["mean_total_time"],
            }

    # Save
    out = os.path.join(
        PROJECT_ROOT, "results", "sa_headless", "se3_random_vs_optimized.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Print summary
    print(f"\n{'='*110}")
    print(f"SE(3) RANDOM vs OPTIMIZED — Boltzmann path-efficiency, "
          f"beta={args.beta}, threshold={args.threshold}")
    print(f"{'='*110}")
    print(f"  {'Tier':>7} {'Layout':>12} {'Pick Time (s)':>16} "
          f"{'Threshold Acc':>14} {'Argmax Acc':>14} {'Speedup':>9}")
    print("  " + "-" * 95)
    for task_path, base_scene, tier in ALL_TASKS:
        rnd = all_results.get(f"{tier}_random")
        opt = all_results.get(f"{tier}_optimized")
        if not rnd or not opt:
            continue
        sp = rnd["mean_pick_time"] / max(opt["mean_pick_time"], 0.01)
        print(f"  {tier:>7} {'random':>12} "
              f"{rnd['mean_pick_time']:>6.2f} ± {rnd['std_pick_time']:>5.2f} "
              f"{rnd['mean_accuracy']:>9.0%} ± {rnd['std_accuracy']:>3.0%} "
              f"{rnd['mean_argmax_accuracy']:>9.0%} ± {rnd['std_argmax_accuracy']:>3.0%} ")
        print(f"  {'':>7} {'optimized':>12} "
              f"{opt['mean_pick_time']:>6.2f} ± {opt['std_pick_time']:>5.2f} "
              f"{opt['accuracy']:>11.0%}{'':>5} "
              f"{opt['argmax_accuracy']:>11.0%}{'':>5} "
              f"{sp:>7.2f}x")


if __name__ == "__main__":
    main()
