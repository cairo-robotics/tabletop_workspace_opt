#!/usr/bin/env python3
"""Threshold sweep evaluation for trajectory-margin workspace optimization.

Runs headless SA once per (environment, layout) with bypass_threshold=True
to record first-crossing steps for each threshold in TAU_SWEEP. Computes
per-threshold metrics offline from the recorded crossings.

Outputs:
  - results/sa_headless/threshold_sweep.json
  - Summary tables printed to stdout

Usage:
    python3 scripts/sweep_threshold_eval.py                    # full sweep
    python3 scripts/sweep_threshold_eval.py --envs cluttered   # one env
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
    run_headless_sa, generate_random_layouts, TAU_SWEEP,
)
from run_sa_all_orderings_headless import (
    ALL_TASKS, MAX_ORDERINGS, get_movable_and_fixed,
)
import yaml


def collect_tau_crossings(task_config_path, scene_name=None,
                          seed=42, max_orderings=MAX_ORDERINGS,
                          layout_override=None, sa_kwargs=None):
    """Run all orderings and collect tau_crossings from each pick step.

    Returns a list of dicts, one per pick step with:
      - target_id, argmax_correct, n_steps, stalled
      - tau_crossings dict {tau: (step, goal) or None}
    """
    with open(task_config_path) as f:
        task_config = yaml.safe_load(f)["task"]

    pick_objects = task_config.get("pick_objects", [])
    if not pick_objects:
        return []

    all_perms = list(permutations(pick_objects))
    n_total = len(all_perms)
    if n_total <= max_orderings:
        perms = all_perms
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=max_orderings, replace=False)
        perms = [all_perms[i] for i in indices]

    picks = []
    for perm in perms:
        seq_str = ",".join(perm)
        r = run_headless_sa(
            task_config_path, scene_name=scene_name,
            user_sequence=seq_str, seed=seed,
            layout_override=layout_override, quiet=True,
            bypass_threshold=True, **sa_kwargs)

        for st in r.get("step_times", []):
            # Only disambiguation pick steps (skip trivial single-goal)
            if st["action"] != "pick" or st["inference_steps"] <= 1:
                continue
            picks.append({
                "target_id": st["target_id"],
                "argmax_correct": st["argmax_correct"],
                "n_steps": st["inference_steps"],
                "stalled": st.get("stalled", False),
                "tau_crossings": st["tau_crossings"],
            })

    return picks


def compute_metrics_at_tau(picks, tau, control_rate=20.0):
    """Compute threshold-dependent metrics from recorded crossings.

    Returns dict with:
      - threshold_accuracy: fraction where first crossing was on correct goal
      - convergence_rate: fraction where any goal crossed tau before stall
      - mean_time_to_convergence: over converged picks only (seconds)
      - premature_wrong_rate: fraction where first crossing was on wrong goal
      - not_converged_rate: fraction that never crossed tau
    """
    if not picks:
        return None

    n_correct = 0        # crossed on correct goal
    n_wrong = 0          # crossed on wrong goal (premature wrong commit)
    n_no_cross = 0       # never crossed the threshold
    converged_times = []

    tau_str = str(tau)
    for p in picks:
        c = p["tau_crossings"].get(tau_str)
        if c is None:
            n_no_cross += 1
            continue
        step, goal = c[0], c[1]
        time_s = step / control_rate
        if goal == p["target_id"]:
            n_correct += 1
            converged_times.append(time_s)
        else:
            n_wrong += 1
            converged_times.append(time_s)

    n = len(picks)
    n_converged = n_correct + n_wrong
    return {
        "n_picks": n,
        "threshold_accuracy": n_correct / n,
        "convergence_rate": n_converged / n,
        "premature_wrong_rate": n_wrong / n,
        "not_converged_rate": n_no_cross / n,
        "mean_time_to_convergence_s": (
            float(np.mean(converged_times)) if converged_times else 0.0),
        "std_time_to_convergence_s": (
            float(np.std(converged_times)) if converged_times else 0.0),
    }


def run_environment(env_task, env_scene, env_tier, n_random, seed,
                    max_orderings, sa_kwargs):
    """Run sweep for one environment: random baselines + optimized."""
    task_path = os.path.join(PROJECT_ROOT, env_task)
    opt_scene = f"{env_scene}_trajectory_optimized"
    opt_xml = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{opt_scene}.xml")

    results = {}

    # --- Random baselines: 30 layouts, aggregate all picks together ---
    if os.path.exists(os.path.join(
            PROJECT_ROOT, "src", "assets", f"{env_scene}.xml")):
        print(f"\n  Random baselines ({n_random} layouts)...")
        movable, fixed = get_movable_and_fixed(task_path, env_scene)
        layouts = generate_random_layouts(n_random, movable, fixed, seed=seed)
        if len(layouts) < n_random:
            print(f"    WARNING: {len(layouts)}/{n_random} layouts generated")

        random_picks = []
        for i, layout in enumerate(layouts):
            picks = collect_tau_crossings(
                task_path, scene_name=env_scene, seed=seed + i,
                max_orderings=max_orderings,
                layout_override=layout, sa_kwargs=sa_kwargs)
            random_picks.extend(picks)
            n_correct = sum(1 for p in picks if p["argmax_correct"])
            print(f"    layout {i+1}/{len(layouts)}: "
                  f"{len(picks)} picks, "
                  f"argmax {n_correct}/{len(picks)}")

        argmax_acc = (sum(1 for p in random_picks if p["argmax_correct"]) /
                      max(len(random_picks), 1))
        print(f"  Random total: {len(random_picks)} picks, "
              f"argmax acc {argmax_acc:.1%}")

        results["random"] = {
            "n_layouts": len(layouts),
            "n_picks": len(random_picks),
            "argmax_accuracy": argmax_acc,
            "by_threshold": {
                str(tau): compute_metrics_at_tau(random_picks, tau)
                for tau in TAU_SWEEP
            },
        }

    # --- Optimized ---
    if os.path.exists(opt_xml):
        print(f"\n  Optimized ({opt_scene})...")
        picks = collect_tau_crossings(
            task_path, scene_name=opt_scene, seed=seed,
            max_orderings=max_orderings, sa_kwargs=sa_kwargs)
        n_correct = sum(1 for p in picks if p["argmax_correct"])
        argmax_acc = n_correct / max(len(picks), 1)
        print(f"  Optimized: {len(picks)} picks, argmax acc {argmax_acc:.1%}")
        results["optimized"] = {
            "n_picks": len(picks),
            "argmax_accuracy": argmax_acc,
            "by_threshold": {
                str(tau): compute_metrics_at_tau(picks, tau)
                for tau in TAU_SWEEP
            },
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Threshold sweep for trajectory-margin evaluation")
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Subset of environment keys to run")
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    sa_kwargs = dict(
        inference_model="path_efficiency",
        beta=args.beta, noise=args.noise,
        threshold=0.5,  # irrelevant with bypass_threshold=True
        max_speed=0.05, control_rate=20.0, motion_2d=True,
    )

    selected_tasks = ALL_TASKS
    if args.envs:
        keep = set(args.envs)
        selected_tasks = [
            (t, s, tier) for (t, s, tier) in ALL_TASKS
            if tier in keep or s in keep or s.replace("scene_", "") in keep
        ]

    all_results = {
        "meta": {
            "beta": args.beta, "noise": args.noise, "n_random": args.n_random,
            "tau_sweep": TAU_SWEEP,
            "motion_2d": True, "stall_terminated": True,
            "max_speed": 0.05, "control_rate": 20.0,
            "seed": args.seed,
        },
        "environments": {},
    }

    for task, scene, tier in selected_tasks:
        print(f"\n{'='*70}")
        print(f"  {tier} — {scene}")
        print(f"{'='*70}")
        env_results = run_environment(
            task, scene, tier, args.n_random, args.seed,
            MAX_ORDERINGS, sa_kwargs)
        all_results["environments"][tier] = env_results

    # Save
    out = args.output or os.path.join(
        PROJECT_ROOT, "results", "sa_headless", "threshold_sweep.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")

    # Print summary tables
    print(f"\n{'='*95}")
    print(f"THRESHOLD SWEEP SUMMARY")
    print(f"beta={args.beta}, motion_2d=True, stall-term, n_random={args.n_random}")
    print(f"{'='*95}\n")

    # Argmax accuracy (threshold-independent)
    print("Argmax accuracy (threshold-free reference):")
    print(f"  {'Tier':>8}  {'random':>12}  {'optimized':>12}")
    for task, scene, tier in selected_tasks:
        env = all_results["environments"].get(tier, {})
        r = env.get("random", {}).get("argmax_accuracy")
        o = env.get("optimized", {}).get("argmax_accuracy")
        r_str = f"{r:.1%}" if r is not None else "—"
        o_str = f"{o:.1%}" if o is not None else "—"
        print(f"  {tier:>8}  {r_str:>12}  {o_str:>12}")

    # Per-threshold tables
    for layout_type in ["random", "optimized"]:
        print(f"\n{layout_type.upper()} — threshold accuracy (commit on correct goal):")
        header = f"  {'Tier':>8}"
        for tau in TAU_SWEEP:
            header += f"  {'τ='+str(tau):>9}"
        print(header)
        for task, scene, tier in selected_tasks:
            env = all_results["environments"].get(tier, {})
            data = env.get(layout_type, {}).get("by_threshold", {})
            row = f"  {tier:>8}"
            for tau in TAU_SWEEP:
                m = data.get(str(tau))
                if m:
                    row += f"  {m['threshold_accuracy']:>8.0%}"
                else:
                    row += f"  {'—':>9}"
            print(row)

        print(f"\n{layout_type.upper()} — mean time to convergence (s, over converged picks):")
        print(header)
        for task, scene, tier in selected_tasks:
            env = all_results["environments"].get(tier, {})
            data = env.get(layout_type, {}).get("by_threshold", {})
            row = f"  {tier:>8}"
            for tau in TAU_SWEEP:
                m = data.get(str(tau))
                if m and m["convergence_rate"] > 0:
                    row += f"  {m['mean_time_to_convergence_s']:>8.2f} "
                else:
                    row += f"  {'—':>9}"
            print(row)

        print(f"\n{layout_type.upper()} — convergence rate (any goal crossed τ):")
        print(header)
        for task, scene, tier in selected_tasks:
            env = all_results["environments"].get(tier, {})
            data = env.get(layout_type, {}).get("by_threshold", {})
            row = f"  {tier:>8}"
            for tau in TAU_SWEEP:
                m = data.get(str(tau))
                if m:
                    row += f"  {m['convergence_rate']:>8.0%}"
                else:
                    row += f"  {'—':>9}"
            print(row)

        print(f"\n{layout_type.upper()} — premature wrong-commit rate (crossed τ on wrong goal):")
        print(header)
        for task, scene, tier in selected_tasks:
            env = all_results["environments"].get(tier, {})
            data = env.get(layout_type, {}).get("by_threshold", {})
            row = f"  {tier:>8}"
            for tau in TAU_SWEEP:
                m = data.get(str(tau))
                if m:
                    row += f"  {m['premature_wrong_rate']:>8.0%}"
                else:
                    row += f"  {'—':>9}"
            print(row)


if __name__ == "__main__":
    main()
