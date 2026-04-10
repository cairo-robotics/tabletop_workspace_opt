#!/usr/bin/env python3
"""Compare MAP-Elites results across all three optimization objectives.

Reads results JSONs from separability, legibility, and trajectory margin
runs and produces a cross-objective comparison table.

Usage:
    python3 scripts/compare_objectives.py
    python3 scripts/compare_objectives.py --output results/objective_comparison.json
"""
import sys
import os
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_all_results(source_name, filenames):
    """Load and merge results from multiple JSON files."""
    merged = {}
    for fname in filenames:
        path = os.path.join(PROJECT_ROOT, "results", fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            merged.update(data)
    return merged


SOURCES = {
    "separability": [
        "map_elites_v2_all_results.json",
        "map_elites_all_results.json",
    ],
    "legibility": [
        "map_elites_legibility_all.json",
        "map_elites_legibility_easy_med.json",
        "map_elites_legibility_hard.json",
    ],
    "trajectory_margin": [
        "map_elites_trajectory_margin_all.json",
        "map_elites_trajectory_margin_easy_med.json",
        "map_elites_trajectory_margin_hard.json",
    ],
}

ENV_ORDER = [
    ("breakfast_easy", "Easy"),
    ("desk", "Med-A"),
    ("breakfast", "Med-B"),
    ("kitchen_prep", "Med-C"),
    ("meal_assembly", "Hard-A"),
    ("cluttered", "Hard-B"),
]


def get_metrics(env_data):
    """Extract key metrics from an environment result."""
    if not env_data:
        return None
    me = env_data.get("map_elites", {})
    bl = env_data.get("baseline", {})
    ss = bl.get("single_step", {})
    return {
        "n_seeds": me.get("n_seeds", 0),
        "bl_accuracy": ss.get("accuracy", 0),
        "bl_steps": ss.get("mean_steps", 0),
        "bl_time_s": ss.get("mean_time_s", 0),
        "bl_slack": env_data.get("baseline", {}).get("separability_slack", 0),
        "me_accuracy": me.get("best_mean_accuracy", 0),
        "me_steps": me.get("best_mean_steps", 0),
        "me_time_s": me.get("best_mean_time_s", 0),
        "me_slack": me.get("best_mean_slack", 0),
        "speedup": env_data.get("speedup_single", 1.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load all results
    all_data = {}
    for source, fnames in SOURCES.items():
        data = load_all_results(source, fnames)
        if data:
            all_data[source] = data
            print(f"Loaded {source}: {list(data.keys())}")
        else:
            print(f"WARNING: no results for {source}")

    if not all_data:
        print("No results found. Run eval_map_elites_tiers.py first.")
        sys.exit(1)

    # Build comparison table
    comparison = {}
    for env_key, tier in ENV_ORDER:
        env_comp = {"tier": tier}
        for source in SOURCES:
            if source in all_data and env_key in all_data[source]:
                env_comp[source] = get_metrics(all_data[source][env_key])
        comparison[env_key] = env_comp

    # Print comparison table
    print(f"\n{'='*130}")
    print("CROSS-OBJECTIVE COMPARISON: MAP-Elites Best Elite")
    print(f"{'='*130}")

    # Header
    print(f"{'Tier':>7} {'Scene':>16}  ", end="")
    for source in ["separability", "legibility", "trajectory_margin"]:
        print(f"{'|':>2} {source:>18} {'Slack':>7} {'Spdup':>6}", end="")
    print()
    print("-" * 130)

    for env_key, tier in ENV_ORDER:
        comp = comparison.get(env_key, {})
        print(f"{tier:>7} {env_key:>16}  ", end="")
        for source in ["separability", "legibility", "trajectory_margin"]:
            m = comp.get(source)
            if m:
                print(f"| {m['me_time_s']:>6.2f}s/{m['me_accuracy']:.0%} "
                      f"{m['me_slack']:>7.1f} {m['speedup']:>5.1f}x", end="")
            else:
                print(f"|       —          —      —   ", end="")
        print()

    # Aggregate stats
    print(f"\n{'='*80}")
    print("AGGREGATE METRICS (mean across environments)")
    print(f"{'='*80}")
    for source in ["separability", "legibility", "trajectory_margin"]:
        if source not in all_data:
            continue
        slacks = []
        speedups = []
        accs = []
        for env_key, _ in ENV_ORDER:
            m = comparison.get(env_key, {}).get(source)
            if m:
                slacks.append(m["me_slack"])
                speedups.append(m["speedup"])
                accs.append(m["me_accuracy"])
        if slacks:
            print(f"  {source:>20}: mean_slack={np.mean(slacks):.1f}  "
                  f"mean_speedup={np.mean(speedups):.2f}x  "
                  f"mean_accuracy={np.mean(accs):.2%}  "
                  f"(n={len(slacks)} envs)")

    # Save
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(PROJECT_ROOT, "results",
                               "objective_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Make serializable
    def to_json(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=to_json)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
