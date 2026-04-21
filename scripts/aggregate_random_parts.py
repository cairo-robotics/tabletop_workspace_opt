#!/usr/bin/env python3
"""Aggregate per-worker random_part_{tier}_*.json files into the
canonical `se3_3d_random_vs_optimized.json`.

Each part file has raw per-layout results; we combine them across
workers for the same tier, then compute the aggregate `random_yaw`
entry using the same formula as `compare_se3_sa_3d.py`.

Existing `se3_optimized`, `me_optimized`, `random_yaw_optimized`
entries in the canonical JSON are preserved.
"""
import os
import sys
import json
import glob
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS_DIR = os.path.join("/tmp/random_parts")
CANONICAL = os.path.join(
    ROOT, "results", "sa_headless", "se3_3d_random_vs_optimized.json")


def agg(results_list):
    if not results_list:
        return None
    def feas(r):
        return r.get("n_feasible_picks", 0) > 0
    any_feas = any(feas(r) for r in results_list)
    return {
        "n_layouts": len(results_list),
        "mean_task_success_rate": float(np.mean(
            [r["task_success_rate"] for r in results_list])),
        "std_task_success_rate": float(np.std(
            [r["task_success_rate"] for r in results_list])),
        "mean_infeasible_pick_rate": float(np.mean(
            [r["infeasible_pick_rate"] for r in results_list])),
        "mean_pick_time": float(np.mean(
            [r["mean_pick_time"] for r in results_list if feas(r)]))
            if any_feas else 0,
        "std_pick_time": float(np.std(
            [r["mean_pick_time"] for r in results_list if feas(r)]))
            if any_feas else 0,
        "mean_threshold_accuracy": float(np.mean(
            [r["threshold_accuracy"] for r in results_list if feas(r)]))
            if any_feas else 0,
        "mean_argmax_accuracy": float(np.mean(
            [r["argmax_accuracy"] for r in results_list if feas(r)]))
            if any_feas else 0,
    }


def main():
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "random_part_*.json")))
    if not parts:
        print(f"No part files in {PARTS_DIR}")
        sys.exit(1)

    by_tier = defaultdict(list)   # tier -> list of (layout_idx, result)
    params_by_tier = {}
    for p in parts:
        with open(p) as f:
            d = json.load(f)
        tier = d["tier"]
        params_by_tier.setdefault(tier, d.get("params", {}))
        for idx, res in zip(d["layout_indices"], d["random_yaw_results"]):
            by_tier[tier].append((idx, res))

    # Deduplicate by layout index (later workers override earlier)
    dedup = {}
    for tier, pairs in by_tier.items():
        seen = {}
        for idx, res in pairs:
            seen[idx] = res
        dedup[tier] = [seen[k] for k in sorted(seen.keys())]
        print(f"{tier}: {len(dedup[tier])} unique layouts across "
              f"{len(pairs)} part entries")

    # Load canonical JSON, update random_yaw per tier, save back
    if os.path.exists(CANONICAL):
        canonical = json.load(open(CANONICAL))
    else:
        canonical = {}

    for tier, results in dedup.items():
        entry = canonical.get(tier, {})
        entry["random_yaw"] = agg(results)
        canonical[tier] = entry

    with open(CANONICAL, "w") as f:
        json.dump(canonical, f, indent=2, default=str)
    print(f"\nUpdated {CANONICAL}")
    print("Random params by tier:")
    for t, p in params_by_tier.items():
        print(f"  {t}: {p}")


if __name__ == "__main__":
    main()
