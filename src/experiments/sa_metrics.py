"""Shared-autonomy experiment metric aggregation helpers."""
from __future__ import annotations

import numpy as np


def aggregate_layout_results(results_list):
    """Aggregate per-layout SA metrics for a random-layout arm."""
    if not results_list:
        return None

    def feasible(r):
        return r.get("n_feasible_picks", 0) > 0

    any_feasible = any(feasible(r) for r in results_list)
    return {
        "n_layouts": len(results_list),
        "n_orderings_total": int(sum(
            r["n_orderings"] for r in results_list)),
        "n_picks_total": int(sum(
            r["n_total_picks"] for r in results_list)),
        "mean_task_success_rate": float(np.mean(
            [r["task_success_rate"] for r in results_list])),
        "std_task_success_rate": float(np.std(
            [r["task_success_rate"] for r in results_list])),
        "mean_infeasible_pick_rate": float(np.mean(
            [r["infeasible_pick_rate"] for r in results_list])),
        "mean_pick_time": float(np.mean(
            [r["mean_pick_time"] for r in results_list if feasible(r)]))
            if any_feasible else 0,
        "std_pick_time": float(np.std(
            [r["mean_pick_time"] for r in results_list if feasible(r)]))
            if any_feasible else 0,
        "mean_threshold_accuracy": float(np.mean(
            [r["threshold_accuracy"] for r in results_list if feasible(r)]))
            if any_feasible else 0,
        "mean_argmax_accuracy": float(np.mean(
            [r["argmax_accuracy"] for r in results_list if feasible(r)]))
            if any_feasible else 0,
    }
