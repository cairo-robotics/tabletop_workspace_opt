#!/usr/bin/env python3
"""Generate condition-comparison CSV and plots from trial/block analysis exports."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLOCK_METRICS = [
    ("mean_teleop_time_sec", "Mean Teleop Time (s)"),
    ("mean_autonomous_time_sec", "Mean Autonomous Time (s)"),
    ("mean_teleop_distance_proportion", "Mean Teleop Distance Proportion"),
    ("mean_avg_teleop_entropy", "Mean Teleop Entropy"),
]

TRIAL_METRICS = [
    ("teleop_time_sec", "Teleop Time (s)"),
    ("autonomous_time_sec", "Autonomous Time (s)"),
    ("teleop_distance_proportion", "Teleop Distance Proportion"),
    ("avg_teleop_entropy", "Teleop Entropy"),
]


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def _safe_diff(a, b):
    if a is None or b is None:
        return ""
    return b - a


def _write_summary_csv(output_path, optimized_row, unoptimized_row):
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "optimized", "unoptimized", "diff_unoptimized_minus_optimized"],
        )
        writer.writeheader()
        for key, _label in BLOCK_METRICS:
            opt = _float_or_none(optimized_row.get(key))
            unopt = _float_or_none(unoptimized_row.get(key))
            writer.writerow(
                {
                    "metric": key,
                    "optimized": "" if opt is None else opt,
                    "unoptimized": "" if unopt is None else unopt,
                    "diff_unoptimized_minus_optimized": _safe_diff(opt, unopt),
                }
            )


def _plot_block_metrics(output_path, optimized_row, unoptimized_row):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for ax, (key, label) in zip(axes, BLOCK_METRICS):
        opt_value = _float_or_none(optimized_row.get(key)) or 0.0
        unopt_value = _float_or_none(unoptimized_row.get(key)) or 0.0
        ax.bar(["optimized", "unoptimized"], [opt_value, unopt_value], color=["#1f77b4", "#d62728"])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_trial_distributions(output_path, optimized_trials, unoptimized_trials):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()
    for ax, (key, label) in zip(axes, TRIAL_METRICS):
        opt_values = [_float_or_none(row.get(key)) for row in optimized_trials]
        opt_values = [value for value in opt_values if value is not None]
        unopt_values = [_float_or_none(row.get(key)) for row in unoptimized_trials]
        unopt_values = [value for value in unopt_values if value is not None]
        ax.boxplot(
            [opt_values, unopt_values],
            labels=["optimized", "unoptimized"],
            patch_artist=True,
            boxprops={"facecolor": "#cfe2f3"},
            medianprops={"color": "#1f1f1f"},
        )
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle("Condition Comparison: Trial-Level Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized-block", required=True, help="Path to optimized block analysis CSV")
    parser.add_argument("--unoptimized-block", required=True, help="Path to unoptimized block analysis CSV")
    parser.add_argument("--optimized-trial", required=True, help="Path to optimized trial analysis CSV")
    parser.add_argument("--unoptimized-trial", required=True, help="Path to unoptimized trial analysis CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for plots and summary CSV")
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    optimized_block_rows = _read_csv_rows(os.path.expanduser(args.optimized_block))
    unoptimized_block_rows = _read_csv_rows(os.path.expanduser(args.unoptimized_block))
    optimized_trial_rows = _read_csv_rows(os.path.expanduser(args.optimized_trial))
    unoptimized_trial_rows = _read_csv_rows(os.path.expanduser(args.unoptimized_trial))
    if not optimized_block_rows or not unoptimized_block_rows:
        raise RuntimeError("block analysis CSV is empty")

    optimized_block = optimized_block_rows[0]
    unoptimized_block = unoptimized_block_rows[0]

    _write_summary_csv(output_dir / "condition_comparison_summary.csv", optimized_block, unoptimized_block)
    _plot_block_metrics(output_dir / "condition_comparison_block_metrics.png", optimized_block, unoptimized_block)
    _plot_trial_distributions(
        output_dir / "condition_comparison_trial_distributions.png",
        optimized_trial_rows,
        unoptimized_trial_rows,
    )

    print("Wrote comparison outputs to {}".format(output_dir))


if __name__ == "__main__":
    main()
