#!/usr/bin/env python3
"""Regenerate result figures under docs/latex/figures/.

Produces:
  fig_argmax_accuracy.png/.pdf   - argmax accuracy bars (random vs ME)
  fig_task_and_time.png/.pdf     - task success + mean pick time (2 panels)
  fig_threshold_heatmaps.png/.pdf - (tier x tau) heatmaps for ME
  fig_pareto.png/.pdf            - accuracy vs time Pareto scatter from sweep
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.se3_catalog import SE3_TIER_LABELS

FIG_DIR = os.path.join(ROOT, "docs", "latex", "figures")
SWEEP = os.path.join(ROOT, "results", "sa_headless", "se3_threshold_sweep.json")
ME = os.path.join(ROOT, "results", "sa_headless", "se3_me_sa_results.json")
CMP = os.path.join(ROOT, "results", "sa_headless", "se3_3d_random_vs_optimized.json")

TIERS = list(SE3_TIER_LABELS)
COLORS = {"Random": "#888888", "ME": "#d62728"}


def load():
    cmp = json.load(open(CMP))
    return (json.load(open(SWEEP)), json.load(open(ME)),
            cmp.get("tiers", cmp))


def collect_bar_data(me, cmp, field_random, field_opt):
    rand = [cmp[t]["random_yaw"][field_random] for t in TIERS]
    me_v = [cmp[t].get("me_optimized", me[t])[field_opt] for t in TIERS]
    return rand, me_v


def plot_argmax(me, cmp, out):
    rand, me_v = collect_bar_data(
        me, cmp, "mean_argmax_accuracy", "argmax_accuracy")
    x = np.arange(len(TIERS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, rand, w, label="Random", color=COLORS["Random"])
    ax.bar(x + w / 2, me_v, w, label="ME-optimized", color=COLORS["ME"])
    ax.set_xticks(x)
    ax.set_xticklabels(TIERS)
    ax.set_ylabel("Argmax accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Argmax accuracy by tier")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    plt.close(fig)


def plot_task_and_time(me, cmp, out):
    ts_r, ts_m = collect_bar_data(
        me, cmp, "mean_task_success_rate", "task_success_rate")
    tm_r, tm_m = collect_bar_data(
        me, cmp, "mean_pick_time", "mean_pick_time")
    x = np.arange(len(TIERS))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (r, m), ylabel, title, ylim in [
        (axes[0], (ts_r, ts_m), "Task success rate",
         "Task success by tier", (0, 1.05)),
        (axes[1], (tm_r, tm_m), "Mean pick time (s)",
         "Time-to-inference by tier", None),
    ]:
        ax.bar(x - w / 2, r, w, label="Random", color=COLORS["Random"])
        ax.bar(x + w / 2, m, w, label="ME-optimized", color=COLORS["ME"])
        ax.set_xticks(x)
        ax.set_xticklabels(TIERS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    plt.close(fig)


def plot_threshold_heatmaps(sweep, out):
    taus = sweep["meta"]["tau_sweep"]
    metrics = [
        ("threshold_accuracy", "Threshold accuracy", (0, 1), "viridis"),
        ("convergence_rate", "Convergence rate", (0, 1), "viridis"),
        ("premature_wrong_rate", "Premature-wrong rate", (0, 1), "magma"),
        ("mean_time_s", "Mean time (s)", None, "cividis"),
    ]
    methods = ["ME"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    axes = np.atleast_2d(axes)
    for i, method in enumerate(methods):
        for j, (key, title, vlim, cmap) in enumerate(metrics):
            M = np.full((len(TIERS), len(taus)), np.nan)
            for ti, tier in enumerate(TIERS):
                md = sweep["tiers"][tier].get(method, {})
                if md.get("n_picks", 0) == 0:
                    continue
                for ki, tau in enumerate(taus):
                    M[ti, ki] = md[str(tau)][key]
            ax = axes[i, j]
            if vlim:
                im = ax.imshow(M, aspect="auto", cmap=cmap,
                               vmin=vlim[0], vmax=vlim[1])
            else:
                im = ax.imshow(M, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(taus)))
            ax.set_xticklabels([str(t) for t in taus], fontsize=8)
            ax.set_yticks(range(len(TIERS)))
            ax.set_yticklabels(TIERS, fontsize=8)
            ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{method}\nTier", fontsize=10)
            ax.set_xlabel(r"$\tau$")
            for ti in range(len(TIERS)):
                for ki in range(len(taus)):
                    v = M[ti, ki]
                    if not np.isnan(v):
                        ax.text(ki, ti, f"{v:.2f}", ha="center",
                                va="center", fontsize=6,
                                color="white" if (vlim and v < 0.5) else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Threshold sweep: ME-optimized (tier x tau) heatmaps")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    plt.close(fig)


def plot_pareto(sweep, out):
    taus = sweep["meta"]["tau_sweep"]
    tier_colors = plt.cm.viridis(np.linspace(0, 0.9, len(TIERS)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for ti, tier in enumerate(TIERS):
        md = sweep["tiers"][tier].get("ME", {})
        if md.get("n_picks", 0) == 0:
            continue
        xs = [md[str(t)]["mean_time_s"] for t in taus]
        ys = [md[str(t)]["threshold_accuracy"] for t in taus]
        ax.plot(xs, ys, "-", color=tier_colors[ti], alpha=0.5, lw=1)
        ax.scatter(xs, ys, marker="o", color=tier_colors[ti],
                   s=40, edgecolors="k", linewidths=0.3)
    # legend: tiers
    from matplotlib.lines import Line2D
    tier_handles = [Line2D([0], [0], marker="o", linestyle="",
                           color=tier_colors[i], label=TIERS[i],
                           markeredgecolor="k", markeredgewidth=0.3)
                    for i in range(len(TIERS))]
    ax.legend(handles=tier_handles, title="Tier",
              loc="lower right", fontsize=8)
    ax.set_xlabel("Mean time (s)")
    ax.set_ylabel("Threshold accuracy")
    ax.set_title(r"Accuracy vs. time (swept over $\tau$)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    plt.close(fig)


def main():
    sweep, me, cmp = load()
    plot_argmax(me, cmp, os.path.join(FIG_DIR, "fig_argmax_accuracy"))
    plot_task_and_time(me, cmp, os.path.join(FIG_DIR, "fig_task_and_time"))
    plot_threshold_heatmaps(sweep, os.path.join(FIG_DIR, "fig_threshold_heatmaps"))
    plot_pareto(sweep, os.path.join(FIG_DIR, "fig_pareto"))
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
