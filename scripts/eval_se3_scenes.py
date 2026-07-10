#!/usr/bin/env python3
"""Cross-evaluation: random vs 2D-optimized vs SE(3)-optimized layouts
across observer variants (2d-center, 3d-center, 3d-grasp-pos, se3-grasp).

For each scene, every layout x every observer mode is run through the
headless shared-autonomy runner with all orderings of pick objects.
Results are aggregated into a JSON payload and a results markdown doc.

Usage:
    python3 scripts/eval_se3_scenes.py --all
    python3 scripts/eval_se3_scenes.py --only scene_desk
"""
import os
import sys
import json
import argparse
import numpy as np
import yaml
from itertools import permutations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from run_sa_headless import (
    run_headless_sa, run_headless_sa_se3,
)
from experiments.se3_catalog import eval_scene_tiers
from envopt.layout_sampling import generate_random_layouts


TIERS = eval_scene_tiers()

OBSERVER_MODES = ["2d-center", "3d-center", "3d-grasp-pos", "se3-grasp"]

MAX_ORDERINGS = 24
N_RANDOM = 5


def summarize_picks(step_times):
    picks = [s for s in step_times
             if s["action"] == "pick" and s["inference_steps"] > 1]
    times = [s["inference_time_s"] for s in picks]
    argmax = [s.get("argmax_correct", s.get("correct", True)) for s in picks]
    threshold = [s.get("correct", False) for s in picks]
    return {
        "n_picks": len(picks),
        "mean_time": float(np.mean(times)) if times else float("nan"),
        "std_time": float(np.std(times)) if times else float("nan"),
        "argmax_acc": float(np.mean(argmax)) if argmax else 0.0,
        "threshold_acc": float(np.mean(threshold)) if threshold else 0.0,
    }


def run_one(task_path, scene_name, mode, perm, layout_override=None,
            layout_yaws=None, seed=42):
    """Run a single ordering of picks and return summarized metrics."""
    seq = ",".join(perm)
    kwargs = dict(
        scene_name=scene_name,
        user_sequence=seq,
        seed=seed,
        quiet=True,
        layout_override=layout_override,
        goal_timeout=15.0,
    )
    if mode == "2d-center":
        r = run_headless_sa(
            task_path,
            inference_model="path_efficiency",
            **kwargs,
        )
    else:
        r = run_headless_sa_se3(
            task_path,
            intent_mode=mode,
            inference_model="path_efficiency",
            layout_yaws=layout_yaws,
            **kwargs,
        )
    return summarize_picks(r["step_times"])


def agg_perms(task_path, scene_name, mode, objects, layout_override=None,
              layout_yaws=None, seed=42, max_orderings=None):
    """Aggregate metrics over multiple orderings."""
    if max_orderings is None:
        max_orderings = MAX_ORDERINGS
    # Cluttered scene: fewer orderings to keep runtime reasonable
    if len(objects) >= 8:
        max_orderings = min(max_orderings, 6)
    perms = list(permutations(objects))
    if len(perms) > max_orderings:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(perms), max_orderings, replace=False)
        perms = [perms[i] for i in idx]

    all_times, all_argmax, all_thresh = [], [], []
    for perm in perms:
        s = run_one(task_path, scene_name, mode, perm,
                    layout_override, layout_yaws, seed)
        if not np.isnan(s["mean_time"]):
            all_times.append(s["mean_time"])
        all_argmax.append(s["argmax_acc"])
        all_thresh.append(s["threshold_acc"])

    return {
        "n_orderings": len(perms),
        "mean_time": float(np.mean(all_times)) if all_times else float("nan"),
        "std_time": float(np.std(all_times)) if all_times else float("nan"),
        "argmax_acc": float(np.mean(all_argmax)),
        "threshold_acc": float(np.mean(all_thresh)),
    }


def load_se3_layout(scene_name):
    """Load the optimizer output JSON for an SE(3)-optimized scene."""
    path = os.path.join(PROJECT_ROOT, "results", "se3_optimize",
                        f"{scene_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    layout = {k: np.asarray(v) for k, v in data["positions"].items()}
    yaws = data["yaws"]
    return layout, yaws


def generate_random_for_scene(scene_name, objects, n, seed=0):
    """Generate N random layouts for a scene."""
    scene_yaml = os.path.join(PROJECT_ROOT, "config", "scenes",
                              f"{scene_name}.yaml")
    with open(scene_yaml) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    # Build (name -> z) from base scene by reading the MuJoCo XML
    import mujoco
    xml_path = os.path.join(PROJECT_ROOT, "src", "assets",
                            f"{scene_name}.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco_names = {o["short_name"]: o["mujoco_name"]
                    for o in scene_cfg["objects"]}
    z_map = {}
    for name in objects:
        mname = mujoco_names.get(name, name)
        try:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mname)
            z_map[name] = float(data.xpos[bid, 2])
        except Exception:
            z_map[name] = 0.95

    # Use the existing random layout generator
    fixed = {}   # no fixed objects
    layouts_2d = generate_random_layouts(
        n_layouts=n, movable_names=objects,
        fixed_positions=fixed, seed=seed)
    layouts_3d = []
    for L in layouts_2d:
        L3 = {k: np.array([v[0], v[1], z_map[k]]) for k, v in L.items()}
        layouts_3d.append(L3)
    return layouts_3d


def evaluate_scene(tier, modes, seed=42, n_random=N_RANDOM):
    """Run the full cross-eval for one scene tier."""
    name = tier["name"]
    objects = tier["objects"]
    task_2d = os.path.join(PROJECT_ROOT, tier["task_2d"])
    task_3d = os.path.join(PROJECT_ROOT, tier["task_3d"])

    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    results = {"scene": name, "objects": objects, "layouts": {}}

    # --- SE(3) optimized layout ---
    se3_data = load_se3_layout(name)
    if se3_data is not None:
        layout, yaws = se3_data
        print(f"  [SE(3)-optimized] yaws={yaws}")
        layouts_map = {"se3_optimized": (layout, yaws)}
    else:
        print(f"  [warn] no SE(3) optimized layout at results/se3_optimize/{name}.json")
        layouts_map = {}

    # --- Random baselines ---
    print(f"  Generating {n_random} random baselines...")
    random_layouts = generate_random_for_scene(name, objects, n_random, seed)
    for i, L in enumerate(random_layouts):
        layouts_map[f"random_{i}"] = (L, None)

    # For each layout, run every observer mode
    for layout_name, (layout, yaws) in layouts_map.items():
        results["layouts"][layout_name] = {}
        for mode in modes:
            task = task_2d if mode == "2d-center" else task_3d
            agg = agg_perms(
                task, name, mode, objects,
                layout_override=layout, layout_yaws=yaws,
                seed=seed)
            results["layouts"][layout_name][mode] = agg
            print(f"    {layout_name:<14} {mode:<15} "
                  f"time={agg['mean_time']:5.2f}s argmax={agg['argmax_acc']*100:5.1f}% "
                  f"thresh={agg['threshold_acc']*100:5.1f}%")

    return results


def aggregate_random(results):
    """Average metrics across random_* layouts for each mode."""
    by_mode = {}
    for layout_name, mode_data in results["layouts"].items():
        if not layout_name.startswith("random_"):
            continue
        for mode, agg in mode_data.items():
            by_mode.setdefault(mode, []).append(agg)
    avg = {}
    for mode, rows in by_mode.items():
        times = [r["mean_time"] for r in rows if not np.isnan(r["mean_time"])]
        argmax = [r["argmax_acc"] for r in rows]
        thresh = [r["threshold_acc"] for r in rows]
        avg[mode] = {
            "mean_time": float(np.mean(times)) if times else float("nan"),
            "std_time": float(np.std(times)) if times else float("nan"),
            "argmax_acc": float(np.mean(argmax)),
            "threshold_acc": float(np.mean(thresh)),
            "n_layouts": len(rows),
        }
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--n-random", type=int, default=N_RANDOM)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", type=str, default=None)
    args = parser.parse_args()

    only = None
    if args.only:
        only = set(s.strip() for s in args.only.split(","))

    modes = OBSERVER_MODES
    if args.modes:
        modes = [s.strip() for s in args.modes.split(",")]

    all_results = {}
    out_dir = os.path.join(PROJECT_ROOT, "results", "se3_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_eval.json")

    # Resume from existing JSON if present
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    for tier in TIERS:
        if only is not None and tier["name"] not in only:
            continue
        # Cluttered scene uses fewer orderings for speed
        max_ord = 6 if tier["name"] == "scene_cluttered" else MAX_ORDERINGS
        global MAX_ORDERINGS_GLOBAL  # noqa
        MAX_ORDERINGS_GLOBAL = max_ord  # not actually used, just docs
        print(f"  (max orderings for this scene: {max_ord})")
        r = evaluate_scene(
            tier, modes, seed=args.seed, n_random=args.n_random)
        r["random_aggregate"] = aggregate_random(r)
        all_results[tier["name"]] = r
        # Save incrementally so we keep partial results on interrupt
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Wrote partial results to {out_path}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
