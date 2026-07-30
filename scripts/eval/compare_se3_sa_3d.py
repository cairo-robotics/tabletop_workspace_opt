#!/usr/bin/env python3
"""3D SE(3) shared autonomy comparison: random vs optimized layouts.

Evaluates intent inference using the SE(3) path-efficiency observer
(matched to the SE(3) optimizer) on three layout conditions:

  1. random-yaw:          random (x,y) + random yaw per object
  2. random-yaw-optimized: random (x,y) + yaw optimized via nested GD
  3. SE(3)-optimized:      (x,y,yaw) from the SE(3) optimizer output

For each layout, grasps are checked for IK feasibility + arm-vs-other-
object collision BEFORE running SA. Infeasible grasps are skipped
(recorded as failures).

Metrics:
  - task_success_rate: fraction of full tasks where ALL picks had feasible grasps
  - infeasible_pick_rate: fraction of individual picks with no IK or collision
  - argmax_accuracy: over feasible picks only
  - threshold_accuracy: over feasible picks only
  - pick_time: over feasible picks only

Usage:
    PYTHONUNBUFFERED=1 python3 scripts/eval/compare_se3_sa_3d.py --n-random 30
"""
import argparse
import json
import os
import sys
from itertools import permutations

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from run_sa_headless import run_headless_sa_se3
from experiments.se3_catalog import START_POS_3D, compare_scene_tiers
from experiments.provenance import (
    build_se3_sa_metadata,
    metadata_compatible,
)
from experiments.sa_metrics import aggregate_layout_results
from envopt.layout_sampling import generate_random_se3_layouts
from envopt.grasp_library import GraspLibrary
from envopt.grasp_feasibility import check_layout_feasibility
from envopt.reachability import ReachabilityOracle
from envopt.yaw_optimizer import optimize_yaw


MAX_ORDERINGS = 120
GRASP_LIB_PATH = os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml")
SCENE_TIERS = compare_scene_tiers()


def load_scene_meta(scene_name):
    """Return names, footprints, heights, and base poses from MuJoCo."""
    yaml_path = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)["scene"]
    import mujoco
    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    mujoco_names = {}
    half_extents = {}
    z_map = {}
    fixed_xy = {}
    fixed_yaw = {}
    for o in cfg["objects"]:
        short = o["short_name"]
        mname = o["mujoco_name"]
        mujoco_names[short] = mname
        he = o.get("half_extents", [0.03, 0.03, 0.03])
        half_extents[short] = (he[0], he[1])
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mname)
        if bid >= 0:
            z_map[short] = float(d.xpos[bid, 2])
            fixed_xy[short] = d.xpos[bid, :2].copy()
            fixed_yaw[short] = float(np.arctan2(
                d.xmat[bid].reshape(3, 3)[1, 0],
                d.xmat[bid].reshape(3, 3)[0, 0]))
    return mujoco_names, half_extents, z_map, fixed_xy, fixed_yaw


def check_feasibility(layout_xy3, yaw_map, half_extents, grasp_lib,
                      oracle, task_objects, mujoco_names,
                      fixed_positions=None):
    """Run the collision-aware feasibility check and return per-pick results."""
    feas = check_layout_feasibility(
        layout_xy3, yaw_map, half_extents, grasp_lib, oracle,
        task_objects=task_objects,
        mujoco_names=mujoco_names,
        check_arm_collisions=True,
        fixed_positions=fixed_positions)
    # Parse which objects failed
    failed_objects = set()
    for f in feas["failures"]:
        obj = f.split(":")[0]
        failed_objects.add(obj)
    # Global footprint failures cannot be attributed safely to one object.
    # Treat every task pick as infeasible rather than silently reporting a
    # feasible task for an invalid layout.
    if not feas["feasible"] and not failed_objects:
        failed_objects.update(task_objects)
    return feas["feasible"], failed_objects


def build_metadata(args, selected_tiers):
    return build_se3_sa_metadata(PROJECT_ROOT, args, selected_tiers)


def run_sa_for_layout(task_path, scene_name, layout_xy, yaw_map,
                      failed_objects, task_objects, sa_kwargs,
                      max_orderings=MAX_ORDERINGS, seed=42):
    """Run SA for all orderings on one layout, skipping infeasible picks.

    Returns dict with aggregated metrics.
    """
    with open(task_path) as f:
        task_config = yaml.safe_load(f)["task"]

    pick_objects = task_config.get("pick_objects", task_objects)
    all_perms = list(permutations(pick_objects))
    if len(all_perms) > max_orderings:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(all_perms), size=max_orderings, replace=False)
        perms = [all_perms[i] for i in indices]
    else:
        perms = all_perms

    all_pick_times = []
    all_thresh_correct = []
    all_argmax_correct = []
    n_infeasible_picks = 0
    n_total_picks = 0
    n_task_success = 0  # tasks where ALL picks were feasible

    for perm in perms:
        seq = ",".join(perm)
        # Check if any pick in this ordering is infeasible
        task_has_infeasible = any(obj in failed_objects for obj in perm)

        r = run_headless_sa_se3(
            task_path, scene_name=scene_name,
            intent_mode="se3-grasp",
            grasp_library_path=GRASP_LIB_PATH,
            user_sequence=seq, seed=seed,
            layout_override=layout_xy,
            layout_yaws=yaw_map,
            quiet=True, benchmark_mode=True, **sa_kwargs)

        if not task_has_infeasible:
            n_task_success += 1

        for s in r.get("step_times", []):
            if s["action"] != "pick":
                continue
            n_total_picks += 1
            obj = s.get("target_object", s.get("object", ""))
            if obj in failed_objects:
                n_infeasible_picks += 1
                continue
            all_pick_times.append(s["inference_time_s"])
            all_thresh_correct.append(s.get("correct", False))
            all_argmax_correct.append(
                s.get("argmax_correct", s.get("correct", False)))

    n_orderings = len(perms)
    expected_picks = n_orderings * len(pick_objects)
    if n_total_picks != expected_picks:
        raise RuntimeError(
            f"Benchmark invariant failed: expected {expected_picks} picks "
            f"({n_orderings} orderings x {len(pick_objects)} objects), "
            f"recorded {n_total_picks}")
    return {
        "n_orderings": n_orderings,
        "n_total_picks": n_total_picks,
        "expected_picks": expected_picks,
        "n_infeasible_picks": n_infeasible_picks,
        "infeasible_pick_rate": (
            n_infeasible_picks / max(n_total_picks, 1)),
        "task_success_rate": n_task_success / max(n_orderings, 1),
        "n_feasible_picks": len(all_pick_times),
        "mean_pick_time": (
            float(np.mean(all_pick_times)) if all_pick_times else 0),
        "std_pick_time": (
            float(np.std(all_pick_times)) if all_pick_times else 0),
        "threshold_accuracy": (
            float(np.mean(all_thresh_correct)) if all_thresh_correct else 0),
        "argmax_accuracy": (
            float(np.mean(all_argmax_correct)) if all_argmax_correct else 0),
    }


def load_optimized_yaml(scene_name, suffix):
    """Load optimized layout from YAML. Returns (layout_xy, yaw_map) or None.

    suffix: currently 'se3_me_optimized' (MAP-Elites).
    """
    p = os.path.join(
        PROJECT_ROOT, "config", "scenes",
        f"{scene_name}_{suffix}.yaml")
    if not os.path.exists(p):
        return None
    with open(p) as f:
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


def load_me_optimized(scene_name):
    return load_optimized_yaml(scene_name, "se3_me_optimized")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--control-rate", type=float, default=5.0)
    parser.add_argument("--lambda-R", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-random", action="store_true",
                        help="Skip random_yaw arm; merge-update only "
                             "me_optimized.")
    parser.add_argument("--tiers", nargs="+", default=None,
                        help="Only run these tiers (e.g. Med-A Hard-B).")
    parser.add_argument("--only-random", action="store_true",
                        help="Skip ME; only run random arm. "
                             "Writes per-layout raw results to --random-parts-dir.")
    parser.add_argument("--random-max-orderings", type=int, default=120,
                        help="Max orderings for random arm (120 = unchanged).")
    parser.add_argument("--random-layout-offset", type=int, default=0,
                        help="Offset into the deterministic random layouts.")
    parser.add_argument("--random-layout-count", type=int, default=-1,
                        help="Number of random layouts to evaluate starting "
                             "at offset (-1 = all).")
    parser.add_argument("--random-parts-dir", type=str,
                        default="/tmp/random_parts",
                        help="Where --only-random writes per-chunk JSON.")
    parser.add_argument("--arrival-dist", type=float, default=0.02)
    parser.add_argument("--only-me", action="store_true",
                        help="Skip random; only run ME layout.")
    parser.add_argument("--force-incompatible-merge", action="store_true",
                        help="Allow merging results with incompatible or "
                             "legacy provenance metadata.")
    args = parser.parse_args()

    sa_kwargs = dict(
        inference_model="path_efficiency",
        beta=args.beta, noise=args.noise, threshold=args.threshold,
        lambda_R=args.lambda_R,
        max_speed=0.05, control_rate=args.control_rate,
        arrival_dist=args.arrival_dist,
    )

    grasp_lib = GraspLibrary.load(GRASP_LIB_PATH)
    all_results = {}

    selected_tiers = SCENE_TIERS
    if args.tiers:
        keep = set(args.tiers)
        selected_tiers = [t for t in SCENE_TIERS if t[2] in keep]
    # Hash the full experiment input set even for tier/worker subsets so all
    # compatible parts carry identical merge provenance.
    metadata = build_metadata(args, SCENE_TIERS)

    for task_path_rel, scene_name, tier, task_objects in selected_tiers:
        task_path = os.path.join(PROJECT_ROOT, task_path_rel)
        print(f"\n{'='*70}")
        print(f"  {tier} — {scene_name}")
        print(f"{'='*70}")

        mujoco_names, half_extents, z_map, fixed_xy, fixed_yaw = load_scene_meta(
            scene_name)
        base_xml = os.path.join(
            PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
        oracle = ReachabilityOracle(
            base_xml, pos_tol=0.02, rot_tol_deg=20.0,
            max_iters=30, n_seeds=4)

        rnd_yaw_results = []
        rnd_yopt_results = []
        rnd_yaw_indices = []  # layout index for each result (for part files)
        if not args.skip_random and not args.only_me:
            fixed_for_random = {n: xy for n, xy in fixed_xy.items()
                                if n not in task_objects}
            all_layouts = generate_random_se3_layouts(
                args.n_random, task_objects, fixed_for_random, half_extents,
                seed=args.seed,
                fixed_yaws={n: fixed_yaw.get(n, 0.0)
                            for n in fixed_for_random})
            # Slice for worker parallelism
            off = args.random_layout_offset
            cnt = args.random_layout_count
            if cnt < 0:
                layouts = all_layouts[off:]
                indices = list(range(off, len(all_layouts)))
            else:
                layouts = all_layouts[off:off + cnt]
                indices = list(range(off, off + len(layouts)))
            print(f"  Using {len(layouts)} random layouts "
                  f"(offset={off} count={cnt}, total={len(all_layouts)})")

            # --- random-yaw baseline ---
            print(f"\n  Random-yaw baseline ({len(layouts)} layouts)...")
            for i, (layout_idx, layout_data) in enumerate(zip(indices, layouts)):
                layout_2d, yaw_map = layout_data
                layout_3d = {n: np.array([xy[0], xy[1], z_map.get(n, 0.95)])
                             for n, xy in layout_2d.items()}
                _, failed = check_feasibility(
                    layout_3d, yaw_map, half_extents, grasp_lib,
                    oracle, task_objects, mujoco_names,
                    fixed_positions=fixed_for_random)
                r = run_sa_for_layout(
                    task_path, scene_name, layout_2d, yaw_map,
                    failed, task_objects, sa_kwargs,
                    max_orderings=args.random_max_orderings,
                    seed=args.seed + layout_idx)
                rnd_yaw_results.append(r)
                rnd_yaw_indices.append(layout_idx)
                print(f"    layout[{layout_idx}]: "
                      f"task_ok={r['task_success_rate']:.0%} "
                      f"argmax={r['argmax_accuracy']:.0%} "
                      f"infeas={r['infeasible_pick_rate']:.0%}")

            # random-yaw-optimized: only run if NOT only-random (we're
            # deprecating this arm; --only-random workers skip it).
            if not args.only_random:
                print(f"\n  Random-yaw-optimized baseline ({len(layouts)} layouts)...")
                for i, (layout_idx, layout_2d) in enumerate(
                        (idx, data[0]) for idx, data in zip(indices, layouts)):
                    layout_3d = {n: np.array([xy[0], xy[1],
                                              z_map.get(n, 0.95)])
                                 for n, xy in layout_2d.items()}
                    try:
                        yaws, _ = optimize_yaw(
                            layout_3d, task_objects, grasp_lib, START_POS_3D,
                            n_steps=20, sigma_v=0.5, sigma_w=0.5, lambda_R=0.04)
                        yaw_map = {n: float(y) for n, y
                                   in zip(task_objects, yaws)}
                    except Exception:
                        yaw_map = {n: 0.0 for n in task_objects}
                    _, failed = check_feasibility(
                        layout_3d, yaw_map, half_extents, grasp_lib,
                        oracle, task_objects, mujoco_names,
                        fixed_positions=fixed_for_random)
                    r = run_sa_for_layout(
                        task_path, scene_name, layout_2d, yaw_map,
                        failed, task_objects, sa_kwargs,
                        max_orderings=args.random_max_orderings,
                        seed=args.seed + layout_idx)
                    rnd_yopt_results.append(r)
                    print(f"    layout[{layout_idx}]: "
                          f"task_ok={r['task_success_rate']:.0%} "
                          f"argmax={r['argmax_accuracy']:.0%} "
                          f"infeas={r['infeasible_pick_rate']:.0%}")

            # Write per-chunk raw results if we're a worker
            if args.only_random:
                os.makedirs(args.random_parts_dir, exist_ok=True)
                part_path = os.path.join(
                    args.random_parts_dir,
                    f"random_part_{tier}_{off}_{len(layouts)}.json")
                part_data = {
                    "schema_version": 2,
                    "experiment": metadata,
                    "tier": tier,
                    "scene": scene_name,
                    "layout_indices": rnd_yaw_indices,
                    "random_yaw_results": rnd_yaw_results,
                    "params": {
                        "n_random_total": args.n_random,
                        "random_max_orderings": args.random_max_orderings,
                        "seed": args.seed,
                        "arrival_dist": args.arrival_dist,
                    },
                }
                with open(part_path, "w") as f:
                    json.dump(part_data, f, indent=2, default=str)
                print(f"  Wrote part: {part_path}")
        else:
            print(f"  --skip-random: preserving existing random arms in JSON")

        # --- ME-optimized ---
        me_data = load_me_optimized(scene_name) if not args.only_random else None
        me_result = None
        if me_data is not None:
            layout_xy_me, yaw_map_me = me_data
            layout_3d_me = {
                n: np.array([xy[0], xy[1], z_map.get(n, 0.95)])
                for n, xy in layout_xy_me.items()}
            _, failed_me = check_feasibility(
                layout_3d_me, yaw_map_me, half_extents, grasp_lib,
                oracle, task_objects, mujoco_names)
            print(f"\n  ME-optimized (failed objects: {failed_me or 'none'})...")
            me_result = run_sa_for_layout(
                task_path, scene_name, layout_xy_me, yaw_map_me,
                failed_me, task_objects, sa_kwargs, seed=args.seed)
            print(f"    task_ok={me_result['task_success_rate']:.0%} "
                  f"argmax={me_result['argmax_accuracy']:.0%} "
                  f"infeas={me_result['infeasible_pick_rate']:.0%}")
        else:
            print(f"\n  SKIP ME-optimized: no YAML for {scene_name}")

        tier_entry = {}
        # In worker mode (--only-random), skip aggregation/main-JSON write;
        # the part file has the raw per-layout results already.
        if args.only_random:
            continue
        if not args.skip_random:
            tier_entry["random_yaw"] = aggregate_layout_results(rnd_yaw_results)
            if rnd_yopt_results:
                tier_entry["random_yaw_optimized"] = aggregate_layout_results(
                    rnd_yopt_results)
        if me_result is not None:
            tier_entry["me_optimized"] = me_result
        all_results[tier] = tier_entry

    if args.only_random:
        return

    document = {"schema_version": 2, "experiment": metadata,
                "tiers": all_results}
    # Save (merge-update only when provenance is compatible).
    out = os.path.join(
        PROJECT_ROOT, "results", "sa_headless",
        "se3_3d_random_vs_optimized.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Always merge into existing JSON to preserve other tiers / arms.
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
        if not metadata_compatible(existing, document):
            if not args.force_incompatible_merge:
                raise RuntimeError(
                    "Refusing to merge incompatible or legacy results. "
                    "Archive the existing file, rerun all arms together, or "
                    "pass --force-incompatible-merge explicitly.")
            print("WARNING: forcing an incompatible result merge")
            if "tiers" not in existing:
                existing = {"schema_version": 2, "experiment": metadata,
                            "tiers": existing}
        for tier, entry in all_results.items():
            merged = existing["tiers"].get(tier, {})
            merged.update(entry)
            existing["tiers"][tier] = merged
        to_write = existing
        print(f"\nMerging into existing file (preserving other tiers/arms).")
    else:
        to_write = document
    tmp_out = out + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(to_write, f, indent=2, default=str)
    os.replace(tmp_out, out)
    print(f"Saved: {out}")

    # Summary table
    print(f"\n{'='*120}")
    print(f"3D SE(3) COMPARISON — SE(3) path-efficiency observer, "
          f"beta={args.beta}, threshold={args.threshold}")
    print(f"{'='*120}")
    print(f"  {'Tier':>7} {'Layout':>18} {'Task Succ':>10} "
          f"{'Infeas%':>8} {'Pick Time':>14} {'Thr Acc':>9} "
          f"{'Argmax':>8}")
    print("  " + "-" * 110)

    for task_path_rel, scene_name, tier, task_objects in SCENE_TIERS:
        tr = all_results.get(tier, {})
        for label, key in [("random-yaw", "random_yaw"),
                           ("random-yopt", "random_yaw_optimized"),
                           ("ME-optimized", "me_optimized")]:
            r = tr.get(key)
            if r is None:
                continue
            if "mean_task_success_rate" in r:
                # aggregated random
                print(f"  {tier:>7} {label:>18} "
                      f"{r['mean_task_success_rate']:>9.0%} "
                      f"{r['mean_infeasible_pick_rate']:>7.0%} "
                      f"{r['mean_pick_time']:>6.2f}s±{r['std_pick_time']:.2f} "
                      f"{r['mean_threshold_accuracy']:>8.0%} "
                      f"{r['mean_argmax_accuracy']:>7.0%}")
            else:
                # single optimized
                print(f"  {tier:>7} {label:>18} "
                      f"{r['task_success_rate']:>9.0%} "
                      f"{r['infeasible_pick_rate']:>7.0%} "
                      f"{r['mean_pick_time']:>6.2f}s±{r['std_pick_time']:.2f} "
                      f"{r['threshold_accuracy']:>8.0%} "
                      f"{r['argmax_accuracy']:>7.0%}")
        print()


if __name__ == "__main__":
    main()
