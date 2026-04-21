#!/usr/bin/env python3
"""SE(3) workspace optimization via CMA-ME (MAP-Elites).

Uses the same SE(3) fitness pipeline as optimize_se3_scenes.py (nested
yaw GD + collision-aware feasibility) but searches with CMA-ME instead
of differential evolution. Produces a diverse archive of feasible
layouts rather than a single best.

Usage:
    python3 scripts/optimize_se3_map_elites.py
    python3 scripts/optimize_se3_map_elites.py --only scene_desk
    python3 scripts/optimize_se3_map_elites.py --quick   # fewer iters
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from envopt.grasp_library import GraspLibrary
from envopt.grasp_feasibility import (
    check_layout_feasibility, object_footprints, pairwise_overlap_penalty)
from envopt.reachability import ReachabilityOracle
from envopt.yaw_optimizer import optimize_yaw

# Reuse scene definitions from the DE optimizer
from optimize_se3_scenes import (
    SCENE_TIERS, START_POS_3D, load_scene_yaml,
    build_scene_maps, read_baseline_z, save_optimized_scene,
)

TABLE_BOUNDS_X = (0.35, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)
ROBOT_HOME_2D = np.array([0.452, 0.160])
ROBOT_EXCLUSION = 0.15


def compute_features(positions_2d: np.ndarray) -> tuple:
    """Compute (mean_pairwise_dist, centroid_offset)."""
    M = len(positions_2d)
    dists = []
    for i in range(M):
        for j in range(i + 1, M):
            dists.append(np.linalg.norm(positions_2d[i] - positions_2d[j]))
    f1 = float(np.mean(dists)) if dists else 0.0
    centroid = positions_2d.mean(axis=0)
    f2 = float(np.linalg.norm(centroid - np.array([0.6, 0.0])))
    return f1, f2


def calibrate_ranges(task_objects, fixed_positions, half_extents,
                     n_samples=1000, seed=99):
    """Sample random layouts and compute feature ranges."""
    from run_sa_headless import generate_random_layouts
    layouts = generate_random_layouts(
        n_samples, task_objects, fixed_positions, seed=seed)
    f1s, f2s = [], []
    for layout in layouts:
        pos = np.array([layout[n] for n in task_objects])
        f1, f2 = compute_features(pos)
        f1s.append(f1)
        f2s.append(f2)
    if not f1s:
        return [(0.05, 0.6), (0.0, 0.4)]
    return [
        (float(np.percentile(f1s, 5)), float(np.percentile(f1s, 95))),
        (float(np.percentile(f2s, 5)), float(np.percentile(f2s, 95))),
    ]


def run_cma_me(task_objects, half_extents, z_map, grasp_lib, oracle,
               fixed_positions, mujoco_names,
               iterations=200, seed=42, log_interval=25):
    """Run CMA-ME MAP-Elites for SE(3) workspace optimization."""
    M = len(task_objects)
    solution_dim = M * 2

    rng = np.random.default_rng(seed)
    x0 = np.zeros(solution_dim)
    for i in range(M):
        x0[i * 2] = rng.uniform(TABLE_BOUNDS_X[0] + 0.1,
                                 TABLE_BOUNDS_X[1] - 0.1)
        x0[i * 2 + 1] = rng.uniform(TABLE_BOUNDS_Y[0] + 0.1,
                                     TABLE_BOUNDS_Y[1] - 0.1)

    lower = np.array([TABLE_BOUNDS_X[0], TABLE_BOUNDS_Y[0]] * M)
    upper = np.array([TABLE_BOUNDS_X[1], TABLE_BOUNDS_Y[1]] * M)

    ranges = calibrate_ranges(task_objects, fixed_positions, half_extents)
    print(f"  Feature ranges: dist={ranges[0]}, offset={ranges[1]}")

    archive = GridArchive(
        solution_dim=solution_dim, dims=[20, 20], ranges=ranges)

    emitters = [
        EvolutionStrategyEmitter(
            archive, x0=x0, sigma0=0.08,
            ranker="2imp",
            bounds=list(zip(lower, upper)),
            batch_size=30,
        )
        for _ in range(3)
    ]
    scheduler = Scheduler(archive, emitters)
    history = []

    for itr in range(1, iterations + 1):
        sols = scheduler.ask()
        objs = []
        measures = []

        for sol in sols:
            layout_xy = {}
            for i, name in enumerate(task_objects):
                layout_xy[name] = np.array([
                    sol[i * 2], sol[i * 2 + 1], z_map[name]])

            # Soft penalty: size-aware separation + robot exclusion
            penalty = 0.0
            all_pos = dict(layout_xy)
            if fixed_positions:
                all_pos.update(fixed_positions)
            all_names = list(all_pos.keys())
            for k in range(len(all_names)):
                for l in range(k + 1, len(all_names)):
                    d = np.linalg.norm(
                        all_pos[all_names[k]][:2] -
                        all_pos[all_names[l]][:2])
                    he_k = half_extents.get(all_names[k], (0.03, 0.03))
                    he_l = half_extents.get(all_names[l], (0.03, 0.03))
                    r_k = (he_k[0] + he_k[1]) * 0.5
                    r_l = (he_l[0] + he_l[1]) * 0.5
                    min_d = r_k + r_l + 0.03
                    if d < min_d:
                        penalty += (min_d - d) ** 2 * 100.0
            for name in task_objects:
                d_home = np.linalg.norm(layout_xy[name][:2] - ROBOT_HOME_2D)
                if d_home < ROBOT_EXCLUSION:
                    penalty += (ROBOT_EXCLUSION - d_home) ** 2 * 100.0

            if penalty > 1e-3:
                objs.append(-1000.0 - penalty)
                pos_2d = np.array([sol[i * 2:i * 2 + 2] for i in range(M)])
                measures.append(list(compute_features(pos_2d)))
                continue

            # Yaw optimization
            try:
                yaws, slack = optimize_yaw(
                    layout_xy, task_objects, grasp_lib, START_POS_3D,
                    n_steps=20, sigma_v=0.5, sigma_w=0.5, lambda_R=0.04, objective="trajectory_margin")
                yaw_map = {n: float(y)
                           for n, y in zip(task_objects, yaws)}
            except Exception:
                objs.append(-500.0)
                pos_2d = np.array([sol[i * 2:i * 2 + 2] for i in range(M)])
                measures.append(list(compute_features(pos_2d)))
                continue

            # Hard feasibility
            feas = check_layout_feasibility(
                layout_xy, yaw_map, half_extents, grasp_lib, oracle,
                task_objects=task_objects,
                min_approach_clearance=0.005,
                min_object_separation=0.005,
                mujoco_names=mujoco_names,
                check_arm_collisions=True,
                fixed_positions=fixed_positions)

            if not feas["feasible"]:
                objs.append(-200.0)
                pos_2d = np.array([sol[i * 2:i * 2 + 2] for i in range(M)])
                measures.append(list(compute_features(pos_2d)))
                continue

            objs.append(float(slack))
            pos_2d = np.array([sol[i * 2:i * 2 + 2] for i in range(M)])
            measures.append(list(compute_features(pos_2d)))

        scheduler.tell(objs, measures)

        if itr % log_interval == 0 or itr == iterations:
            stats = archive.stats
            history.append({
                "iteration": itr,
                "best": float(stats.obj_max),
                "coverage": float(stats.coverage),
                "qd_score": float(stats.qd_score),
                "num_elites": int(stats.num_elites),
            })
            print(f"    itr {itr:4d}: best={stats.obj_max:.2f} "
                  f"elites={stats.num_elites} "
                  f"coverage={stats.coverage:.3f} "
                  f"qd={stats.qd_score:.1f}")

    # Extract best and top-5
    best_elite = archive.best_elite
    best_sol = best_elite["solution"]
    best_positions = {}
    for i, name in enumerate(task_objects):
        best_positions[name] = np.array([
            best_sol[i * 2], best_sol[i * 2 + 1], z_map[name]])

    # Re-run yaw GD with feasibility tracking on the best elite
    def _feas_fn(yaw_dict):
        fr = check_layout_feasibility(
            best_positions, yaw_dict, half_extents, grasp_lib, oracle,
            task_objects=task_objects,
            min_approach_clearance=0.005,
            min_object_separation=0.005,
            mujoco_names=mujoco_names,
            check_arm_collisions=True,
            fixed_positions=fixed_positions)
        return fr["feasible"]

    best_yaws, best_slack = optimize_yaw(
        best_positions, task_objects, grasp_lib, START_POS_3D,
        n_steps=80, sigma_v=0.5, sigma_w=0.5, lambda_R=0.04, objective="trajectory_margin",
        feasibility_fn=_feas_fn)
    best_yaw_map = {n: float(y) for n, y in zip(task_objects, best_yaws)}

    # Final feasibility check
    final_feas = check_layout_feasibility(
        best_positions, best_yaw_map, half_extents, grasp_lib, oracle,
        task_objects=task_objects,
        min_approach_clearance=0.005,
        min_object_separation=0.005,
        mujoco_names=mujoco_names,
        check_arm_collisions=True,
        fixed_positions=fixed_positions)

    return {
        "best_positions": best_positions,
        "best_yaws": best_yaw_map,
        "best_slack": float(best_slack),
        "feasibility": final_feas,
        "archive_stats": {
            "num_elites": int(archive.stats.num_elites),
            "coverage": float(archive.stats.coverage),
            "qd_score": float(archive.stats.qd_score),
            "best_fitness": float(archive.stats.obj_max),
        },
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    lib = GraspLibrary.load(
        os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml"))
    out_dir = os.path.join(PROJECT_ROOT, "results", "se3_map_elites")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for tier in SCENE_TIERS:
        scene_name = tier["name"]
        if args.only and scene_name != args.only:
            continue

        print(f"\n{'='*60}\n  {scene_name} (MAP-Elites)\n{'='*60}")
        scene_cfg = load_scene_yaml(scene_name)
        half_ext, z_map, mujoco_names = build_scene_maps(
            scene_cfg, tier["objects"])
        oracle = ReachabilityOracle(
            os.path.join(PROJECT_ROOT, "src", "assets",
                         f"{scene_name}.xml"),
            pos_tol=0.02, rot_tol_deg=20.0, max_iters=30, n_seeds=3)

        # Fixed positions
        import mujoco
        _m = mujoco.MjModel.from_xml_path(
            os.path.join(PROJECT_ROOT, "src", "assets",
                         f"{scene_name}.xml"))
        _d = mujoco.MjData(_m)
        if _m.nkey > 0:
            mujoco.mj_resetDataKeyframe(_m, _d, 0)
        mujoco.mj_forward(_m, _d)
        fixed_positions = {}
        for o in scene_cfg["objects"]:
            name = o["short_name"]
            if name not in tier["objects"]:
                mname = mujoco_names.get(name, name)
                bid = mujoco.mj_name2id(
                    _m, mujoco.mjtObj.mjOBJ_BODY, mname)
                if bid >= 0:
                    fixed_positions[name] = _d.xpos[bid].copy()
        if fixed_positions:
            print(f"  Fixed objects: {list(fixed_positions.keys())}")

        iters = max(50, tier["maxiter"] * 3) if not args.quick else \
            max(25, tier["maxiter"])

        t0 = time.time()
        result = run_cma_me(
            tier["objects"], half_ext, z_map, lib, oracle,
            fixed_positions, mujoco_names,
            iterations=iters, seed=args.seed)
        elapsed = time.time() - t0

        print(f"  Time: {elapsed:.0f}s")
        print(f"  Best slack: {result['best_slack']:.2f}")
        print(f"  Feasible: {result['feasibility']['feasible']}")
        print(f"  Archive: {result['archive_stats']['num_elites']} elites, "
              f"coverage={result['archive_stats']['coverage']:.3f}")

        # Save optimized scene YAML
        out_result = {
            "positions": {k: v for k, v in result["best_positions"].items()},
            "yaws": result["best_yaws"],
        }
        out_path = save_optimized_scene(
            out_result, scene_cfg, tier["objects"],
            suffix="se3_me_optimized")
        print(f"  Wrote: {out_path}")

        results[scene_name] = {
            "slack": result["best_slack"],
            "feasible": result["feasibility"]["feasible"],
            "archive_stats": result["archive_stats"],
            "elapsed_s": elapsed,
            "positions": {k: v.tolist()
                          for k, v in result["best_positions"].items()},
            "yaws": result["best_yaws"],
        }

        json_path = os.path.join(out_dir, f"{scene_name}.json")
        with open(json_path, "w") as f:
            json.dump(results[scene_name], f, indent=2)

    # Summary
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, r in results.items():
        print(f"  {name}: slack={r['slack']:.2f} "
              f"feas={r['feasible']} "
              f"elites={r['archive_stats']['num_elites']} "
              f"coverage={r['archive_stats']['coverage']:.3f} "
              f"{r['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
