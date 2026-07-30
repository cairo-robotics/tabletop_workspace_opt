#!/usr/bin/env python3
"""SE(3) workspace optimization via CMA-ME (MAP-Elites).

Runs nested yaw GD + collision-aware feasibility inside a CMA-ME search.
The objective is task-structure aware: each candidate layout is scored by
the worst SE(3) trajectory-margin slack over the task's unique competing
pick states, rather than assuming every task object competes in every
state.

Usage:
    python3 scripts/optimize/optimize_se3_map_elites.py
    python3 scripts/optimize/optimize_se3_map_elites.py --only scene_desk
    python3 scripts/optimize/optimize_se3_map_elites.py --quick   # fewer iters
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from envopt.grasp_library import GraspLibrary
from envopt.grasp_feasibility import check_layout_feasibility
from envopt.reachability import ReachabilityOracle
from envopt.yaw_optimizer import optimize_yaw

from experiments.se3_catalog import START_POS_3D, SE3_TIERS

TABLE_BOUNDS_X = (0.35, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)
ROBOT_HOME_2D = np.array([0.452, 0.160])
ROBOT_EXCLUSION = 0.15

SCENE_TIERS = [
    {
        "name": tier.scene,
        "task": tier.task_3d,
        "objects": list(tier.objects),
        "maxiter": tier.maxiter,
        "popsize": tier.popsize,
        "yaw_steps": tier.yaw_steps,
    }
    for tier in SE3_TIERS
]


def load_scene_yaml(scene_name):
    path = os.path.join(PROJECT_ROOT, "config", "scenes",
                        f"{scene_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["scene"]


def load_task_yaml(task_path):
    if not os.path.isabs(task_path):
        task_path = os.path.join(PROJECT_ROOT, task_path)
    with open(task_path) as f:
        return yaml.safe_load(f)["task"]


def read_baseline_z(scene_name, object_name):
    """Read the baseline z-height for an object from its MuJoCo XML."""
    import mujoco
    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        return float(data.xpos[bid, 2])
    except Exception:
        return 0.943


def build_scene_maps(scene_cfg, task_objects):
    """Extract half_extents, z_map, and MuJoCo body-name map."""
    half_extents = {}
    mujoco_names = {}
    for obj in scene_cfg["objects"]:
        name = obj["short_name"]
        he = obj.get("half_extents", [0.03, 0.03, 0.03])
        half_extents[name] = (he[0], he[1])
        mujoco_names[name] = obj["mujoco_name"]
    scene_name = scene_cfg["name"]
    z_map = {}
    for name in task_objects:
        mname = mujoco_names.get(name, name)
        z_map[name] = read_baseline_z(scene_name, mname)
    return half_extents, z_map, mujoco_names


def save_optimized_scene(result, scene_cfg, task_objects,
                         suffix="se3_me_optimized"):
    """Write optimized positions and yaws to a scene YAML."""
    out_name = f"{scene_cfg['name']}_{suffix}"
    out_path = os.path.join(PROJECT_ROOT, "config", "scenes",
                            f"{out_name}.yaml")
    new_cfg = {"name": out_name, "table": scene_cfg["table"],
               "objects": []}
    for obj in scene_cfg["objects"]:
        name = obj["short_name"]
        new_obj = dict(obj)
        if name in task_objects:
            pos = result["positions"][name]
            new_obj["position"] = [float(pos[0]), float(pos[1]), float(pos[2])]
            new_obj["yaw"] = float(result["yaws"].get(name, 0.0))
        new_cfg["objects"].append(new_obj)
    with open(out_path, "w") as f:
        yaml.safe_dump({"scene": new_cfg}, f, default_flow_style=False)
    return out_path


def extract_competing_pick_sets(task_cfg, fallback_objects):
    """Return unique task-state pick competitors.

    State-machine tasks contribute one set per state with two or more pick
    valid_goals. Pick-and-return tasks fall back to the declared pick_objects,
    because all remaining pick objects are candidates in each pick state.
    """
    pick_sets = []
    if "states" in task_cfg:
        for state_name, state_data in task_cfg["states"].items():
            pick_objects = [
                goal["object"]
                for goal in state_data.get("valid_goals", [])
                if goal.get("action") == "pick" and goal.get("object")
            ]
            if len(pick_objects) >= 2:
                pick_sets.append((state_name, pick_objects))
    if not pick_sets:
        pick_objects = list(task_cfg.get("pick_objects") or fallback_objects)
        if len(pick_objects) >= 2:
            pick_sets.append(("pick", pick_objects))

    seen = set()
    unique = []
    for name, objects in pick_sets:
        key = tuple(sorted(objects))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, objects))
    return unique


def optimize_task_yaws(layout_xy, pick_sets, grasp_lib, start_pos,
                       n_steps=20):
    """Optimize yaws per competing pick set and return worst slack.

    The returned yaw map merges the state-wise yaw optima, preferring larger
    competing sets first. Feasibility is still checked over the full task
    object union by the caller.
    """
    state_results = []
    worst_slack = np.inf
    for state_name, objects in pick_sets:
        sub_layout = {name: layout_xy[name] for name in objects}
        yaws, slack = optimize_yaw(
            sub_layout, objects, grasp_lib, start_pos,
            n_steps=n_steps, sigma_v=0.5, sigma_w=0.5, lambda_R=0.04,
            objective="trajectory_margin")
        state_yaws = {n: float(y) for n, y in zip(objects, yaws)}
        state_results.append((state_name, objects, state_yaws, float(slack)))
        worst_slack = min(worst_slack, float(slack))

    merged = {}
    for _state_name, objects, state_yaws, _slack in sorted(
            state_results, key=lambda item: len(item[1]), reverse=True):
        for name in objects:
            merged.setdefault(name, state_yaws[name])
    for name in layout_xy:
        merged.setdefault(name, 0.0)
    return merged, float(worst_slack), state_results


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
    from envopt.layout_sampling import generate_random_layouts
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
               fixed_positions, mujoco_names, pick_sets,
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

            # Task-aware yaw optimization: score by the worst competing
            # pick-state slack, not by all task objects unconditionally.
            try:
                yaw_map, slack, _state_results = optimize_task_yaws(
                    layout_xy, pick_sets, grasp_lib, START_POS_3D,
                    n_steps=20)
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

    candidate_yaw_map, candidate_slack, candidate_state_results = \
        optimize_task_yaws(
            best_positions, pick_sets, grasp_lib, START_POS_3D,
            n_steps=20)
    candidate_feas = check_layout_feasibility(
        best_positions, candidate_yaw_map, half_extents, grasp_lib, oracle,
        task_objects=task_objects,
        min_approach_clearance=0.005,
        min_object_separation=0.005,
        mujoco_names=mujoco_names,
        check_arm_collisions=True,
        fixed_positions=fixed_positions)

    best_yaw_map, best_slack, state_results = optimize_task_yaws(
        best_positions, pick_sets, grasp_lib, START_POS_3D,
        n_steps=80)

    # Final feasibility check
    final_feas = check_layout_feasibility(
        best_positions, best_yaw_map, half_extents, grasp_lib, oracle,
        task_objects=task_objects,
        min_approach_clearance=0.005,
        min_object_separation=0.005,
        mujoco_names=mujoco_names,
        check_arm_collisions=True,
        fixed_positions=fixed_positions)
    if not final_feas["feasible"] and candidate_feas["feasible"]:
        best_yaw_map = candidate_yaw_map
        best_slack = candidate_slack
        state_results = candidate_state_results
        final_feas = candidate_feas

    return {
        "best_positions": best_positions,
        "best_yaws": best_yaw_map,
        "best_slack": float(best_slack),
        "pick_state_slacks": {
            name: slack for name, _objects, _yaws, slack in state_results
        },
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
        task_cfg = load_task_yaml(tier["task"])
        pick_sets = extract_competing_pick_sets(task_cfg, tier["objects"])
        if not pick_sets:
            raise RuntimeError(f"No competing pick states for {tier['task']}")
        print("  Competing pick states:")
        for state_name, objects in pick_sets:
            print(f"    {state_name}: {objects}")

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
            fixed_positions, mujoco_names, pick_sets,
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
            "pick_states": [
                {"state": name, "objects": objects}
                for name, objects in pick_sets
            ],
            "pick_state_slacks": result["pick_state_slacks"],
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
