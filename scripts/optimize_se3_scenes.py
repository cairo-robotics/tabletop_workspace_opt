#!/usr/bin/env python3
"""Run SE(3) layout optimization on every scene tier.

Produces:
  - config/scenes/scene_<name>_se3_optimized.yaml (layout + yaws)
  - results/se3_optimize/<name>.json (slack, feasibility, stats)

Each optimization call wraps the DE + nested yaw GD optimizer.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from envopt.se3_layout_optimizer import optimize_scene
from envopt.grasp_library import GraspLibrary
from envopt.reachability import ReachabilityOracle


# Scene-level task definitions. Matches the tiers referenced in the
# plan's §7.2 table and the existing eval_map_elites_tiers.
SCENE_TIERS = [
    {
        "name": "scene_breakfast_easy",
        "task": "breakfast_easy_pick_and_return_sa_3d",
        "objects": ["cereal", "banana"],
        "maxiter": 25, "popsize": 10, "yaw_steps": 15,
    },
    {
        "name": "scene_breakfast",
        "task": "breakfast_pick_and_return_sa_3d",
        "objects": ["cereal", "banana", "milk_carton"],
        "maxiter": 30, "popsize": 10, "yaw_steps": 15,
    },
    {
        "name": "scene_desk",
        "task": "desk_pick_and_return_sa_3d",
        "objects": ["mug", "stapler", "pen_cup"],
        "maxiter": 30, "popsize": 10, "yaw_steps": 15,
    },
    {
        "name": "scene_kitchen_prep",
        "task": "kitchen_pick_and_return_sa_3d",
        "objects": ["apple", "can", "bottle"],
        "maxiter": 30, "popsize": 10, "yaw_steps": 15,
    },
    {
        "name": "scene_meal_assembly",
        "task": "meal_pick_and_return_sa_3d",
        "objects": ["cereal", "banana", "apple", "can", "bottle"],
        "maxiter": 35, "popsize": 12, "yaw_steps": 12,
    },
    {
        "name": "scene_cluttered",
        "task": "cluttered_pick_and_return_sa_3d",
        "objects": ["red_block", "blue_block", "green_cylinder",
                    "yellow_block", "orange_cylinder", "purple_block",
                    "white_cylinder", "pink_block"],
        "maxiter": 40, "popsize": 12, "yaw_steps": 12,
    },
]

START_POS_3D = np.array([0.452, 0.160, 1.05])
TABLE_Z = 0.943  # resting z for small objects (table top + small margin)


def load_scene_yaml(scene_name):
    path = os.path.join(PROJECT_ROOT, "config", "scenes",
                        f"{scene_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["scene"]


def read_baseline_z(scene_name, object_name):
    """Read the baseline z-height for an object from its MuJoCo XML.

    We do this by loading the XML via HeadlessSim (already wraps
    MuJoCo) so we get the actual per-object resting z.
    """
    import mujoco
    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        return float(data.xpos[bid, 2])
    except Exception:
        return TABLE_Z


def build_scene_maps(scene_cfg, task_objects):
    """Extract half_extents, z_map, and mujoco_name map from the scene YAML.

    Returns ``(half_extents, z_map, mujoco_names)`` where ``mujoco_names``
    maps every object short_name (movable AND fixed) to its full
    mujoco body name. The optimizer needs the *full* map so it can
    teleport every object — including non-task fixed objects — into the
    oracle's internal model when running collision-aware feasibility
    checks.
    """
    half_extents = {}
    mujoco_names = {}
    for obj in scene_cfg["objects"]:
        name = obj["short_name"]
        he = obj.get("half_extents", [0.03, 0.03, 0.03])
        half_extents[name] = (he[0], he[1])
        mujoco_names[name] = obj["mujoco_name"]
    # Resolve z from the MuJoCo XML's resting state
    scene_name = scene_cfg["name"]
    z_map = {}
    for name in task_objects:
        mname = mujoco_names.get(name, name)
        z_map[name] = read_baseline_z(scene_name, mname)
    return half_extents, z_map, mujoco_names


def save_optimized_scene(result, scene_cfg, task_objects, suffix="se3_optimized"):
    """Write the optimized scene YAML alongside the originals."""
    out_name = f"{scene_cfg['name']}_{suffix}"
    out_path = os.path.join(PROJECT_ROOT, "config", "scenes",
                            f"{out_name}.yaml")

    # Copy scene_cfg and update movable object positions (not yaws in the
    # YAML — the yaws are saved in the sidecar JSON because the existing
    # scene YAML format doesn't have a yaw field).
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated subset of scene names to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Shorter DE / yaw steps for smoke testing")
    args = parser.parse_args()

    only = None
    if args.only:
        only = set(s.strip() for s in args.only.split(","))

    lib = GraspLibrary.load(
        os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml"))

    results = {}
    out_dir = os.path.join(PROJECT_ROOT, "results", "se3_optimize")
    os.makedirs(out_dir, exist_ok=True)

    for tier in SCENE_TIERS:
        scene_name = tier["name"]
        if only is not None and scene_name not in only:
            continue

        print(f"\n{'='*60}\n  {scene_name}\n{'='*60}")
        scene_cfg = load_scene_yaml(scene_name)
        half_ext, z_map, mujoco_names = build_scene_maps(
            scene_cfg, tier["objects"])
        oracle = ReachabilityOracle(
            os.path.join(PROJECT_ROOT, "src", "assets", f"{scene_name}.xml"),
            pos_tol=0.02, rot_tol_deg=20.0, max_iters=30, n_seeds=3)

        # Build fixed_positions: non-task objects whose positions come
        # from the baseline scene XML. Included in overlap/separation
        # checks so movable objects can't land on top of fixed objects.
        fixed_positions = {}
        for obj in scene_cfg["objects"]:
            name = obj["short_name"]
            if name not in tier["objects"]:
                mname = mujoco_names.get(name, name)
                z = read_baseline_z(scene_name, mname)
                # Read xy from the MuJoCo model (already loaded by
                # read_baseline_z). Quick re-read for xy.
                import mujoco as _mj
                _m = _mj.MjModel.from_xml_path(
                    os.path.join(PROJECT_ROOT, "src", "assets",
                                 f"{scene_name}.xml"))
                _d = _mj.MjData(_m)
                if _m.nkey > 0:
                    _mj.mj_resetDataKeyframe(_m, _d, 0)
                _mj.mj_forward(_m, _d)
                try:
                    _bid = _mj.mj_name2id(
                        _m, _mj.mjtObj.mjOBJ_BODY, mname)
                    fixed_positions[name] = np.array([
                        float(_d.xpos[_bid, 0]),
                        float(_d.xpos[_bid, 1]),
                        float(_d.xpos[_bid, 2])])
                except Exception:
                    pass
        if fixed_positions:
            print(f"  Fixed objects: {list(fixed_positions.keys())}")

        maxiter = max(5, tier["maxiter"] // 4) if args.quick else tier["maxiter"]
        popsize = max(4, tier["popsize"] // 2) if args.quick else tier["popsize"]
        yaw_steps = max(5, tier["yaw_steps"] // 2) if args.quick else tier["yaw_steps"]

        t0 = time.time()
        result = optimize_scene(
            tier["objects"], half_ext, z_map, lib, oracle, START_POS_3D,
            sigma_v=0.5, sigma_w=0.5, lambda_R=0.04,
            maxiter=maxiter, popsize=popsize, seed=args.seed,
            yaw_steps=yaw_steps, verbose=True,
            mujoco_names=mujoco_names,
            check_arm_collisions=True,
            fixed_positions=fixed_positions,
            objective="trajectory_margin")
        dt = time.time() - t0

        out_path = save_optimized_scene(
            result, scene_cfg, tier["objects"], suffix="se3_optimized")
        print(f"  Wrote: {out_path}")

        json_payload = {
            "scene": scene_name,
            "task_objects": tier["objects"],
            "positions": {k: v.tolist() for k, v in result["positions"].items()},
            "yaws": result["yaws"],
            "slack": result["slack"],
            "feasibility": result["feasibility"],
            "eval_count": result["eval_count"],
            "elapsed_s": dt,
            "optimizer": {"maxiter": maxiter, "popsize": popsize,
                          "yaw_steps": yaw_steps, "seed": args.seed},
        }
        json_path = os.path.join(out_dir, f"{scene_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_payload, f, indent=2)
        results[scene_name] = json_payload
        print(f"  slack={result['slack']:.3f} "
              f"feasible={result['feasibility']['feasible']} "
              f"time={dt:.1f}s")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, r in results.items():
        print(f"  {name}: slack={r['slack']:.2f} "
              f"feas={r['feasibility']['feasible']} "
              f"{r['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
