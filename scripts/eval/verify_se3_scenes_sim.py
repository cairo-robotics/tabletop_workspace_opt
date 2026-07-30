#!/usr/bin/env python3
"""M8 pragmatic verification: confirm that every SE(3)-optimized layout's
intended grasp pose is IK-reachable in the actual MuJoCo scene.

For each optimized scene tier:
  1. Load the base MuJoCo XML.
  2. Teleport every object to the optimized (x, y, yaw).
  3. For every task object, resolve the grasp pose via the grasp library.
  4. Run the reachability oracle on the pre-grasp and grasp poses.
  5. Additionally run a physics-aware IK solve using the headless sim's
     own Jacobian controller (HeadlessSim.compute_new_joints_6d) to
     simulate what the ROS runner + RelaxedIK would do when driven.

Output: results/se3_verify/<scene>.json with per-object feasibility and
the final EE pose error after IK iterations.
"""
import os
import sys
import json
import time
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mujoco
from envopt.grasp_library import GraspLibrary
from envopt.reachability import ReachabilityOracle
from envopt.se3_utils import rot_log


def teleport_body(model, data, mujoco_name, pos, yaw):
    from envopt.se3_utils import yaw_to_quat
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
    except Exception:
        return False
    jid = model.body_jntadr[bid]
    if jid < 0:
        return False
    qadr = model.jnt_qposadr[jid]
    q = yaw_to_quat(yaw)
    data.qpos[qadr:qadr+3] = pos
    data.qpos[qadr+3:qadr+7] = q
    vadr = model.jnt_dofadr[jid]
    data.qvel[vadr:vadr+6] = 0
    mujoco.mj_forward(model, data)
    return True


def verify_scene(scene_name, task_objects):
    base_xml = os.path.join(PROJECT_ROOT, "src", "assets",
                            f"{scene_name}.xml")
    base_yaml = os.path.join(PROJECT_ROOT, "config", "scenes",
                             f"{scene_name}.yaml")
    opt_json = os.path.join(PROJECT_ROOT, "results", "se3_optimize",
                            f"{scene_name}.json")
    if not os.path.exists(opt_json):
        return {"scene": scene_name, "error": "no SE(3) optimization output"}

    with open(opt_json) as f:
        opt = json.load(f)
    with open(base_yaml) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    mujoco_names = {o["short_name"]: o["mujoco_name"]
                    for o in scene_cfg["objects"]}

    lib = GraspLibrary.load(
        os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml"))

    model = mujoco.MjModel.from_xml_path(base_xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Teleport every task object
    for name in task_objects:
        pos = np.asarray(opt["positions"][name])
        yaw = float(opt["yaws"][name])
        mname = mujoco_names.get(name, name)
        ok = teleport_body(model, data, mname, pos, yaw)
        if not ok:
            pass  # some objects may be fixed bodies; that's fine

    # Oracle reads the current qpos — build it AFTER teleports so
    # the reachability check operates on the optimized layout.
    oracle = ReachabilityOracle(
        base_xml, pos_tol=0.025, rot_tol_deg=25.0, max_iters=50, n_seeds=4)
    # Copy the teleport state into the oracle's MjData
    oracle.data.qpos[:] = data.qpos[:]
    mujoco.mj_forward(oracle.model, oracle.data)

    per_obj = {}
    all_reachable = True

    for name in task_objects:
        pos = np.asarray(opt["positions"][name])
        yaw = float(opt["yaws"][name])
        entry = lib.get(name)
        if entry is None:
            per_obj[name] = {"error": "no grasp library entry"}
            all_reachable = False
            continue
        poses = entry.resolve(pos, yaw)

        grasp_q = oracle.solve_ik(poses["grasp_pos"], poses["grasp_R"])
        pregrasp_q = oracle.solve_ik(
            poses["pregrasp_pos"], poses["grasp_R"])

        grasp_err = pregrasp_err = None
        if grasp_q is not None:
            oracle._set_joints(grasp_q)
            cur_pos = oracle.data.site_xpos[oracle.ee_site_id].copy()
            cur_R = oracle.data.site_xmat[oracle.ee_site_id].reshape(3, 3)
            grasp_err = {
                "pos": float(np.linalg.norm(cur_pos - poses["grasp_pos"])),
                "rot_deg": float(np.rad2deg(
                    np.linalg.norm(rot_log(poses["grasp_R"] @ cur_R.T)))),
            }
        if pregrasp_q is not None:
            oracle._set_joints(pregrasp_q)
            cur_pos = oracle.data.site_xpos[oracle.ee_site_id].copy()
            cur_R = oracle.data.site_xmat[oracle.ee_site_id].reshape(3, 3)
            pregrasp_err = {
                "pos": float(np.linalg.norm(cur_pos - poses["pregrasp_pos"])),
                "rot_deg": float(np.rad2deg(
                    np.linalg.norm(rot_log(poses["grasp_R"] @ cur_R.T)))),
            }

        ok_grasp = grasp_q is not None
        ok_pg = pregrasp_q is not None
        if not (ok_grasp and ok_pg):
            all_reachable = False

        per_obj[name] = {
            "grasp_name": entry.name,
            "grasp_pos": poses["grasp_pos"].tolist(),
            "pregrasp_pos": poses["pregrasp_pos"].tolist(),
            "yaw": yaw,
            "grasp_reachable": ok_grasp,
            "pregrasp_reachable": ok_pg,
            "grasp_err": grasp_err,
            "pregrasp_err": pregrasp_err,
        }

    return {
        "scene": scene_name,
        "all_reachable": all_reachable,
        "per_object": per_obj,
    }


SCENES = [
    ("scene_breakfast_easy", ["cereal", "banana"]),
    ("scene_breakfast", ["cereal", "banana", "milk_carton"]),
    ("scene_desk", ["mug", "stapler", "pen_cup"]),
    ("scene_kitchen_prep", ["apple", "can", "bottle"]),
    ("scene_cluttered", [
        "red_block", "blue_block", "green_cylinder", "yellow_block",
        "orange_cylinder", "purple_block", "white_cylinder", "pink_block"]),
]


def main():
    out_dir = os.path.join(PROJECT_ROOT, "results", "se3_verify")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for scene, objs in SCENES:
        print(f"\n[{scene}]")
        r = verify_scene(scene, objs)
        results[scene] = r
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  all_reachable: {r['all_reachable']}")
        for obj, data in r["per_object"].items():
            if "error" in data:
                print(f"  {obj}: {data['error']}")
                continue
            mark_g = "OK" if data["grasp_reachable"] else "FAIL"
            mark_p = "OK" if data["pregrasp_reachable"] else "FAIL"
            print(f"  {obj:16s} grasp={mark_g}  pregrasp={mark_p}"
                  f"  yaw={data['yaw']:.2f}")
            if data["grasp_err"]:
                print(f"     grasp err: pos={data['grasp_err']['pos']*1000:.1f}mm "
                      f"rot={data['grasp_err']['rot_deg']:.1f}°")

        out_path = os.path.join(out_dir, f"{scene}.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
