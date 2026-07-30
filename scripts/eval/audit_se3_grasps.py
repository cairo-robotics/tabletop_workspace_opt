#!/usr/bin/env python3
"""Audit grasps for every SE(3)-optimized scene.

For each (scene, pick_object) pair this script checks:

1. Bounding-box validity — does the resolved grasp tip lie inside (or
   on the surface of) the object's bounding box, and is the gripper
   opening axis roughly perpendicular to one of the object's principal
   extents? Pure geometry — no IK.

2. IK reachability — does ReachabilityOracle.solve_ik converge for the
   grasp pose and the pre-grasp pose?

3. Self-collision and table-collision — at the IK solution, does the
   robot arm/gripper collide with itself OR with the table? OTHER
   objects on the table are intentionally ignored, per request.

The motivation for this auditor: the current SE(3) workspace optimizer
only checks (a) IK convergence and (b) the 2D projection of the
approach segment against other-object footprints. It does NOT check
collisions of the arm geometry against anything, and the grasp library
itself is hand-authored with no validity check.

Outputs:
  - results/se3_audit/grasp_audit.json with full per-(scene, object)
    detail.
  - A printed summary table.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mujoco
import yaml

from envopt.grasp_library import GraspLibrary
from envopt.reachability import ReachabilityOracle
from envopt.se3_utils import yaw_to_quat
from experiments.se3_catalog import grasp_audit_scenes


# Sawyer arm + gripper body names. Anything whose body is in this set
# (or has an arm body as an ancestor up to the world root, but here all
# arm/gripper geoms are direct children of these named bodies) is
# treated as part of the robot.
ARM_BODY_NAMES = {
    "pedestal", "base", "head",
    "right_l0", "right_l1", "right_l2", "right_l3",
    "right_l4", "right_l5", "right_l6",
    "gripper_body", "gripper_finger_left", "gripper_finger_right",
}

TABLE_BODY_NAMES = {"tablelink"}

# Bounding-box validity tolerances
BBOX_SURFACE_TOL = 0.015   # fingertip can be up to 1.5 cm outside bbox
BBOX_INTERIOR_TOL = 0.05   # at least one axis must be within bbox interior + tol

# Distance from the EE site to the fingertip midpoint, measured along
# the EE site's +z axis (the gripper tool axis). Empirically determined
# from src/assets/sawyer-gripper.xml: gripper_body site `end_effector`
# at z=0.0545, fingerpad geom centers at body-frame z=0.084 →
# fingertip midpoint is +0.0295 m along EE +z. For a top-down grasp
# (EE z points world -z), this means the fingertips are 29.5 mm BELOW
# the EE site in world space.
FINGERTIP_EE_OFFSET = np.array([0.0, 0.0, 0.0295])


# ---------------------------------------------------------------------------
# Geom categorization
# ---------------------------------------------------------------------------

def categorize_geoms(model) -> Dict[int, str]:
    """Map every geom id to one of {'arm', 'table', 'object', 'other'}.

    Object geoms are identified by walking up the body tree until we
    find a body with a free joint (the object root) — its descendants
    are 'object'. Bodies in ARM_BODY_NAMES are 'arm', TABLE_BODY_NAMES
    are 'table', and the rest are 'other'.
    """
    cat = {}
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname in ARM_BODY_NAMES:
            cat[gid] = "arm"
        elif bname in TABLE_BODY_NAMES:
            cat[gid] = "table"
        else:
            # Walk up to see if this body or an ancestor has a free joint.
            cur = bid
            is_object = False
            while cur > 0:
                jadr = model.body_jntadr[cur]
                if jadr >= 0 and model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                    is_object = True
                    break
                cur = model.body_parentid[cur]
            cat[gid] = "object" if is_object else "other"
    return cat


# ---------------------------------------------------------------------------
# Layout loading and teleport
# ---------------------------------------------------------------------------

def load_optimized_layout(scene_yaml_path: str):
    """Return list of dicts with short_name, mujoco_name, position, yaw, half_extents."""
    with open(scene_yaml_path) as f:
        cfg = yaml.safe_load(f)["scene"]
    objs = []
    for obj in cfg["objects"]:
        # Some entries have only short_name (no position) — those are fixed.
        objs.append({
            "short_name": obj["short_name"],
            "mujoco_name": obj["mujoco_name"],
            "position": np.asarray(
                obj.get("position", [0.0, 0.0, 0.94]), dtype=float),
            "yaw": float(obj.get("yaw", 0.0)),
            "half_extents": np.asarray(obj["half_extents"], dtype=float),
        })
    return cfg["name"], objs


def teleport_body(model, data, mujoco_name, pos, yaw) -> bool:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
    if bid < 0:
        return False
    jid = model.body_jntadr[bid]
    if jid < 0:
        return False
    qadr = model.jnt_qposadr[jid]
    q = yaw_to_quat(yaw)
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = q
    return True


# ---------------------------------------------------------------------------
# Bounding-box check
# ---------------------------------------------------------------------------

def bbox_check(grasp_pos, grasp_R, obj_pos, obj_yaw, half_extents):
    """Verify the gripper FINGERTIP is at the object surface.

    The library's `grasp_pos` is the target for the EE site, not the
    fingertip — fingers extend FINGERTIP_EE_OFFSET further along the
    EE +z axis. We compute the fingertip position in world space, then
    transform it into the object body frame and check that it lies
    inside (or within BBOX_SURFACE_TOL of) the bounding box.

    Returns dict with both EE-site and fingertip overshoots so failures
    are interpretable.
    """
    # Fingertip position in world frame
    fingertip_world = grasp_pos + grasp_R @ FINGERTIP_EE_OFFSET

    # Transform points into object body frame: undo yaw rotation around z.
    cos_y, sin_y = np.cos(-obj_yaw), np.sin(-obj_yaw)
    Rz_inv = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    fingertip_body = Rz_inv @ (fingertip_world - obj_pos)
    ee_body = Rz_inv @ (grasp_pos - obj_pos)

    # Distance to bbox surface (signed; negative inside)
    finger_overshoots = np.abs(fingertip_body) - half_extents
    ee_overshoots = np.abs(ee_body) - half_extents
    max_finger_overshoot = float(np.max(finger_overshoots))
    max_ee_overshoot = float(np.max(ee_overshoots))

    in_bbox = max_finger_overshoot <= BBOX_SURFACE_TOL

    # Gripper opening axis = the y-axis of the grasp frame (parallel
    # jaws close along the gripper's local y). Verify it is perpendicular
    # to one of the object's principal axes — i.e. the |dot| with the
    # nearest body axis should be small. Equivalently, the gripper
    # opening axis should NOT be aligned with the object's longest
    # extent if the fingers are to straddle it.
    grip_y_world = grasp_R[:, 1]
    grip_y_body = Rz_inv @ grip_y_world
    dots = np.abs(grip_y_body)  # |dot| with body x, y, z

    # The "best" alignment we want is the gripper-y aligned with whichever
    # body axis the fingers should close ACROSS — typically the smallest
    # half-extent direction (so the fingers close on the narrower
    # dimension). max(dots) close to 1 means the gripper opens along
    # exactly one body axis.
    alignment = float(np.max(dots))

    ok = in_bbox and alignment >= 0.85
    details = ""
    if not in_bbox:
        details = (f"fingertip {max_finger_overshoot*1000:+.0f}mm "
                   f"outside bbox (EE site {max_ee_overshoot*1000:+.0f}mm)")
    elif alignment < 0.85:
        details = f"gripper axis weakly aligned (|dot|_max={alignment:.2f})"
    return {
        "ok": bool(ok),
        "in_bbox": bool(in_bbox),
        "max_fingertip_overshoot_mm": max_finger_overshoot * 1000,
        "max_ee_overshoot_mm": max_ee_overshoot * 1000,
        "gripper_axis_alignment": alignment,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Collision check (self + table only)
# ---------------------------------------------------------------------------

def arm_set_joints(model, data, q_arm):
    for i in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_j{i}")
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = q_arm[i]


def collision_at_pose(model, data, q_arm, geom_cat) -> Dict:
    """Set the arm to q_arm, run kinematics + collision, return arm-vs-(self|table) contacts.

    Other-object contacts are silently ignored. Returns:
        n_self: number of arm-arm contacts
        n_table: number of arm-table contacts
        ok: True iff both are 0
        details: list of (geom1_body, geom2_body) for failing pairs
    """
    arm_set_joints(model, data, q_arm)
    mujoco.mj_forward(model, data)
    # mj_forward also runs collision detection (it calls mj_collision
    # internally as part of mj_fwdPosition), so data.ncon is populated.
    n_self = 0
    n_table = 0
    details = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        c1 = geom_cat.get(g1, "other")
        c2 = geom_cat.get(g2, "other")
        b1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1]) or "?"
        b2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2]) or "?"
        if c1 == "arm" and c2 == "arm":
            n_self += 1
            details.append(("self", b1, b2))
        elif (c1 == "arm" and c2 == "table") or (c1 == "table" and c2 == "arm"):
            n_table += 1
            details.append(("table", b1, b2))
    return {
        "n_self": n_self,
        "n_table": n_table,
        "ok": n_self == 0 and n_table == 0,
        "details": details[:6],   # cap to avoid noise in JSON
    }


# ---------------------------------------------------------------------------
# Per-scene audit
# ---------------------------------------------------------------------------

def audit_scene(scene_yaml_path: str, base_xml_path: str,
                grasp_lib: GraspLibrary, pick_objects: List[str]):
    scene_name, all_objs = load_optimized_layout(scene_yaml_path)

    model = mujoco.MjModel.from_xml_path(base_xml_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    for obj in all_objs:
        teleport_body(model, data, obj["mujoco_name"], obj["position"], obj["yaw"])

    geom_cat = categorize_geoms(model)
    oracle = ReachabilityOracle(
        base_xml_path, pos_tol=0.01, rot_tol_deg=10.0,
        max_iters=120, n_seeds=8)
    # Mirror the teleported state into the oracle so its IK sees the
    # right object positions for any contact-aware step we add later.
    oracle.data.qpos[:] = data.qpos[:]
    mujoco.mj_forward(oracle.model, oracle.data)

    obj_by_name = {o["short_name"]: o for o in all_objs}
    results = {}

    for short in pick_objects:
        if short not in obj_by_name:
            results[short] = {"status": "missing_from_scene"}
            continue
        obj = obj_by_name[short]
        entry = grasp_lib.get(short)
        if entry is None:
            results[short] = {"status": "missing_grasp_library_entry"}
            continue

        poses = entry.resolve(obj["position"], obj["yaw"])
        grasp_pos = poses["grasp_pos"]
        grasp_R = poses["grasp_R"]
        pregrasp_pos = poses["pregrasp_pos"]

        bb = bbox_check(grasp_pos, grasp_R,
                        obj["position"], obj["yaw"], obj["half_extents"])

        # IK
        q_grasp = oracle.solve_ik(grasp_pos, grasp_R)
        q_pregrasp = oracle.solve_ik(pregrasp_pos, grasp_R)

        ik_grasp_ok = q_grasp is not None
        ik_pregrasp_ok = q_pregrasp is not None

        col_grasp = None
        col_pregrasp = None
        if ik_grasp_ok:
            col_grasp = collision_at_pose(model, data, q_grasp, geom_cat)
        if ik_pregrasp_ok:
            col_pregrasp = collision_at_pose(model, data, q_pregrasp, geom_cat)

        # Restore the home pose so the next object's collision check
        # starts from the same baseline.
        if model.nkey > 0:
            home_qpos = model.key_qpos[0].copy()
            for i in range(7):
                jid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"right_j{i}")
                addr = model.jnt_qposadr[jid]
                data.qpos[addr] = home_qpos[addr]
        mujoco.mj_forward(model, data)

        verdict = "ok"
        if not bb["ok"]:
            verdict = "bbox_invalid"
        elif not ik_grasp_ok or not ik_pregrasp_ok:
            verdict = "ik_failed"
        elif (col_grasp and not col_grasp["ok"]) or \
             (col_pregrasp and not col_pregrasp["ok"]):
            verdict = "collision"

        results[short] = {
            "verdict": verdict,
            "bbox": bb,
            "ik_grasp": ik_grasp_ok,
            "ik_pregrasp": ik_pregrasp_ok,
            "collision_grasp": col_grasp,
            "collision_pregrasp": col_pregrasp,
            "grasp_name": entry.name,
            "grasp_pos": grasp_pos.tolist(),
            "object_pos": obj["position"].tolist(),
            "object_yaw": obj["yaw"],
        }

    return scene_name, results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Map SE(3) optimized scene → base scene XML and pick objects.
SCENES = grasp_audit_scenes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "results", "se3_audit"))
    parser.add_argument("--grasp-library", type=str,
                        default=os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    grasp_lib = GraspLibrary.load(args.grasp_library)

    all_results = {"meta": {"grasp_library": args.grasp_library}, "scenes": {}}

    print(f"\n{'='*100}")
    print(f"GRASP AUDIT — {len(SCENES)} scenes")
    print(f"{'='*100}")

    for scene_yaml_name, base_scene_name, pick_objects in SCENES:
        scene_yaml = os.path.join(
            PROJECT_ROOT, "config", "scenes", f"{scene_yaml_name}.yaml")
        base_xml = os.path.join(
            PROJECT_ROOT, "src", "assets", f"{base_scene_name}.xml")
        if not os.path.exists(scene_yaml):
            print(f"\nSKIP {scene_yaml_name}: yaml not found")
            continue
        if not os.path.exists(base_xml):
            print(f"\nSKIP {scene_yaml_name}: base XML not found")
            continue

        print(f"\n--- {scene_yaml_name} ---")
        scene_name, results = audit_scene(
            scene_yaml, base_xml, grasp_lib, pick_objects)
        all_results["scenes"][scene_name] = results

        for short, r in results.items():
            verdict = r.get("verdict", r.get("status", "?"))
            line = f"  {short:18s} {verdict:14s}"
            if "bbox" in r:
                bb = r["bbox"]
                line += (f"  bbox=({'OK' if bb['ok'] else 'BAD'}, "
                         f"finger={bb['max_fingertip_overshoot_mm']:+.0f}mm, "
                         f"ee={bb['max_ee_overshoot_mm']:+.0f}mm, "
                         f"align={bb['gripper_axis_alignment']:.2f})")
            if "ik_grasp" in r:
                line += f"  ik_g={'Y' if r['ik_grasp'] else 'N'}"
                line += f" ik_pg={'Y' if r['ik_pregrasp'] else 'N'}"
            if r.get("collision_grasp"):
                cg = r["collision_grasp"]
                line += f"  c_g[s={cg['n_self']},t={cg['n_table']}]"
            if r.get("collision_pregrasp"):
                cp = r["collision_pregrasp"]
                line += f"  c_pg[s={cp['n_self']},t={cp['n_table']}]"
            print(line)

    # Save JSON
    out_path = os.path.join(args.out_dir, "grasp_audit.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Aggregate summary
    print(f"\n{'='*100}")
    print(f"AGGREGATE SUMMARY")
    print(f"{'='*100}")
    print(f"  {'scene':>32s}  {'total':>5}  {'ok':>4}  {'bbox':>5}  {'ik':>4}  {'col':>4}")
    for scene_name, results in all_results["scenes"].items():
        n = len(results)
        n_ok = sum(1 for r in results.values()
                   if r.get("verdict") == "ok")
        n_bb = sum(1 for r in results.values()
                   if r.get("verdict") == "bbox_invalid")
        n_ik = sum(1 for r in results.values()
                   if r.get("verdict") == "ik_failed")
        n_col = sum(1 for r in results.values()
                    if r.get("verdict") == "collision")
        print(f"  {scene_name:>32s}  {n:>5d}  {n_ok:>4d}  "
              f"{n_bb:>5d}  {n_ik:>4d}  {n_col:>4d}")


if __name__ == "__main__":
    main()
