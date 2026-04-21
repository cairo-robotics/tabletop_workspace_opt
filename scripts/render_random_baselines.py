#!/usr/bin/env python3
"""Render figures of randomly sampled baseline layouts.

For each environment, loads the baseline scene XML, samples a valid
random placement for the pick objects (using the same constraints
as the numerical evaluation: min pairwise distance, robot exclusion,
table bounds), teleports the objects in the MuJoCo model, settles
physics, and saves a PNG.

Usage:
    MUJOCO_GL=osmesa python3 scripts/render_random_baselines.py
    MUJOCO_GL=osmesa python3 scripts/render_random_baselines.py --seed 0
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import yaml
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from run_sa_headless import generate_random_layouts


# Map environment to (scene name, task config)
ENVS = [
    ("scene_breakfast_easy",  "breakfast_easy_pick_and_return_sa.yaml"),
    ("scene_desk",            "desk_pick_and_return_sa.yaml"),
    ("scene_breakfast",       "breakfast_pick_and_return_sa.yaml"),
    ("scene_kitchen_prep",    "kitchen_pick_and_return_sa.yaml"),
    ("scene_meal_assembly",   "meal_pick_and_return_sa.yaml"),
    ("scene_cluttered",       "cluttered_pick_and_return_sa.yaml"),
]

DEFAULT_LOOKAT = (0.55, 0.0, 0.95)
DEFAULT_DISTANCE = 1.3
DEFAULT_AZIMUTH = 180.0
DEFAULT_ELEVATION = -50.0
SETTLE_STEPS = 200


def load_movable_and_fixed(scene_name, task_yaml_name):
    """Return (pick_object_names, {fixed_name: (x, y)}, mujoco_name_map,
    {fixed_name: (hx, hy)}, {movable_name: (hx, hy)})."""
    scene_yaml = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    task_yaml = os.path.join(
        PROJECT_ROOT, "config", "tasks", task_yaml_name)

    with open(scene_yaml) as f:
        scene_cfg = yaml.safe_load(f)["scene"]
    with open(task_yaml) as f:
        task_cfg = yaml.safe_load(f)["task"]

    pick_objects = list(task_cfg["pick_objects"])
    mujoco_map = {o["short_name"]: o["mujoco_name"]
                  for o in scene_cfg["objects"]}

    # Read the baseline XML to get the current xy of fixed objects
    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    fixed_positions = {}
    fixed_half_extents = {}
    movable_half_extents = {}
    for obj in scene_cfg["objects"]:
        short = obj["short_name"]
        he = obj.get("half_extents")
        he_xy = tuple(he[:2]) if he and len(he) >= 2 else None
        if short in pick_objects:
            if he_xy is not None:
                movable_half_extents[short] = he_xy
            continue
        mname = obj["mujoco_name"]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mname)
        pos = data.xpos[bid][:2].copy()
        fixed_positions[short] = pos
        if he_xy is not None:
            fixed_half_extents[short] = he_xy

    return (pick_objects, fixed_positions, mujoco_map,
            fixed_half_extents, movable_half_extents)


def teleport_object(model, data, mujoco_name, xy):
    """Move a freejoint body in xy, preserving z and orientation."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
    if bid < 0:
        return
    jid = model.body_jntadr[bid]
    if jid < 0:
        return
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr] = xy[0]
    data.qpos[qadr + 1] = xy[1]
    # qpos[qadr+2] is z; leave alone
    # quaternion qpos[qadr+3 .. qadr+6] reset to identity
    data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
    vadr = model.jnt_dofadr[jid]
    data.qvel[vadr:vadr + 6] = 0


def render_random_baseline(scene_name, task_yaml_name, seed,
                           output_path, width=800, height=600,
                           max_attempts=10, displace_thresh=0.08):
    (pick_objects, fixed_positions, mujoco_map,
     fixed_he, movable_he) = load_movable_and_fixed(
        scene_name, task_yaml_name)

    # Try seeds seed, seed+100, seed+200, ... until post-settle displacement
    # stays under threshold (i.e., the arm didn't knock an object off).
    for attempt in range(max_attempts):
        trial_seed = seed + 100 * attempt
        layouts = generate_random_layouts(
            n_layouts=1, movable_names=pick_objects,
            fixed_positions=fixed_positions, seed=trial_seed,
            fixed_half_extents=fixed_he,
            movable_half_extents=movable_he)
        if not layouts:
            continue
        layout = layouts[0]
        if attempt > 0:
            print(f"  retrying {scene_name} with seed {trial_seed}")
        break
    else:
        print(f"WARN: could not generate a valid layout for {scene_name}")
        return

    xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Teleport pick objects to the sampled random positions
    for short in pick_objects:
        if short not in layout:
            continue
        mname = mujoco_map.get(short)
        if mname:
            teleport_object(model, data, mname, layout[short])
    mujoco.mj_forward(model, data)

    # Hold the arm in home pose during physics settling
    if model.nu > 0:
        for i in range(min(model.nu, model.nq)):
            joint_id = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[joint_id]
            data.ctrl[i] = data.qpos[qadr]
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    # Sanity check: if any pick object was displaced far from its sampled
    # xy (i.e. the arm knocked it), try a new seed.
    max_disp = 0.0
    for short in pick_objects:
        mname = mujoco_map.get(short)
        if not mname:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mname)
        if bid < 0:
            continue
        actual_xy = data.xpos[bid][:2]
        target_xy = np.asarray(layout[short])
        disp = float(np.linalg.norm(actual_xy - target_xy))
        if disp > max_disp:
            max_disp = disp
    if max_disp > displace_thresh:
        print(f"  seed {trial_seed} displaced {max_disp:.3f} m "
              f"(>{displace_thresh}); retrying")
        return render_random_baseline(
            scene_name, task_yaml_name, seed + 100 * (attempt + 1),
            output_path, width=width, height=height,
            max_attempts=max_attempts - attempt - 1,
            displace_thresh=displace_thresh)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = DEFAULT_LOOKAT
    camera.distance = DEFAULT_DISTANCE
    camera.azimuth = DEFAULT_AZIMUTH
    camera.elevation = DEFAULT_ELEVATION
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    Image.fromarray(pixels).save(output_path)
    print(f"Saved {output_path}")
    print(f"  Random layout: " +
          ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})"
                    for n, xy in layout.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (each env uses seed + env_index)")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "docs/latex/figures"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for i, (scene, task) in enumerate(ENVS):
        out = os.path.join(args.out_dir, f"{scene}_baseline.png")
        render_random_baseline(scene, task, seed=args.seed + i,
                               output_path=out)


if __name__ == "__main__":
    main()
