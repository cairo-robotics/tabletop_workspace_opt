#!/usr/bin/env python3
"""Render paper figures for SE(3) MAP-Elites optimized scenes.

Loads each `scene_<env>_se3_me_optimized.yaml`, teleports every object
(with `position` + `yaw`) in the matching baseline MuJoCo XML, holds the
arm at the scene's home keyframe, settles physics, and renders a
front-tilted PNG matching `render_scene_figure.py`'s camera to
`results/figures/scene_<env>_optimized.png`.

Usage:
    MUJOCO_GL=osmesa python3 scripts/render/render_me_optimized_figures.py
"""
import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from envopt.grasp_visualization import load_optimized_scene
from envopt.se3_utils import yaw_to_quat


ENVS = [
    "scene_breakfast_easy",
    "scene_breakfast",
    "scene_desk",
    "scene_kitchen_prep",
    "scene_meal_assembly",
    "scene_cluttered",
]

LOOKAT = (0.55, 0.0, 0.95)
DISTANCE = 1.3
AZIMUTH = 180.0
ELEVATION = -50.0
SETTLE_STEPS = 200


def teleport_body(model, data, mujoco_name, pos, yaw) -> bool:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
    if bid < 0:
        return False
    jid = model.body_jntadr[bid]
    if jid < 0:
        return False
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr:qadr + 3] = pos
    data.qpos[qadr + 3:qadr + 7] = yaw_to_quat(yaw)
    vadr = model.jnt_dofadr[jid]
    data.qvel[vadr:vadr + 6] = 0
    return True


def render_one(env, output_path, width=800, height=600):
    yaml_path = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{env}_se3_me_optimized.yaml")
    xml_path = os.path.join(PROJECT_ROOT, "src", "assets", f"{env}.xml")
    if not os.path.exists(yaml_path):
        print(f"SKIP: {yaml_path} missing")
        return False
    if not os.path.exists(xml_path):
        print(f"SKIP: {xml_path} missing")
        return False

    scene = load_optimized_scene(yaml_path)

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    failed = []
    for obj in scene.objects:
        if not teleport_body(model, data, obj.mujoco_name, obj.position, obj.yaw):
            failed.append(obj.mujoco_name)
    mujoco.mj_forward(model, data)

    # Hold the arm at its home pose via position actuators.
    if model.nu > 0:
        for i in range(min(model.nu, model.nq)):
            joint_id = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[joint_id]
            data.ctrl[i] = data.qpos[qadr]
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = LOOKAT
    camera.distance = DISTANCE
    camera.azimuth = AZIMUTH
    camera.elevation = ELEVATION
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    Image.fromarray(pixels).save(output_path)
    print(f"Saved {output_path}  (objects teleported: {len(scene.objects)}"
          + (f", failed: {failed}" if failed else "") + ")")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             "results/figures"))
    parser.add_argument("--env", type=str, default=None,
                        help="Only render a single env (e.g. scene_desk).")
    parser.add_argument("-W", "--width", type=int, default=800)
    parser.add_argument("-H", "--height", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    envs = [args.env] if args.env else ENVS
    for env in envs:
        out = os.path.join(args.out_dir, f"{env}_optimized.png")
        render_one(env, out, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
