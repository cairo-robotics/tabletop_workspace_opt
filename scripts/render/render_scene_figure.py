#!/usr/bin/env python3
"""Render front-tilted MuJoCo screenshots for paper figures.

Resets the robot to its 'home' keyframe (arm folded up out of the
way) and renders from a front-tilted camera so all objects on the
table are visible.

Usage:
    MUJOCO_GL=osmesa python3 scripts/render/render_scene_figure.py \
        src/assets/scene_breakfast_easy.xml \
        -o results/figures/scene_breakfast_easy.png
"""
import argparse
import os
import mujoco
import numpy as np
from PIL import Image


def render(model_path, output_path, width=800, height=600,
           lookat=(0.55, 0.0, 0.95), distance=1.3,
           azimuth=180.0, elevation=-50.0,
           settle_steps=200):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    # Reset to the 'home' keyframe so the arm is in its folded pose
    # (otherwise the gripper hangs down and blocks the table view).
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    # Hold the arm in the home pose during physics by feeding the
    # home joint angles as actuator targets (Sawyer uses position
    # actuators with affine bias, so ctrl = desired joint angle).
    if model.nu > 0:
        for i in range(min(model.nu, model.nq)):
            joint_id = model.actuator_trnid[i, 0]
            qadr = model.jnt_qposadr[joint_id]
            data.ctrl[i] = data.qpos[qadr]
    # Step physics so any floating objects settle onto the table.
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation

    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    Image.fromarray(pixels).save(output_path)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-W", "--width", type=int, default=800)
    parser.add_argument("-H", "--height", type=int, default=600)
    parser.add_argument("--lookat", type=float, nargs=3,
                        default=[0.55, 0.0, 0.95])
    parser.add_argument("--distance", type=float, default=1.3)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--elevation", type=float, default=-50.0)
    parser.add_argument("--settle-steps", type=int, default=200)
    args = parser.parse_args()
    render(args.model, args.output, args.width, args.height,
           tuple(args.lookat), args.distance, args.azimuth, args.elevation,
           args.settle_steps)


if __name__ == "__main__":
    main()
