#!/usr/bin/env python3
"""Minimal MCP-style helper: renders MuJoCo scenes to PNG for Claude Code."""

import mujoco
import numpy as np
from PIL import Image
import argparse
import os

def capture_mujoco_frame(model_path, output_path="/tmp/mujoco_frame.png",
                          width=640, height=480,
                          lookat=None, distance=None, azimuth=None,
                          elevation=None):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)

    # Set custom camera if specified
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    if lookat is not None:
        camera.lookat[:] = lookat
    if distance is not None:
        camera.distance = distance
    if azimuth is not None:
        camera.azimuth = azimuth
    if elevation is not None:
        camera.elevation = elevation

    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    img.save(output_path)
    print(f"Saved frame to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to MJCF XML file")
    parser.add_argument("-o", "--output", default="/tmp/mujoco_frame.png")
    parser.add_argument("-W", "--width", type=int, default=640)
    parser.add_argument("-H", "--height", type=int, default=480)
    parser.add_argument("--azimuth", type=float, default=None,
                        help="Camera azimuth angle (degrees)")
    parser.add_argument("--elevation", type=float, default=None,
                        help="Camera elevation angle (degrees)")
    parser.add_argument("--distance", type=float, default=None,
                        help="Camera distance from lookat point")
    parser.add_argument("--lookat", type=float, nargs=3, default=None,
                        help="Camera lookat point (x y z)")
    args = parser.parse_args()
    capture_mujoco_frame(args.model, args.output, args.width, args.height,
                          lookat=args.lookat, distance=args.distance,
                          azimuth=args.azimuth, elevation=args.elevation)