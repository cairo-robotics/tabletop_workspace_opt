#!/usr/bin/env python3
"""Run workspace optimization for all scenes and generate optimized scene XMLs.

This reads each scene config YAML, runs the intent separability optimizer,
and creates an optimized variant of the MuJoCo scene XML with the new positions.
"""

import sys
import os
import re
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "envopt"))

from workspace_optimizer import optimize_workspace, DEFAULT_START_POS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES_DIR = os.path.join(PROJECT_ROOT, "config", "scenes")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "src", "assets")


def load_scene_positions(scene_xml_path, object_mujoco_names):
    """Extract object (x, y, z) positions from a scene XML keyframe."""
    with open(scene_xml_path) as f:
        xml_content = f.read()

    # Parse keyframe qpos to get object positions
    # Format: 7 arm joints + 2 gripper + N objects x 7 (pos3 + quat4)
    match = re.search(r'<key\s+name="home"\s+qpos="([^"]+)"', xml_content)
    if not match:
        raise ValueError(f"No keyframe found in {scene_xml_path}")

    qpos_str = match.group(1).strip()
    qpos = np.array([float(x) for x in qpos_str.split()])

    # Skip 9 (7 arm + 2 gripper), then 7 per object (pos3 + quat4)
    robot_dof = 9
    positions = {}
    for i, name in enumerate(object_mujoco_names):
        idx = robot_dof + i * 7
        positions[name] = qpos[idx:idx + 3]

    return positions


def update_scene_xml(original_path, output_path, short_to_mujoco,
                     optimized_xy, original_positions):
    """Create a new scene XML with optimized (x, y) positions.

    Only x, y are changed; z is preserved from the original.
    Updates both the <body> pos attributes and the keyframe qpos.
    """
    with open(original_path) as f:
        xml = f.read()

    for short_name, mujoco_name in short_to_mujoco.items():
        if short_name not in optimized_xy:
            continue

        new_x, new_y = optimized_xy[short_name]
        orig_z = original_positions[mujoco_name][2]

        # Update <body name="..." pos="x y z">
        body_pattern = rf'(<body\s+name="{re.escape(mujoco_name)}"\s+pos=")([^"]+)(")'
        match = re.search(body_pattern, xml)
        if match:
            xml = re.sub(body_pattern,
                         rf'\g<1>{new_x:.3f} {new_y:.3f} {orig_z:.3f}\3',
                         xml)

    # Update keyframe qpos
    match = re.search(r'(<key\s+name="home"\s+qpos=")([^"]+)(")', xml)
    if match:
        qpos = [float(x) for x in match.group(2).strip().split()]
        robot_dof = 9
        mujoco_names_ordered = list(short_to_mujoco.values())
        for i, mname in enumerate(mujoco_names_ordered):
            short = [k for k, v in short_to_mujoco.items() if v == mname][0]
            if short in optimized_xy:
                idx = robot_dof + i * 7
                qpos[idx] = optimized_xy[short][0]      # x
                qpos[idx + 1] = optimized_xy[short][1]  # y
                # z stays the same

        qpos_str = " ".join(f"{v:.3f}" if abs(v) > 0.0005 else "0" for v in qpos)

        # Format: 9 robot values on first line, then 7 per object per line
        lines = []
        vals_all = qpos_str.split()
        lines.append(" ".join(vals_all[:robot_dof]))
        for i in range(len(mujoco_names_ordered)):
            start = robot_dof + i * 7
            end = start + 7
            lines.append("      " + " ".join(vals_all[start:end]))
        formatted = lines[0] + "\n" + "\n".join(lines[1:])

        # Use a function for replacement to avoid regex backreference issues
        def keyframe_replacer(m):
            return m.group(1) + formatted + m.group(2)
        xml = re.sub(r'(<key\s+name="home"\s+qpos=")[^"]+(")',
                     keyframe_replacer, xml)

    # Update model name
    xml = re.sub(r'(<mujoco\s+model=")([^"]+)(")',
                 lambda m: f'{m.group(1)}{m.group(2)}_optimized{m.group(3)}',
                 xml, count=1)

    with open(output_path, 'w') as f:
        f.write(xml)

    print(f"  Written: {output_path}")


def optimize_scene(scene_name):
    """Run optimization for a single scene and create the optimized XML."""
    config_path = os.path.join(SCENES_DIR, f"{scene_name}.yaml")
    xml_path = os.path.join(ASSETS_DIR, f"{scene_name}.xml")
    output_path = os.path.join(ASSETS_DIR, f"{scene_name}_optimized.xml")

    if not os.path.exists(config_path):
        print(f"SKIP: No config at {config_path}")
        return None

    if not os.path.exists(xml_path):
        print(f"SKIP: No XML at {xml_path}")
        return None

    with open(config_path) as f:
        scene_cfg = yaml.safe_load(f)["scene"]

    object_names = [obj["short_name"] for obj in scene_cfg["objects"]]
    mujoco_names = {obj["short_name"]: obj["mujoco_name"] for obj in scene_cfg["objects"]}

    # Get original positions from XML
    orig_positions = load_scene_positions(xml_path, list(mujoco_names.values()))
    initial_xy = np.array([
        orig_positions[mujoco_names[name]][:2] for name in object_names
    ])

    print(f"\n{'=' * 60}")
    print(f"Optimizing: {scene_name} ({len(object_names)} objects)")
    print(f"Objects: {object_names}")
    print(f"{'=' * 60}")

    # Run optimization
    result = optimize_workspace(
        object_names=object_names,
        initial_positions=initial_xy,
        start_pos=DEFAULT_START_POS,
        T=50,
        step_size=0.01,
        maxiter=200,
        seed=42,
        method="differential_evolution",
        verbose=True,
    )

    # Build optimized (x, y) dict
    optimized_xy = {}
    for i, name in enumerate(object_names):
        optimized_xy[name] = result["optimized_positions"][i]

    print(f"\nOriginal positions:")
    for name in object_names:
        orig = orig_positions[mujoco_names[name]]
        print(f"  {name}: ({orig[0]:.3f}, {orig[1]:.3f}, {orig[2]:.3f})")

    print(f"\nOptimized positions (x, y only, z preserved):")
    for name in object_names:
        orig_z = orig_positions[mujoco_names[name]][2]
        new_x, new_y = optimized_xy[name]
        print(f"  {name}: ({new_x:.3f}, {new_y:.3f}, {orig_z:.3f})")

    # Create optimized scene XML
    update_scene_xml(xml_path, output_path, mujoco_names,
                     optimized_xy, orig_positions)

    # Also create an optimized scene config YAML
    opt_config_path = os.path.join(SCENES_DIR, f"{scene_name}_optimized.yaml")
    opt_cfg = yaml.safe_load(open(config_path))
    opt_cfg["scene"]["name"] = f"{scene_name}_optimized"
    with open(opt_config_path, 'w') as f:
        yaml.dump(opt_cfg, f, default_flow_style=False)
    print(f"  Written: {opt_config_path}")

    return result


if __name__ == "__main__":
    scenes = ["scene_breakfast", "scene_desk", "scene_kitchen_prep"]

    results = {}
    for scene in scenes:
        result = optimize_scene(scene)
        if result:
            results[scene] = result

    print(f"\n{'=' * 60}")
    print("All optimizations complete!")
    print(f"{'=' * 60}")
    for scene, result in results.items():
        print(f"  {scene}: slack {result['initial_slack']:.1f} -> {result['optimized_slack']:.1f} "
              f"(+{result['optimized_slack'] - result['initial_slack']:.1f})")
