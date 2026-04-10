#!/usr/bin/env python3
"""Generate a MuJoCo scene XML from MAP-Elites optimized positions.

Reads the MAP-Elites best positions from the v2 results JSON, loads the
baseline scene XML, replaces the (x, y) positions of the optimized
objects in both body pos attributes and the keyframe qpos, and writes
a new XML + copies the matching scene YAML.

Usage:
    python3 scripts/write_me_scene_xml.py --scene desk
    python3 scripts/write_me_scene_xml.py --scene all
    python3 scripts/write_me_scene_xml.py --scene desk --seed-idx 1
"""
import sys
import os
import re
import json
import argparse
import shutil
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# Map env_key (from results JSON) to scene XML filename (baseline)
SCENE_MAP = {
    "breakfast_easy": "scene_breakfast_easy",
    "desk": "scene_desk",
    "breakfast": "scene_breakfast",
    "kitchen_prep": "scene_kitchen_prep",
    "meal_assembly": "scene_meal_assembly",
    "cluttered": "scene_cluttered",
}


RESULTS_FILES = {
    "separability": ["map_elites_v2_all_results.json",
                     "map_elites_all_results.json"],
    "legibility": ["map_elites_legibility_all.json",
                   "map_elites_legibility_easy_med.json",
                   "map_elites_legibility_hard.json"],
    "trajectory_margin": ["map_elites_trajectory_margin_all.json",
                          "map_elites_trajectory_margin_easy_med.json",
                          "map_elites_trajectory_margin_hard.json"],
}

SUFFIX_MAP = {
    "separability": "_me_optimized",
    "legibility": "_legibility_optimized",
    "trajectory_margin": "_trajectory_optimized",
}


def load_results(source="separability"):
    """Load results JSON for the given optimization source."""
    fnames = RESULTS_FILES.get(source, RESULTS_FILES["separability"])
    all_data = {}
    for fname in fnames:
        path = os.path.join(PROJECT_ROOT, "results", fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            all_data.update(data)
    if not all_data:
        raise FileNotFoundError(
            f"No {source} results found. Run eval_map_elites_tiers.py "
            f"--objective {source} first.")
    return all_data, source


def load_scene_yaml(scene_name: str) -> dict:
    """Load scene YAML config."""
    path = os.path.join(PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)["scene"]


def body_has_freejoint(xml: str, mujoco_name: str) -> bool:
    """Check if a body has a freejoint in the XML."""
    match = re.search(
        rf'<body\s+name="{re.escape(mujoco_name)}"[^>]*>(.*?)</body>',
        xml, re.DOTALL)
    return bool(match and "freejoint" in match.group(1))


def update_body_pos(xml: str, mujoco_name: str,
                    new_x: float, new_y: float) -> str:
    """Replace body pos attribute, preserving z."""
    pattern = rf'(<body\s+name="{re.escape(mujoco_name)}"\s+pos=")([^"]+)(")'
    match = re.search(pattern, xml)
    if not match:
        return xml
    old_pos = match.group(2).split()
    if len(old_pos) < 3:
        return xml
    old_z = old_pos[2]
    new_pos = f"{new_x:.3f} {new_y:.3f} {old_z}"
    return xml[:match.start(2)] + new_pos + xml[match.end(2):]


def update_keyframe_qpos(xml: str, scene_cfg: dict,
                         new_positions: dict) -> str:
    """Update keyframe qpos with new (x, y) for freejoint bodies.

    qpos layout: 9 robot DOF + 7 entries per freejoint body (pos + quat).
    """
    match = re.search(r'(<key\s+name="home"\s+qpos=")([^"]+)(")', xml)
    if not match:
        print("  Warning: no 'home' keyframe found, skipping qpos update")
        return xml

    qpos_str = match.group(2).strip()
    tokens = qpos_str.split()
    try:
        qpos = [float(t) for t in tokens]
    except ValueError:
        print("  Warning: failed to parse qpos, skipping")
        return xml

    # Walk through objects in YAML order; those with freejoints occupy
    # 7 qpos entries (x, y, z, qw, qx, qy, qz).
    freejoint_idx = 0
    for obj in scene_cfg["objects"]:
        short = obj["short_name"]
        mujoco_name = obj["mujoco_name"]
        if not body_has_freejoint(xml, mujoco_name):
            continue
        base = 9 + freejoint_idx * 7
        if base + 1 < len(qpos) and short in new_positions:
            new_x, new_y = new_positions[short]
            qpos[base] = float(new_x)
            qpos[base + 1] = float(new_y)
        freejoint_idx += 1

    # Rebuild qpos string preserving the original multi-line format if present
    if "\n" in qpos_str:
        # Recreate with 9 robot DOF on first line, 7 per object per line
        lines = [" ".join(f"{v:g}" for v in qpos[:9])]
        for i in range(9, len(qpos), 7):
            chunk = qpos[i:i+7]
            lines.append("      " + " ".join(f"{v:g}" for v in chunk))
        new_qpos_str = "\n".join(lines)
    else:
        new_qpos_str = " ".join(f"{v:g}" for v in qpos)

    return xml[:match.start(2)] + new_qpos_str + xml[match.end(2):]


def write_optimized_scene(env_key: str, results: dict, seed_idx: int = 0,
                          source: str = "separability") -> bool:
    """Generate optimized XML and YAML for one environment."""
    if env_key not in results:
        print(f"  ERROR: {env_key} not in results, skipping")
        return False

    baseline_scene = SCENE_MAP[env_key]
    suffix = SUFFIX_MAP.get(source, "_me_optimized")
    output_scene = f"{baseline_scene}{suffix}"
    baseline_xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{baseline_scene}.xml")
    output_xml_path = os.path.join(
        PROJECT_ROOT, "src", "assets", f"{output_scene}.xml")
    baseline_yaml_path = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{baseline_scene}.yaml")
    output_yaml_path = os.path.join(
        PROJECT_ROOT, "config", "scenes", f"{output_scene}.yaml")

    if not os.path.exists(baseline_xml_path):
        print(f"  ERROR: {baseline_xml_path} not found")
        return False

    # Extract ME best positions
    me_seeds = results[env_key].get("map_elites", {}).get("per_seed", [])
    if not me_seeds or seed_idx >= len(me_seeds):
        print(f"  ERROR: no seed {seed_idx} in results for {env_key}")
        return False
    best_positions = me_seeds[seed_idx].get("best_positions", {})
    if not best_positions:
        print(f"  ERROR: no best_positions in results for {env_key}")
        return False

    print(f"\n  Processing {env_key}")
    print(f"    Baseline: {baseline_scene}")
    print(f"    Output:   {output_scene}")
    print(f"    Optimized objects: {list(best_positions.keys())}")

    # Read baseline XML
    with open(baseline_xml_path) as f:
        xml = f.read()

    # Load scene config (shared between baseline and optimized)
    scene_cfg = load_scene_yaml(baseline_scene)

    # Update body pos attributes for each optimized object
    mujoco_name_map = {o["short_name"]: o["mujoco_name"]
                       for o in scene_cfg["objects"]}
    for short, pos in best_positions.items():
        mname = mujoco_name_map.get(short)
        if mname is None:
            print(f"    Warning: {short} not in scene config")
            continue
        xml = update_body_pos(xml, mname, pos[0], pos[1])
        print(f"    {short}: ({pos[0]:.3f}, {pos[1]:.3f})")

    # Update keyframe qpos
    xml = update_keyframe_qpos(xml, scene_cfg, best_positions)

    # Update model name inside XML so MuJoCo distinguishes it
    xml = re.sub(
        r'<mujoco model="[^"]+">',
        f'<mujoco model="mesatask_{env_key}_me_optimized">',
        xml, count=1)

    # Write output XML
    with open(output_xml_path, "w") as f:
        f.write(xml)
    print(f"    Wrote {output_xml_path}")

    # Create YAML by copying and updating name field
    scene_yaml = load_scene_yaml(baseline_scene)
    # Re-read the full YAML structure (load_scene_yaml returns "scene" contents)
    with open(baseline_yaml_path) as f:
        full_yaml = yaml.safe_load(f)
    full_yaml["scene"]["name"] = output_scene
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(full_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"    Wrote {output_yaml_path}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="all",
                        choices=list(SCENE_MAP.keys()) + ["all"],
                        help="Environment key to process")
    parser.add_argument("--seed-idx", type=int, default=0,
                        help="Which MAP-Elites seed's best to use (default: 0)")
    parser.add_argument("--source", type=str, default="separability",
                        choices=["separability", "legibility", "trajectory_margin"],
                        help="Which optimization results to use")
    args = parser.parse_args()

    results, source = load_results(args.source)
    print(f"Loaded {args.source} results")
    print(f"Available envs: {list(results.keys())}")

    if args.scene == "all":
        env_keys = list(SCENE_MAP.keys())
    else:
        env_keys = [args.scene]

    success_count = 0
    for env_key in env_keys:
        if write_optimized_scene(env_key, results, seed_idx=args.seed_idx,
                                source=args.source):
            success_count += 1

    suffix = SUFFIX_MAP[args.source]
    print(f"\nWrote {success_count}/{len(env_keys)} optimized scenes")
    print(f"\nTo run the simulator:")
    print(f"  roslaunch tabletop_workspace_opt sim_moveit.launch "
          f"scene_name:=scene_desk{suffix}")


if __name__ == "__main__":
    main()
