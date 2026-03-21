#!/usr/bin/env python3
"""Download a PartNet Mobility object and convert its URDF to MuJoCo XML.

Usage:
    # Set your SAPIEN token first:
    export SAPIEN_ACCESS_TOKEN="your_token_here"

    # Download and convert a specific object:
    python3 download_and_convert.py --id 7119 --category Microwave

    # Just convert an already-downloaded object:
    python3 download_and_convert.py --id 7119 --skip-download

    # List available categories:
    python3 download_and_convert.py --list-categories

    # List objects in a category:
    python3 download_and_convert.py --list-objects Microwave
"""

import argparse
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ASSETS_DIR = PROJECT_ROOT / "src" / "assets"
PARTNET_DIR = PROJECT_ROOT / "src" / "assets" / "partnet_mobility"

SAPIEN_API = "https://sapien.ucsd.edu/api"


def get_token():
    token = os.environ.get("SAPIEN_ACCESS_TOKEN", "")
    if not token:
        token_file = Path.home() / ".sapien_token"
        if token_file.exists():
            token = token_file.read_text().strip()
    if not token:
        print("ERROR: No SAPIEN access token found.")
        print("Set SAPIEN_ACCESS_TOKEN env var or create ~/.sapien_token")
        print("Register at https://sapien.ucsd.edu to get a token.")
        sys.exit(1)
    return token


def list_categories():
    """List all available categories from the SAPIEN API."""
    resp = requests.get(f"{SAPIEN_API}/categories")
    resp.raise_for_status()
    categories = resp.json()
    print(f"{'Category':<25} {'Count':>6}")
    print("-" * 32)
    for cat in sorted(categories, key=lambda c: c.get("count", 0), reverse=True):
        print(f"{cat['name']:<25} {cat.get('count', '?'):>6}")


def list_objects(category, limit=20):
    """List object IDs for a given category."""
    resp = requests.get(f"{SAPIEN_API}/models", params={
        "category": category, "limit": limit
    })
    resp.raise_for_status()
    models = resp.json()
    print(f"\n{category} objects (showing {min(limit, len(models))}):")
    print(f"{'ID':<10} {'Name':<40}")
    print("-" * 50)
    for m in models[:limit]:
        model_id = m.get("model_id", m.get("id", "?"))
        name = m.get("name", m.get("model_id", ""))
        print(f"{model_id:<10} {name:<40}")


def download_object(object_id, token):
    """Download and extract a PartNet Mobility object."""
    dest = PARTNET_DIR / str(object_id)
    if dest.exists():
        print(f"Object {object_id} already exists at {dest}")
        return dest

    print(f"Downloading object {object_id}...")
    url = f"{SAPIEN_API}/download/compressed/{object_id}.zip?token={token}"
    resp = requests.get(url, stream=True)
    if resp.status_code == 401:
        print("ERROR: Invalid or expired SAPIEN token.")
        sys.exit(1)
    resp.raise_for_status()

    PARTNET_DIR.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    z.extractall(PARTNET_DIR)
    print(f"Extracted to {dest}")
    return dest


def inspect_object(object_dir):
    """Print the folder structure and key metadata."""
    object_dir = Path(object_dir)
    print(f"\n--- Object: {object_dir.name} ---")

    # Metadata
    meta_path = object_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"Category: {meta.get('model_cat', 'unknown')}")
        print(f"Name: {meta.get('model_id', object_dir.name)}")

    # URDF
    urdf_path = object_dir / "mobility.urdf"
    if urdf_path.exists():
        print(f"URDF: {urdf_path}")
        # Count links and joints
        content = urdf_path.read_text()
        n_links = content.count("<link")
        n_joints = content.count("<joint")
        print(f"  Links: {n_links}, Joints: {n_joints}")
    else:
        print("WARNING: No mobility.urdf found!")

    # Meshes
    mesh_dir = object_dir / "textured_objs"
    if mesh_dir.exists():
        obj_files = list(mesh_dir.glob("*.obj"))
        tex_files = list(mesh_dir.glob("*.png")) + list(mesh_dir.glob("*.jpg"))
        print(f"Meshes: {len(obj_files)} OBJ files, {len(tex_files)} textures")
    else:
        print("WARNING: No textured_objs/ directory found!")

    # Bounding box
    bb_path = object_dir / "bounding_box.json"
    if bb_path.exists():
        bb = json.loads(bb_path.read_text())
        print(f"Bounding box: {bb}")

    # Part semantics
    sem_path = object_dir / "semantics.txt"
    if sem_path.exists():
        lines = sem_path.read_text().strip().split("\n")
        print(f"Semantic parts: {len(lines)}")
        for line in lines[:10]:
            print(f"  {line}")

    return urdf_path if urdf_path.exists() else None


def convert_urdf_to_mjcf(urdf_path, output_name=None, scale=1.0):
    """Convert a PartNet Mobility URDF to MuJoCo XML.

    MuJoCo can load URDFs directly via mujoco.MjModel.from_xml_path(),
    but we want a standalone XML file. We use MuJoCo's built-in URDF
    compiler to parse the URDF, then save the compiled model as XML.

    Args:
        urdf_path: Path to mobility.urdf
        output_name: Name for the output XML (default: object_id)
        scale: Scale factor for the model

    Returns:
        Path to the generated MuJoCo XML file
    """
    import mujoco

    urdf_path = Path(urdf_path)
    object_dir = urdf_path.parent
    object_id = object_dir.name

    if output_name is None:
        output_name = f"partnet_{object_id}"

    # MuJoCo can load URDF directly. We wrap it in a minimal MJCF
    # that includes the URDF and adds necessary MuJoCo elements.
    # The key trick: MuJoCo's compiler resolves mesh paths relative
    # to the XML file location, so we need to work from the object dir.

    # First, try direct URDF loading
    print(f"\nConverting {urdf_path} to MuJoCo XML...")

    # Create a wrapper MJCF that includes the URDF
    wrapper_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="{output_name}">
  <compiler meshdir="textured_objs" angle="radian"
            fusestatic="true" balanceinertia="true"
            strippath="true" discardvisual="false"/>
  <option gravity="0 0 -9.81"/>

  <worldbody>
    <light name="top" pos="0 0 2" dir="0 0 -1" directional="true"/>
  </worldbody>
</mujoco>
"""

    # Try loading the URDF directly with MuJoCo
    try:
        # MuJoCo 2.3+ can load URDF files directly
        old_cwd = os.getcwd()
        os.chdir(str(object_dir))

        # First attempt: load URDF directly
        try:
            model = mujoco.MjModel.from_xml_path(str(urdf_path))
            print("  Loaded URDF directly with MuJoCo")
        except Exception as e:
            print(f"  Direct URDF load failed: {e}")
            print("  Trying with wrapper XML...")

            # Write wrapper and try again
            wrapper_path = object_dir / f"_wrapper_{output_name}.xml"
            wrapper_path.write_text(wrapper_xml)
            try:
                model = mujoco.MjModel.from_xml_path(str(wrapper_path))
                print("  Loaded via wrapper XML")
            finally:
                wrapper_path.unlink(missing_ok=True)

        os.chdir(old_cwd)

        # Save the compiled model as MJCF XML
        output_path = object_dir / f"{output_name}.xml"
        xml_string = mujoco.mj_saveLastXML(str(output_path), model)

        print(f"  Saved MuJoCo XML to: {output_path}")
        print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, "
              f"Geoms: {model.ngeom}, Meshes: {model.nmesh}")

        return output_path

    except Exception as e:
        os.chdir(old_cwd)
        print(f"  ERROR: Conversion failed: {e}")
        print("  You may need to manually fix the URDF or use dm_control.")
        print("  Install dm_control: pip3 install dm_control")
        return None


def generate_scene_include(object_id, output_name=None, pos=(0.6, 0.0, 1.0),
                           scale=0.001, category="object"):
    """Generate a snippet to include the converted object in a scene XML.

    Args:
        object_id: PartNet object ID
        output_name: Name used during conversion
        pos: (x, y, z) position in world frame
        scale: Scale factor (PartNet models are often in mm)
        category: Object category for naming

    Returns:
        XML string to paste into a scene file
    """
    if output_name is None:
        output_name = f"partnet_{object_id}"

    snippet = f"""
    <!-- {category} (PartNet Mobility ID: {object_id}) -->
    <!-- Option A: Include the converted XML as a sub-model -->
    <!-- <include file="partnet_mobility/{object_id}/{output_name}.xml"/> -->

    <!-- Option B: Reference the URDF directly (MuJoCo 2.3+) -->
    <body name="{category.lower()}_{object_id}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
      <!-- Add visual/collision geoms from the converted XML -->
      <!-- You may need to adjust scale and position offsets -->
    </body>
"""
    return snippet


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert PartNet Mobility objects to MuJoCo XML"
    )
    parser.add_argument("--id", type=int, help="PartNet Mobility object ID")
    parser.add_argument("--category", type=str, help="Object category (for naming)")
    parser.add_argument("--list-categories", action="store_true",
                        help="List all available categories")
    parser.add_argument("--list-objects", type=str, metavar="CATEGORY",
                        help="List objects in a category")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing files")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only inspect, don't convert")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor for conversion")
    parser.add_argument("--output-name", type=str,
                        help="Custom name for output XML")
    args = parser.parse_args()

    if args.list_categories:
        list_categories()
        return

    if args.list_objects:
        list_objects(args.list_objects)
        return

    if args.id is None:
        parser.print_help()
        return

    object_id = args.id
    category = args.category or "object"

    # Download
    if not args.skip_download:
        token = get_token()
        object_dir = download_object(object_id, token)
    else:
        object_dir = PARTNET_DIR / str(object_id)
        if not object_dir.exists():
            print(f"ERROR: Object directory not found: {object_dir}")
            sys.exit(1)

    # Inspect
    urdf_path = inspect_object(object_dir)

    if args.inspect_only:
        return

    if urdf_path is None:
        print("ERROR: No URDF found, cannot convert.")
        sys.exit(1)

    # Convert
    output_name = args.output_name or f"partnet_{object_id}"
    xml_path = convert_urdf_to_mjcf(urdf_path, output_name, args.scale)

    if xml_path:
        # Print scene include snippet
        print("\n--- Scene Include Snippet ---")
        print(generate_scene_include(
            object_id, output_name,
            pos=(0.6, 0.0, 1.0), category=category,
        ))
        print(f"To use in a scene, reference: {xml_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
