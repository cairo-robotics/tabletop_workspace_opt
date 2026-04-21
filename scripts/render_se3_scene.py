#!/usr/bin/env python3
"""Offscreen MuJoCo render of an SE(3)-optimized scene with grasp overlays.

For each object in a scene_*_se3_optimized.yaml layout this script:
  1. Loads the corresponding base MuJoCo XML (src/assets/scene_<name>.xml).
  2. Teleports every object to its optimized (x, y, yaw).
  3. Resolves the object's grasp pose via config/grasp_poses_3d.yaml.
  4. Injects a grasp axis triad (world-frame x=red, y=green, z=blue) at
     the grasp tip, a small sphere at the grasp tip, and a thin arrow
     from the pre-grasp to the grasp showing the approach segment.
  5. Renders multiple camera views to PNG under results/se3_visualize/.

Run `mujoco.viewer` is NOT used — all rendering is offscreen via
`mujoco.Renderer`, so this works on a headless machine.

Usage:
    # Single scene, default iso/top/side views
    python3 scripts/render_se3_scene.py \
        --scene config/scenes/scene_breakfast_easy_se3_optimized.yaml

    # Batch every SE(3)-optimized scene
    python3 scripts/render_se3_scene.py --all
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import mujoco
import imageio.v2 as imageio

from envopt.grasp_library import GraspLibrary
from envopt.reachability import ReachabilityOracle
from envopt.se3_utils import yaw_to_quat
from envopt.grasp_visualization import (
    SceneLayout,
    load_optimized_scene,
    resolve_all_grasps,
    find_missing_grasps,
)


# Tucked "display" pose used as the render baseline. This is NOT the
# runtime home pose from the scene XML keyframe — it's a compact,
# closer-to-body pose chosen specifically so the arm doesn't dominate
# the view of the table. Users can override with --default-joints.
TUCKED_JOINTS = np.array(
    [0.0, -1.40, 0.0, 2.70, 0.0, 1.40, 3.1416], dtype=float
)


# ---------------------------------------------------------------------------
# Scene teleport (same pattern as verify_se3_scenes_sim.py)
# ---------------------------------------------------------------------------

def teleport_body(model, data, mujoco_name, pos, yaw) -> bool:
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mujoco_name)
    except Exception:
        return False
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
# User-geom injection helpers
# ---------------------------------------------------------------------------

def _add_geom(scn, geom_type, size, pos, mat, rgba) -> bool:
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=int(geom_type),
        size=np.asarray(size, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.asarray(mat, dtype=np.float64).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1
    return True


def add_sphere(scn, pos, radius, rgba) -> None:
    _add_geom(
        scn, mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0.0, 0.0],
        pos=pos,
        mat=np.eye(3),
        rgba=rgba,
    )


def _rotation_to_align_z(direction) -> np.ndarray:
    """Return a 3x3 rotation whose local +z axis points along `direction`."""
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3)
    z = d / n
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def add_cylinder_segment(scn, p1, p2, rgba, radius=0.004) -> None:
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return
    R = _rotation_to_align_z(d)
    _add_geom(
        scn, mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, length * 0.5, 0.0],
        pos=(p1 + p2) * 0.5,
        mat=R,
        rgba=rgba,
    )


def add_arrow(scn, p1, p2, rgba, radius=0.004) -> None:
    """Thin cylinder from p1 to p2 (arrow-like visual)."""
    add_cylinder_segment(scn, p1, p2, rgba, radius=radius)


def add_grasp_axes(scn, pos, R, length=0.05, radius=0.0035) -> None:
    colors = [
        (1.0, 0.1, 0.1, 1.0),  # x = red
        (0.1, 1.0, 0.1, 1.0),  # y = green
        (0.1, 0.3, 1.0, 1.0),  # z = blue
    ]
    for col in range(3):
        axis = R[:, col]
        add_cylinder_segment(
            scn, pos, pos + length * axis, colors[col], radius=radius)


def add_infeasible_marker(scn, pos, size=0.025) -> None:
    """Draw a bright red X at a position to mark an infeasible grasp."""
    red = (1.0, 0.0, 0.0, 1.0)
    # Large red sphere
    add_sphere(scn, pos, size * 0.6, red)
    # Two crossed cylinders forming an X (in the xy plane)
    for dx, dy in [(1, 1), (1, -1)]:
        d = np.array([dx, dy, 0.0], dtype=float)
        d /= np.linalg.norm(d)
        add_cylinder_segment(
            scn, pos - size * d, pos + size * d, red, radius=0.003)


def add_grasp_overlay(scn, rg, axis_length=0.06, feasible=True) -> None:
    """Draw a grasp pose: axis triad + tip sphere + approach cylinder.

    If feasible=False, draws a large red X marker at the grasp tip
    instead of the normal axis triad, so infeasible grasps are
    immediately visible in the rendered image.
    """
    if not feasible:
        # Infeasible: red X + dimmed approach line
        add_infeasible_marker(scn, rg.grasp_pos, size=0.03)
        add_cylinder_segment(
            scn, rg.pregrasp_pos, rg.grasp_pos,
            (1.0, 0.2, 0.2, 0.6), radius=0.003)
        add_sphere(scn, rg.pregrasp_pos, 0.008,
                   (1.0, 0.2, 0.2, 0.7))
        return

    # Feasible: normal axes + colored tip
    add_grasp_axes(scn, rg.grasp_pos, rg.grasp_R,
                   length=axis_length, radius=0.004)
    if rg.approach_type == "top":
        tip_color = (0.1, 0.8, 0.1, 1.0)   # green for top-down
    else:
        tip_color = (0.1, 0.4, 1.0, 1.0)    # blue for side
    add_sphere(scn, rg.grasp_pos, 0.008, tip_color)
    add_sphere(scn, rg.pregrasp_pos, 0.006,
               (tip_color[0], tip_color[1], tip_color[2], 0.85))
    add_cylinder_segment(
        scn, rg.pregrasp_pos, rg.grasp_pos,
        (tip_color[0], tip_color[1], tip_color[2], 0.85),
        radius=0.0025,
    )


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def default_views():
    base_look = [0.6, 0.0, 1.02]
    views = {}

    # Iso view from the opposite side of the robot arm so the arm
    # doesn't occlude the objects on the table.
    cam = mujoco.MjvCamera()
    cam.lookat[:] = base_look
    cam.distance = 1.45
    cam.azimuth = 215
    cam.elevation = -32
    views["iso"] = cam

    # Near-top-down, slightly tilted to preserve some depth cues.
    cam = mujoco.MjvCamera()
    cam.lookat[:] = base_look
    cam.distance = 1.25
    cam.azimuth = 90
    cam.elevation = -82
    views["top"] = cam

    # Raised front-of-robot view tilted down toward the table. Gives a
    # clearer look at where the grasps sit on each object and how side
    # vs top approaches differ in orientation.
    cam = mujoco.MjvCamera()
    cam.lookat[:] = base_look
    cam.distance = 1.7
    cam.azimuth = 180
    cam.elevation = -28
    views["front"] = cam

    return views


# ---------------------------------------------------------------------------
# Main render pipeline
# ---------------------------------------------------------------------------

def resolve_base_xml(scene_name: str) -> str:
    """Map a scene layout name to its base MuJoCo XML.

    `scene_breakfast_easy_se3_optimized` -> `scene_breakfast_easy.xml`.
    """
    stem = scene_name
    for suffix in ("_se3_optimized", "_me_optimized", "_legibility_optimized",
                   "_trajectory_optimized", "_optimized"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return os.path.join(PROJECT_ROOT, "src", "assets", f"{stem}.xml")


def composite_ghost_over(baseline, grasp_frames, alpha=0.55, diff_thresh=18):
    """Alpha-composite grasp-pose frames over a solid baseline.

    The baseline (e.g. a render with the arm at the tucked pose) stays
    fully opaque. For each grasp frame, pixels that differ from the
    baseline by more than `diff_thresh` are considered "the arm moved
    here" and are alpha-blended onto the composite at `alpha`. Pixels
    that match the baseline are left untouched.

    Overlapping ghost regions compound naturally: after k frames, a
    pixel that had an arm in all k frames ends up as
        alpha * f_k + alpha*(1-alpha) * f_{k-1} + ... + (1-alpha)^k * baseline,
    so heavily-overlapped regions still read as "lots of arm motion".
    """
    base_f = baseline.astype(np.float32)
    out = base_f.copy()
    for frame in grasp_frames:
        f = frame.astype(np.float32)
        diff = np.max(np.abs(f - base_f), axis=2)
        mask = diff > diff_thresh
        if not mask.any():
            continue
        out[mask] = alpha * f[mask] + (1.0 - alpha) * out[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def _solve_ik_for_grasps(oracle, model, data, resolved):
    """Solve IK for each grasp tip pose on the already-teleported scene.

    Copies the current qpos into the oracle so its reachability check
    sees the teleported objects, then calls oracle.solve_ik per grasp.
    Returns a list of (resolved_grasp, joint_array_or_None) in order.
    """
    oracle.data.qpos[:] = data.qpos[:]
    mujoco.mj_forward(oracle.model, oracle.data)

    ik_results = []
    for rg in resolved:
        q = oracle.solve_ik(rg.grasp_pos, rg.grasp_R)
        ik_results.append((rg, q))
    return ik_results


def _apply_arm_joints(model, data, arm_qpos_addrs, q):
    for i, addr in enumerate(arm_qpos_addrs):
        data.qpos[addr] = q[i]
    mujoco.mj_forward(model, data)


def _arm_qpos_addrs(model):
    addrs = []
    for i in range(7):
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"right_j{i}")
        addrs.append(model.jnt_qposadr[jid])
    return addrs


def _render_one(renderer, data, cam, resolved, axis_length,
                infeasible_names=None):
    """Update scene at current data state, inject overlays, render, return uint8 image.

    infeasible_names: set of short_names whose grasps are infeasible.
        Those get a red X marker instead of the normal axis triad.
    """
    if infeasible_names is None:
        infeasible_names = set()
    renderer.update_scene(data, camera=cam)
    scn = renderer.scene
    for rg in resolved:
        feasible = rg.short_name not in infeasible_names
        add_grasp_overlay(scn, rg, axis_length=axis_length,
                          feasible=feasible)
    return renderer.render()


def render_scene(
    scene: SceneLayout,
    library: GraspLibrary,
    output_dir: str,
    width: int = 1280,
    height: int = 800,
    axis_length: float = 0.05,
    ghost_arms: bool = False,
    ghost_alpha: float = 0.75,
    ghost_diff_thresh: int = 18,
    default_joints: np.ndarray = None,
) -> dict:
    base_xml = resolve_base_xml(scene.name)
    if not os.path.exists(base_xml):
        raise FileNotFoundError(f"Base MuJoCo XML not found: {base_xml}")

    model = mujoco.MjModel.from_xml_path(base_xml)
    # Resize the offscreen framebuffer to match the requested render size.
    # MuJoCo's default `vis.global_.offwidth/offheight` is 640x480 or 1920x1080
    # depending on the model, and creating a Renderer larger than that fails.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)

    teleport_failures = []
    for obj in scene.objects:
        ok = teleport_body(model, data, obj.mujoco_name, obj.position, obj.yaw)
        if not ok:
            teleport_failures.append(obj.mujoco_name)

    # Apply the tucked display pose before any rendering. This replaces
    # the scene XML's `<keyframe name="home">` (which extends the arm
    # across the table) with a compact, close-to-body pose so the arm
    # doesn't dominate the view. Ghost-arm IK is solved relative to this.
    arm_addrs = _arm_qpos_addrs(model)
    default_q = (
        np.asarray(default_joints, dtype=float)
        if default_joints is not None
        else TUCKED_JOINTS.copy()
    )
    _apply_arm_joints(model, data, arm_addrs, default_q)

    resolved = resolve_all_grasps(scene, library)
    missing = find_missing_grasps(scene, library)

    ik_results = []
    ghost_failures = []
    if ghost_arms:
        oracle = ReachabilityOracle(
            base_xml, pos_tol=0.02, rot_tol_deg=20.0,
            max_iters=60, n_seeds=4)
        # Teleport objects in the oracle so collision checks see the layout.
        for obj in scene.objects:
            oracle.teleport_object(obj.mujoco_name, obj.position, obj.yaw)
        ik_results = _solve_ik_for_grasps(oracle, model, data, resolved)
        # Mark grasps as failed if IK didn't converge OR if the IK
        # solution has arm-vs-other-object collisions.
        for rg, q in ik_results:
            if q is None:
                ghost_failures.append(rg.short_name)
            else:
                contacts = oracle.arm_object_collisions(
                    q, target_body_name=None)
                # Find the target body name for this object
                target_body = None
                for obj in scene.objects:
                    if obj.short_name == rg.short_name:
                        target_body = obj.mujoco_name
                        break
                contacts = oracle.arm_object_collisions(
                    q, target_body_name=target_body)
                if contacts:
                    ghost_failures.append(rg.short_name)

    renderer = mujoco.Renderer(model, height=height, width=width)
    os.makedirs(output_dir, exist_ok=True)

    # Build the set of infeasible object names for overlay annotation.
    infeasible_set = set(ghost_failures)

    written = []
    stem = scene.name
    for view_name, cam in default_views().items():
        if ghost_arms and ik_results:
            # Baseline: render with the arm at the tucked default pose.
            # This stays fully opaque in the final composite.
            _apply_arm_joints(model, data, arm_addrs, default_q)
            baseline = _render_one(renderer, data, cam, resolved,
                                   axis_length, infeasible_set)

            # Per-grasp frames: render with the arm at each IK-solved
            # grasp pose. Every frame also draws the grasp overlays so
            # they're identical to the baseline in overlay pixels and
            # survive alpha compositing untouched.
            grasp_frames = []
            for rg, q in ik_results:
                if q is None:
                    continue
                _apply_arm_joints(model, data, arm_addrs, q)
                grasp_frames.append(
                    _render_one(renderer, data, cam, resolved, axis_length,
                                infeasible_set))
            # Restore tucked pose for any subsequent view.
            _apply_arm_joints(model, data, arm_addrs, default_q)

            composite = composite_ghost_over(
                baseline, grasp_frames,
                alpha=ghost_alpha, diff_thresh=ghost_diff_thresh)
            out_path = os.path.join(
                output_dir, f"{stem}_{view_name}_ghost.png")
            imageio.imwrite(out_path, composite)
        else:
            img = _render_one(renderer, data, cam, resolved, axis_length,
                              infeasible_set)
            out_path = os.path.join(output_dir, f"{stem}_{view_name}.png")
            imageio.imwrite(out_path, img)
        written.append(out_path)

    return {
        "scene": scene.name,
        "base_xml": base_xml,
        "objects": len(scene.objects),
        "grasps_resolved": len(resolved),
        "missing_grasps": missing,
        "teleport_failures": teleport_failures,
        "ghost_failures": ghost_failures,
        "outputs": written,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument(
        "--grasp-library", type=str,
        default=os.path.join(PROJECT_ROOT, "config", "grasp_poses_3d.yaml"),
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(PROJECT_ROOT, "results", "se3_visualize"),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--axis-length", type=float, default=0.07,
        help="Length of each grasp-frame axis cylinder in meters (default 0.07).",
    )
    parser.add_argument(
        "--ghost-arms", action="store_true",
        help="Render the robot arm at every grasp pose with IK-solved "
             "joints and alpha-composite them over the tucked-pose "
             "baseline. Baseline stays fully opaque; each grasp arm "
             "is blended at --ghost-alpha opacity. Outputs use the "
             "*_ghost.png suffix.",
    )
    parser.add_argument(
        "--ghost-alpha", type=float, default=0.75,
        help="Alpha for ghost-arm blending (0 = invisible, 1 = fully "
             "opaque). Default 0.75.",
    )
    parser.add_argument(
        "--ghost-diff-thresh", type=int, default=18,
        help="Pixel-diff threshold (0-255) used to detect where the arm "
             "moved relative to the baseline. Higher = stricter.",
    )
    parser.add_argument(
        "--default-joints", type=str, default=None,
        help="Override the 7-element arm pose used for the rendering "
             "baseline, as a comma-separated list "
             "(e.g. '0,-1.4,0,2.7,0,1.4,3.14'). Defaults to the tucked "
             "pose defined in the script.",
    )
    args = parser.parse_args()

    default_q = None
    if args.default_joints:
        try:
            default_q = np.array(
                [float(x) for x in args.default_joints.split(",")])
            if default_q.shape != (7,):
                parser.error("--default-joints must have exactly 7 values")
        except ValueError as e:
            parser.error(f"--default-joints parse error: {e}")

    library = GraspLibrary.load(args.grasp_library)

    if args.all:
        scenes_dir = os.path.join(PROJECT_ROOT, "config", "scenes")
        scene_files = sorted(
            os.path.join(scenes_dir, f)
            for f in os.listdir(scenes_dir)
            if f.endswith("_se3_optimized.yaml")
        )
        if not scene_files:
            print(f"No scene_*_se3_optimized.yaml files in {scenes_dir}", file=sys.stderr)
            return 1
        for scene_path in scene_files:
            scene = load_optimized_scene(scene_path)
            try:
                info = render_scene(
                    scene, library, args.output_dir,
                    width=args.width, height=args.height,
                    axis_length=args.axis_length,
                    ghost_arms=args.ghost_arms,
                    ghost_alpha=args.ghost_alpha,
                    ghost_diff_thresh=args.ghost_diff_thresh,
                    default_joints=default_q,
                )
            except Exception as e:
                print(f"[{scene.name}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            print(
                f"[{scene.name}]  objs={info['objects']}  "
                f"grasps={info['grasps_resolved']}  "
                f"-> {len(info['outputs'])} views"
            )
            if info["missing_grasps"]:
                print(f"    missing: {info['missing_grasps']}")
            if info["teleport_failures"]:
                print(f"    teleport failed: {info['teleport_failures']}")
            if info.get("ghost_failures"):
                print(f"    IK failed for: {info['ghost_failures']}")
        return 0

    if not args.scene:
        parser.error("--scene is required (or pass --all).")
    scene = load_optimized_scene(args.scene)
    info = render_scene(
        scene, library, args.output_dir,
        width=args.width, height=args.height,
        axis_length=args.axis_length,
        ghost_arms=args.ghost_arms,
        ghost_alpha=args.ghost_alpha,
        ghost_diff_thresh=args.ghost_diff_thresh,
        default_joints=default_q,
    )
    print(
        f"[{scene.name}]  objs={info['objects']}  "
        f"grasps={info['grasps_resolved']}"
    )
    for p in info["outputs"]:
        print(f"  {p}")
    if info["missing_grasps"]:
        print(f"  missing grasps: {info['missing_grasps']}")
    if info["teleport_failures"]:
        print(f"  teleport failed: {info['teleport_failures']}")
    if info.get("ghost_failures"):
        print(f"  IK failed for: {info['ghost_failures']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
