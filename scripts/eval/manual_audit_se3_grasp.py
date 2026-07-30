#!/usr/bin/env python3
"""Manually audit one SE(3) grasp in the MuJoCo + MoveIt simulator.

Launch the base scene first, then run this script with the optimized layout
YAML scene name. Example:

    roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_breakfast_easy

    python3 scripts/eval/manual_audit_se3_grasp.py \\
      --scene scene_breakfast_easy_se3_me_optimized \\
      --object banana

The script resolves the grasp/pregrasp pose from config/grasp_poses_3d.yaml,
teleports objects that have positions in the chosen scene YAML, then pauses
between each motion so the grasp can be inspected in RViz/MuJoCo.
"""
import argparse
import glob
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# catkin_make_isolated creates separate devel spaces per package. Add the ROS
# and package paths explicitly so the script can be run as plain python3.
_ROS_PYTHON_PATH = "/opt/ros/noetic/lib/python3/dist-packages"
if _ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROS_PYTHON_PATH)
_DEVEL_ISOLATED = os.path.expanduser("~/sawyer_ws/devel_isolated")
for _d in glob.glob(os.path.join(_DEVEL_ISOLATED, "*/lib/python3/dist-packages")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
# Keep ROS Noetic's Python packages ahead of isolated workspace paths. Some
# isolated devel spaces can contain namespace/stub packages that shadow the
# real MoveIt Python package and lack moveit_commander.roscpp_initialize.
if _ROS_PYTHON_PATH in sys.path:
    sys.path.insert(0, sys.path.pop(sys.path.index(_ROS_PYTHON_PATH)))
else:
    sys.path.insert(0, _ROS_PYTHON_PATH)
_SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import numpy as np
import moveit_commander
import rospy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from moveit_msgs.msg import DisplayRobotState, DisplayTrajectory, RobotState
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionIKRequest,
    GetStateValidity,
    GetStateValidityRequest,
)
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tabletop_workspace_opt.srv import (
    MoveToCartesianPose,
    OperateGripper,
    TeleportObject,
)
from visualization_msgs.msg import Marker, MarkerArray

from envopt.grasp_library import GraspLibrary, default_library_path
from envopt.se3_utils import quat_to_rot, yaw_to_quat


BASE_Z_OFFSET = 0.92
FINGERTIP_EE_OFFSET = np.array([0.0, 0.0, 0.0295])


def world_to_base(pos_world):
    pos_world = np.asarray(pos_world, dtype=float)
    return np.array([pos_world[0], pos_world[1], pos_world[2] - BASE_Z_OFFSET])


def wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float)
    return q[1], q[2], q[3], q[0]


def to_pose_stamped(pos_world, quat_wxyz, frame_id):
    pos_base = world_to_base(pos_world)
    qx, qy, qz, qw = wxyz_to_xyzw(quat_wxyz)
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(pos_base[0])
    msg.pose.position.y = float(pos_base[1])
    msg.pose.position.z = float(pos_base[2])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def load_scene(scene_name):
    path = os.path.join(PROJECT_ROOT, "config", "scenes", f"{scene_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scene YAML not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)["scene"]
    objects = {o["short_name"]: o for o in cfg["objects"]}
    return path, cfg, objects


def resolve_object_pose(obj_cfg, scene_name=None):
    if "position" not in obj_cfg:
        if scene_name is None:
            raise ValueError(
                f"Object '{obj_cfg['short_name']}' has no position in this "
                "scene YAML and no scene name was provided for XML fallback.")
        return resolve_object_pose_from_xml(scene_name, obj_cfg)
    pos = np.asarray(obj_cfg["position"], dtype=float)
    if pos.shape[0] == 2:
        pos = np.array([pos[0], pos[1], 0.95])
    yaw = float(obj_cfg.get("yaw", 0.0))
    return pos, yaw


def resolve_object_pose_from_xml(scene_name, obj_cfg):
    """Read an object's reset pose from the MuJoCo XML/keyframe.

    Base scene YAMLs intentionally do not duplicate object positions; those
    positions live in src/assets/<scene>.xml. Optimized scene YAMLs include
    explicit positions/yaws and use resolve_object_pose() above.
    """
    import mujoco

    xml_path = os.path.join(PROJECT_ROOT, "src", "assets", f"{scene_name}.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"Object '{obj_cfg['short_name']}' has no position in the scene "
            f"YAML, and XML fallback does not exist: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, obj_cfg["mujoco_name"])
    if bid < 0:
        raise ValueError(
            f"Body '{obj_cfg['mujoco_name']}' not found in {xml_path}")

    pos = data.xpos[bid].copy()
    R = data.xmat[bid].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    print(
        f"Resolved {obj_cfg['short_name']} pose from XML fallback "
        f"{xml_path}: world=[{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}], "
        f"yaw={yaw:.4f}")
    return pos, yaw


def pause(args, label):
    if args.no_pause:
        return
    input(f"\n[{label}] Press Enter to continue, or Ctrl-C to stop... ")


def call_move(move_srv, pos_world, quat_wxyz, label, cartesian=False):
    pos_base = world_to_base(pos_world)
    qx, qy, qz, qw = wxyz_to_xyzw(quat_wxyz)
    rospy.loginfo(
        "%s world=[%.4f %.4f %.4f] base=[%.4f %.4f %.4f] "
        "quat_xyzw=[%.4f %.4f %.4f %.4f]",
        label, pos_world[0], pos_world[1], pos_world[2],
        pos_base[0], pos_base[1], pos_base[2], qx, qy, qz, qw)
    resp = move_srv(
        x=float(pos_base[0]), y=float(pos_base[1]), z=float(pos_base[2]),
        qx=float(qx), qy=float(qy), qz=float(qz), qw=float(qw),
        cartesian=cartesian)
    if not resp.success:
        rospy.logwarn("%s motion failed: %s", label, resp.message)
    return bool(resp.success)


def make_pose_stamped_base(pos_world, quat_wxyz, frame_id="base"):
    pos_base = world_to_base(pos_world)
    qx, qy, qz, qw = wxyz_to_xyzw(quat_wxyz)
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(pos_base[0])
    msg.pose.position.y = float(pos_base[1])
    msg.pose.position.z = float(pos_base[2])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def robot_state_with_goal_joints(robot, joint_names, joint_values):
    """Return full current RobotState with active arm joints overwritten."""
    state = robot.get_current_state()
    names = list(state.joint_state.name)
    positions = list(state.joint_state.position)
    for name, value in zip(joint_names, joint_values):
        if name in names:
            positions[names.index(name)] = float(value)
        else:
            names.append(name)
            positions.append(float(value))
    state.joint_state.name = names
    state.joint_state.position = positions
    return state


def compute_ik(group, robot, pos_world, quat_wxyz, label, num_seeds=12,
               avoid_collisions=True, ik_link_name=None):
    """Compute one IK solution for a world-frame EE pose."""
    ik_srv = rospy.ServiceProxy("/compute_ik", GetPositionIK)
    ik_srv.wait_for_service(timeout=5.0)
    joint_names = group.get_active_joints()
    qx, qy, qz, qw = wxyz_to_xyzw(quat_wxyz)
    pos_base = world_to_base(pos_world)

    joint_limits = []
    for name in joint_names:
        joint_limits.append(robot.get_joint(name).bounds())

    for seed_idx in range(num_seeds):
        req = GetPositionIKRequest()
        req.ik_request.group_name = group.get_name()
        if ik_link_name:
            req.ik_request.ik_link_name = ik_link_name
        req.ik_request.pose_stamped.header.frame_id = group.get_planning_frame()
        req.ik_request.pose_stamped.header.stamp = rospy.Time.now()
        req.ik_request.pose_stamped.pose.position.x = float(pos_base[0])
        req.ik_request.pose_stamped.pose.position.y = float(pos_base[1])
        req.ik_request.pose_stamped.pose.position.z = float(pos_base[2])
        req.ik_request.pose_stamped.pose.orientation.x = float(qx)
        req.ik_request.pose_stamped.pose.orientation.y = float(qy)
        req.ik_request.pose_stamped.pose.orientation.z = float(qz)
        req.ik_request.pose_stamped.pose.orientation.w = float(qw)
        req.ik_request.timeout = rospy.Duration(1.0)
        req.ik_request.avoid_collisions = avoid_collisions

        seed = RobotState()
        seed.joint_state.name = joint_names
        if seed_idx == 0:
            seed.joint_state.position = group.get_current_joint_values()
        else:
            seed.joint_state.position = [
                float(np.random.uniform(lo, hi)) for lo, hi in joint_limits
            ]
        req.ik_request.robot_state = seed

        resp = ik_srv(req)
        if resp.error_code.val != 1:
            continue
        solution_names = list(resp.solution.joint_state.name)
        values = []
        for name in joint_names:
            values.append(resp.solution.joint_state.position[
                solution_names.index(name)])
        rospy.loginfo("%s IK solution found on seed %d/%d (avoid_collisions=%s)",
                      label, seed_idx + 1, num_seeds, avoid_collisions)
        return joint_names, values

    rospy.logwarn("%s IK failed for all %d seeds (avoid_collisions=%s)",
                  label, num_seeds, avoid_collisions)
    return joint_names, None


def report_state_validity(robot_state, group_name, label, max_contacts=20):
    """Ask MoveIt which contacts make a candidate robot state invalid."""
    try:
        validity_srv = rospy.ServiceProxy(
            "/check_state_validity", GetStateValidity)
        validity_srv.wait_for_service(timeout=3.0)
    except (rospy.ROSException, rospy.ServiceException) as exc:
        rospy.logwarn("Could not contact /check_state_validity: %s", exc)
        return

    req = GetStateValidityRequest()
    req.robot_state = robot_state
    req.group_name = group_name

    try:
        resp = validity_srv(req)
    except rospy.ServiceException as exc:
        rospy.logwarn("/check_state_validity failed: %s", exc)
        return

    if resp.valid:
        rospy.loginfo("%s state validity: VALID", label)
        return

    rospy.logwarn("%s state validity: INVALID", label)
    if not resp.contacts:
        rospy.logwarn(
            "%s invalid but MoveIt returned no contact pairs. This can happen "
            "with joint-limit, bounds, or plugin-level validity failures.",
            label)
        return

    by_pair = {}
    for contact in resp.contacts:
        pair = tuple(sorted((contact.contact_body_1, contact.contact_body_2)))
        row = by_pair.setdefault(pair, {
            "contacts": [],
            "max_depth": float("-inf"),
            "deepest": None,
        })
        row["contacts"].append(contact)
        if contact.depth > row["max_depth"]:
            row["max_depth"] = contact.depth
            row["deepest"] = contact

    rows = sorted(by_pair.items(),
                  key=lambda item: item[1]["max_depth"],
                  reverse=True)
    deepest_pair, deepest_row = rows[0]
    deepest_contact = deepest_row["deepest"]
    recommended_clearance = max(0.0, deepest_row["max_depth"]) + 0.005

    rospy.logwarn(
        "%s collision summary: %d raw contacts, %d unique pairs, "
        "deepest=%s <-> %s %.1f mm",
        label, len(resp.contacts), len(rows),
        deepest_pair[0], deepest_pair[1], deepest_row["max_depth"] * 1000.0)
    rospy.logwarn(
        "%s rough tuning hint: move the relevant pose/link at least %.1f mm "
        "away from the obstacle, then add extra margin for MoveIt padding.",
        label, recommended_clearance * 1000.0)
    rospy.logwarn(
        "%s deepest contact point=[%.3f %.3f %.3f] normal=[%.3f %.3f %.3f]",
        label,
        deepest_contact.position.x,
        deepest_contact.position.y,
        deepest_contact.position.z,
        deepest_contact.normal.x,
        deepest_contact.normal.y,
        deepest_contact.normal.z)

    rospy.logwarn("%s contact pairs by max penetration:", label)
    for i, (pair, row) in enumerate(rows):
        depths = [c.depth for c in row["contacts"]]
        rospy.logwarn("  %s <-> %s  max=%.1f mm  mean=%.1f mm  n=%d",
                      pair[0], pair[1],
                      max(depths) * 1000.0,
                      (sum(depths) / len(depths)) * 1000.0,
                      len(depths))
        if i + 1 >= max_contacts:
            rospy.logwarn("  ... truncated after %d unique contact pairs",
                          max_contacts)
            break


def publish_moveit_goal_state(args, robot, group, display_state_pub,
                              display_traj_pub, pos_world, quat_wxyz, label):
    """Publish MoveIt goal state/path visualization without executing motion."""
    if args.no_moveit_preview:
        return True

    joint_names, joint_values = compute_ik(
        group, robot, pos_world, quat_wxyz, label, num_seeds=args.ik_seeds,
        avoid_collisions=True, ik_link_name=args.ik_link)
    if joint_values is None and args.ik_diagnose_collisions:
        rospy.logwarn(
            "%s collision-aware IK failed; retrying IK with collisions disabled "
            "for diagnosis only.", label)
        joint_names, joint_values = compute_ik(
            group, robot, pos_world, quat_wxyz,
            f"{label} (collision-ignored)",
            num_seeds=args.ik_seeds, avoid_collisions=False,
            ik_link_name=args.ik_link)
        if joint_values is not None:
            rospy.logwarn(
                "%s IK succeeds only when collisions are ignored. Treat this "
                "pose as collision-constrained, not valid for final audit.",
                label)
            diagnostic_state = robot_state_with_goal_joints(
                robot, joint_names, joint_values)
            report_state_validity(diagnostic_state, group.get_name(), label)
    if joint_values is None:
        return False

    goal_state = robot_state_with_goal_joints(robot, joint_names, joint_values)
    display_state = DisplayRobotState()
    display_state.state = goal_state
    display_state_pub.publish(display_state)
    rospy.loginfo("Published %s goal robot state to /move_group/display_robot_state",
                  label)

    if args.no_plan_preview:
        return True

    group.set_start_state_to_current_state()
    group.set_joint_value_target(joint_values)
    success, plan, planning_time, error_code = group.plan()
    group.clear_pose_targets()

    if success and plan.joint_trajectory.points:
        display = DisplayTrajectory()
        display.model_id = "sawyer"
        display.trajectory_start = robot.get_current_state()
        display.trajectory.append(plan)
        display_traj_pub.publish(display)
        rospy.loginfo(
            "Published %s preview plan to /move_group/display_planned_path "
            "(%.2fs, %d waypoints)",
            label, planning_time, len(plan.joint_trajectory.points))
        return True

    rospy.logwarn("%s preview planning failed before execution (error=%d)",
                  label, error_code.val)
    return False


def reset_and_apply_layout(args, scene_cfg, objects_by_name, reset_srv, teleport_srv):
    if args.no_reset:
        rospy.loginfo("Skipping /reset_sim because --no-reset was set")
    else:
        rospy.loginfo("Resetting simulation")
        reset_srv()
        rospy.sleep(args.settle)

    if args.no_teleport_layout:
        rospy.loginfo("Skipping layout teleport because --no-teleport-layout was set")
        return

    rospy.loginfo("Teleporting objects with explicit positions from scene YAML")
    for short, obj in objects_by_name.items():
        if "position" not in obj:
            continue
        pos, yaw = resolve_object_pose(obj, scene_cfg["name"])
        quat = yaw_to_quat(yaw)
        resp = teleport_srv(
            object_name=obj["mujoco_name"],
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
            qw=float(quat[0]), qx=float(quat[1]),
            qy=float(quat[2]), qz=float(quat[3]))
        if not resp.success:
            rospy.logwarn("Teleport failed for %s/%s: %s",
                          short, obj["mujoco_name"], resp.message)
    rospy.sleep(args.settle)


def print_pose_summary(obj_name, obj_pos, obj_yaw, entry, poses):
    print("\n" + "=" * 78)
    print(f"SE(3) manual grasp audit: {obj_name} / {entry.name}")
    print("=" * 78)
    print(f"object world pos:   [{obj_pos[0]: .4f}, {obj_pos[1]: .4f}, {obj_pos[2]: .4f}]")
    print(f"object yaw:         {obj_yaw: .4f} rad")
    print(f"local offset:       {entry.local_offset.tolist()}")
    print(f"local quat wxyz:    {entry.local_quat.tolist()}")
    print(f"approach standoff:  {entry.approach_standoff:.4f} m")
    print(f"approach axis:      {entry.approach_axis.tolist()}  (grasp frame)")
    for label, key_pos, key_quat in [
            ("pregrasp", "pregrasp_pos", "pregrasp_quat"),
            ("grasp", "grasp_pos", "grasp_quat")]:
        p = poses[key_pos]
        q = poses[key_quat]
        pb = world_to_base(p)
        print(f"{label:8s} world:   [{p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f}]")
        print(f"{label:8s} base:    [{pb[0]: .4f}, {pb[1]: .4f}, {pb[2]: .4f}]")
        print(f"{label:8s} quat:    wxyz={q.tolist()} xyzw={wxyz_to_xyzw(q)}")
    dz = poses["pregrasp_pos"][2] - poses["grasp_pos"][2]
    print(f"pregrasp - grasp dz: {dz:+.4f} m")
    print("=" * 78 + "\n")


def make_arrow_marker(frame_id, ns, marker_id, start, end, color):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.points = [
        Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
        Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
    ]
    marker.scale.x = 0.008
    marker.scale.y = 0.018
    marker.scale.z = 0.035
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    marker.lifetime = rospy.Duration(0)
    return marker


def make_sphere_marker(frame_id, ns, marker_id, pos, radius, color):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = float(pos[0])
    marker.pose.position.y = float(pos[1])
    marker.pose.position.z = float(pos[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x = radius
    marker.scale.y = radius
    marker.scale.z = radius
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    marker.lifetime = rospy.Duration(0)
    return marker


def make_cube_marker(frame_id, ns, marker_id, pos, half_extents, color):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = float(pos[0])
    marker.pose.position.y = float(pos[1])
    marker.pose.position.z = float(pos[2])
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(half_extents[0] * 2.0)
    marker.scale.y = float(half_extents[1] * 2.0)
    marker.scale.z = float(half_extents[2] * 2.0)
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    marker.lifetime = rospy.Duration(0)
    return marker


def make_text_marker(frame_id, ns, marker_id, pos, text, color):
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.x = float(pos[0])
    marker.pose.position.y = float(pos[1])
    marker.pose.position.z = float(pos[2])
    marker.pose.orientation.w = 1.0
    marker.scale.z = 0.035
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    marker.text = text
    marker.lifetime = rospy.Duration(0)
    return marker


def add_pose_axes(markers, frame_id, ns, start_id, pos_world, quat_wxyz, length):
    pos = world_to_base(pos_world)
    R = quat_to_rot(quat_wxyz)
    axes = [
        ("x", R[:, 0], (1.0, 0.0, 0.0, 1.0)),
        ("y", R[:, 1], (0.0, 0.8, 0.0, 1.0)),
        ("z", R[:, 2], (0.1, 0.3, 1.0, 1.0)),
    ]
    for i, (_axis_name, axis, color) in enumerate(axes):
        markers.append(make_arrow_marker(
            frame_id, ns, start_id + i, pos, pos + length * axis, color))


def add_gripper_proxy_markers(markers, frame_id, ns, start_id,
                              pos_world, quat_wxyz, label):
    R = quat_to_rot(quat_wxyz)
    ee_base = world_to_base(pos_world)
    fingertip_world = np.asarray(pos_world) + R @ FINGERTIP_EE_OFFSET
    fingertip_base = world_to_base(fingertip_world)
    tool_axis_end = ee_base + 0.08 * R[:, 2]
    markers.append(make_sphere_marker(
        frame_id, ns, start_id, ee_base, 0.025,
        (0.2, 0.6, 1.0, 0.9)))
    markers.append(make_sphere_marker(
        frame_id, ns, start_id + 1, fingertip_base, 0.022,
        (1.0, 0.2, 0.8, 0.9)))
    markers.append(make_arrow_marker(
        frame_id, ns, start_id + 2, ee_base, tool_axis_end,
        (0.2, 0.6, 1.0, 1.0)))
    markers.append(make_text_marker(
        frame_id, ns, start_id + 3,
        ee_base + np.array([0.0, 0.0, 0.05]),
        f"{label}: blue=EE magenta=fingertip", (0.8, 0.8, 1.0, 1.0)))


def add_table_marker(markers, frame_id, scene_cfg):
    table = scene_cfg.get("table", {})
    if not table:
        return
    pos_world = np.asarray(table.get("pos_world", [0.6, 0.0, 0.915]),
                           dtype=float)
    half_extents = np.asarray(table.get("half_extents", [0.4, 0.7, 0.027]),
                              dtype=float)
    # Scene YAML table.pos_world.z is the tabletop surface height, matching the
    # MuJoCo tablelink convention.  The collision box center is one half-height
    # below the surface.
    table_center_world = pos_world.copy()
    table_center_world[2] -= half_extents[2]
    table_base = world_to_base(table_center_world)
    markers.append(make_cube_marker(
        frame_id, "table_collision_box", 100, table_base, half_extents,
        (0.0, 1.0, 1.0, 0.20)))
    top_z = table_base[2] + half_extents[2]
    bottom_z = table_base[2] - half_extents[2]
    markers.append(make_text_marker(
        frame_id, "table_collision_box", 101,
        table_base + np.array([0.0, 0.0, half_extents[2] + 0.04]),
        f"table collision z=[{bottom_z:+.3f}, {top_z:+.3f}] base",
        (0.0, 1.0, 1.0, 1.0)))


def publish_visualization(args, scene_cfg, obj_cfg, obj_pos, poses,
                          pre_pub, grasp_pub, marker_pub):
    frame_id = args.viz_frame
    pre_pub.publish(to_pose_stamped(
        poses["pregrasp_pos"], poses["pregrasp_quat"], frame_id))
    grasp_pub.publish(to_pose_stamped(
        poses["grasp_pos"], poses["grasp_quat"], frame_id))

    markers = []
    add_pose_axes(markers, frame_id, "pregrasp_axes", 0,
                  poses["pregrasp_pos"], poses["pregrasp_quat"], 0.08)
    add_pose_axes(markers, frame_id, "grasp_axes", 10,
                  poses["grasp_pos"], poses["grasp_quat"], 0.08)

    pre_base = world_to_base(poses["pregrasp_pos"])
    grasp_base = world_to_base(poses["grasp_pos"])
    obj_base = world_to_base(obj_pos)
    add_table_marker(markers, frame_id, scene_cfg)
    markers.append(make_arrow_marker(
        frame_id, "approach_segment", 20, pre_base, grasp_base,
        (1.0, 1.0, 0.0, 1.0)))
    markers.append(make_sphere_marker(
        frame_id, "object_center", 21, obj_base, 0.035,
        (1.0, 0.55, 0.0, 0.9)))

    half_extents = obj_cfg.get("half_extents")
    if half_extents is not None:
        markers.append(make_cube_marker(
            frame_id, "object_bbox", 22, obj_base,
            np.asarray(half_extents, dtype=float),
            (1.0, 0.55, 0.0, 0.22)))

    add_gripper_proxy_markers(
        markers, frame_id, "pregrasp_gripper_proxy", 30,
        poses["pregrasp_pos"], poses["pregrasp_quat"], "pregrasp")
    add_gripper_proxy_markers(
        markers, frame_id, "grasp_gripper_proxy", 40,
        poses["grasp_pos"], poses["grasp_quat"], "grasp")

    text = (f"{obj_cfg['short_name']} pre_z={pre_base[2]:+.3f} "
            f"grasp_z={grasp_base[2]:+.3f} base frame")
    text_pos = pre_base + np.array([0.0, 0.0, 0.10])
    color = (1.0, 0.2, 0.2, 1.0) if pre_base[2] < 0.0 else (1.0, 1.0, 1.0, 1.0)
    markers.append(make_text_marker(
        frame_id, "pose_labels", 23, text_pos, text, color))

    marker_pub.publish(MarkerArray(markers=markers))
    rospy.loginfo("Published RViz visualization:")
    rospy.loginfo("  PoseStamped: /manual_audit_se3/pregrasp_pose")
    rospy.loginfo("  PoseStamped: /manual_audit_se3/grasp_pose")
    rospy.loginfo("  MarkerArray: /manual_audit_se3/markers")


def clear_rviz_visualization(args, robot, marker_pub, display_state_pub,
                             display_traj_pub):
    """Clear stale RViz markers and MoveIt preview displays.

    This does not restart RViz or reset MuJoCo. It clears the latched topics
    this script publishes plus the standard MoveIt planned-path topic.
    """
    if args.no_clear_rviz:
        return

    delete_all = Marker()
    delete_all.header.stamp = rospy.Time.now()
    delete_all.header.frame_id = args.viz_frame
    delete_all.action = Marker.DELETEALL
    marker_pub.publish(MarkerArray(markers=[delete_all]))

    empty_traj = DisplayTrajectory()
    empty_traj.model_id = "sawyer"
    empty_traj.trajectory_start = robot.get_current_state()
    display_traj_pub.publish(empty_traj)

    display_state = DisplayRobotState()
    display_state.state = robot.get_current_state()
    display_state_pub.publish(display_state)

    rospy.loginfo("Cleared manual audit RViz markers and MoveIt preview displays")


def main():
    parser = argparse.ArgumentParser(
        description="Manual SE(3) grasp audit in MuJoCo + MoveIt sim")
    parser.add_argument("--scene", required=True,
                        help="Scene YAML name, e.g. scene_desk_se3_me_optimized. "
                             "Launch the corresponding base XML scene first.")
    parser.add_argument("--object", required=True,
                        help="Object short_name to audit.")
    parser.add_argument("--grasp-library", default=default_library_path(PROJECT_ROOT))
    parser.add_argument("--settle", type=float, default=1.0,
                        help="Seconds to wait after reset/teleport/service calls.")
    parser.add_argument("--no-pause", action="store_true",
                        help="Run without waiting for Enter between steps.")
    parser.add_argument("--no-reset", action="store_true",
                        help="Do not call /reset_sim at startup.")
    parser.add_argument("--no-teleport-layout", action="store_true",
                        help="Do not teleport objects from the scene YAML.")
    parser.add_argument("--skip-close-lift", action="store_true",
                        help="Only move to pregrasp and grasp; do not close/lift.")
    parser.set_defaults(exclude_target=True)
    parser.add_argument("--exclude-target", dest="exclude_target",
                        action="store_true",
                        help="Exclude target object from MoveIt planning scene "
                             "only for the grasp step. The target remains in "
                             "the scene during pregrasp planning. This is the "
                             "default.")
    parser.add_argument("--no-exclude-target", dest="exclude_target",
                        action="store_false",
                        help="Keep the target object in the MoveIt planning "
                             "scene even for the grasp step.")
    parser.add_argument("--keep-target-collision", action="store_true",
                        help="Alias for --no-exclude-target.")
    parser.add_argument("--exclude-table", action="store_true",
                        help="Exclude table from MoveIt planning scene. Use only "
                             "for debugging, not final table-collision audit.")
    parser.add_argument("--cartesian", action="store_true",
                        help="Set cartesian=True on MoveToCartesianPose calls.")
    parser.add_argument("--viz-frame", default="base",
                        help="Frame for published RViz poses/markers.")
    parser.add_argument("--no-moveit-preview", action="store_true",
                        help="Do not publish MoveIt goal-state/path previews.")
    parser.add_argument("--no-plan-preview", action="store_true",
                        help="Publish IK goal state only; do not run preview planning.")
    parser.add_argument("--ik-seeds", type=int, default=12,
                        help="Number of IK seeds to try for MoveIt preview.")
    parser.add_argument("--no-ik-diagnose-collisions",
                        dest="ik_diagnose_collisions", action="store_false",
                        default=True,
                        help="Do not retry IK with collisions disabled when "
                             "collision-aware IK fails.")
    parser.add_argument("--no-clear-rviz", action="store_true",
                        help="Do not clear old manual-audit/MoveIt preview "
                             "visualizations at startup.")
    parser.add_argument("--ik-link", default=None,
                        help="Override MoveIt IK link name for previews "
                             "(default: MoveGroup end-effector link).")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    scene_path, scene_cfg, objects_by_name = load_scene(args.scene)
    if args.object not in objects_by_name:
        raise SystemExit(
            f"Object '{args.object}' not in {scene_path}. Available: "
            f"{sorted(objects_by_name)}")

    grasp_lib = GraspLibrary.load(args.grasp_library)
    entry = grasp_lib.get(args.object)
    if entry is None:
        raise SystemExit(
            f"No grasp entry for '{args.object}' in {args.grasp_library}. "
            f"Available: {sorted(grasp_lib.entries)}")

    obj_cfg = objects_by_name[args.object]
    obj_pos, obj_yaw = resolve_object_pose(obj_cfg, scene_cfg["name"])
    poses = entry.resolve(obj_pos, obj_yaw)
    print_pose_summary(args.object, obj_pos, obj_yaw, entry, poses)

    rospy.init_node("manual_audit_se3_grasp", anonymous=True)
    rospy.loginfo("Waiting for simulator/MoveIt services")
    rospy.wait_for_service("/move_to_cartesian_pose", timeout=30.0)
    rospy.wait_for_service("/operate_gripper", timeout=30.0)
    rospy.wait_for_service("/reset_sim", timeout=30.0)
    rospy.wait_for_service("/sim/teleport_object", timeout=30.0)
    if not args.no_moveit_preview:
        rospy.wait_for_service("/compute_ik", timeout=30.0)
    if not args.no_moveit_preview and args.ik_diagnose_collisions:
        rospy.wait_for_service("/check_state_validity", timeout=30.0)

    if not hasattr(moveit_commander, "roscpp_initialize"):
        raise RuntimeError(
            "Imported moveit_commander without roscpp_initialize from "
            f"{getattr(moveit_commander, '__file__', '<unknown>')}. "
            "This usually means a stale workspace package is shadowing "
            "/opt/ros/noetic/lib/python3/dist-packages.")
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    group = moveit_commander.MoveGroupCommander("right_arm")
    group.set_planner_id("RRTConnectkConfigDefault")
    group.set_planning_time(10.0)
    group.set_num_planning_attempts(10)
    group.set_goal_orientation_tolerance(0.26)
    group.set_goal_position_tolerance(0.005)
    rospy.loginfo("MoveIt planning frame: %s", group.get_planning_frame())
    rospy.loginfo("MoveIt default end-effector link: %s",
                  group.get_end_effector_link())
    if args.ik_link:
        rospy.loginfo("Using IK link override for preview: %s", args.ik_link)

    move_srv = rospy.ServiceProxy("/move_to_cartesian_pose", MoveToCartesianPose)
    gripper_srv = rospy.ServiceProxy("/operate_gripper", OperateGripper)
    reset_srv = rospy.ServiceProxy("/reset_sim", Trigger)
    teleport_srv = rospy.ServiceProxy("/sim/teleport_object", TeleportObject)
    exclude_pub = rospy.Publisher("/sim/exclude_from_scene", String, queue_size=1)
    pregrasp_pose_pub = rospy.Publisher(
        "/manual_audit_se3/pregrasp_pose", PoseStamped,
        queue_size=1, latch=True)
    grasp_pose_pub = rospy.Publisher(
        "/manual_audit_se3/grasp_pose", PoseStamped,
        queue_size=1, latch=True)
    marker_pub = rospy.Publisher(
        "/manual_audit_se3/markers", MarkerArray,
        queue_size=1, latch=True)
    display_state_pub = rospy.Publisher(
        "/move_group/display_robot_state", DisplayRobotState,
        queue_size=1, latch=True)
    display_traj_pub = rospy.Publisher(
        "/move_group/display_planned_path", DisplayTrajectory,
        queue_size=1, latch=True)
    rospy.sleep(0.5)

    clear_rviz_visualization(
        args, robot, marker_pub, display_state_pub, display_traj_pub)
    rospy.sleep(0.2)

    reset_and_apply_layout(args, scene_cfg, objects_by_name, reset_srv, teleport_srv)
    publish_visualization(
        args, scene_cfg, obj_cfg, obj_pos, poses,
        pregrasp_pose_pub, grasp_pose_pub, marker_pub)

    rospy.loginfo("Clearing planning-scene exclusions before pregrasp")
    exclude_pub.publish(String(data=""))
    rospy.sleep(0.5)

    rospy.loginfo("Opening gripper")
    gripper_srv(open=True)
    rospy.sleep(args.settle)

    if args.keep_target_collision:
        rospy.loginfo("Keeping target object in MoveIt planning scene for all steps")
    else:
        rospy.loginfo("Keeping target object in MoveIt planning scene for pregrasp")
    if args.exclude_table:
        rospy.logwarn("Excluding table from planning scene")
        exclude_pub.publish(String(data="table"))
        rospy.sleep(0.5)

    publish_moveit_goal_state(
        args, robot, group, display_state_pub, display_traj_pub,
        poses["pregrasp_pos"], poses["pregrasp_quat"], "Pregrasp")
    pause(args, "pregrasp")
    if not call_move(move_srv, poses["pregrasp_pos"], poses["pregrasp_quat"],
                     "Pregrasp", cartesian=args.cartesian):
        rospy.logwarn("Pregrasp failed; continuing only if you choose to proceed")
    rospy.sleep(args.settle)

    if not args.keep_target_collision and args.exclude_target:
        rospy.loginfo("Excluding target object from MoveIt planning scene for grasp: %s",
                      obj_cfg["mujoco_name"])
        exclude_pub.publish(String(data=obj_cfg["mujoco_name"]))
        rospy.sleep(0.5)

    publish_moveit_goal_state(
        args, robot, group, display_state_pub, display_traj_pub,
        poses["grasp_pos"], poses["grasp_quat"], "Grasp")
    pause(args, "grasp")
    if not call_move(move_srv, poses["grasp_pos"], poses["grasp_quat"],
                     "Grasp", cartesian=args.cartesian):
        rospy.logwarn("Grasp failed")
        if args.skip_close_lift:
            return
    rospy.sleep(args.settle)

    if args.skip_close_lift:
        rospy.loginfo("Done after grasp pose because --skip-close-lift was set")
        return

    pause(args, "close gripper")
    rospy.loginfo("Closing gripper")
    gripper_srv(open=False)
    rospy.sleep(args.settle)

    pause(args, "lift to pregrasp")
    call_move(move_srv, poses["pregrasp_pos"], poses["pregrasp_quat"],
              "Lift/pregrasp", cartesian=args.cartesian)
    rospy.sleep(args.settle)

    rospy.loginfo("Manual audit sequence complete")
    rospy.loginfo("To clear planning-scene exclusions: "
                  "rostopic pub -1 /sim/exclude_from_scene std_msgs/String "
                  "\"data: 'clear'\"")


if __name__ == "__main__":
    main()
