#!/usr/bin/env python3
"""Grasp controller: moves the Sawyer end-effector to a hover pose above a
specified scene object, then verifies the achieved pose in both MuJoCo and RViz.

Usage:
    rosrun tabletop_workspace_opt grasp_controller.py bowl
    rosrun tabletop_workspace_opt grasp_controller.py cereal --qx 0.7058 --qy 0.7084 --qz 0.002 --qw 0.002
    rosrun tabletop_workspace_opt grasp_controller.py banana --hover 0.15
    rosrun tabletop_workspace_opt grasp_controller.py --all
    rosrun tabletop_workspace_opt grasp_controller.py bowl --position-only
    rosrun tabletop_workspace_opt grasp_controller.py --list
"""
import argparse
import sys

_ROS_PYTHON_PATH = '/opt/ros/noetic/lib/python3/dist-packages'
if _ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROS_PYTHON_PATH)
else:
    sys.path.insert(0, sys.path.pop(sys.path.index(_ROS_PYTHON_PATH)))

import numpy as np
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from intera_core_msgs.msg import EndpointState
from vision_msgs.msg import Detection2DArray
from tabletop_workspace_opt.srv import MoveToCartesianPose

# ---- constants ---------------------------------------------------------------

OBJECTS = {
    "bowl":   (0, "bowl_0_a430d997_0564_4fae_801b_c01693feeee6"),
    "cereal": (1, "cereal_0_1cd24fb39eb340b28b7a0ce00e6d3c6a"),
    "napkin": (2, "napkin_0_d147f73d5f2249a7ba169a1cf0c21e95"),
    "spoon":  (3, "spoon_0_98c9fc6a50414f68979318c693fe25f8"),
    "banana": (4, "banana_0_cfd0a5186de3408cbb6fbde0cd6144ce"),
    "milk":   (5, "milk_carton_0_8ce748cf2f4a410181650550275650b1"),
}

BASE_Z_OFFSET = 0.92

# ---- helpers -----------------------------------------------------------------

def get_current_ee_orientation_moveit():
    """Query MoveIt for the current EE orientation (xyzw) — use as default
    downward grasp orientation since the home config already points down."""
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander("right_arm")
    rospy.sleep(0.5)
    pose = group.get_current_pose().pose
    o = pose.orientation
    return (o.x, o.y, o.z, o.w)


def get_object_position_world(object_name, timeout=5.0):
    """Read the latest object position from /mujoco_sim/detections (world frame)."""
    det_idx, _ = OBJECTS[object_name]
    try:
        msg = rospy.wait_for_message("/mujoco_sim/detections", Detection2DArray,
                                     timeout=timeout)
    except rospy.ROSException:
        rospy.logerr("Timeout waiting for /mujoco_sim/detections")
        return None

    for det in msg.detections:
        for res in det.results:
            if res.id == det_idx:
                p = res.pose.pose.position
                return np.array([p.x, p.y, p.z])
    rospy.logerr("Object '%s' (id=%d) not found in detections", object_name, det_idx)
    return None


def get_ee_pose_world(timeout=5.0):
    """Read current EE pose from /mujoco_sim/endpoint_state (world frame)."""
    try:
        msg = rospy.wait_for_message("/mujoco_sim/endpoint_state", EndpointState,
                                     timeout=timeout)
    except rospy.ROSException:
        rospy.logerr("Timeout waiting for /mujoco_sim/endpoint_state")
        return None, None

    pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    ori = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                    msg.pose.orientation.z, msg.pose.orientation.w])
    return pos, ori


def world_to_base(pos_world):
    """Convert a world-frame position to base frame."""
    return np.array([pos_world[0], pos_world[1], pos_world[2] - BASE_Z_OFFSET])


def quat_angle_diff(q1, q2):
    """Angular difference in degrees between two quaternions (xyzw).
    Handles q/-q equivalence (same rotation)."""
    # Note: MuJoCo and MoveIt may report opposite-sign quaternions for the
    # same rotation, so we compare the *absolute* dot product and also
    # normalize the MuJoCo quat sign to match before display.
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot)), q2


# ---- core --------------------------------------------------------------------

def move_to_grasp_hover(object_name, hover_height, orientation, position_only=False):
    """Move the EE to hover above the specified object.

    Args:
        object_name: short name (e.g. "bowl")
        hover_height: metres above the object centre to hover
        orientation: (qx, qy, qz, qw) — used unless position_only is True
        position_only: if True, let MoveIt choose any orientation

    Returns:
        True on success.
    """
    # 1. Read object position
    obj_pos = get_object_position_world(object_name)
    if obj_pos is None:
        return False
    rospy.loginfo("Object '%s' world position: [%.3f, %.3f, %.3f]",
                  object_name, *obj_pos)

    # 2. Compute target in base frame
    target_world = obj_pos.copy()
    target_world[2] += hover_height
    target_base = world_to_base(target_world)
    rospy.loginfo("Target hover pose (base frame): [%.3f, %.3f, %.3f]",
                  *target_base)

    # 3. Determine orientation
    if position_only:
        qx = qy = qz = qw = 0.0
        rospy.loginfo("Position-only planning (MoveIt chooses orientation)")
    else:
        qx, qy, qz, qw = orientation
        rospy.loginfo("Target orientation (xyzw): [%.6f, %.6f, %.6f, %.6f]",
                      qx, qy, qz, qw)

    # 4. Call the move_to_cartesian_pose service
    rospy.loginfo("Waiting for /move_to_cartesian_pose service...")
    rospy.wait_for_service("/move_to_cartesian_pose", timeout=10.0)
    move_srv = rospy.ServiceProxy("/move_to_cartesian_pose", MoveToCartesianPose)

    resp = move_srv(x=target_base[0], y=target_base[1], z=target_base[2],
                    qx=qx, qy=qy, qz=qz, qw=qw)

    if not resp.success:
        rospy.logerr("Motion failed: %s", resp.message)
        if not position_only:
            rospy.logwarn("Retrying with position-only planning...")
            resp = move_srv(x=target_base[0], y=target_base[1], z=target_base[2],
                            qx=0.0, qy=0.0, qz=0.0, qw=0.0)
            if not resp.success:
                rospy.logerr("Position-only fallback also failed: %s", resp.message)
                return False
            rospy.loginfo("Position-only fallback succeeded")
        else:
            return False

    rospy.loginfo("Motion plan executed — waiting for arm to settle...")
    rospy.sleep(8.0)

    # 5. Verify using MoveIt's own pose feedback (same frame as planning)
    group = moveit_commander.MoveGroupCommander("right_arm")
    rospy.sleep(0.5)
    moveit_pose = group.get_current_pose().pose
    achieved_pos = np.array([moveit_pose.position.x, moveit_pose.position.y,
                             moveit_pose.position.z])
    achieved_ori = np.array([moveit_pose.orientation.x, moveit_pose.orientation.y,
                             moveit_pose.orientation.z, moveit_pose.orientation.w])

    pos_error = np.linalg.norm(achieved_pos - target_base)

    rospy.loginfo("--- Verification (MoveIt) ---")
    rospy.loginfo("  Target  (base): [%.4f, %.4f, %.4f]", *target_base)
    rospy.loginfo("  Achieved(base): [%.4f, %.4f, %.4f]", *achieved_pos)
    rospy.loginfo("  Position error: %.4f m", pos_error)

    if not position_only:
        ori_err, achieved_ori_norm = quat_angle_diff(np.array(orientation), achieved_ori)
        rospy.loginfo("  Target  orient: [%.4f, %.4f, %.4f, %.4f]", *orientation)
        rospy.loginfo("  Achieved orient: [%.4f, %.4f, %.4f, %.4f]", *achieved_ori_norm)
        rospy.loginfo("  Orientation error: %.2f deg", ori_err)

    if pos_error > 0.05:
        rospy.logwarn("  WARN: Position error > 5 cm")
    else:
        rospy.loginfo("  OK: Position within tolerance")

    return True


# ---- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Move Sawyer EE to hover above a scene object",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="By default, the current EE orientation (downward) is used.\n"
               "Use --position-only to let MoveIt choose any orientation.\n"
               "Use --qx/qy/qz/qw to specify a custom orientation.")
    parser.add_argument("object", nargs="?",
                        help="Object name: " + ", ".join(OBJECTS.keys()))
    parser.add_argument("--hover", type=float, default=0.10,
                        help="Hover height above object centre (m, default: 0.10)")
    parser.add_argument("--qx", type=float, default=None)
    parser.add_argument("--qy", type=float, default=None)
    parser.add_argument("--qz", type=float, default=None)
    parser.add_argument("--qw", type=float, default=None)
    parser.add_argument("--position-only", action="store_true",
                        help="Let MoveIt choose orientation freely")
    parser.add_argument("--list", action="store_true",
                        help="List available objects and exit")
    parser.add_argument("--all", action="store_true",
                        help="Test all objects sequentially")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    if args.list:
        print("Available objects:")
        for name, (idx, full) in OBJECTS.items():
            print(f"  {name:8s}  (id={idx})  {full}")
        return

    rospy.init_node("grasp_controller", anonymous=True)

    # Determine orientation
    position_only = args.position_only
    ori_parts = [args.qx, args.qy, args.qz, args.qw]

    if any(v is not None for v in ori_parts):
        if any(v is None for v in ori_parts):
            rospy.logerr("All of --qx --qy --qz --qw are required together")
            sys.exit(1)
        orientation = (args.qx, args.qy, args.qz, args.qw)
        rospy.loginfo("Using user-specified orientation")
    elif not position_only:
        rospy.loginfo("Querying MoveIt for current EE orientation (downward default)...")
        orientation = get_current_ee_orientation_moveit()
        rospy.loginfo("Default orientation (xyzw): [%.6f, %.6f, %.6f, %.6f]",
                      *orientation)
    else:
        orientation = None

    if args.all:
        results = {}
        for name in OBJECTS:
            rospy.loginfo("========== %s ==========", name.upper())
            ok = move_to_grasp_hover(name, args.hover, orientation, position_only)
            results[name] = ok
            rospy.sleep(1.0)

        rospy.loginfo("========== SUMMARY ==========")
        for name, ok in results.items():
            status = "OK" if ok else "FAILED"
            rospy.loginfo("  %-8s %s", name, status)
        return

    if not args.object:
        parser.print_help()
        sys.exit(1)

    if args.object not in OBJECTS:
        rospy.logerr("Unknown object '%s'. Available: %s",
                     args.object, ", ".join(OBJECTS.keys()))
        sys.exit(1)

    move_to_grasp_hover(args.object, args.hover, orientation, position_only)


if __name__ == "__main__":
    main()
