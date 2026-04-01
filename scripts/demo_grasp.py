#!/usr/bin/env python3
"""Demo a single grasp from config/grasp_poses.yaml.

Picks an object while avoiding collisions with other scene objects, then
verifies the grasp by moving the held object to a different position.

Usage (with sim_moveit.launch already running):
    python3 demo_grasp.py                    # defaults to banana
    python3 demo_grasp.py --object cereal
    python3 demo_grasp.py --object banana --grasp down_y_higher
    python3 demo_grasp.py --object spoon
"""
import argparse
import sys
import os

_ROS_PYTHON_PATH = '/opt/ros/noetic/lib/python3/dist-packages'
if _ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROS_PYTHON_PATH)
else:
    sys.path.insert(0, sys.path.pop(sys.path.index(_ROS_PYTHON_PATH)))

import yaml
import numpy as np
import rospy
from vision_msgs.msg import Detection2DArray
from intera_core_msgs.msg import EndpointState
from std_msgs.msg import String
from tabletop_workspace_opt.srv import MoveToCartesianPose, OperateGripper
from std_srvs.srv import Trigger

BASE_Z_OFFSET = 0.92

# Object detection IDs (must match simulation_server publish order)
OBJECT_DET_IDS = {
    "bowl": 0,
    "cereal": 1,
    "napkin": 2,
    "spoon": 3,
    "banana": 4,
    "milk_carton": 5,
}

OBJECT_MUJOCO_NAMES = {
    "cereal": "cereal_0_1cd24fb39eb340b28b7a0ce00e6d3c6a",
    "banana": "banana_0_cfd0a5186de3408cbb6fbde0cd6144ce",
    "spoon": "spoon_0_98c9fc6a50414f68979318c693fe25f8",
    "napkin": "napkin_0_d147f73d5f2249a7ba169a1cf0c21e95",
    "bowl": "bowl_0_a430d997_0564_4fae_801b_c01693feeee6",
    "milk_carton": "milk_carton_0_8ce748cf2f4a410181650550275650b1",
}


def get_object_position(det_id, timeout=5.0):
    msg = rospy.wait_for_message("/mujoco_sim/detections", Detection2DArray,
                                 timeout=timeout)
    for det in msg.detections:
        for res in det.results:
            if res.id == det_id:
                p = res.pose.pose.position
                return np.array([p.x, p.y, p.z])
    return None


def world_to_base(pos):
    return [pos[0], pos[1], pos[2] - BASE_Z_OFFSET]


def get_ee_pose(timeout=2.0):
    """Read current EE pose from /mujoco_sim/endpoint_state."""
    try:
        msg = rospy.wait_for_message("/mujoco_sim/endpoint_state",
                                      EndpointState, timeout=timeout)
        p = msg.pose.position
        o = msg.pose.orientation
        return np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w])
    except Exception:
        return None, None


def move_and_wait(move_srv, pos, qx, qy, qz, qw, label, settle=8.0,
                  position_fallback=False):
    """Call move service, check success, wait for settling."""
    rospy.loginfo("  %s -> base [%.3f, %.3f, %.3f]", label, *pos)
    resp = move_srv(x=pos[0], y=pos[1], z=pos[2],
                    qx=qx, qy=qy, qz=qz, qw=qw)
    if not resp.success and position_fallback:
        rospy.logwarn("  Oriented plan failed, retrying with position-only...")
        resp = move_srv(x=pos[0], y=pos[1], z=pos[2],
                        qx=0.0, qy=0.0, qz=0.0, qw=0.0)
    if not resp.success:
        rospy.logerr("  Motion failed: %s", resp.message)
        return False
    rospy.sleep(settle)
    # Report actual EE position after settling
    ee_pos, ee_quat = get_ee_pose()
    if ee_pos is not None:
        target_world = [pos[0], pos[1], pos[2] + BASE_Z_OFFSET]
        err = np.array(target_world) - ee_pos
        rospy.loginfo("  %s actual EE world [%.3f, %.3f, %.3f]  err [%.1fmm, %.1fmm, %.1fmm]",
                      label, *ee_pos, err[0]*1000, err[1]*1000, err[2]*1000)
    return True


def main():
    parser = argparse.ArgumentParser(description="Demo a single grasp")
    parser.add_argument("--object", type=str, default="banana",
                        help="Object to grasp (cereal, banana, spoon)")
    parser.add_argument("--grasp", type=str, default=None,
                        help="Grasp name (default: first one for the object)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    # Load grasp config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config", "grasp_poses.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    obj_name = args.object
    if obj_name not in config["grasps"]:
        print(f"No grasps saved for '{obj_name}'. Available: {list(config['grasps'].keys())}")
        sys.exit(1)

    grasps = config["grasps"][obj_name]
    if args.grasp:
        grasp = next((g for g in grasps if g["name"] == args.grasp), None)
        if grasp is None:
            names = [g["name"] for g in grasps]
            print(f"Grasp '{args.grasp}' not found. Available for {obj_name}: {names}")
            sys.exit(1)
    else:
        grasp = grasps[0]

    offset = np.array([grasp["offset"]["x"], grasp["offset"]["y"], grasp["offset"]["z"]])
    qx = grasp["orientation"]["qx"]
    qy = grasp["orientation"]["qy"]
    qz = grasp["orientation"]["qz"]
    qw = grasp["orientation"]["qw"]
    approach_dz = 0.10

    print(f"\n{'='*60}")
    print(f"  DEMO GRASP: {obj_name} / {grasp['name']}")
    print(f"  offset:      [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]")
    print(f"  orientation: [{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]")
    print(f"{'='*60}\n")

    rospy.init_node("demo_grasp", anonymous=True)

    rospy.loginfo("Waiting for services...")
    rospy.wait_for_service("/move_to_cartesian_pose", timeout=15.0)
    rospy.wait_for_service("/operate_gripper", timeout=15.0)
    rospy.wait_for_service("/reset_sim", timeout=15.0)

    move_srv = rospy.ServiceProxy("/move_to_cartesian_pose", MoveToCartesianPose)
    gripper_srv = rospy.ServiceProxy("/operate_gripper", OperateGripper)
    reset_srv = rospy.ServiceProxy("/reset_sim", Trigger)
    exclude_pub = rospy.Publisher("/sim/exclude_from_scene", String, queue_size=1)
    rospy.sleep(1.0)

    det_id = OBJECT_DET_IDS[obj_name]
    target_mujoco = OBJECT_MUJOCO_NAMES[obj_name]

    # ---- Step 1: Reset ----
    rospy.loginfo("Step 1: Resetting simulation...")
    reset_srv()
    rospy.sleep(3.0)

    # ---- Step 2: Open gripper ----
    rospy.loginfo("Step 2: Opening gripper...")
    gripper_srv(open=True)
    rospy.sleep(1.0)

    # ---- Step 3: Read object position ----
    obj_pos = get_object_position(det_id)
    if obj_pos is None:
        rospy.logerr("Could not find object '%s'", obj_name)
        return
    rospy.loginfo("Step 3: Object at world [%.3f, %.3f, %.3f]", *obj_pos)

    # ---- Step 4: Exclude ONLY the target object from MoveIt planning scene ----
    # All other objects remain as collision obstacles so MoveIt plans around them.
    rospy.loginfo("Step 4: Excluding target '%s' from planning scene...", obj_name)
    exclude_pub.publish(String(data=target_mujoco))
    rospy.sleep(0.3)
    # Also exclude the table for thin objects whose grasp goes near the surface.
    if obj_name in ("spoon", "napkin"):
        exclude_pub.publish(String(data="table"))
        rospy.sleep(0.1)
    rospy.sleep(0.5)

    # Compute poses
    grasp_world = obj_pos + offset
    pre_grasp_world = grasp_world.copy()
    pre_grasp_world[2] += approach_dz

    pre_grasp_base = world_to_base(pre_grasp_world)
    grasp_base = world_to_base(grasp_world)

    # ---- Step 5 + 6: Approach and descend (with retries) ----
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        rospy.loginfo("Attempt %d/%d", attempt, max_attempts)

        if attempt > 1:
            # Re-reset and re-open gripper for retry
            rospy.loginfo("  Resetting for retry...")
            reset_srv()
            rospy.sleep(3.0)
            gripper_srv(open=True)
            rospy.sleep(1.0)
            # Re-read object position (may have shifted)
            obj_pos = get_object_position(det_id)
            if obj_pos is None:
                rospy.logerr("Lost object after reset")
                return
            grasp_world = obj_pos + offset
            pre_grasp_world = grasp_world.copy()
            pre_grasp_world[2] += approach_dz
            pre_grasp_base = world_to_base(pre_grasp_world)
            grasp_base = world_to_base(grasp_world)
            # Re-exclude target
            exclude_pub.publish(String(data=target_mujoco))
            rospy.sleep(0.3)
            if obj_name in ("spoon", "napkin"):
                exclude_pub.publish(String(data="table"))
                rospy.sleep(0.1)
            rospy.sleep(0.5)

        rospy.loginfo("Step 5: Moving to pre-grasp hover...")
        if not move_and_wait(move_srv, pre_grasp_base, qx, qy, qz, qw, "Pre-grasp"):
            continue

        rospy.loginfo("Step 6: Descending to grasp pose...")
        if not move_and_wait(move_srv, grasp_base, qx, qy, qz, qw, "Grasp",
                             position_fallback=True):
            continue

        # Check if object is still near expected position (wasn't knocked away)
        obj_at_grasp = get_object_position(det_id)
        if obj_at_grasp is not None:
            dist_from_start = np.linalg.norm(obj_at_grasp[:2] - obj_pos[:2])
            if dist_from_start > 0.05:  # >5cm displacement = knocked aside
                rospy.logwarn("  Object displaced %.0fmm — retrying...", dist_from_start * 1000)
                continue

        break  # Both moves succeeded, proceed to grasp
    else:
        rospy.logerr("All %d attempts failed.", max_attempts)
        return

    # ---- Step 6b: Verify object is between fingers ----
    obj_at_grasp = get_object_position(det_id)
    ee_pos_at_grasp, _ = get_ee_pose()
    if obj_at_grasp is not None and ee_pos_at_grasp is not None:
        # Finger centers are 84mm below EE (right_hand) when pointing down
        finger_center_z = ee_pos_at_grasp[2] - 0.084
        obj_to_finger = obj_at_grasp[2] - finger_center_z
        rospy.loginfo("  Object world z=%.3f, Finger center z=%.3f, diff=%.1fmm",
                      obj_at_grasp[2], finger_center_z, obj_to_finger*1000)
        rospy.loginfo("  Object-EE xy offset: [%.1fmm, %.1fmm]",
                      (obj_at_grasp[0]-ee_pos_at_grasp[0])*1000,
                      (obj_at_grasp[1]-ee_pos_at_grasp[1])*1000)

    # ---- Step 7: Close gripper ----
    rospy.loginfo("Step 7: Closing gripper...")
    gripper_srv(open=False)
    rospy.sleep(3.0)

    obj_after_close = get_object_position(det_id)
    if obj_after_close is not None:
        rospy.loginfo("  Object after close: [%.3f, %.3f, %.3f]", *obj_after_close)

    # ---- Step 8: Lift ----
    lift_base = list(grasp_base)
    lift_base[2] += 0.15
    rospy.loginfo("Step 8: Lifting...")
    if not move_and_wait(move_srv, lift_base, qx, qy, qz, qw, "Lift"):
        return

    obj_after_lift = get_object_position(det_id)
    if obj_after_close is not None and obj_after_lift is not None:
        dz = obj_after_lift[2] - obj_after_close[2]
        rospy.loginfo("  Lift z-change: %.1f cm  %s",
                      dz * 100, "LIFTED" if dz > 0.05 else "NOT LIFTED")
        if dz < 0.05:
            rospy.logerr("Grasp failed - object did not lift. Aborting.")
            return
    else:
        rospy.logwarn("Could not verify lift")

    # ---- Step 9: Verify grasp — move laterally and check object follows ----
    rospy.loginfo("Step 9: Verifying grasp — moving laterally...")

    # Move +10cm in y while keeping the lifted height
    verify_base_1 = list(lift_base)
    verify_base_1[1] += 0.10
    if not move_and_wait(move_srv, verify_base_1, qx, qy, qz, qw, "Verify-1 (+y)"):
        return

    obj_v1 = get_object_position(det_id)
    if obj_v1 is not None:
        dy1 = obj_v1[1] - obj_after_lift[1]
        rospy.loginfo("  Object y-shift: %.1f cm (expected ~10 cm)  %s",
                      dy1 * 100,
                      "OK" if abs(dy1) > 0.05 else "SLIP!")

    # Move -20cm in y (to the other side)
    verify_base_2 = list(lift_base)
    verify_base_2[1] -= 0.10
    if not move_and_wait(move_srv, verify_base_2, qx, qy, qz, qw, "Verify-2 (-y)"):
        return

    obj_v2 = get_object_position(det_id)
    if obj_v2 is not None and obj_v1 is not None:
        dy2 = obj_v2[1] - obj_v1[1]
        rospy.loginfo("  Object y-shift: %.1f cm (expected ~-20 cm)  %s",
                      dy2 * 100,
                      "OK" if abs(dy2) > 0.10 else "SLIP!")

    # Final z-check: object should still be well above where it started
    if obj_v2 is not None and obj_after_close is not None:
        final_dz = obj_v2[2] - obj_after_close[2]
        if final_dz > 0.05:
            rospy.loginfo("  GRASP VERIFIED: object still held (z +%.1f cm from start)", final_dz * 100)
        else:
            rospy.logerr("  GRASP FAILED: object dropped (z %.1f cm from start)", final_dz * 100)

    # ---- Hold so you can see it ----
    rospy.loginfo("Holding for 10 seconds — look at the MuJoCo window!")
    rospy.sleep(10.0)

    rospy.loginfo("Done.")


if __name__ == "__main__":
    main()
