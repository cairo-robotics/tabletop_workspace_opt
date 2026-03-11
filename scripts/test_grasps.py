#!/usr/bin/env python3
"""Test candidate grasp poses on scene objects in MuJoCo simulation.

For each candidate grasp, the script:
  1. Resets the simulation to the initial keyframe
  2. Excludes the target object from the MoveIt planning scene
  3. Opens the gripper and moves to a pre-grasp hover pose
  4. Descends to the grasp pose
  5. Closes the gripper
  6. Lifts the object
  7. Checks whether the object followed (z increased >= 5 cm)

Passing grasps are saved to config/grasp_poses.yaml.

Usage:
    rosrun tabletop_workspace_opt test_grasps.py
    rosrun tabletop_workspace_opt test_grasps.py --object banana
"""
import argparse
import sys
import os

_ROS_PYTHON_PATH = '/opt/ros/noetic/lib/python3/dist-packages'
if _ROS_PYTHON_PATH not in sys.path:
    sys.path.insert(0, _ROS_PYTHON_PATH)
else:
    sys.path.insert(0, sys.path.pop(sys.path.index(_ROS_PYTHON_PATH)))

import numpy as np
import yaml
import rospy
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from tabletop_workspace_opt.srv import MoveToCartesianPose, OperateGripper
from std_srvs.srv import Trigger
from scipy.spatial.transform import Rotation

# ---- constants ---------------------------------------------------------------

BASE_Z_OFFSET = 0.92

# Downward EE orientations (xyzw) that the KDL IK solver can actually reach.
# 180° about x-axis: EE z→down, EE y→world -y
DOWN_X = [1.0, 0.0, 0.0, 0.0]
# 180° about y-axis: EE z→down, EE y→world +y
DOWN_Y = [0.0, 1.0, 0.0, 0.0]

# Wrist rotations from DOWN_X (z-axis rotations change finger alignment)
_r_down_x = Rotation.from_quat(DOWN_X)

_r_dx_rot45 = _r_down_x * Rotation.from_euler('z', 45, degrees=True)
DOWN_X_ROT45 = _r_dx_rot45.as_quat().tolist()

_r_dx_rot90 = _r_down_x * Rotation.from_euler('z', 90, degrees=True)
DOWN_X_ROT90 = _r_dx_rot90.as_quat().tolist()

_r_dx_rot_neg45 = _r_down_x * Rotation.from_euler('z', -45, degrees=True)
DOWN_X_ROTN45 = _r_dx_rot_neg45.as_quat().tolist()

# Wrist rotations from DOWN_Y
_r_down_y = Rotation.from_quat(DOWN_Y)

_r_dy_rot45 = _r_down_y * Rotation.from_euler('z', 45, degrees=True)
DOWN_Y_ROT45 = _r_dy_rot45.as_quat().tolist()

_r_dy_rot90 = _r_down_y * Rotation.from_euler('z', 90, degrees=True)
DOWN_Y_ROT90 = _r_dy_rot90.as_quat().tolist()

# Object detection IDs (must match simulation_server publish order)
# Object half-extents in world frame (identity object rotation)
OBJECTS = {
    "cereal": {"det_id": 1, "mujoco_name": "cereal_0_1cd24fb39eb340b28b7a0ce00e6d3c6a",
               "half": [0.04, 0.03, 0.08]},
    "napkin":  {"det_id": 2, "mujoco_name": "napkin_0_d147f73d5f2249a7ba169a1cf0c21e95",
               "half": [0.08, 0.08, 0.005]},
    "spoon":  {"det_id": 3, "mujoco_name": "spoon_0_98c9fc6a50414f68979318c693fe25f8",
               "half": [0.075, 0.015, 0.01]},
    "banana": {"det_id": 4, "mujoco_name": "banana_0_cfd0a5186de3408cbb6fbde0cd6144ce",
               "half": [0.09, 0.03, 0.025]},
}

# ---- candidate grasps --------------------------------------------------------
# offset: [dx,dy,dz] from object centre to EE (world frame)
# orientation: [qx,qy,qz,qw] of the EE
# approach_dz: extra z above grasp pose for pre-grasp hover (world frame)
#
# Gripper geometry: finger centre is ~3cm below EE, fingers span ~68mm along
# approach axis. Max opening ~69mm. Finger bottom is ~6.4cm below EE when
# pointing straight down.


# Gripper geometry (from sawyer-gripper.xml):
#   gripper_body origin = MoveIt EE (right_hand) position
#   Finger body z-offset from gripper_body: 0.045m
#   Finger collision cylinder center offset within finger body: z=0.039m
#   => Finger collision center is ~0.084m below gripper_body origin (EE)
#   Finger collision cylinder half-length: 0.034m
#   => Finger tips at ~0.118m below EE, finger tops at ~0.050m below EE
#
# For a top-down grasp to wrap around an object at height obj_z:
#   EE_z = obj_z + offset_z
#   Finger center height = obj_z + offset_z - 0.084
#   For fingers centered on object: offset_z ≈ 0.084
#   For fingers slightly above center (better grip on tall objects): offset_z ≈ 0.09-0.10
#   Finger tips reach down to: obj_z + offset_z - 0.118

CANDIDATES = {
    "cereal": [
        # Cereal: x=80mm, y=60mm, z=160mm. Fingers along y spans 60mm ✓
        # down_x_mid and down_x_center already saved - try to find a 3rd
        {"name": "down_x_top",     "offset": [0, 0, 0.12],  "orientation": DOWN_X,
         "approach_dz": 0.12},
        {"name": "down_y_top",     "offset": [0, 0, 0.12],  "orientation": DOWN_Y,
         "approach_dz": 0.12},
        {"name": "down_x_r45_mid", "offset": [0, 0, 0.09],  "orientation": DOWN_X_ROT45,
         "approach_dz": 0.10},
        {"name": "down_x_r90_mid", "offset": [0, 0, 0.09],  "orientation": DOWN_X_ROT90,
         "approach_dz": 0.10},
        {"name": "down_x_mid",     "offset": [0, 0, 0.09],  "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_x_center",  "offset": [0, 0, 0.084], "orientation": DOWN_X,
         "approach_dz": 0.10},
    ],
    "banana": [
        # Banana: x=180mm, y=60mm, z=50mm. Fingers along y spans 60mm ✓
        # Short object: fingers need to be near the center
        {"name": "down_x",         "offset": [0, 0, 0.084], "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_y",         "offset": [0, 0, 0.084], "orientation": DOWN_Y,
         "approach_dz": 0.10},
        {"name": "down_x_higher",  "offset": [0, 0, 0.10],  "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_y_higher",  "offset": [0, 0, 0.10],  "orientation": DOWN_Y,
         "approach_dz": 0.10},
        {"name": "down_x_lower",   "offset": [0, 0, 0.07],  "orientation": DOWN_X,
         "approach_dz": 0.10},
    ],
    "spoon": [
        # Spoon: x=150mm, y=30mm, z=20mm. Very thin, near table surface.
        # Finger tips at 118mm below EE. Table at z≈0.915 (world).
        # Need fingers to close around spoon without hitting table.
        {"name": "down_x",         "offset": [0, 0, 0.084], "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_x_low",     "offset": [0, 0, 0.065], "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_x_vlow",    "offset": [0, 0, 0.055], "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_x_r45",     "offset": [0, 0, 0.065], "orientation": DOWN_X_ROT45,
         "approach_dz": 0.10},
        {"name": "down_x_r90",     "offset": [0, 0, 0.065], "orientation": DOWN_X_ROT90,
         "approach_dz": 0.10},
    ],
    "napkin": [
        # Napkin: 160x160x10mm. Only 10mm z-thickness, very close to table.
        # Best prior result: 3cm lift with offset 0.07. Need lower offsets.
        {"name": "down_x_low",     "offset": [0, 0, 0.06],  "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_x_vlow",    "offset": [0, 0, 0.05],  "orientation": DOWN_X,
         "approach_dz": 0.10},
        {"name": "down_y_low",     "offset": [0, 0, 0.06],  "orientation": DOWN_Y,
         "approach_dz": 0.10},
        {"name": "down_y_vlow",    "offset": [0, 0, 0.05],  "orientation": DOWN_Y,
         "approach_dz": 0.10},
        {"name": "down_x_r45",     "offset": [0, 0, 0.055], "orientation": DOWN_X_ROT45,
         "approach_dz": 0.10},
    ],
}

# ---- helpers -----------------------------------------------------------------

def world_to_base(pos):
    return [pos[0], pos[1], pos[2] - BASE_Z_OFFSET]


def get_object_position(det_id, timeout=5.0):
    """Read object world position from /mujoco_sim/detections."""
    try:
        msg = rospy.wait_for_message("/mujoco_sim/detections", Detection2DArray,
                                     timeout=timeout)
    except rospy.ROSException:
        return None
    for det in msg.detections:
        for res in det.results:
            if res.id == det_id:
                p = res.pose.pose.position
                return np.array([p.x, p.y, p.z])
    return None


# ---- test one grasp ----------------------------------------------------------

def test_grasp(object_name, grasp, move_srv, gripper_srv, reset_srv, exclude_pub):
    """Test a single grasp candidate. Returns (success, z_change)."""
    info = OBJECTS[object_name]
    det_id = info["det_id"]
    mujoco_name = info["mujoco_name"]
    grasp_name = grasp["name"]
    offset = np.array(grasp["offset"])
    orientation = grasp["orientation"]
    approach_dz = grasp["approach_dz"]

    rospy.loginfo("--- Testing %s / %s ---", object_name, grasp_name)

    # 1. Reset simulation (also clears exclusions)
    rospy.loginfo("  Resetting simulation...")
    try:
        reset_srv()
    except Exception as e:
        rospy.logerr("  Reset failed: %s", e)
        return False, 0.0
    rospy.sleep(3.0)

    # 2. Open gripper
    gripper_srv(open=True)
    rospy.sleep(1.0)

    # 3. Get object position
    obj_pos = get_object_position(det_id)
    if obj_pos is None:
        rospy.logerr("  Could not read object position")
        return False, 0.0
    rospy.loginfo("  Object world pos: [%.3f, %.3f, %.3f]", *obj_pos)

    # 4. Compute poses in base frame
    grasp_world = obj_pos + offset
    pre_grasp_world = grasp_world.copy()
    pre_grasp_world[2] += approach_dz

    grasp_base = world_to_base(grasp_world)
    pre_grasp_base = world_to_base(pre_grasp_world)

    qx, qy, qz, qw = orientation

    # 5. Exclude ALL objects from planning scene for collision-free motion.
    #    Also exclude the table for thin objects (spoon/napkin) where the
    #    fingers must descend near/below the table surface.
    rospy.loginfo("  Excluding all objects from planning scene...")
    for obj_info in OBJECTS.values():
        exclude_pub.publish(String(data=obj_info["mujoco_name"]))
        rospy.sleep(0.1)
    if object_name in ("spoon", "napkin"):
        exclude_pub.publish(String(data="table"))
        rospy.sleep(0.1)
    rospy.sleep(0.5)

    # 6. Move to pre-grasp
    rospy.loginfo("  Moving to pre-grasp [%.3f, %.3f, %.3f]...", *pre_grasp_base)
    resp = move_srv(x=pre_grasp_base[0], y=pre_grasp_base[1], z=pre_grasp_base[2],
                    qx=qx, qy=qy, qz=qz, qw=qw)
    if not resp.success:
        rospy.logwarn("  Pre-grasp motion failed: %s", resp.message)
        return False, 0.0
    rospy.sleep(8.0)

    # 7. Keep all objects excluded during descent to avoid collision-induced
    #    planning failures (error_code=-6).  The target object is in the
    #    gripper workspace, and nearby objects often block the planner.

    # 8. Move to grasp pose
    rospy.loginfo("  Moving to grasp [%.3f, %.3f, %.3f]...", *grasp_base)
    resp = move_srv(x=grasp_base[0], y=grasp_base[1], z=grasp_base[2],
                    qx=qx, qy=qy, qz=qz, qw=qw)
    if not resp.success:
        rospy.logwarn("  Grasp motion failed: %s", resp.message)
        return False, 0.0
    rospy.sleep(8.0)

    # 9. Close gripper
    rospy.loginfo("  Closing gripper...")
    gripper_srv(open=False)
    rospy.sleep(3.0)

    # 10. Record object position before lift
    obj_before = get_object_position(det_id)
    if obj_before is None:
        rospy.logerr("  Could not read object position before lift")
        return False, 0.0
    rospy.loginfo("  Object before lift: z=%.4f", obj_before[2])

    # 11. Lift
    lift_base = list(grasp_base)
    lift_base[2] += 0.10
    rospy.loginfo("  Lifting to [%.3f, %.3f, %.3f]...", *lift_base)
    resp = move_srv(x=lift_base[0], y=lift_base[1], z=lift_base[2],
                    qx=qx, qy=qy, qz=qz, qw=qw)
    if not resp.success:
        rospy.logwarn("  Lift motion failed: %s", resp.message)
        return False, 0.0
    rospy.sleep(8.0)

    # 12. Check result
    obj_after = get_object_position(det_id)
    if obj_after is None:
        rospy.logerr("  Could not read object position after lift")
        return False, 0.0

    z_change = obj_after[2] - obj_before[2]
    rospy.loginfo("  Object after lift: z=%.4f  (delta=%.4f)", obj_after[2], z_change)

    success = z_change > 0.05
    if success:
        rospy.loginfo("  PASS: object lifted %.1f cm", z_change * 100)
    else:
        rospy.logwarn("  FAIL: object only moved %.1f cm in z", z_change * 100)

    return success, z_change


# ---- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test grasp poses in MuJoCo sim")
    parser.add_argument("--object", type=str, default=None,
                        help="Test only this object (e.g. banana)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("test_grasps", anonymous=True)

    rospy.loginfo("Waiting for services...")
    rospy.wait_for_service("/move_to_cartesian_pose", timeout=15.0)
    rospy.wait_for_service("/operate_gripper", timeout=15.0)
    rospy.wait_for_service("/reset_sim", timeout=15.0)

    move_srv = rospy.ServiceProxy("/move_to_cartesian_pose", MoveToCartesianPose)
    gripper_srv = rospy.ServiceProxy("/operate_gripper", OperateGripper)
    reset_srv = rospy.ServiceProxy("/reset_sim", Trigger)
    exclude_pub = rospy.Publisher("/sim/exclude_from_scene", String, queue_size=1)

    rospy.sleep(1.0)  # let publishers connect

    # Determine which objects to test
    if args.object:
        if args.object not in CANDIDATES:
            rospy.logerr("No candidates for '%s'. Available: %s",
                         args.object, list(CANDIDATES.keys()))
            sys.exit(1)
        test_objects = [args.object]
    else:
        test_objects = list(CANDIDATES.keys())

    # Run tests
    results = {}

    for obj_name in test_objects:
        results[obj_name] = []
        candidates = CANDIDATES[obj_name]
        rospy.loginfo("========== %s (%d candidates) ==========",
                      obj_name.upper(), len(candidates))

        for grasp in candidates:
            if len(results[obj_name]) >= 3:
                rospy.loginfo("  Already have 3 passing grasps, skipping remaining")
                break

            ok, z_change = test_grasp(obj_name, grasp, move_srv, gripper_srv,
                                      reset_srv, exclude_pub)
            if ok:
                results[obj_name].append({
                    "name": grasp["name"],
                    "offset": [float(v) for v in grasp["offset"]],
                    "orientation": [round(v, 6) for v in grasp["orientation"]],
                    "z_lift_achieved": round(float(z_change), 4),
                })

    # Print summary
    rospy.loginfo("========== SUMMARY ==========")
    total_pass = 0
    for obj_name in test_objects:
        passing = results[obj_name]
        total_pass += len(passing)
        names = [g["name"] for g in passing]
        rospy.loginfo("  %-8s %d/%d passed: %s", obj_name,
                      len(passing), len(CANDIDATES[obj_name]),
                      ", ".join(names) if names else "none")

    # Save to YAML — merge with existing file to accumulate across runs
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grasp_poses.yaml")

    # Load existing grasps
    existing = {"grasps": {}}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing = yaml.safe_load(f) or {"grasps": {}}

    # Merge new results with existing (deduplicate by name, cap at 3 per object)
    yaml_data = existing
    for obj_name, passing in results.items():
        if not passing:
            continue
        if obj_name not in yaml_data["grasps"]:
            yaml_data["grasps"][obj_name] = []
        existing_names = {g["name"] for g in yaml_data["grasps"][obj_name]}
        for g in passing:
            if g["name"] not in existing_names and len(yaml_data["grasps"][obj_name]) < 3:
                yaml_data["grasps"][obj_name].append({
                    "name": g["name"],
                    "offset": {"x": g["offset"][0], "y": g["offset"][1],
                               "z": g["offset"][2]},
                    "orientation": {"qx": g["orientation"][0], "qy": g["orientation"][1],
                                    "qz": g["orientation"][2], "qw": g["orientation"][3]},
                })
                existing_names.add(g["name"])

    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    total_saved = sum(len(v) for v in yaml_data["grasps"].values())
    rospy.loginfo("Saved %d total grasps (%d new this run) to %s",
                  total_saved, total_pass, output_path)


if __name__ == "__main__":
    main()
