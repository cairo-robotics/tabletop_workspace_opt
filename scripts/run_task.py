#!/usr/bin/env python3
"""General task executor for tabletop scenes.

Reads a task config YAML and executes a sequence of pick/place actions,
then verifies goal conditions by checking object positions in MuJoCo.

Usage (with sim_moveit.launch already running):
    python3 run_task.py config/tasks/banana_in_bowl.yaml
    python3 run_task.py config/tasks/desk_organize.yaml --scene scene_desk
    python3 run_task.py config/tasks/kitchen_prep_full.yaml --scene scene_kitchen_prep
    python3 run_task.py config/tasks/banana_in_bowl.yaml --no-reset
"""
import argparse
import sys
import os
import glob

# catkin_make_isolated creates separate devel spaces per package.
# Add all Python paths so we can import intera_core_msgs, vision_msgs, etc.
_ws = os.path.expanduser("~/sawyer_ws/devel_isolated")
for _d in glob.glob(os.path.join(_ws, "*/lib/python3/dist-packages")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

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
from sensor_msgs.msg import JointState as JointStateMsg
from std_srvs.srv import Trigger

BASE_Z_OFFSET = 0.92
APPROACH_DZ = 0.10       # hover height above grasp/place pose
SETTLE_TIME = 8.0        # seconds to wait after each move

# Safe home joint configuration (arm tucked, clear of table)
HOME_JOINTS = [0.0, -1.1775, 0.0, 2.1761, 0.0, 0.5663, 3.3124]
GRIPPER_CLOSE_TIME = 3.0
LIFT_HEIGHT = 0.15       # how high to lift after grasping
PLACE_RELEASE_DZ = 0.04  # lower this much below place hover before releasing
MAX_PICK_ATTEMPTS = 5

# ---------------------------------------------------------------------------
# Scene-aware object config loading
# ---------------------------------------------------------------------------

def _load_scene_config(scene_name):
    """Load object config from scene YAML. Returns (det_ids, mujoco_names, half_extents)."""
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(pkg_root, 'config', 'scenes', f'{scene_name}.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            scene_cfg = yaml.safe_load(f)['scene']
        det_ids = {}
        mujoco_names = {}
        half_extents = {}
        for i, obj in enumerate(scene_cfg['objects']):
            short = obj['short_name']
            det_ids[short] = i
            mujoco_names[short] = obj['mujoco_name']
            half_extents[short] = obj['half_extents']
        return det_ids, mujoco_names, half_extents
    else:
        return None, None, None


# Default: breakfast scene (fallback if no --scene arg or no YAML)
OBJECT_DET_IDS = {
    "bowl": 0, "cereal": 1, "napkin": 2,
    "spoon": 3, "banana": 4, "milk_carton": 5,
}

OBJECT_MUJOCO_NAMES = {
    "cereal": "cereal_0_1cd24fb39eb340b28b7a0ce00e6d3c6a",
    "banana": "banana_0_cfd0a5186de3408cbb6fbde0cd6144ce",
    "spoon": "spoon_0_98c9fc6a50414f68979318c693fe25f8",
    "napkin": "napkin_0_d147f73d5f2249a7ba169a1cf0c21e95",
    "bowl": "bowl_0_a430d997_0564_4fae_801b_c01693feeee6",
    "milk_carton": "milk_carton_0_8ce748cf2f4a410181650550275650b1",
}

# Half-extents [x, y, z] in metres — must match scene XML
OBJECT_HALF_EXTENTS = {
    "bowl": [0.08, 0.08, 0.04],
    "cereal": [0.035, 0.023, 0.08],
    "napkin": [0.08, 0.08, 0.005],
    "spoon": [0.075, 0.015, 0.01],
    "banana": [0.09, 0.02, 0.025],
    "milk_carton": [0.04, 0.04, 0.11],
}


# ---------------------------------------------------------------------------
# Sensor helpers
# ---------------------------------------------------------------------------

def get_all_object_positions(timeout=5.0):
    """Return dict mapping object short name -> np.array([x, y, z]) world."""
    msg = rospy.wait_for_message("/mujoco_sim/detections", Detection2DArray,
                                 timeout=timeout)
    id_to_name = {v: k for k, v in OBJECT_DET_IDS.items()}
    positions = {}
    for det in msg.detections:
        for res in det.results:
            name = id_to_name.get(res.id)
            if name:
                p = res.pose.pose.position
                positions[name] = np.array([p.x, p.y, p.z])
    return positions


def get_object_position(name, timeout=5.0):
    det_id = OBJECT_DET_IDS[name]
    msg = rospy.wait_for_message("/mujoco_sim/detections", Detection2DArray,
                                 timeout=timeout)
    for det in msg.detections:
        for res in det.results:
            if res.id == det_id:
                p = res.pose.pose.position
                return np.array([p.x, p.y, p.z])
    return None


def get_ee_pose(timeout=2.0):
    try:
        msg = rospy.wait_for_message("/mujoco_sim/endpoint_state",
                                      EndpointState, timeout=timeout)
        p = msg.pose.position
        return np.array([p.x, p.y, p.z])
    except Exception:
        return None


def world_to_base(pos):
    return [pos[0], pos[1], pos[2] - BASE_Z_OFFSET]


def snapshot_object_positions():
    """Snapshot all object positions for collision detection."""
    return get_all_object_positions()


def detect_displacements(before, after, threshold_mm=5.0):
    """Compare object snapshots and report any displaced objects.

    Returns list of (name, displacement_mm) for objects displaced > threshold.
    """
    displaced = []
    for name in before:
        if name not in after:
            continue
        dist = np.linalg.norm(before[name] - after[name]) * 1000  # mm
        if dist > threshold_mm:
            displaced.append((name, dist))
    return displaced


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------

def move_and_wait(move_srv, pos, qx, qy, qz, qw, label,
                  settle=SETTLE_TIME, position_fallback=False,
                  cartesian=False):
    rospy.loginfo("  %s -> base [%.3f, %.3f, %.3f]%s", label, *pos,
                  " (cartesian)" if cartesian else "")
    resp = move_srv(x=pos[0], y=pos[1], z=pos[2],
                    qx=qx, qy=qy, qz=qz, qw=qw,
                    cartesian=cartesian)
    if not resp.success and cartesian:
        rospy.logwarn("  Cartesian path failed, falling back to RRT...")
        resp = move_srv(x=pos[0], y=pos[1], z=pos[2],
                        qx=qx, qy=qy, qz=qz, qw=qw,
                        cartesian=False)
    if not resp.success and position_fallback:
        rospy.logwarn("  Oriented plan failed, retrying position-only...")
        resp = move_srv(x=pos[0], y=pos[1], z=pos[2],
                        qx=0.0, qy=0.0, qz=0.0, qw=0.0)
    if not resp.success:
        rospy.logerr("  Motion failed: %s", resp.message)
        return False
    rospy.sleep(settle)
    return True


# ---------------------------------------------------------------------------
# High-level primitives
# ---------------------------------------------------------------------------

class TaskExecutor:
    def __init__(self, grasp_config_path):
        rospy.loginfo("Waiting for services...")
        rospy.wait_for_service("/move_to_cartesian_pose", timeout=15.0)
        rospy.wait_for_service("/operate_gripper", timeout=15.0)
        rospy.wait_for_service("/reset_sim", timeout=15.0)

        self.move_srv = rospy.ServiceProxy("/move_to_cartesian_pose",
                                           MoveToCartesianPose)
        self.gripper_srv = rospy.ServiceProxy("/operate_gripper",
                                              OperateGripper)
        self.reset_srv = rospy.ServiceProxy("/reset_sim", Trigger)
        self.exclude_pub = rospy.Publisher("/sim/exclude_from_scene",
                                          String, queue_size=1)
        self.joint_pub = rospy.Publisher(
            "relaxed_ik/joint_angle_solutions", JointStateMsg, queue_size=10)
        rospy.sleep(1.0)

        with open(grasp_config_path, 'r') as f:
            self.grasp_config = yaml.safe_load(f)

        self.holding = None  # name of currently held object (or None)
        self.steps_completed = 0  # track progress for reset decisions

    # ---- retract helpers ----

    def return_home(self):
        """Move the arm to a safe retracted position above the table.

        Uses MoveIt RRT planning to avoid collisions during the retract,
        targeting a high position above the table center.
        """
        rospy.loginfo("  Returning to safe home position...")
        self._include_all()
        rospy.sleep(0.3)
        # High position above table center (well clear of all objects)
        safe_pos = [0.3, 0.0, 0.35]
        return move_and_wait(self.move_srv, safe_pos,
                             0.0, 0.0, 0.0, 0.0, "Home-retract",
                             settle=3.0)

    # ---- exclusion helpers ----

    def _exclude(self, obj_name):
        mujoco_name = OBJECT_MUJOCO_NAMES.get(obj_name)
        if mujoco_name:
            self.exclude_pub.publish(String(data=mujoco_name))
            rospy.sleep(0.05)

    def _exclude_table(self):
        self.exclude_pub.publish(String(data="table"))
        rospy.sleep(0.05)

    def _include_table(self):
        self.exclude_pub.publish(String(data="-table"))
        rospy.sleep(0.05)

    def _exclude_nearby(self, target_xy, radius=0.10, skip=None):
        """Exclude objects within `radius` of target_xy in the XY plane.

        This gives the RRT planner freedom near the descent target while
        keeping distant objects as collision obstacles.
        """
        if skip is None:
            skip = set()
        positions = get_all_object_positions()
        for name, pos in positions.items():
            if name in skip:
                continue
            dist = np.linalg.norm(np.array(target_xy) - pos[:2])
            if dist < radius:
                rospy.loginfo("  Excluding nearby '%s' (%.0fmm from target)",
                              name, dist * 1000)
                self._exclude(name)

    def _exclude_all(self):
        for name in OBJECT_MUJOCO_NAMES.values():
            self.exclude_pub.publish(String(data=name))
            rospy.sleep(0.05)
        self.exclude_pub.publish(String(data="table"))
        rospy.sleep(0.3)

    def _include_all(self):
        """Re-include everything by publishing empty string (clears set)."""
        self.exclude_pub.publish(String(data=""))
        rospy.sleep(0.3)

    # ---- PICK primitive ----

    def pick(self, obj_name):
        rospy.loginfo("=" * 50)
        rospy.loginfo("PICK: %s", obj_name)
        rospy.loginfo("=" * 50)

        if obj_name not in self.grasp_config.get("grasps", {}):
            rospy.logerr("No grasp config for '%s'", obj_name)
            return False

        grasp = self.grasp_config["grasps"][obj_name][0]
        offset = np.array([grasp["offset"]["x"],
                           grasp["offset"]["y"],
                           grasp["offset"]["z"]])
        qx = grasp["orientation"]["qx"]
        qy = grasp["orientation"]["qy"]
        qz = grasp["orientation"]["qz"]
        qw = grasp["orientation"]["qw"]

        # Open gripper
        self.gripper_srv(open=True)
        rospy.sleep(1.0)

        for attempt in range(1, MAX_PICK_ATTEMPTS + 1):
            rospy.loginfo("  Pick attempt %d/%d", attempt, MAX_PICK_ATTEMPTS)

            # Read object position
            obj_pos = get_object_position(obj_name)
            if obj_pos is None:
                rospy.logerr("  Cannot find '%s'", obj_name)
                return False
            rospy.loginfo("  Object at world [%.3f, %.3f, %.3f]", *obj_pos)

            # All objects + table in scene → RRT avoids everything
            self._include_all()
            rospy.sleep(0.5)

            # Compute poses
            grasp_world = obj_pos + offset
            pre_grasp_world = grasp_world.copy()
            pre_grasp_world[2] += APPROACH_DZ + 0.05  # extra clearance
            pre_grasp_base = world_to_base(pre_grasp_world)
            grasp_base = world_to_base(grasp_world)

            # Snapshot positions BEFORE pre-grasp approach
            snap_before = snapshot_object_positions()

            # Move to pre-grasp hover (all objects + table are collision
            # obstacles, so RRT plans a collision-free trajectory)
            if not move_and_wait(self.move_srv, pre_grasp_base,
                                 qx, qy, qz, qw, "Pre-grasp"):
                continue

            # Check for collisions during pre-grasp approach
            snap_after = snapshot_object_positions()
            displaced = detect_displacements(snap_before, snap_after)
            if displaced:
                rospy.logwarn("  COLLISION DETECTED during pre-grasp:")
                for name, dist_mm in displaced:
                    rospy.logwarn("    %s displaced %.0fmm", name, dist_mm)

            # Re-read object position after approach (it may have shifted)
            obj_after_approach = get_object_position(obj_name)
            if obj_after_approach is not None:
                approach_disp = np.linalg.norm(
                    obj_after_approach[:2] - obj_pos[:2])
                rospy.loginfo("  After approach: obj disp %.0fmm",
                              approach_disp * 1000)
                if approach_disp > 0.005:
                    rospy.loginfo("  Updating grasp target to current obj pos")
                    grasp_world = obj_after_approach + offset
                    grasp_base = world_to_base(grasp_world)

            # Exclude the target object (gripper enters its collision
            # volume), the table (arm links pass close to it), and
            # nearby objects that constrain the RRT too much.
            # Distant objects remain as collision obstacles.
            self._exclude(obj_name)
            self._exclude_table()
            self._exclude_nearby(grasp_world[:2], radius=0.10,
                                 skip={obj_name})
            rospy.sleep(0.5)

            # Snapshot before descent
            snap_before_descent = snapshot_object_positions()

            # Descend to grasp pose using Cartesian straight-line path
            if not move_and_wait(self.move_srv, grasp_base,
                                 qx, qy, qz, qw, "Grasp",
                                 position_fallback=True,
                                 cartesian=True):
                continue

            # Check for collisions during descent
            snap_after_descent = snapshot_object_positions()
            displaced = detect_displacements(snap_before_descent,
                                             snap_after_descent)
            if displaced:
                rospy.logwarn("  COLLISION DETECTED during descent:")
                for name, dist_mm in displaced:
                    rospy.logwarn("    %s displaced %.0fmm", name, dist_mm)

            # Check object wasn't knocked away
            obj_now = get_object_position(obj_name)
            if obj_now is not None:
                disp = np.linalg.norm(obj_now[:2] - obj_pos[:2])
                if disp > 0.05:
                    if self.steps_completed == 0:
                        rospy.logwarn("  Object displaced %.0fmm, resetting",
                                     disp * 1000)
                        self.reset_srv()
                        rospy.sleep(3.0)
                        self.gripper_srv(open=True)
                        rospy.sleep(1.0)
                    else:
                        rospy.logwarn("  Object displaced %.0fmm (mid-task, "
                                     "no reset)", disp * 1000)
                    continue

            # Close gripper
            rospy.loginfo("  Closing gripper...")
            self.gripper_srv(open=False)
            rospy.sleep(GRIPPER_CLOSE_TIME)

            # Snapshot before lift
            snap_before_lift = snapshot_object_positions()

            # Lift using Cartesian straight-line path (straight up)
            lift_base = list(grasp_base)
            lift_base[2] += LIFT_HEIGHT
            if not move_and_wait(self.move_srv, lift_base,
                                 qx, qy, qz, qw, "Lift",
                                 cartesian=True):
                return False

            # Check for collisions during lift
            snap_after_lift = snapshot_object_positions()
            displaced = detect_displacements(snap_before_lift, snap_after_lift)
            if displaced:
                rospy.logwarn("  COLLISION DETECTED during lift:")
                for name, dist_mm in displaced:
                    if name != obj_name:  # skip the held object
                        rospy.logwarn("    %s displaced %.0fmm", name, dist_mm)

            # Re-include table now that we're above the surface
            self._include_table()
            rospy.sleep(0.3)

            # Verify lift
            obj_after = get_object_position(obj_name)
            if obj_after is not None and obj_now is not None:
                dz = obj_after[2] - obj_now[2]
                rospy.loginfo("  Lift dz: %.1f cm  %s",
                              dz * 100, "OK" if dz > 0.05 else "FAIL")
                if dz > 0.05:
                    self.holding = obj_name
                    rospy.loginfo("  PICK SUCCESS: holding %s", obj_name)
                    return True
                else:
                    rospy.logwarn("  Lift failed, retrying...")
                    self.gripper_srv(open=True)
                    rospy.sleep(0.5)
                    if self.steps_completed == 0:
                        self.reset_srv()
                        rospy.sleep(3.0)
                    self.gripper_srv(open=True)
                    rospy.sleep(1.0)
                    continue
            else:
                rospy.logwarn("  Could not verify lift")
                self.holding = obj_name
                return True

        rospy.logerr("  PICK FAILED after %d attempts", MAX_PICK_ATTEMPTS)
        return False

    # ---- PLACE primitive ----

    def place(self, destination, orientation, is_last_step=False):
        """Place the currently held object at a destination.

        destination: dict with either:
          - 'position': {x, y, z} absolute world position
          - 'reference' + 'offset': place relative to another object
        orientation: dict with qx, qy, qz, qw
        is_last_step: if True, skip return_home to avoid disturbing
                      just-placed objects
        """
        if self.holding is None:
            rospy.logerr("PLACE called but not holding anything!")
            return False

        obj_name = self.holding
        rospy.loginfo("=" * 50)
        rospy.loginfo("PLACE: %s", obj_name)
        rospy.loginfo("=" * 50)

        # Compute target world position
        if "reference" in destination:
            ref_name = destination["reference"]
            ref_pos = get_object_position(ref_name)
            if ref_pos is None:
                rospy.logerr("  Cannot find reference object '%s'", ref_name)
                return False
            off = destination.get("offset", {})
            target_world = ref_pos + np.array([
                off.get("x", 0), off.get("y", 0), off.get("z", 0)
            ])
            rospy.loginfo("  Reference '%s' at [%.3f, %.3f, %.3f]",
                          ref_name, *ref_pos)
        else:
            p = destination["position"]
            target_world = np.array([p["x"], p["y"], p["z"]])

        rospy.loginfo("  Place target world [%.3f, %.3f, %.3f]", *target_world)

        qx = orientation.get("qx", 1.0)
        qy = orientation.get("qy", 0.0)
        qz = orientation.get("qz", 0.0)
        qw = orientation.get("qw", 0.0)

        # Carry phase: exclude only held object, keep table + others
        # as collision obstacles so RRT avoids them during carry
        self._include_all()
        rospy.sleep(0.3)
        self._exclude(obj_name)
        rospy.sleep(0.3)

        # Snapshot before carry
        snap_before_carry = snapshot_object_positions()

        # Move above the target (hover)
        hover_world = target_world.copy()
        hover_world[2] += APPROACH_DZ
        hover_base = world_to_base(hover_world)

        if not move_and_wait(self.move_srv, hover_base,
                             qx, qy, qz, qw, "Place-hover"):
            return False

        # Check for collisions during carry
        snap_after_carry = snapshot_object_positions()
        displaced = detect_displacements(snap_before_carry, snap_after_carry)
        if displaced:
            rospy.logwarn("  COLLISION DETECTED during carry/hover:")
            for name, dist_mm in displaced:
                if name != obj_name:
                    rospy.logwarn("    %s displaced %.0fmm", name, dist_mm)

        # Verify object is still held after carry
        obj_pos = get_object_position(self.holding)
        ee_pos = get_ee_pose()
        if obj_pos is not None and ee_pos is not None:
            carry_dist = np.linalg.norm(obj_pos[:2] - ee_pos[:2])
            rospy.loginfo("  After carry: obj-EE dist %.0fmm", carry_dist * 1000)
            if carry_dist > 0.10:
                rospy.logwarn("  Object may have been dropped during carry!")

        # Only exclude reference object if the place target is geometrically
        # inside its collision volume (both xy AND z overlap). If the target
        # is next to it (different xy) but same z-height, keep it as a
        # collision obstacle so the arm avoids it during descent.
        need_exclude_ref = False
        if "reference" in destination:
            ref_name = destination["reference"]
            ref_pos_now = get_object_position(ref_name)
            need_exclude_ref = False
            if ref_pos_now is not None:
                ref_half = OBJECT_HALF_EXTENTS.get(ref_name, [0.05, 0.05, 0.04])
                moveit_padding = 0.05
                ref_top = ref_pos_now[2] + ref_half[2] + moveit_padding
                # Check xy overlap: target must be within ref's xy footprint
                xy_dist = np.linalg.norm(target_world[:2] - ref_pos_now[:2])
                ref_radius_xy = max(ref_half[0], ref_half[1]) + moveit_padding
                z_inside = target_world[2] < ref_top
                xy_inside = xy_dist < ref_radius_xy
                if z_inside and xy_inside:
                    need_exclude_ref = True
                    rospy.loginfo("  Target inside ref volume (xy=%.0fmm,"
                                  " z=%.3f < %.3f), excluding %s",
                                  xy_dist * 1000, target_world[2], ref_top,
                                  ref_name)
                elif z_inside:
                    rospy.loginfo("  Target z inside ref but xy outside"
                                  " (%.0fmm > %.0fmm), keeping %s as obstacle",
                                  xy_dist * 1000, ref_radius_xy * 1000,
                                  ref_name)
            if need_exclude_ref:
                self._exclude(ref_name)
                rospy.sleep(0.3)

        # Exclude table and nearby objects for descent near the surface
        # (arm links need clearance). Distant objects remain as obstacles.
        self._exclude_table()
        skip_objs = {obj_name}
        if "reference" in destination and need_exclude_ref:
            skip_objs.add(destination["reference"])
        self._exclude_nearby(target_world[:2], radius=0.10,
                             skip=skip_objs)
        rospy.sleep(0.3)

        # Lower to release height using Cartesian straight-line path
        release_world = target_world.copy()
        release_base = world_to_base(release_world)

        if not move_and_wait(self.move_srv, release_base,
                             qx, qy, qz, qw, "Place-lower",
                             cartesian=True):
            return False

        # Open gripper
        rospy.loginfo("  Opening gripper...")
        self.gripper_srv(open=True)
        rospy.sleep(2.0)

        self.holding = None

        if is_last_step:
            # Skip retreat and return_home to avoid disturbing just-placed objects
            rospy.loginfo("  Last step — skipping retreat to avoid displacement")
        else:
            # Retreat upward using Cartesian straight-line path
            if not move_and_wait(self.move_srv, hover_base,
                                 qx, qy, qz, qw, "Place-retreat", settle=3.0,
                                 cartesian=True):
                pass  # non-critical

        # Re-include all objects + table in planning scene
        self._include_all()
        rospy.sleep(0.5)

        if not is_last_step:
            self.return_home()

        rospy.loginfo("  PLACE COMPLETE: released %s", obj_name)
        return True

    # ---- MOVE_TO primitive (no grasp) ----

    def move_to(self, position, orientation):
        """Move the EE to a position without grasping."""
        qx = orientation.get("qx", 0.0)
        qy = orientation.get("qy", 0.0)
        qz = orientation.get("qz", 0.0)
        qw = orientation.get("qw", 0.0)
        base = world_to_base([position["x"], position["y"], position["z"]])
        return move_and_wait(self.move_srv, base, qx, qy, qz, qw, "MoveTo")


# ---------------------------------------------------------------------------
# Goal verification
# ---------------------------------------------------------------------------

def verify_goals(goals):
    rospy.loginfo("")
    rospy.loginfo("=" * 50)
    rospy.loginfo("VERIFYING GOALS")
    rospy.loginfo("=" * 50)

    positions = get_all_object_positions()
    all_pass = True

    for i, goal in enumerate(goals):
        desc = goal.get("description", f"Goal {i+1}")
        gtype = goal["type"]

        if gtype == "near":
            obj = goal["object"]
            ref = goal["reference"]
            max_d = goal.get("max_distance_xy", 0.10)

            obj_pos = positions.get(obj)
            ref_pos = positions.get(ref)
            if obj_pos is None or ref_pos is None:
                rospy.logerr("  [FAIL] %s: could not find objects", desc)
                all_pass = False
                continue

            dist_xy = np.linalg.norm(obj_pos[:2] - ref_pos[:2])
            passed = dist_xy <= max_d
            status = "PASS" if passed else "FAIL"
            rospy.loginfo("  [%s] %s", status, desc)
            rospy.loginfo("         %s at [%.3f, %.3f, %.3f]", obj, *obj_pos)
            rospy.loginfo("         %s at [%.3f, %.3f, %.3f]", ref, *ref_pos)
            rospy.loginfo("         xy distance: %.1f mm (max: %.1f mm)",
                          dist_xy * 1000, max_d * 1000)
            if not passed:
                all_pass = False

        elif gtype == "above":
            obj = goal["object"]
            ref = goal["reference"]
            min_z = goal.get("min_z_offset", 0.0)

            obj_pos = positions.get(obj)
            ref_pos = positions.get(ref)
            if obj_pos is None or ref_pos is None:
                rospy.logerr("  [FAIL] %s: could not find objects", desc)
                all_pass = False
                continue

            z_diff = obj_pos[2] - ref_pos[2]
            passed = z_diff >= min_z
            status = "PASS" if passed else "FAIL"
            rospy.loginfo("  [%s] %s", status, desc)
            rospy.loginfo("         %s z=%.3f, %s z=%.3f, diff=%.1fmm (min: %.1fmm)",
                          obj, obj_pos[2], ref, ref_pos[2],
                          z_diff * 1000, min_z * 1000)
            if not passed:
                all_pass = False

        elif gtype == "on_table":
            obj = goal["object"]
            table_z = goal.get("table_z_world", 0.942)
            tol = goal.get("tolerance", 0.05)

            obj_pos = positions.get(obj)
            if obj_pos is None:
                rospy.logerr("  [FAIL] %s: could not find object", desc)
                all_pass = False
                continue

            diff = abs(obj_pos[2] - table_z)
            passed = diff <= tol
            status = "PASS" if passed else "FAIL"
            rospy.loginfo("  [%s] %s", status, desc)
            rospy.loginfo("         %s z=%.3f, table z=%.3f, diff=%.1fmm (tol: %.1fmm)",
                          obj, obj_pos[2], table_z, diff * 1000, tol * 1000)
            if not passed:
                all_pass = False

        else:
            rospy.logwarn("  [SKIP] Unknown goal type: %s", gtype)

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="General task executor")
    parser.add_argument("task_config", type=str,
                        help="Path to task YAML config file")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip initial sim reset")
    parser.add_argument("--scene", type=str, default="scene_breakfast",
                        help="Scene name (matches config/scenes/<name>.yaml)")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    # Load scene-specific object config if available
    global OBJECT_DET_IDS, OBJECT_MUJOCO_NAMES, OBJECT_HALF_EXTENTS
    det_ids, mujoco_names, half_extents = _load_scene_config(args.scene)
    if det_ids is not None:
        OBJECT_DET_IDS = det_ids
        OBJECT_MUJOCO_NAMES = mujoco_names
        OBJECT_HALF_EXTENTS = half_extents
        print(f"Loaded scene config: {args.scene} ({len(det_ids)} objects)")
    else:
        print(f"No scene config for '{args.scene}', using breakfast defaults")

    # Resolve config path relative to package root if not absolute
    if not os.path.isabs(args.task_config):
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.task_config = os.path.join(pkg_root, args.task_config)

    with open(args.task_config, 'r') as f:
        task = yaml.safe_load(f)["task"]

    grasp_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "grasp_poses.yaml")

    print(f"\n{'=' * 60}")
    print(f"  TASK: {task['name']}")
    print(f"  {task.get('description', '')}")
    print(f"  Steps: {len(task['steps'])}")
    print(f"  Goals: {len(task.get('goals', []))}")
    print(f"{'=' * 60}\n")

    rospy.init_node("run_task", anonymous=True)
    executor = TaskExecutor(grasp_config_path)

    # Reset
    if not args.no_reset:
        rospy.loginfo("Resetting simulation...")
        executor.reset_srv()
        rospy.sleep(3.0)

    # Execute steps
    for i, step in enumerate(task["steps"]):
        action = step["action"]
        rospy.loginfo("")
        rospy.loginfo("--- Step %d/%d: %s ---", i + 1, len(task["steps"]), action)

        if action == "pick":
            success = executor.pick(step["object"])
        elif action == "place":
            is_last = (i == len(task["steps"]) - 1)
            success = executor.place(step["destination"],
                                     step.get("orientation", {}),
                                     is_last_step=is_last)
        elif action == "move_to":
            success = executor.move_to(step["position"],
                                       step.get("orientation", {}))
        else:
            rospy.logerr("Unknown action: %s", action)
            success = False

        if not success:
            rospy.logerr("Step %d FAILED — aborting task.", i + 1)
            # Still verify goals so we can see what happened
            if task.get("goals"):
                rospy.sleep(2.0)
                verify_goals(task["goals"])
            return

        executor.steps_completed += 1

    # Verify goals
    rospy.sleep(3.0)  # let physics settle
    goals = task.get("goals", [])
    if goals:
        all_pass = verify_goals(goals)
        print(f"\n{'=' * 60}")
        if all_pass:
            print(f"  TASK '{task['name']}': ALL GOALS PASSED")
        else:
            print(f"  TASK '{task['name']}': SOME GOALS FAILED")
        print(f"{'=' * 60}\n")
    else:
        rospy.loginfo("No goals to verify.")

    rospy.loginfo("Task execution complete.")


if __name__ == "__main__":
    main()
