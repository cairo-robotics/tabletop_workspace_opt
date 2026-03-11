# Collision Box Discrepancy Investigation

## Problem Statement

The MoveIt motion planner plans collision-free paths in RViz, but when executed
in MuJoCo the robot end effector and arm links collide with objects on the table.
This investigation identifies the root causes and documents changes made.

## Issues Found

### 1. Object Collision Box Size Mismatch (Fixed)

**Files:** `src/mujoco_sim/simulation_server.py`, `scripts/run_task.py`

The `simulation_server.py` published collision boxes to MoveIt's planning scene
that were **larger** than the actual MuJoCo collision geoms in `scene_breakfast.xml`.
MoveIt planned paths around inflated boxes, but the MuJoCo physics used smaller
boxes, creating a gap where the arm could pass through in the planner but contact
objects in the sim.

| Object  | simulation_server (old)   | MuJoCo scene XML (correct) |
|---------|---------------------------|----------------------------|
| cereal  | [0.04, 0.03, 0.08]       | [0.035, 0.023, 0.08]       |
| banana  | [0.09, 0.03, 0.025]      | [0.09, 0.02, 0.025]        |

**Fix:** Updated `object_sizes` in `simulation_server.py` and `OBJECT_HALF_EXTENTS`
in `run_task.py` to match the exact MuJoCo collision geom sizes from
`scene_breakfast.xml`.

### 2. URDF right_hand Link Has Zero-Size Collision (Fixed)

**File:** `src/assets/sawyer_with_electric_gripper.urdf`

The `right_hand` link (connecting the wrist to the gripper base) had collision
geometry of `cylinder length="0" radius="0"` -- effectively invisible to MoveIt.
This meant MoveIt did not check for collisions between the hand link and scene
objects.

**Fix:** Changed to `cylinder length="0.05" radius="0.04"` centered at z=0.025.

### 3. URDF right_gripper_base Has No Collision (Fixed)

**File:** `src/assets/sawyer_with_electric_gripper.urdf`

The `right_gripper_base` link had no `<collision>` tag at all, making it invisible
to collision checking.

**Fix:** Added a `cylinder length="0.04" radius="0.035"` collision geom.

### 4. Arm Link Collision Geometry Mismatch (Documented, Not Fixed)

**Scope:** All arm links right_l0 through right_l5

The URDF arm collision geometry is systematically **smaller** than the MuJoCo
collision geometry:

| Link     | URDF Shape      | URDF Size           | MuJoCo Shape | MuJoCo Size          |
|----------|-----------------|---------------------|--------------|----------------------|
| right_l0 | sphere r=0.07   | point               | capsule      | len=0.255, r=0.07    |
| right_l1 | sphere r=0.07   | point               | capsule      | len=0.19, r=0.068    |
| right_l2 | cylinder        | len=0.34, r=0.06    | capsule      | len=0.41, r=0.055    |
| right_l3 | sphere r=0.06   | point               | capsule      | len=0.145, r=0.055   |
| right_l4 | cylinder        | len=0.30, r=0.045   | capsule      | len=0.40, r=0.045    |
| right_l5 | sphere r=0.06   | point               | 2 capsules   | len=0.12+0.08, r=0.045 |

This means MoveIt's RRT planner underestimates the arm's swept volume. Paths
that clear objects in MoveIt's model may clip them in MuJoCo.

**Why not fixed:** Enlarging the URDF collision to match MuJoCo makes MoveIt too
conservative. In testing, enlarging the arm collisions caused the cereal pick to
fail entirely (the arm couldn't approach without MoveIt detecting self-collisions
with nearby objects). The original URDF values are a necessary compromise between
collision accuracy and planning feasibility.

**Recommendation:** Use MoveIt's `padding` parameter (collision padding applied
uniformly to all links) to add a small safety margin rather than exactly matching
MuJoCo capsule geometry. A padding of 0.01-0.02m may help without over-constraining
the planner. This can be set in `planning_context.launch` or programmatically.

### 5. Trajectory Execution Fidelity (Documented, Not Fixed)

MoveIt plans a densely-sampled collision-free trajectory, but execution in MuJoCo
streams waypoints through a PD controller. The MuJoCo controller interpolates
between waypoints, and the actual arm path may deviate from the planned path,
especially during fast motions or when the arm has significant momentum.

This is likely the primary cause of the large milk carton displacement (167mm)
observed during carry motions, where the upper arm links sweep through the milk
carton's position even though MoveIt planned a collision-free path.

## Collision Detection Instrumentation Added

**File:** `scripts/run_task.py`

Added `snapshot_object_positions()` and `detect_displacements()` functions that
compare all object positions before and after each motion phase:

- **Pre-grasp approach**: Detects if the arm clips objects while moving to hover
- **Descent**: Detects if objects are knocked during the grasp approach
- **Lift**: Detects if objects are disturbed while lifting the grasped object
- **Carry/hover**: Detects if the arm hits objects while carrying to the place target

Displacements >5mm are reported as warnings in the ROS log.

## Test Results

### Before fixes (original code):
- cereal_next_to_bowl: PASS (no collision detection)
- banana_in_bowl: PASS (no collision detection)

### After object size fix + collision detection:
- cereal_next_to_bowl: PASS (minor displacements, no failures)
- banana_in_bowl: FAIL (milk_carton displaced 163mm during carry, bowl displaced
  causing goal miss)

### After arm collision enlargement (reverted):
- cereal_next_to_bowl: FAIL (arm too "fat" to approach cereal)
- banana_in_bowl: FAIL (same milk_carton issue)

### Final state (object sizes fixed, arm collision reverted):
- Both tasks experience some object displacement due to the fundamental
  MuJoCo-URDF arm geometry mismatch (issue #4) and trajectory execution
  fidelity (issue #5)
- These are known limitations requiring deeper architectural changes

## Files Changed

1. `src/mujoco_sim/simulation_server.py` - Fixed `object_sizes` dict
2. `scripts/run_task.py` - Fixed `OBJECT_HALF_EXTENTS` + added collision detection
3. `src/assets/sawyer_with_electric_gripper.urdf` - Fixed right_hand and
   right_gripper_base collision geometry

## Recommendations for Future Work

1. **MoveIt collision padding**: Add 1-2cm padding in MoveIt config to partially
   compensate for URDF-MuJoCo geometry mismatch without over-constraining planning.
2. **Denser waypoint streaming**: Reduce the waypoint interpolation step in
   `_execute_on_mujoco()` to improve trajectory fidelity.
3. **Post-execution collision check**: After each trajectory segment, check if
   any objects were displaced and re-plan if necessary.
4. **Scene layout optimization**: Move objects farther apart on the table to give
   the arm more clearance during carry motions.
