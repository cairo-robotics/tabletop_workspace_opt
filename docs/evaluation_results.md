# Workspace Optimization Evaluation Results

## Overview

This document reports the results of testing optimized vs. un-optimized
workspace layouts across 3 scenes and 6 task configurations. The evaluation
uses Monte Carlo simulation of noisy joystick-based goal inference (Section
1.6 of the Shared Autonomy Notes).

Two types of evaluation were performed:

1. **Monte Carlo intent separability** — simulates noisy joystick inputs and
   Bayesian goal inference (no ROS/MuJoCo required)
2. **Full ROS + MuJoCo simulation** — actual pick/place execution with
   MoveIt planning, MuJoCo physics, and goal verification

## Optimization Results

| Scene | Objects | Original Slack | Optimized Slack | Improvement |
|-------|---------|---------------|----------------|-------------|
| scene_breakfast | 6 (bowl, cereal, napkin, spoon, banana, milk) | 66.8 | 272.7 | +205.9 (4.1x) |
| scene_desk | 5 (mug, book, phone, pen_cup, stapler) | 143.3 | 361.1 | +217.8 (2.5x) |
| scene_kitchen_prep | 5 (cutting_board, apple, can, bottle, sponge) | 150.9 | 361.1 | +210.1 (2.4x) |

## Single-Step Goal Identification

How many noisy joystick inputs until the robot correctly identifies which
object the user is reaching for (95% posterior confidence):

| Scene | Original (steps) | Optimized (steps) | Speedup |
|-------|-------------------|-------------------|---------|
| scene_breakfast | 8.2 | 1.6 | 5.1x |
| scene_desk | 2.6 | 1.3 | 2.0x |
| scene_kitchen_prep | 2.5 | 1.3 | 1.9x |

All layouts achieve 100% accuracy. The breakfast scene benefits most because
its original layout has objects clustered together (bowl/spoon/milk in similar
directions from the EE start).

## Multi-Step Task Results

### Breakfast Scene

| Task | Steps | Orig Total | Opt Total | Speedup | Orig Acc | Opt Acc |
|------|-------|-----------|-----------|---------|----------|---------|
| banana_in_bowl | 2 | 203.8 | 3.9 | **52.0x** | 89% | 100% |
| cereal_next_to_bowl | 2 | 36.5 | 4.1 | **8.9x** | 99% | 100% |
| set_breakfast | 4 | 42.1 | 23.4 | **1.8x** | 99.5% | 100% |
| full_breakfast | 6 | 274.8 | 135.3 | **2.0x** | 82.7% | 98% |

**Key findings:**
- `banana_in_bowl` sees **52x speedup** — in the original layout, banana is
  between bowl and cereal making it hard to distinguish. Optimization moves
  banana to a unique angle.
- `full_breakfast` improves from 82.7% → 98% accuracy. The original layout
  fails completely on step 6 (place milk near bowl, 0% accuracy) because
  the place target is near other objects. Optimization fixes this.

### Desk Scene

| Task | Steps | Orig Total | Opt Total | Speedup | Orig Acc | Opt Acc |
|------|-------|-----------|-----------|---------|----------|---------|
| desk_organize | 6 | 198.1 | 24.2 | **8.2x** | 87% | 100% |

**Key findings:**
- The original layout fails on step 6 (place stapler on book, 22% accuracy)
  because after the mug is moved, the place target near the book is
  ambiguous. Optimization spreads objects so all place targets become
  distinguishable.

### Kitchen Prep Scene

| Task | Steps | Orig Total | Opt Total | Speedup | Orig Acc | Opt Acc |
|------|-------|-----------|-----------|---------|----------|---------|
| kitchen_prep_full | 10 | 636.8 | 294.0 | **2.2x** | 88.2% | 95.2% |

**Key findings:**
- The longest task (10 steps) shows consistent improvement across most steps.
- Step 4 (place can near bottle) remains difficult even with optimization
  (52% accuracy) because the place target position happens to be close to
  other objects in the optimized layout. This suggests place-step targets
  should be considered during optimization.
- Steps 2 and 10 (place near cutting_board) improve dramatically:
  174→2.7 and 200→7.8 steps respectively.

## Per-Step Analysis: Where Optimization Helps Most

**Pick steps** (reaching for a known object) almost always benefit from
optimization. The optimized layout ensures each object is in a unique
angular direction from the EE.

**Place steps** (reaching for a computed position like "left of bowl") show
mixed results. The optimization only considers object positions, not the
derived place targets. Place targets that happen to be near other objects
remain ambiguous.

### Failure Mode: Ambiguous Place Targets

The remaining failure cases are all place steps where the destination
(reference + offset) ends up close to another object:

| Scene | Task | Step | Target | Issue |
|-------|------|------|--------|-------|
| breakfast | full_breakfast | 2 | place_near_bowl | Bowl moved far from other objects, but cereal place offset points toward napkin |
| kitchen_prep | kitchen_prep_full | 4 | place_near_bottle | Place offset puts target near apple's optimized position |

**Recommendation:** Future work should extend the optimization to also
consider place-step target positions as candidate goals in the separability
objective.

## Optimized Object Positions

### Breakfast (6 objects)

| Object | Original (x, y) | Optimized (x, y) |
|--------|-----------------|-------------------|
| bowl | (0.600, 0.300) | (0.730, -0.346) |
| cereal | (0.480, -0.250) | (0.374, -0.295) |
| napkin | (0.820, 0.200) | (0.820, -0.042) |
| spoon | (0.480, 0.250) | (0.678, 0.230) |
| banana | (0.500, 0.050) | (0.310, -0.042) |
| milk_carton | (0.780, 0.300) | (0.417, 0.299) |

### Desk (5 objects)

| Object | Original (x, y) | Optimized (x, y) |
|--------|-----------------|-------------------|
| mug | (0.500, 0.200) | (0.301, 0.265) |
| book | (0.650, -0.150) | (0.302, -0.173) |
| phone | (0.450, -0.250) | (0.848, -0.039) |
| pen_cup | (0.750, 0.300) | (0.659, 0.264) |
| stapler | (0.550, -0.350) | (0.695, -0.365) |

### Kitchen Prep (5 objects)

| Object | Original (x, y) | Optimized (x, y) |
|--------|-----------------|-------------------|
| cutting_board | (0.600, 0.000) | (0.301, 0.265) |
| apple | (0.500, 0.150) | (0.302, -0.173) |
| can | (0.450, -0.200) | (0.848, -0.039) |
| bottle | (0.750, 0.250) | (0.659, 0.264) |
| sponge | (0.700, -0.200) | (0.695, -0.365) |

## Changes Made

### New Files
| File | Purpose |
|------|---------|
| `scripts/optimize_all_scenes.py` | Runs optimization for all scenes, generates optimized XMLs |
| `scripts/eval_all_scenes.py` | Evaluates all scenes/tasks with Monte Carlo simulation |
| `scripts/test_all_tasks.sh` | Shell script for full sim testing (requires display) |
| `src/assets/scene_breakfast_optimized.xml` | Optimized breakfast layout |
| `src/assets/scene_desk_optimized.xml` | Optimized desk layout |
| `src/assets/scene_kitchen_prep_optimized.xml` | Optimized kitchen prep layout |
| `config/scenes/scene_breakfast_optimized.yaml` | Optimized breakfast scene config |
| `config/scenes/scene_desk_optimized.yaml` | Optimized desk scene config |
| `config/scenes/scene_kitchen_prep_optimized.yaml` | Optimized kitchen prep scene config |

### Modified Files
| File | Change |
|------|--------|
| `scripts/run_task.py` | Added `--scene` argument and `_load_scene_config()` to load object config from YAML instead of hardcoding breakfast objects. Backward compatible. |
| `scripts/optimize_all_scenes.py` | Fixed regex backreference bug in keyframe qpos replacement |

### Bug Fix
- **`optimize_all_scenes.py` regex error**: The keyframe qpos string
  contained values like "0.310" which Python's `re.sub` interpreted as
  backreference `\310`. Fixed by using a replacement function instead of
  a replacement string.

## How to Run

```bash
# Monte Carlo evaluation (no display needed):
cd /home/yi-shiuan/sawyer_ws/src/tabletop_workspace_opt
python3 scripts/eval_all_scenes.py

# Full sim testing (requires display):
# 1. Original breakfast scene:
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_breakfast
python3 scripts/run_task.py config/tasks/banana_in_bowl.yaml

# 2. Optimized breakfast scene:
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_breakfast_optimized
python3 scripts/run_task.py config/tasks/banana_in_bowl.yaml --scene scene_breakfast_optimized

# 3. Desk scene:
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk
python3 scripts/run_task.py config/tasks/desk_organize.yaml --scene scene_desk

# 4. Full automated test (requires display):
bash /tmp/run_all_sim_tests.sh
```

---

## Full ROS + MuJoCo Simulation Results

### Test Summary

Full simulation tests were run on 2026-03-21 with MuJoCo physics,
MoveIt motion planning, and actual pick/place execution.

| Scene / Task | Result |
|---|---|
| scene_breakfast / banana_in_bowl | **PASS** |
| scene_breakfast / cereal_next_to_bowl | **PASS** |
| scene_breakfast / set_breakfast | **PASS** |
| scene_breakfast_optimized / banana_in_bowl | **PASS** |
| scene_breakfast_optimized / cereal_next_to_bowl | ERROR (service crash during pick) |
| scene_breakfast_optimized / set_breakfast | ERROR (bowl displaced 1169mm during lift) |
| scene_desk / desk_organize | ERROR (phone pick failed after 5 attempts) |
| scene_desk_optimized / desk_organize | ERROR (service crash during stapler pick) |
| scene_kitchen_prep / kitchen_prep_full | ERROR (service crash during place) |
| scene_kitchen_prep_optimized / kitchen_prep_full | ERROR (service crash during sponge pick) |

### Analysis

**Original breakfast scene: 3/3 PASS** — All tasks completed with goals
verified. Minor collisions detected during lift/carry phases but the retry
mechanism handled them.

**Optimized breakfast scene: 1/3 PASS** — `banana_in_bowl` passed despite
needing a retry and RRT fallback. The other two tasks hit planning failures:
- `cereal_next_to_bowl`: cereal at (0.374, -0.295) was repeatedly displaced
  during pre-grasp approach. After 4 retries the MoveIt planner service
  crashed.
- `set_breakfast`: bowl was displaced 1169mm during the cereal lift phase,
  then place planning failed completely.

**New scenes (desk, kitchen_prep): 0/4 PASS** — Both original and optimized
layouts had grasping failures:
- **Desk scene**: The phone (15mm thin slab) was displaced ~100mm on every
  pre-grasp approach. After 5 failed attempts the task aborted. This is a
  known issue — very thin flat objects on the table are hard to grasp because
  the arm sweeps them aside. The phone's collision box is only 15mm tall.
- **Kitchen prep scene**: Pick/place of apple and sponge succeeded for the
  first few steps but the MoveIt planner service crashed during later steps,
  likely due to cumulative physics state divergence.

### Root Causes of Failures

1. **MoveIt planner service crash** (`service returned no response`):
   The `move_to_cartesian_pose.py` node dies when MoveIt's internal state
   becomes inconsistent after many planning attempts. This is an existing
   issue (see memory: "MoveIt planning is non-deterministic").

2. **Arm-object collisions**: The URDF arm geometry (spheres/short cylinders)
   underestimates the actual MuJoCo arm volume (long capsules). MoveIt plans
   that appear collision-free in the planning scene clip objects in MuJoCo.
   Objects near workspace edges (where the optimized layout places them) are
   more susceptible.

3. **Thin object grasping**: Objects with very small z-extent (phone at 15mm,
   cutting_board at 20mm) are hard to grasp because the descent path has
   almost no clearance above the table surface.

### Recommendations

1. **Increase collision padding** for optimized scenes (currently 3cm,
   consider 5cm for objects near workspace edges)
2. **Add grasp pose validation**: Before attempting a grasp, verify the
   grasp pose is reachable by checking IK solutions
3. **Phone grasp**: Use side-approach rather than top-down, or increase the
   phone collision z-extent
4. **Service recovery**: Add watchdog/restart for the `move_to_cartesian_pose`
   node when it crashes
5. **Re-run tests**: Due to non-deterministic MoveIt planning, each task
   should be run 3-5 times and results aggregated

### Changes Made During Testing

| File | Change |
|------|--------|
| `src/mujoco_sim/simulation_server.py` | Added `devel_isolated` Python path auto-discovery at top of file (fixes `ModuleNotFoundError: No module named 'intera_core_msgs'`) |
| `scripts/run_task.py` | Added same `devel_isolated` Python path fix (fixes import errors when launched from bash scripts) |
| `launch/sim_moveit.launch` | Added `<env name="PYTHONPATH">` tag with all `devel_isolated` package paths (partially effective — nodes still need the sys.path fix for reliable imports) |
