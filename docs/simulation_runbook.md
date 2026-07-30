# Simulation Runbook

Living document with instructions for running tasks in the MuJoCo
simulator. Updated as new scripts and workflows are added.

**Last updated: 2026-07-16**

> **Note:** For the current SE(3) shared-autonomy pipeline
> (workspace optimization, headless evaluation, threshold sweep, paper
> figures), see `docs/shared_autonomy_experiments.md`. This runbook now
> primarily covers the ROS + MoveIt simulation workflow, full grasp execution,
> and manual grasp auditing.

---

## 0. Current Baseline Parameters and Results Paths

For the current SE(3) shared-autonomy pipeline and reported experiment
parameters, use `docs/shared_autonomy_experiments.md`.

**Baseline parameters (moderate-n_steps regime, as of 2026-04-27):**

| Parameter | Value |
|-----------|-------|
| `sigma_u` (joystick noise) | 0.03 m/s |
| `n_steps` (pick horizon) | 75 |
| `control_rate` | 5.0 Hz |
| `arrival_dist` | 0.02 m |
| `beta` (Boltzmann rationality) | 5.0 |
| `threshold` (commit) | 0.9 |

Defaults live in `src/envopt/yaw_optimizer.py` and the SA runners. Match
the same values between the optimizer and the runtime.

**Canonical results paths (as of 2026-04-15+):**

- `results/se3_map_elites/` — ME archives (per-scene JSON)
- `results/sa_headless/` — SA evaluation
  - `se3_3d_random_vs_optimized.json` — main paper table
  - `se3_threshold_sweep.json` — threshold sweep

Legacy paths (`results/map_elites_v2_*.json`, `results/map_elites_v3_*.json`,
`results/map_elites_tier_*.json`, `results/sa_benchmark.json`) are archived
from earlier iterations. **Do not consult them for current comparisons.**

---

## Table of Contents

0. [Current Baseline Parameters and Results Paths](#0-current-baseline-parameters-and-results-paths)
1. [Quick Reference](#1-quick-reference)
2. [Prerequisites](#2-prerequisites)
3. [Launching the Simulator](#3-launching-the-simulator)
4. [Shared Autonomy Mode (No Motion Planning)](#4-shared-autonomy-mode-no-motion-planning)
5. [Motion Planning Mode (Full Grasping)](#5-motion-planning-mode-full-grasping)
6. [Manual SE(3) Grasp Audit](#6-manual-se3-grasp-audit)
7. [Evaluating All Pick Orderings](#7-evaluating-all-pick-orderings)
8. [Comparing Baseline vs Optimized Layouts](#8-comparing-baseline-vs-optimized-layouts)
9. [Using Optimized Scene YAMLs](#9-generating-optimized-scene-xmls)
10. [Analytical Evaluation (No Simulator) — ⚠️ LEGACY](#10-analytical-evaluation-no-simulator--legacy)
11. [Available Scenes and Tasks](#11-available-scenes-and-tasks)
12. [Troubleshooting](#12-troubleshooting)
13. [Cleanup](#13-cleanup)

---

## 1. Quick Reference

```bash
# Shared autonomy (recommended — no grasping, uses object teleportation)
# Terminal 1:
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk
# Terminal 2:
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml

# Motion planning (full grasping pipeline — may fail on some objects)
# Terminal 1:
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk
# Terminal 2:
python3 scripts/run_task.py config/tasks/desk_organize_v2_sa.yaml --scene scene_desk

# Analytical evaluation (no simulator needed)
python3 scripts/eval/eval_map_elites_tiers.py --scenes desk --seeds 1 --n-trials 50
```

---

## 2. Prerequisites

- ROS Noetic workspace at `~/sawyer_ws/` built with `catkin build_isolated`
- Source the workspace: `source ~/sawyer_ws/devel_isolated/setup.bash`
- Required packages: `mujoco`, `ribs`, `scipy`, `shapely`, `opencv-python`
- Optional: `xdotool` (for dismissing ROS1 EOL popup)

---

## 3. Launching the Simulator

All workflows require the simulator running first.

### Start the sim

```bash
# Clean up any stale processes first
pkill -f "mujoco" || true; pkill -f "rviz" || true
pkill -f "roslaunch" || true; pkill -f "simulation_server.py" || true
pkill -f "relaxed_ik" || true

# Launch with a specific scene
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk
```

This starts: RelaxedIK, MuJoCo simulation server, TF publishers,
MoveIt, RViz, and the move_to_cartesian_pose service.

### Dismiss the ROS1 EOL popup

RViz shows a popup that blocks interaction. Dismiss it:

```bash
sleep 5 && xdotool search --name "ROS End of Life" windowactivate --sync key Return 2>/dev/null &
```

Run this right after launching, or manually click the popup.

---

## 4. Shared Autonomy Mode (No Motion Planning)

**Recommended for evaluation.** Uses a simulated joystick to move the
end-effector toward objects. When the Bayesian intent inference reaches
the confidence threshold, the object is **teleported** to its
destination. No grasping, no motion planning, no arm-object collisions.

### How it works

1. A simulated user moves the EE toward a target object (with noise)
2. The inference engine tracks the EE trajectory and computes a
   posterior over candidate goals
3. When confidence exceeds the threshold (default 90%), the system
   auto-completes the **pick** action
4. **Place** actions are automatic (no inference) — the object is
   teleported back to its original pick location immediately
5. The EE snaps back to home, and the state machine advances
6. Only pick actions are recorded in the results JSON

### Basic usage

```bash
# Terminal 2 (sim must be running in terminal 1):
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml
```

### Specifying pick order

By default, the simulated user follows the first branch at each
state. To specify which objects to pick and in what order:

```bash
# Pick stapler first, then pen_cup, then mug
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml \
    user_sequence:=stapler,pen_cup,mug
```

The `user_sequence` only controls the pick order. Place/pour actions
are automatically determined by the state machine.

### Debug and visualization modes

```bash
# Debug mode: pauses for keyboard input between goals
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml \
    debug:=true

# Live posterior visualization (matplotlib bar chart)
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml \
    visualize:=true

# Both
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml \
    debug:=true visualize:=true
```

Debug mode prints the full posterior distribution after each pick and
waits for Enter. Type `q` + Enter to abort early.

The visualizer shows a live horizontal bar chart with P(goal) for each
candidate, a threshold line, and the true target highlighted in green.

### Tuning hyperparameters

```bash
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml \
    noise:=0.02 \
    threshold:=0.85 \
    sigma:=0.8 \
    max_speed:=0.05
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inference_model` | `gaussian` | Intent inference algorithm: `gaussian` (direction model, matches optimizer) or `path_efficiency` (Dragan-style) |
| `noise` | 0.03 | Gaussian noise sigma on velocity commands (m/s) |
| `threshold` | 0.9 | Posterior confidence threshold for auto-completion |
| `sigma` | 1.0 | Gaussian direction model: noise sigma for likelihood computation |
| `beta` | 5.0 | Path efficiency model: rationality coefficient |
| `max_speed` | 0.05 | Max EE Cartesian speed (m/s) |
| `control_rate` | 5.0 | Control loop frequency (Hz) |
| `debug` | `false` | Pause for keyboard input between goals |
| `visualize` | `false` | Show live matplotlib posterior chart |

**Inference models:**
- `gaussian` (default): Accumulates Gaussian log-likelihoods of velocity
  direction vs expected direction toward each goal. Matches the intent
  separability model used by the MAP-Elites optimizer.
- `path_efficiency`: Boltzmann-rational path efficiency ratio (Dragan
  legibility model). Does not use velocity observations directly.

### Optimized layouts

Optimized layouts are now stored as YAML only, e.g.
`config/scenes/scene_desk_se3_me_optimized.yaml`. The simulator launch uses the
base XML scene:

```bash
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk
```

For current random-vs-optimized SE(3) experiments, use the headless workflow in
`docs/shared_autonomy_experiments.md`. For manual visual checks, use
`scripts/eval/manual_audit_se3_grasp.py`, which launches against the base XML and
teleports objects from the optimized YAML.

### Output

The runner prints per-step inference results:

```
State: initial | User intent: pick_mug | Goals: ['pick_mug', 'pick_stapler', 'pick_pen_cup']
  d=119mm | pick_mug=0.58 pick_pen_cup=0.22 pick_stapler=0.21
  d=92mm  | pick_mug=0.76 pick_pen_cup=0.13 pick_stapler=0.11
  INTENT INFERRED: pick_mug (p=0.801, t=14.2s)
```

Results are saved as JSON to `results/sa_runs/` with per-step
inference times, confidence, and ordering.

---

## 5. Motion Planning Mode (Full Grasping)

Uses MoveIt motion planning + IK to execute pick-and-place with the
physical gripper in MuJoCo. This is more realistic but:

- **Non-deterministic** — MoveIt planning may find different paths
- **Collision-prone** — arm can knock objects during reach
- **Some objects are ungraspable** — flat/thin objects like book

### Basic usage

```bash
# Terminal 2 (sim must be running):
python3 scripts/run_task.py config/tasks/desk_organize_v2_sa.yaml \
    --scene scene_desk
```

### State-machine tasks

`run_task.py` supports state-machine tasks (`*_sa.yaml`). It
extracts a linear step sequence by following the first branch at
each state. The `--scene` flag overrides scene detection if needed.

### Known limitations

- **book** (desk scene): too flat to grasp — use `desk_organize_v2_sa`
  which replaces book with stapler
- **napkin, spoon** (breakfast): very thin, table blocks fingers
- **Grasps are non-deterministic**: ~50% pass rate per attempt, the
  runner retries up to 5 times
- **Arm-object collisions**: MuJoCo capsule geometry is larger than
  MoveIt's URDF collision model, so MoveIt plans that pass in RViz
  may collide in MuJoCo

### Using with optimized layouts

```bash
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk

python3 scripts/run_task.py config/tasks/desk_organize_v2_sa.yaml \
    --scene scene_desk_se3_me_optimized
```

---

## 6. Manual SE(3) Grasp Audit

Use this workflow before trusting full MoveIt grasp execution or before a
large SE(3) rerun whose grasp poses changed.

Terminal 1:

```bash
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_breakfast_easy
```

Terminal 2:

```bash
python3 scripts/eval/manual_audit_se3_grasp.py \
    --scene scene_breakfast_easy \
    --object banana
```

The audit script:

- loads `config/grasp_poses_3d.yaml`;
- resolves object poses from the scene YAML or base XML fallback;
- calls `/reset_sim`;
- clears its own RViz markers and MoveIt preview displays;
- teleports objects with explicit positions from optimized/layout YAMLs;
- keeps the target object in the MoveIt planning scene for pregrasp planning;
- excludes the target object only for the grasp step, unless
  `--no-exclude-target` or `--keep-target-collision` is passed;
- publishes RViz poses/markers before planning:
  - `/manual_audit_se3/pregrasp_pose`
  - `/manual_audit_se3/grasp_pose`
  - `/manual_audit_se3/markers`
- publishes MoveIt goal-state/path previews:
  - `/move_group/display_robot_state`
  - `/move_group/display_planned_path`

Useful flags:

```bash
# Inspect pregrasp/grasp only; do not close or lift.
python3 scripts/eval/manual_audit_se3_grasp.py \
    --scene scene_desk_se3_me_optimized --object stapler --skip-close-lift

# Run without Enter prompts.
python3 scripts/eval/manual_audit_se3_grasp.py \
    --scene scene_desk_se3_me_optimized --object stapler --no-pause

# Publish IK goal state only; skip preview planning.
python3 scripts/eval/manual_audit_se3_grasp.py \
    --scene scene_desk_se3_me_optimized --object stapler --no-plan-preview

# Keep the target object in the planning scene even for the grasp step.
python3 scripts/eval/manual_audit_se3_grasp.py \
    --scene scene_desk_se3_me_optimized --object stapler --no-exclude-target
```

Collision diagnostics:

- If collision-aware IK fails but collision-ignored IK succeeds, the script
  calls `/check_state_validity`.
- It prints raw contacts, unique collision pairs, deepest penetration in
  millimeters, mean/max penetration per pair, and a rough clearance hint.
- Treat the depth as an order-of-magnitude tuning guide. Changing a Cartesian
  grasp pose can change the whole arm IK solution.

Frame convention:

```text
MoveIt base z = MuJoCo/world z - 0.92
```

RViz reset behavior:

- `/reset_sim` resets MuJoCo, not RViz UI state.
- The script clears its own latched topics at startup.
- If the RViz MotionPlanning panel still shows stale state, use the panel
  clear/reset controls or restart RViz.

---

## 7. Evaluating All Pick Orderings

For tasks with N pick objects, there are N! possible orderings.
Use the pick-and-return task format where objects are picked, used,
then returned to their original position. All objects remain as
candidates at every pick state.

### Headless SE(3) comparison (recommended — fast, no ROS needed)

```bash
# Single scene, single ordering
python3 scripts/eval/run_sa_headless.py \
    config/tasks/desk_pick_and_return_sa_3d.yaml \
    --intent-mode se3-grasp \
    --grasp-library config/grasp_poses_3d.yaml \
    --user-sequence stapler,pen_cup,mug

# Current random-vs-SE(3)-ME comparison for one tier
python3 scripts/eval/compare_se3_sa_3d.py \
    --only-me --tiers Med-A

# Current random-vs-SE(3)-ME comparison for all tiers
python3 scripts/eval/compare_se3_sa_3d.py \
    --n-random 10 --random-max-orderings 30
```

For scenes with many objects, `compare_se3_sa_3d.py` samples orderings
according to `--random-max-orderings` instead of exhaustively testing all
N! permutations.

Results are saved to
`results/sa_headless/se3_3d_random_vs_optimized.json`.

### With ROS/MuJoCo GUI

```bash
# Make sure sim is running first, then:
bash scripts/eval/run_sa_all_orderings.sh \
    config/tasks/desk_pick_and_return_sa.yaml \
    mug,stapler,pen_cup

# With optimized layout (launch sim with optimized scene first):
bash scripts/eval/run_sa_all_orderings.sh \
    config/tasks/desk_pick_and_return_sa.yaml \
    mug,stapler,pen_cup \
    scene_desk_se3_me_optimized
```

### Pick-and-return task files

| Scene | Task File |
|-------|-----------|
| breakfast_easy | `config/tasks/breakfast_easy_pick_and_return_sa.yaml` |
| desk | `config/tasks/desk_pick_and_return_sa.yaml` |
| breakfast | `config/tasks/breakfast_pick_and_return_sa.yaml` |
| kitchen_prep | `config/tasks/kitchen_pick_and_return_sa.yaml` |
| meal_assembly | `config/tasks/meal_pick_and_return_sa.yaml` |
| cluttered | `config/tasks/cluttered_pick_and_return_sa.yaml` |

---

## 8. Comparing Baseline vs Optimized Layouts

### Full workflow for one scene

```bash
# 1. Run baseline in ROS/MuJoCo
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk
# (in another terminal)
roslaunch tabletop_workspace_opt shared_autonomy.launch \
    task_config:=config/tasks/desk_organize_v2_sa.yaml
# (wait for completion, then kill sim)

# 2. For optimized SE(3) layouts, use the headless comparison pipeline:
python3 scripts/eval/compare_se3_sa_3d.py --skip-random --tiers Med-A
```

### Visual comparison

Layout screenshots and comparison overlays are in `results/layouts/`:

```bash
xdg-open results/layouts/desk_baseline.png
xdg-open results/layouts/desk_me_best.png
xdg-open results/layouts/desk_comparison.png
```

Inference videos are in `results/videos/`:

```bash
vlc results/videos/desk_baseline_inference.mp4
vlc results/videos/desk_me_best_inference.mp4
```

---

## 9. Using Optimized Scene YAMLs

The current optimized-layout source of truth is:

```text
config/scenes/scene_<name>_se3_me_optimized.yaml
```

The corresponding optimized XML files are intentionally not maintained. Launch
the base XML scene and use scripts that explicitly apply the optimized YAML
layout:

- headless evaluation: `scripts/eval/compare_se3_sa_3d.py`;
- manual grasp audit: `scripts/eval/manual_audit_se3_grasp.py`;
- full task execution: `scripts/run_task.py --scene scene_<name>_se3_me_optimized`.

---

## 10. Analytical Evaluation (No Simulator) — ⚠️ LEGACY

> **This section is retained for historical reference only.** The
> analytical evaluation predates the SE(3) MAP-Elites and headless SA
> pipeline. It uses a simplified 2D user/inference model and writes to
> the archived `results/map_elites_v2_*.json` paths.
>
> **For current SE(3) workspace optimization and evaluation, use:**
>
> - `scripts/optimize/optimize_se3_map_elites.py` — MAP-Elites (SE(3) yaw-aware)
> - `scripts/eval/compare_se3_sa_3d.py` — headless SA evaluation
> - `scripts/eval/sweep_threshold_se3.py` — threshold sensitivity
>
> Full instructions are in `docs/shared_autonomy_experiments.md`. The
> commands below are unchanged from the Apr 2026 pipeline but should
> not be used to produce new comparisons.

Runs Monte Carlo simulation of Bayesian intent inference without
MuJoCo. Faster but uses a simplified user/inference model.

```bash
# Run all 6 scenes
python3 scripts/eval/eval_map_elites_tiers.py --seeds 3 --n-trials 50

# Run one scene
python3 scripts/eval/eval_map_elites_tiers.py --scenes desk --seeds 1 --n-trials 50

# Generate result figures
python3 scripts/render/plot_results_figures.py
```

Results are saved to `results/map_elites_v2_*.json` and
`results/map_elites_evaluation_results.md`.

**Note:** The analytical evaluation uses a different inference model
than the MuJoCo shared autonomy runner. Results show the same trends
(optimized is faster) but the absolute numbers differ significantly.
See `results/map_elites_evaluation_results.md` for details.

---

## 11. Available Scenes and Tasks

### Scenes

| Scene Name | Objects | Description |
|------------|:-------:|-------------|
| `scene_breakfast_easy` | 3 | Bowl (fixed) + cereal, banana |
| `scene_desk` | 5 | Mug, book, pen_cup, phone, stapler |
| `scene_breakfast` | 6 | Bowl, cereal, napkin, spoon, banana, milk_carton |
| `scene_kitchen_prep` | 5 | Cutting_board (fixed), apple, can, bottle, sponge |
| `scene_meal_assembly` | 7 | Bowl, cutting_board (fixed), cereal, banana, apple, can, bottle |
| `scene_cluttered` | 11 | 8 colored blocks/cylinders + 3 fixed trays/bin |

Each baseline scene has a `*_se3_me_optimized.yaml` variant with MAP-Elites
optimized object positions and yaws.

### Tasks (State-Machine Format)

| Task File | Scene | Pick Objects | Description |
|-----------|-------|--------------|-------------|
| `set_table_easy_sa` | breakfast_easy | cereal, banana | 2-pick, place near bowl |
| `desk_organize_v2_sa` | desk | mug, stapler, pen_cup | 3-pick, place near phone/mug |
| `full_breakfast_sa` | breakfast | cereal, banana, milk_carton | 3-pick with pour actions |
| `set_table_sa` | breakfast | cereal, banana | 2-pick, place near bowl |
| `sort_by_zone_sa` | kitchen_prep | apple, can, bottle | 3-pick, sort into zones |
| `kitchen_prep_sa` | kitchen_prep | apple, can, bottle | 3-pick, place on cutting board |
| `meal_assembly_sa` | meal_assembly | apple, banana, bottle, can, cereal | 5-pick, long horizon |
| `cluttered_sort_sa` | cluttered | 8 blocks/cylinders | 8-pick, sort into trays/bin |

**Recommended for testing:** Start with `desk_organize_v2_sa` (3
objects, graspable, well-tested).

---

## 12. Troubleshooting

### RViz popup blocks interaction

```bash
xdotool search --name "ROS End of Life" windowactivate --sync key Return
```

### "Failed to load scene" or missing YAML

The base XML must exist under `src/assets/`, and the selected scene metadata
must exist under `config/scenes/`. Optimized layouts are YAML-only; do not pass
`scene_name:=scene_<name>_se3_me_optimized` to `sim_moveit.launch` unless a
matching XML was intentionally generated outside the current maintained
workflow.

### Shared autonomy: "No detections received"

The simulation server needs a few seconds after launch to start
publishing detections. The SA runner waits automatically, but if
it times out, restart the sim.

### Motion planning: "Step N FAILED"

MoveIt planning is non-deterministic. Re-run the task — it may
succeed on the next attempt. For persistent failures:

- Check if the object is reachable (within robot workspace)
- Check for arm-object collisions in MuJoCo
- Try adjusting grasp poses in `config/grasp_poses.yaml`

### Objects at wrong height after optimization

Check that the base scene YAML and the corresponding
`*_se3_me_optimized.yaml` agree on object `half_extents`. Explicit optimized
object `position.z` values should place the object on the tabletop for the
current short/flat geometry variant.

### Inference never reaches threshold

Try lowering the threshold or increasing noise:

```bash
roslaunch ... threshold:=0.7 noise:=0.02
```

---

## 13. Cleanup

**Always clean up after testing** to avoid stale processes:

```bash
# Full teardown
rosnode kill -a 2>/dev/null || true
pkill -f "roslaunch" || true
pkill -f "rviz" || true
pkill -f "mujoco" || true
pkill -f "simulation_server.py" || true
pkill -f "move_to_cartesian_pose.py" || true
pkill -f "relaxed_ik_rust.py" || true
pkill -f "moveit_ros_move_group/move_group" || true
pkill -f "rosout/rosout" || true
pkill -f "static_transform_publisher" || true
pkill -f "robot_state_publisher" || true
pkill -f "shared_autonomy_runner" || true
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/eval/run_sa_headless.py` | Headless SA runner (no ROS, recommended) |
| `scripts/eval/compare_se3_sa_3d.py` | Current SE(3) random-vs-MAP-Elites shared-autonomy comparison |
| `scripts/eval/sweep_threshold_se3.py` | Current SE(3) threshold sensitivity sweep |
| `scripts/eval/compare_se3_grasp_feasibility.py` | Random-vs-MAP-Elites grasp feasibility comparison |
| `scripts/eval/audit_se3_grasps.py` | Batch SE(3) grasp audit |
| `scripts/eval/run_sa_all_orderings.sh` | All orderings via ROS (requires sim running) |
| `scripts/eval/eval_map_elites_tiers.py` | MAP-Elites optimization + analytical evaluation |
| `scripts/render/plot_results_figures.py` | Generate result figures |
| `scripts/run_task.py` | Run task with full motion planning + grasping |
| `scripts/eval/run_sa_benchmark.py` | Benchmark shared autonomy across scenes |
