# MuJoCo Intent-Grasp Migration Guide

This guide is for continuing development on a lab machine with Codex or Claude.
It focuses only on the current project direction:

- use the published `intent_inference` module for object-level intent
- use the MuJoCo + GraspNet pipeline for candidate grasp generation
- use the existing shared-autonomy selector as the current grasp-level baseline
- collect training data first
- add VLM scoring and lightweight fusion training after the data pipeline is stable

## Branch

Work on branch:

```bash
fix/static-scene-visualization
```

Repository root:

```bash
~/catkin_ws/src/tabletop_workspace_opt
```

## Current Focus

The current integration milestone is:

1. launch MuJoCo simulation
2. run object-level intent inference
3. run candidate grasp generation
4. run the baseline shared-autonomy selector
5. log image + detections + intent outputs + grasp candidates + selector outputs

This milestone is implemented by:

- `launch/intent_grasp_data_collection.launch`
- `scripts/intent_grasp_dataset_logger.py`

## What Is Already Wired

`launch/intent_grasp_data_collection.launch` includes:

- `shared_autonomy_graspnet_validation.launch`
- `intent_inference_node.py`
- `intent_grasp_dataset_logger.py`

Main topics used by the logger:

- `/realsense/color/image_raw`
- `/mujoco_sim/detections`
- `/intent_inference/distribution`
- `/intent_inference/top_goal`
- `/intent_inference/top_pose`
- `/shared_autonomy/candidate_grasps`
- `/shared_autonomy/grasp_scores`
- `/shared_autonomy/selected_grasp`
- `/relaxed_ik/ee_vel_goals`

## Lab Machine Setup

Clone or update the repo:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone <repo-url> tabletop_workspace_opt
cd tabletop_workspace_opt
git checkout fix/static-scene-visualization
git pull origin fix/static-scene-visualization
```

Build:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make --pkg tabletop_workspace_opt
source ~/catkin_ws/devel/setup.bash
```

## Main Run Command

Use this as the default development entry point:

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch tabletop_workspace_opt intent_grasp_data_collection.launch
```

Useful overrides:

```bash
roslaunch tabletop_workspace_opt intent_grasp_data_collection.launch \
  task_instruction:="pick up the intended object" \
  output_dir:=/tmp/intent_grasp_dataset_run1 \
  min_sample_period_sec:=1.0 \
  with_rviz:=true \
  with_keyboard_teleop:=true
```

## Dataset Output

Default output location:

```bash
/tmp/intent_grasp_dataset
```

Files:

- `records.jsonl`
- `images/*.png`

Each sample should include:

- task instruction
- saved RGB image path
- detections
- object-level intent distribution
- top inferred goal
- candidate grasps
- selector score breakdown
- final selected grasp
- latest user input vector if available

## Expected Development Order

Do not skip straight to VLM integration on a fresh machine.

Recommended order:

1. verify that `intent_grasp_data_collection.launch` runs
2. verify that `records.jsonl` and `images/` are being populated
3. inspect logged samples for topic consistency
4. add an offline VLM labeling script
5. add a lightweight training script for grasp fusion
6. only then consider reconnecting the trained model back into online selection

## Recommended Next Files

If continuing development, the next likely files are:

- `scripts/offline_vlm_labeler.py`
- `scripts/train_grasp_fusion.py`
- `launch/intent_grasp_data_collection.launch`
- `scripts/intent_grasp_dataset_logger.py`

## What Codex/Claude Should Treat As Stable

These components should be treated as existing baselines unless explicitly asked to redesign them:

- `src/shared_autonomy/intent_inference_node.py`
- `scripts/shared_autonomy_grasp_selector.py`
- `scripts/graspnet_realsense_node.py`
- `launch/shared_autonomy_graspnet_validation.launch`

The goal is to build on top of them, not replace them all at once.

## What Not To Change Casually

Avoid large refactors to:

- the published intent inference algorithm
- the AprilTag real-world demo pipeline
- GraspNet runtime internals
- MoveIt execution logic

Those are not the current bottleneck.

## How To Use Codex or Claude Effectively

When handing work to Codex or Claude on the lab machine, ask for one bounded task at a time.

Good prompts:

- "Run `intent_grasp_data_collection.launch`, inspect failing topics, and patch only the logger or launch file."
- "Add `offline_vlm_labeler.py` that reads `records.jsonl` and writes `vlm_semantic_score` per candidate."
- "Train a logistic regression baseline from the logged dataset and save weights to JSON."

Bad prompts:

- "Redesign the whole project."
- "Refactor all shared autonomy code."
- "Integrate VLM, training, and real robot demo all at once."

## Commit Discipline

Only commit files directly related to the current task.

Before committing:

```bash
cd ~/catkin_ws/src/tabletop_workspace_opt
git status --short
```

For the current milestone, the core files are:

- `CMakeLists.txt`
- `launch/intent_grasp_data_collection.launch`
- `scripts/intent_grasp_dataset_logger.py`
- `MIGRATION_DEV_GUIDE.md`

Commit example:

```bash
git add CMakeLists.txt \
  launch/intent_grasp_data_collection.launch \
  scripts/intent_grasp_dataset_logger.py \
  MIGRATION_DEV_GUIDE.md
git commit -m "Add MuJoCo intent-grasp data collection pipeline"
git push origin fix/static-scene-visualization
```

## Notes On Existing Unrelated Changes

This repo currently also has uncommitted AprilTag and documentation work.
Do not bundle those into the MuJoCo data-collection commit unless explicitly requested.

Keep the data-collection pipeline commit small and clean.
