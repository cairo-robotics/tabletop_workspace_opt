# Codex Handoff

## Current State

We now have two scene concepts:

- `separated scene`: the original milk-selection workspace using the legacy pose file
- `grouped scene`: similar objects are grouped within category, while categories are still spatially separated

The grouped scene currently contains:

- milk cluster: milk, soy, oat
- fruit cluster: apple, orange
- cereal cluster: Quaker cereal, GM cereal

## Current Files

- separated scene grasp file:
  - `config/fixed_grasp_candidates.yaml`
- grouped scene grasp file:
  - `config/fixed_grasp_candidates_grouped.yaml`

## What Has Been Implemented

- Added runtime target-selection GUI:
  - `src/shared_autonomy/grasp_target_selector_gui.py`
- `shared_autonomy_pregrasp_selector.py` now listens to the selected grasp label topic
- `test_pour_task_sequence.py` now listens to the selected grasp label topic
- Launch files were updated so the GUI can change the active target at runtime
- Pour return path was made safer by inserting an extra upward lift phase
- Pour sequence was updated to use the actual grasp reference pose for carry/place-back behavior

## Grouped Scene Grasp Coverage

Already available in `config/fixed_grasp_candidates_grouped.yaml`:

- `side_pregrasp_milk`
- `side_grasp_milk`
- `apple_top_grasp`
- `orange_top_grasp`
- `oat_side_grasp`
- `soy_side_grasp`
- `quaker_cereal_side_grasp`
- `gm_cereal_side_grasp`

These currently contain pregrasp/grasp stages only.

## Research Framing

The current project stage is best framed as:

- workspace optimization for clearer object-intent inference in task-oriented shared autonomy

At this stage, the user mainly specifies the object, while the system executes the corresponding task policy:

- milk/soy/oat -> cup-related task
- fruit -> plate-related task
- cereal -> bowl-related task

This is not yet full destination-preference inference. It is object-intent disambiguation under different workspace layouts.

## Immediate Next Steps

1. Define semantic destination regions:
   - `plate_region`
   - `bowl_region`
   - `cup_region`

2. Record one hover/reference EE pose for each region
   - not a precise release pose
   - just a safe pose above the region

3. Add an object-to-region mapping, for example:
   - `apple -> plate_region`
   - `orange -> plate_region`
   - `milk_box -> cup_region`
   - `soy_box -> cup_region`
   - `oat_box -> cup_region`
   - `quaker_box -> bowl_region`
   - `gm_box -> bowl_region`

4. Implement a simple `pick_to_region` execution sequence
   - wait for grasp completion
   - lift from current grasp pose
   - move to region hover
   - descend
   - open gripper
   - retreat

5. After that, extend cereal with task-specific pour poses if needed

## Scene Taxonomy

- `separated`: all targets are clearly separated
- `grouped`: similar objects grouped within category, categories still separated
- `cluttered`: categories and targets intermixed more tightly

Right now, the active new scene is the `grouped` scene.
