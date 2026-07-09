# Sandwich Task Migration

This note describes the minimum changes required to replace the old
`Make Breakfast` user-study task with a `Make Sandwich` task.

## What Was Changed

- `task_context_manager.py` now loads step commands and `allowed_tag_ids`
  directly from the active tasks YAML.
- `apriltag_filtered_executor.py` now validates the selected target against the
  current `/task_context/allowed_tag_ids` set instead of relying only on
  breakfast-specific phase names.
- `apriltag_task_context.launch`, `apriltag_user_study.launch`, and
  `sam_lego_user_study.launch` now pass a configurable `tasks_yaml`.
- Example sandwich configs were added:
  - `config/user_study_tasks_sandwich.example.yaml`
  - `config/apriltag_object_map_sandwich.example.yaml`

## What You Still Need To Fill In

1. Assign real tag IDs for:
   - bottom bread
   - top bread
   - ham
   - tomato
   - onion
   - optional cheese

2. Replace the placeholder values in:
   - `config/user_study_tasks_sandwich.example.yaml`
   - `config/apriltag_object_map_sandwich.example.yaml`

3. Replace placeholder `grasp_complete_label` values with the real labels
   published by your AprilTag or SAM/PCA grasp pipeline.

4. Point the user-study launch to the sandwich tasks YAML, for example:

```bash
roslaunch tabletop_workspace_opt apriltag_user_study.launch \
  tasks_yaml:=$(rospack find tabletop_workspace_opt)/config/user_study_tasks_sandwich.yaml
```

## Recommended Transition Path

1. Keep `sorting` and `lego_sorting` unchanged.
2. Copy `user_study_tasks_sandwich.example.yaml` to a real
   `user_study_tasks_sandwich.yaml`.
3. Copy `apriltag_object_map_sandwich.example.yaml` into either:
   - your main object map, or
   - a dedicated sandwich object map used in a separate launch.
4. Pilot the sandwich task with AprilTag-based bread slices first.
5. Add SAM/PCA-generated candidates for small fillings after the task flow is
   stable.

## Research Framing

The sandwich task should be treated as a high-ambiguity target-selection task,
not as a generic grasping benchmark.

- `Sorting` remains the lower-ambiguity comparison task.
- `Sandwich Assembly` is the higher-ambiguity task because pieces are visually
  similar, thin, and order-dependent.
- The main comparison should be `unoptimized` versus `optimized` scene layout
  within each task.
- The main dependent variables should focus on target disambiguation:
  - target selection correctness
  - time to commit
  - top-goal switching
  - cancel count
  - downstream task completion
