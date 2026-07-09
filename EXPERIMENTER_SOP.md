# Experimenter SOP

This document is intended as a printable standard operating procedure for running participant sessions.

## Study Units

- `session`: one participant's full visit or run day
- `block`: one `condition + task` combination
- `trial`: one logged task step

For multi-step tasks such as `Sorting`, one full pick-and-place cycle is normally logged as two trials:

- `pickup`
- `destination`

## Condition Structure

Keep the condition definition explicit and stable across participants.

Recommended metadata fields:

- `session_id`
- `participant_id`
- `condition_id`
- `block_id`
- `condition_order`
- `task_type`

Recommended rule:

- change only one main factor per condition whenever possible

Example condition comparison:

- `unoptimized`: crowded or ambiguous object arrangement
- `optimized`: layout arranged to improve target separability while keeping the task itself unchanged

## Pre-Session Setup

Before the participant arrives:

1. Power on the robot, RealSense camera, and controller.
2. Enter the ROS environment with `source /opt/ros/noetic/setup.bash` and `source ~/catkin_ws/devel/setup.bash`.
3. Launch the user study stack.
4. Open both pages:
   - on the same computer: `http://127.0.0.1:8766/operator` and `http://127.0.0.1:8766/participant`
   - on another computer: `http://<dashboard_host_ip>:8766/operator` and `http://<dashboard_host_ip>:8766/participant`
5. Confirm that:
   - the camera image is visible
   - the Web UI loads correctly
   - the controller is connected
   - the robot responds normally
   - the log directory is writable

## Session Metadata

Before each real participant block, set:

- `session_id`
- `participant_id`
- `condition_id`
- `block_id`

Example:

```bash
roslaunch tabletop_workspace_opt user_study.launch \
  frontend_stack:=sandwich \
  camera_mode:=light \
  session_id:=pilot_20260701 \
  participant_id:=P01 \
  condition_id:=optimized \
  block_id:=B3
```

Camera mode rule:

- use `camera_mode:=light` for normal participant sessions
- use `camera_mode:=full` only for debugging or troubleshooting
- `light` keeps the RGB stream active for AprilTag and LEGO grasping, but disables the heavier depth pipeline to reduce RealSense crashes

Recommended naming:

- `participant_id`: `P01`, `P02`, ...
- `condition_id`: `baseline`, `unoptimized`, `optimized`, ...
- `block_id`: `B1`, `B2`, ...
- `session_id`: one label for the full participant run or run date

## Participant Instruction Script

Use a fixed script so all participants receive the same instructions:

1. "You will use the controller to show which target you intend to select. Sometimes the target is an object to pick up, and sometimes it is a destination where you want to place the item."
2. "Move clearly toward the target you want. Avoid very small or hesitant motions."
3. "Before the system is confident enough, just keep moving toward the target. Do not press `X` yet."
4. "When the interface says the target is locked and asks if you are going for that target, press `X` to confirm or `Y` to cancel."
5. "After you confirm a pickup target, the robot will move to a pregrasp pose. You may make a small adjustment there. Press `X` again to execute the straight-down grasp motion."
6. "At the grasp pose, press `A` to close the gripper."
7. "After pickup, move clearly toward the next target shown by the task. For placement steps, keep moving until the placement target is locked, then press `X` to confirm prerelease."
8. "At the prerelease pose, you may make a small height adjustment. Press `B` to open the gripper."
9. "If anything looks wrong, stop and wait for the experimenter."
10. "Your goal is to complete the task as naturally and accurately as possible."

## Practice Block

Before formal data collection:

1. Run one short practice block.
2. Confirm that the participant understands:
   - how to move the robot
   - how to confirm with `X`
   - why `X` should only be pressed after the interface says the target is locked
   - how to cancel with `Y`
   - how to close with `A`
   - how to release with `B`
   - that slight adjustment is allowed at pregrasp and prerelease
   - how the system transitions between steps
3. Exclude practice data from the final analysis.

## Block Execution Procedure

For each block:

1. Arrange the physical objects for the assigned task and condition.
2. Confirm that the scene layout matches the intended condition.
3. Scan the workspace until all required candidates are visible and stable.
4. In the operator UI, start the assigned task.
5. Tell the participant to begin.
6. Observe the run without coaching the participant unless recovery is required.
7. If a recoverable error occurs, follow the recovery procedure below.
8. At block end, stop interaction and verify that the logs were recorded correctly.
9. Have the participant complete the questionnaire for that block.

## Send Robot Home

Use `Send Robot Home` from the operator page when you need the robot to return to a safe neutral position before the next action.

### When To Use It

Use `Send Robot Home`:

- before rescanning the workspace after a confused or interrupted run
- when the robot is left in an awkward pose between blocks
- before rearranging task objects by hand
- when you want to reset the physical robot pose without restarting the whole system

### How To Use It

1. Stop the current interaction.
2. If needed, press `Reset` or return the task to a neutral study state.
3. Press `Send Robot Home` in the operator UI.
4. Wait until the robot reaches neutral and stops moving.
5. Only then begin rescanning, rearranging objects, or starting the next block.

### What It Does

`Send Robot Home` now:

- pauses the active shared-autonomy motion
- clears the currently selected target
- moves the robot to neutral
- prevents the robot from jumping back to the previous target after home motion completes

### Important Caution

Do not use `Send Robot Home` while the participant is still actively trying to complete the current task unless you have decided to interrupt or restart that run.

## Task-Specific Execution Notes

- `Sorting`
  - step 1: select one object to sort
  - step 2: select the destination container
  - after a successful placement, the dashboard returns to the object-selection step so the remaining objects can continue to be sorted
  - the final remaining object can still be sorted even if only one sortable object is left in the scene
- `Make Breakfast`
  - step 1: select one breakfast ingredient
  - step 2: select one milk carton
  - after a successful ingredient pickup, the interface should guide the participant toward the milk-selection step
- `Sandwich Assembly`
  - step 1: select any sandwich piece to pick up
  - step 2: select one placement target such as the plate
  - after a successful placement, the dashboard returns to sandwich-piece selection so the participant can decide how many layers to build
  - the analysis focus is whether the intended piece or destination was easy or hard to disambiguate under the current scene layout
- `LEGO Sorting`
  - step 1: select one LEGO brick
  - step 2: select the destination container
  - after a successful placement, the dashboard returns to LEGO selection so the participant can continue with another brick

## Fault Handling

Use three fault classes.

### 1. Recoverable Fault

Examples:

- image temporarily missing
- incomplete scan
- wrong target lock
- UI not updating correctly

Recovery order:

1. cancel the current action
2. rescan the workspace if needed
3. restart the current task step if needed
4. continue the block only if the system state is clear

### 2. Restart-Block Fault

Examples:

- wrong session metadata
- block started with the wrong task or step
- robot state and UI state are no longer synchronized
- the participant clearly did not understand the task instructions

Recovery action:

1. terminate the current block
2. mark the block as invalid or restarted
3. restart the block from the beginning

### 3. Terminate-Session Fault

Examples:

- persistent camera failure
- robot safety concern
- controller failure
- repeated system instability

Recovery action:

1. stop the session
2. record the failure reason
3. do not continue until the system is stable again

## End-of-Block Checks

Before moving to the next block, confirm:

1. the task has actually finished
2. the robot is in a safe stable state
3. the trial log was written
4. `participant_id`, `condition_id`, and `block_id` are correct
5. the exporter runs successfully
6. the questionnaire response can be matched to this block

## Recommended Analysis Rule

Use one questionnaire response set per block, and keep one block tied to exactly one:

- participant
- task
- condition

Recommended interpretation by task:

- `Sorting`
  - use object-selection and destination-selection quality as the main intent outcomes
- `Sandwich Assembly`
  - use per-layer target-selection quality as the main intent outcome
  - do not frame the study around generic grasp capability
  - treat pickup and placement as downstream consequences of better or worse target disambiguation
