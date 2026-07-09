# Data Analysis Checklist

This document is intended for the research team during pilot review, data cleaning, and paper writing.

## Purpose

Use this file after data collection, not as participant-facing or tester-facing instructions.

It is meant for:

- checking whether a block should be kept for analysis
- confirming that exported logs match the study design
- aligning questionnaire data with objective metrics
- freezing the analysis protocol before writing the paper

## Block Validity Rules

### Valid Block

A block is valid if:

- the intended task and condition were run
- the participant understood the instructions
- the system remained usable throughout the block
- the log was written successfully
- the block can be exported and matched to questionnaire data

### Restarted Block

Treat a block as restarted and exclude the earlier attempt from final analysis if:

- wrong `participant_id`, `condition_id`, or `block_id` was used
- the wrong task was started
- the scene was scanned incorrectly before the task began
- the participant clearly misunderstood the instructions
- the robot state and UI state became inconsistent
- the run was interrupted before meaningful interaction began

Recommended label:

- `restart`

### Aborted Block

Treat a block as aborted if:

- the camera stream failed and could not be recovered quickly
- the controller stopped working
- the robot entered an unsafe or unusable state
- repeated software instability prevented meaningful completion

Recommended label:

- `aborted`

Exclude aborted blocks from main analysis unless you have a separate failure-analysis plan.

### Aborted Session

Treat a full session as aborted if:

- a safety issue occurred
- the hardware remained unstable
- multiple blocks failed because of system faults
- the participant could not continue

Recommended label:

- `session_aborted`

## Practice Block Policy

- Every participant should complete one practice block before formal data collection.
- Practice data should not be used in the final analysis.
- Practice should use a fixed task and instruction script across participants.

## Export Integrity Checks

For each exported participant block, confirm:

- the trial log exists
- the block analysis CSV exists
- `participant_id` is correct
- `condition_id` is correct
- `block_id` is correct
- the number of exported rows is plausible for the task
- `success`, `correct_inference`, and `failure_reason` match the observed run
- no unexpected inactive shutdown rows remain in the analysis export

## Questionnaire Mapping

Treat the questionnaire as block-level data unless a form is explicitly session-level.

### Join Keys

Use these fields to join questionnaire data with exported block analysis:

- `participant_id`
- `condition_id`
- `block_id`

Use `session_id` as an additional join key if needed.

### Recommended Interpretation By Task

- `Sorting`
  - treat object selection and destination selection as the main intent-inference outcomes
- `Sandwich Assembly`
  - treat each sandwich layer as a target-disambiguation event
  - analyze whether the correct piece becomes easier to infer under the optimized layout
  - do not frame the main claim as generic grasp success

## Objective-To-Subjective Mapping

- `Intuitiveness`
  - compare against `mean_time_to_commit_sec`, `mean_teleop_time_sec`, and `mean_confirmation_count`
- `Perceived Robot Understanding`
  - compare against `destination_correct_inference_rate`, `success_rate_destination`, and `mean_top_goal_switch_count`
- `Sense of Agency and Control`
  - compare against `mean_teleop_time_sec`, `mean_cancel_count`, and `mean_confirmation_count`
- `Human-Robot Fluency`
  - compare against `mean_duration_sec`, `mean_autonomous_time_sec`, and interruption counts
- `Usability`
  - compare against `success_rate_all`, `mean_duration_sec`, and `mean_timeout_count`
- `Workload`
  - compare against `mean_teleop_time_sec`, `mean_duration_sec`, `mean_cancel_count`, and `mean_top_goal_switch_count`

## Recommended Primary Metrics

For the current paper framing, prioritize:

- target selection correctness
- time to commit to the intended target
- top-goal switching / ambiguity indicators
- cancel count and wrong-target attempts
- sorting success rate
- sandwich completion rate
- task completion time
- autonomous assistance time
- joystick / teleoperation time

Recommended comparison rule:

- compare `unoptimized` vs `optimized` within the same task
- use `Sorting` as the lower-ambiguity task
- use `Sandwich Assembly` as the higher-ambiguity task
- test whether optimization helps both tasks and whether the effect is larger for sandwich assembly

## Version Lock Recommendation

Before formal participant data collection, freeze the following:

- launch files
- task YAML files
- questionnaire wording
- condition definitions
- trial exporter
- block exporter

If any of these change, record the change and treat later data as a different system version.
