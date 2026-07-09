# User Study Requirements

## Goal

Evaluate whether robot-optimized tabletop layouts improve shared-autonomy performance without making the workspace feel confusing, unnatural, or harder to use.

## Main Questions

1. Do optimized layouts improve intent inference compared with human-intuitive layouts?
2. Do optimized layouts reduce time, control effort, and wrong assistance?
3. How do optimized layouts affect workload, trust, predictability, control, and naturalness?
4. Is there a tradeoff between robot-legibility and human-intuitive organization?

## Core Hypotheses

- Optimized layouts improve inference accuracy.
- Optimized layouts reduce time to correct intent lock-in.
- Optimized layouts reduce task time and control effort.
- Optimized layouts reduce wrong assistance and user corrections.
- Optimized layouts may feel less natural than human-intuitive layouts.
- Users may prefer a compromise between highly optimized and highly intuitive layouts.

## Experimental Conditions

Required conditions:

- Human-Intuitive Layout
  - Objects are grouped in a way that feels natural to people.
- Robot-Optimized Layout
  - Objects are placed to improve goal separability for the inference model.

Optional conditions:

- Random Feasible Layout
  - Random placement that still satisfies reachability and clearance constraints.
- Balanced Layout
  - A compromise between robot-legibility and human naturalness.

## Interaction Mode

- Use shared autonomy for all main trials.
- The participant teleoperates the robot with a joystick or controller.
- The system updates its belief over candidate goals during motion.
- Once confidence passes a threshold, the system commits to an inferred goal.
- If enabled, the participant confirms or rejects the inferred goal.
- The robot then runs the assistance behavior such as grasp, pour, place, or return.

Manual-only control can be included as a small secondary baseline, but it should not be the main comparison.

## Study Design

- Within-subject design.
- Every participant sees every layout condition.
- Counterbalance layout order across participants.
- Randomize or counterbalance scene order and target order within each condition.

## Participants

Recommended sample sizes:

- Pilot: 4 to 6 participants.
- Main study: 18 to 24 participants.
- Smaller validation study: 12 to 16 participants.

Inclusion criteria:

- Age 18 or older.
- Able to understand instructions.
- Able to use a joystick or game controller.
- No prior experience with the exact system required.

Record:

- Participant ID.
- Age range.
- Dominant hand.
- Prior teleoperation experience.
- Prior joystick or game controller experience.
- Prior robotic arm experience.
- Prior shared autonomy or assistive robotics experience.

## Recommended Pilot Task

Use drink selection and pouring as the first pilot task:

- Milk
- Soy milk
- Oat milk

Reason:

- The objects are visually similar.
- The task is simple and repeatable.
- It directly tests intent disambiguation.

Additional domains can be added later:

- Tea and snack preparation
- LEGO sorting
- Assistive feeding

## Workspace Requirements

All layouts should satisfy:

- Workspace bounds
- Minimum inter-object clearance
- Minimum distance from robot start pose
- Reachability for all candidate grasps
- Collision-free access to targets
- Safety margins around fragile or hazardous objects

## Scene Benchmark

Recommended scene difficulty levels:

- Easy
  - Targets are well separated and visually distinct.
- Ambiguous
  - Similar objects are near each other, making intent harder to infer.
- Cluttered
  - Distractors or obstacles make selection and manipulation harder.

Recommended pilot scene set:

- 1 easy scene
- 2 ambiguous scenes
- 1 cluttered scene

If both layouts produce near-perfect performance, increase ambiguity.

## Session Procedure

1. Consent and introduction.
2. Demographic and background questionnaire.
3. Explain robot, controller, and safety rules.
4. Explain shared-autonomy behavior.
5. Run training trials.
6. Run experimental trials for each layout condition.
7. Collect post-condition questionnaires.
8. Collect final comparison questionnaire.
9. Run short interview and debrief.

## Training Requirements

- Familiarize participants with the controller.
- Let them move the robot in free space.
- Show how shared autonomy assists.
- Include one practice trial not used in the real study.
- Do not expose the exact experimental layouts during training.

## Trial Procedure

1. Load the selected scene and layout.
2. Tell the participant the target object or goal.
3. Reset the robot to its start pose.
4. Begin teleoperation.
5. Update belief over candidate goals during motion.
6. Commit when confidence crosses threshold.
7. Ask for confirmation if confirmation is enabled.
8. Execute the assistance routine if confirmed.
9. End the trial on success, failure, abort, or timeout.
10. Save logs and experimenter notes.

Recommended pilot timeout:

- 120 seconds
- Reduce to 90 seconds if pilot runs are consistently much shorter

## Failure Conditions

Mark a trial as failed if:

- The robot reaches for the wrong object.
- The robot grasps the wrong object.
- The robot collides with an object or the environment.
- The participant aborts the trial.
- The trial times out.
- The robot enters an unsafe or unrecoverable state.

## Metrics

Primary objective metrics:

- Intent inference accuracy
- Time to correct intent lock-in
- Task completion time
- Task success rate
- Wrong assistance rate
- Correction or override count
- User control effort

Useful control-effort signals:

- Joystick input magnitude
- End-effector path length
- Number of control commands

Secondary objective metrics:

- Posterior entropy over time
- Confidence at commitment
- Number of confirmation steps
- Confirmation latency
- Reset or aborted trial count
- Collision or near-collision count
- Idle time before assistance
- Number of belief switches before commitment

Subjective metrics after each condition:

- Workload
- Ease of use
- Trust
- Predictability
- Sense of control
- Helpfulness of assistance
- Naturalness of layout
- Preference for repeated use

Suggested tools:

- NASA-TLX for workload
- 1 to 7 Likert-scale questions for the other ratings

## Suggested Questionnaire Items

Participants can rate statements such as:

- The layout felt natural for the task.
- The layout made it easy to decide where to move.
- The robot seemed to understand my intent.
- The robot assisted at the right time.
- I felt in control of the robot.
- The robot behavior was predictable.
- The assistance reduced my effort.
- The assistance was distracting.
- I trusted the robot assistance.
- I would use this layout again.

## Required Logging

Per trial, record at minimum:

- Participant ID
- Trial ID
- Layout condition
- Task domain
- Scene ID
- Target object
- Target grasp, if relevant
- Layout ID
- Layout score or separability score
- Trial start and end times
- Completion time
- Success or failure
- Failure reason
- Final inferred goal
- Whether inference matched the target
- Belief trajectory over time
- Posterior entropy over time
- Time to correct intent inference
- Time to commitment
- Commitment confidence
- Belief switch count
- Override count
- Confirmation latency
- Wrong assistance count
- End-effector path length
- Joystick effort
- Collision count
- Experimenter notes

## Analysis Plan

Primary comparisons:

- Human-Intuitive vs Robot-Optimized
- Easy vs Ambiguous vs Cluttered
- Layout-by-difficulty interaction

For pilot analysis:

- Use descriptive statistics
- Use paired comparisons where appropriate

For a larger study:

- Use mixed-effects models
- Use linear models for continuous outcomes
- Use logistic models for binary outcomes
- Use Poisson or negative binomial models for count outcomes

If the sample is small, nonparametric paired tests are acceptable.

## Expected Outcomes

- Optimized layouts should improve inference accuracy.
- Optimized layouts should reduce time to correct intent.
- Optimized layouts should reduce wrong assistance and corrections.
- Gains should be largest in ambiguous and cluttered scenes.
- Optimized layouts may be rated as less natural.
- Balanced layouts may be preferred if purely optimized layouts feel strange.

## Pilot Success Criteria

The pilot is useful if it can separate conditions on at least one of:

- Inference accuracy
- Time to correct intent lock-in
- Wrong assistance rate
- Task completion time
- Control effort
- Workload
- Perceived layout naturalness

The pilot should also answer:

- Are the tasks too easy or too hard?
- Does optimization create measurable behavioral differences?
- Do users think optimized layouts are unnatural?
- Is the assistance understandable?
- Is confirmation necessary?
- Is the confidence threshold tuned well?
- Do users adopt strategies that affect inference?
- How long does each condition take?
- Are there safety or reset problems?

## Post-Study Interview

Ask participants:

1. Which layout felt easiest to use?
2. Which layout made the robot seem most helpful?
3. Which layout felt most natural?
4. When did the robot misunderstand your intent?
5. Did any layout make target expression easier or harder?
6. Did the robot assist too early or too late?
7. Did you feel in control?
8. Did the optimized layout feel unusual?
9. Would you prefer robot-friendly, human-friendly, or balanced layouts?
10. What would make the interaction more natural?

## Takeaway

The study should not only test whether optimized layouts improve inference. It should also test whether those layouts still support user understanding, comfort, and agency. The strongest result is not simply "optimization helps," but whether a good workspace can balance robot legibility with human usability.
