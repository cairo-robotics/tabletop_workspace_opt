# Study Operator Guide

This document is for invited testers and study helpers. It explains the user study in simple terms and describes what to expect during a test session.

## What This Study Is About

In this study, the operator will use a controller to guide a robot during tabletop tasks.

The system tries to infer the operator's intended target from their motion and then assists with grasping or placement.

The main tasks currently used in the study are:

- `Make Breakfast`
- `Sorting`
- `LEGO Sorting`
- `Sandwich Assembly`

## What The Tester Will Do

During the session, the tester will:

1. receive a short explanation of the task
2. complete a short practice round
3. perform one or more study tasks using the controller
4. complete short surveys during and after the session

## What The Tester Should Know

- The tester does not need prior experience with robotics.
- The goal is to interact naturally and clearly.
- The tester should move toward the object or destination they intend to select.
- Before the system is confident enough, the tester should keep moving toward the intended target.
- If the system selects the correct target, the tester confirms it.
- If the system selects the wrong target, the tester cancels and continues.
- Small adjustment is allowed at the pregrasp pose and prerelease pose.

## Controller Actions

The tester will be told these controls before starting:

- `X`: confirm the current target, and later continue from pregrasp to grasp
- `Y`: cancel the current target
- `A`: close the gripper when prompted
- `B`: open the gripper when prompted

The tester does not need to memorize everything in advance. The experimenter will explain the controls before the task begins.

## What The Tasks Look Like

### Make Breakfast

- first choose one breakfast ingredient
- then choose one milk carton
- after the ingredient is grasped, the task advances to the milk-selection step

### Sorting

- first choose one object
- then choose the destination container
- after placing one object, the task may continue with the remaining objects until the table is finished

### LEGO Sorting

- first choose one LEGO brick
- then choose the destination container
- after placing one brick, the task may continue with the remaining bricks

### Sandwich Assembly

- first choose one sandwich piece
- then choose the placement target
- after placing one piece, the task returns to sandwich-piece selection
- the participant decides how many layers to build

## Study Conditions

The same task may be run in different scene layouts.

- `Unoptimized scene`: objects are more crowded or visually ambiguous, so target selection is harder.
- `Optimized scene`: objects are arranged to make the intended target easier to disambiguate.

The study focus is whether scene layout makes the intended target easier or harder for the system to infer. It is not mainly a test of generic robot grasping.

## Surveys

The study uses three surveys.

Whenever possible, surveys should be completed on a separate survey device such as a tablet or a second laptop. This keeps the survey process independent from the ROS control computer and avoids interference from the experimenter's browser session.

- Interest survey:
  `https://docs.google.com/forms/d/e/1FAIpQLSfaBFg-U7iyXgra0fg_WstYixT402ES40mRpHRJIUQGPB7jcg/viewform`
- Post-condition survey:
  `https://docs.google.com/forms/d/e/1FAIpQLSfmKlQRKYBoFq-img7ED_Inh8qectd-nVpi_7vqXwMh1aEUMw/viewform`
- Final survey:
  `https://docs.google.com/forms/d/e/1FAIpQLSdUuuwf04hOBwj14NWEfH3horZXOuiPygKczDfi31ycvxj56Q/viewform`

Expected timing:

- `Interest survey`: before the study tasks begin
- `Post-condition survey`: after each study block
- `Final survey`: after the full session ends

## If Something Goes Wrong

- If the robot or interface behaves unexpectedly, the tester should stop and wait for the experimenter.
- If the tester is unsure what to do, they should ask the experimenter.
- Safety takes priority over task completion.

## Short Participant Script

The experimenter can read the following short version aloud:

1. "Use the controller to show which target you want."
2. "Move clearly toward that target. Before the system is confident enough, just keep moving."
3. "When the screen says the target is locked, press `X` to confirm or `Y` to cancel."
4. "At pregrasp, you may make a small adjustment, then press `X` again."
5. "At grasp, press `A` to close the gripper."
6. "For placement, move toward where you want to place the item. When that target is locked, press `X`."
7. "At prerelease, you may make a small height adjustment, then press `B` to release."

## What The Experimenter Will Handle

The experimenter will:

- launch the system
- prepare the task objects
- explain the task
- help if the system needs to be reset
- manage the study records and exported logs

The tester only needs to focus on performing the task and answering the surveys.
