# Shared Autonomy System — Implementation Plan

> Archived note: this document describes an older prototype shared-autonomy architecture.
> Several files and launch entries referenced below were removed during repository cleanup and are no longer part of the maintained user study pipeline.
> For the current maintained stack, use `apriltag_user_study.launch`, `apriltag_shared_control.launch`, `sam_lego_user_study.launch`, and the top-level workflow in `README.md`.

## Architecture Overview

Four new components interact with the existing simulation:

```
 task_state_machine ──valid_goals──> intent_inference
       │                                    │
       │ teleport_object                    │ auto_complete_trigger
       v                                    v
 simulation_server <────────────── task_auto_complete
       ^                                    │
       │ joint targets                      │ gripper
 simulated_joystick ────────────────────────┘
```

1. **Simulated Joystick Controller** — Generates velocity commands (with configurable noise) to drive the EE directly in MuJoCo, bypassing MoveIt.
2. **Task State Machine** — Defines a graph of possible goals at each state, replacing the linear step list.
3. **Intent Inference Module** — Observes EE trajectory and predicts which goal the user is pursuing.
4. **Task Auto-Completion** — When confidence exceeds a threshold, teleports the held object to its place position and moves the EE to the post-place position.

---

## 1. New Task YAML Format (State Machine)

The current YAML format is a flat list of sequential steps. The new format represents a directed graph of states where multiple goals are valid at each point.

```yaml
task:
  name: full_breakfast
  description: "Full breakfast preparation"

  objects: [cereal, banana, milk_carton, bowl]

  states:
    initial:
      description: "Nothing picked up yet"
      valid_goals:
        - id: pick_cereal
          action: pick
          object: cereal
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: holding_cereal
        - id: pick_banana
          action: pick
          object: banana
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: holding_banana
        - id: pick_milk
          action: pick
          object: milk_carton
          orientation: {qx: 0.707, qy: 0.707, qz: 0.0, qw: 0.0}
          next_state: holding_milk

    holding_cereal:
      description: "Holding cereal box"
      valid_goals:
        - id: pour_cereal
          action: pour
          destination:
            reference: bowl
            offset: {x: 0.0, y: 0.0, z: 0.45}
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          pour_orientation: {qx: 0.924, qy: 0.0, qz: 0.0, qw: 0.383}
          hold_time: 3.0
          next_state: holding_cereal_poured
        - id: place_cereal
          action: place
          destination:
            reference: bowl
            offset: {x: -0.15, y: -0.15, z: 0.08}
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: cereal_placed

    holding_cereal_poured:
      description: "Holding cereal after pouring"
      valid_goals:
        - id: place_cereal_after_pour
          action: place
          destination:
            reference: bowl
            offset: {x: -0.15, y: -0.15, z: 0.08}
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: cereal_placed

    cereal_placed:
      description: "Cereal placed, pick next object"
      valid_goals:
        - id: pick_banana
          action: pick
          object: banana
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: holding_banana
        - id: pick_milk
          action: pick
          object: milk_carton
          orientation: {qx: 0.707, qy: 0.707, qz: 0.0, qw: 0.0}
          next_state: holding_milk

    holding_banana:
      valid_goals:
        - id: place_banana_in_bowl
          action: place
          destination:
            reference: bowl
            offset: {x: 0.0, y: 0.0, z: 0.06}
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: banana_placed

    banana_placed:
      valid_goals:
        - id: pick_milk_after_banana
          action: pick
          object: milk_carton
          orientation: {qx: 0.707, qy: 0.707, qz: 0.0, qw: 0.0}
          next_state: holding_milk

    holding_milk:
      valid_goals:
        - id: pour_milk
          action: pour
          destination:
            reference: bowl
            offset: {x: 0.0, y: 0.0, z: 0.45}
          orientation: {qx: 0.707, qy: 0.707, qz: 0.0, qw: 0.0}
          pour_orientation: {qx: 0.653, qy: 0.653, qz: -0.271, qw: 0.271}
          hold_time: 3.0
          next_state: holding_milk_poured
        - id: place_milk
          action: place
          destination:
            reference: bowl
            offset: {x: 0.15, y: 0.0, z: 0.0}
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: milk_placed

    holding_milk_poured:
      valid_goals:
        - id: place_milk_after_pour
          action: place
          destination:
            reference: bowl
            offset: {x: 0.15, y: 0.0, z: 0.0}
          orientation: {qx: 1.0, qy: 0.0, qz: 0.0, qw: 0.0}
          next_state: milk_placed

    milk_placed:
      valid_goals: []  # terminal state (or link back for more objects)

    done:
      description: "All tasks complete"
      valid_goals: []

  # Post-task verification goals (same as current format)
  goals:
    - description: "Cereal is near the bowl"
      type: near
      object: cereal
      reference: bowl
      max_distance_xy: 0.25
    - description: "Banana is above the bowl"
      type: above
      object: banana
      reference: bowl
      min_z_offset: -0.03
    - description: "Milk is near the bowl"
      type: near
      object: milk_carton
      reference: bowl
      max_distance_xy: 0.25
```

**Key design decisions:**
- Each state has `valid_goals` defining what the user could be trying to do.
- Each goal has a `next_state` to transition to upon completion.
- The `id` field uniquely identifies each goal for intent inference.
- For `pick` goals, the inference target is the object's current position.
- For `place`/`pour` goals, the target is computed from reference + offset.
- The `goals` section at the bottom is retained for post-task verification.

---

## 2. Simulated Joystick Velocity Controller

**New file:** `src/shared_autonomy/simulated_joystick_node.py`

This node simulates a human controlling the EE via a joystick. It generates velocity commands toward a secretly-chosen goal, with configurable Gaussian noise.

### How it works

1. Read current EE pose from `/mujoco_sim/endpoint_state`
2. Read current joint positions from `/joint_states`
3. Compute direction vector from EE to the user's intended target
4. Scale to a fixed speed (configurable)
5. Add Gaussian noise: `v_noisy = v_ideal + N(0, sigma²)`
6. Convert Cartesian velocity to joint velocities using MuJoCo Jacobian (`mj_jacSite`)
7. Integrate: `q_new = q_current + J_pinv * v_cartesian * dt`
8. Publish `q_new` to `relaxed_ik/joint_angle_solutions` (reuses existing PID controller)

### Why position-based control (not raw velocity)

By integrating velocity into position targets and using the existing PID controller in `mujoco_visualizer.py`, we avoid modifying the core simulation loop. The PID controller naturally handles smoothing and stability.

### Jacobian computation

The node loads the MuJoCo model read-only for Jacobian computation. It does NOT step physics — it only uses `mj_jacSite` with a temporary `MjData` set to the current joint positions from `/joint_states`.

### Hyperparameters (ROS params)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `~noise_sigma_linear` | Std dev of Gaussian noise on linear velocity (m/s) | 0.005 |
| `~noise_sigma_angular` | Std dev of Gaussian noise on angular velocity (rad/s) | 0.01 |
| `~max_speed` | Maximum linear speed (m/s) | 0.05 |
| `~control_rate` | Control loop rate (Hz) | 30 |
| `~approach_gain` | Proportional gain for direction toward goal | 1.0 |

### Gripper control

The simulated user also opens/closes the gripper at appropriate times. When the EE is within a threshold distance of a pick target, the node sends a gripper close command via `/operate_gripper`. This is a simple distance-based trigger.

---

## 3. Intent Inference Module

**Modify:** `src/shared_autonomy/intent_inference_node.py`

The existing node uses a softmax over path-efficiency scores. Changes needed:

### 3a. Subscribe to valid goals from state machine

- Subscribe to `/shared_autonomy/valid_goals` (published by the task state machine node)
- This replaces the current object detection-based goal list
- Valid goals include computed positions (object positions for pick, destination positions for place/pour)

### 3b. Inference algorithm

For each valid goal g:
```
P(g) = softmax(-beta * (L_obs + dist(EE, g)) / dist(start, g))
```
Where:
- `L_obs` = total path length observed so far
- `dist(EE, g)` = straight-line distance from current EE to goal g
- `dist(start, g)` = straight-line distance from start position to goal g
- `beta` = rationality coefficient (hyperparameter, higher = more confident)

### 3c. Auto-completion trigger

When `max(P(g)) > intent_action_threshold`:
- Publish `AutoCompleteTrigger` message with the goal ID and confidence
- The threshold is a hyperparameter (default 0.8)

### Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `~beta` | Rationality coefficient | 1.0 |
| `~intent_action_threshold` | Confidence threshold for auto-completion | 0.8 |

---

## 4. Task Auto-Completion (Object Teleportation)

**New file:** `src/shared_autonomy/task_auto_complete_node.py`

When triggered by intent inference, instead of running full motion planning:

### For pick goals:
1. The EE is already near the object (that's how intent was inferred)
2. Close gripper via `/operate_gripper`
3. Lift EE slightly by publishing new joint targets

### For place goals:
1. Open gripper via `/operate_gripper`
2. Teleport object to destination via new `/sim/teleport_object` service
3. Move EE to post-place position (lift up 15cm)

### For pour goals:
1. Teleport object to pour position above bowl
2. Hold briefly (simulating pour)
3. Transition to next state

### Object teleportation implementation

Objects in MuJoCo use `freejoint` (7 qpos: x, y, z, qw, qx, qy, qz). To teleport:
1. Find body ID via `mujoco.mj_name2id`
2. Find joint qpos address
3. Set `data.qpos[addr:addr+7] = [x, y, z, qw, qx, qy, qz]`
4. Zero velocity: `data.qvel[dof:dof+6] = 0`
5. Call `mj_forward` to update

This requires a new method in `mujoco_visualizer.py` and a new ROS service in `simulation_server.py`.

---

## 5. Task State Machine Node

**New file:** `src/shared_autonomy/task_state_machine_node.py`

The orchestrator that ties everything together:

1. Loads the new YAML task config
2. Tracks the current state in the state machine
3. Computes target positions for each valid goal:
   - For `pick` goals: query object position from `/mujoco_sim/detections`
   - For `place`/`pour` goals: query reference object position + offset
4. Publishes valid goals to `/shared_autonomy/valid_goals`
5. Subscribes to `/shared_autonomy/auto_complete_trigger`
6. On trigger: calls auto-complete, transitions to `next_state`
7. Publishes current state to `/shared_autonomy/current_state` for debugging

---

## 6. New ROS Messages and Services

### `srv/TeleportObject.srv`
```
string object_name
float64 x
float64 y
float64 z
float64 qw
float64 qx
float64 qy
float64 qz
---
bool success
string message
```

### `msg/ValidGoal.msg`
```
string goal_id
string action_type    # pick, place, pour
geometry_msgs/Point target_position
string object_name    # which object is involved
```

### `msg/ValidGoals.msg`
```
std_msgs/Header header
string current_state
ValidGoal[] goals
```

### `msg/AutoCompleteTrigger.msg`
```
string goal_id
float64 confidence
```

---

## 7. Files Summary

### New files to create

| File | Purpose |
|------|---------|
| `src/shared_autonomy/simulated_joystick_node.py` | Velocity control with Gaussian noise |
| `src/shared_autonomy/task_state_machine_node.py` | State machine orchestrator |
| `src/shared_autonomy/task_auto_complete_node.py` | Object teleportation + auto-completion |
| `config/tasks/full_breakfast_sa.yaml` | State machine task config |
| `srv/TeleportObject.srv` | Teleport service definition |
| `msg/ValidGoal.msg` | Single valid goal message |
| `msg/ValidGoals.msg` | Set of valid goals message |
| `msg/AutoCompleteTrigger.msg` | Auto-completion trigger message |
| `launch/shared_autonomy.launch` | Launch file for entire stack |

### Existing files to modify

| File | Changes |
|------|---------|
| `src/mujoco_sim/mujoco_visualizer.py` | Add `set_object_pose()` and `get_jacobian()` methods |
| `src/mujoco_sim/simulation_server.py` | Add `/sim/teleport_object` service |
| `src/shared_autonomy/intent_inference_node.py` | Subscribe to `ValidGoals`, publish `AutoCompleteTrigger` |
| `CMakeLists.txt` | Add new msg/srv generation |
| `package.xml` | Add message generation dependencies |

---

## 8. Implementation Phases

### Phase 1 — Infrastructure
- Create `TeleportObject.srv` and custom messages
- Update `CMakeLists.txt` and `package.xml`
- Add `set_object_pose()` to `mujoco_visualizer.py`
- Add `/sim/teleport_object` service to `simulation_server.py`

### Phase 2 — Task YAML + State Machine
- Create `full_breakfast_sa.yaml` in the new state machine format
- Implement `task_state_machine_node.py`

### Phase 3 — Simulated Joystick
- Implement `simulated_joystick_node.py`
- Jacobian-based velocity control
- Gaussian noise injection
- Distance-based gripper triggering

### Phase 4 — Intent Inference
- Modify `intent_inference_node.py` to consume `ValidGoals`
- Publish `AutoCompleteTrigger` when confidence exceeds threshold

### Phase 5 — Auto-Completion
- Implement `task_auto_complete_node.py`
- Teleportation logic for place/pour
- Post-action EE repositioning

### Phase 6 — Integration
- Create `shared_autonomy.launch`
- End-to-end testing
- Parameter tuning (noise, threshold, beta)

---

## 9. Interaction Flow (Step by Step)

1. **task_state_machine** loads YAML, starts in `initial` state
2. **task_state_machine** queries object positions and publishes `ValidGoals`:
   - pick_cereal → cereal position [0.480, -0.250, 1.068]
   - pick_banana → banana position [0.500, 0.050, 0.940]
   - pick_milk → milk position [0.780, 0.300, 1.091]
3. **simulated_joystick** secretly picks "pick_cereal" as its intent
4. **simulated_joystick** moves EE toward cereal position with noise
5. **intent_inference** observes EE motion, computes P(pick_cereal), P(pick_banana), P(pick_milk)
6. As EE gets closer to cereal, P(pick_cereal) increases
7. When P(pick_cereal) > 0.8, **intent_inference** publishes AutoCompleteTrigger(goal_id="pick_cereal")
8. **task_auto_complete** closes gripper, lifts slightly
9. **task_state_machine** transitions to `holding_cereal` state
10. **task_state_machine** publishes new ValidGoals: pour_cereal, place_cereal
11. **simulated_joystick** picks "pour_cereal" as next intent
12. Cycle repeats...
