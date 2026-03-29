# Shared Autonomy Implementation — Summary

## Changes Made

### New Files Created

| File | Description |
|------|-------------|
| `src/shared_autonomy/shared_autonomy_runner.py` | Main shared autonomy node combining joystick simulation, intent inference, state machine, and auto-completion |
| `config/tasks/full_breakfast_sa.yaml` | State-machine task config for shared autonomy |
| `msg/ValidGoal.msg` | ROS message for a single valid goal |
| `msg/ValidGoals.msg` | ROS message for set of valid goals |
| `msg/AutoCompleteTrigger.msg` | ROS message for auto-completion trigger |
| `srv/TeleportObject.srv` | ROS service for teleporting objects in MuJoCo |
| `launch/shared_autonomy.launch` | Launch file for shared autonomy stack |
| `docs/shared_autonomy_plan.md` | Detailed implementation plan |

### Existing Files Modified

| File | Changes |
|------|---------|
| `src/mujoco_sim/mujoco_visualizer.py` | Added `set_object_pose()` for object teleportation, `get_jacobian()` for EE Jacobian computation |
| `src/mujoco_sim/simulation_server.py` | Added `/sim/teleport_object` ROS service, imported `TeleportObject` service type |
| `CMakeLists.txt` | Added 3 new messages (`ValidGoal`, `ValidGoals`, `AutoCompleteTrigger`) and 1 new service (`TeleportObject`) |

### System Components

1. **Simulated Joystick Controller** — Generates velocity commands toward a target goal with configurable Gaussian noise. Uses MuJoCo Jacobian (`mj_jacSite`) for Cartesian-to-joint velocity conversion. Publishes incremental joint position targets to the existing PID controller.

2. **Intent Inference Engine** — Bayesian inference using path efficiency: `P(g) = softmax(-β * (L_obs + d(EE,g)) / d(start,g))` with a proximity bonus for goals within 10cm. Triggers auto-completion when confidence exceeds threshold.

3. **Task State Machine** — Loads state-machine YAML config. Each state has valid goals with `next_state` transitions. Computes target positions by resolving object references from MuJoCo detections.

4. **Auto-Completion** — Bypasses motion planning entirely:
   - Pick: closes gripper at current position
   - Place: opens gripper, teleports object to destination
   - Pour: teleports object to pour position for 2s, then continues

### Hyperparameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| Noise sigma | `--noise` | 0.01 | Gaussian noise std dev on velocity (m/s) |
| Threshold | `--threshold` | 0.8 | Intent confidence threshold for auto-completion |
| Beta | `--beta` | 5.0 | Rationality coefficient (higher = more confident) |
| Max speed | `--max-speed` | 0.05 | Maximum EE linear speed (m/s) |
| Control rate | `--control-rate` | 20 | Control loop frequency (Hz) |

---

## Tests Performed

### Test 1: Initial Run (full_breakfast_sa.yaml)
**Parameters:** noise=0.005, threshold=0.7, beta=5.0, max_speed=0.03, rate=20Hz

**Result: PARTIAL PASS (8/8 steps completed, then infinite loop)**

Steps completed:
1. pick_cereal — inferred at p=0.704 ✓
2. pour_cereal — inferred at p=0.700 ✓
3. place_cereal — inferred at p=1.000 (only goal) ✓
4. pick_banana — inferred at p=0.709 ✓
5. place_banana — inferred at p=1.000 (only goal) ✓
6. pick_milk — inferred at p=0.704 ✓
7. pour_milk — inferred at p=0.710 ✓
8. place_milk — inferred at p=1.000 (only goal) ✓

**Issue found:** After completing all 8 steps, the `milk_done` state still had `pick_banana` and `pick_cereal` as valid goals, causing infinite looping. The EE also got stuck 1000mm from targets after teleportation moved it beyond reach.

**Fixes applied:**
- Added proximity bonus to inference: goals within 10cm get a strong boost
- Made `milk_done` a terminal state (empty valid_goals)
- Made `banana_done` only allow `pick_milk` (no backtracking)
- Removed `move_ee_to(home)` after place auto-complete

### Test 2: Fixed Run (terminal states)
**Parameters:** same as Test 1

**Result: PASS (8/8 steps, clean termination)**

All 8 steps completed in ~60 seconds:
1. pick_cereal — p=0.701, ~18s
2. pour_cereal — p=0.701, ~3s
3. place_cereal — p=1.000, instant
4. pick_banana — p=0.710, ~13s
5. place_banana — p=1.000, instant
6. pick_milk — p=1.000, instant (only goal)
7. pour_milk — p=0.701, ~16s
8. place_milk — p=1.000, instant

Final output:
```
SHARED AUTONOMY COMPLETE
  Steps completed: 8
  Final state: milk_done
```

### Test Summary

| Test | Steps Completed | Termination | Status |
|------|----------------|-------------|--------|
| Test 1 (initial) | 8/8 | Infinite loop | PARTIAL PASS |
| Test 2 (fixed) | 8/8 | Clean exit | PASS |

---

## Known Limitations

1. **No real gripper interaction:** The auto-complete for pick just closes the gripper — it doesn't verify the object is actually grasped in MuJoCo. The teleportation handles object placement regardless.

2. **EE position after teleport:** After place auto-completion, the EE stays at its current position. If the arm is at joint limits, subsequent velocity commands may be slow or ineffective. Could be improved by moving the EE to a neutral position between goals.

3. **Single-goal inference for single-goal states:** When only one goal is valid, inference instantly returns p=1.0 without the user needing to move. This is correct behavior but skips the joystick simulation for those steps.

4. **Fixed task order:** The state machine graph defines a relatively fixed order. True shared autonomy would allow more flexible goal selection, which the state machine supports but the test YAML constrains.

5. **No collision avoidance in joystick control:** The Jacobian-based velocity controller doesn't check for collisions. The arm could push through objects in MuJoCo. This is acceptable for the simulated user but would need fixing for real deployment.
