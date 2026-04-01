# Intent Separability Optimization & New Scene Infrastructure

## Overview

This document summarizes two bodies of work:

1. **Intent Separability Optimization** — Implementation of the workspace design
   objective from the Shared Autonomy Notes (Section 1.6), which optimizes
   object placements on the table to maximize how quickly a robot can identify
   the user's intended goal from noisy joystick inputs.

2. **New MuJoCo Scenes & Dynamic Scene Configuration** — Two new tabletop
   scenes (desk, kitchen prep), a YAML-based scene config system that replaces
   hardcoded object definitions, task configs for multi-step evaluation, and a
   PartNet Mobility download/conversion script.

---

## 1. Intent Separability Optimization

### Motivation

In shared autonomy, a user controls the robot with a noisy joystick and the
robot must infer which object the user intends to reach. The workspace layout
(where objects are placed) directly affects how quickly this inference can be
made. Objects placed in similar directions from the end-effector starting
position are harder to distinguish than objects spread apart angularly.

### Mathematical Formulation (Section 1.6 of Shared Autonomy Notes)

The objective maximizes the worst-case "intent separability slack":

```
max_θ  min_{g≠g*}  [ m̃(g;θ) - √Ṽ(g;θ) · Φ⁻¹(1-α_g) ]
```

Where:
- **θ** = object (x,y) positions on the table (what we optimize)
- **g\*** = true user goal; **g** = each alternative goal
- **Ṽ(g;θ)** = cumulative Mahalanobis separation between predicted mean
  commands for g vs g\* over T timesteps
- **m̃(g;θ)** = ½Ṽ(g;θ) - log(p(g)/p(g\*)) — the separation margin
- **Φ⁻¹** = standard normal quantile function
- **α_g** = per-goal failure probability budget (union bound)

The goal-conditioned policy is a simple "point toward goal":
`μ_t(g) = (goal - x_t) / ||goal - x_t||`

### New Files

| File | Purpose |
|------|---------|
| `src/envopt/intent_separability.py` | Core math: policy, trajectory sim, separation, margin, slack, MAP inference simulation, multi-step task evaluation |
| `src/envopt/workspace_optimizer.py` | Optimization (DE + MAP-Elites), single/multi-step evaluation, method comparison |
| `tests/test_intent_separability.py` | 43 tests covering math, properties, optimization, and goal prediction speed |

### Optimization Methods

Two global optimizers are supported via the `method` parameter:

- **`differential_evolution`** (default) — scipy's global optimizer. Finds
  higher-slack solutions (503.5 vs 432.3) but slower (348s vs 256s).
- **`map_elites`** — Quality-Diversity optimization via pyribs (CMA-ME).
  Faster and produces an archive of 400+ diverse high-quality layouts.

### Results: Single-Step Goal Identification

| Method | Slack | Accuracy | Steps to ID | Speedup |
|--------|-------|----------|-------------|---------|
| Original (un-optimized) | 111.1 | 100% | 6.6 | 1.0x |
| Differential Evolution | 503.5 | 100% | 1.1 | 6.0x |
| MAP-Elites (CMA-ME) | 432.3 | 100% | 1.1 | 6.0x |

Both methods achieve ~6x faster goal identification. The optimizer spreads
objects into roughly the four quadrants of the reachable table, maximizing
angular separation from the EE start position.

### Results: Multi-Step Tasks

Longer-horizon tasks were created to test intent identification across
sequential pick/place steps:

| Task | Steps | Original (steps) | DE (steps) | ME (steps) | Best Speedup |
|------|-------|-------------------|------------|------------|-------------|
| banana_in_bowl | 2 | 133.9 | 3.1 | 2.4 | 55x |
| full_breakfast | 6 | 250.5 | 92.6 | 150.9 | 2.7x |
| tidy_table | 8 | 467.5 | 375.0 | 162.8 | 2.9x |
| sort_by_size | 10 | 507.0 | 191.5 | 322.0 | 2.6x |

Key findings:
- Both methods improve all tasks over baseline
- Place steps (reaching derived positions like "left of bowl") are harder
  to identify than pick steps (reaching known object positions)
- DE and ME find different layouts that trade off differently across tasks
- Accuracy improves from 82-91% to 93-100% alongside speed gains

### Usage

```python
from workspace_optimizer import optimize_workspace, compare_methods

# Single method
result = optimize_workspace(method="differential_evolution", maxiter=200)
result = optimize_workspace(method="map_elites", maxiter=200)

# Full comparison with multi-step tasks
compare_methods(task_dir="config/tasks/")
```

---

## 2. New MuJoCo Scenes

### Scene: `scene_desk.xml` — Office Desk

Five primitive-shape objects (no mesh dependencies):

| Object | Visual Shape | Color | Collision (half-extents) | Position |
|--------|-------------|-------|-------------------------|----------|
| mug | cylinder r=35mm h=110mm | white | box 28×28×55mm | (0.50, 0.20) |
| book | box 150×100×30mm | blue | box 75×50×15mm | (0.65, -0.15) |
| phone | box 70×15×150mm | black | box 35×7.5×75mm | (0.45, -0.25) |
| pen_cup | cylinder r=30mm h=100mm | dark red | box 25×25×50mm | (0.75, 0.30) |
| stapler | box 50×40×30mm | dark gray | box 25×20×15mm | (0.55, -0.35) |

### Scene: `scene_kitchen_prep.xml` — Kitchen Prep

| Object | Visual Shape | Color | Collision (half-extents) | Position |
|--------|-------------|-------|-------------------------|----------|
| cutting_board | box 240×160×20mm | wood brown | box 120×80×10mm | (0.60, 0.00) |
| apple | sphere r=38mm | red | box 28×28×28mm | (0.50, 0.15) |
| can | cylinder r=33mm h=110mm | red | box 27×27×55mm | (0.45, -0.20) |
| bottle | cylinder r=30mm h=190mm | green | box 25×25×95mm | (0.75, 0.25) |
| sponge | box 100×50×40mm | yellow | box 50×25×20mm | (0.70, -0.20) |

All collision boxes have their narrowest graspable dimension < 59mm (Sawyer
gripper gap). Visual geoms are `contype=0` (no collision); invisible collision
geoms are `contype=1`.

### Design Conventions

- Same table geometry as `scene_breakfast.xml` (pos 0.6,0,0.915; 90° yaw)
- Same robot (sawyer-gripper.xml include), lighting, and visual settings
- Object z = table_top (0.942) + object half-height
- High friction (8.0) on all graspable objects
- Keyframe includes all object initial poses

---

## 3. Scene Configuration System

### Problem

The simulation server (`simulation_server.py`) previously hardcoded object
names, sizes, and table parameters for the breakfast scene only. Adding new
scenes required code changes.

### Solution

Scene configs are now YAML files in `config/scenes/`:

```yaml
# config/scenes/scene_desk.yaml
scene:
  name: scene_desk
  table:
    pos_world: [0.6, 0.0, 0.915]
    half_extents: [0.4, 0.7, 0.027]
  objects:
    - short_name: mug
      mujoco_name: mug
      half_extents: [0.028, 0.028, 0.055]
    - short_name: book
      ...
```

The simulation server loads the config matching the `scene_name` ROS parameter:
```python
scene_config_path = os.path.join(package_path, 'config', 'scenes', f'{scene_name}.yaml')
```

Falls back to hardcoded breakfast defaults if no YAML exists.

### Files

| File | Purpose |
|------|---------|
| `config/scenes/scene_breakfast.yaml` | Breakfast scene object config |
| `config/scenes/scene_desk.yaml` | Desk scene object config |
| `config/scenes/scene_kitchen_prep.yaml` | Kitchen prep scene object config |

---

## 4. Task Configs

### New Multi-Step Tasks (Breakfast Scene)

| Task | Steps | Description |
|------|-------|-------------|
| `full_breakfast.yaml` | 6 | Pick/place cereal, banana, and milk around bowl |
| `tidy_table.yaml` | 8 | Gather all items near bowl, rearrange banana |
| `sort_by_size.yaml` | 10 | Arrange items by size, reposition twice |

### New Scene Tasks

| Task | Scene | Steps | Description |
|------|-------|-------|-------------|
| `desk_organize.yaml` | desk | 6 | Phone→book, mug→pen_cup, stapler→book |
| `kitchen_prep_full.yaml` | kitchen_prep | 10 | Apple→board, group near bottle, rearrange |

### Launching

```bash
# Desk scene
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk

# Kitchen prep scene
roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_kitchen_prep
```

---

## 5. PartNet Mobility Integration

Script to download articulated objects from the PartNet Mobility dataset
(2,347 objects across 47 categories) and convert their URDFs to MuJoCo XML.

### Location

`scripts/partnet/download_and_convert.py`

### Usage

```bash
# Register at https://sapien.ucsd.edu for an access token
export SAPIEN_ACCESS_TOKEN="your_token"

# Browse available categories (47 total)
python3 scripts/partnet/download_and_convert.py --list-categories

# List objects in a category
python3 scripts/partnet/download_and_convert.py --list-objects Microwave

# Download and convert an object
python3 scripts/partnet/download_and_convert.py --id 7119 --category Microwave

# Inspect an already-downloaded object
python3 scripts/partnet/download_and_convert.py --id 7119 --skip-download --inspect-only
```

### What It Does

1. Downloads the object zip from SAPIEN API
2. Extracts to `src/assets/partnet_mobility/{object_id}/`
3. Inspects the folder structure (URDF, meshes, semantics, bounding box)
4. Converts `mobility.urdf` → MuJoCo XML using MuJoCo's built-in URDF compiler
5. Prints a scene include snippet for integrating into a scene XML

---

## 6. Modified Files

| File | Change |
|------|--------|
| `src/mujoco_sim/simulation_server.py` | Load object config from YAML with hardcoded fallback |
| `config/grasp_poses.yaml` | Added top-down grasps for 10 new objects (desk + kitchen) |

---

## 7. Test Suite

43 tests in `tests/test_intent_separability.py`:

| Category | Count | What's Tested |
|----------|-------|---------------|
| Core math | 8 | Policy direction, unit vectors, trajectory simulation |
| Pairwise separation | 4 | Zero/positive separation, step/noise scaling |
| Margin & slack | 4 | Prior effects, positive/negative slack |
| Objective properties | 4 | Well-separated > close, orthogonal > collinear |
| Overlap penalty | 2 | Detection correctness |
| DE optimization | 3 | Slack improves, bounds, no overlap |
| MAP-Elites optimization | 4 | Slack improves, bounds, no overlap, invalid method |
| Single-step prediction (DE) | 2 | Faster + accurate |
| Single-step prediction (ME) | 2 | Faster + accurate |
| Task parsing | 5 | All task YAMLs parse correctly |
| Multi-step inference | 2 | Correct structure, fast identification |
| Multi-step optimization | 3 | full_breakfast, tidy_table, sort_by_size faster |

Run with:
```bash
python3 -m pytest tests/test_intent_separability.py -v
```
