# Shared Autonomy: Running Experiments

This document covers how to run the headless shared autonomy (SA)
experiments, from single-task evaluation through full multi-tier
comparisons, workspace optimization, and figure generation.

This is the canonical runbook for the current SE(3) random-vs-optimized
shared-autonomy experiments. For ROS, MoveIt, MuJoCo GUI launch, and manual
grasp auditing, use `docs/simulation_runbook.md`.

## Prerequisites

- Python 3.8+ with numpy, scipy, torch, mujoco, yaml, matplotlib.
- MuJoCo scene XMLs in `src/assets/`.
- Grasp library at `config/grasp_poses_3d.yaml` (required for SE(3) modes).
- Task configs in `config/tasks/`.
- Scene layout configs in `config/scenes/`.

No ROS required for headless experiments. The headless runner
(`run_sa_headless.py`) uses MuJoCo directly with a Jacobian-based
velocity controller.

---

## Repository Entry Points

The six SE(3) tiers are defined in `src/experiments/se3_catalog.py`. Scripts
should import that catalog instead of redefining scene/task/object lists.

Reusable experiment/runtime code lives under `src/`:

- `src/experiments/se3_catalog.py`: canonical tier/task/object definitions.
- `src/experiments/provenance.py`: schema-v2 metadata and compatibility checks.
- `src/experiments/sa_metrics.py`: aggregate metrics for random layout arms.
- `src/envopt/layout_sampling.py`: random 2D and footprint-aware SE(3) layout samplers.
- `src/shared_autonomy/headless_core.py`: ROS-free inference/user-model primitives.
- `src/shared_autonomy/headless_state.py`: ROS-free task state machines.

`scripts/eval/run_sa_headless.py` still re-exports some helpers for older scripts,
but new code should import reusable pieces from the package modules above.

Before a full rerun, run a quick grasp audit:

```bash
python3 scripts/eval/audit_se3_grasps.py --out-dir /tmp/tabletop_grasp_audit
```

For visual/manual MoveIt inspection, see the manual audit workflow in
`docs/simulation_runbook.md`.

---

## 0. Current Baseline Parameters and Generated Results

This runbook records the current experiment parameters and commands. Generated
result JSON, figures, and paper drafts are local artifacts and are ignored by
Git.

**Baseline parameters (moderate-n_steps regime, as of 2026-04-27):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `sigma_u` | 0.03 m/s | Joystick velocity noise |
| `n_steps` | 75 | Expected pick horizon in control steps (~15 s) |
| `dt` | 0.20 s | Control timestep (= 1/control_rate) |
| `control_rate` | 5.0 Hz | Control loop frequency |
| `arrival_dist` | 0.02 m | Arrival-based termination distance |
| `beta` | 5.0 | Boltzmann rationality |
| `lambda_R` | 0.04 | Rotation weight for SE(3) observer |
| `threshold` | 0.9 | Posterior commit threshold |

Defaults are hardcoded in `src/envopt/yaw_optimizer.py` and the SA
runners. Do NOT change without updating both the optimizer and the
runtime; mismatches will break slack predictions.

**Generated results paths:**

- `results/se3_map_elites/{scene}.json` — ME archives (per-scene)
- `results/sa_headless/se3_3d_random_vs_optimized.json` — main paper table
- `results/sa_headless/se3_threshold_sweep.json` — threshold sensitivity

Legacy result files (`results/map_elites_v2_*.json`, `map_elites_v3_*.json`,
`sa_benchmark.json`, `map_elites_tier_*.json`, etc.) are from earlier
iterations. Do not use them for current comparisons; they used different
parameters, older intent modes, or pre-yaw-fix optimizations.

---

## 1. Single-Task Headless SA

Run one SA task with a simulated user (straight-line + Gaussian noise
joystick) and a Boltzmann path-efficiency observer.

```bash
python3 scripts/eval/run_sa_headless.py config/tasks/breakfast_easy_pick_and_return_sa.yaml \
  --intent-mode se3-grasp \
  --grasp-library config/grasp_poses_3d.yaml \
  --inference-model path_efficiency \
  --beta 5.0 --noise 0.03 --threshold 0.9 \
  --max-speed 0.05 --control-rate 5.0 \
  --seed 42
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `task_config` (positional) | required | Path to task YAML |
| `--intent-mode` | `2d-center` | One of: `2d-center`, `3d-center`, `3d-grasp-pos`, `se3-grasp` |
| `--inference-model` | `path_efficiency` | `gaussian` or `path_efficiency` |
| `--grasp-library` | None | Path to `grasp_poses_3d.yaml` (required for 3D/SE3 modes) |
| `--noise` | 0.03 | Simulated joystick velocity noise sigma (m/s) |
| `--beta` | 5.0 | Boltzmann rationality parameter |
| `--threshold` | 0.9 | Posterior probability threshold for commit |
| `--max-speed` | 0.05 | Max EE velocity (m/s) |
| `--control-rate` | 5.0 | Control loop frequency (Hz) |
| `--lambda-R` | 0.04 | Rotation weight for SE(3) observer |
| `--sigma-v` / `--sigma-w` | 0.5 | Gaussian observer translation/rotation sigma |
| `--scene` | None | Override scene name (uses task config's scene by default) |
| `--user-sequence` | None | Comma-separated pick order (e.g., `cereal,banana,milk_carton`) |
| `--seed` | 42 | Random seed |

### Output

Prints per-pick results: inferred goal, confidence, time, correct/wrong.
Returns a dict with `step_times` (per-pick details) and summary stats.

### Specifying a pick ordering

```bash
python3 scripts/eval/run_sa_headless.py config/tasks/desk_pick_and_return_sa.yaml \
  --intent-mode se3-grasp \
  --grasp-library config/grasp_poses_3d.yaml \
  --user-sequence mug,stapler,pen_cup
```

---

## 2. Workspace Optimization

Optimize object layout positions and yaw angles to maximize the
trajectory-margin slack (Boltzmann path-efficiency separability).

### 2a. MAP-Elites (ME) optimization

Produces an archive of diverse feasible layouts. Best-elite layout
is saved to `config/scenes/{scene}_se3_me_optimized.yaml`.

```bash
# All scenes
python3 scripts/optimize/optimize_se3_map_elites.py

# Single scene
python3 scripts/optimize/optimize_se3_map_elites.py --only scene_desk

# Quick smoke test (fewer iterations)
python3 scripts/optimize/optimize_se3_map_elites.py --only scene_breakfast_easy --quick
```

Results saved to `results/se3_map_elites/{scene}.json`.

### Optimizer parameters

The optimizer calls `optimize_yaw()` internally. The key parameters
controlling the slack formula are set as defaults in:

- `src/envopt/yaw_optimizer.py`: `sigma_u`, `traj_steps`
- `src/envopt/intent_separability_torch.py`: `sigma_u`, `dt`, `n_steps`

Current defaults (moderate-n_steps regime):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `sigma_u` | 0.03 m/s | Assumed joystick noise (should match SA runtime) |
| `dt` | 0.20 s | Control timestep (= 1/control_rate = 1/5 Hz) |
| `n_steps` | 75 | Expected pick horizon in control steps (~15 s) |
| `alpha` | 0.05 | Bonferroni significance level |
| `beta` | 5.0 | Boltzmann rationality |

---

## 3. Full Experiment: Random vs. ME

The main comparison script evaluates random layouts against the SE(3)
MAP-Elites layout across six difficulty tiers (Easy, Med-A/B/C, Hard-A,
Hard-B).

### 3a. Full run (all conditions, all tiers)

```bash
python3 scripts/eval/compare_se3_sa_3d.py \
  --n-random 10 --random-max-orderings 30 \
  --noise 0.03 --control-rate 5.0 --seed 42
```

This takes many hours. See Section 4 for parallelization.

### 3b. Run only optimized layouts (skip random)

```bash
python3 scripts/eval/compare_se3_sa_3d.py --skip-random --tiers Easy Med-A
```

### 3c. Run only ME (skip random)

```bash
python3 scripts/eval/compare_se3_sa_3d.py --only-me --tiers Hard-A Hard-B
```

### 3d. Run only random (worker mode for parallelization)

```bash
python3 scripts/eval/compare_se3_sa_3d.py \
  --only-random --tiers Hard-B \
  --n-random 10 --random-max-orderings 30 \
  --random-layout-offset 0 --random-layout-count 5
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-random` | 30 | Number of random layouts per tier |
| `--random-max-orderings` | 120 | Max orderings per random layout |
| `--noise` | 0.03 | Joystick noise sigma |
| `--control-rate` | 5.0 | Control frequency (Hz) |
| `--threshold` | 0.9 | Commit threshold |
| `--arrival-dist` | 0.02 | Arrival termination distance (m) |
| `--seed` | 42 | Random seed |
| `--tiers` | all | Subset of tiers to run |
| `--skip-random` | false | Skip random arm; merge only ME |
| `--only-random` | false | Run only random; write to parts dir |
| `--only-me` | false | Run only ME layout |
| `--random-layout-offset` | 0 | Start index for random layout slice |
| `--random-layout-count` | -1 | Number of layouts in slice (-1=all) |
| `--random-parts-dir` | `/tmp/random_parts` | Output dir for worker mode |

### Output

Results merge into
`results/sa_headless/se3_3d_random_vs_optimized.json` with keys per
tier: `random_yaw`, `random_yaw_optimized`, and `me_optimized`.

---

## 4. Parallelized Experiment Runs

The random arm is the bottleneck (10+ layouts x 30+ orderings x
N picks). Parallelize by splitting random layouts across workers.

### Step 1: Run ME (fast, one process)

```bash
python3 scripts/eval/compare_se3_sa_3d.py --skip-random
```

### Step 2: Run random in parallel workers

```bash
mkdir -p /tmp/random_parts

# Easy + Med tiers (fast, one worker)
python3 scripts/eval/compare_se3_sa_3d.py \
  --only-random --tiers Easy Med-A Med-B Med-C \
  --n-random 10 --random-max-orderings 30 &

# Hard-A (one worker)
python3 scripts/eval/compare_se3_sa_3d.py \
  --only-random --tiers Hard-A \
  --n-random 10 --random-max-orderings 30 &

# Hard-B split across 2 workers
python3 scripts/eval/compare_se3_sa_3d.py \
  --only-random --tiers Hard-B \
  --n-random 10 --random-max-orderings 30 \
  --random-layout-offset 0 --random-layout-count 5 &

python3 scripts/eval/compare_se3_sa_3d.py \
  --only-random --tiers Hard-B \
  --n-random 10 --random-max-orderings 30 \
  --random-layout-offset 5 --random-layout-count 5 &

wait
```

### Step 3: Aggregate random results

```bash
python3 scripts/eval/aggregate_random_parts.py
```

This reads all `random_part_*.json` files from `/tmp/random_parts/`,
aggregates per-layout results by tier, and merges into the canonical
`results/sa_headless/se3_3d_random_vs_optimized.json`.

---

## 5. Threshold Sweep

Evaluate intent inference across multiple decision thresholds
(tau = 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95) using bypass-threshold
mode. Records first-crossing step for each tau in a single run.

```bash
python3 scripts/eval/sweep_threshold_se3.py \
  --n-random 10 --beta 5.0 --noise 0.03 --seed 42
```

Results saved to `results/sa_headless/se3_threshold_sweep.json`.

---

## 6. Generating Figures

After results are computed, regenerate all figures:

```bash
python3 scripts/render/plot_results_figures.py
```

Reads from:
- `results/sa_headless/se3_3d_random_vs_optimized.json`
- `results/sa_headless/se3_me_sa_results.json`
- `results/sa_headless/se3_threshold_sweep.json`

Writes figures under `results/figures/` or another local output directory:
- `fig_argmax_accuracy.{png,pdf}` — argmax accuracy bars (random/ME per tier)
- `fig_task_and_time.{png,pdf}` — task success + pick time (2-panel)
- `fig_threshold_heatmaps.{png,pdf}` — (tier x tau) heatmaps for ME
- `fig_pareto.{png,pdf}` — accuracy vs. time Pareto scatter

---

## 7. Real Robot Pipeline (ROS)

Requires ROS Noetic, MoveIt, and the full `sim_moveit.launch` stack.

### Launch the simulation + planning stack

```bash
roslaunch tabletop_workspace_opt sim_moveit.launch
```

### Run shared autonomy with the real observer

```bash
roslaunch tabletop_workspace_opt shared_autonomy.launch \
  task_config:=config/tasks/breakfast_easy_pick_and_return_sa.yaml \
  intent_mode:=se3-grasp \
  grasp_library:=config/grasp_poses_3d.yaml \
  inference_model:=path_efficiency \
  beta:=5.0 noise:=0.03 threshold:=0.9 \
  control_rate:=5.0 lambda_R:=0.04 \
  scene:=scene_breakfast_easy
```

Key launch arguments mirror the headless runner. The launch file
starts `shared_autonomy_runner.py` which connects to MoveIt for
motion execution and subscribes to `/joy` for real joystick input.

---

## 8. Tiers and Scenes

| Tier   | Scene               | M (objects) | Task config |
|--------|---------------------|:-----------:|-------------|
| Easy   | scene_breakfast_easy| 2           | breakfast_easy_pick_and_return_sa.yaml |
| Med-A  | scene_desk          | 3           | desk_pick_and_return_sa.yaml |
| Med-B  | scene_breakfast     | 3           | breakfast_pick_and_return_sa.yaml |
| Med-C  | scene_kitchen_prep  | 3           | kitchen_pick_and_return_sa.yaml |
| Hard-A | scene_meal_assembly | 5           | meal_pick_and_return_sa.yaml |
| Hard-B | scene_cluttered     | 8           | cluttered_pick_and_return_sa.yaml |

### Object height variants (2026-07-01)

Several objects have been swapped from tall to short collision-box
variants to avoid EE-object collisions during joystick approach
(commit 163201e). Base scene XMLs and their corresponding
`config/scenes/*.yaml` + `*_se3_me_optimized.yaml` files have been
updated in place:

| Scene | Object | Height (tall → short) |
|-------|--------|-----------------------|
| scene_breakfast | cereal | 30.6 → 8 cm |
| scene_breakfast | milk_carton | 35.2 → 8 cm |
| scene_breakfast_easy | cereal | 30.6 → 8 cm |
| scene_meal_assembly | cereal | 30.6 → 8 cm |
| scene_meal_assembly | bottle | 19 → 8 cm |
| scene_kitchen_prep | bottle | 19 → 8 cm |
| scene_desk | phone | 15 cm upright → 1.5 cm flat |

**To restore tall variants:** uncomment the `<!-- ORIGINAL ... -->` body
blocks in each XML, revert the keyframe z coordinates, and restore the
old `half_extents` values in the YAMLs (originals are noted in inline
YAML comments and XML block comments).

**Known limitation of the current short-variant swap:**

1. **`_se3_me_optimized.yaml` positions were optimized against tall
   objects.** They were updated in place (dimensions + z) but x/y remain
   the tall-object optima. Re-running `optimize_se3_map_elites.py` with
   the short variants would produce different (and possibly better)
   layouts, but would also invalidate the paper's reported results.

---

## 9. Quick Reference: End-to-End Workflow

```bash
# 1. Optimize layouts (ME)
python3 scripts/optimize/optimize_se3_map_elites.py

# 2. Evaluate (parallelized)
python3 scripts/eval/compare_se3_sa_3d.py --skip-random  # ME
mkdir -p /tmp/random_parts
python3 scripts/eval/compare_se3_sa_3d.py --only-random --n-random 10 --random-max-orderings 30
python3 scripts/eval/aggregate_random_parts.py

# 3. Threshold sweep
python3 scripts/eval/sweep_threshold_se3.py --n-random 10

# 4. Generate figures
python3 scripts/render/plot_results_figures.py

# 5. Check results
cat results/sa_headless/se3_3d_random_vs_optimized.json | python3 -m json.tool | head -40
ls results/figures/fig_*.png
```
