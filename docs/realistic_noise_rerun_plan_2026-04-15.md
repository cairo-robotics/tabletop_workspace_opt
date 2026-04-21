# Realistic Noise + Low-Rate Rerun Plan (2026-04-15)

## Context
The optimizer's slack formula was under-estimating endpoint variance
by ${\sim}14{-}15\times$ because (a) `traj_steps=30` was shorter than
the actual control-loop horizon and (b) `sigma_u=0.03` m/s combined
with `dt=0.05` s gives tiny per-step noise. At the same time, the SA
runtime used `control_rate=20` Hz which is unrealistic for a human
joystick user. This rerun aligns both sides to a more realistic
regime and restricts comparison to random vs. ME-optimized.

## Frozen parameters

### Optimizer (slack objective)
- `objective = trajectory_margin`
- `beta = 5.0`
- **`sigma_u = 0.05`** m/s (was 0.03)
- **`dt = 0.20`** s (was 0.05) — matches 5 Hz control
- **`traj_steps / n_steps = 150`** (was 30) — 30 s × 5 Hz horizon
- `alpha = 0.05` (Bonferroni per-pair)
- `lambda_R = 0.04`
- `sigma_v = 0.5`, `sigma_w = 0.5` (Gaussian observer, unused here)

### SA runtime (simulated user)
- `inference_model = path_efficiency`
- `beta = 5.0`
- **`noise = 0.05`** m/s (was 0.03) — joystick velocity std
- `threshold = 0.9`
- `lambda_R = 0.04`
- `max_speed = 0.05` m/s
- **`control_rate = 5.0`** Hz (was 20.0) — 200 ms per step
- `goal_timeout = 30.0` s
- `arrival_dist = 0.02` m
- `seed = 42`
- `intent_mode = se3-grasp`

### Evaluation scope
- Conditions: **random_yaw and me_optimized only** (DE skipped).
- Random: 10 layouts per tier, 30 orderings per layout.
- ME: 1 best-elite layout per tier, up to 120 orderings.
- Tiers: Easy, Med-A, Med-B, Med-C, Hard-A, Hard-B.

## Tasks

- [x] Update `src/envopt/yaw_optimizer.py` defaults (sigma_u=0.05, traj_steps=150).
- [x] Update `src/envopt/intent_separability_torch.py` slack defaults (sigma_u=0.05, dt=0.20, n_steps=150).
- [x] Update `scripts/compare_se3_sa_3d.py` defaults (noise=0.05, control_rate=5.0); add `--only-me` flag.
- [x] Update `scripts/plot_results_figures.py` to drop DE from figures.
- [x] Re-run ME optimization for all 6 scenes.
- [x] Re-run SA evaluation: random_yaw all tiers, me_optimized all tiers.
- [x] Aggregate random parts.
- [x] Regenerate figures.
- [x] Update latex `tab:main_results` and prose.
- [x] Summarize results in this doc.

## Process layout

### Phase 1: ME optimization (2 concurrent)

| Proc | ME opt scenes | Est wall |
|------|---------------|---------:|
| X | breakfast_easy, desk, breakfast | ~45 min |
| Y | kitchen_prep, meal_assembly, cluttered | ~45 min |

### Phase 2: SA evaluation (5 concurrent, after Phase 1 completes)

| Proc | Work | Est wall |
|------|------|---------:|
| Z | ME all 6 tiers (sequential) + random Easy/Med-A/B/C | ~2 h |
| W | random Hard-A, 10 layouts | ~2.5 h |
| V | random Hard-B, layouts 0-4 | ~2 h |
| U | random Hard-B, layouts 5-9 | ~2 h |

Total wall time ≈ 3.5 h.

## Results

_Filled in as processes complete._

### Per-process status
| Proc | Phase | Status | Launched | Completed | Notes |
|------|-------|--------|----------|-----------|-------|
| X    | 1     | done    | 2026-04-15 | 2026-04-15 | slacks: Easy=6.79, Med-A=3.52, Med-B=3.53 |
| Y    | 1     | done    | 2026-04-15 | 2026-04-15 | slacks: Med-C=4.44, Hard-A=0.15, Hard-B=0.00 |
| Z    | 2     | done    | 2026-04-15 | 2026-04-15 | ME all 6 tiers + random Easy/Med-A/B/C. ME Easy/Med 100%, Hard-A 15%, Hard-B 10% argmax. |
| W    | 2     | done    | 2026-04-15 | 2026-04-15 | Hard-A random 10 layouts, argmax mean 54% |
| V    | 2     | done    | 2026-04-15 | 2026-04-15 | Hard-B random 0-4 argmax mean 16% |
| U    | 2     | done    | 2026-04-15 | 2026-04-15 | Hard-B random 5-9 argmax mean 13% |

### Final aggregate

All 6 processes completed without crashes.

### Optimizer slacks (trajectory_margin, new formula: sigma_u=0.05, dt=0.20, n_steps=150, alpha=0.05)

| Tier   | ME slack |
|--------|---------:|
| Easy   | 6.79     |
| Med-A  | 3.52     |
| Med-B  | 3.53     |
| Med-C  | 4.44     |
| Hard-A | 0.15     |
| Hard-B | 0.00     |

### SA results (5 Hz, noise=0.05, arrival_dist=0.02)

| Tier   | Cond   | task % | infeas % | pick time (s) | argmax |
|--------|--------|-------:|---------:|--------------:|-------:|
| Easy   | random |  90    |  5       |  7.03         |  77%   |
| Easy   | ME     | 100    |  0       |  5.75         | 100%   |
| Med-A  | random |  20    | 40       |  8.23         |  74%   |
| Med-A  | ME     | 100    |  0       |  4.00         | 100%   |
| Med-B  | random |  30    | 47       | 10.70         |  91%   |
| Med-B  | ME     | 100    |  0       |  7.31         | 100%   |
| Med-C  | random |  30    | 24       |  7.85         |  82%   |
| Med-C  | ME     | 100    |  0       |  4.97         | 100%   |
| Hard-A | random |  10    | 57       | 12.89         |  60%   |
| Hard-A | ME     | 100    |  0       | 11.78         | **15%**|
| Hard-B | random |  20    |  8       | 11.65         |  15%   |
| Hard-B | ME     | 100    |  0       | 18.49         | **10%**|

### Key findings

- **Easy through Med-C:** ME achieves 100% argmax and 100% task
  success; random degrades to 74–91% argmax and 20–30% task success
  (with 24–47% infeasibility). ME is also faster by 2–4 s on these
  tiers.
- **Hard-A / Hard-B collapse for ME:** slacks of 0.15 and 0.00
  indicate the optimizer could not find a layout with meaningful
  pairwise separation under the realistic noise model. SA argmax
  drops to 15% / 10%, actually below random on Hard-A (60%) — which
  benefits from infeasible picks being skipped (57% infeasibility),
  so its 60% argmax is over a smaller, easier subset. Hard-B random
  sees only 15% argmax with low infeasibility, consistent with
  layouts having no separability by construction.
- **Interpretation:** with the realistic noise model, the Hard tiers
  are essentially un-optimizable against the Boltzmann
  path-efficiency observer. The workspace is too constrained for
  the observer to disambiguate 5 or 8 goals under 50 mm/s joystick
  noise and 5 Hz control. The prior "100% Hard-B ME" result was
  under the under-specified slack formula; the realistic calibration
  now correctly predicts the regime change.
- **Pick time on Hard-B ME (18.49 s)** is 2× the random arm's
  11.65 s because ME runs near the full 30 s timeout trying to
  converge on a layout that the observer cannot disambiguate.

### Crashes
None. All 6 procs completed on first launch.
