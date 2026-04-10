# Trajectory-Margin Workspace Optimization: Evaluation Results

**Experiment date:** 2026-04-10
**Author:** generated from `scripts/run_sa_all_orderings_headless.py`
**Raw results:** `results/sa_headless/random_vs_optimized.json`
**Plan reference:** `docs/unified_inference_plan.md`

## Setup

- **Optimizer:** MAP-Elites with trajectory-margin objective (linearized
  Boltzmann-Gaussian slack), run per `scripts/eval_map_elites_tiers.py`.
- **Evaluator:** Headless shared autonomy with Boltzmann path-efficiency
  observer (matching the optimizer's observer model).
- **Inference model:** path_efficiency, β=5.0, threshold=0.9.
- **Simulator dynamics:** 2D EE motion (`motion_2d=True`), Jacobian-IK control
  with joystick noise σ=0.03 m/s, max speed 0.05 m/s, 20 Hz control rate,
  30 s goal timeout.
- **Path-length accumulation:** skip steps below 1 mm; windowed stall
  termination (loop breaks after 20 steps with net displacement < 5 mm).
- **Random baselines:** 30 layouts per environment, generated with object
  min-distance 12 cm, robot-exclusion radius 15 cm, table bounds
  [0.30, 0.85] × [−0.45, 0.45].
- **Orderings:** all permutations of pick objects, capped at 120 for
  environments with more than 5! permutations.
- **Environments:** 6 tiers from 2-object (Easy) to 8-object (Hard-B),
  defined in `scripts/eval_map_elites_tiers.py::ENVIRONMENTS`.

## Headline metrics

Two accuracy metrics are reported:

- **Threshold accuracy**: fraction of picks where the Boltzmann posterior
  crossed P ≥ 0.9 on the *correct* goal before stall/timeout.
- **Argmax accuracy**: fraction of picks where the highest-probability goal
  at termination (threshold cross, stall, or timeout) was the true goal.
  This is threshold-independent.

Threshold accuracy tests a decision rule; argmax accuracy tests whether the
optimizer produced a layout where the observer's cost function identifies
the correct goal at all. The slack guarantee in the unified plan is about
**argmax accuracy**.

## Results

| Tier   | M | Layout    | Pick Time (s) | Thresh Acc   | Argmax Acc   | Speedup |
|--------|---|-----------|---------------|--------------|--------------|---------|
| Easy   | 2 | random    | 5.0 ± 2.4     | 73% ± 25%    | 87% ± 22%    | —       |
|        |   | optimized | **1.7 ± 0.1** | **100%**     | **100%**     | **2.9x**|
| Med-A  | 3 | random    | 5.8 ± 2.3     | 64% ± 23%    | 84% ± 18%    | —       |
|        |   | optimized | **2.0 ± 0.1** | **100%**     | **100%**     | **3.0x**|
| Med-B  | 3 | random    | 7.6 ± 1.8     | 52% ± 19%    | 77% ± 21%    | —       |
|        |   | optimized | **1.4 ± 0.1** | **100%**     | **100%**     | **5.5x**|
| Med-C  | 3 | random    | 6.1 ± 2.2     | 67% ± 24%    | 86% ± 16%    | —       |
|        |   | optimized | **1.6 ± 0.1** | **100%**     | **100%**     | **3.9x**|
| Hard-A | 5 | random    | 7.3 ± 1.1     | 46% ± 14%    | 72% ± 15%    | —       |
|        |   | optimized | **5.1 ± 0.6** | **100%**     | **100%**     | **1.4x**|
| Hard-B | 8 | random    | 8.7 ± 0.8     | 18% ± 9%     | 59% ± 8%     | —       |
|        |   | optimized | 9.0 ± 1.5     | 12%          | **100%**     | 1.0x    |

## Threshold sensitivity

Convergence rate = fraction of picks where the true goal's posterior
reached at least τ at some point during the trajectory. Computed post hoc
from the recorded `max_prob_true` per pick.

| Tier   | Layout    | τ=0.8  | τ=0.85 | τ=0.9  | τ=0.95 |
|--------|-----------|--------|--------|--------|--------|
| Easy   | random    |  80%   |  74%   |  73%   |   0%   |
|        | optimized | 100%   | 100%   | 100%   |   0%   |
| Med-A  | random    |  75%   |  70%   |  64%   |   0%   |
|        | optimized | 100%   | 100%   | 100%   |   0%   |
| Med-B  | random    |  62%   |  57%   |  52%   |   0%   |
|        | optimized | 100%   | 100%   | 100%   |   0%   |
| Med-C  | random    |  75%   |  70%   |  67%   |   0%   |
|        | optimized | 100%   | 100%   | 100%   |   0%   |
| Hard-A | random    |  57%   |  54%   |  46%   |   0%   |
|        | optimized | 100%   | 100%   | 100%   |   0%   |
| Hard-B | random    |  29%   |  24%   |  18%   |   0%   |
|        | optimized |  25%   |  25%   |  12%   |   0%   |

## Takeaways

1. **The trajectory-margin optimizer produces layouts that achieve 100%
   argmax accuracy on every environment, including Hard-B.** This matches
   the slack guarantee from `docs/unified_inference_plan.md` (≥ 95% under
   α = 0.05). Per-pair slack for the optimized Hard-B layout was verified
   positive (worst pair: 1.08) before running the simulator.

2. **Layout optimization gives 1.0x–5.5x speedup over random baselines**,
   with the largest gains on 3-object environments (Med-B 5.5x) and the
   smallest on Hard-A (1.4x) and Hard-B (1.0x — no time speedup because
   both stall at similar times). Accuracy gains are large across every
   tier: random argmax is 59%–87% vs 100% for optimized.

3. **Random baselines now deserve comparison.** After the stall-termination
   fix, random layouts on Easy through Hard-A reach 72%–87% argmax
   accuracy. Before the fix they were near 10%–50% because every run
   loitered into a corrupt posterior.

4. **Three orthogonal failure modes were identified and each has a fix.**
   The paper can present these as separate concerns:
   - **Optimizer-simulator mismatch** (2D optimizer vs 3D-with-Jacobian
     simulator): resolved by constraining the EE to 2D motion
     (`motion_2d=True`). On Hard-B optimized this raised argmax accuracy
     from 50% to 75%.
   - **Loitering corruption** (path length grew while the EE wandered in
     place after arrival, biasing the cost numerator toward far-from-start
     goals): resolved by stall termination (break when 20-step windowed
     net displacement < 5 mm). On Hard-B optimized this raised argmax
     accuracy from 75% to 100%.
   - **Boltzmann saturation** (for M alternatives at similar d_start, the
     softmax posterior cannot concentrate above 1 / (1 + (M−1)·exp(−β·ε)),
     which is ~0.91 for M=8, β=5 even with perfect cost ordering):
     decoupled from argmax accuracy. Only affects threshold-based
     decisions. Fix would be either argmax-with-margin commitment or
     higher β (which amplifies noise sensitivity mid-trajectory).

5. **Hard-B threshold accuracy stays at 12% even with 100% argmax
   accuracy.** This is not an optimization failure — it is the Boltzmann
   saturation. The posterior on the true goal physically cannot exceed
   ~0.91 at β=5 when there are 8 alternatives packed on the table. For
   that tier, a threshold-based decision rule is the wrong abstraction.

6. **Verifying slack on the layout is a good sanity check.** For Hard-B
   optimized we computed all 56 pairwise slacks — every one was positive,
   with worst 1.08. This matched the empirical 100% argmax accuracy after
   the simulator fixes. When we saw the earlier 50% accuracy, the
   theory-vs-practice gap was fully explained by simulator artifacts, not
   by a flawed linearization.

7. **Pure 2D Monte Carlo confirmed the theory independently of the
   simulator.** Running 200 noisy straight-line trajectories per goal on
   the optimized Hard-B layout gave 100% argmax accuracy across all 8
   goals, corroborating that the slack condition holds empirically.

## Open questions for future work

- **Does stall termination help the optimizer see the same objective the
  evaluator measures?** Currently the optimizer's `trajectory_margin_objective`
  assumes a fixed-length trajectory; the evaluator uses early termination.
  The optimizer's variance estimate may be miscalibrated under the new
  simulator rules. A Monte Carlo margin in the optimizer (instead of the
  linearized one) would remove this mismatch entirely.
- **Does an argmax-with-margin decision rule (commit when
  P(g₁) − P(g₂) ≥ δ) close the Hard-B threshold-accuracy gap without
  raising β?** The saturation issue is threshold-specific; a relative
  rule should sidestep it.
- **What happens with higher β (say 10 or 15) under the fixed simulator?**
  Earlier β=10 raised Hard-B argmax from 12% to 25% but only in the old
  (broken) simulator. With the fix, argmax is already 100%, so higher β
  would only affect threshold accuracy — worth measuring once.

## Supporting scripts and files

- Optimizer: `scripts/eval_map_elites_tiers.py`
- Simulator: `scripts/run_sa_headless.py`
- Evaluation harness: `scripts/run_sa_all_orderings_headless.py`
- Slack computation: `scripts/eval_map_elites_tiers.py::trajectory_margin_objective`
- Comparison helper: `scripts/compare_objectives.py`
- Optimized layouts:
  - `config/scenes/scene_*_trajectory_optimized.yaml`
  - `src/assets/scene_*_trajectory_optimized.xml`
- Raw MAP-Elites results: `results/map_elites_trajectory_margin_all.json`
- Raw SA evaluation results: `results/sa_headless/random_vs_optimized.json`
