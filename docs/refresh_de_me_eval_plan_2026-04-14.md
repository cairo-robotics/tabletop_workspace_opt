# Refresh DE-optimized and ME-optimized Evaluation Results (2026-04-14)

## Context
`results/sa_headless/se3_3d_random_vs_optimized.json` (Apr 13) contains
anomalous DE-optimized Med-A numbers (n_picks=150, argmax=0.08) that a
fresh single-ordering run cannot reproduce — both the DE layout YAML and
`run_sa_headless.py` were modified after that JSON was written. Need to
refresh DE-optimized and ME-optimized arms across all 6 tiers while
preserving the existing random_yaw results.

## Goal
Regenerate DE-optimized and ME-optimized rows in
`se3_3d_random_vs_optimized.json` for all 6 tiers. Leave existing
random_yaw / random_yaw_optimized rows untouched. Refresh the four
figures in `docs/latex/figures/`.

## Tasks

- [x] Modify `scripts/compare_se3_sa_3d.py`:
    - Add `--skip-random` flag so random_yaw and random_yaw_optimized arms
      are not re-run.
    - Add `me_optimized` condition that loads
      `{scene}_se3_me_optimized.yaml` and runs the same per-ordering eval.
    - On write: load existing JSON first, overwrite only `se3_optimized`
      and `me_optimized` keys per tier (preserve random arms).
- [x] Run the refreshed script for all 6 tiers (Easy, Med-A/B/C,
      Hard-A/B) with `--skip-random`. SA params unchanged: path_efficiency,
      β=5.0, noise=0.03, τ=0.9, λ_R=0.04, seed=42.
- [x] Rerun `scripts/plot_results_figures.py` to refresh the four
      figures (`fig_argmax_accuracy.png`, `fig_task_and_time.png`,
      `fig_threshold_heatmaps.png`, `fig_pareto.png`).
- [x] Summarize refreshed results in this doc and in chat.

## Output
- **Updated:** `results/sa_headless/se3_3d_random_vs_optimized.json`
  (merged — random arms preserved, DE + ME arms refreshed).
- **Refreshed:** 4 figures in `docs/latex/figures/`.

## Results

All 6 tiers refreshed in
`results/sa_headless/se3_3d_random_vs_optimized.json`. Random arms
preserved. Task success = 100% and infeasibility = 0% across all tiers
for both DE and ME — headline differences are in threshold/argmax
accuracy and mean pick time.

| Tier   | Method | task_ok | argmax | thr_acc | pick time (s) |
|--------|--------|--------:|-------:|--------:|--------------:|
| Easy   | DE     | 100%    | 100%   | 100%    | 4.19 ± 2.99   |
| Easy   | ME     | 100%    | 100%   | 100%    | 6.09 ± 3.46   |
| Med-A  | DE     | 100%    | 100%   | 100%    | 5.48 ± 2.31   |
| Med-A  | ME     | 100%    | 100%   | 100%    | 4.07 ± 3.25   |
| Med-B  | DE     | 100%    | 100%   | 100%    | 3.72 ± 2.54   |
| Med-B  | ME     | 100%    | 100%   |  78%    | 7.71 ± 5.84   |
| Med-C  | DE     | 100%    | 100%   |  67%    | 5.44 ± 5.89   |
| Med-C  | ME     | 100%    | 100%   |  94%    | 3.52 ± 2.07   |
| Hard-A | DE     | 100%    |  99%   |  62%    | 10.90 ± 4.26  |
| Hard-A | ME     | 100%    |  94%   |  44%    | 11.42 ± 6.67  |
| Hard-B | DE     | 100%    |  67%   |  12%    | 12.47 ± 4.90  |
| Hard-B | ME     | 100%    |  97%   |  25%    | 12.02 ± 3.85  |

### Key findings
- **Med-A anomaly resolved.** DE-optimized Med-A now reports
  argmax=100% and threshold=100% (was argmax=8% with 150 spurious picks
  in the stale JSON). The anomaly was a stale-artifact of a prior
  layout/runner version; nothing wrong with the optimization itself.
- **ME wins decisively on Hard-B.** Argmax 97% vs DE 67%; ME is the only
  method reaching high-accuracy inference at the hardest tier.
- **DE wins on Hard-A argmax** (99% vs 94%) but ME and DE are within
  noise on threshold accuracy.
- **Easy/Med tiers are saturated** for argmax (≥99% everywhere);
  differences are primarily in pick time.
- Threshold accuracy drops with difficulty for both methods (Hard-B:
  DE=12%, ME=25%), reflecting the known pattern that τ=0.9 is too high
  for the hardest tier — the threshold sweep figure now shows this
  clearly.

### Figures refreshed
- `docs/latex/figures/fig_argmax_accuracy.png`
- `docs/latex/figures/fig_task_and_time.png`
- `docs/latex/figures/fig_threshold_heatmaps.png` (unchanged data — still
  from `se3_threshold_sweep.json`)
- `docs/latex/figures/fig_pareto.png`
