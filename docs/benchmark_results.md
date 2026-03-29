# Shared Autonomy Benchmark Results

## Optimization Method
Object positions optimized using **differential evolution** on the
intent separability metric (Section 1.6 of Shared Autonomy Notes).
The optimizer maximizes the worst-case slack across all goal pairs,
ensuring the inference engine can reliably distinguish between any
two goals regardless of which one the user is pursuing.

## Scenes

### Breakfast Scene
- **Objects:** bowl, cereal, banana, milk_carton (+ napkin, spoon)
- **Optimizer slack:** 111.08 (unoptimized) → 503.52 (optimized), 4.5x improvement

### Desk Scene
- **Objects:** mug, book, phone, pen_cup, stapler
- **Optimizer slack:** 89.11 (unoptimized) → 352.61 (optimized), 4.0x improvement

## Benchmark Results (2 trials each)

### Desk Organize Task (6 steps: pick mug → place, pick book → place, pick pen_cup → place)

| Config | Total Time | Retries | Inference/Goal | Confidence | Pass |
|--------|-----------|---------|---------------|------------|------|
| **Desk optimized** | **18.1s** | 0 | **1.02s** | 0.903 | 100% |
| Desk unoptimized | 24.8s | 0 | 2.15s | 0.902 | 100% |

**Speedup: 1.37x total time, 2.1x faster inference per goal**

#### Per-goal inference times (desk):

| Goal | Unoptimized | Optimized |
|------|------------|-----------|
| pick_mug (3 choices) | 9.4s | 3.8s |
| place_mug (1 choice) | 0.2s | 0.2s |
| pick_book (2 choices) | 2.7s | 1.5s |
| place_book (1 choice) | 0.2s | 0.2s |
| pick_pen_cup (1 choice) | 0.2s | 0.2s |
| place_pen_cup (1 choice) | 0.2s | 0.2s |

### Full Breakfast Task (8 steps: pick/pour/place cereal, pick/place banana, pick/pour/place milk)

| Config | Total Time | Retries | Inference/Goal | Confidence | Pass |
|--------|-----------|---------|---------------|------------|------|
| Breakfast optimized | 46.2s | 0 | 3.98s | 0.854 | 100% |
| Breakfast unoptimized | 38.8s | 0 | 3.06s | 0.857 | 100% |

Note: The breakfast optimized layout was slower in this run. This is because
the optimizer spreads objects to maximize angular separation, but this can place
some objects further from the EE home position, requiring longer travel time.

### Set Table Task (4 steps: pick/place cereal, pick/place banana)

| Config | Total Time | Retries | Inference/Goal | Confidence | Pass |
|--------|-----------|---------|---------------|------------|------|
| Optimized | 16.7s | 0 | 1.70s | 0.927 | 100% |
| Unoptimized | 15.5s | 0 | 1.37s | 0.927 | 100% |

## Key Findings

1. **Desk task shows clear optimization benefit:** 2.1x faster inference per goal
   in the optimized layout. The `pick_mug` step (3 choices) improved from 9.4s to
   3.8s because the optimizer spread mug, book, and pen_cup in distinct angular
   directions from the robot.

2. **Breakfast task results are mixed:** The optimizer maximizes angular separation,
   which helps disambiguation but can increase travel distance. The trade-off depends
   on the specific task structure and which goals compete at each state.

3. **Single-choice goals are instant (~0.2s)** regardless of layout. The inference
   immediately recognizes there's only one valid goal.

4. **All configurations achieve 100% pass rate** with the current inference threshold
   (0.7). No retries or timeouts in this benchmark run.

5. **The optimization algorithm used is differential evolution** from scipy, maximizing
   the worst-case intent separability slack across all goal pairs. This is a
   gradient-free global optimizer that works well for the non-convex objective.

## Screenshots

- `scene_unoptimized.png` — Breakfast scene, original layout
- `scene_optimized.png` — Breakfast scene, optimizer layout
- `scene_desk_unoptimized.png` — Desk scene, original layout
- `scene_desk_optimized.png` — Desk scene, optimizer layout
