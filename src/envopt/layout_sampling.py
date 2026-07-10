"""Random layout samplers used by SA and optimizer baselines."""
from __future__ import annotations

import numpy as np


TABLE_BOUNDS_X = (0.30, 0.85)
TABLE_BOUNDS_Y = (-0.45, 0.45)
MIN_DIST = 0.12
ROBOT_EXCLUSION_RADIUS = 0.15
START_POS_2D = np.array([0.452, 0.160])


def generate_random_se3_layouts(n_layouts, movable_names, fixed_positions,
                                half_extents, seed=0, fixed_yaws=None,
                                footprint_margin=0.01,
                                min_separation=0.02):
    """Generate deterministic, footprint-valid random ``(x, y, yaw)`` layouts.

    Each layout index uses an independent seed sequence, so a layout is
    identical whether generated in one process or consumed by split workers.
    Layout validity covers rotated footprints, fixed objects, table edges, and
    the robot exclusion region. Grasp feasibility is intentionally not used
    for rejection because it is an experiment outcome.
    """
    from envopt.grasp_feasibility import _rotated_rect
    from shapely.geometry import Point, box

    fixed_yaws = fixed_yaws or {}
    table = box(TABLE_BOUNDS_X[0], TABLE_BOUNDS_Y[0],
                TABLE_BOUNDS_X[1], TABLE_BOUNDS_Y[1])
    robot_exclusion = Point(*START_POS_2D).buffer(ROBOT_EXCLUSION_RADIUS)
    fixed_footprints = []
    for name, pos in fixed_positions.items():
        he = half_extents.get(name)
        if he is not None:
            fixed_footprints.append(_rotated_rect(
                np.asarray(pos)[:2], he, fixed_yaws.get(name, 0.0)))

    layouts = []
    for layout_idx in range(n_layouts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, layout_idx]))
        accepted = None
        for _ in range(200):
            positions = {}
            yaws = {}
            placed = list(fixed_footprints)
            valid = True
            for name in movable_names:
                he = half_extents.get(name)
                if he is None:
                    raise ValueError(f"Missing half_extents for {name}")
                found = False
                for _ in range(500):
                    yaw = float(rng.uniform(-np.pi, np.pi))
                    pos = np.array([
                        rng.uniform(TABLE_BOUNDS_X[0], TABLE_BOUNDS_X[1]),
                        rng.uniform(TABLE_BOUNDS_Y[0], TABLE_BOUNDS_Y[1]),
                    ])
                    footprint = _rotated_rect(pos, he, yaw)
                    if not table.covers(footprint):
                        continue
                    if footprint.intersects(robot_exclusion):
                        continue
                    if any(footprint.distance(other) <
                           min_separation + footprint_margin
                           for other in placed):
                        continue
                    positions[name] = pos
                    yaws[name] = yaw
                    placed.append(footprint)
                    found = True
                    break
                if not found:
                    valid = False
                    break
            if valid:
                accepted = (positions, yaws)
                break
        if accepted is None:
            raise RuntimeError(
                f"Could not generate valid random layout index {layout_idx}")
        layouts.append(accepted)
    return layouts


def generate_random_layouts(n_layouts, movable_names, fixed_positions,
                            seed=0, fixed_half_extents=None,
                            movable_half_extents=None,
                            footprint_margin=0.01):
    """Generate collision-free random 2D layouts avoiding robot position."""
    rng = np.random.default_rng(seed)
    layouts = []
    fx_he = fixed_half_extents or {}
    mv_he = movable_half_extents or {}

    def _aabb_overlap(pa, ha, pb, hb):
        return (abs(pa[0] - pb[0]) < (ha[0] + hb[0] + footprint_margin)
                and abs(pa[1] - pb[1]) < (ha[1] + hb[1] + footprint_margin))

    for _ in range(n_layouts * 20):
        if len(layouts) >= n_layouts:
            break
        placed = [(n, np.asarray(p[:2]), fx_he.get(n))
                  for n, p in fixed_positions.items()]
        positions = {}
        valid = True

        for name in movable_names:
            success = False
            mh = mv_he.get(name)
            for _ in range(200):
                x = rng.uniform(TABLE_BOUNDS_X[0] + 0.05,
                                TABLE_BOUNDS_X[1] - 0.05)
                y = rng.uniform(TABLE_BOUNDS_Y[0] + 0.05,
                                TABLE_BOUNDS_Y[1] - 0.05)
                pos = np.array([x, y])

                if np.linalg.norm(pos - START_POS_2D) < ROBOT_EXCLUSION_RADIUS:
                    continue

                clash = False
                for _oname, opos, ohe in placed:
                    if ohe is not None and mh is not None:
                        if _aabb_overlap(pos, mh, opos, ohe):
                            clash = True
                            break
                    else:
                        if np.linalg.norm(pos - opos) < MIN_DIST:
                            clash = True
                            break
                if clash:
                    continue

                positions[name] = pos
                placed.append((name, pos, mh))
                success = True
                break

            if not success:
                valid = False
                break

        if valid:
            layouts.append(positions)

    return layouts[:n_layouts]
