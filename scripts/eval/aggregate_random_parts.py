#!/usr/bin/env python3
"""Aggregate per-worker random_part_{tier}_*.json files into the
canonical `se3_3d_random_vs_optimized.json`.

Each part file has raw per-layout results; we combine them across
workers for the same tier, then compute the aggregate `random_yaw`
entry using the same formula as `compare_se3_sa_3d.py`.

Existing `me_optimized` and `random_yaw_optimized` entries in the canonical
JSON are preserved.
"""
import os
import sys
import json
import glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.provenance import metadata_compatible
from experiments.sa_metrics import aggregate_layout_results

PARTS_DIR = os.path.join("/tmp/random_parts")
CANONICAL = os.path.join(
    ROOT, "results", "sa_headless", "se3_3d_random_vs_optimized.json")


def main():
    parts = sorted(glob.glob(os.path.join(PARTS_DIR, "random_part_*.json")))
    if not parts:
        print(f"No part files in {PARTS_DIR}")
        sys.exit(1)

    by_tier = defaultdict(list)   # tier -> list of (layout_idx, result)
    params_by_tier = {}
    reference_document = None
    for p in parts:
        with open(p) as f:
            d = json.load(f)
        if d.get("schema_version") != 2 or "experiment" not in d:
            raise RuntimeError(f"Legacy or unversioned part file: {p}")
        part_document = {"schema_version": 2, "experiment": d["experiment"]}
        if reference_document is None:
            reference_document = part_document
        elif not metadata_compatible(reference_document, part_document):
            raise RuntimeError(f"Incompatible provenance in part file: {p}")
        tier = d["tier"]
        params_by_tier.setdefault(tier, d.get("params", {}))
        for idx, res in zip(d["layout_indices"], d["random_yaw_results"]):
            by_tier[tier].append((idx, res))

    # Deduplicate by layout index (later workers override earlier)
    dedup = {}
    for tier, pairs in by_tier.items():
        seen = {}
        for idx, res in pairs:
            seen[idx] = res
        dedup[tier] = [seen[k] for k in sorted(seen.keys())]
        print(f"{tier}: {len(dedup[tier])} unique layouts across "
              f"{len(pairs)} part entries")

    # Load canonical JSON, update random_yaw per tier, save back
    if os.path.exists(CANONICAL):
        canonical = json.load(open(CANONICAL))
        if canonical.get("schema_version") != 2:
            raise RuntimeError(
                "Canonical result uses a legacy schema; archive it before "
                "aggregating new worker parts")
        if not metadata_compatible(canonical, reference_document):
            raise RuntimeError("Canonical result provenance differs from parts")
    else:
        canonical = {"schema_version": 2,
                     "experiment": reference_document["experiment"],
                     "tiers": {}}

    for tier, results in dedup.items():
        entry = canonical["tiers"].get(tier, {})
        entry["random_yaw"] = aggregate_layout_results(results)
        canonical["tiers"][tier] = entry

    tmp = CANONICAL + ".tmp"
    with open(tmp, "w") as f:
        json.dump(canonical, f, indent=2, default=str)
    os.replace(tmp, CANONICAL)
    print(f"\nUpdated {CANONICAL}")
    print("Random params by tier:")
    for t, p in params_by_tier.items():
        print(f"  {t}: {p}")


if __name__ == "__main__":
    main()
