#!/usr/bin/env python3
"""Merge trajectory_margin easy_med and hard results into one file."""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

files = [
    "map_elites_trajectory_margin_easy_med.json",
    "map_elites_trajectory_margin_hard.json",
]

merged = {}
for fname in files:
    path = os.path.join(PROJECT_ROOT, "results", fname)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        merged.update(data)
        print(f"Loaded {fname}: {list(data.keys())}")
    else:
        print(f"Missing: {fname}")

out_path = os.path.join(PROJECT_ROOT, "results",
                       "map_elites_trajectory_margin_all.json")
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)
print(f"\nMerged {len(merged)} environments -> {out_path}")
