#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize teleop-only and teleop+VLM shared-autonomy CSV logs."""

import argparse
import csv
import json
import os


def load_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_probs(text):
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def summarize_single(path, expected_object):
    rows = load_rows(path)
    if not rows:
        raise RuntimeError("empty csv: {}".format(path))
    final = rows[-1]

    teleop_wrong_bias = False
    if expected_object:
        for row in rows:
            teleop = parse_probs(row.get("teleop_object_probs_json", ""))
            if not teleop:
                continue
            expected_prob = float(teleop.get(expected_object, 0.0))
            best_other = max(
                (float(prob) for obj, prob in teleop.items() if obj != expected_object),
                default=0.0,
            )
            if best_other > expected_prob:
                teleop_wrong_bias = True
                break

    return {
        "csv_path": path,
        "rows": len(rows),
        "mode": final.get("fusion_mode", ""),
        "expected_object": expected_object or "",
        "final_selected_object": final.get("selected_object", ""),
        "final_selected_grasp": final.get("selected_grasp", ""),
        "matches_instruction": str(bool(expected_object) and final.get("selected_object", "") == expected_object).lower(),
        "teleop_wrong_bias_observed": str(teleop_wrong_bias).lower(),
    }


def summarize_pair(baseline_csv, fusion_csv, expected_object):
    baseline = summarize_single(baseline_csv, expected_object)
    fusion = summarize_single(fusion_csv, expected_object)
    return {
        "expected_object": expected_object or "",
        "baseline_csv": baseline_csv,
        "fusion_csv": fusion_csv,
        "baseline_selected_object": baseline["final_selected_object"],
        "fusion_selected_object": fusion["final_selected_object"],
        "baseline_matches_instruction": baseline["matches_instruction"],
        "fusion_matches_instruction": fusion["matches_instruction"],
        "baseline_motion_bias": baseline["teleop_wrong_bias_observed"],
        "fusion_corrected": str(
            bool(expected_object)
            and baseline["final_selected_object"] != expected_object
            and fusion["final_selected_object"] == expected_object
        ).lower(),
    }


def write_summary_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize teleop-only vs teleop+VLM intent fusion logs.")
    parser.add_argument("--baseline-csv", default="", help="CSV log for teleop-only run")
    parser.add_argument("--fusion-csv", default="", help="CSV log for teleop+VLM run")
    parser.add_argument("--csv", default="", help="Single CSV log to summarize")
    parser.add_argument("--expected-object", default="", help="Expected canonical object, e.g. whole_milk")
    parser.add_argument("--output-csv", default="", help="Optional output summary CSV")
    args = parser.parse_args()

    rows = []
    if args.baseline_csv and args.fusion_csv:
        rows.append(summarize_pair(args.baseline_csv, args.fusion_csv, args.expected_object))
    elif args.csv:
        rows.append(summarize_single(args.csv, args.expected_object))
    else:
        raise SystemExit("provide either --csv or both --baseline-csv and --fusion-csv")

    for row in rows:
        for key in sorted(row.keys()):
            print("{}: {}".format(key, row[key]))
        print("")

    if args.output_csv:
        output_path = os.path.expanduser(args.output_csv)
        write_summary_csv(output_path, rows)
        print("wrote summary csv: {}".format(output_path))


if __name__ == "__main__":
    main()
