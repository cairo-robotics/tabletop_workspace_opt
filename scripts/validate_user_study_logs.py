#!/usr/bin/env python3
"""Validate user study trial logs for common data integrity issues."""

import argparse
import glob
import json
import os
import sys
from collections import Counter


def _latest_log(log_dir):
    candidates = sorted(glob.glob(os.path.join(log_dir, "user_study_trials_*.jsonl")))
    return candidates[-1] if candidates else ""


def _load_trials(path):
    trials = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise ValueError("line {} is not valid json: {}".format(line_no, exc))
            payload["_line_no"] = line_no
            trials.append(payload)
    return trials


def _report(problem_type, message, problems):
    problems.append("{}: {}".format(problem_type, message))


def validate_trials(trials, require_metadata):
    problems = []
    counts = Counter(str(trial.get("trial_id") or "") for trial in trials if str(trial.get("trial_id") or "").strip())
    duplicate_ids = sorted(trial_id for trial_id, count in counts.items() if count > 1)
    for trial_id in duplicate_ids:
        _report("duplicate_trial_id", trial_id, problems)

    for trial in trials:
        trial_id = str(trial.get("trial_id") or "<missing>")
        line_no = trial.get("_line_no")
        if trial.get("end_time_sec") is None:
            _report("missing_end_time", "{} (line {})".format(trial_id, line_no), problems)
        if bool(trial.get("success")) and not bool(trial.get("correct_inference")):
            _report("success_but_wrong_inference", "{} (line {})".format(trial_id, line_no), problems)
        if require_metadata:
            for field in ("session_id", "participant_id", "condition_id"):
                if not str(trial.get(field) or "").strip():
                    _report(
                        "missing_metadata",
                        "{} missing {} (line {})".format(trial_id, field, line_no),
                        problems,
                    )
        if trial.get("trial_index_within_block") in (None, ""):
            _report("missing_trial_index", "{} (line {})".format(trial_id, line_no), problems)
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Path to a user_study_trials_*.jsonl file")
    parser.add_argument(
        "--log-dir",
        default=os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "logs",
        ),
        help="Directory to search when path is omitted",
    )
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Skip session/participant/condition completeness checks",
    )
    args = parser.parse_args()

    path = args.path or _latest_log(os.path.expanduser(args.log_dir))
    if not path:
        print("No user study trial log found.", file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print("Log file does not exist: {}".format(path), file=sys.stderr)
        return 2

    trials = _load_trials(path)
    problems = validate_trials(trials, require_metadata=not args.allow_missing_metadata)

    print("Validated {} trial rows from {}".format(len(trials), path))
    if not problems:
        print("No integrity problems found.")
        return 0

    for problem in problems:
        print(problem)
    return 1


if __name__ == "__main__":
    sys.exit(main())
