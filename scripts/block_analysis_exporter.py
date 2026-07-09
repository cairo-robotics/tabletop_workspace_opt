#!/usr/bin/env python3
"""Export user study trial logs to a block-level analysis CSV."""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict


DEFAULT_COLUMNS = [
    "session_id",
    "participant_id",
    "condition_id",
    "block_id",
    "condition_order",
    "task_type",
    "target_selection_source",
    "destination_inference_mode",
    "autonomy_mode",
    "n_trials",
    "n_pickup_trials",
    "n_destination_trials",
    "n_interrupted_after_commit",
    "n_interrupted_during_execution",
    "n_failed_after_commit",
    "n_failed_without_commit",
    "success_rate_all",
    "success_rate_pickup",
    "success_rate_destination",
    "destination_correct_inference_rate",
    "mean_duration_sec",
    "mean_duration_pickup_sec",
    "mean_duration_destination_sec",
    "mean_time_to_commit_sec",
    "mean_time_to_commit_pickup_sec",
    "mean_time_to_commit_destination_sec",
    "mean_autonomous_time_sec",
    "mean_autonomous_time_pickup_sec",
    "mean_autonomous_time_destination_sec",
    "mean_teleop_time_sec",
    "mean_teleop_time_pickup_sec",
    "mean_teleop_time_destination_sec",
    "mean_teleop_distance_proportion",
    "mean_teleop_distance_proportion_pickup",
    "mean_teleop_distance_proportion_destination",
    "mean_autonomous_distance_proportion",
    "mean_autonomous_distance_proportion_pickup",
    "mean_autonomous_distance_proportion_destination",
    "mean_avg_teleop_entropy",
    "mean_avg_teleop_entropy_pickup",
    "mean_avg_teleop_entropy_destination",
    "mean_confirmation_count",
    "mean_top_goal_switch_count",
    "mean_cancel_count",
    "mean_timeout_count",
]


def _latest_trial_log(log_dir):
    candidates = sorted(glob.glob(os.path.join(log_dir, "user_study_trials_*.jsonl")))
    return candidates[-1] if candidates else ""


def _default_output_path(input_path):
    root, _ = os.path.splitext(input_path)
    return root + "_block_analysis.csv"


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


def _step_role(step_id):
    value = str(step_id or "").strip().lower()
    if "destination" in value or value.startswith("place_"):
        return "destination"
    if (
        "brick" in value
        or "pick" in value
        or "object" in value
        or value.startswith("select_")
        or "item" in value
    ):
        return "pickup"
    return "other"


def _trial_has_meaningful_activity(trial):
    if bool(trial.get("success")):
        return True
    if int(trial.get("confirmation_count", 0) or 0) > 0:
        return True
    if int(trial.get("cancel_count", 0) or 0) > 0:
        return True
    if int(trial.get("timeout_count", 0) or 0) > 0:
        return True
    if int(trial.get("intent_locked_count", 0) or 0) > 0:
        return True
    if int(trial.get("auto_stalled_count", 0) or 0) > 0:
        return True
    if float(trial.get("autonomous_time_sec", 0.0) or 0.0) > 0.0:
        return True
    if trial.get("time_to_commit_sec") is not None:
        return True
    events = list(trial.get("events") or [])
    meaningful_events = {
        "confirm_prompt",
        "confirm_accept",
        "confirm_cancel",
        "confirm_timeout",
        "auto_start",
        "auto_complete",
        "auto_stalled",
        "grasp_complete",
        "release_complete",
        "manual_advance",
        "quick_rescan",
        "send_home",
    }
    for event in events:
        if str(event.get("event") or "").strip() in meaningful_events:
            return True
    return False


def _include_in_analysis(trial):
    failure_reason = str(trial.get("failure_reason") or "").strip()
    if failure_reason == "node_shutdown" and not _trial_has_meaningful_activity(trial):
        return False
    return True


def _analysis_outcome(trial):
    if bool(trial.get("success")):
        return "success"
    failure_reason = str(trial.get("failure_reason") or "").strip()
    committed = trial.get("time_to_commit_sec") is not None
    if failure_reason == "node_shutdown":
        if _trial_has_meaningful_activity(trial):
            if committed:
                return "interrupted_after_commit"
            return "interrupted_during_execution"
        return "aborted_inactive"
    if committed:
        return "failed_after_commit"
    return "failed_without_commit"


def _safe_mean(values):
    if not values:
        return ""
    return sum(values) / float(len(values))


def _safe_rate(numerator, denominator):
    if denominator <= 0:
        return ""
    return float(numerator) / float(denominator)


def _float_or_none(value):
    if value in (None, ""):
        return None
    return float(value)


def _stringify(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _unique_or_mixed(values):
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not cleaned:
        return ""
    unique = []
    for value in cleaned:
        if value not in unique:
            unique.append(value)
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _group_key(trial):
    return (
        str(trial.get("session_id") or "").strip(),
        str(trial.get("participant_id") or "").strip(),
        str(trial.get("condition_id") or "").strip(),
        str(trial.get("task_id") or "").strip(),
    )


def _build_block_row(trials, autonomy_mode):
    pickup_trials = [trial for trial in trials if _step_role(trial.get("step_id")) == "pickup"]
    destination_trials = [trial for trial in trials if _step_role(trial.get("step_id")) == "destination"]
    outcomes = [_analysis_outcome(trial) for trial in trials]

    all_duration = [_float_or_none(trial.get("duration_sec")) for trial in trials]
    pickup_duration = [_float_or_none(trial.get("duration_sec")) for trial in pickup_trials]
    destination_duration = [_float_or_none(trial.get("duration_sec")) for trial in destination_trials]

    all_commit = [_float_or_none(trial.get("time_to_commit_sec")) for trial in trials]
    pickup_commit = [_float_or_none(trial.get("time_to_commit_sec")) for trial in pickup_trials]
    destination_commit = [_float_or_none(trial.get("time_to_commit_sec")) for trial in destination_trials]

    auto_time = [_float_or_none(trial.get("autonomous_time_sec")) for trial in trials]
    pickup_auto_time = [_float_or_none(trial.get("autonomous_time_sec")) for trial in pickup_trials]
    destination_auto_time = [_float_or_none(trial.get("autonomous_time_sec")) for trial in destination_trials]
    teleop_time = [_float_or_none(trial.get("teleop_time_sec")) for trial in trials]
    pickup_teleop_time = [_float_or_none(trial.get("teleop_time_sec")) for trial in pickup_trials]
    destination_teleop_time = [_float_or_none(trial.get("teleop_time_sec")) for trial in destination_trials]
    teleop_distance_prop = [_float_or_none(trial.get("teleop_distance_proportion")) for trial in trials]
    pickup_teleop_distance_prop = [_float_or_none(trial.get("teleop_distance_proportion")) for trial in pickup_trials]
    destination_teleop_distance_prop = [_float_or_none(trial.get("teleop_distance_proportion")) for trial in destination_trials]
    autonomous_distance_prop = []
    pickup_autonomous_distance_prop = []
    destination_autonomous_distance_prop = []
    for trial in trials:
        value = _float_or_none(trial.get("teleop_distance_proportion"))
        autonomous_distance_prop.append(None if value is None else max(0.0, 1.0 - value))
    for trial in pickup_trials:
        value = _float_or_none(trial.get("teleop_distance_proportion"))
        pickup_autonomous_distance_prop.append(None if value is None else max(0.0, 1.0 - value))
    for trial in destination_trials:
        value = _float_or_none(trial.get("teleop_distance_proportion"))
        destination_autonomous_distance_prop.append(None if value is None else max(0.0, 1.0 - value))
    avg_teleop_entropy = [_float_or_none(trial.get("avg_teleop_entropy")) for trial in trials]
    pickup_avg_teleop_entropy = [_float_or_none(trial.get("avg_teleop_entropy")) for trial in pickup_trials]
    destination_avg_teleop_entropy = [_float_or_none(trial.get("avg_teleop_entropy")) for trial in destination_trials]
    confirmations = [_float_or_none(trial.get("confirmation_count")) for trial in trials]
    switches = [_float_or_none(trial.get("top_goal_switch_count")) for trial in trials]
    cancels = [_float_or_none(trial.get("cancel_count")) for trial in trials]
    timeouts = [_float_or_none(trial.get("timeout_count")) for trial in trials]

    def present(values):
        return [value for value in values if value is not None]

    row = {
        "session_id": str(trials[0].get("session_id") or "").strip(),
        "participant_id": str(trials[0].get("participant_id") or "").strip(),
        "condition_id": str(trials[0].get("condition_id") or "").strip(),
        "block_id": str(trials[0].get("block_id") or "").strip(),
        "condition_order": "",
        "task_type": _unique_or_mixed(trial.get("task_id") for trial in trials),
        "target_selection_source": _unique_or_mixed(trial.get("target_source") for trial in pickup_trials),
        "destination_inference_mode": _unique_or_mixed(trial.get("target_source") for trial in destination_trials),
        "autonomy_mode": autonomy_mode,
        "n_trials": len(trials),
        "n_pickup_trials": len(pickup_trials),
        "n_destination_trials": len(destination_trials),
        "n_interrupted_after_commit": sum(1 for outcome in outcomes if outcome == "interrupted_after_commit"),
        "n_interrupted_during_execution": sum(
            1 for outcome in outcomes if outcome == "interrupted_during_execution"
        ),
        "n_failed_after_commit": sum(1 for outcome in outcomes if outcome == "failed_after_commit"),
        "n_failed_without_commit": sum(1 for outcome in outcomes if outcome == "failed_without_commit"),
        "success_rate_all": _safe_rate(sum(1 for trial in trials if bool(trial.get("success"))), len(trials)),
        "success_rate_pickup": _safe_rate(
            sum(1 for trial in pickup_trials if bool(trial.get("success"))), len(pickup_trials)
        ),
        "success_rate_destination": _safe_rate(
            sum(1 for trial in destination_trials if bool(trial.get("success"))), len(destination_trials)
        ),
        "destination_correct_inference_rate": _safe_rate(
            sum(1 for trial in destination_trials if bool(trial.get("correct_inference"))),
            len(destination_trials),
        ),
        "mean_duration_sec": _safe_mean(present(all_duration)),
        "mean_duration_pickup_sec": _safe_mean(present(pickup_duration)),
        "mean_duration_destination_sec": _safe_mean(present(destination_duration)),
        "mean_time_to_commit_sec": _safe_mean(present(all_commit)),
        "mean_time_to_commit_pickup_sec": _safe_mean(present(pickup_commit)),
        "mean_time_to_commit_destination_sec": _safe_mean(present(destination_commit)),
        "mean_autonomous_time_sec": _safe_mean(present(auto_time)),
        "mean_autonomous_time_pickup_sec": _safe_mean(present(pickup_auto_time)),
        "mean_autonomous_time_destination_sec": _safe_mean(present(destination_auto_time)),
        "mean_teleop_time_sec": _safe_mean(present(teleop_time)),
        "mean_teleop_time_pickup_sec": _safe_mean(present(pickup_teleop_time)),
        "mean_teleop_time_destination_sec": _safe_mean(present(destination_teleop_time)),
        "mean_teleop_distance_proportion": _safe_mean(present(teleop_distance_prop)),
        "mean_teleop_distance_proportion_pickup": _safe_mean(present(pickup_teleop_distance_prop)),
        "mean_teleop_distance_proportion_destination": _safe_mean(present(destination_teleop_distance_prop)),
        "mean_autonomous_distance_proportion": _safe_mean(present(autonomous_distance_prop)),
        "mean_autonomous_distance_proportion_pickup": _safe_mean(present(pickup_autonomous_distance_prop)),
        "mean_autonomous_distance_proportion_destination": _safe_mean(present(destination_autonomous_distance_prop)),
        "mean_avg_teleop_entropy": _safe_mean(present(avg_teleop_entropy)),
        "mean_avg_teleop_entropy_pickup": _safe_mean(present(pickup_avg_teleop_entropy)),
        "mean_avg_teleop_entropy_destination": _safe_mean(present(destination_avg_teleop_entropy)),
        "mean_confirmation_count": _safe_mean(present(confirmations)),
        "mean_top_goal_switch_count": _safe_mean(present(switches)),
        "mean_cancel_count": _safe_mean(present(cancels)),
        "mean_timeout_count": _safe_mean(present(timeouts)),
    }
    return {column: _stringify(row.get(column)) for column in DEFAULT_COLUMNS}


def export_blocks(input_path, output_path, include_all=False, autonomy_mode="shared_autonomy"):
    trials = _load_trials(input_path)
    if not include_all:
        trials = [trial for trial in trials if _include_in_analysis(trial)]

    grouped = defaultdict(list)
    for trial in trials:
        grouped[_group_key(trial)].append(trial)

    rows = []
    for key in sorted(grouped.keys()):
        group_trials = sorted(
            grouped[key],
            key=lambda trial: (
                int(trial.get("trial_index_within_block") or 0),
                str(trial.get("start_time_iso") or ""),
            ),
        )
        rows.append(_build_block_row(group_trials, autonomy_mode=autonomy_mode))

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


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
        "--output",
        default="",
        help="Destination CSV path. Defaults to <input>_block_analysis.csv",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include administrative or incomplete rows that are normally excluded from block analysis export.",
    )
    parser.add_argument(
        "--autonomy-mode",
        default="shared_autonomy",
        help="Label to store in the autonomy_mode column.",
    )
    args = parser.parse_args()

    input_path = args.path or _latest_trial_log(os.path.expanduser(args.log_dir))
    if not input_path:
        print("No user study trial log found.", file=sys.stderr)
        return 2
    if not os.path.exists(input_path):
        print("Log file does not exist: {}".format(input_path), file=sys.stderr)
        return 2

    output_path = os.path.expanduser(args.output) if args.output else _default_output_path(input_path)
    try:
        row_count = export_blocks(
            input_path,
            output_path,
            include_all=args.include_all,
            autonomy_mode=args.autonomy_mode,
        )
    except Exception as exc:
        print("Failed to export block analysis CSV: {}".format(exc), file=sys.stderr)
        return 1

    print("Exported {} block rows to {}".format(row_count, output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
