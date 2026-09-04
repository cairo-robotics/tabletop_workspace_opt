#!/usr/bin/env python3
"""Export user study trial logs to a questionnaire-friendly analysis CSV."""

import argparse
import csv
import glob
import json
import os
import sys
import yaml


DEFAULT_COLUMNS = [
    "trial_id",
    "session_id",
    "participant_id",
    "condition_id",
    "layout_condition",
    "block_id",
    "trial_index_within_block",
    "task_id",
    "task_name",
    "step_id",
    "step_index",
    "step_title",
    "step_role",
    "analysis_focus",
    "command",
    "target_source",
    "target_completion_label",
    "target_completion_labels",
    "target_candidate_ids",
    "target_completion_category",
    "target_completion_categories",
    "start_time_iso",
    "end_time_iso",
    "duration_sec",
    "time_to_commit_sec",
    "time_to_first_correct_lock_sec",
    "time_teleop",
    "autonomous_time_sec",
    "teleop_time_sec",
    "teleop_distance_m",
    "autonomous_distance_m",
    "teleop_distance_proportion",
    "avg_teleop_entropy",
    "confirmation_count",
    "cancel_count",
    "timeout_count",
    "auto_stalled_count",
    "intent_locked_count",
    "top_goal_switch_count",
    "distribution_snapshot_count",
    "max_top_probability",
    "casper_prediction_count",
    "casper_confident_count",
    "casper_intent_switch_count",
    "casper_distinct_intents",
    "casper_mean_latency_sec",
    "casper_latest_candidate_id",
    "casper_latest_confidence",
    "casper_agreement_with_top_goal_count",
    "casper_agreement_with_top_goal_rate",
    "casper_final_matches_target",
    "casper_first_target_prediction_sec",
    "casper_first_confident_target_prediction_sec",
    "casper_first_stable_target_prediction_sec",
    "casper_wrong_confident_count",
    "casper_wrong_confident_rate",
    "top_goal_first_target_sec",
    "top_goal_first_stable_target_sec",
    "top_goal_wrong_switch_count",
    "top_goal_label_at_end",
    "top_probability_at_end",
    "selected_grasp_label_at_end",
    "final_inferred_goal",
    "final_inferred_object_name",
    "final_inferred_category",
    "analysis_outcome",
    "correct_inference",
    "success",
    "failure_reason",
]


def _default_object_map_path():
    return os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "config",
        "apriltag_object_map.yaml",
    )


def _load_label_to_candidate_id(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    entries = data.get("tag_objects") or data.get("candidate_objects") or {}
    mapping = {}
    for tag_id, meta in entries.items():
        if not isinstance(meta, dict):
            continue
        label = str(meta.get("grasp_complete_label") or "").strip()
        if label:
            mapping[label] = str(tag_id).strip()
    return mapping


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


def _analysis_focus(task_id, step_id):
    task_value = str(task_id or "").strip().lower()
    role = _step_role(step_id)
    if task_value == "lego_sorting":
        if role == "pickup":
            return "acquisition"
        if role == "destination":
            return "intent_inference"
    if role == "destination":
        return "intent_inference"
    if role == "pickup":
        return "acquisition"
    return "other"


def _latest_trial_log(log_dir):
    candidates = sorted(glob.glob(os.path.join(log_dir, "user_study_trials_*.jsonl")))
    return candidates[-1] if candidates else ""


def _default_output_path(input_path):
    root, _ = os.path.splitext(input_path)
    return root + "_analysis.csv"


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
            trials.append(payload)
    return trials


def _stringify(value):
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _as_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _target_candidate_ids(trial, label_to_candidate_id):
    labels = []
    final_goal = str(trial.get("final_inferred_goal") or "").strip()
    if final_goal:
        labels.append(final_goal)
    else:
        labels.extend(str(item).strip() for item in _as_list(trial.get("target_completion_labels")) if str(item).strip())
        target_label = str(trial.get("target_completion_label") or "").strip()
        if target_label:
            labels.append(target_label)
    ids = []
    for label in labels:
        candidate_id = label_to_candidate_id.get(label, "")
        if candidate_id and candidate_id not in ids:
            ids.append(candidate_id)
    return ids


def _first_time(items, predicate):
    times = []
    for item in items:
        if not isinstance(item, dict) or not predicate(item):
            continue
        try:
            times.append(float(item.get("t_rel_sec")))
        except Exception:
            continue
    return "" if not times else min(times)


def _first_stable_time(items, predicate, min_count=2):
    streak = 0
    first_time = None
    for item in sorted(
        [entry for entry in items if isinstance(entry, dict)],
        key=lambda entry: float(entry.get("t_rel_sec", 0.0) or 0.0),
    ):
        if predicate(item):
            if streak == 0:
                try:
                    first_time = float(item.get("t_rel_sec"))
                except Exception:
                    first_time = ""
            streak += 1
            if streak >= min_count:
                return first_time
        else:
            streak = 0
            first_time = None
    return ""


def _derived_metrics(trial, label_to_candidate_id):
    target_ids = _target_candidate_ids(trial, label_to_candidate_id)
    target_set = set(target_ids)
    casper_predictions = [item for item in _as_list(trial.get("casper_predictions")) if isinstance(item, dict)]
    intent_timeline = [item for item in _as_list(trial.get("intent_timeline")) if isinstance(item, dict)]
    confident = [item for item in casper_predictions if bool(item.get("confident"))]
    wrong_confident = [
        item
        for item in confident
        if target_set and str(item.get("predicted_candidate_id") or "").strip() not in target_set
    ]
    latest = str(trial.get("casper_latest_candidate_id") or "").strip()
    return {
        "target_candidate_ids": target_ids,
        "casper_final_matches_target": bool(target_set and latest in target_set),
        "casper_first_target_prediction_sec": _first_time(
            casper_predictions,
            lambda item: str(item.get("predicted_candidate_id") or "").strip() in target_set,
        ),
        "casper_first_confident_target_prediction_sec": _first_time(
            confident,
            lambda item: str(item.get("predicted_candidate_id") or "").strip() in target_set,
        ),
        "casper_first_stable_target_prediction_sec": _first_stable_time(
            confident,
            lambda item: str(item.get("predicted_candidate_id") or "").strip() in target_set,
            min_count=2,
        ),
        "casper_wrong_confident_count": len(wrong_confident),
        "casper_wrong_confident_rate": (
            float(len(wrong_confident)) / float(len(confident)) if confident else ""
        ),
        "top_goal_first_target_sec": _first_time(
            intent_timeline,
            lambda item: str(item.get("top_goal_candidate_id") or "").strip() in target_set,
        ),
        "top_goal_first_stable_target_sec": _first_stable_time(
            intent_timeline,
            lambda item: str(item.get("top_goal_candidate_id") or "").strip() in target_set,
            min_count=2,
        ),
        "top_goal_wrong_switch_count": sum(
            1
            for item in intent_timeline
            if str(item.get("event") or "").strip() == "top_goal_switch"
            and target_set
            and str(item.get("top_goal_candidate_id") or "").strip() not in target_set
        ),
    }


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


def _row_from_trial(trial, label_to_candidate_id):
    derived = _derived_metrics(trial, label_to_candidate_id)
    row = {}
    for column in DEFAULT_COLUMNS:
        if column == "analysis_outcome":
            row[column] = _analysis_outcome(trial)
            continue
        if column == "step_role":
            row[column] = _step_role(trial.get("step_id"))
            continue
        if column == "analysis_focus":
            row[column] = _analysis_focus(trial.get("task_id"), trial.get("step_id"))
            continue
        row[column] = _stringify(derived.get(column, trial.get(column)))
    return row


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


def export_trials(input_path, output_path, include_all=False, object_map_yaml=""):
    trials = _load_trials(input_path)
    if not include_all:
        trials = [trial for trial in trials if _include_in_analysis(trial)]
    label_to_candidate_id = _load_label_to_candidate_id(object_map_yaml or _default_object_map_path())
    rows = [_row_from_trial(trial, label_to_candidate_id) for trial in trials]

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
        help="Destination CSV path. Defaults to <input>_analysis.csv",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include administrative or incomplete rows that are normally excluded from analysis export.",
    )
    parser.add_argument(
        "--object-map-yaml",
        default=_default_object_map_path(),
        help="Object map YAML used to convert completion labels to candidate ids.",
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
        row_count = export_trials(
            input_path,
            output_path,
            include_all=args.include_all,
            object_map_yaml=os.path.expanduser(args.object_map_yaml),
        )
    except Exception as exc:
        print("Failed to export analysis CSV: {}".format(exc), file=sys.stderr)
        return 1

    print("Exported {} trial rows to {}".format(row_count, output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
