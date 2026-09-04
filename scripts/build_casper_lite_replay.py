#!/usr/bin/env python3
"""Build CASPER-lite replay episodes from user-study probability logs.

The output JSONL can be consumed by:

  python3 scripts/casper_lite_evaluator.py --replay-jsonl results/casper_lite_replay_episodes.jsonl
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import yaml


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise ValueError("{}:{} invalid json: {}".format(path, line_no, exc))
            payload["_source_path"] = path
            payload["_line_no"] = line_no
            rows.append(payload)
    return rows


def _write_jsonl(path, rows):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _expand_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            paths.extend(matches)
        else:
            paths.append(os.path.expanduser(pattern))
    return sorted(dict.fromkeys(paths))


def _parse_iso_timestamp(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _load_object_map(path):
    if not path or not os.path.exists(path):
        return {}, {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    entries = raw.get("tag_objects") or raw.get("candidate_objects") or {}
    tag_to_meta = {}
    label_to_tag = {}
    for key, meta in entries.items():
        if not isinstance(meta, dict):
            continue
        tag = str(key).strip()
        tag_to_meta[tag] = dict(meta)
        label = str(meta.get("grasp_complete_label") or "").strip()
        if label:
            label_to_tag[label] = tag
    return tag_to_meta, label_to_tag


def _trial_key(row):
    return (
        str(row.get("session_id") or ""),
        str(row.get("participant_id") or ""),
        str(row.get("condition_id") or ""),
        str(row.get("block_id") or ""),
        str(row.get("task_id") or row.get("active_task_id") or ""),
        str(row.get("step_id") or row.get("active_step_id") or ""),
    )


def _prob_key(row):
    return (
        str(row.get("session_id") or ""),
        str(row.get("participant_id") or ""),
        str(row.get("condition_id") or ""),
        str(row.get("block_id") or ""),
        str(row.get("active_task_id") or ""),
        str(row.get("active_step_id") or ""),
    )


def _load_trials(paths, label_to_tag):
    trials = []
    for path in paths:
        if not os.path.exists(path):
            continue
        for row in _read_jsonl(path):
            start_sec = row.get("start_time_sec")
            end_sec = row.get("end_time_sec")
            try:
                start_sec = float(start_sec) if start_sec is not None else None
                end_sec = float(end_sec) if end_sec is not None else None
            except Exception:
                start_sec, end_sec = None, None
            correct = _correct_candidate_from_trial(row, label_to_tag)
            trials.append(
                {
                    "key": _trial_key(row),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "correct_candidate_id": correct,
                    "trial": row,
                }
            )
    return trials


def _correct_candidate_from_trial(trial, label_to_tag):
    labels = []
    committed = str(trial.get("committed_goal_label") or "").strip()
    final_goal = str(trial.get("final_inferred_goal") or "").strip()
    target = str(trial.get("target_completion_label") or "").strip()
    labels.extend([committed, target, final_goal])
    labels.extend(str(item).strip() for item in list(trial.get("target_completion_labels") or []) if str(item).strip())
    for label in labels:
        if not label:
            continue
        if label in label_to_tag:
            return label_to_tag[label]
        if label.lstrip("-").isdigit():
            return label
    return ""


def _matching_trial(row, trials):
    key = _prob_key(row)
    stamp = _parse_iso_timestamp(row.get("timestamp"))
    candidates = [trial for trial in trials if trial["key"] == key]
    if not candidates:
        return None
    if stamp is not None:
        for trial in candidates:
            start_sec = trial.get("start_sec")
            end_sec = trial.get("end_sec")
            if start_sec is None or end_sec is None:
                continue
            if start_sec <= stamp <= end_sec:
                return trial
    return candidates[0]


def _task_type_from_row(row):
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("active_step_id", "active_step_title", "current_phase", "target_source")
    )
    if any(word in text for word in ("destination", "place", "release")):
        return "place"
    if "pour" in text:
        return "pour"
    if any(word in text for word in ("select", "grasp", "pick", "ingredient", "item")):
        return "pickup"
    return str(row.get("active_step_id") or "unknown")


def _candidate_from_probability_object(obj, tag_to_meta):
    label = str(obj.get("label") or "").strip()
    meta = tag_to_meta.get(label, {})
    object_name = str(obj.get("object_name") or meta.get("object_name") or "").strip()
    category = str(obj.get("category") or meta.get("category") or "").strip()
    task = str(meta.get("task") or "").strip()
    if not task:
        if category == "destination":
            task = "place"
        elif "side" in str(meta.get("grasp_complete_label") or ""):
            task = "pour"
        else:
            task = "pickup"
    grasp_type = "destination" if task == "place" or category == "destination" else ""
    label_text = str(meta.get("grasp_complete_label") or "").strip()
    if not grasp_type:
        if "side" in label_text:
            grasp_type = "side"
        elif "top" in label_text:
            grasp_type = "top"
        else:
            grasp_type = task
    return {
        "candidate_id": label,
        "label": label,
        "object_name": object_name,
        "category": category,
        "grasp_type": grasp_type,
        "task_suitability": task,
        "grasp_complete_label": label_text,
        "candidate_text": "Object: {}. Category: {}. Skill: {}. Label: {}.".format(
            object_name or "unknown",
            category or "unknown",
            task or "unknown",
            label_text or label,
        ),
    }


def _candidate_ids(row):
    return [str(v).strip() for v in list(row.get("candidate_labels") or []) if str(v).strip()]


def _candidate_rows(row, tag_to_meta):
    objects = list(row.get("candidate_objects") or [])
    if not objects:
        objects = [{"label": label} for label in _candidate_ids(row)]
    candidates = [_candidate_from_probability_object(obj, tag_to_meta) for obj in objects]
    allowed = set(str(v) for v in list(row.get("allowed_tag_ids") or []))
    if allowed:
        filtered = [item for item in candidates if str(item.get("candidate_id")) in allowed]
        if filtered:
            candidates = filtered
    seen = set()
    out = []
    for item in candidates:
        cid = str(item.get("candidate_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(item)
    return out


def _row_is_candidate_for_replay(row, args):
    if not str(row.get("active_step_id") or "").strip():
        return False
    if len(_candidate_ids(row)) < args.min_candidates:
        return False
    if not args.include_after_selection and str(row.get("selected_grasp_label") or "").strip():
        return False
    return True


def _should_sample(row, last_sample_sec, args):
    if args.sample_interval_sec <= 0:
        return True
    stamp = _parse_iso_timestamp(row.get("timestamp"))
    if stamp is None:
        return last_sample_sec is None
    if last_sample_sec is None:
        return True
    return (stamp - last_sample_sec) >= args.sample_interval_sec


def _instruction_from_row(row, trial):
    if trial is not None:
        trial_row = trial.get("trial") or {}
        for key in ("step_description", "step_title", "command"):
            value = str(trial_row.get(key) or "").strip()
            if value:
                return value
    for key in ("active_step_title", "active_step_id", "current_phase"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def build_replay(probability_paths, trial_paths, object_map_yaml, args):
    tag_to_meta, label_to_tag = _load_object_map(object_map_yaml)
    trials = _load_trials(trial_paths, label_to_tag)
    rows = []
    for path in probability_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows.extend(_read_jsonl(path))
    rows.sort(key=lambda row: (str(row.get("_source_path") or ""), row.get("_line_no", 0)))

    last_sample_by_key = {}
    episodes = []
    skipped_no_target = 0
    for row in rows:
        if not _row_is_candidate_for_replay(row, args):
            continue
        key = _prob_key(row)
        last_sample_sec = last_sample_by_key.get(key)
        if not _should_sample(row, last_sample_sec, args):
            continue
        stamp = _parse_iso_timestamp(row.get("timestamp"))
        last_sample_by_key[key] = stamp if stamp is not None else 0.0

        trial = _matching_trial(row, trials)
        correct_candidate_id = trial.get("correct_candidate_id", "") if trial else ""
        selected = str(row.get("selected_grasp_label") or "").strip()
        if not correct_candidate_id and selected in label_to_tag:
            correct_candidate_id = label_to_tag[selected]
        if not correct_candidate_id and args.allow_top_goal_as_target:
            correct_candidate_id = str(row.get("top_goal_label") or "").strip()
        if not correct_candidate_id and not args.include_unlabeled:
            skipped_no_target += 1
            continue

        candidates = _candidate_rows(row, tag_to_meta)
        if not candidates:
            continue
        valid_ids = set(str(item.get("candidate_id")) for item in candidates)
        if correct_candidate_id and correct_candidate_id not in valid_ids and not args.include_unlabeled:
            skipped_no_target += 1
            continue

        source_name = os.path.basename(str(row.get("_source_path") or "probability_log"))
        episode_id = "{}__line_{:06d}".format(os.path.splitext(source_name)[0], int(row.get("_line_no", 0)))
        episodes.append(
            {
                "episode_id": episode_id,
                "source_probability_log": row.get("_source_path", ""),
                "source_line_no": row.get("_line_no"),
                "timestamp": row.get("timestamp", ""),
                "session_id": row.get("session_id", ""),
                "participant_id": row.get("participant_id", ""),
                "condition_id": row.get("condition_id", ""),
                "block_id": row.get("block_id", ""),
                "active_task_id": row.get("active_task_id", ""),
                "active_task_name": row.get("active_task_name", ""),
                "active_step_id": row.get("active_step_id", ""),
                "active_step_title": row.get("active_step_title", ""),
                "instruction": _instruction_from_row(row, trial),
                "task_type": _task_type_from_row(row),
                "correct_candidate_id": correct_candidate_id,
                "candidates": candidates,
                "probabilities": row.get("probabilities") or [],
                "top_goal_label": row.get("top_goal_label", ""),
                "top_probability": row.get("top_probability", 0.0),
                "selected_grasp_label": row.get("selected_grasp_label", ""),
                "execution_state": row.get("execution_state", ""),
                "trajectory_history": row.get("trajectory_history") or [],
                "trajectory_summary": row.get("trajectory_summary") or {"available": False},
            }
        )
        if args.limit > 0 and len(episodes) >= args.limit:
            break

    metrics = {
        "episodes": len(episodes),
        "probability_rows_read": len(rows),
        "trials_read": len(trials),
        "skipped_no_target": skipped_no_target,
        "probability_paths": probability_paths,
        "trial_paths": trial_paths,
        "object_map_yaml": object_map_yaml,
    }
    return episodes, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "probability_logs",
        nargs="+",
        help="Probability log path(s) or glob(s), e.g. logs/user_study_probability_*.jsonl",
    )
    parser.add_argument(
        "--trial-logs",
        nargs="*",
        default=[os.path.join(PACKAGE_ROOT, "logs", "user_study_trials_*.jsonl")],
        help="Trial log path(s) or glob(s) used to recover target labels.",
    )
    parser.add_argument(
        "--object-map-yaml",
        default=os.path.join(PACKAGE_ROOT, "config", "apriltag_object_map.yaml"),
        help="AprilTag object map YAML.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PACKAGE_ROOT, "results", "casper_lite_replay_episodes.jsonl"),
        help="Replay episode JSONL output path.",
    )
    parser.add_argument(
        "--metrics-output",
        default=os.path.join(PACKAGE_ROOT, "results", "casper_lite_replay_metrics.json"),
        help="Replay build metrics JSON output path.",
    )
    parser.add_argument("--sample-interval-sec", type=float, default=1.0, help="Minimum time between samples per step.")
    parser.add_argument("--min-candidates", type=int, default=2, help="Minimum candidate count for a replay row.")
    parser.add_argument("--limit", type=int, default=0, help="Optional output episode limit.")
    parser.add_argument("--include-after-selection", action="store_true", help="Keep rows after selected_grasp_label is set.")
    parser.add_argument("--include-unlabeled", action="store_true", help="Keep rows with no recoverable correct candidate.")
    parser.add_argument(
        "--allow-top-goal-as-target",
        action="store_true",
        help="Use top_goal_label as target when no trial/selection target is available. Useful only for smoke tests.",
    )
    args = parser.parse_args()

    probability_paths = _expand_paths(args.probability_logs)
    trial_paths = _expand_paths(args.trial_logs)
    episodes, metrics = build_replay(
        probability_paths,
        trial_paths,
        os.path.expanduser(args.object_map_yaml),
        args,
    )
    _write_jsonl(os.path.expanduser(args.output), episodes)
    metrics_path = os.path.expanduser(args.metrics_output)
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
