#!/usr/bin/env python3
"""Offline CASPER-lite intent evaluator over candidate-classification episodes.

This script evaluates the CASPER-style step that asks a VLM to choose one
intent from an explicit candidate set. It deliberately avoids ROS runtime
dependencies so that prompts, parsing, and metrics can be debugged before
adding an online shared-autonomy node.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "data",
    "milk_candidate_cls",
)


def _ascii_safe(text):
    return str(text or "").encode("ascii", "replace").decode("ascii")


def _normalize_candidate_row(row):
    candidate_id = str(row.get("candidate_id") or row.get("label") or row.get("id") or "").strip()
    object_name = str(row.get("object_name") or row.get("object") or "").strip()
    category = str(row.get("category") or "").strip()
    grasp_type = str(row.get("grasp_type") or "").strip()
    task_suitability = str(row.get("task_suitability") or row.get("task") or "").strip()
    if not grasp_type:
        if "destination" in category or task_suitability == "place":
            grasp_type = "destination"
        elif "side" in candidate_id:
            grasp_type = "side"
        elif "top" in candidate_id:
            grasp_type = "top"
    if not task_suitability:
        if grasp_type == "side":
            task_suitability = "pour"
        elif grasp_type == "destination":
            task_suitability = "place"
        elif grasp_type == "top":
            task_suitability = "pickup"
    out = dict(row)
    out["candidate_id"] = candidate_id
    out["object_name"] = object_name
    out["category"] = category
    out["grasp_type"] = grasp_type
    out["task_suitability"] = task_suitability
    if not out.get("candidate_text"):
        out["candidate_text"] = "Object: {}. Category: {}. Skill: {}.".format(
            object_name or "unknown",
            category or "unknown",
            task_suitability or "unknown",
        )
    return out


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


def _candidate_sort_key(row):
    candidate_id = str(row.get("candidate_id") or "")
    parts = candidate_id.split("_", 1)
    object_order = {"whole": 0, "oat": 1, "soy": 2}
    grasp_order = {"top": 0, "side": 1}
    return (
        object_order.get(parts[0], 99),
        grasp_order.get(parts[1] if len(parts) > 1 else "", 99),
        candidate_id,
    )


def _load_trajectory_lookup(path):
    if not path:
        return {}
    rows = _read_jsonl(path)
    lookup = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or row.get("trial_id") or "").strip()
        if not episode_id:
            continue
        lookup[episode_id] = {
            "trajectory_history": row.get("trajectory_history") or [],
            "trajectory_summary": row.get("trajectory_summary") or {},
        }
    return lookup


def load_milk_episodes(episodes_path, candidate_samples_path, limit=None, trajectory_jsonl=""):
    episodes = _read_jsonl(episodes_path)
    samples = _read_jsonl(candidate_samples_path)
    trajectory_lookup = _load_trajectory_lookup(trajectory_jsonl)

    by_episode = defaultdict(list)
    for row in samples:
        by_episode[str(row.get("episode_id") or "")].append(row)

    loaded = []
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        candidates = sorted(by_episode.get(episode_id, []), key=_candidate_sort_key)
        if not candidates:
            continue
        allowed = set(str(v) for v in episode.get("allowed_candidates") or [])
        if allowed:
            candidates = [row for row in candidates if str(row.get("candidate_id") or "") in allowed]
        trajectory = trajectory_lookup.get(episode_id, {})
        loaded.append(
            {
                "episode_id": episode_id,
                "scene_id": str(episode.get("scene_id") or ""),
                "view_id": str(episode.get("view_id") or ""),
                "image_path": str(episode.get("image_path") or ""),
                "instruction": str(episode.get("instruction") or ""),
                "task_type": str(episode.get("task_type") or ""),
                "correct_candidate_id": str(episode.get("correct_candidate_id") or ""),
                "slot_assignment": episode.get("slot_assignment") or {},
                "trajectory_history": trajectory.get("trajectory_history") or episode.get("trajectory_history") or [],
                "trajectory_summary": trajectory.get("trajectory_summary") or episode.get("trajectory_summary") or {},
                "task_history": episode.get("task_history") or episode.get("completed_actions") or [],
                "candidates": candidates,
            }
        )
        if limit is not None and len(loaded) >= limit:
            break
    return loaded


def load_replay_episodes(replay_jsonl, limit=None):
    rows = _read_jsonl(replay_jsonl)
    loaded = []
    for row in rows:
        candidates = [_normalize_candidate_row(item) for item in list(row.get("candidates") or [])]
        candidates = [item for item in candidates if item.get("candidate_id")]
        if not candidates:
            continue
        episode_id = str(row.get("episode_id") or row.get("trial_id") or "replay_{:06d}".format(len(loaded))).strip()
        loaded.append(
            {
                "episode_id": episode_id,
                "scene_id": str(row.get("scene_id") or row.get("active_task_id") or row.get("task_id") or ""),
                "view_id": str(row.get("view_id") or "replay"),
                "image_path": str(row.get("image_path") or ""),
                "instruction": str(
                    row.get("instruction")
                    or row.get("step_description")
                    or row.get("active_step_title")
                    or row.get("command")
                    or ""
                ),
                "task_type": str(row.get("task_type") or row.get("command") or row.get("active_step_id") or ""),
                "correct_candidate_id": str(row.get("correct_candidate_id") or row.get("target_candidate_id") or ""),
                "slot_assignment": row.get("slot_assignment") or {},
                "trajectory_history": row.get("trajectory_history") or [],
                "trajectory_summary": row.get("trajectory_summary") or {},
                "task_history": row.get("task_history") or row.get("completed_actions") or [],
                "candidates": candidates,
            }
        )
        if limit is not None and len(loaded) >= limit:
            break
    return loaded


def load_observation_episodes(
    observation_jsonl,
    limit=None,
    min_displacement_m=0.0,
    require_ground_truth=False,
    task_type_filter="",
    save_reason_filter="",
):
    rows = _read_jsonl(observation_jsonl)
    truth_by_trial_key = {}
    truth_by_task_step = {}
    for row in rows:
        candidate_id = str(
            row.get("correct_candidate_id")
            or row.get("intended_candidate_id")
            or row.get("terminal_candidate_id")
            or row.get("selected_candidate_id")
            or row.get("selected_tag_id")
            or ""
        ).strip()
        if not candidate_id:
            continue
        trial_key = str(row.get("trial_key") or "").strip()
        if trial_key:
            truth_by_trial_key[trial_key] = candidate_id
        context = row.get("trial_context") or row.get("context") or {}
        task_step_key = "::".join(
            str(context.get(key) or row.get(key) or "").strip()
            for key in ("session_id", "participant_id", "condition_id", "block_id", "active_task_id", "active_step_id")
        )
        if task_step_key.strip(":"):
            truth_by_task_step[task_step_key] = candidate_id

    loaded = []
    for row in rows:
        candidates = [_normalize_candidate_row(item) for item in list(row.get("candidates") or [])]
        candidates = [item for item in candidates if item.get("candidate_id")]
        if not candidates:
            continue

        summary = row.get("trajectory_summary") or {}
        try:
            displacement = float(summary.get("displacement_m", 0.0))
        except Exception:
            displacement = 0.0
        if displacement < float(min_displacement_m):
            continue

        context = row.get("trial_context") or row.get("context") or {}
        task_id = str(context.get("active_task_id") or row.get("scene_id") or row.get("active_task_id") or "")
        step_id = str(context.get("active_step_id") or row.get("task_type") or row.get("active_step_id") or "")
        if task_type_filter:
            allowed_steps = set(
                item.strip().lower()
                for item in str(task_type_filter).replace(",", " ").split()
                if item.strip()
            )
            if step_id.strip().lower() not in allowed_steps:
                continue
        if save_reason_filter:
            allowed_reasons = set(
                item.strip().lower()
                for item in str(save_reason_filter).replace(",", " ").split()
                if item.strip()
            )
            if str(row.get("save_reason") or "").strip().lower() not in allowed_reasons:
                continue
        instruction = str(
            context.get("active_step_title")
            or row.get("instruction")
            or "Infer the user's intended sandwich candidate from the visual observation and teleop trajectory."
        )
        task_step_key = "::".join(
            str(context.get(key) or row.get(key) or "").strip()
            for key in ("session_id", "participant_id", "condition_id", "block_id", "active_task_id", "active_step_id")
        )
        correct_candidate_id = str(
            row.get("correct_candidate_id")
            or row.get("intended_candidate_id")
            or row.get("terminal_candidate_id")
            or row.get("selected_candidate_id")
            or row.get("selected_tag_id")
            or truth_by_trial_key.get(str(row.get("trial_key") or "").strip(), "")
            or truth_by_task_step.get(task_step_key, "")
            or ""
        ).strip()
        if require_ground_truth and not correct_candidate_id:
            continue
        episode_id = str(
            row.get("episode_id")
            or row.get("event_id")
            or "{}__line_{:06d}".format(os.path.splitext(os.path.basename(observation_jsonl))[0], row.get("_line_no", len(loaded)))
        )

        loaded.append(
            {
                "episode_id": episode_id,
                "trial_key": str(row.get("trial_key") or task_step_key or episode_id),
                "save_reason": str(row.get("save_reason") or ""),
                "frame_index": int(row.get("_line_no", len(loaded))),
                "scene_id": task_id or "casper_observation",
                "view_id": str(row.get("save_reason") or row.get("view_id") or "observation"),
                "image_path": str(row.get("image_path") or ""),
                "semantic_map_path": str(row.get("semantic_map_path") or ""),
                "instruction": instruction,
                "task_type": step_id,
                "correct_candidate_id": correct_candidate_id,
                "slot_assignment": row.get("slot_assignment") or {},
                "trajectory_history": row.get("trajectory_history") or [],
                "trajectory_summary": summary,
                "task_history": row.get("task_history") or row.get("completed_actions") or [],
                "candidates": candidates,
                "source_row": {
                    "save_reason": row.get("save_reason"),
                    "control_phase": row.get("control_phase"),
                    "trajectory_source": row.get("trajectory_source"),
                    "trajectory_freeze_reason": row.get("trajectory_freeze_reason"),
                    "top_goal": row.get("top_goal"),
                    "top_goal_label": row.get("top_goal_label"),
                    "top_probability": row.get("top_probability"),
                    "distribution": row.get("distribution"),
                    "selected_grasp_label": row.get("selected_grasp_label"),
                    "selected_candidate_id": row.get("selected_candidate_id"),
                    "intended_grasp_label": row.get("intended_grasp_label"),
                    "intended_candidate_id": row.get("intended_candidate_id"),
                    "terminal_grasp_label": row.get("terminal_grasp_label"),
                    "terminal_candidate_id": row.get("terminal_candidate_id"),
                },
            }
        )
        if limit is not None and len(loaded) >= limit:
            break
    return loaded


def _as_xyz(value):
    if isinstance(value, dict):
        if all(key in value for key in ("x", "y", "z")):
            return [float(value["x"]), float(value["y"]), float(value["z"])]
        if "position" in value:
            return _as_xyz(value.get("position"))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def _format_xyz(value):
    xyz = _as_xyz(value)
    if xyz is None:
        return "unknown"
    return "({:.3f}, {:.3f}, {:.3f})".format(xyz[0], xyz[1], xyz[2])


def _direction_sign(value, eps=1e-4):
    value = float(value)
    if value > eps:
        return "+1"
    if value < -eps:
        return "-1"
    return "0"


def summarize_trajectory(episode):
    summary = episode.get("trajectory_summary") or {}
    history = episode.get("trajectory_history") or []
    if isinstance(summary, dict) and summary:
        return summary
    if not isinstance(history, list) or len(history) < 2:
        return {}

    points = []
    for item in history:
        if isinstance(item, dict):
            xyz = _as_xyz(item.get("ee_position") or item.get("position") or item.get("xyz"))
            stamp = item.get("stamp", item.get("t", item.get("time_sec")))
        else:
            xyz = _as_xyz(item)
            stamp = None
        if xyz is not None:
            points.append({"stamp": stamp, "xyz": xyz})
    if len(points) < 2:
        return {}

    start = points[0]["xyz"]
    end = points[-1]["xyz"]
    delta = [end[i] - start[i] for i in range(3)]
    displacement = math.sqrt(sum(v * v for v in delta))
    duration = None
    try:
        if points[0]["stamp"] is not None and points[-1]["stamp"] is not None:
            duration = max(0.0, float(points[-1]["stamp"]) - float(points[0]["stamp"]))
    except Exception:
        duration = None

    return {
        "available": True,
        "num_points": len(points),
        "start_xyz": start,
        "end_xyz": end,
        "delta_xyz": delta,
        "displacement_m": displacement,
        "duration_sec": duration,
        "subsampled_points": points[:3] + points[-3:] if len(points) > 6 else points,
    }


def format_trajectory_for_prompt(episode, geometry_mode="full"):
    summary = summarize_trajectory(episode)
    if not summary:
        return "Trajectory history: not available for this episode."

    lines = [
        "Trajectory history:",
        "- samples: {}".format(summary.get("num_points", "unknown")),
    ]
    geometry_mode = str(geometry_mode or "full").strip().lower()
    if geometry_mode == "full":
        lines.extend(
            [
                "- start end-effector xyz: {}".format(_format_xyz(summary.get("start_xyz"))),
                "- end end-effector xyz: {}".format(_format_xyz(summary.get("end_xyz"))),
                "- delta xyz: {}".format(_format_xyz(summary.get("delta_xyz"))),
            ]
        )
    else:
        lines.append("- absolute xyz hidden; use the visual marks and motion arrows as primary evidence.")
        if summary.get("delta_xyz") is not None:
            delta = _as_xyz(summary.get("delta_xyz"))
            if delta is not None:
                lines.append(
                    "- motion delta direction sign: x={}, y={}, z={}".format(
                        _direction_sign(delta[0]),
                        _direction_sign(delta[1]),
                        _direction_sign(delta[2]),
                    )
                )
    if summary.get("duration_sec") is not None:
        lines.append("- duration sec: {:.3f}".format(float(summary.get("duration_sec"))))
    if summary.get("displacement_m") is not None:
        lines.append("- displacement m: {:.3f}".format(float(summary.get("displacement_m"))))

    nearest_start = summary.get("nearest_candidate_start") or summary.get("nearest_start")
    nearest_end = summary.get("nearest_candidate_end") or summary.get("nearest_end")
    if nearest_start:
        lines.append("- nearest candidate at start: {}".format(nearest_start))
    if nearest_end:
        lines.append("- nearest candidate at end: {}".format(nearest_end))
    if summary.get("motion_direction"):
        lines.append("- motion direction: {}".format(summary.get("motion_direction")))
    if summary.get("notes"):
        lines.append("- notes: {}".format(summary.get("notes")))

    points = summary.get("subsampled_points") or []
    if points:
        formatted = []
        for item in points[:8]:
            if isinstance(item, dict):
                stamp = item.get("stamp", item.get("t", item.get("time_sec")))
                xyz = item.get("xyz") or item.get("ee_position") or item.get("position")
                prefix = "t={:.3f} ".format(float(stamp)) if stamp is not None else ""
                if geometry_mode == "full":
                    formatted.append("{}xyz={}".format(prefix, _format_xyz(xyz)))
                else:
                    formatted.append("{}point={}".format(prefix, len(formatted) + 1))
            else:
                formatted.append("xyz={}".format(_format_xyz(item)) if geometry_mode == "full" else "point={}".format(len(formatted) + 1))
        lines.append("- subsampled points: {}".format("; ".join(formatted)))
    return "\n".join(lines)


def _skill_class_for_candidate(row, task_type):
    task = str(row.get("task_suitability") or row.get("task") or "").strip().lower()
    category = str(row.get("category") or "").strip().lower()
    task_type = str(task_type or "").strip().lower()
    if task == "place" or category == "destination" or "destination" in task_type or "place" in task_type:
        return "Place"
    if task == "pour" or "pour" in task_type:
        return "Pour"
    return "Pick"


def _precondition_for_skill(skill_class):
    if skill_class == "Place":
        return "robot is holding an object"
    if skill_class == "Pour":
        return "robot should choose a pour-capable object or destination according to the task"
    return "gripper is empty and target object is pickable"


def _affordance_for_candidate(row, skill_class):
    category = str(row.get("category") or "").strip()
    object_name = str(row.get("object_name") or "").strip()
    if skill_class == "Place":
        return "stable destination surface for placing the held item"
    if skill_class == "Pour":
        return "pour-capable object or pour destination"
    if category:
        return "pickable {}".format(category)
    if object_name:
        return "pickable object"
    return "task-relevant manipulation target"


def _format_action_candidate(idx, row, task_type, geometry_mode):
    skill_class = _skill_class_for_candidate(row, task_type)
    position = row.get("position") or row.get("xyz") or row.get("pose")
    parts = [
        "{}. candidate_id={}".format(idx, row.get("candidate_id", "")),
        "intent_id={}".format(row.get("candidate_id", "")),
        "skill_class={}".format(skill_class),
        "target_object={}".format(row.get("object_name", "") or "unknown"),
        "object_category={}".format(row.get("category", "") or "unknown"),
        "preconditions={}".format(_precondition_for_skill(skill_class)),
        "affordance={}".format(_affordance_for_candidate(row, skill_class)),
        "expected_user_motion={}".format(
            "move held object toward this destination"
            if skill_class == "Place"
            else "move gripper toward this object"
        ),
    ]
    if str(geometry_mode or "full").strip().lower() == "full":
        parts.append("xyz={}".format(_format_xyz(position)))
    else:
        parts.append("xyz=hidden")
    rel = row.get("relative_to_gripper") or {}
    if isinstance(rel, dict):
        rel_parts = []
        if rel.get("distance_xy_m") is not None:
            rel_parts.append("distance_xy_m={:.3f}".format(float(rel.get("distance_xy_m"))))
        direction = rel.get("direction_xy")
        if isinstance(direction, (list, tuple)) and len(direction) >= 2:
            rel_parts.append("direction_xy=({:+.2f},{:+.2f})".format(float(direction[0]), float(direction[1])))
        if rel.get("rank_by_distance") is not None:
            rel_parts.append("rank_by_distance={}".format(int(rel.get("rank_by_distance"))))
        if rel_parts:
            parts.append("relative_to_gripper={}".format(",".join(rel_parts)))
    if row.get("candidate_text"):
        parts.append("description={}".format(row.get("candidate_text", "")))
    return " ".join(parts)


def _format_simple_candidate(idx, row, include_distances=False):
    parts = [
        "{}. candidate_id={}".format(idx, row.get("candidate_id", "")),
        "object={}".format(row.get("object_name", "") or "unknown"),
    ]
    if include_distances:
        rel = row.get("relative_to_gripper") or {}
        if isinstance(rel, dict):
            if rel.get("distance_xy_m") is not None:
                parts.append("distance_to_gripper_xy_m={:.3f}".format(float(rel.get("distance_xy_m"))))
            if rel.get("rank_by_distance") is not None:
                parts.append("rank_by_distance={}".format(int(rel.get("rank_by_distance"))))
    return " ".join(parts)


def _format_motion_candidate(idx, row):
    skill_class = _skill_class_for_candidate(row, row.get("task_type"))
    rel = row.get("relative_to_gripper") or {}
    parts = [
        "{}. visual_mark={}".format(idx, row.get("candidate_id", "")),
        "candidate_id={}".format(row.get("candidate_id", "")),
        "skill={}".format(skill_class),
    ]
    if isinstance(rel, dict):
        if rel.get("distance_xy_m") is not None:
            parts.append("distance_to_gripper_xy_m={:.3f}".format(float(rel.get("distance_xy_m"))))
        if rel.get("rank_by_distance") is not None:
            parts.append("rank_by_distance={}".format(int(rel.get("rank_by_distance"))))
        direction = rel.get("direction_xy")
        if isinstance(direction, (list, tuple)) and len(direction) >= 2:
            parts.append("direction_from_gripper_xy=({:+.2f},{:+.2f})".format(float(direction[0]), float(direction[1])))
    if len(parts) == 2:
        parts.append("distance_to_gripper_xy_m=unknown")
    return " ".join(parts)


def _format_object_reference(idx, row):
    skill_class = _skill_class_for_candidate(row, row.get("task_type"))
    return "{}. candidate_id={} object={} category={} skill={}".format(
        idx,
        row.get("candidate_id", ""),
        row.get("object_name", "") or "unknown",
        row.get("category", "") or "unknown",
        skill_class,
    )


def _format_task_history_for_prompt(episode, max_items=6):
    history = episode.get("task_history") or episode.get("completed_actions") or []
    if not isinstance(history, list) or not history:
        return "Task/action history: none."
    lines = ["Task/action history:"]
    for item in history[-max(1, int(max_items)):]:
        if isinstance(item, dict):
            status = str(item.get("status") or item.get("event") or item.get("type") or "").strip()
            skill = str(item.get("skill") or item.get("skill_class") or "").strip()
            candidate_id = str(item.get("candidate_id") or item.get("intent_id") or "").strip()
            obj = str(item.get("object_name") or item.get("object") or item.get("target_object") or "").strip()
            phase = str(item.get("task_phase") or item.get("step_id") or "").strip()
            parts = []
            if status:
                parts.append(status)
            if skill:
                parts.append(skill)
            if obj:
                parts.append(obj)
            if candidate_id:
                parts.append("(candidate_id={})".format(candidate_id))
            if phase:
                parts.append("phase={}".format(phase))
            text = " ".join(parts).strip()
        else:
            text = str(item).strip()
        if text:
            lines.append("- {}".format(text))
    return "\n".join(lines)


def build_prompt(episode, prompt_style=None, geometry_mode=None):
    prompt_style = str(prompt_style or episode.get("prompt_style") or "object_candidates").strip().lower()
    geometry_mode = str(geometry_mode or episode.get("prompt_geometry_mode") or "full").strip().lower()
    candidate_lines = []
    object_reference_lines = []
    for idx, row in enumerate(episode["candidates"], start=1):
        position = row.get("position") or row.get("xyz") or row.get("pose")
        if prompt_style == "simple":
            candidate_lines.append(_format_simple_candidate(idx, row, include_distances=False))
        elif prompt_style in ("simple_with_distances", "simple_distances", "distance_prompt"):
            candidate_lines.append(_format_motion_candidate(idx, row))
            object_reference_lines.append(_format_object_reference(idx, row))
        elif prompt_style in ("action_candidates", "actions", "skill_candidates", "casper_v2"):
            candidate_lines.append(_format_action_candidate(idx, row, episode.get("task_type"), geometry_mode))
        else:
            candidate_lines.append(
                "{}. candidate_id={} object={} grasp_type={} task_suitability={} xyz={} description={}".format(
                    idx,
                    row.get("candidate_id", ""),
                    row.get("object_name", ""),
                    row.get("grasp_type", ""),
                    row.get("task_suitability", ""),
                    _format_xyz(position) if geometry_mode == "full" else "hidden",
                    row.get("candidate_text", ""),
                )
            )

    slot_assignment = episode.get("slot_assignment") or {}
    slot_lines = []
    for slot in ("left", "center", "right"):
        if slot in slot_assignment:
            slot_lines.append("{}={}".format(slot, slot_assignment[slot]))

    image_history = [str(path) for path in episode.get("image_history_paths") or [] if str(path).strip()]
    if not image_history and episode.get("image_path"):
        image_history = [str(episode.get("image_path"))]
    semantic_map_path = str(episode.get("semantic_map_path") or "").strip()
    if semantic_map_path and semantic_map_path not in image_history:
        image_history.append(semantic_map_path)
    image_lines = []
    for idx, path in enumerate(image_history, start=1):
        image_lines.append("Image path {}: {}".format(idx, path))

    if prompt_style in ("simple", "simple_with_distances", "simple_distances", "distance_prompt"):
        if prompt_style in ("simple_with_distances", "simple_distances", "distance_prompt"):
            lines = [
                "You are a robot shared-autonomy intent inference module.",
                "Infer the user's current target from motion evidence only.",
                "Return only JSON with keys: intent_id, confidence, reason.",
                "",
                "Task: {}".format(episode["instruction"]),
                "Task phase: {}".format(episode["task_type"] or "unknown"),
                "",
                "Images:",
                "\n".join(image_lines) if image_lines else "Image path: unknown",
                "The wrist-camera image is local gripper context and may not show all objects.",
                "The top-down semantic map shows numbered pseudo-mask candidate regions when it is included.",
                "The red G mark is the robot gripper/end-effector.",
                "Use the top-down semantic map as the main workspace evidence and the wrist image only as local visual context.",
                "",
                _format_task_history_for_prompt(episode),
                "",
                "Decision rules:",
                "1. First look at where red G moved and where it ended relative to the numbered pseudo-mask regions.",
                "2. Then use distance_to_gripper_xy_m and rank_by_distance as supporting evidence.",
                "3. Use Task/action history only to understand the current phase and what has already been completed.",
                "4. Object names are only references for reporting; they are not task-order priors.",
                "5. Do not choose bread, patty, cheese, plate, or any other item because it is the common next recipe step.",
                "6. If motion is too small, ambiguous, or not aimed at a numbered mark, choose no_intent with low confidence.",
                "7. Use high confidence only when the motion, endpoint, and distance table point to the same candidate.",
                "",
                format_trajectory_for_prompt(episode, geometry_mode="relative_only"),
                "",
                "Action candidates:",
                "\n".join(candidate_lines),
            ]
            if object_reference_lines:
                lines.extend(
                    [
                        "",
                        "Object reference:",
                        "\n".join(object_reference_lines),
                    ]
                )
            lines.extend(
                [
                    "Uncertain option: candidate_id=no_intent object=none. Choose this if the motion evidence is insufficient.",
                    "",
                    "JSON:",
                ]
            )
            return _ascii_safe("\n".join(lines))

        lines = [
            "You are an intent inference module for robot shared autonomy.",
            "Choose the candidate that best matches the user's current motion.",
            "Return only JSON with keys: intent_id, confidence, reason.",
            "",
            "Task: {}".format(episode["instruction"]),
            "Task phase: {}".format(episode["task_type"] or "unknown"),
            "Images:",
            "\n".join(image_lines) if image_lines else "Image path: unknown",
            "Image marks use candidate_id numbers. The red G/gripper marker is the end-effector.",
        ]
        lines.extend(
            [
                "",
                format_trajectory_for_prompt(episode, geometry_mode="relative_only"),
                "",
                "Candidates:",
                "\n".join(candidate_lines),
                "Uncertain option: candidate_id=no_intent object=none. Choose this if the motion evidence is insufficient.",
                "",
                "JSON:",
            ]
        )
        return _ascii_safe("\n".join(lines))

    prompt = "\n".join(
        [
            "You are a CASPER-lite V2 shared-autonomy intent inference module.",
            "Choose exactly one candidate intent from the list.",
            "Your goal is to infer the user's intended target, not the next recipe step.",
            "Use the multi-frame visual observation history, visual marks, gripper marker, and trajectory arrows as primary evidence.",
            "Use object identity only to understand labels and affordances; do not prefer common sandwich ingredients by default.",
            "If the task says to choose any sandwich piece, every sandwich piece is valid, so do not use sandwich assembly order as a prior.",
            "Do not infer intent from recipe order. Bread-bottom-first or patty-next is not evidence of user intent.",
            "If the user has not moved meaningfully, or the gripper/trajectory evidence is ambiguous, choose no_intent with low confidence.",
            "Prefer the candidate that the end-effector moved toward, ended closest to, or visually appears under/near the gripper.",
            "When relative_to_gripper is available, rank_by_distance=1 and smaller distance_xy_m are strong evidence unless trajectory arrows clearly contradict them.",
            "Candidates are action/skill intents when skill_class is provided.",
            "Return only JSON with keys: intent_id, confidence, reason.",
            "",
            "Task instruction: {}".format(episode["instruction"]),
            "Task type hint: {}".format(episode["task_type"] or "unknown"),
            "Scene id: {}".format(episode["scene_id"]),
            "View id: {}".format(episode["view_id"]),
            "Prompt style: {}".format(prompt_style),
            "Prompt geometry mode: {}".format(geometry_mode),
            "Observation history images:",
            "\n".join(image_lines) if image_lines else "Image path: unknown",
            "Image marks use candidate_id numbers. The red G/gripper marker is the end-effector.",
            "If a top-down semantic map is included, use it as the workspace-level view and use the wrist image as local visual evidence.",
            "Known spatial assignment: {}".format(", ".join(slot_lines) if slot_lines else "unknown"),
            "",
            format_trajectory_for_prompt(episode, geometry_mode=geometry_mode),
            "",
            "Action candidates:" if prompt_style in ("action_candidates", "actions", "skill_candidates", "casper_v2") else "Candidates:",
            "\n".join(candidate_lines),
            "Uncertain option: candidate_id=no_intent intent_id=no_intent skill_class=Wait target_object=none object_category=none preconditions=insufficient user motion evidence affordance=no takeover expected_user_motion=none description=Choose this when evidence is insufficient.",
            "",
            "JSON:",
        ]
    )
    return _ascii_safe(prompt)


def _normalize_candidate_id(text, valid_ids):
    value = str(text or "").strip()
    if value in valid_ids:
        return value
    lower_to_id = {candidate_id.lower(): candidate_id for candidate_id in valid_ids}
    if value.lower() in lower_to_id:
        return lower_to_id[value.lower()]
    for candidate_id in valid_ids:
        if re.search(r"\b{}\b".format(re.escape(candidate_id)), value, flags=re.IGNORECASE):
            return candidate_id
    return ""


def parse_model_response(text, valid_ids):
    raw_text = str(text or "").strip()
    payload = {}
    try:
        payload = json.loads(raw_text)
    except Exception:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = {}

    intent_id = ""
    confidence = 0.0
    reason = ""
    if isinstance(payload, dict):
        intent_id = _normalize_candidate_id(
            payload.get("intent_id")
            or payload.get("candidate_id")
            or payload.get("intent")
            or payload.get("answer"),
            valid_ids,
        )
        try:
            confidence = float(payload.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        reason = str(payload.get("reason") or "")

    if not intent_id:
        intent_id = _normalize_candidate_id(raw_text, valid_ids)
    confidence = max(0.0, min(1.0, confidence))
    return {"intent_id": intent_id, "confidence": confidence, "reason": reason, "raw_response": raw_text}


class RuleBackend(object):
    """Deterministic dry-run backend for validating the evaluator plumbing."""

    def predict(self, episode, prompt):
        instruction = episode["instruction"].lower()
        task_type = episode["task_type"].lower()
        slot_assignment = episode.get("slot_assignment") or {}

        wanted_object = ""
        for key, object_name in slot_assignment.items():
            key = str(key).lower()
            object_name = str(object_name)
            if key in instruction:
                wanted_object = object_name
                break

        if not wanted_object:
            for object_name in ("whole_milk", "oat_milk", "soy_milk"):
                short = object_name.split("_", 1)[0]
                if short in instruction or object_name.replace("_", " ") in instruction:
                    wanted_object = object_name
                    break

        if not task_type:
            if any(word in instruction for word in ("pour", "dispense")):
                task_type = "pour"
            elif any(word in instruction for word in ("pick", "grab", "take", "choose")):
                task_type = "pickup"

        best_id = ""
        best_score = -1.0
        for row in episode["candidates"]:
            score = 0.0
            if wanted_object and str(row.get("object_name") or "") == wanted_object:
                score += 2.0
            if task_type and str(row.get("task_suitability") or "") == task_type:
                score += 1.0
            if score > best_score:
                best_score = score
                best_id = str(row.get("candidate_id") or "")

        confidence = 0.5 if best_score <= 0.0 else min(0.95, 0.45 + 0.2 * best_score)
        return {
            "intent_id": best_id,
            "confidence": confidence,
            "reason": "rule backend dry run",
            "raw_response": json.dumps({"intent_id": best_id, "confidence": confidence}),
        }


class TrajectoryRuleBackend(object):
    """Geometry-only sanity check over teleop trajectory and candidate positions."""

    def predict(self, episode, prompt):
        summary = summarize_trajectory(episode)
        start = _as_xyz(summary.get("start_xyz"))
        end = _as_xyz(summary.get("end_xyz"))
        delta = _as_xyz(summary.get("delta_xyz"))
        if start is None or end is None:
            history = episode.get("trajectory_history") or []
            points = []
            for item in history:
                if isinstance(item, dict):
                    xyz = _as_xyz(item.get("ee_position") or item.get("position") or item.get("xyz"))
                else:
                    xyz = _as_xyz(item)
                if xyz is not None:
                    points.append(xyz)
            if len(points) >= 2:
                start, end = points[0], points[-1]
                delta = [end[i] - start[i] for i in range(3)]
        if start is None or end is None or delta is None:
            first_id = str(episode["candidates"][0].get("candidate_id") or "")
            return {
                "intent_id": first_id,
                "confidence": 0.0,
                "reason": "no usable trajectory; returned first candidate",
                "raw_response": json.dumps({"intent_id": first_id, "confidence": 0.0}),
            }

        delta_norm = math.sqrt(sum(v * v for v in delta))
        scores = []
        for row in episode["candidates"]:
            candidate_id = str(row.get("candidate_id") or "")
            goal = _as_xyz(row.get("position") or row.get("xyz") or row.get("pose"))
            if goal is None:
                continue
            d_end = math.sqrt(sum((end[i] - goal[i]) ** 2 for i in range(3)))
            d_start = math.sqrt(sum((start[i] - goal[i]) ** 2 for i in range(3)))
            improvement = d_start - d_end
            alignment = 0.0
            perpendicular = d_end
            if delta_norm > 1e-6:
                to_goal = [goal[i] - start[i] for i in range(3)]
                to_goal_norm = math.sqrt(sum(v * v for v in to_goal))
                if to_goal_norm > 1e-6:
                    alignment = sum(delta[i] * to_goal[i] for i in range(3)) / (delta_norm * to_goal_norm)
                projection = sum(delta[i] * to_goal[i] for i in range(3)) / max(delta_norm, 1e-6)
                closest = [start[i] + (projection / max(delta_norm, 1e-6)) * delta[i] for i in range(3)]
                perpendicular = math.sqrt(sum((closest[i] - goal[i]) ** 2 for i in range(3)))
            score = 2.0 * improvement + 0.15 * alignment - 0.35 * d_end - 0.15 * perpendicular
            scores.append((score, candidate_id, d_start, d_end, improvement, alignment))

        if not scores:
            first_id = str(episode["candidates"][0].get("candidate_id") or "")
            return {
                "intent_id": first_id,
                "confidence": 0.0,
                "reason": "no candidate positions; returned first candidate",
                "raw_response": json.dumps({"intent_id": first_id, "confidence": 0.0}),
            }
        scores.sort(reverse=True)
        best = scores[0]
        margin = best[0] - scores[1][0] if len(scores) > 1 else abs(best[0])
        confidence = max(0.05, min(0.95, 0.45 + 2.0 * margin))
        reason = "trajectory_rule: candidate {} had best approach score; d_start={:.3f}, d_end={:.3f}, improvement={:.3f}, alignment={:.3f}".format(
            best[1], best[2], best[3], best[4], best[5]
        )
        return {
            "intent_id": best[1],
            "confidence": confidence,
            "reason": reason,
            "raw_response": json.dumps({"intent_id": best[1], "confidence": confidence, "reason": reason}),
        }


class CommandBackend(object):
    """Backend that sends the prompt to an external command over stdin.

    The command must print JSON such as:
      {"intent_id": "soy_side", "confidence": 0.82, "reason": "..."}
    """

    def __init__(self, command, timeout_sec):
        if not command:
            raise ValueError("--command is required for backend=command")
        self.command = command
        self.timeout_sec = float(timeout_sec)

    def predict(self, episode, prompt):
        proc = subprocess.run(
            self.command,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
            timeout=self.timeout_sec,
        )
        valid_ids = [str(row.get("candidate_id") or "") for row in episode["candidates"]]
        if proc.returncode != 0:
            if proc.stdout.strip():
                parsed = parse_model_response(proc.stdout, valid_ids)
                parsed["reason"] = parsed.get("reason") or "command backend failed rc={}".format(proc.returncode)
                parsed["raw_response"] = proc.stdout.strip()
                return parsed
            raise RuntimeError(
                "command backend failed rc={} stderr={} stdout={}".format(
                    proc.returncode,
                    proc.stderr.strip(),
                    proc.stdout.strip(),
                )
            )
        return parse_model_response(proc.stdout, valid_ids)


class PromptDumpBackend(object):
    def predict(self, episode, prompt):
        first_id = str(episode["candidates"][0].get("candidate_id") or "")
        return {
            "intent_id": first_id,
            "confidence": 0.0,
            "reason": "prompt_dump placeholder",
            "raw_response": "",
        }


def build_backend(args):
    name = str(args.backend or "rule").strip().lower()
    if name == "rule":
        return RuleBackend()
    if name == "trajectory_rule":
        return TrajectoryRuleBackend()
    if name == "command":
        return CommandBackend(args.command, args.timeout_sec)
    if name == "prompt_dump":
        return PromptDumpBackend()
    raise ValueError("unsupported backend: {}".format(args.backend))


def _predict_with_latency(backend, episode, prompt):
    start_time = time.time()
    pred = backend.predict(episode, prompt)
    pred["latency_sec"] = time.time() - start_time
    return pred


def _self_consistency_predictions(backend, episode, prompt, k):
    k = max(1, int(k))
    batch_start = time.time()
    if k == 1:
        predictions = [_predict_with_latency(backend, episode, prompt)]
    else:
        predictions = []
        with ThreadPoolExecutor(max_workers=k) as pool:
            futures = [pool.submit(_predict_with_latency, backend, episode, prompt) for _ in range(k)]
            for future in as_completed(futures):
                predictions.append(future.result())
    return predictions, time.time() - batch_start


def evaluate(episodes, backend, self_consistency_k=1, agreement_threshold=1):
    prediction_rows = []
    correct = 0
    attempted = 0
    confident = 0
    confident_correct = 0
    latencies = []

    for episode in episodes:
        prompt = build_prompt(episode)
        valid_ids = [str(row.get("candidate_id") or "") for row in episode["candidates"]]
        votes = []
        raw_predictions, batch_latency_sec = _self_consistency_predictions(
            backend,
            episode,
            prompt,
            self_consistency_k,
        )
        for pred in raw_predictions:
            pred["intent_id"] = _normalize_candidate_id(pred.get("intent_id"), valid_ids)
            latencies.append(float(pred.get("latency_sec") or 0.0))
            votes.append(pred.get("intent_id") or "")

        counts = Counter(v for v in votes if v)
        top_id, vote_count = ("", 0) if not counts else counts.most_common(1)[0]
        is_confident = vote_count >= int(agreement_threshold)
        confidence = float(vote_count) / float(max(1, int(self_consistency_k)))
        is_correct = top_id == episode["correct_candidate_id"]

        if top_id:
            attempted += 1
            if is_correct:
                correct += 1
        if is_confident:
            confident += 1
            if is_correct:
                confident_correct += 1

        prediction_rows.append(
            {
                "episode_id": episode["episode_id"],
                "trial_key": episode.get("trial_key") or episode["episode_id"],
                "save_reason": episode.get("save_reason") or episode.get("view_id") or "",
                "frame_index": episode.get("frame_index"),
                "scene_id": episode["scene_id"],
                "view_id": episode["view_id"],
                "instruction": episode["instruction"],
                "task_type": episode["task_type"],
                "image_path": episode.get("image_path") or "",
                "correct_candidate_id": episode["correct_candidate_id"],
                "predicted_candidate_id": top_id,
                "correct": bool(is_correct),
                "confident": bool(is_confident),
                "self_consistency_votes": dict(counts),
                "self_consistency_agreement": vote_count,
                "self_consistency_confidence": confidence,
                "latency_sec_wall": batch_latency_sec,
                "latency_sec_api_sum": sum(float(pred.get("latency_sec") or 0.0) for pred in raw_predictions),
                "prompt": prompt,
                "raw_predictions": raw_predictions,
                "source_row": episode.get("source_row") or {},
            }
        )

    total = len(episodes)
    metrics = {
        "episodes": total,
        "attempted": attempted,
        "accuracy": (float(correct) / float(total)) if total else 0.0,
        "attempted_accuracy": (float(correct) / float(attempted)) if attempted else 0.0,
        "confident": confident,
        "confident_rate": (float(confident) / float(total)) if total else 0.0,
        "confident_accuracy": (float(confident_correct) / float(confident)) if confident else 0.0,
        "self_consistency_k": int(self_consistency_k),
        "agreement_threshold": int(agreement_threshold),
    }
    if latencies:
        sorted_latencies = sorted(latencies)
        p95_idx = min(len(sorted_latencies) - 1, int(math.ceil(0.95 * len(sorted_latencies))) - 1)
        metrics.update(
            {
                "latency_sec_min": min(latencies),
                "latency_sec_mean": sum(latencies) / float(len(latencies)),
                "latency_sec_median": sorted_latencies[len(sorted_latencies) // 2],
                "latency_sec_p95": sorted_latencies[p95_idx],
                "latency_sec_max": max(latencies),
            }
        )
    return metrics, prediction_rows


def aggregate_trial_predictions(prediction_rows):
    grouped = defaultdict(list)
    for row in prediction_rows:
        correct_id = str(row.get("correct_candidate_id") or "").strip()
        group_key = (
            str(row.get("trial_key") or row.get("episode_id") or "").strip(),
            str(row.get("scene_id") or "").strip(),
            str(row.get("task_type") or "").strip(),
            correct_id,
        )
        grouped[group_key].append(row)

    trial_rows = []
    attempted = 0
    correct = 0
    confident = 0
    confident_correct = 0

    for index, (group_key, rows) in enumerate(sorted(grouped.items()), start=1):
        _trial_key, scene_id, task_type, correct_id = group_key
        counts = Counter(str(row.get("predicted_candidate_id") or "").strip() for row in rows if row.get("predicted_candidate_id"))
        top_id = ""
        vote_count = 0
        if counts:
            candidates = counts.most_common()
            best_vote_count = candidates[0][1]
            tied_ids = [candidate_id for candidate_id, count in candidates if count == best_vote_count]
            if len(tied_ids) == 1:
                top_id = tied_ids[0]
                vote_count = best_vote_count
            else:
                mean_conf = {}
                for candidate_id in tied_ids:
                    vals = [
                        float(row.get("self_consistency_confidence") or 0.0)
                        for row in rows
                        if str(row.get("predicted_candidate_id") or "").strip() == candidate_id
                    ]
                    mean_conf[candidate_id] = sum(vals) / float(max(1, len(vals)))
                top_id = sorted(tied_ids, key=lambda cid: (-mean_conf.get(cid, 0.0), cid))[0]
                vote_count = best_vote_count
        is_attempted = bool(top_id)
        agreement = float(vote_count) / float(max(1, len(rows)))
        is_confident = is_attempted and vote_count >= max(1, int(math.ceil(0.5 * len(rows))))
        is_correct = bool(top_id and correct_id and top_id == correct_id)

        if is_attempted:
            attempted += 1
            if is_correct:
                correct += 1
        if is_confident:
            confident += 1
            if is_correct:
                confident_correct += 1

        trial_rows.append(
            {
                "trial_index": index,
                "trial_key": group_key[0],
                "scene_id": scene_id,
                "task_type": task_type,
                "correct_candidate_id": correct_id,
                "predicted_candidate_id": top_id,
                "correct": bool(is_correct),
                "confident": bool(is_confident),
                "vote_count": vote_count,
                "frame_count": len(rows),
                "agreement": agreement,
                "votes": dict(counts),
                "frame_episode_ids": [row.get("episode_id") for row in rows],
                "frame_save_reasons": [row.get("save_reason") for row in rows],
            }
        )

    total = len(trial_rows)
    metrics = {
        "trials": total,
        "attempted": attempted,
        "accuracy": (float(correct) / float(total)) if total else 0.0,
        "attempted_accuracy": (float(correct) / float(attempted)) if attempted else 0.0,
        "confident": confident,
        "confident_rate": (float(confident) / float(total)) if total else 0.0,
        "confident_accuracy": (float(confident_correct) / float(confident)) if confident else 0.0,
    }
    return metrics, trial_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        default=os.path.join(DEFAULT_DATA_DIR, "episodes_semantic_spatial.jsonl"),
        help="Episode JSONL path.",
    )
    parser.add_argument(
        "--replay-jsonl",
        default="",
        help="CASPER-lite replay JSONL. When set, milk episode/candidate paths are ignored.",
    )
    parser.add_argument(
        "--observation-jsonl",
        default="",
        help="CASPER observation JSONL from casper_observation_logger.py. When set, other episode paths are ignored.",
    )
    parser.add_argument(
        "--candidate-samples",
        default=os.path.join(DEFAULT_DATA_DIR, "candidate_samples_semantic_spatial.jsonl"),
        help="Candidate sample JSONL path.",
    )
    parser.add_argument(
        "--trajectory-jsonl",
        default="",
        help="Optional JSONL with episode_id/trial_id plus trajectory_history or trajectory_summary fields.",
    )
    parser.add_argument(
        "--backend",
        choices=("rule", "trajectory_rule", "prompt_dump", "command"),
        default="rule",
        help="Prediction backend.",
    )
    parser.add_argument(
        "--min-displacement-m",
        type=float,
        default=0.0,
        help="For --observation-jsonl, skip rows with less teleop displacement than this.",
    )
    parser.add_argument(
        "--require-ground-truth",
        action="store_true",
        help="For --observation-jsonl, evaluate only rows with recovered correct_candidate_id.",
    )
    parser.add_argument(
        "--task-type-filter",
        default="",
        help="For --observation-jsonl, keep only these active_step_id/task_type values, comma or space separated.",
    )
    parser.add_argument(
        "--save-reason-filter",
        default="",
        help="For --observation-jsonl, keep only these save_reason values, comma or space separated.",
    )
    parser.add_argument(
        "--command",
        default="",
        help="Shell command for backend=command. Prompt is passed on stdin; JSON must be printed on stdout.",
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-call timeout for backend=command.")
    parser.add_argument("--self-consistency-k", type=int, default=1, help="Number of repeated backend calls.")
    parser.add_argument("--agreement-threshold", type=int, default=1, help="Minimum matching votes for confident=True.")
    parser.add_argument(
        "--prompt-style",
        choices=("simple", "simple_with_distances", "object_candidates", "action_candidates", "casper_v2"),
        default="object_candidates",
        help="Prompt candidate format.",
    )
    parser.add_argument(
        "--prompt-geometry-mode",
        choices=("full", "relative_only", "no_xyz"),
        default="full",
        help="How much absolute xyz geometry to expose in the prompt.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum episode count.")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "results",
            "casper_lite_predictions.jsonl",
        ),
        help="Prediction JSONL output path.",
    )
    parser.add_argument(
        "--metrics-output",
        default=os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "results",
            "casper_lite_metrics.json",
        ),
        help="Metrics JSON output path.",
    )
    parser.add_argument(
        "--trial-output",
        default="",
        help="Optional trial-level aggregated prediction JSONL output path.",
    )
    parser.add_argument(
        "--trial-metrics-output",
        default="",
        help="Optional trial-level aggregated metrics JSON output path.",
    )
    args = parser.parse_args()

    if args.observation_jsonl:
        episodes = load_observation_episodes(
            os.path.expanduser(args.observation_jsonl),
            limit=args.limit if args.limit > 0 else None,
            min_displacement_m=args.min_displacement_m,
            require_ground_truth=args.require_ground_truth,
            task_type_filter=args.task_type_filter,
            save_reason_filter=args.save_reason_filter,
        )
    elif args.replay_jsonl:
        episodes = load_replay_episodes(
            os.path.expanduser(args.replay_jsonl),
            limit=args.limit if args.limit > 0 else None,
        )
    else:
        episodes = load_milk_episodes(
            os.path.expanduser(args.episodes),
            os.path.expanduser(args.candidate_samples),
            limit=args.limit if args.limit > 0 else None,
            trajectory_jsonl=os.path.expanduser(args.trajectory_jsonl),
        )
    for episode in episodes:
        episode["prompt_style"] = args.prompt_style
        episode["prompt_geometry_mode"] = args.prompt_geometry_mode
    backend = build_backend(args)
    metrics, rows = evaluate(
        episodes,
        backend,
        self_consistency_k=args.self_consistency_k,
        agreement_threshold=args.agreement_threshold,
    )

    _write_jsonl(os.path.expanduser(args.output), rows)
    metrics_path = os.path.expanduser(args.metrics_output)
    metrics_dir = os.path.dirname(os.path.abspath(metrics_path))
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.trial_output or args.trial_metrics_output:
        trial_metrics, trial_rows = aggregate_trial_predictions(rows)
        if args.trial_output:
            _write_jsonl(os.path.expanduser(args.trial_output), trial_rows)
        if args.trial_metrics_output:
            trial_metrics_path = os.path.expanduser(args.trial_metrics_output)
            trial_metrics_dir = os.path.dirname(os.path.abspath(trial_metrics_path))
            if trial_metrics_dir:
                os.makedirs(trial_metrics_dir, exist_ok=True)
            with open(trial_metrics_path, "w", encoding="utf-8") as handle:
                json.dump(trial_metrics, handle, indent=2, sort_keys=True)
                handle.write("\n")
        metrics["trial_level"] = trial_metrics

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
