#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate spatial-instruction episodes from existing recorded scenes.

This script reuses existing images and slot assignments to build a new dataset
with more vision-dependent prompts such as:
  - Pick the carton on the left.
  - Pour from the middle carton.

Outputs:
  data/milk_candidate_cls/episodes_spatial.jsonl
  data/milk_candidate_cls/candidate_samples_spatial.jsonl
"""

import json
import os
from collections import OrderedDict


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def object_short_name(object_name):
    txt = str(object_name).strip().lower()
    if txt.startswith("whole"):
        return "whole"
    if txt.startswith("oat"):
        return "oat"
    if txt.startswith("soy"):
        return "soy"
    raise ValueError("Unsupported object name: {}".format(object_name))


def candidate_metadata(candidate_id):
    object_part, grasp_part = candidate_id.split("_", 1)
    object_name_map = {
        "whole": "whole_milk",
        "oat": "oat_milk",
        "soy": "soy_milk",
    }
    grasp_description_map = {
        "top": "top grasp",
        "side": "side grasp",
    }
    suitability_map = {
        "top": "pickup",
        "side": "pour",
    }
    return {
        "candidate_id": candidate_id,
        "object_short": object_part,
        "object_name": object_name_map.get(object_part, object_part),
        "grasp_type": grasp_part,
        "grasp_description": grasp_description_map.get(grasp_part, grasp_part),
        "task_suitability": suitability_map.get(grasp_part, ""),
        "candidate_text": (
            "Object: {}. Grasp: {}. Task suitability: {}."
        ).format(
            object_name_map.get(object_part, object_part),
            grasp_description_map.get(grasp_part, grasp_part),
            suitability_map.get(grasp_part, ""),
        ),
        "candidate_text_reduced": "Candidate grasp type: {}.".format(grasp_description_map.get(grasp_part, grasp_part)),
    }


def base_scene_records(rows):
    by_scene = OrderedDict()
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id not in by_scene:
            by_scene[scene_id] = row
    return list(by_scene.values())


def build_spatial_episodes(scene_rows):
    prompt_templates = [
        ("left", "pickup", "Pick the carton on the left."),
        ("left", "pickup", "Pick up the left carton."),
        ("center", "pickup", "Pick the carton in the middle."),
        ("center", "pickup", "Pick up the center carton."),
        ("right", "pickup", "Pick the carton on the right."),
        ("right", "pickup", "Pick up the right carton."),
        ("left", "pour", "Pour from the carton on the left."),
        ("left", "pour", "Grasp the left carton for pouring."),
        ("center", "pour", "Pour from the carton in the middle."),
        ("center", "pour", "Grasp the center carton for pouring."),
        ("right", "pour", "Pour from the carton on the right."),
        ("right", "pour", "Grasp the right carton for pouring."),
    ]

    episodes = []
    for scene in scene_rows:
        slot_assignment = dict(scene["slot_assignment"])
        for idx, (slot_name, task_type, instruction) in enumerate(prompt_templates, start=1):
            object_name = slot_assignment[slot_name]
            object_short = object_short_name(object_name)
            grasp_type = "top" if task_type == "pickup" else "side"
            correct_candidate_id = "{}_{}".format(object_short, grasp_type)
            episodes.append(
                {
                    "episode_id": "{}__spatial_{:02d}".format(scene["scene_id"], idx),
                    "scene_id": scene["scene_id"],
                    "image_path": scene["image_path"],
                    "instruction": instruction,
                    "correct_candidate_id": correct_candidate_id,
                    "task_type": task_type,
                    "slot_assignment": slot_assignment,
                    "view_id": scene.get("view_id", ""),
                    "scene_notes": "{}; synthetic_spatial_prompts".format(scene.get("scene_notes", "")).strip("; "),
                    "episode_notes": "spatial_prompt",
                    "allowed_candidates": list(scene["allowed_candidates"]),
                    "image_topic": scene.get("image_topic", ""),
                    "image_stamp": scene.get("image_stamp", ""),
                    "recorded_at": scene.get("recorded_at", ""),
                }
            )
    return episodes


def expand_episode(episode):
    allowed_candidates = list(episode["allowed_candidates"])
    correct_candidate_id = episode["correct_candidate_id"]
    rows = []
    for candidate_id in allowed_candidates:
        row = {
            "sample_id": "{}__{}".format(episode["episode_id"], candidate_id),
            "episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "view_id": episode.get("view_id", ""),
            "image_path": episode["image_path"],
            "instruction": episode["instruction"],
            "correct_candidate_id": correct_candidate_id,
            "label": 1 if candidate_id == correct_candidate_id else 0,
            "task_type": episode.get("task_type", ""),
            "scene_notes": episode.get("scene_notes", ""),
            "episode_notes": episode.get("episode_notes", ""),
            "slot_assignment": episode.get("slot_assignment", {}),
        }
        row.update(candidate_metadata(candidate_id))
        rows.append(row)
    return rows


def main():
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(package_root, "data", "milk_candidate_cls")
    episodes_path = os.path.join(dataset_root, "episodes.jsonl")
    spatial_episodes_path = os.path.join(dataset_root, "episodes_spatial.jsonl")
    spatial_candidates_path = os.path.join(dataset_root, "candidate_samples_spatial.jsonl")

    base_rows = load_jsonl(episodes_path)
    scene_rows = base_scene_records(base_rows)
    spatial_episodes = build_spatial_episodes(scene_rows)

    spatial_candidate_rows = []
    for episode in spatial_episodes:
        spatial_candidate_rows.extend(expand_episode(episode))

    write_jsonl(spatial_episodes_path, spatial_episodes)
    write_jsonl(spatial_candidates_path, spatial_candidate_rows)

    print("Base recorded rows      :", len(base_rows))
    print("Unique scene/view images:", len(scene_rows))
    print("Spatial episodes        :", len(spatial_episodes))
    print("Spatial candidate rows  :", len(spatial_candidate_rows))
    print("Wrote                   :", spatial_episodes_path)
    print("Wrote                   :", spatial_candidates_path)


if __name__ == "__main__":
    main()
