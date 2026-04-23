#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate semantic + spatial instruction episodes from recorded scenes.

This script reuses existing images and slot assignments to build a more
vision-dependent dataset with three prompt families:
  1. spatial prompts:
       - Pick the carton on the left.
       - Pour from the carton in the center.
  2. semantic prompts:
       - Pick the dairy milk.
       - Pour the plant-based milk that is not soy.
  3. semantic + spatial prompts:
       - Pick the dairy carton on the right.
       - Pour from the soy-based carton in the center.

The generated prompts are designed to keep a unique target candidate for each
scene and avoid ambiguous labels such as "pick the plant-based milk".

Outputs:
  data/milk_candidate_cls/episodes_semantic_spatial.jsonl
  data/milk_candidate_cls/candidate_samples_semantic_spatial.jsonl
"""

import json
import os
from collections import OrderedDict


OBJECT_NAME_MAP = {
    "whole": "whole_milk",
    "oat": "oat_milk",
    "soy": "soy_milk",
}

GRASP_DESCRIPTION_MAP = {
    "top": "top grasp",
    "side": "side grasp",
}

SUITABILITY_MAP = {
    "top": "pickup",
    "side": "pour",
}

SEMANTIC_PHRASES = {
    "whole_milk": {
        "noun": "dairy",
        "pickup": [
            "Pick the dairy milk.",
            "Pick up the dairy carton.",
        ],
        "pour": [
            "Pour the dairy milk.",
            "Pour from the dairy carton.",
        ],
    },
    "oat_milk": {
        "noun": "plant_based_not_soy",
        "pickup": [
            "Pick the plant-based milk that is not soy.",
            "Pick up the non-dairy carton that is not soy.",
        ],
        "pour": [
            "Pour the plant-based milk that is not soy.",
            "Pour from the non-dairy carton that is not soy.",
        ],
    },
    "soy_milk": {
        "noun": "soy_based",
        "pickup": [
            "Pick the soy-based milk alternative.",
            "Pick up the soy milk carton.",
        ],
        "pour": [
            "Pour the soy-based milk alternative.",
            "Pour from the soy milk carton.",
        ],
    },
}


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
    return {
        "candidate_id": candidate_id,
        "object_short": object_part,
        "object_name": OBJECT_NAME_MAP.get(object_part, object_part),
        "grasp_type": grasp_part,
        "grasp_description": GRASP_DESCRIPTION_MAP.get(grasp_part, grasp_part),
        "task_suitability": SUITABILITY_MAP.get(grasp_part, ""),
        "candidate_text": (
            "Object: {}. Grasp: {}. Task suitability: {}."
        ).format(
            OBJECT_NAME_MAP.get(object_part, object_part),
            GRASP_DESCRIPTION_MAP.get(grasp_part, grasp_part),
            SUITABILITY_MAP.get(grasp_part, ""),
        ),
        "candidate_text_reduced": "Candidate grasp type: {}.".format(
            GRASP_DESCRIPTION_MAP.get(grasp_part, grasp_part)
        ),
    }


def base_scene_records(rows):
    by_scene = OrderedDict()
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id not in by_scene:
            by_scene[scene_id] = row
    return list(by_scene.values())


def append_episode(episodes, seen_keys, scene, instruction, correct_candidate_id, task_type, prompt_family):
    key = (scene["scene_id"], instruction, correct_candidate_id, task_type)
    if key in seen_keys:
        return
    seen_keys.add(key)

    episode_idx = len(episodes) + 1
    episodes.append(
        {
            "episode_id": "{}__semsp_{:03d}".format(scene["scene_id"], episode_idx),
            "scene_id": scene["scene_id"],
            "image_path": scene["image_path"],
            "instruction": instruction,
            "correct_candidate_id": correct_candidate_id,
            "task_type": task_type,
            "slot_assignment": dict(scene["slot_assignment"]),
            "view_id": scene.get("view_id", ""),
            "scene_notes": "{}; synthetic_semantic_spatial_prompts".format(scene.get("scene_notes", "")).strip("; "),
            "episode_notes": prompt_family,
            "allowed_candidates": list(scene["allowed_candidates"]),
            "image_topic": scene.get("image_topic", ""),
            "image_stamp": scene.get("image_stamp", ""),
            "recorded_at": scene.get("recorded_at", ""),
        }
    )


def spatial_templates():
    return [
        ("left", "pickup", "Pick the carton on the left."),
        ("left", "pickup", "Pick up the left carton."),
        ("left", "pickup", "Grab the leftmost carton."),
        ("left", "pickup", "Choose the carton on the left side."),
        ("left", "pickup", "Take the carton on the far left."),
        ("left", "pickup", "Lift the left-side carton."),
        ("center", "pickup", "Pick the carton in the center."),
        ("center", "pickup", "Pick up the middle carton."),
        ("center", "pickup", "Grab the center carton."),
        ("center", "pickup", "Choose the carton in the middle."),
        ("center", "pickup", "Take the carton in the center position."),
        ("center", "pickup", "Lift the middle carton."),
        ("right", "pickup", "Pick the carton on the right."),
        ("right", "pickup", "Pick up the right carton."),
        ("right", "pickup", "Grab the rightmost carton."),
        ("right", "pickup", "Choose the carton on the right side."),
        ("right", "pickup", "Take the carton on the far right."),
        ("right", "pickup", "Lift the right-side carton."),
        ("left", "pour", "Pour from the carton on the left."),
        ("left", "pour", "Use the left carton for pouring."),
        ("left", "pour", "Grasp the leftmost carton for pouring."),
        ("left", "pour", "Pour using the carton on the left side."),
        ("left", "pour", "Use the far-left carton to pour."),
        ("left", "pour", "Choose the left carton for pouring."),
        ("center", "pour", "Pour from the carton in the center."),
        ("center", "pour", "Use the middle carton for pouring."),
        ("center", "pour", "Grasp the center carton for pouring."),
        ("center", "pour", "Pour using the carton in the middle."),
        ("center", "pour", "Use the center-position carton to pour."),
        ("center", "pour", "Choose the middle carton for pouring."),
        ("right", "pour", "Pour from the carton on the right."),
        ("right", "pour", "Use the right carton for pouring."),
        ("right", "pour", "Grasp the rightmost carton for pouring."),
        ("right", "pour", "Pour using the carton on the right side."),
        ("right", "pour", "Use the far-right carton to pour."),
        ("right", "pour", "Choose the right carton for pouring."),
    ]


def semantic_spatial_templates(slot_name, object_name):
    slot_phrase = {
        "left": "on the left",
        "center": "in the center",
        "right": "on the right",
    }[slot_name]
    phrases = SEMANTIC_PHRASES[object_name]
    if object_name == "whole_milk":
        return [
            ("pickup", "Pick the dairy carton {}.".format(slot_phrase)),
            ("pour", "Pour from the dairy carton {}.".format(slot_phrase)),
        ]
    if object_name == "oat_milk":
        return [
            ("pickup", "Pick the plant-based carton that is not soy {}.".format(slot_phrase)),
            ("pour", "Pour from the plant-based carton that is not soy {}.".format(slot_phrase)),
        ]
    if object_name == "soy_milk":
        return [
            ("pickup", "Pick the soy-based carton {}.".format(slot_phrase)),
            ("pour", "Pour from the soy-based carton {}.".format(slot_phrase)),
        ]
    raise ValueError("Unsupported object name: {}".format(object_name))


def build_episodes(scene_rows):
    episodes = []
    seen_keys = set()

    for scene in scene_rows:
        slot_assignment = dict(scene["slot_assignment"])

        for slot_name, task_type, instruction in spatial_templates():
            object_name = slot_assignment[slot_name]
            object_short = object_short_name(object_name)
            grasp_type = "top" if task_type == "pickup" else "side"
            append_episode(
                episodes,
                seen_keys,
                scene,
                instruction,
                "{}_{}".format(object_short, grasp_type),
                task_type,
                "spatial_prompt",
            )

        for object_name, phrase_group in SEMANTIC_PHRASES.items():
            object_short = object_short_name(object_name)
            for task_type in ("pickup", "pour"):
                grasp_type = "top" if task_type == "pickup" else "side"
                for instruction in phrase_group[task_type]:
                    append_episode(
                        episodes,
                        seen_keys,
                        scene,
                        instruction,
                        "{}_{}".format(object_short, grasp_type),
                        task_type,
                        "semantic_prompt",
                    )

        for slot_name, object_name in slot_assignment.items():
            object_short = object_short_name(object_name)
            for task_type, instruction in semantic_spatial_templates(slot_name, object_name):
                grasp_type = "top" if task_type == "pickup" else "side"
                append_episode(
                    episodes,
                    seen_keys,
                    scene,
                    instruction,
                    "{}_{}".format(object_short, grasp_type),
                    task_type,
                    "semantic_spatial_prompt",
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
    output_episodes_path = os.path.join(dataset_root, "episodes_semantic_spatial.jsonl")
    output_candidates_path = os.path.join(dataset_root, "candidate_samples_semantic_spatial.jsonl")

    base_rows = load_jsonl(episodes_path)
    scene_rows = base_scene_records(base_rows)
    generated_episodes = build_episodes(scene_rows)

    candidate_rows = []
    for episode in generated_episodes:
        candidate_rows.extend(expand_episode(episode))

    write_jsonl(output_episodes_path, generated_episodes)
    write_jsonl(output_candidates_path, candidate_rows)

    print("Base recorded rows           :", len(base_rows))
    print("Unique scene/view images     :", len(scene_rows))
    print("Generated semantic episodes  :", len(generated_episodes))
    print("Generated candidate rows     :", len(candidate_rows))
    print("Wrote                        :", output_episodes_path)
    print("Wrote                        :", output_candidates_path)


if __name__ == "__main__":
    main()
