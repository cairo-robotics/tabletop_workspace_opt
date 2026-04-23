#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand episode-level candidate labels into candidate-level training rows.

Input:
  data/milk_candidate_cls/episodes.jsonl

Output:
  data/milk_candidate_cls/candidate_samples.jsonl

Each episode row contains the correct candidate id for one image + instruction.
This script expands it into one row per allowed candidate with a binary label.
"""

import json
import os
from typing import Dict, Iterable, List


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def candidate_metadata(candidate_id: str) -> Dict:
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
    }


def expand_episode(episode: Dict) -> List[Dict]:
    allowed_candidates = list(episode.get("allowed_candidates", []))
    correct_candidate_id = episode["correct_candidate_id"]
    expanded = []

    for candidate_id in allowed_candidates:
        row = {
            "sample_id": "{}__{}".format(episode["episode_id"], candidate_id),
            "episode_id": episode["episode_id"],
            "scene_id": episode.get("scene_id", ""),
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
        expanded.append(row)

    return expanded


def main():
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(package_root, "data", "milk_candidate_cls")
    episodes_path = os.path.join(dataset_root, "episodes.jsonl")
    output_path = os.path.join(dataset_root, "candidate_samples.jsonl")

    if not os.path.exists(episodes_path):
        raise FileNotFoundError("Missing episodes file: {}".format(episodes_path))

    episodes = load_jsonl(episodes_path)
    expanded_rows: List[Dict] = []
    for episode in episodes:
        expanded_rows.extend(expand_episode(episode))

    count = write_jsonl(output_path, expanded_rows)
    print("Episodes      :", len(episodes))
    print("Candidates/ep :", len(episodes[0].get("allowed_candidates", [])) if episodes else 0)
    print("Expanded rows :", count)
    print("Wrote         :", output_path)


if __name__ == "__main__":
    main()
