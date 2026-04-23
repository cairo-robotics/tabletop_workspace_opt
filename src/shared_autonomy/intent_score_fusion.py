#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for object-level semantic fusion in shared autonomy."""

import json


OBJECT_ORDER = ("whole_milk", "oat_milk", "soy_milk")


def canonicalize_object_name(name):
    txt = str(name or "").strip().lower()
    if not txt:
        return ""
    if "whole" in txt or txt in ("milk_box", "milkbox", "milk", "whole_milk"):
        return "whole_milk"
    if "oat" in txt:
        return "oat_milk"
    if "soy" in txt:
        return "soy_milk"
    return txt


def candidate_task_action(grasp_id):
    txt = str(grasp_id or "").strip().lower()
    if txt.startswith("side_") or "_side_" in txt or txt.endswith("_side"):
        return "pour"
    if txt.startswith("top_") or "_top_" in txt or txt.endswith("_top"):
        return "pickup"
    return ""


def normalize_object_probs(probs, object_order=OBJECT_ORDER):
    out = {obj: max(0.0, float(probs.get(obj, 0.0))) for obj in object_order}
    total = sum(out.values())
    if total <= 1e-9:
        uniform = 1.0 / float(len(object_order))
        return {obj: uniform for obj in object_order}
    return {obj: val / total for obj, val in out.items()}


def aggregate_candidate_probs_by_object(candidate_rows, object_order=OBJECT_ORDER, prob_key="probability"):
    totals = {obj: 0.0 for obj in object_order}
    for row in candidate_rows:
        obj = canonicalize_object_name(row.get("canonical_object") or row.get("object_name"))
        if obj not in totals:
            continue
        totals[obj] += max(0.0, float(row.get(prob_key, 0.0)))
    return normalize_object_probs(totals, object_order=object_order)


def fuse_object_probs(teleop_probs, semantic_probs, method="weighted_sum", alpha=0.7, beta=0.3, object_order=OBJECT_ORDER):
    teleop = normalize_object_probs(teleop_probs, object_order=object_order)
    semantic = normalize_object_probs(semantic_probs, object_order=object_order)

    if str(method).strip().lower() == "multiplicative":
        fused = {obj: teleop[obj] * semantic[obj] for obj in object_order}
    else:
        fused = {obj: alpha * teleop[obj] + beta * semantic[obj] for obj in object_order}
    return normalize_object_probs(fused, object_order=object_order)


def project_object_probs_to_candidates(candidate_rows, fused_object_probs, object_order=OBJECT_ORDER, teleop_prob_key="probability"):
    fused_object_probs = normalize_object_probs(fused_object_probs, object_order=object_order)
    teleop_object_probs = aggregate_candidate_probs_by_object(candidate_rows, object_order=object_order, prob_key=teleop_prob_key)

    per_object_candidate_count = {obj: 0 for obj in object_order}
    for row in candidate_rows:
        obj = canonicalize_object_name(row.get("canonical_object") or row.get("object_name"))
        if obj in per_object_candidate_count:
            per_object_candidate_count[obj] += 1

    fused_candidate_probs = []
    for row in candidate_rows:
        obj = canonicalize_object_name(row.get("canonical_object") or row.get("object_name"))
        teleop_prob = max(0.0, float(row.get(teleop_prob_key, 0.0)))
        object_mass = fused_object_probs.get(obj, 0.0)
        teleop_object_mass = teleop_object_probs.get(obj, 0.0)
        if teleop_object_mass > 1e-9:
            fused_prob = object_mass * (teleop_prob / teleop_object_mass)
        else:
            count = max(1, per_object_candidate_count.get(obj, 1))
            fused_prob = object_mass / float(count)
        fused_candidate_probs.append(fused_prob)

    total = sum(fused_candidate_probs)
    if total <= 1e-9:
        uniform = 1.0 / max(1.0, float(len(fused_candidate_probs)))
        return [uniform for _ in fused_candidate_probs]
    return [val / total for val in fused_candidate_probs]


def top_object_label(probs, object_order=OBJECT_ORDER):
    norm = normalize_object_probs(probs, object_order=object_order)
    return max(object_order, key=lambda obj: norm.get(obj, 0.0))


def probs_to_json(probs, object_order=OBJECT_ORDER):
    norm = normalize_object_probs(probs, object_order=object_order)
    return json.dumps({obj: round(norm[obj], 6) for obj in object_order}, sort_keys=True)
