#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modular semantic scorer interface for shared-autonomy fusion."""

import importlib
import json
import os

from intent_score_fusion import OBJECT_ORDER, normalize_object_probs


class BaseSemanticScorer(object):
    def score(self, instruction, image_path=None, image_bgr=None):
        raise NotImplementedError


class UniformSemanticScorer(BaseSemanticScorer):
    def score(self, instruction, image_path=None, image_bgr=None):
        return normalize_object_probs({})


class KeywordRuleSemanticScorer(BaseSemanticScorer):
    """Lightweight fallback for demos when no trained callable is exported yet."""

    def score(self, instruction, image_path=None, image_bgr=None):
        text = str(instruction or "").strip().lower()
        scores = {obj: 0.2 for obj in OBJECT_ORDER}

        if "allergic to soy" in text or "not soy" in text or "soy-free" in text:
            scores["soy_milk"] = 0.01
            scores["whole_milk"] += 0.49
            scores["oat_milk"] += 0.30
        if "whole milk" in text or "dairy" in text or "milk" in text:
            scores["whole_milk"] += 0.45
        if "oat" in text:
            scores["oat_milk"] += 0.55
        if "soy" in text and "allergic to soy" not in text and "not soy" not in text and "soy-free" not in text:
            scores["soy_milk"] += 0.55

        return normalize_object_probs(scores)


class JsonSemanticScorer(BaseSemanticScorer):
    """Load normalized scores from a JSON lookup table.

    Accepted JSON formats:
    1. { "<instruction>": {"whole_milk": 0.7, ...}, ... }
    2. { "<scene_key>|<instruction>": {...}, ... }
    """

    def __init__(self, json_path):
        self.json_path = os.path.expanduser(str(json_path))
        if not os.path.exists(self.json_path):
            raise FileNotFoundError("semantic scores json not found: {}".format(self.json_path))
        with open(self.json_path, "r", encoding="utf-8") as handle:
            self.lookup = json.load(handle) or {}

    def score(self, instruction, image_path=None, image_bgr=None):
        key_instruction = str(instruction or "").strip()
        scene_key = os.path.basename(str(image_path or "")).strip()
        lookup_key = "{}|{}".format(scene_key, key_instruction) if scene_key else key_instruction
        raw = self.lookup.get(lookup_key, self.lookup.get(key_instruction, {}))
        return normalize_object_probs(raw)


class CallableSemanticScorer(BaseSemanticScorer):
    """Call a user-exported Python function from a module path string.

    Callable spec format:
      package.module:function_name
    """

    def __init__(self, callable_spec):
        spec = str(callable_spec or "").strip()
        if ":" not in spec:
            raise ValueError("callable spec must be package.module:function_name")
        module_name, function_name = spec.split(":", 1)
        module = importlib.import_module(module_name)
        self.fn = getattr(module, function_name)

    def score(self, instruction, image_path=None, image_bgr=None):
        raw = self.fn(
            instruction=str(instruction or ""),
            image_path=image_path,
            image_bgr=image_bgr,
            object_order=list(OBJECT_ORDER),
        )
        return normalize_object_probs(raw)


def build_semantic_scorer(backend="disabled", callable_spec="", json_path=""):
    backend_name = str(backend or "disabled").strip().lower()
    if backend_name in ("", "disabled", "none", "uniform"):
        return UniformSemanticScorer()
    if backend_name in ("keyword", "keyword_rules", "rule", "rules"):
        return KeywordRuleSemanticScorer()
    if backend_name in ("json", "precomputed_json"):
        return JsonSemanticScorer(json_path=json_path)
    if backend_name in ("callable", "python_callable", "module_callable"):
        return CallableSemanticScorer(callable_spec=callable_spec)
    raise ValueError("unsupported semantic scorer backend: {}".format(backend))
