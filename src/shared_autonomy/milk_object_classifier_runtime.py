#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime inference for the deployed milk object classifier."""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor

from intent_score_fusion import normalize_object_probs


OBJECT_TO_INDEX = {"whole_milk": 0, "oat_milk": 1, "soy_milk": 2}
INDEX_TO_OBJECT = {v: k for k, v in OBJECT_TO_INDEX.items()}


def get_image_features(model, pixel_values):
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)
    return model.vision_model(pixel_values=pixel_values).pooler_output


def get_text_features(model, input_ids, attention_mask=None):
    if hasattr(model, "get_text_features"):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return model.get_text_features(**kwargs)
    if attention_mask is not None:
        return model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    return model.text_model(input_ids=input_ids).pooler_output


def l2_normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


class ImageTextObjectClassifier(nn.Module):
    def __init__(self, vlm_backbone, embed_dim, num_classes=3):
        super().__init__()
        self.vlm_backbone = vlm_backbone
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            image_feat = get_image_features(self.vlm_backbone, pixel_values)
            text_feat = get_text_features(self.vlm_backbone, input_ids, attention_mask)
            image_feat = l2_normalize(image_feat)
            text_feat = l2_normalize(text_feat)
        joint = torch.cat([image_feat, text_feat, image_feat * text_feat], dim=1)
        return self.classifier(joint)


class MilkObjectClassifierRuntime(object):
    def __init__(self, checkpoint_path, device=None):
        self.checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError("checkpoint not found: {}".format(self.checkpoint_path))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        self.model_name = str(payload["model_name"])
        self.embed_dim = int(payload["embed_dim"])

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.model = ImageTextObjectClassifier(self.backbone, self.embed_dim).to(self.device)
        self.model.classifier.load_state_dict(payload["classifier_state_dict"])
        self.model.eval()

    def _prepare_image(self, image_path=None, image_bgr=None):
        if image_bgr is not None:
            rgb = image_bgr[:, :, ::-1]
            return Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        if image_path:
            return Image.open(os.path.expanduser(str(image_path))).convert("RGB")
        raise ValueError("either image_path or image_bgr must be provided")

    def predict(self, instruction, image_path=None, image_bgr=None):
        image = self._prepare_image(image_path=image_path, image_bgr=image_bgr)
        text = str(instruction or "").strip()
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True)

        pixel_values = inputs["pixel_values"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().tolist()

        raw = {INDEX_TO_OBJECT[idx]: float(prob) for idx, prob in enumerate(probs)}
        return normalize_object_probs(raw)


_RUNTIME_CACHE = {}


def predict_object_scores(instruction, image_path=None, image_bgr=None, object_order=None):
    checkpoint_path = os.environ.get("MILK_OBJECT_CLASSIFIER_CKPT", "").strip()
    if not checkpoint_path:
        raise RuntimeError("MILK_OBJECT_CLASSIFIER_CKPT is not set")
    runtime = _RUNTIME_CACHE.get(checkpoint_path)
    if runtime is None:
        runtime = MilkObjectClassifierRuntime(checkpoint_path=checkpoint_path)
        _RUNTIME_CACHE[checkpoint_path] = runtime
    return runtime.predict(instruction=instruction, image_path=image_path, image_bgr=image_bgr)
