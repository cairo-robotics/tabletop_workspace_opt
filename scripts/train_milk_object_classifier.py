#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and export the image+text CLIP milk object classifier for deployment."""

import argparse
import gc
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor


SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OBJECT_TO_INDEX = {"whole": 0, "oat": 1, "soy": 2}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def base_scene_id(scene_id):
    for suffix in ("_top", "_side", "_lean"):
        if scene_id.endswith(suffix):
            return scene_id[: -len(suffix)]
    return scene_id


def object_from_candidate_id(candidate_id):
    return str(candidate_id).split("_", 1)[0]


class ImageTextObjectDataset(Dataset):
    def __init__(self, records, image_root):
        self.records = records
        self.image_root = image_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(os.path.join(self.image_root, rec["image_path"])).convert("RGB")
        return {
            "image": image,
            "instruction": rec["instruction"],
            "target_index": rec["target_index"],
        }


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


def collate_image_text(batch, processor):
    images = [item["image"] for item in batch]
    texts = [item["instruction"] for item in batch]
    image_inputs = processor(images=images, return_tensors="pt")
    text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    out = {
        "pixel_values": image_inputs["pixel_values"],
        "input_ids": text_inputs["input_ids"],
        "targets": torch.tensor([item["target_index"] for item in batch], dtype=torch.long),
    }
    if "attention_mask" in text_inputs:
        out["attention_mask"] = text_inputs["attention_mask"]
    return out


def build_records(df):
    records = []
    for episode_id, g in df.groupby("episode_id", sort=True):
        row0 = g.iloc[0]
        records.append(
            {
                "episode_id": episode_id,
                "scene_id": row0["scene_id"],
                "base_scene_id": row0["base_scene_id"],
                "image_path": row0["image_path"],
                "instruction": row0["instruction"],
                "target_index": int(row0["target_object_index"]),
            }
        )
    return records


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    n = 0
    correct = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        targets = batch["targets"].to(device)
        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, targets)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        n += len(targets)
        total_loss += float(loss.item()) * len(targets)
    return {"loss": total_loss / max(n, 1), "top1": correct / max(n, 1)}


def main():
    parser = argparse.ArgumentParser(description="Train/export image+text CLIP milk object classifier.")
    parser.add_argument("--samples-path", default="", help="candidate_samples_semantic_spatial.jsonl path")
    parser.add_argument("--project-root", default="/home/gyanig/catkin_ws/src/tabletop_workspace_opt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", required=True, help="deployable checkpoint output path")
    args = parser.parse_args()

    project_root = os.path.expanduser(args.project_root)
    data_root = os.path.join(project_root, "data", "milk_candidate_cls")
    samples_path = os.path.expanduser(args.samples_path) if args.samples_path else os.path.join(data_root, "candidate_samples_semantic_spatial.jsonl")
    image_root = data_root

    df = pd.DataFrame(load_jsonl(samples_path))
    df = df[df["episode_notes"] == "spatial_prompt"].copy().reset_index(drop=True)
    df["base_scene_id"] = df["scene_id"].map(base_scene_id)
    df["target_object"] = df["correct_candidate_id"].map(object_from_candidate_id)
    df["target_object_index"] = df["target_object"].map(OBJECT_TO_INDEX)

    base_scenes = sorted(df["base_scene_id"].unique())
    val_scene_count = 2 if len(base_scenes) >= 4 else 1
    train_scenes = base_scenes[:-val_scene_count]
    val_scenes = base_scenes[-val_scene_count:]

    train_df = df[df["base_scene_id"].isin(train_scenes)].reset_index(drop=True)
    val_df = df[df["base_scene_id"].isin(val_scenes)].reset_index(drop=True)
    train_records = build_records(train_df)
    val_records = build_records(val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy_img = Image.open(os.path.join(image_root, train_records[0]["image_path"])).convert("RGB")
        dummy_text = train_records[0]["instruction"]
        dummy = processor(images=dummy_img, text=dummy_text, return_tensors="pt", padding=True)
        embed_dim = int(get_image_features(backbone, dummy["pixel_values"].to(device)).shape[-1])

    train_ds = ImageTextObjectDataset(train_records, image_root)
    val_ds = ImageTextObjectDataset(val_records, image_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate_image_text(batch, processor))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_image_text(batch, processor))

    model = ImageTextObjectClassifier(backbone, embed_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        print(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
            }
        )
        if val_metrics["top1"] >= best_val:
            best_val = val_metrics["top1"]
            best_state = {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()}

    payload = {
        "model_name": args.model_name,
        "embed_dim": embed_dim,
        "classifier_state_dict": best_state,
        "num_classes": 3,
        "label_order": ["whole_milk", "oat_milk", "soy_milk"],
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "best_val_top1": best_val,
    }

    output_path = os.path.expanduser(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, output_path)
    print("saved deployable checkpoint:", output_path)

    del model, backbone, processor, train_loader, val_loader, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
