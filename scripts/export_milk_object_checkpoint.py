#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a trained object-classifier state dict into a deployable checkpoint.

Expected input is the image+text CLIP object-classifier head from the final
spatial-object notebook.
"""

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="Export deployable milk object-classifier checkpoint.")
    parser.add_argument("--classifier-state", required=True, help="Path to torch state_dict or payload containing classifier weights")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="HF backbone name")
    parser.add_argument("--embed-dim", type=int, default=512, help="Backbone embedding dimension")
    args = parser.parse_args()

    classifier_state_path = os.path.expanduser(args.classifier_state)
    output_path = os.path.expanduser(args.output)
    payload = torch.load(classifier_state_path, map_location="cpu")

    if isinstance(payload, dict) and "classifier_state_dict" in payload:
        classifier_state = payload["classifier_state_dict"]
    elif isinstance(payload, dict):
        classifier_state = payload
    else:
        raise RuntimeError("unsupported payload in {}".format(classifier_state_path))

    export_payload = {
        "model_name": str(args.model_name),
        "embed_dim": int(args.embed_dim),
        "classifier_state_dict": classifier_state,
        "num_classes": 3,
        "label_order": ["whole_milk", "oat_milk", "soy_milk"],
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(export_payload, output_path)
    print("wrote deployable checkpoint:", output_path)


if __name__ == "__main__":
    main()
