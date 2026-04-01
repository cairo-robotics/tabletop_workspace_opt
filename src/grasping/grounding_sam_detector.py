"""
GroundingSAM Detector — GroundingDINO + Segment Anything for text-prompted
zero-shot object detection and segmentation.

Provides the same interface as YOLODetector so it can be swapped in
graspnet_realsense_node.py.

References:
    - /home/yi-shiuan/assistive_plating/grounded_sam.py
    - GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
    - SAM: https://github.com/facebookresearch/segment-anything
"""

import os
import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any

from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert

from segment_anything import sam_model_registry, SamPredictor


# Default paths — weights and configs live in assets/models/ within this package
_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "assets", "models"
)
_DEFAULT_GDINO_CONFIG = os.path.join(_ASSETS_DIR, "GroundingDINO_SwinT_OGC.py")
_DEFAULT_GDINO_WEIGHTS = os.path.join(_ASSETS_DIR, "groundingdino_swint_ogc.pth")
_DEFAULT_SAM_WEIGHTS = os.path.join(_ASSETS_DIR, "sam_vit_h_4b8939.pth")
_DEFAULT_SAM_MODEL_TYPE = "vit_h"


class GroundingSAMDetector:
    """Text-prompted object detector using GroundingDINO + SAM.

    Mirrors the ``YOLODetector`` interface so it can be used as a drop-in
    replacement in ``graspnet_realsense_node.py``.

    Parameters
    ----------
    gdino_config_path : str
        Path to the GroundingDINO config file.
    gdino_weights_path : str
        Path to the GroundingDINO checkpoint.
    sam_weights_path : str
        Path to the SAM checkpoint.
    sam_model_type : str
        SAM model type (``"vit_h"``, ``"vit_l"``, or ``"vit_b"``).
    text_prompt : str
        Default text prompt for detection. Labels separated by `` . ``
        (e.g. ``"cup . bottle . object"``).
    box_threshold : float
        GroundingDINO box confidence threshold.
    text_threshold : float
        GroundingDINO text confidence threshold.
    """

    def __init__(
        self,
        gdino_config_path: str = _DEFAULT_GDINO_CONFIG,
        gdino_weights_path: str = _DEFAULT_GDINO_WEIGHTS,
        sam_weights_path: str = _DEFAULT_SAM_WEIGHTS,
        sam_model_type: str = _DEFAULT_SAM_MODEL_TYPE,
        text_prompt: str = "object",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        print(f"GroundingSAMDetector using device: {self.device}")

        # --- Load GroundingDINO ---
        print(f"Loading GroundingDINO from: {gdino_weights_path}")
        self.gdino_model = load_model(gdino_config_path, gdino_weights_path)
        print("GroundingDINO loaded.")

        # Pre-build the image transform (same as GroundingDINO's default)
        self._transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # --- Load SAM ---
        print(f"Loading SAM ({sam_model_type}) from: {sam_weights_path}")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_weights_path)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        print("SAM loaded.")

    # ------------------------------------------------------------------
    # Public API (mirrors YOLODetector)
    # ------------------------------------------------------------------

    def perform_inference(
        self,
        frame: np.ndarray,
        text_prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[List[int]], List[np.ndarray]]:
        """Run GroundingDINO detection + SAM segmentation on a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).
        text_prompt : str, optional
            Override the default text prompt for this call.
        box_threshold : float, optional
            Override the default box confidence threshold.
        text_threshold : float, optional
            Override the default text confidence threshold.

        Returns
        -------
        detections : list of dict
            Each dict has ``"bounding_box"`` (x1, y1, x2, y2),
            ``"confidence"`` (float), and ``"class_name"`` (str).
            Same format as ``YOLODetector.perform_yolo_inference``.
        predicted_boxes : list of [x1, y1, x2, y2]
            Integer bounding boxes.
        masks : list of np.ndarray
            Boolean masks (H, W) for each detection, from SAM.
        """
        text_prompt = text_prompt or self.text_prompt
        box_threshold = box_threshold if box_threshold is not None else self.box_threshold
        text_threshold = text_threshold if text_threshold is not None else self.text_threshold

        # Convert BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        h, w = frame.shape[:2]

        # --- GroundingDINO detection ---
        image_tensor, _ = self._transform(pil_image, None)

        boxes, logits, phrases = predict(
            model=self.gdino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(boxes) == 0:
            return [], [], []

        # Convert cxcywh (normalized) -> xyxy (pixel)
        boxes_pixel = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes_xyxy = box_convert(
            boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy"
        ).cpu().numpy()

        # Build detections list (YOLODetector-compatible)
        detections = []
        predicted_boxes = []
        for i in range(len(phrases)):
            x1, y1, x2, y2 = (
                int(boxes_xyxy[i][0]),
                int(boxes_xyxy[i][1]),
                int(boxes_xyxy[i][2]),
                int(boxes_xyxy[i][3]),
            )
            detections.append({
                "bounding_box": (x1, y1, x2, y2),
                "confidence": float(logits[i]),
                "class_name": phrases[i],
            })
            predicted_boxes.append([x1, y1, x2, y2])

        # --- SAM segmentation ---
        self.sam_predictor.set_image(rgb)
        masks = []
        for box in boxes_xyxy:
            mask, _, _ = self.sam_predictor.predict(
                box=box, multimask_output=False
            )
            # mask shape: (1, H, W) -> (H, W) boolean
            masks.append(mask[0].astype(bool))

        return detections, predicted_boxes, masks

    def detect_best_object(
        self,
        frame: np.ndarray,
        text_prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """Detect the best (highest-confidence) object and return its mask.

        Convenience method for ``graspnet_realsense_node.py`` which only
        needs the single best detection mask.

        Returns
        -------
        best_detection : dict or None
            The highest-confidence detection, or None if nothing detected.
        best_mask : np.ndarray or None
            Boolean mask (H, W) for the best detection, or None.
        """
        detections, _, masks = self.perform_inference(
            frame,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if not detections:
            return None, None

        # Find the best detection (highest confidence, largest area tiebreaker)
        best_idx = max(
            range(len(detections)),
            key=lambda i: (
                detections[i]["confidence"],
                (detections[i]["bounding_box"][2] - detections[i]["bounding_box"][0])
                * (detections[i]["bounding_box"][3] - detections[i]["bounding_box"][1]),
            ),
        )

        return detections[best_idx], masks[best_idx]
