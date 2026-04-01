import math
from ultralytics import YOLO
import torch


class YOLODetector:
    """Lightweight detector using only YOLO (no FastSAM)."""

    def __init__(self, yolo_weight_path, yolo_conf_threshold=0.5):
        print(f"Loading YOLO model from: {yolo_weight_path}")
        self.yolo_model = YOLO(yolo_weight_path)
        self.yolo_conf_threshold = yolo_conf_threshold
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print("YOLO model loaded.")

    def perform_yolo_inference(self, frame, confidence_threshold=None):
        """Run YOLO inference on a single frame.

        Returns:
            detections: list of dicts with 'bounding_box', 'confidence', 'class_name'
            predicted_boxes: list of [x1, y1, x2, y2]
        """
        if confidence_threshold is None:
            confidence_threshold = self.yolo_conf_threshold

        results = self.yolo_model(frame, stream=True, verbose=False)
        detections = []
        predicted_boxes = []

        for r in results:
            for box in r.boxes:
                if box.conf[0] >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = self.yolo_model.names[cls]

                    detections.append({
                        "bounding_box": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "class_name": class_name,
                    })
                    predicted_boxes.append([x1, y1, x2, y2])

        return detections, predicted_boxes
