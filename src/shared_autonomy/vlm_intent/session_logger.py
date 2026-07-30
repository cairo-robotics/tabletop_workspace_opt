"""Standalone JSONL session logger for CASPER runs.

One file per session; every record flushed immediately so a crash loses
nothing. Record types: session_meta, inference, offer, decision,
skill_outcome, state_change. Enough to compute intent accuracy,
time-to-offer, and false-offer rate per layout.
"""
import json
import os
import time
from typing import Any, Dict, Optional


class SessionLogger:
    def __init__(self, out_dir: str, scene: str, task_name: str,
                 meta: Optional[Dict[str, Any]] = None,
                 save_images: bool = True):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = "%s_%s_%s" % (scene, task_name, stamp)
        self.directory = os.path.join(os.path.expanduser(out_dir),
                                      self.session_id)
        self.images_dir = os.path.join(self.directory, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.save_images = save_images
        self._path = os.path.join(self.directory, "session.jsonl")
        self._file = open(self._path, "a", buffering=1)
        self._image_count = 0
        self.log("session_meta", scene=scene, task=task_name,
                 **(meta or {}))

    def log(self, record_type: str, **fields) -> None:
        record = {"type": record_type, "t": time.time()}
        record.update(fields)
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def save_image(self, bgr_image, tag: str = "annotated") -> Optional[str]:
        if not self.save_images or bgr_image is None:
            return None
        try:
            import cv2
        except ImportError:
            return None
        rel = os.path.join("images",
                           "%04d_%s.jpg" % (self._image_count, tag))
        self._image_count += 1
        if cv2.imwrite(os.path.join(self.directory, rel), bgr_image):
            return rel
        return None

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
