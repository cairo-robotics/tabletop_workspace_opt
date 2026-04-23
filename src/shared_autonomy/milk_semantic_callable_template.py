#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Template callable for plugging a trained semantic scorer into ROS.

Replace the body of `predict_object_scores` with your exported notebook/runtime
inference code. The return value must be a dict over:
  - whole_milk
  - oat_milk
  - soy_milk
"""


def predict_object_scores(instruction, image_path=None, image_bgr=None, object_order=None):
    """Return raw or normalized object scores.

    Parameters
    ----------
    instruction : str
        Natural language instruction for the current trial.
    image_path : str or None
        Optional path to a scene image on disk.
    image_bgr : np.ndarray or None
        Optional live BGR image captured from a ROS image topic.
    object_order : list[str] or None
        Canonical object label order requested by the caller.
    """
    raise NotImplementedError(
        "Replace milk_semantic_callable_template.predict_object_scores with your exported trained-model inference."
    )
