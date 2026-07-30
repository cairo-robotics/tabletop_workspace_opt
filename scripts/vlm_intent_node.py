#!/usr/bin/env python3
"""CASPER VLM intent-inference node.

Maintains a teleoperation history buffer (static camera frames + EE pose +
gripper state + user command velocity) and, while the coordinator reports
TELEOP state and the user is actually moving, periodically fires K parallel
VLM calls that classify which candidate subtask the user intends. Publishes
the self-consistency vote; the coordinator decides whether to offer.

Subscribed control channel (/casper/inference_ctl, latched String JSON from
the coordinator):
  {"state": "TELEOP", "generation": 3, "current_state": "initial",
   "holding": null, "session_dir": "/path", "rejections": ["Pick up the X"],
   "candidates": [{"letter": "A", "goal_spec": {...},
                   "position_world": [x, y, z]}, ...]}

Published votes (/casper/intent_vote, String JSON):
  {"generation": 3, "winner": "A"|null, "confidence": 0.8,
   "tally": {"A": 4, "UNCLEAR": 1}, "n_parsed": 5, "latency_s": 2.1,
   "annotated_image": "images/0007_annotated.jpg"}
"""
import json
import os
import sys
import threading
import time

import rospy
from cv_bridge import CvBridge
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEVelGoals
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Make shared_autonomy importable when run from source without a build.
_PKG_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")
if _PKG_SRC not in sys.path:
    sys.path.insert(0, _PKG_SRC)

from shared_autonomy.vlm_intent.camera import StaticCameraModel
from shared_autonomy.vlm_intent.history_buffer import (
    HistoryBuffer,
    HistoryRecord,
)
from shared_autonomy.vlm_intent.prompt_builder import build_messages
from shared_autonomy.vlm_intent.vlm_client import (
    VLMClient,
    render_messages_text,
)
from shared_autonomy.vlm_intent.vote import aggregate


class VLMIntentNode:
    def __init__(self):
        cfg = rospy.get_param("~casper", rospy.get_param("casper", {}))
        vlm_cfg = cfg.get("vlm", {})
        self.k = int(cfg.get("self_consistency", {}).get("k", 5))
        self.eta = int(cfg.get("self_consistency", {}).get("eta", 4))
        history_cfg = cfg.get("history", {})
        self.history_sec = float(history_cfg.get("history_sec", 4.5))
        self.n_frames = int(history_cfg.get("n_frames", 6))
        self.arrow_window_s = float(history_cfg.get("arrow_window_s", 1.5))
        self.image_send_mode = history_cfg.get(
            "image_send_mode", "annotated_plus_text")
        gating = cfg.get("gating", {})
        self.min_motion_m = float(gating.get("min_motion_m", 0.03))
        self.min_motion_window_s = float(
            gating.get("min_motion_window_s", 1.5))
        self.retry_sec = float(gating.get("retry_sec", 2.0))
        # Preview/dry-run: build and expose the exact prompt + annotated
        # image, but never query the VLM (no offers). For verifying inputs
        # before a real run. In dry-run the inputs are published at
        # preview_period_s (live rqt view, even at rest) and saved to disk at
        # the slower preview_save_interval_s so the session dir stays tidy.
        self.dry_run = bool(vlm_cfg.get("dry_run", False))
        self.preview_period = float(gating.get("preview_period_s", 0.7))
        self.save_interval_s = float(gating.get("preview_save_interval_s", 2.0))
        # Gripper mask overlay (published by the sim via segmentation render).
        gm = cfg.get("gripper_mask", {})
        self.gripper_mask_enabled = bool(gm.get("enabled", True))
        self.mask_color = tuple(gm.get("color_bgr", [255, 255, 0]))
        self.mask_opacity = float(gm.get("opacity", 0.5))
        self._gripper_mask_topic = gm.get("topic",
                                          "/static_camera/gripper_mask")
        self._latest_mask = None

        self.client = VLMClient(
            base_url=vlm_cfg.get("base_url", "http://localhost:8000/v1"),
            model=vlm_cfg.get("model", ""),
            api_key_env=vlm_cfg.get("api_key_env", "VLM_API_KEY"),
            temperature=float(vlm_cfg.get("temperature", 0.6)),
            max_tokens=int(vlm_cfg.get("max_tokens", 256)),
            request_timeout_s=float(vlm_cfg.get("request_timeout_s", 20.0)),
        )
        self.camera = StaticCameraModel()
        self.buffer = HistoryBuffer(max_seconds=self.history_sec + 6.0)
        self.bridge = CvBridge()

        self._lock = threading.Lock()
        self._ctl = {"state": "IDLE", "generation": -1, "candidates": []}
        self._latest_ee = None
        self._gripper_state = "open"
        self._latest_cmd = (0.0, 0.0, 0.0)
        self._image_count = 0
        self._last_save_t = 0.0

        self.pub_vote = rospy.Publisher("/casper/intent_vote", String,
                                        queue_size=4)
        # Live inspection of exactly what the VLM receives each cycle.
        self.pub_input_image = rospy.Publisher("/casper/vlm_input_image",
                                               Image, queue_size=1)
        self.pub_prompt = rospy.Publisher("/casper/vlm_prompt", String,
                                          queue_size=2)
        rospy.Subscriber("/casper/inference_ctl", String, self._ctl_cb,
                         queue_size=2)
        rospy.Subscriber("/casper/gripper_state", String,
                         self._gripper_cb, queue_size=1)
        rospy.Subscriber("/mujoco_sim/endpoint_state", EndpointState,
                         self._ee_cb, queue_size=1)
        rospy.Subscriber("/teleop/ee_vel_goals", EEVelGoals,
                         self._cmd_cb, queue_size=1)
        rospy.Subscriber("/static_camera/color/image_raw", Image,
                         self._image_cb, queue_size=1, buff_size=2 ** 24)
        if self.gripper_mask_enabled:
            rospy.Subscriber(self._gripper_mask_topic, Image,
                             self._mask_cb, queue_size=1)

        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()
        rospy.loginfo("vlm_intent_node ready (K=%d, eta=%d, model=%s%s)",
                      self.k, self.eta, self.client.model,
                      ", DRY-RUN: inputs only, VLM not queried"
                      if self.dry_run else "")

    # ---- callbacks ---------------------------------------------------------

    def _ctl_cb(self, msg):
        try:
            ctl = json.loads(msg.data)
        except ValueError:
            rospy.logwarn("bad inference_ctl JSON")
            return
        with self._lock:
            self._ctl = ctl

    def _gripper_cb(self, msg):
        self._gripper_state = msg.data

    def _mask_cb(self, msg):
        try:
            self._latest_mask = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="mono8")
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "mask cv_bridge failed: %s", exc)

    def _ee_cb(self, msg):
        p = msg.pose.position
        self._latest_ee = (p.x, p.y, p.z)

    def _cmd_cb(self, msg):
        if msg.ee_vels:
            v = msg.ee_vels[0].linear
            self._latest_cmd = (v.x, v.y, v.z)

    def _image_cb(self, msg):
        if self._latest_ee is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge failed: %s", exc)
            return
        self.buffer.append(HistoryRecord(
            stamp=msg.header.stamp.to_sec(),
            image=image,
            ee_pos=self._latest_ee,
            gripper_state=self._gripper_state,
            cmd_vel=self._latest_cmd,
        ))

    # ---- inference worker --------------------------------------------------

    def _worker_loop(self):
        while not rospy.is_shutdown():
            time.sleep(self.preview_period if self.dry_run else self.retry_sec)
            try:
                self._maybe_infer()
            except Exception as exc:
                rospy.logwarn("inference cycle failed: %r", exc)

    def _maybe_infer(self):
        with self._lock:
            ctl = dict(self._ctl)
        if ctl.get("state") != "TELEOP":
            return
        candidates = ctl.get("candidates") or []
        if not candidates:
            return
        # Gate window: is the user moving enough to be worth inferring?
        motion, _, _ = self.buffer.motion_since(self.min_motion_window_s)
        # Dry-run publishes a live preview even at rest; real inference only
        # fires once the user is actually moving (the arrow needs motion).
        if not self.dry_run and motion < self.min_motion_m:
            return
        # Arrow window: how far back the drawn motion vector reaches (its own
        # knob, independent of the gate window above).
        arrow_motion, arrow_start, arrow_end = \
            self.buffer.motion_since(self.arrow_window_s)
        frames = self.buffer.snapshot(self.n_frames, self.history_sec)
        if not frames:
            return

        marks = [{"mark_id": c["letter"],
                  "position_world": c["position_world"]}
                 for c in candidates if c.get("position_world")]
        arrow = (arrow_start, arrow_end) \
            if arrow_start is not None and arrow_motion > 0.005 else None
        annotated = self.camera.annotate(
            frames[-1].image, marks, arrow,
            gripper_mask=(self._latest_mask
                          if self.gripper_mask_enabled else None),
            mask_color=self.mask_color, mask_opacity=self.mask_opacity)

        letter_pairs = [(c["letter"], c["goal_spec"]) for c in candidates]
        messages = build_messages(
            current_state=ctl.get("current_state", "?"),
            holding=ctl.get("holding"),
            frames=frames,
            annotated_image=annotated,
            candidates=letter_pairs,
            rejected_descriptions=ctl.get("rejections", []),
            image_send_mode=self.image_send_mode,
        )

        # Expose the exact inputs (annotated image + readable prompt) every
        # cycle so they can be inspected live (rqt_image_view / rostopic echo).
        prompt_text = render_messages_text(messages)
        self._publish_inputs(annotated, prompt_text, ctl)

        # Save to the session dir. Every cycle for real inference (paired with
        # a vote); throttled in dry-run so a live preview doesn't flood disk.
        now = time.time()
        do_save = (not self.dry_run
                   or (now - self._last_save_t) >= self.save_interval_s)
        prompt_rel = image_rel = None
        if do_save:
            self._last_save_t = now
            # Save prompt first (shares the index), then the image (which
            # increments), so NNNN_prompt.txt pairs with NNNN_annotated.jpg.
            prompt_rel = self._save_prompt(ctl.get("session_dir"), prompt_text)
            image_rel = self._save_annotated(ctl.get("session_dir"), annotated)

        if self.dry_run:
            rospy.loginfo_throttle(
                2.0, "DRY-RUN preview: publishing /casper/vlm_input_image + "
                "/casper/vlm_prompt (%d candidates, motion=%.3fm) — VLM not "
                "queried", len(candidates), motion)
            if do_save:
                self.pub_vote.publish(String(data=json.dumps({
                    "generation": ctl.get("generation", -1),
                    "dry_run": True, "winner": None,
                    "annotated_image": image_rel, "prompt_file": prompt_rel,
                    "motion_m": round(motion, 4),
                })))
            return

        started = time.time()
        votes = self.client.vote_batch(messages, self.k)
        latency = time.time() - started
        valid_ids = [c["letter"] for c in candidates]
        winner, confidence, tally = aggregate(
            votes, self.k, self.eta, valid_ids=valid_ids)

        self.pub_vote.publish(String(data=json.dumps({
            "generation": ctl.get("generation", -1),
            "winner": winner,
            "confidence": confidence,
            "tally": tally,
            "n_parsed": len(votes),
            "latency_s": round(latency, 2),
            "annotated_image": image_rel,
            "prompt_file": prompt_rel,
            "motion_m": round(motion, 4),
        })))
        rospy.loginfo("intent vote: %s conf=%.2f tally=%s (%.1fs)",
                      winner, confidence, tally, latency)

    def _publish_inputs(self, annotated, prompt_text, ctl):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            img_msg.header.stamp = rospy.Time.now()
            self.pub_input_image.publish(img_msg)
        except Exception as exc:
            rospy.logwarn_throttle(10.0, "input image publish failed: %s", exc)
        self.pub_prompt.publish(String(data=json.dumps({
            "generation": ctl.get("generation", -1),
            "current_state": ctl.get("current_state"),
            "candidates": [c["letter"] + ": " + c["goal_spec"].get("id", "")
                           for c in (ctl.get("candidates") or [])],
            "prompt_text": prompt_text,
        })))

    def _save_prompt(self, session_dir, prompt_text):
        if not session_dir:
            return None
        try:
            prompts_dir = os.path.join(session_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)
            rel = os.path.join("prompts", "%04d_prompt.txt" % self._image_count)
            with open(os.path.join(session_dir, rel), "w") as handle:
                handle.write(prompt_text)
            return rel
        except Exception:
            return None

    def _save_annotated(self, session_dir, annotated):
        if not session_dir:
            return None
        try:
            import cv2
            images_dir = os.path.join(session_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            rel = os.path.join(
                "images", "%04d_annotated.jpg" % self._image_count)
            self._image_count += 1
            cv2.imwrite(os.path.join(session_dir, rel), annotated)
            return rel
        except Exception:
            return None


if __name__ == "__main__":
    rospy.init_node("vlm_intent")
    VLMIntentNode()
    rospy.spin()
