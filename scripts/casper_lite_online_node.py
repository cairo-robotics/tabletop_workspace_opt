#!/usr/bin/env python3
"""Observe-only online CASPER-lite VLM intent predictor.

This node mirrors the CASPER Section 3.2 pattern at runtime: collect a short
teleoperation observation history, render lightweight visual prompts, run a VLM
in a background thread, and publish the inferred intent. It does not command the
robot or modify the existing intent distribution.
"""

import json
import math
import os
import subprocess
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2
import rospy
import yaml
from cv_bridge import CvBridge
from intera_core_msgs.msg import EndpointState
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32MultiArray, String
from vision_msgs.msg import Detection2DArray

from casper_lite_evaluator import build_prompt, parse_model_response
from render_casper_visual_prompts import render_row, render_semantic_map


def _parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class CasperLiteOnlineNode(object):
    def __init__(self):
        rospy.init_node("casper_lite_online_node")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_frame_dir = os.path.join(package_root, "logs", "casper_online_frames")
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.end_effector_topic = str(
            rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        ).strip()
        self.candidates_topic = str(rospy.get_param("~candidates_topic", "/hybrid_grasp_registry/detections")).strip()
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.trial_context_topic = str(rospy.get_param("~trial_context_topic", "/user_study/trial_context")).strip()
        self.study_event_topic = str(rospy.get_param("~study_event_topic", "/user_study/events")).strip()
        self.execution_state_topic = str(
            rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")
        ).strip()
        self.top_goal_topic = str(rospy.get_param("~top_goal_topic", "/apriltag_intent_inference/top_goal")).strip()
        self.top_probability_topic = str(
            rospy.get_param("~top_probability_topic", "/apriltag_intent_inference/top_probability")
        ).strip()
        self.distribution_topic = str(rospy.get_param("~distribution_topic", "/apriltag_intent_inference/distribution")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.manual_label_state_topic = str(rospy.get_param("~manual_label_state_topic", "/sam_manual_labeler/state")).strip()
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param("~object_map_yaml", os.path.join(package_root, "config", "apriltag_object_map.yaml"))
        )
        self.frame_dir = os.path.expanduser(rospy.get_param("~frame_dir", default_frame_dir))
        default_log_dir = os.path.dirname(self.frame_dir)
        self.log_dir = os.path.expanduser(rospy.get_param("~log_dir", default_log_dir))
        self.log_predictions = _parse_bool(rospy.get_param("~log_predictions", True))
        self.predict_command = str(
            rospy.get_param(
                "~predict_command",
                "python3 scripts/casper_lite_vlm_predict_one.py --provider openai --model gpt-4o-mini --image-detail low",
            )
        )
        self.timeout_sec = float(rospy.get_param("~timeout_sec", 12.0))
        self.min_trigger_interval_sec = float(rospy.get_param("~min_trigger_interval_sec", 2.0))
        self.min_displacement_m = float(rospy.get_param("~min_displacement_m", 0.02))
        self.trajectory_history_sec = float(rospy.get_param("~trajectory_history_sec", 3.0))
        self.trajectory_history_max_points = int(rospy.get_param("~trajectory_history_max_points", 20))
        self.save_image_width = int(rospy.get_param("~save_image_width", 384))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 70))
        self.self_consistency_k = int(rospy.get_param("~self_consistency_k", 1))
        self.agreement_threshold = int(rospy.get_param("~agreement_threshold", 1))
        self.confidence_threshold = float(rospy.get_param("~confidence_threshold", 0.0))
        self.prompt_style = str(rospy.get_param("~prompt_style", "action_candidates")).strip().lower()
        self.prompt_geometry_mode = str(rospy.get_param("~prompt_geometry_mode", "relative_only")).strip().lower()
        self.image_history_count = int(rospy.get_param("~image_history_count", 3))
        self.task_history_count = int(rospy.get_param("~task_history_count", 6))
        self.enable_semantic_map = _parse_bool(rospy.get_param("~enable_semantic_map", True))
        self.semantic_map_size = int(rospy.get_param("~semantic_map_size", 512))
        self.visual_label_mode = str(rospy.get_param("~visual_label_mode", "id")).strip().lower()
        self.semantic_label_mode = str(rospy.get_param("~semantic_label_mode", "id")).strip().lower()
        self.wrist_candidate_marks = _parse_bool(rospy.get_param("~wrist_candidate_marks", False))
        self.semantic_region_mode = str(rospy.get_param("~semantic_region_mode", "ellipse")).strip().lower()
        self.trigger_on_top_goal_change = _parse_bool(rospy.get_param("~trigger_on_top_goal_change", True))
        self.trigger_on_probability_crossing = _parse_bool(rospy.get_param("~trigger_on_probability_crossing", True))
        self.trigger_on_selection = _parse_bool(rospy.get_param("~trigger_on_selection", True))
        self.trigger_on_timer = _parse_bool(rospy.get_param("~trigger_on_timer", False))
        self.timer_period_sec = float(rospy.get_param("~timer_period_sec", 3.0))
        self.status_period_sec = float(rospy.get_param("~status_period_sec", 2.0))
        self.post_reset_cooldown_sec = float(rospy.get_param("~post_reset_cooldown_sec", 3.0))
        self.probability_crossing_threshold = float(rospy.get_param("~probability_crossing_threshold", 0.6))

        os.makedirs(self.frame_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_handle = None
        if self.log_predictions:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(self.log_dir, "casper_online_predictions_{}.jsonl".format(stamp))
            self.log_handle = open(self.log_path, "a", encoding="utf-8")
        else:
            self.log_path = ""
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.condition = threading.Condition()
        self.label_to_meta, self.tag_id_to_meta = self._load_object_map()

        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_image_frame = ""
        self.ee_history = deque()
        self.observation_history = deque(maxlen=max(1, self.image_history_count))
        self.task_history = deque(maxlen=max(1, self.task_history_count))
        self.current_context = {}
        self.current_trial_key = ""
        self.candidate_labels = []
        self.candidate_poses = {}
        self.candidate_uv = {}
        self.allowed_tag_ids = set()
        self.last_distribution = []
        self.top_goal_label = ""
        self.top_probability = 0.0
        self.selected_grasp_label = ""
        self.execution_state = ""
        self.control_phase = "idle"
        self.pending_snapshot = None
        self.worker_busy = False
        self.last_trigger_at = 0.0
        self.last_status_reason = "starting"
        self.reset_generation = 0
        self.suppress_until_sec = 0.0

        self.prediction_pub = rospy.Publisher("/casper_lite/prediction", String, queue_size=10)
        self.predicted_intent_pub = rospy.Publisher("/casper_lite/predicted_intent", String, queue_size=10)
        self.confidence_pub = rospy.Publisher("/casper_lite/confidence", Float32, queue_size=10)
        self.agreement_pub = rospy.Publisher("/casper_lite/agrees_with_online", Bool, queue_size=10)
        self.status_pub = rospy.Publisher("/casper_lite/status", String, queue_size=10)

        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)
        rospy.Subscriber(self.end_effector_topic, EndpointState, self._ee_cb, queue_size=50)
        rospy.Subscriber(self.candidates_topic, Detection2DArray, self._candidates_cb, queue_size=1)
        rospy.Subscriber(self.allowed_ids_topic, Int32MultiArray, self._allowed_ids_cb, queue_size=1)
        rospy.Subscriber(self.trial_context_topic, String, self._trial_context_cb, queue_size=10)
        rospy.Subscriber(self.study_event_topic, String, self._study_event_cb, queue_size=50)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=10)
        rospy.Subscriber(self.top_goal_topic, String, self._top_goal_cb, queue_size=10)
        rospy.Subscriber(self.top_probability_topic, Float32, self._top_probability_cb, queue_size=10)
        rospy.Subscriber(self.distribution_topic, Float32MultiArray, self._distribution_cb, queue_size=10)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_label_cb, queue_size=10)
        rospy.Subscriber(self.manual_label_state_topic, String, self._manual_label_state_cb, queue_size=10)
        rospy.on_shutdown(self._shutdown)

        self.worker = threading.Thread(target=self._worker_loop)
        self.worker.daemon = True
        self.worker.start()
        rospy.Timer(rospy.Duration(max(0.5, self.status_period_sec)), self._status_timer_cb)
        if self.trigger_on_timer:
            rospy.Timer(rospy.Duration(max(0.2, self.timer_period_sec)), lambda _event: self._trigger("timer"))
        rospy.loginfo("[casper_lite_online] observe-only VLM node started prediction_log=%s", self.log_path or "disabled")

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}, {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        entries = raw.get("tag_objects") or raw.get("candidate_objects") or {}
        label_map = {}
        tag_id_map = {}
        for key, meta in entries.items():
            if not isinstance(meta, dict):
                continue
            tag = str(key).strip()
            meta_with_id = dict(meta)
            meta_with_id["candidate_id"] = tag
            tag_id_map[tag] = meta_with_id
            label = str(meta.get("grasp_complete_label") or "").strip()
            if label:
                label_map[label] = meta_with_id
        return label_map, tag_id_map

    def _candidate_id_for_grasp_label(self, label):
        label = str(label or "").strip()
        if not label:
            return ""
        if label in self.tag_id_to_meta:
            return label
        meta = self.label_to_meta.get(label) or {}
        return str(meta.get("candidate_id") or "").strip()

    def _meta_for_candidate_id(self, candidate_id):
        return self.tag_id_to_meta.get(str(candidate_id or "").strip(), {}) or {}

    def _meta_for_grasp_label(self, label):
        label = str(label or "").strip()
        if not label:
            return {}
        if label in self.tag_id_to_meta:
            return self.tag_id_to_meta.get(label, {}) or {}
        return self.label_to_meta.get(label, {}) or {}

    def _skill_for_event_locked(self, event_name, meta):
        event_name = str(event_name or "").strip().lower()
        step = str((self.current_context or {}).get("active_step_id") or "").strip().lower()
        task = str((meta or {}).get("task") or "").strip().lower()
        category = str((meta or {}).get("category") or "").strip().lower()
        if event_name in ("release_complete",) or "place" in step or task == "place" or category == "destination":
            return "Place"
        if "pour" in step or task == "pour" or event_name == "pour_complete":
            return "Pour"
        return "Pick"

    def _append_task_history_locked(self, event_name, grasp_label="", status=""):
        event_name = str(event_name or "").strip().lower()
        grasp_label = str(grasp_label or "").strip()
        candidate_id = self._candidate_id_for_grasp_label(grasp_label)
        meta = self._meta_for_grasp_label(grasp_label) if grasp_label else self._meta_for_candidate_id(candidate_id)
        if not candidate_id and not event_name:
            return
        status = str(status or "").strip()
        if not status:
            if event_name in ("auto_complete", "grasp_complete", "release_complete", "pour_complete"):
                status = "completed"
            elif event_name in ("confirm_cancel", "casper_confirm_reject"):
                status = "cancelled"
            elif event_name in ("auto_start", "pour_start"):
                status = "started"
            else:
                status = "selected"
        self.task_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "event": event_name,
                "status": status,
                "skill": self._skill_for_event_locked(event_name, meta),
                "candidate_id": candidate_id,
                "object_name": str((meta or {}).get("object_name") or "").strip(),
                "category": str((meta or {}).get("category") or "").strip(),
                "grasp_label": grasp_label,
                "task_phase": str((self.current_context or {}).get("active_step_id") or "").strip(),
            }
        )

    def _trial_key(self, context):
        if not context:
            return ""
        return "::".join(
            str(context.get(key) or "").strip()
            for key in ("session_id", "participant_id", "condition_id", "block_id", "active_task_id", "active_step_id")
        )

    def _image_cb(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[casper_lite_online] image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time(0) else rospy.Time.now().to_sec()
            self.latest_image_frame = str(msg.header.frame_id or "")

    def _ee_cb(self, msg):
        now_sec = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time(0) else rospy.Time.now().to_sec()
        pose = msg.pose
        sample = {
            "stamp": now_sec,
            "ee_position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
        }
        with self.lock:
            if self.control_phase != "teleop":
                return
            if time.time() < self.suppress_until_sec:
                self.ee_history.clear()
                return
            self.ee_history.append(sample)
            cutoff = now_sec - max(0.1, self.trajectory_history_sec)
            while self.ee_history and float(self.ee_history[0].get("stamp", 0.0)) < cutoff:
                self.ee_history.popleft()

    def _candidates_cb(self, msg):
        labels = []
        poses = {}
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            label = str(int(hyp.id))
            labels.append(label)
            p = hyp.pose.pose.position
            poses[label] = {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
        with self.lock:
            self.candidate_labels = sorted(set(labels), key=lambda value: int(value) if value.lstrip("-").isdigit() else value)
            self.candidate_poses = poses

    def _manual_label_state_cb(self, msg):
        try:
            payload = json.loads(str(msg.data))
        except Exception:
            return
        uv_by_id = {}
        for item in payload.get("labeled_objects") or []:
            try:
                candidate_id = str(int(item.get("candidate_id")))
                center_uv = item.get("center_uv") or []
                if len(center_uv) >= 2:
                    uv_by_id[candidate_id] = [float(center_uv[0]), float(center_uv[1])]
            except Exception:
                continue
        with self.lock:
            self.candidate_uv.update(uv_by_id)

    def _allowed_ids_cb(self, msg):
        with self.lock:
            self.allowed_tag_ids = set(str(int(v)) for v in list(msg.data))

    def _trial_context_cb(self, msg):
        try:
            context = json.loads(str(msg.data))
        except Exception:
            context = {}
        should_reset = False
        with self.lock:
            next_key = self._trial_key(context)
            if next_key != self.current_trial_key:
                self.ee_history.clear()
                self.observation_history.clear()
                self.task_history.clear()
                self.pending_snapshot = None
                self.reset_generation += 1
                self.suppress_until_sec = time.time() + max(0.0, self.post_reset_cooldown_sec)
                self.control_phase = "teleop" if next_key else "idle"
                self.last_status_reason = "context_changed"
                should_reset = True
            self.current_context = context
            self.current_trial_key = next_key
        if should_reset:
            self._publish_reset_prediction("context_changed")

    def _study_event_cb(self, msg):
        try:
            event = json.loads(str(msg.data))
        except Exception:
            event = {}
        name = str(event.get("event") or "").strip().lower()
        if not name:
            return
        if name in ("auto_start", "pour_start"):
            with self.lock:
                grasp_label = str(event.get("grasp_id") or event.get("selected_grasp_label") or "").strip()
                self._append_task_history_locked(name, grasp_label=grasp_label, status="started")
                self.control_phase = "autonomous"
            return
        if name in ("auto_complete", "grasp_complete", "release_complete", "pour_complete", "confirm_cancel"):
            with self.lock:
                grasp_label = str(event.get("grasp_id") or event.get("selected_grasp_label") or "").strip()
                self._append_task_history_locked(name, grasp_label=grasp_label)
                self.control_phase = "teleop" if self.current_trial_key else "idle"
                self.ee_history.clear()
                self.observation_history.clear()
                self.pending_snapshot = None
                self.reset_generation += 1
                self.suppress_until_sec = time.time() + max(0.0, self.post_reset_cooldown_sec)
                self.last_status_reason = "{}_reset_history".format(name)
            self._publish_reset_prediction("{}_reset_history".format(name))
            return
        if self.trigger_on_selection and name in (
            "auto_select_pregrasp",
            "joystick_event_select",
            "joystick_retarget_select",
            "confirm_accept",
        ):
            self._trigger(name)

    def _execution_state_cb(self, msg):
        state = str(msg.data).strip().lower()
        with self.lock:
            self.execution_state = state
            if state.startswith("exec_") or state in ("retreat_before_pregrasp",):
                self.control_phase = "autonomous"
            elif state in ("", "idle", "done", "wait_pregrasp_confirm") and self.current_trial_key:
                self.control_phase = "teleop"

    def _publish_reset_prediction(self, reason):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "reset": True,
            "predicted_candidate_id": "",
            "model_confidence": 0.0,
            "self_consistency_confidence": 0.0,
            "confident": False,
            "failure_reason": str(reason),
        }
        self.prediction_pub.publish(json.dumps(payload, sort_keys=True))
        self.status_pub.publish(String("prediction_reset reason={}".format(reason)))

    def _top_goal_cb(self, msg):
        with self.lock:
            next_goal = str(msg.data).strip()
            changed = bool(next_goal and next_goal != self.top_goal_label)
            self.top_goal_label = next_goal
        if changed and self.trigger_on_top_goal_change:
            self._trigger("top_goal_change")

    def _top_probability_cb(self, msg):
        should_trigger = False
        with self.lock:
            previous = float(self.top_probability)
            self.top_probability = float(msg.data)
            should_trigger = (
                self.trigger_on_probability_crossing
                and previous < self.probability_crossing_threshold
                and self.top_probability >= self.probability_crossing_threshold
            )
        if should_trigger:
            self._trigger("probability_crossing")

    def _distribution_cb(self, msg):
        with self.lock:
            self.last_distribution = [float(v) for v in list(msg.data)]

    def _selected_label_cb(self, msg):
        with self.lock:
            self.selected_grasp_label = str(msg.data).strip()

    def _subsample_ee_history_locked(self):
        points = list(self.ee_history)
        max_points = max(2, int(self.trajectory_history_max_points))
        if len(points) <= max_points:
            return points
        stride = float(len(points) - 1) / float(max_points - 1)
        return [points[int(round(i * stride))] for i in range(max_points)]

    def _trajectory_summary(self, points):
        if len(points) < 2:
            return {"available": False, "num_points": len(points)}
        start = points[0]["ee_position"]
        end = points[-1]["ee_position"]
        delta = {axis: float(end[axis]) - float(start[axis]) for axis in ("x", "y", "z")}
        displacement = math.sqrt(sum(delta[axis] * delta[axis] for axis in ("x", "y", "z")))
        duration = max(0.0, float(points[-1]["stamp"]) - float(points[0]["stamp"]))
        return {
            "available": True,
            "num_points": len(points),
            "duration_sec": duration,
            "start_xyz": start,
            "end_xyz": end,
            "delta_xyz": delta,
            "displacement_m": displacement,
        }

    def _candidate_rows_locked(self, image_scale=1.0):
        labels = list(self.candidate_labels)
        if self.allowed_tag_ids:
            filtered = [label for label in labels if label in self.allowed_tag_ids]
            if filtered:
                labels = filtered
        gripper_xy = None
        if self.ee_history:
            latest = self.ee_history[-1].get("ee_position") or {}
            if "x" in latest and "y" in latest:
                gripper_xy = (float(latest["x"]), float(latest["y"]))
        rows = []
        for label in labels:
            meta = self.tag_id_to_meta.get(label, {})
            category = str(meta.get("category") or "").strip()
            task = str(meta.get("task") or "").strip()
            if not task:
                task = "place" if category == "destination" else "pickup"
            position = self.candidate_poses.get(label, {})
            relative_to_gripper = {}
            if gripper_xy is not None and "x" in position and "y" in position:
                dx = float(position["x"]) - float(gripper_xy[0])
                dy = float(position["y"]) - float(gripper_xy[1])
                distance_xy = math.sqrt(dx * dx + dy * dy)
                relative_to_gripper = {
                    "distance_xy_m": distance_xy,
                    "direction_xy": [dx, dy],
                }
            rows.append(
                {
                    "candidate_id": label,
                    "label": label,
                    "object_name": str(meta.get("object_name") or "").strip(),
                    "category": category,
                    "task_type": str((self.current_context or {}).get("active_step_id") or "").strip(),
                    "task_suitability": task,
                    "grasp_complete_label": str(meta.get("grasp_complete_label") or "").strip(),
                    "position": position,
                    "relative_to_gripper": relative_to_gripper,
                    "center_uv": [float(v) * float(image_scale) for v in self.candidate_uv.get(label, [])],
                }
            )
        rows.sort(
            key=lambda row: (
                float((row.get("relative_to_gripper") or {}).get("distance_xy_m", 1e9)),
                int(row.get("candidate_id")) if str(row.get("candidate_id") or "").lstrip("-").isdigit() else 999999,
            )
        )
        for idx, row in enumerate(rows, start=1):
            rel = row.get("relative_to_gripper") or {}
            if rel:
                rel["rank_by_distance"] = idx
        return rows

    def _resized_image_locked(self):
        if self.latest_image is None:
            return None, 1.0
        image = self.latest_image
        width = int(self.save_image_width)
        if width <= 0 or image.shape[1] <= width:
            return image.copy(), 1.0
        scale = float(width) / float(image.shape[1])
        height = max(1, int(round(float(image.shape[0]) * scale)))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), scale

    def _trigger(self, reason):
        now = time.time()
        with self.lock:
            if self.worker_busy:
                self.last_status_reason = "worker_busy"
                return False
            suppress_left = self.suppress_until_sec - now
            if suppress_left > 0.0:
                self.ee_history.clear()
                self.last_status_reason = "post_reset_cooldown {:.1f}s".format(suppress_left)
                return False
            if self.pending_snapshot is not None:
                self.last_status_reason = "prediction_pending"
                return False
            if self.latest_image is None:
                self.last_status_reason = "no_image"
                return False
            if not self.candidate_labels:
                self.last_status_reason = "no_candidates"
                return False
            if (now - self.last_trigger_at) < self.min_trigger_interval_sec:
                self.last_status_reason = "trigger_interval"
                return False
            trajectory = self._subsample_ee_history_locked()
            summary = self._trajectory_summary(trajectory)
            if float(summary.get("displacement_m") or 0.0) < self.min_displacement_m:
                self.last_status_reason = "displacement_low {:.3f}m min={:.3f}m".format(
                    float(summary.get("displacement_m") or 0.0), self.min_displacement_m
                )
                return False
            image, image_scale = self._resized_image_locked()
            if image is None:
                self.last_status_reason = "image_resize_failed"
                return False
            self.last_trigger_at = now
            self.last_status_reason = "queued {}".format(reason)
            self.pending_snapshot = {
                "reason": str(reason),
                "timestamp": datetime.now().isoformat(),
                "stamp_sec": rospy.Time.now().to_sec(),
                "image": image,
                "image_stamp_sec": self.latest_image_stamp,
                "image_frame": self.latest_image_frame,
                "trial_context": dict(self.current_context),
                "trial_key": self.current_trial_key,
                "candidates": self._candidate_rows_locked(image_scale=image_scale),
                "allowed_tag_ids": sorted(self.allowed_tag_ids),
                "top_goal_label": self.top_goal_label,
                "top_probability": self.top_probability,
                "distribution": list(self.last_distribution),
                "selected_grasp_label": self.selected_grasp_label,
                "selected_candidate_id": self._candidate_id_for_grasp_label(self.selected_grasp_label),
                "execution_state": self.execution_state,
                "control_phase": self.control_phase,
                "trajectory_history": trajectory,
                "trajectory_summary": summary,
                "task_history": list(self.task_history),
                "_reset_generation": self.reset_generation,
            }
        with self.condition:
            self.condition.notify()
        return True

    def _status_timer_cb(self, _event):
        with self.lock:
            if self.worker_busy:
                reason = "worker_busy"
            elif self.pending_snapshot is not None:
                reason = "prediction_pending"
            elif self.latest_image is None:
                reason = "no_image"
            elif not self.candidate_labels:
                reason = "no_candidates"
            else:
                suppress_left = self.suppress_until_sec - time.time()
                if suppress_left > 0.0:
                    self.ee_history.clear()
                    reason = "post_reset_cooldown {:.1f}s".format(suppress_left)
                else:
                    trajectory = self._subsample_ee_history_locked()
                    summary = self._trajectory_summary(trajectory)
                    displacement = float(summary.get("displacement_m") or 0.0)
                    if displacement < self.min_displacement_m:
                        reason = "displacement_low {:.3f}m min={:.3f}m".format(displacement, self.min_displacement_m)
                    elif not self.trigger_on_timer:
                        reason = "ready_waiting_for_event timer_disabled"
                    else:
                        reason = self.last_status_reason or "ready"
            self.last_status_reason = reason
        self.status_pub.publish(String("waiting_for_trigger reason={}".format(reason)))

    def _write_visual_prompt(self, snapshot):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = os.path.join(self.frame_dir, "{}_{}_raw.jpg".format(stamp, snapshot["reason"]))
        cv2.imwrite(raw_path, snapshot["image"], [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        row = dict(snapshot)
        row.pop("image", None)
        row["image_path"] = raw_path
        rendered_path = render_row(
            row,
            self.frame_dir,
            label_mode=self.visual_label_mode,
            draw_candidates=self.wrist_candidate_marks,
        )
        if rendered_path:
            row["original_image_path"] = raw_path
            row["image_path"] = rendered_path
            row["visual_prompting"] = {
                "type": "casper_lite_overlay",
                "candidate_marks": bool(self.wrist_candidate_marks),
                "gripper_end_marker": True,
                "trajectory_arrows": True,
                "projection": "table_xy_affine_to_image",
                "uses_center_uv_when_available": True,
            }
        if self.enable_semantic_map:
            semantic_map_path = render_semantic_map(
                row,
                self.frame_dir,
                size_px=self.semantic_map_size,
                label_mode=self.semantic_label_mode,
                region_mode=self.semantic_region_mode,
            )
            if semantic_map_path:
                row["semantic_map_path"] = semantic_map_path
                row.setdefault("visual_prompting", {})
                row["visual_prompting"]["semantic_topdown_map"] = True
                row["visual_prompting"]["semantic_region_mode"] = self.semantic_region_mode
        return row

    def _episode_from_row(self, row, image_history_paths=None):
        context = row.get("trial_context") or {}
        task_id = str(context.get("active_task_id") or "casper_observation")
        step_id = str(context.get("active_step_id") or "")
        instruction = self._english_instruction_for_step(step_id)
        return {
            "episode_id": "{}_{}".format(row.get("trial_key") or "online", row.get("reason") or "trigger"),
            "trial_key": str(row.get("trial_key") or ""),
            "save_reason": str(row.get("reason") or ""),
            "scene_id": task_id,
            "view_id": str(row.get("reason") or "online"),
            "image_path": str(row.get("image_path") or ""),
            "semantic_map_path": str(row.get("semantic_map_path") or ""),
            "image_history_paths": list(image_history_paths or []),
            "prompt_style": self.prompt_style,
            "prompt_geometry_mode": self.prompt_geometry_mode,
            "instruction": instruction,
            "task_type": step_id,
            "correct_candidate_id": "",
            "slot_assignment": {},
            "trajectory_history": row.get("trajectory_history") or [],
            "trajectory_summary": row.get("trajectory_summary") or {},
            "task_history": row.get("task_history") or [],
            "candidates": row.get("candidates") or [],
        }

    def _english_instruction_for_step(self, step_id):
        step = str(step_id or "").strip().lower()
        if step == "select_sandwich_item":
            return "Infer which sandwich ingredient the user intends to pick."
        if step == "place_sandwich_item":
            return "Infer where the user intends to place the held sandwich ingredient."
        if "place" in step:
            return "Infer which destination the user intends to place the held object on."
        if "select" in step or "pick" in step or "grasp" in step:
            return "Infer which object the user intends to pick."
        return "Infer the user's intended target from the visual observation and teleoperation trajectory."

    def _run_prediction(self, episode, prompt):
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("LC_ALL", "C.UTF-8")
        env.setdefault("LANG", "C.UTF-8")
        proc = subprocess.run(
            self.predict_command,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
            env=env,
            shell=True,
            timeout=self.timeout_sec,
        )
        valid_ids = [str(row.get("candidate_id") or "") for row in episode["candidates"]]
        valid_ids.append("no_intent")
        parsed = parse_model_response(proc.stdout, valid_ids)
        if str(parsed.get("intent_id") or "").strip() == "no_intent":
            parsed["intent_id"] = ""
            parsed["reason"] = "no_intent: {}".format(parsed.get("reason") or "insufficient evidence")
        parsed["returncode"] = int(proc.returncode)
        parsed["stderr"] = proc.stderr.strip()
        parsed["stdout"] = proc.stdout.strip()
        if proc.returncode != 0 and not parsed.get("intent_id"):
            parsed["reason"] = parsed.get("reason") or "prediction_failed rc={} stderr={}".format(proc.returncode, proc.stderr.strip())
        return parsed

    def _run_prediction_with_latency(self, episode, prompt):
        start = time.time()
        try:
            pred = self._run_prediction(episode, prompt)
        except Exception as exc:
            pred = {
                "intent_id": "",
                "confidence": 0.0,
                "reason": "prediction_failed: {}".format(exc),
                "raw_response": "",
                "returncode": -1,
                "stderr": str(exc),
                "stdout": "",
            }
        pred["latency_sec"] = time.time() - start
        return pred

    def _run_self_consistency_predictions(self, episode, prompt):
        k = max(1, int(self.self_consistency_k))
        batch_start = time.time()
        if k == 1:
            predictions = [self._run_prediction_with_latency(episode, prompt)]
        else:
            predictions = []
            with ThreadPoolExecutor(max_workers=k) as pool:
                futures = [pool.submit(self._run_prediction_with_latency, episode, prompt) for _ in range(k)]
                for future in as_completed(futures):
                    predictions.append(future.result())
        batch_latency = time.time() - batch_start
        latencies = [float(pred.get("latency_sec") or 0.0) for pred in predictions]
        return predictions, latencies, batch_latency

    def _worker_loop(self):
        while not rospy.is_shutdown():
            with self.condition:
                self.condition.wait(0.2)
            with self.lock:
                snapshot = self.pending_snapshot
                self.pending_snapshot = None
                if snapshot is None:
                    continue
                self.worker_busy = True
            try:
                row = self._write_visual_prompt(snapshot)
                with self.lock:
                    self.observation_history.append(row)
                    image_history_paths = [
                        str(item.get("image_path") or "")
                        for item in list(self.observation_history)[-max(1, self.image_history_count):]
                        if str(item.get("image_path") or "").strip()
                    ]
                    semantic_map_path = str(row.get("semantic_map_path") or "").strip()
                    if semantic_map_path:
                        image_history_paths.append(semantic_map_path)
                episode = self._episode_from_row(row, image_history_paths=image_history_paths)
                prompt = build_prompt(episode)
                predictions, latencies, batch_latency_sec = self._run_self_consistency_predictions(episode, prompt)
                votes = Counter(str(p.get("intent_id") or "") for p in predictions if p.get("intent_id"))
                intent_id, vote_count = ("", 0) if not votes else votes.most_common(1)[0]
                agreement = float(vote_count) / float(max(1, self.self_consistency_k))
                model_confidence = max([float(p.get("confidence") or 0.0) for p in predictions if p.get("intent_id") == intent_id] or [0.0])
                confident = vote_count >= self.agreement_threshold and model_confidence >= self.confidence_threshold
                online_candidate_id = self._candidate_id_for_grasp_label(row.get("top_goal_label"))
                agrees = bool(intent_id and online_candidate_id and intent_id == online_candidate_id)
                failure_reasons = [
                    str(p.get("reason") or "").strip()
                    for p in predictions
                    if not p.get("intent_id") and str(p.get("reason") or "").strip()
                ]
                result = {
                    "timestamp": datetime.now().isoformat(),
                    "trigger_reason": row.get("reason"),
                    "trial_key": row.get("trial_key"),
                    "task_type": episode.get("task_type"),
                    "image_path": row.get("image_path"),
                    "semantic_map_path": row.get("semantic_map_path"),
                    "image_history_paths": image_history_paths,
                    "prompt_style": self.prompt_style,
                    "prompt_geometry_mode": self.prompt_geometry_mode,
                    "prompt": prompt,
                    "candidates": episode.get("candidates") or [],
                    "task_history": episode.get("task_history") or [],
                    "predicted_candidate_id": intent_id,
                    "model_confidence": model_confidence,
                    "self_consistency_agreement": vote_count,
                    "self_consistency_confidence": agreement,
                    "confident": bool(confident),
                    "top_goal_label": row.get("top_goal_label"),
                    "top_goal_candidate_id": online_candidate_id,
                    "top_probability": row.get("top_probability"),
                    "agrees_with_online": agrees,
                    "latency_sec_total": batch_latency_sec,
                    "latency_sec_wall": batch_latency_sec,
                    "latency_sec_api_sum": sum(latencies),
                    "latency_sec_api_max": max(latencies or [0.0]),
                    "prediction_failed": not bool(intent_id),
                    "failure_reason": failure_reasons[0] if failure_reasons else "",
                    "raw_predictions": predictions,
                }
                with self.lock:
                    if int(row.get("_reset_generation", -1)) != int(self.reset_generation):
                        self.last_status_reason = "stale_prediction_dropped"
                        continue
                self._write_prediction_log(result)
                self.prediction_pub.publish(json.dumps(result, sort_keys=True))
                self.predicted_intent_pub.publish(String(intent_id))
                self.confidence_pub.publish(Float32(model_confidence))
                self.agreement_pub.publish(Bool(agrees))
                if not intent_id:
                    self.status_pub.publish(
                        String(
                            "prediction_failed latency={:.3f} reason={}".format(
                                batch_latency_sec,
                                result["failure_reason"] or "no valid candidate_id",
                            )
                        )
                    )
                else:
                    self.status_pub.publish(
                        String("ok confident={} intent={} latency={:.3f}".format(confident, intent_id, batch_latency_sec))
                    )
            except Exception as exc:
                rospy.logwarn("[casper_lite_online] prediction failed: %s", exc)
                self.status_pub.publish(String("prediction_failed: {}".format(exc)))
            finally:
                with self.lock:
                    self.worker_busy = False

    def _write_prediction_log(self, result):
        if self.log_handle is None:
            return
        self.log_handle.write(json.dumps(result, sort_keys=True) + "\n")
        self.log_handle.flush()

    def _shutdown(self):
        if self.log_handle is not None:
            try:
                self.log_handle.flush()
                self.log_handle.close()
            except Exception:
                pass


if __name__ == "__main__":
    CasperLiteOnlineNode()
    rospy.spin()
