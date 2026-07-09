#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trial-level logger for the tabletop shared-autonomy user study."""

import json
import math
import os
import threading
from datetime import datetime

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Pose
from intera_core_msgs.msg import EndpointState
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray, String
from vision_msgs.msg import Detection2DArray


def _pose_to_xyz(pose):
    return np.array(
        [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
        dtype=np.float64,
    )


def _distribution_entropy(values):
    probs = np.array(list(values or []), dtype=np.float64)
    if probs.size == 0:
        return None
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return None
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 1e-12:
        return None
    probs = probs / total
    probs = probs[probs > 1e-12]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


class UserStudyTrialLogger:
    def __init__(self):
        rospy.init_node("user_study_trial_logger")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_log_dir = os.path.join(package_root, "logs")
        object_map_default = os.path.join(package_root, "config", "apriltag_object_map.yaml")

        self.study_event_topic = str(rospy.get_param("~study_event_topic", "/user_study/events")).strip()
        self.trial_context_topic = str(rospy.get_param("~trial_context_topic", "/user_study/trial_context")).strip()
        self.top_goal_topic = str(rospy.get_param("~top_goal_topic", "/apriltag_intent_inference/top_goal")).strip()
        self.top_probability_topic = str(
            rospy.get_param("~top_probability_topic", "/apriltag_intent_inference/top_probability")
        ).strip()
        self.distribution_topic = str(
            rospy.get_param("~distribution_topic", "/apriltag_intent_inference/distribution")
        ).strip()
        self.phase_topic = str(rospy.get_param("~phase_topic", "/task_context/phase")).strip()
        self.candidates_topic = str(rospy.get_param("~candidates_topic", "/hybrid_grasp_registry/detections")).strip()
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.execution_state_topic = str(rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.apriltag_registry_status_topic = str(
            rospy.get_param("~apriltag_registry_status_topic", "/apriltag_grasp_registry/status")
        ).strip()
        self.lego_registry_status_topic = str(
            rospy.get_param("~lego_registry_status_topic", "/lego_grasp_registry/status")
        ).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()
        self.end_effector_topic = str(
            rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        ).strip()
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param("~object_map_yaml", object_map_default)
        )
        self.log_dir = os.path.expanduser(rospy.get_param("~log_dir", default_log_dir))
        self.flush_every_write = bool(rospy.get_param("~flush_every_write", True))
        self.active_joystick_threshold = float(rospy.get_param("~active_joystick_threshold", 0.25))
        self.joystick_motion_deadzone = float(rospy.get_param("~joystick_motion_deadzone", 0.12))
        self.joystick_motion_axes = [
            int(value)
            for value in list(rospy.get_param("~joystick_motion_axes", [0, 1, 3, 4]) or [0, 1, 3, 4])
        ]
        self.lock = threading.RLock()
        self.run_trial_counts = {}

        os.makedirs(self.log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, "user_study_trials_{}.jsonl".format(stamp))
        self.event_log_path = os.path.join(self.log_dir, "user_study_trial_events_{}.jsonl".format(stamp))

        self.label_to_meta, self.tag_id_to_meta = self._load_object_map()
        self.current_context = {}
        self.current_phase = ""
        self.top_goal_label = ""
        self.top_probability = 0.0
        self.max_top_probability = 0.0
        self.last_distribution = []
        self.selected_grasp_label = ""
        self.execution_state = ""
        self.apriltag_registry_status = ""
        self.lego_registry_status = ""
        self.current_candidate_labels = []
        self.current_allowed_ids = set()
        self.last_joy_stamp = None
        self.last_joy_effort = 0.0
        self.last_ee_pose = None
        self.trial_counter = 0
        self.current_trial = None

        self.log_handle = open(self.log_path, "a", encoding="utf-8")
        self.event_log_handle = open(self.event_log_path, "a", encoding="utf-8")

        rospy.Subscriber(self.trial_context_topic, String, self._trial_context_cb, queue_size=10)
        rospy.Subscriber(self.study_event_topic, String, self._study_event_cb, queue_size=50)
        rospy.Subscriber(self.top_goal_topic, String, self._top_goal_cb, queue_size=10)
        rospy.Subscriber(self.top_probability_topic, Float32, self._top_probability_cb, queue_size=10)
        rospy.Subscriber(self.distribution_topic, Float32MultiArray, self._distribution_cb, queue_size=10)
        rospy.Subscriber(self.phase_topic, String, self._phase_cb, queue_size=10)
        rospy.Subscriber(self.candidates_topic, Detection2DArray, self._candidates_cb, queue_size=10)
        rospy.Subscriber(self.allowed_ids_topic, Int32MultiArray, self._allowed_ids_cb, queue_size=10)
        rospy.Subscriber(self.execution_state_topic, String, self._execution_state_cb, queue_size=10)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_grasp_label_cb, queue_size=10)
        rospy.Subscriber(self.apriltag_registry_status_topic, String, self._apriltag_registry_status_cb, queue_size=10)
        rospy.Subscriber(self.lego_registry_status_topic, String, self._lego_registry_status_cb, queue_size=10)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=50)
        rospy.Subscriber(self.end_effector_topic, EndpointState, self._ee_cb, queue_size=50)
        rospy.on_shutdown(self._shutdown)

        rospy.loginfo("[user_study_trial_logger] writing trials to %s", self.log_path)

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}, {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        label_map = {}
        tag_id_map = {}
        if isinstance(raw.get("tag_objects"), dict):
            entries = raw.get("tag_objects", {}) or {}
        elif isinstance(raw.get("candidate_objects"), dict):
            entries = raw.get("candidate_objects", {}) or {}
        else:
            entries = {}
        for key, meta in entries.items():
            if not isinstance(meta, dict):
                continue
            try:
                tag_id = int(key)
            except Exception:
                tag_id = None
            label = str(meta.get("grasp_complete_label", "")).strip()
            if label:
                label_map[label] = dict(meta)
            if tag_id is not None:
                tag_id_map[str(tag_id)] = dict(meta)
        return label_map, tag_id_map

    def _normalize_goal_label(self, label):
        value = str(label or "").strip()
        if not value:
            return ""
        if value in self.label_to_meta:
            return value
        meta = self.tag_id_to_meta.get(value, {})
        normalized = str(meta.get("grasp_complete_label", "")).strip()
        return normalized or value

    def _meta_for_label(self, label):
        normalized = self._normalize_goal_label(label)
        return self.label_to_meta.get(normalized, {})

    def _write_jsonl(self, handle, payload):
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.flush_every_write:
            handle.flush()

    def _context_key(self, context):
        if not context:
            return None
        task_id = str(context.get("active_task_id") or "").strip()
        step_id = str(context.get("active_step_id") or "").strip()
        if not step_id:
            return None
        return "{}::{}".format(task_id, step_id)

    def _trial_context_cb(self, msg):
        try:
            context = json.loads(str(msg.data))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[user_study_trial_logger] bad trial context json: %s", exc)
            return

        with self.lock:
            previous_key = self._context_key(self.current_context)
            new_key = self._context_key(context)
            self.current_context = context

            if previous_key == new_key:
                return

            if self.current_trial is not None:
                self._finalize_trial_locked(
                    success=False,
                    failure_reason="context_changed" if new_key else "context_cleared",
                )

            if new_key is not None:
                self._start_trial_locked(context)

    def _run_key(self, context):
        return (
            str(context.get("session_id") or "").strip(),
            str(context.get("participant_id") or "").strip(),
            str(context.get("condition_id") or "").strip(),
            str(context.get("active_task_id") or "").strip(),
        )

    def _start_trial_locked(self, context):
        self.trial_counter += 1
        run_key = self._run_key(context)
        trial_index_within_block = self.run_trial_counts.get(run_key, 0) + 1
        self.run_trial_counts[run_key] = trial_index_within_block
        self.last_joy_stamp = None
        self.last_joy_effort = 0.0
        now = rospy.Time.now().to_sec()
        candidate_labels = list(self.current_candidate_labels)
        allowed_ids = sorted(int(v) for v in self.current_allowed_ids)
        available_allowed = [
            int(label) for label in candidate_labels
            if str(label).lstrip("-").isdigit() and int(label) in self.current_allowed_ids
        ]
        self.current_trial = {
            "trial_id": "trial_{:04d}_{}".format(self.trial_counter, datetime.now().strftime("%H%M%S")),
            "task_id": str(context.get("active_task_id") or ""),
            "task_name": str(context.get("active_task_name") or ""),
            "step_id": str(context.get("active_step_id") or ""),
            "step_index": context.get("active_step_index"),
            "step_title": str(context.get("active_step_title") or ""),
            "step_description": str(context.get("active_step_description") or ""),
            "command": str(context.get("command") or ""),
            "success_event": str(context.get("success_event") or ""),
            "session_id": str(context.get("session_id") or ""),
            "participant_id": str(context.get("participant_id") or ""),
            "condition_id": str(context.get("condition_id") or ""),
            "block_id": str(context.get("block_id") or ""),
            "trial_index_within_block": trial_index_within_block,
            "required_allowed_ids": [int(v) for v in list(context.get("allowed_tag_ids") or [])],
            "target_completion_label": str(context.get("completion_label") or ""),
            "target_completion_labels": [
                str(item).strip()
                for item in list(context.get("completion_labels") or [])
                if str(item).strip()
            ],
            "target_completion_category": str(context.get("completion_category") or ""),
            "target_completion_categories": list(context.get("completion_categories") or []),
            "min_required_recorded_count": int(context.get("min_required_recorded_count") or 0),
            "target_source": str(context.get("target_source") or "").strip(),
            "allowed_ids_start": allowed_ids,
            "allowed_ids_end": [],
            "candidate_labels_start": candidate_labels,
            "candidate_labels_end": [],
            "candidate_count_start": len(candidate_labels),
            "candidate_count_end": 0,
            "available_allowed_candidate_ids_start": available_allowed,
            "available_allowed_candidate_ids_end": [],
            "available_allowed_candidate_count_start": len(available_allowed),
            "available_allowed_candidate_count_end": 0,
            "scene_ready_at_start": False,
            "scene_ready_at_end": False,
            "apriltag_registry_status_start": self.apriltag_registry_status,
            "apriltag_registry_status_end": "",
            "lego_registry_status_start": self.lego_registry_status,
            "lego_registry_status_end": "",
            "start_time_sec": now,
            "start_time_iso": datetime.now().isoformat(),
            "start_phase": str(context.get("current_phase") or self.current_phase or ""),
            "end_phase": "",
            "end_time_sec": None,
            "end_time_iso": None,
            "duration_sec": None,
            "teleop_time_sec": 0.0,
            "time_teleop": 0.0,
            "autonomous_time_sec": 0.0,
            "time_to_commit_sec": None,
            "time_to_first_correct_lock_sec": None,
            "first_confirmation_latency_sec": None,
            "active_joystick_time_sec": 0.0,
            "confirmation_latencies_sec": [],
            "confirmation_count": 0,
            "cancel_count": 0,
            "timeout_count": 0,
            "auto_stalled_count": 0,
            "intent_locked_count": 0,
            "top_goal_switch_count": 0,
            "distinct_top_goals": [],
            "max_top_probability": 0.0,
            "joystick_effort": 0.0,
            "control_command_count": 0,
            "ee_path_length_m": 0.0,
            "teleop_distance_m": 0.0,
            "autonomous_distance_m": 0.0,
            "teleop_distance_proportion": 0.0,
            "top_goal_label_at_end": "",
            "top_probability_at_end": 0.0,
            "selected_grasp_label_at_end": "",
            "committed_goal_label": "",
            "final_inferred_goal": "",
            "final_inferred_object_name": "",
            "final_inferred_category": "",
            "avg_teleop_entropy": "",
            "success": False,
            "failure_reason": "",
            "dashboard_action_counts": {
                "start_task": 0,
                "reset_task": 0,
                "scan_scene": 0,
                "quick_rescan": 0,
                "manual_advance": 0,
                "send_home": 0,
            },
            "events": [],
            "_auto_started_at": None,
            "_last_confirm_prompt_at": None,
            "_last_ee_pose": self.last_ee_pose,
            "_last_top_goal_label": "",
            "_teleop_entropy_weighted_sum": 0.0,
            "_teleop_entropy_time_sec": 0.0,
            "_last_distribution_stamp": None,
            "_finalized": False,
        }
        min_required_count = int(self.current_trial.get("min_required_recorded_count", 0) or 0)
        if min_required_count <= 0:
            min_required_count = len(allowed_ids)
            self.current_trial["min_required_recorded_count"] = min_required_count
        self.current_trial["scene_ready_at_start"] = len(available_allowed) >= min_required_count
        rospy.loginfo(
            "[user_study_trial_logger] started %s task=%s step=%s",
            self.current_trial["trial_id"],
            self.current_trial["task_id"],
            self.current_trial["step_id"],
        )

    def _trial_targets_destination(self, trial, label=""):
        success_event = str(trial.get("success_event") or "").strip().lower()
        if success_event:
            return success_event == "release_complete"
        step_id = str(trial.get("step_id") or "").strip().lower()
        if "destination" in step_id:
            return True
        meta = self._meta_for_label(label)
        if str(meta.get("category", "")).strip() == "destination":
            return True
        target_labels = [
            str(item).strip()
            for item in list(trial.get("target_completion_labels") or [])
            if str(item).strip()
        ]
        if not target_labels:
            target_label = str(trial.get("target_completion_label") or "").strip()
            if target_label:
                target_labels = [target_label]
        if target_labels:
            categories = {
                str(self._meta_for_label(item).get("category", "")).strip()
                for item in target_labels
                if str(item).strip()
            }
            categories.discard("")
            if categories == {"destination"}:
                return True
        return False

    def _expected_terminal_event(self, trial, label=""):
        success_event = str(trial.get("success_event") or "").strip().lower()
        if success_event:
            return success_event
        if self._trial_targets_destination(trial, label=label):
            return "release_complete"
        return "grasp_complete"

    def _matches_target(self, label, trial=None):
        trial = self.current_trial if trial is None else trial
        if not trial or not label:
            return False
        label = self._normalize_goal_label(label)
        target_label = str(trial.get("target_completion_label") or "").strip()
        if target_label and label == target_label:
            return True
        target_labels = [
            str(item).strip()
            for item in list(trial.get("target_completion_labels") or [])
            if str(item).strip()
        ]
        if target_labels and label in target_labels:
            return True
        meta = self._meta_for_label(label)
        category = str(meta.get("category", "")).strip()
        target_category = str(trial.get("target_completion_category") or "").strip()
        if target_category and category == target_category:
            return True
        target_categories = [
            str(item).strip()
            for item in list(trial.get("target_completion_categories") or [])
            if str(item).strip()
        ]
        if target_categories and category in target_categories:
            return True
        return False

    def _event_matches_current_trial(self, event_name, event):
        if self.current_trial is None:
            return False
        grasp_id = str(event.get("grasp_id") or "").strip()
        if not self._matches_target(grasp_id, trial=self.current_trial):
            return False
        return event_name == self._expected_terminal_event(self.current_trial, label=grasp_id)

    def _append_trial_event(self, event):
        if self.current_trial is None:
            return
        self.current_trial["events"].append(event)
        self._write_jsonl(
            self.event_log_handle,
            {
                "trial_id": self.current_trial["trial_id"],
                "task_id": self.current_trial["task_id"],
                "step_id": self.current_trial["step_id"],
                "target_source": self.current_trial.get("target_source", ""),
                "event_payload": event,
            },
        )

    def _study_event_cb(self, msg):
        try:
            event = json.loads(str(msg.data))
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[user_study_trial_logger] bad study event json: %s", exc)
            return

        with self.lock:
            if self.current_trial is None:
                return

            event_name = str(event.get("event", "")).strip().lower()
            now_sec = float(event.get("stamp", rospy.Time.now().to_sec()))
            expected_terminal_event = self._expected_terminal_event(self.current_trial)
            terminal_success_event = event_name == expected_terminal_event
            if terminal_success_event and not self._event_matches_current_trial(event_name, event):
                rospy.loginfo(
                    "[user_study_trial_logger] ignoring terminal event=%s grasp_id=%s for active step=%s",
                    event_name,
                    str(event.get("grasp_id") or "").strip() or "<none>",
                    str(self.current_trial.get("step_id") or "") or "<none>",
                )
                return

            self._append_trial_event(event)

            if event_name == "intent_locked":
                self.current_trial["intent_locked_count"] += 1
                if (
                    self.current_trial["time_to_first_correct_lock_sec"] is None
                    and self._matches_target(event.get("grasp_id", ""))
                ):
                    self.current_trial["time_to_first_correct_lock_sec"] = max(
                        0.0, now_sec - self.current_trial["start_time_sec"]
                    )
                return

            if event_name in self.current_trial["dashboard_action_counts"]:
                self.current_trial["dashboard_action_counts"][event_name] += 1
                return

            if event_name == "confirm_prompt":
                self.current_trial["_last_confirm_prompt_at"] = now_sec
                return

            if event_name == "confirm_accept":
                committed_label = self._normalize_goal_label(event.get("grasp_id", ""))
                if committed_label:
                    self.current_trial["committed_goal_label"] = committed_label
                self.current_trial["confirmation_count"] += 1
                if self.current_trial["time_to_commit_sec"] is None:
                    self.current_trial["time_to_commit_sec"] = max(
                        0.0, now_sec - self.current_trial["start_time_sec"]
                    )
                prompt_at = self.current_trial.get("_last_confirm_prompt_at")
                if prompt_at is not None:
                    latency = max(0.0, now_sec - prompt_at)
                    self.current_trial["confirmation_latencies_sec"].append(latency)
                    if self.current_trial["first_confirmation_latency_sec"] is None:
                        self.current_trial["first_confirmation_latency_sec"] = latency
                return

            if event_name == "confirm_cancel":
                self.current_trial["cancel_count"] += 1
                self.current_trial["committed_goal_label"] = ""
                return

            if event_name == "confirm_timeout":
                self.current_trial["timeout_count"] += 1
                return

            if event_name == "auto_start":
                if self.current_trial["_auto_started_at"] is None:
                    self.current_trial["_auto_started_at"] = now_sec
                if self.current_trial["time_to_commit_sec"] is None:
                    self.current_trial["time_to_commit_sec"] = max(
                        0.0, now_sec - self.current_trial["start_time_sec"]
                    )
                return

            if event_name == "pour_start":
                if self.current_trial["_auto_started_at"] is None:
                    self.current_trial["_auto_started_at"] = now_sec
                return

            if event_name in ("auto_complete", "auto_stalled", "grasp_complete", "release_complete", "pour_complete"):
                auto_started_at = self.current_trial.get("_auto_started_at")
                if auto_started_at is not None:
                    self.current_trial["autonomous_time_sec"] += max(0.0, now_sec - auto_started_at)
                    self.current_trial["_auto_started_at"] = None
                if event_name == "auto_stalled":
                    self.current_trial["auto_stalled_count"] += 1
                if terminal_success_event:
                    self._finalize_trial_locked(success=True, failure_reason="")

    def _top_goal_cb(self, msg):
        with self.lock:
            self.top_goal_label = str(msg.data).strip()
            if self.current_trial is not None:
                prev = str(self.current_trial.get("_last_top_goal_label") or "").strip()
                cur = self._normalize_goal_label(self.top_goal_label)
                if cur:
                    distinct = self.current_trial["distinct_top_goals"]
                    if cur not in distinct:
                        distinct.append(cur)
                if prev and cur and prev != cur:
                    self.current_trial["top_goal_switch_count"] += 1
                self.current_trial["_last_top_goal_label"] = cur
            if (
                self.current_trial is not None
                and self.current_trial["time_to_first_correct_lock_sec"] is None
                and self._matches_target(self.top_goal_label)
            ):
                self.current_trial["time_to_first_correct_lock_sec"] = max(
                    0.0, rospy.Time.now().to_sec() - self.current_trial["start_time_sec"]
                )

    def _top_probability_cb(self, msg):
        with self.lock:
            self.top_probability = float(msg.data)
            self.max_top_probability = max(float(self.max_top_probability), self.top_probability)
            if self.current_trial is not None:
                self.current_trial["max_top_probability"] = max(
                    float(self.current_trial.get("max_top_probability", 0.0)),
                    self.top_probability,
                )

    def _distribution_cb(self, msg):
        now_sec = rospy.Time.now().to_sec()
        with self.lock:
            if self.current_trial is not None:
                entropy = _distribution_entropy(msg.data)
                last_stamp = self.current_trial.get("_last_distribution_stamp")
                if entropy is not None and last_stamp is not None:
                    dt = max(0.0, now_sec - float(last_stamp))
                    if dt > 0.0 and self.current_trial.get("_auto_started_at") is None:
                        self.current_trial["_teleop_entropy_weighted_sum"] += entropy * dt
                        self.current_trial["_teleop_entropy_time_sec"] += dt
                self.current_trial["_last_distribution_stamp"] = now_sec
            self.last_distribution = [float(value) for value in list(msg.data)]

    def _phase_cb(self, msg):
        with self.lock:
            self.current_phase = str(msg.data).strip()

    def _candidates_cb(self, msg):
        labels = []
        for det in msg.detections:
            if not det.results:
                continue
            try:
                labels.append(int(det.results[0].id))
            except Exception:
                continue
        labels = sorted(set(labels))
        with self.lock:
            self.current_candidate_labels = labels

    def _allowed_ids_cb(self, msg):
        with self.lock:
            self.current_allowed_ids = set(int(v) for v in list(msg.data))

    def _execution_state_cb(self, msg):
        with self.lock:
            self.execution_state = str(msg.data).strip()

    def _selected_grasp_label_cb(self, msg):
        with self.lock:
            self.selected_grasp_label = str(msg.data).strip()

    def _apriltag_registry_status_cb(self, msg):
        with self.lock:
            self.apriltag_registry_status = str(msg.data).strip()

    def _lego_registry_status_cb(self, msg):
        with self.lock:
            self.lego_registry_status = str(msg.data).strip()

    def _joy_cb(self, msg):
        now_sec = rospy.Time.now().to_sec()
        raw_axes = list(msg.axes) if msg.axes else []
        motion_values = []
        for idx in self.joystick_motion_axes:
            if 0 <= idx < len(raw_axes):
                value = float(raw_axes[idx])
                if abs(value) < self.joystick_motion_deadzone:
                    value = 0.0
                motion_values.append(value)
        axes = np.array(motion_values, dtype=np.float64) if motion_values else np.zeros(0, dtype=np.float64)
        effort = float(np.linalg.norm(axes)) if axes.size else 0.0
        with self.lock:
            if self.current_trial is not None:
                self.current_trial["control_command_count"] += 1
                if self.last_joy_stamp is not None:
                    dt = max(0.0, now_sec - self.last_joy_stamp)
                    self.current_trial["joystick_effort"] += effort * dt
                    if (
                        self.current_trial.get("_auto_started_at") is None
                        and effort >= self.active_joystick_threshold
                    ):
                        self.current_trial["active_joystick_time_sec"] += dt
            self.last_joy_stamp = now_sec
            self.last_joy_effort = effort

    def _ee_cb(self, msg):
        pose = msg.pose
        with self.lock:
            self.last_ee_pose = pose
            if self.current_trial is not None and self.current_trial.get("_last_ee_pose") is not None:
                prev_xyz = _pose_to_xyz(self.current_trial["_last_ee_pose"])
                curr_xyz = _pose_to_xyz(pose)
                delta_m = float(np.linalg.norm(curr_xyz - prev_xyz))
                self.current_trial["ee_path_length_m"] += delta_m
                if self.current_trial.get("_auto_started_at") is None:
                    self.current_trial["teleop_distance_m"] += delta_m
                else:
                    self.current_trial["autonomous_distance_m"] += delta_m
            if self.current_trial is not None:
                self.current_trial["_last_ee_pose"] = pose

    def _finalize_trial_locked(self, success, failure_reason):
        if self.current_trial is None or self.current_trial.get("_finalized", False):
            return
        self.current_trial["_finalized"] = True
        now_sec = rospy.Time.now().to_sec()
        auto_started_at = self.current_trial.get("_auto_started_at")
        if auto_started_at is not None:
            self.current_trial["autonomous_time_sec"] += max(0.0, now_sec - auto_started_at)
            self.current_trial["_auto_started_at"] = None

        self.current_trial["end_time_sec"] = now_sec
        self.current_trial["end_time_iso"] = datetime.now().isoformat()
        self.current_trial["duration_sec"] = max(0.0, now_sec - self.current_trial["start_time_sec"])
        self.current_trial["end_phase"] = self.current_phase
        self.current_trial["teleop_time_sec"] = max(
            0.0, self.current_trial["duration_sec"] - self.current_trial["autonomous_time_sec"]
        )
        self.current_trial["time_teleop"] = float(self.current_trial["teleop_time_sec"])
        total_path = float(self.current_trial.get("ee_path_length_m", 0.0) or 0.0)
        teleop_path = float(self.current_trial.get("teleop_distance_m", 0.0) or 0.0)
        self.current_trial["teleop_distance_proportion"] = (
            teleop_path / total_path if total_path > 1e-9 else 0.0
        )
        candidate_labels_end = list(self.current_candidate_labels)
        allowed_ids_end = sorted(int(v) for v in self.current_allowed_ids)
        available_allowed_end = [
            int(label) for label in candidate_labels_end
            if str(label).lstrip("-").isdigit() and int(label) in self.current_allowed_ids
        ]
        self.current_trial["allowed_ids_end"] = allowed_ids_end
        self.current_trial["candidate_labels_end"] = candidate_labels_end
        self.current_trial["candidate_count_end"] = len(candidate_labels_end)
        self.current_trial["available_allowed_candidate_ids_end"] = available_allowed_end
        self.current_trial["available_allowed_candidate_count_end"] = len(available_allowed_end)
        min_required_count = int(self.current_trial.get("min_required_recorded_count", 0) or 0)
        self.current_trial["scene_ready_at_end"] = len(available_allowed_end) >= min_required_count
        self.current_trial["apriltag_registry_status_end"] = self.apriltag_registry_status
        self.current_trial["lego_registry_status_end"] = self.lego_registry_status
        normalized_top_goal = self._normalize_goal_label(self.top_goal_label)
        committed_goal = self._normalize_goal_label(self.current_trial.get("committed_goal_label", ""))
        final_goal = committed_goal or normalized_top_goal
        self.current_trial["top_goal_label_at_end"] = self.top_goal_label
        self.current_trial["top_probability_at_end"] = float(self.top_probability)
        self.current_trial["selected_grasp_label_at_end"] = self.selected_grasp_label
        self.current_trial["execution_state_at_end"] = self.execution_state
        self.current_trial["final_inferred_goal"] = final_goal
        meta = self._meta_for_label(final_goal)
        self.current_trial["final_inferred_object_name"] = str(meta.get("object_name", "")).strip()
        self.current_trial["final_inferred_category"] = str(meta.get("category", "")).strip()
        self.current_trial["success"] = bool(success)
        self.current_trial["failure_reason"] = str(failure_reason or "")
        self.current_trial["correct_inference"] = self._matches_target(final_goal)
        self.current_trial["distribution_at_end"] = list(self.last_distribution)
        teleop_entropy_time = float(self.current_trial.get("_teleop_entropy_time_sec", 0.0) or 0.0)
        if teleop_entropy_time > 1e-9:
            self.current_trial["avg_teleop_entropy"] = (
                float(self.current_trial.get("_teleop_entropy_weighted_sum", 0.0) or 0.0)
                / teleop_entropy_time
            )
        else:
            self.current_trial["avg_teleop_entropy"] = ""

        finished = dict(self.current_trial)
        finished.pop("_auto_started_at", None)
        finished.pop("_last_confirm_prompt_at", None)
        finished.pop("_last_ee_pose", None)
        finished.pop("_last_top_goal_label", None)
        finished.pop("_teleop_entropy_weighted_sum", None)
        finished.pop("_teleop_entropy_time_sec", None)
        finished.pop("_last_distribution_stamp", None)
        finished.pop("_finalized", None)
        self._write_jsonl(self.log_handle, finished)
        rospy.loginfo(
            "[user_study_trial_logger] finished %s success=%s reason=%s",
            finished["trial_id"],
            finished["success"],
            finished["failure_reason"] or "<none>",
        )
        self.current_trial = None
        self.last_joy_stamp = None
        self.last_joy_effort = 0.0

    def _finalize_trial(self, success, failure_reason):
        with self.lock:
            self._finalize_trial_locked(success=success, failure_reason=failure_reason)

    def _trial_has_meaningful_activity_locked(self):
        if self.current_trial is None:
            return False
        if self.current_trial.get("confirmation_count", 0) > 0:
            return True
        if self.current_trial.get("cancel_count", 0) > 0:
            return True
        if self.current_trial.get("timeout_count", 0) > 0:
            return True
        if self.current_trial.get("intent_locked_count", 0) > 0:
            return True
        if self.current_trial.get("auto_stalled_count", 0) > 0:
            return True
        if float(self.current_trial.get("autonomous_time_sec", 0.0) or 0.0) > 0.0:
            return True
        if self.current_trial.get("time_to_commit_sec") is not None:
            return True
        if str(self.current_trial.get("committed_goal_label") or "").strip():
            return True
        for event in list(self.current_trial.get("events") or []):
            event_name = str(event.get("event") or "").strip()
            if event_name in (
                "confirm_prompt",
                "confirm_accept",
                "confirm_cancel",
                "confirm_timeout",
                "auto_start",
                "pour_start",
                "auto_complete",
                "auto_stalled",
                "grasp_complete",
                "release_complete",
                "pour_complete",
                "manual_advance",
                "quick_rescan",
                "send_home",
            ):
                return True
        return False

    def _shutdown(self):
        with self.lock:
            if self.current_trial is not None:
                if self._trial_has_meaningful_activity_locked():
                    self._finalize_trial_locked(success=False, failure_reason="node_shutdown")
                else:
                    rospy.loginfo(
                        "[user_study_trial_logger] dropping inactive tail trial %s on shutdown",
                        self.current_trial.get("trial_id", "<unknown>"),
                    )
                    self.current_trial = None
        try:
            self.log_handle.close()
        except Exception:
            pass
        try:
            self.event_log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    UserStudyTrialLogger()
    rospy.spin()
