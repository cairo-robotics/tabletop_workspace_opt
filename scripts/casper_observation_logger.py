#!/usr/bin/env python3
"""Event-triggered CASPER observation logger.

This node records a lightweight visual/trajectory history for CASPER Section
3.2-style offline evaluation. It does not save every camera frame; it writes a
downsized JPEG only when a decision-relevant event occurs.
"""

import json
import math
import os
import threading
from collections import deque
from datetime import datetime

import cv2
import rospy
import yaml
from cv_bridge import CvBridge
from intera_core_msgs.msg import EndpointState
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray, String
from vision_msgs.msg import Detection2DArray


def _parse_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class CasperObservationLogger(object):
    def __init__(self):
        rospy.init_node("casper_observation_logger")

        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_log_dir = os.path.join(package_root, "logs")
        default_frame_dir = os.path.join(default_log_dir, "casper_frames")

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
        self.distribution_topic = str(
            rospy.get_param("~distribution_topic", "/apriltag_intent_inference/distribution")
        ).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.manual_label_state_topic = str(
            rospy.get_param("~manual_label_state_topic", "/sam_manual_labeler/state")
        ).strip()
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param("~object_map_yaml", os.path.join(package_root, "config", "apriltag_object_map.yaml"))
        )
        self.log_dir = os.path.expanduser(rospy.get_param("~log_dir", default_log_dir))
        self.frame_dir = os.path.expanduser(rospy.get_param("~frame_dir", default_frame_dir))
        self.save_image_width = int(rospy.get_param("~save_image_width", 384))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 70))
        self.min_save_interval_sec = float(rospy.get_param("~min_save_interval_sec", 1.0))
        self.max_images_per_trial = int(rospy.get_param("~max_images_per_trial", 8))
        self.probability_crossing_threshold = float(rospy.get_param("~probability_crossing_threshold", 0.6))
        self.trajectory_history_sec = float(rospy.get_param("~trajectory_history_sec", 3.0))
        self.trajectory_history_max_points = int(rospy.get_param("~trajectory_history_max_points", 20))
        self.task_history_count = int(rospy.get_param("~task_history_count", 6))
        self.save_on_top_goal_change = _parse_bool(rospy.get_param("~save_on_top_goal_change", True))
        self.save_on_probability_crossing = _parse_bool(rospy.get_param("~save_on_probability_crossing", True))
        self.save_on_selection = _parse_bool(rospy.get_param("~save_on_selection", True))
        self.save_on_trial_start = _parse_bool(rospy.get_param("~save_on_trial_start", True))
        self.flush_every_write = _parse_bool(rospy.get_param("~flush_every_write", True))
        self.intent_select_mode = str(rospy.get_param("~intent_select_mode", "threshold")).strip().lower()
        self.auto_select_pregrasp = _parse_bool(rospy.get_param("~auto_select_pregrasp", False))

        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.label_to_meta, self.tag_id_to_meta = self._load_object_map()

        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_image_frame = ""
        self.ee_history = deque()
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
        self.intended_grasp_label = ""
        self.intended_candidate_id = ""
        self.terminal_grasp_label = ""
        self.terminal_candidate_id = ""
        self.execution_state = ""
        self.control_phase = "idle"
        self.frozen_trajectory = []
        self.freeze_reason = ""
        self.last_saved_at = None
        self.images_this_trial = 0
        self.last_saved_top_goal = ""
        self.crossing_saved_for_goal = set()

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.log_dir, "casper_observations_{}.jsonl".format(stamp))
        self.log_handle = open(self.log_path, "a", encoding="utf-8")

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

        rospy.loginfo("[casper_observation_logger] writing observations to %s", self.log_path)

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
        candidate_id = str(meta.get("candidate_id") or "").strip()
        return candidate_id

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
            elif event_name in ("confirm_cancel",):
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
            rospy.logwarn_throttle(5.0, "[casper_observation_logger] image conversion failed: %s", exc)
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
                if len(center_uv) < 2:
                    continue
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
        with self.lock:
            next_key = self._trial_key(context)
            previous_key = self.current_trial_key
            self.current_context = context
            self.current_trial_key = next_key
            if next_key != previous_key:
                self.images_this_trial = 0
                self.last_saved_at = None
                self.last_saved_top_goal = ""
                self.crossing_saved_for_goal.clear()
                self.ee_history.clear()
                self.task_history.clear()
                self.frozen_trajectory = []
                self.freeze_reason = ""
                self.intended_grasp_label = ""
                self.intended_candidate_id = ""
                self.terminal_grasp_label = ""
                self.terminal_candidate_id = ""
                self.control_phase = "teleop" if next_key else "idle"
                if next_key and self.save_on_trial_start:
                    self._maybe_save_locked("trial_start", force=True)

    def _study_event_cb(self, msg):
        try:
            event = json.loads(str(msg.data))
        except Exception:
            event = {}
        name = str(event.get("event") or "").strip().lower()
        if not name:
            return
        with self.lock:
            grasp_label = str(event.get("grasp_id") or event.get("selected_grasp_label") or "").strip()
            if name in (
                "auto_select_pregrasp",
                "joystick_event_select",
                "joystick_retarget_select",
                "confirm_accept",
            ):
                if grasp_label:
                    self.intended_grasp_label = grasp_label
                    self.intended_candidate_id = self._candidate_id_for_grasp_label(grasp_label)
                self._freeze_trajectory_locked(name)
                self.control_phase = "selected"
                if self.save_on_selection:
                    self._maybe_save_locked(name, force=True)
                return
            if name in ("auto_start", "pour_start"):
                if grasp_label and not self.intended_grasp_label:
                    self.intended_grasp_label = grasp_label
                    self.intended_candidate_id = self._candidate_id_for_grasp_label(grasp_label)
                self._append_task_history_locked(name, grasp_label=grasp_label, status="started")
                self._freeze_trajectory_locked(name)
                self.control_phase = "autonomous"
                return
            if name in ("auto_complete", "grasp_complete", "release_complete", "pour_complete"):
                if grasp_label:
                    self.terminal_grasp_label = grasp_label
                    self.terminal_candidate_id = self._candidate_id_for_grasp_label(grasp_label)
                    if not self.intended_grasp_label:
                        self.intended_grasp_label = grasp_label
                        self.intended_candidate_id = self.terminal_candidate_id
                self._maybe_save_locked(name, force=True)
                self._append_task_history_locked(name, grasp_label=grasp_label)
                if self.current_trial_key:
                    self.control_phase = "teleop"
                self.frozen_trajectory = []
                self.freeze_reason = ""
                return
            if name in ("confirm_cancel",):
                self._append_task_history_locked(name, grasp_label=grasp_label)
                if self.current_trial_key:
                    self.control_phase = "teleop"
                self.frozen_trajectory = []
                self.freeze_reason = ""
                self.intended_grasp_label = ""
                self.intended_candidate_id = ""

    def _execution_state_cb(self, msg):
        state = str(msg.data).strip().lower()
        with self.lock:
            self.execution_state = state
            if state.startswith("exec_") or state in ("retreat_before_pregrasp",):
                self._freeze_trajectory_locked("execution_state_{}".format(state))
                self.control_phase = "autonomous"
            elif state in ("", "idle", "done", "wait_pregrasp_confirm"):
                if self.current_trial_key and self.control_phase == "autonomous":
                    self.control_phase = "teleop"
                    self.frozen_trajectory = []
                    self.freeze_reason = ""

    def _top_goal_cb(self, msg):
        with self.lock:
            next_goal = str(msg.data).strip()
            changed = bool(next_goal and next_goal != self.top_goal_label)
            self.top_goal_label = next_goal
            if changed and self.save_on_top_goal_change:
                self._maybe_save_locked("top_goal_change")

    def _top_probability_cb(self, msg):
        with self.lock:
            previous = float(self.top_probability)
            self.top_probability = float(msg.data)
            if not self.save_on_probability_crossing:
                return
            goal = self.top_goal_label
            if not goal:
                return
            if (
                previous < self.probability_crossing_threshold
                and self.top_probability >= self.probability_crossing_threshold
                and goal not in self.crossing_saved_for_goal
            ):
                if self._maybe_save_locked("probability_crossing"):
                    self.crossing_saved_for_goal.add(goal)

    def _distribution_cb(self, msg):
        with self.lock:
            self.last_distribution = [float(v) for v in list(msg.data)]

    def _selected_label_cb(self, msg):
        with self.lock:
            next_label = str(msg.data).strip()
            changed = bool(next_label and next_label != self.selected_grasp_label)
            self.selected_grasp_label = next_label
            if changed and self.save_on_selection:
                self.intended_grasp_label = next_label
                self.intended_candidate_id = self._candidate_id_for_grasp_label(next_label)
                self._freeze_trajectory_locked("selected_grasp_label")
                self.control_phase = "selected"
                self._maybe_save_locked("selection", force=True)

    def _subsample_ee_history_locked(self):
        points = list(self.ee_history)
        max_points = max(2, int(self.trajectory_history_max_points))
        if len(points) <= max_points:
            return points
        stride = float(len(points) - 1) / float(max_points - 1)
        return [points[int(round(i * stride))] for i in range(max_points)]

    def _freeze_trajectory_locked(self, reason):
        if not self.frozen_trajectory:
            self.frozen_trajectory = self._subsample_ee_history_locked()
            self.freeze_reason = str(reason or "freeze")

    def _trajectory_for_save_locked(self):
        if self.frozen_trajectory:
            return list(self.frozen_trajectory)
        return self._subsample_ee_history_locked()

    def _trajectory_summary_locked(self, points):
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
        rows = []
        for label in labels:
            meta = self.tag_id_to_meta.get(label, {})
            category = str(meta.get("category") or "").strip()
            task = str(meta.get("task") or "").strip()
            if not task:
                task = "place" if category == "destination" else "pickup"
            rows.append(
                {
                    "candidate_id": label,
                    "label": label,
                    "object_name": str(meta.get("object_name") or "").strip(),
                    "category": category,
                    "task_suitability": task,
                    "grasp_complete_label": str(meta.get("grasp_complete_label") or "").strip(),
                    "position": self.candidate_poses.get(label, {}),
                    "center_uv": [
                        float(v) * float(image_scale)
                        for v in self.candidate_uv.get(label, [])
                    ],
                }
            )
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

    def _maybe_save_locked(self, reason, force=False):
        now = rospy.Time.now().to_sec()
        if self.latest_image is None:
            return False
        if (
            self.max_images_per_trial >= 0
            and self.images_this_trial >= self.max_images_per_trial
            and not force
        ):
            return False
        if not force and self.last_saved_at is not None:
            if (now - float(self.last_saved_at)) < self.min_save_interval_sec:
                return False

        image, image_scale = self._resized_image_locked()
        if image is None:
            return False
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = "{}_{}.jpg".format(stamp, reason)
        path = os.path.join(self.frame_dir, filename)
        ok = cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not ok:
            rospy.logwarn("[casper_observation_logger] failed to write image %s", path)
            return False

        trajectory = self._trajectory_for_save_locked()
        payload = {
            "timestamp": datetime.now().isoformat(),
            "stamp_sec": now,
            "save_reason": reason,
            "image_path": path,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
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
            "intended_grasp_label": self.intended_grasp_label,
            "intended_candidate_id": self.intended_candidate_id,
            "terminal_grasp_label": self.terminal_grasp_label,
            "terminal_candidate_id": self.terminal_candidate_id,
            "correct_candidate_id": self.intended_candidate_id or self.terminal_candidate_id,
            "execution_state": self.execution_state,
            "control_phase": self.control_phase,
            "trajectory_source": "teleop_only",
            "trajectory_freeze_reason": self.freeze_reason,
            "intent_select_mode": self.intent_select_mode,
            "auto_select_pregrasp": self.auto_select_pregrasp,
            "trajectory_history": trajectory,
            "trajectory_summary": self._trajectory_summary_locked(trajectory),
            "task_history": list(self.task_history),
        }
        self.log_handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.flush_every_write:
            self.log_handle.flush()
        self.last_saved_at = now
        self.images_this_trial += 1
        self.last_saved_top_goal = self.top_goal_label
        return True

    def _shutdown(self):
        try:
            self.log_handle.flush()
            self.log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    CasperObservationLogger()
    rospy.spin()
