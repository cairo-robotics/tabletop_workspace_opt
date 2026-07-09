#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute filtered AprilTag pregrasp/grasp with joystick confirmation."""

import copy
import json
import math
import os

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEPoseGoals
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Int32MultiArray, String
from visualization_msgs.msg import Marker
import yaml


def _as_np_pos(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)


def _normalize_quat(q):
    q = np.array(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_angle_rad(a, b):
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _pose_stamped_close(a, b, pos_tol=1e-4, rot_tol=1e-3):
    if a is None or b is None:
        return False
    pa = _as_np_pos(a.pose)
    pb = _as_np_pos(b.pose)
    if float(np.linalg.norm(pa - pb)) > float(pos_tol):
        return False
    qa = [a.pose.orientation.x, a.pose.orientation.y, a.pose.orientation.z, a.pose.orientation.w]
    qb = [b.pose.orientation.x, b.pose.orientation.y, b.pose.orientation.z, b.pose.orientation.w]
    return _quat_angle_rad(qa, qb) <= float(rot_tol)


class AprilTagFilteredExecutor:
    def __init__(self):
        rospy.init_node("apriltag_filtered_executor")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.endpoint_topic = str(rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")).strip()
        self.pregrasp_topic = str(rospy.get_param("~pregrasp_topic", "/tag_grasp_demo/pregrasp_pose")).strip()
        self.grasp_topic = str(rospy.get_param("~grasp_topic", "/tag_grasp_demo/grasp_pose")).strip()
        self.joy_topic = str(rospy.get_param("~joy_topic", "joy")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.prompt_topic = str(
            rospy.get_param("~prompt_topic", "/intent_inference/confirmation_prompt")
        ).strip()
        self.execution_state_topic = str(
            rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")
        ).strip()
        self.study_event_topic = str(
            rospy.get_param("~study_event_topic", "/user_study/events")
        ).strip()
        self.task_phase_topic = str(rospy.get_param("~task_phase_topic", "/task_context/phase")).strip()
        self.allowed_ids_topic = str(rospy.get_param("~allowed_ids_topic", "/task_context/allowed_tag_ids")).strip()
        self.grasp_complete_label = str(rospy.get_param("~grasp_complete_label", "apriltag_id_0")).strip()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param(
                "~object_map_yaml",
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    "config",
                    "apriltag_object_map.yaml",
                ),
            )
        )
        self.prompt_marker_topic = str(rospy.get_param("~prompt_marker_topic", "~prompt_marker")).strip()
        self.prompt_marker_offset = float(rospy.get_param("~prompt_marker_offset_z", 0.15))
        self.show_status_window = bool(rospy.get_param("~show_status_window", True))
        self.status_window_name = str(rospy.get_param("~status_window_name", "AprilTag Executor")).strip()

        self.confirm_button_index = int(rospy.get_param("~confirm_button_index", 2))
        self.cancel_button_index = int(rospy.get_param("~cancel_button_index", 3))
        self.close_button_index = int(rospy.get_param("~close_button_index", 0))
        self.open_button_index = int(rospy.get_param("~open_button_index", 1))
        self.max_speed_mps = float(rospy.get_param("~max_speed_mps", 0.10))
        self.max_angular_step = float(rospy.get_param("~max_angular_step", 0.12))
        self.pos_tol = float(rospy.get_param("~position_tolerance_m", 0.01))
        self.rot_tol = float(rospy.get_param("~orientation_tolerance_rad", 0.25))
        self.pregrasp_pos_tol = float(rospy.get_param("~pregrasp_position_tolerance_m", self.pos_tol))
        self.pregrasp_rot_tol = float(rospy.get_param("~pregrasp_orientation_tolerance_rad", self.rot_tol))
        self.use_goal_orientation = bool(rospy.get_param("~use_goal_orientation", True))
        self.manual_assist_enabled = bool(rospy.get_param("~manual_assist_enabled", True))
        self.manual_assist_speed_mps = float(rospy.get_param("~manual_assist_speed_mps", 0.035))
        self.manual_assist_deadzone = float(rospy.get_param("~manual_assist_deadzone", 0.12))
        self.done_reset_sec = float(rospy.get_param("~done_reset_sec", 0.75))
        self.stall_timeout_sec = float(rospy.get_param("~stall_timeout_sec", 8.0))
        self.stall_progress_epsilon_m = float(rospy.get_param("~stall_progress_epsilon_m", 0.003))
        self.auto_start_pregrasp_on_new_target = bool(
            rospy.get_param("~auto_start_pregrasp_on_new_target", False)
        )
        self.required_control_mode = str(rospy.get_param("~required_control_mode", "shared_autonomy")).strip()
        self.pause_topic = str(rospy.get_param("~pause_topic", "/shared_autonomy/home_motion_active")).strip()

        self.latest_ee = None
        self.latest_axes = []
        self.latest_buttons = []
        self.prev_buttons = []
        self.last_joy_time = None
        self.pregrasp = None
        self.grasp = None
        self.exec_pregrasp = None
        self.exec_grasp = None
        self.locked_grasp_label = ""
        self.locked_grasp_pose = None
        self.state = "WAIT_PREGRASP_CONFIRM"
        self.state_started_at = rospy.Time.now()
        self.phase_best_distance = None
        self.phase_last_progress_time = self.state_started_at
        self.last_cmd_time = None
        self.last_status = ""
        self._cv_window_initialized = False
        self.current_phase = "scan_workspace"
        self.current_allowed_ids = set()
        self.paused = False
        self.label_to_meta = self._load_label_metadata()

        self.goal_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.prompt_pub = rospy.Publisher(self.prompt_topic, String, queue_size=1, latch=True)
        self.exec_state_pub = rospy.Publisher(self.execution_state_topic, String, queue_size=1, latch=True)
        self.study_event_pub = rospy.Publisher(self.study_event_topic, String, queue_size=20)
        self.prompt_marker_pub = rospy.Publisher(self.prompt_marker_topic, Marker, queue_size=1, latch=True)

        rospy.Subscriber(self.endpoint_topic, EndpointState, self._ee_cb, queue_size=10)
        rospy.Subscriber(self.pregrasp_topic, PoseStamped, self._pre_cb, queue_size=1)
        rospy.Subscriber(self.grasp_topic, PoseStamped, self._grasp_cb, queue_size=1)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_grasp_label_cb, queue_size=1)
        rospy.Subscriber(self.task_phase_topic, String, self._task_phase_cb, queue_size=1)
        rospy.Subscriber(self.allowed_ids_topic, Int32MultiArray, self._allowed_ids_cb, queue_size=1)
        rospy.Subscriber(self.pause_topic, Bool, self._pause_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self._tick)
        rospy.Timer(rospy.Duration(0.5), self._guard)
        rospy.Timer(rospy.Duration(0.1), self._ui_tick)
        rospy.on_shutdown(self._shutdown)

        self._init_status_window()
        self.exec_state_pub.publish(String(data=str(self.state).strip().lower()))
        self._publish_status("waiting_for_targets")
        rospy.loginfo(
            "[apriltag_filtered_executor] ready pre=%s grasp=%s endpoint=%s joy=%s",
            self.pregrasp_topic,
            self.grasp_topic,
            self.endpoint_topic,
            self.joy_topic,
        )

    def _guard(self, _evt):
        mode = str(rospy.get_param("/tabletop_workspace_opt/control_mode", "")).strip()
        if mode and mode != self.required_control_mode:
            rospy.logwarn(
                "[apriltag_filtered_executor] control_mode=%s required=%s; shutting down",
                mode,
                self.required_control_mode,
            )
            rospy.signal_shutdown("control mode mismatch")

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[apriltag_filtered_executor] %s", text)
        self.status_pub.publish(String(data=text))
        self.prompt_pub.publish(String(data=text))
        self._publish_prompt_marker(text)

    def _set_state(self, new_state):
        if new_state != self.state:
            old_state = self.state
            self.state = new_state
            self.state_started_at = rospy.Time.now()
            self.phase_best_distance = None
            self.phase_last_progress_time = self.state_started_at
            self.exec_state_pub.publish(String(data=str(new_state).strip().lower()))
            self._publish_study_event(
                "executor_state",
                old_state=old_state,
                new_state=new_state,
                grasp_id=self.grasp_complete_label or "",
            )

    def _publish_study_event(self, event_type, **fields):
        payload = {
            "event": str(event_type),
            "node": rospy.get_name(),
            "stamp": rospy.Time.now().to_sec(),
        }
        for key, value in fields.items():
            if value is None:
                continue
            payload[str(key)] = value
        self.study_event_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _load_label_metadata(self):
        if not os.path.exists(self.object_map_yaml):
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict) and isinstance(data.get("tag_objects"), dict):
            tag_objects = data.get("tag_objects", {}) or {}
        elif isinstance(data, dict) and isinstance(data.get("candidate_objects"), dict):
            tag_objects = data.get("candidate_objects", {}) or {}
        else:
            tag_objects = {}
        mapping = {}
        for key, meta in tag_objects.items():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("grasp_complete_label", "")).strip()
            if label:
                enriched_meta = dict(meta)
                try:
                    enriched_meta["tag_id"] = int(key)
                except Exception:
                    enriched_meta["tag_id"] = key
                mapping[label] = enriched_meta
        return mapping

    def _selected_grasp_label_cb(self, msg):
        selected_label = str(msg.data).strip()
        if self.locked_grasp_label and selected_label and selected_label != self.locked_grasp_label:
            rospy.loginfo(
                "[apriltag_filtered_executor] ignoring selected grasp label change while locked: %s -> %s",
                self.locked_grasp_label,
                selected_label,
            )
            return
        if not selected_label:
            self.grasp_complete_label = ""
            self.exec_pregrasp = None
            self.exec_grasp = None
            self.locked_grasp_label = ""
            self.locked_grasp_pose = None
            self.last_cmd_time = None
            self._set_state("WAIT_PREGRASP_CONFIRM")
            rospy.loginfo("[apriltag_filtered_executor] cleared selected grasp label")
            return
        if selected_label == self.grasp_complete_label:
            return
        self.grasp_complete_label = selected_label
        rospy.loginfo(
            "[apriltag_filtered_executor] updated grasp_complete_label to %s",
            self.grasp_complete_label,
        )
        self._maybe_accept_loaded_target()

    def _task_phase_cb(self, msg):
        self.current_phase = str(msg.data).strip().lower() or "scan_workspace"

    def _allowed_ids_cb(self, msg):
        parsed = set()
        for value in list(msg.data):
            try:
                parsed.add(int(value))
            except Exception:
                continue
        self.current_allowed_ids = parsed

    def _pause_cb(self, msg):
        self.paused = bool(msg.data)
        if self.paused:
            self.exec_pregrasp = None
            self.exec_grasp = None
            self.locked_grasp_label = ""
            self.locked_grasp_pose = None
            self.last_cmd_time = None
            self._set_state("WAIT_PREGRASP_CONFIRM")
            self._publish_status("paused_for_home_motion")

    def _maybe_accept_loaded_target(self):
        if self.state != "WAIT_PREGRASP_CONFIRM":
            return False
        if not self.grasp_complete_label or self.pregrasp is None or self.grasp is None:
            return False
        if not self._selected_target_matches_phase():
            return False
        self.locked_grasp_label = self.grasp_complete_label
        self.exec_pregrasp = copy.deepcopy(self.pregrasp)
        self.locked_grasp_pose = copy.deepcopy(self.grasp)
        self.exec_grasp = None
        self.last_cmd_time = None
        self._set_state("EXEC_PREGRASP")
        self._publish_study_event(
            "confirm_accept",
            grasp_id=self.grasp_complete_label or "",
            stage="pregrasp",
        )
        self._publish_study_event(
            "auto_start",
            grasp_id=self.grasp_complete_label or "",
            stage="pregrasp",
        )
        self._publish_status("confirmed_pregrasp_start")
        return True

    def _required_categories_for_phase(self):
        if self.current_phase in ("select_breakfast_milk", "breakfast_milk"):
            return {"milk"}
        if self.current_phase in ("select_breakfast_ingredient", "breakfast_ingredient"):
            return {"cereal", "chocolate"}
        if self.current_phase in ("select_sort_destination", "sort_destination"):
            return {"destination"}
        if self.current_phase in ("grasp_fruit", "select_fruit", "fruit"):
            return {"fruit"}
        if self.current_phase in ("grasp_destination", "select_destination", "destination"):
            return {"destination"}
        if self.current_phase in ("grasp_milk", "select_milk", "milk"):
            return {"milk"}
        if self.current_phase in ("grasp_condiment", "select_condiment", "condiment"):
            return {"cereal", "chocolate"}
        if self.current_phase in ("grasp_cereal", "select_cereal", "cereal"):
            return {"cereal"}
        if self.current_phase in ("select_lego_brick", "lego_brick", "grasp_lego", "select_lego", "lego"):
            return {"lego"}
        if self.current_phase in ("grasp_chocolate", "select_chocolate", "chocolate"):
            return {"chocolate"}
        return set()

    def _required_tag_ids_for_phase(self):
        if self.current_phase in ("select_sort_object", "sort_object"):
            return {
                int(meta.get("tag_id"))
                for meta in self.label_to_meta.values()
                if str(meta.get("sorting_group", "")).strip() in ("fruit", "cereal_like")
            }
        if self.current_phase in ("select_sort_destination", "sort_destination"):
            return {
                int(meta.get("tag_id"))
                for meta in self.label_to_meta.values()
                if str(meta.get("destination_group", "")).strip() == "sorting_target"
            }
        if self.current_phase in ("grasp_destination", "select_destination", "destination"):
            return {
                int(meta.get("tag_id"))
                for meta in self.label_to_meta.values()
                if str(meta.get("category", "")).strip() == "destination"
            }
        if self.current_phase in ("avoid_soy_milk", "select_soy_free_milk", "soy_free_milk"):
            return {
                int(meta.get("tag_id"))
                for meta in self.label_to_meta.values()
                if str(meta.get("object_name", "")).strip() in ("pure_milk", "oat_milk")
            }
        if self.current_phase in ("avoid_dairy_milk", "select_dairy_free_milk", "dairy_free_milk"):
            return {
                int(meta.get("tag_id"))
                for meta in self.label_to_meta.values()
                if str(meta.get("object_name", "")).strip() in ("oat_milk", "soy_milk")
            }
        return set()

    def _selected_target_matches_phase(self):
        meta = self.label_to_meta.get(self.grasp_complete_label, {})
        if self.current_allowed_ids:
            try:
                return int(meta.get("tag_id")) in self.current_allowed_ids
            except Exception:
                return False
        required_tag_ids = self._required_tag_ids_for_phase()
        if required_tag_ids:
            try:
                return int(meta.get("tag_id")) in required_tag_ids
            except Exception:
                return False
        required_categories = self._required_categories_for_phase()
        if not required_categories:
            return True
        actual_category = str(meta.get("category", "")).strip().lower()
        return actual_category in required_categories

    def _is_destination_phase(self):
        return self.current_phase in (
            "grasp_destination", "select_destination", "destination",
            "select_sort_destination", "sort_destination",
        )

    def _ignore_goal_orientation_for_active_target(self):
        if not self.use_goal_orientation:
            return True
        if self._is_destination_phase():
            return True
        meta = self.label_to_meta.get(self.grasp_complete_label, {})
        if not isinstance(meta, dict):
            return False
        if str(meta.get("destination_group", "")).strip() == "sorting_target":
            return True
        return str(meta.get("category", "")).strip() == "destination"

    def _use_tag_aligned_grasp_motion(self):
        return self.current_phase in (
            "select_breakfast_ingredient",
            "breakfast_ingredient",
            "select_breakfast_milk",
            "breakfast_milk",
        )

    def _init_status_window(self):
        if not self.show_status_window or self._cv_window_initialized:
            return
        try:
            cv2.namedWindow(self.status_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.status_window_name, 760, 320)
            cv2.startWindowThread()
            self._cv_window_initialized = True
        except Exception as exc:
            rospy.logwarn("[apriltag_filtered_executor] failed to init status window: %s", exc)
            self.show_status_window = False

    def _distance_to(self, pose_stamped):
        if self.latest_ee is None or pose_stamped is None:
            return None
        return float(np.linalg.norm(_as_np_pos(self.latest_ee) - _as_np_pos(pose_stamped.pose)))

    def _render_status_window(self, text):
        if not self.show_status_window:
            return
        self._init_status_window()
        canvas = np.zeros((320, 760, 3), dtype=np.uint8)
        canvas[:, :] = (28, 28, 28)

        title = "AprilTag Execute Status"
        cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "State: {}".format(self.state), (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "EE:{}  PRE:{}  GRASP:{}  JOY:{}".format(
                int(self.latest_ee is not None),
                int(self.pregrasp is not None),
                int(self.grasp is not None),
                int(self.last_joy_time is not None),
            ),
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 220, 255),
            2,
            cv2.LINE_AA,
        )

        lines = []
        for chunk in str(text).split(" "):
            if not lines or len(lines[-1]) + len(chunk) + 1 > 50:
                lines.append(chunk)
            else:
                lines[-1] += " " + chunk
        y = 130
        for line in lines[:4]:
            cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 255, 180), 2, cv2.LINE_AA)
            y += 34

        pre_dist = self._distance_to(self.exec_pregrasp if self.exec_pregrasp is not None else self.pregrasp)
        grasp_dist = self._distance_to(self.exec_grasp if self.exec_grasp is not None else self.grasp)
        dist_text_pre = "n/a" if pre_dist is None else "{:.3f} m".format(pre_dist)
        dist_text_grasp = "n/a" if grasp_dist is None else "{:.3f} m".format(grasp_dist)
        cv2.putText(canvas, "Pregrasp dist: {}".format(dist_text_pre), (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Grasp dist: {}".format(dist_text_grasp), (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "X:{}  Y:{}  A:{}  B:{}".format(
                self.confirm_button_index,
                self.cancel_button_index,
                self.close_button_index,
                self.open_button_index,
            ),
            (390, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Joy seen: {}".format("yes" if self.last_joy_time is not None else "no"),
            (390, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 180, 255),
            2,
            cv2.LINE_AA,
        )
        button_text = ",".join(str(v) for v in self.latest_buttons[:8]) if self.latest_buttons else "none"
        cv2.putText(canvas, "Buttons: {}".format(button_text), (20, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        try:
            cv2.imshow(self.status_window_name, canvas)
            cv2.waitKey(1)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[apriltag_filtered_executor] status window update failed: %s", exc)

    def _ui_tick(self, _evt):
        self._render_status_window(self.last_status if self.last_status else "starting")

    def _shutdown(self):
        if self.show_status_window:
            try:
                cv2.destroyWindow(self.status_window_name)
            except Exception:
                pass

    def _publish_prompt_marker(self, text):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "apriltag_exec_prompt"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        if self.latest_ee is not None:
            marker.pose.position.x = float(self.latest_ee.position.x)
            marker.pose.position.y = float(self.latest_ee.position.y)
            marker.pose.position.z = float(self.latest_ee.position.z + self.prompt_marker_offset)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = text
        self.prompt_marker_pub.publish(marker)

    def _ee_cb(self, msg):
        self.latest_ee = msg.pose

    def _pre_cb(self, msg):
        previous_pregrasp = self.pregrasp
        self.pregrasp = copy.deepcopy(msg)
        if (
            self.auto_start_pregrasp_on_new_target
            and self.state in ("WAIT_PREGRASP_CONFIRM", "DONE")
            and not _pose_stamped_close(previous_pregrasp, self.pregrasp)
        ):
            self.exec_pregrasp = copy.deepcopy(self.pregrasp)
            self.exec_grasp = None
            self.last_cmd_time = None
            self._set_state("EXEC_PREGRASP")
            self._publish_status("auto_start_pregrasp")
            return
        self._maybe_accept_loaded_target()

    def _grasp_cb(self, msg):
        self.grasp = copy.deepcopy(msg)
        self._maybe_accept_loaded_target()

    def _joy_cb(self, msg):
        self.latest_axes = list(msg.axes)
        self.prev_buttons = list(self.latest_buttons)
        self.latest_buttons = list(msg.buttons)
        self.last_joy_time = rospy.Time.now()
        if len(self.prev_buttons) == 0 and len(self.latest_buttons) > 0:
            rospy.loginfo(
                "[apriltag_filtered_executor] joy connected on %s (buttons=%d, confirm=%d cancel=%d close=%d)",
                self.joy_topic,
                len(self.latest_buttons),
                self.confirm_button_index,
                self.cancel_button_index,
                self.close_button_index,
            )

    def _pressed(self, idx):
        if idx < 0 or idx >= len(self.latest_buttons):
            return False
        return bool(self.latest_buttons[idx])

    def _pressed_edge(self, idx):
        cur = idx >= 0 and idx < len(self.latest_buttons) and bool(self.latest_buttons[idx])
        prev = idx >= 0 and idx < len(self.prev_buttons) and bool(self.prev_buttons[idx])
        return cur and not prev

    def _compute_dt(self, now):
        if self.last_cmd_time is None:
            self.last_cmd_time = now
            return 0.05
        dt = max(0.005, (now - self.last_cmd_time).to_sec())
        self.last_cmd_time = now
        return dt

    def _axis_value(self, idx):
        if idx < 0 or idx >= len(self.latest_axes):
            return 0.0
        return float(self.latest_axes[idx])

    def _apply_deadzone(self, value):
        return value if abs(value) >= self.manual_assist_deadzone else 0.0

    def _manual_assist_velocity(self):
        if not self.manual_assist_enabled:
            return np.zeros(3, dtype=np.float64)
        if self.state == "EXEC_GRASP":
            return np.zeros(3, dtype=np.float64)
        assist = np.array(
            [
                self._apply_deadzone(self._axis_value(1)),
                self._apply_deadzone(self._axis_value(0)),
                self._apply_deadzone(self._axis_value(4)),
            ],
            dtype=np.float64,
        )
        return assist * self.manual_assist_speed_mps

    def _build_cmd_pose(self, target_pose, now):
        cur = self.latest_ee
        dt = self._compute_dt(now)
        cur_p = _as_np_pos(cur)
        tgt_p = _as_np_pos(target_pose)
        delta = tgt_p - cur_p
        dist = float(np.linalg.norm(delta))
        auto_vel = np.zeros(3, dtype=np.float64)
        if dist > 1e-9:
            auto_vel = delta / dist * self.max_speed_mps
        assist_vel = self._manual_assist_velocity()
        cmd_step = (auto_vel + assist_vel) * dt
        step_norm = float(np.linalg.norm(cmd_step))
        max_step = max(1e-4, (self.max_speed_mps + self.manual_assist_speed_mps) * dt)
        if step_norm > max_step:
            cmd_step = cmd_step / step_norm * max_step
        cmd_p = cur_p + cmd_step
        if dist <= max(1e-4, self.max_speed_mps * dt) and float(np.linalg.norm(assist_vel)) < 1e-6:
            cmd_p = tgt_p

        out = Pose()
        out.position.x = float(cmd_p[0])
        out.position.y = float(cmd_p[1])
        out.position.z = float(cmd_p[2])

        if self._ignore_goal_orientation_for_active_target():
            out.orientation = copy.deepcopy(cur.orientation)
            return out

        qa = _normalize_quat([cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w])
        qb = _normalize_quat([target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w])
        if float(np.dot(qa, qb)) < 0.0:
            qb = -qb
        q = _normalize_quat(qa + min(1.0, self.max_angular_step) * (qb - qa))
        out.orientation.x = float(q[0])
        out.orientation.y = float(q[1])
        out.orientation.z = float(q[2])
        out.orientation.w = float(q[3])
        return out

    def _at_target(self, target_pose, pos_tol=None, rot_tol=None):
        cur = self.latest_ee
        if cur is None:
            return False
        if pos_tol is None:
            pos_tol = self.pos_tol
        if rot_tol is None:
            rot_tol = self.rot_tol
        cur_p = _as_np_pos(cur)
        tgt_p = _as_np_pos(target_pose)
        dist = float(np.linalg.norm(cur_p - tgt_p))
        if self._ignore_goal_orientation_for_active_target():
            return dist <= pos_tol
        ang = _quat_angle_rad(
            [cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w],
            [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w],
        )
        return dist <= pos_tol and ang <= rot_tol

    def _publish_goal(self, pose, now):
        msg = EEPoseGoals()
        msg.header.stamp = now
        msg.header.frame_id = self.base_frame
        msg.ee_poses.append(pose)
        self.goal_pub.publish(msg)

    def _publish_current_pose_hold_goal(self, now=None):
        if self.latest_ee is None:
            return False
        if now is None:
            now = rospy.Time.now()
        pose = Pose()
        pose.position.x = float(self.latest_ee.position.x)
        pose.position.y = float(self.latest_ee.position.y)
        pose.position.z = float(self.latest_ee.position.z)
        pose.orientation.x = float(self.latest_ee.orientation.x)
        pose.orientation.y = float(self.latest_ee.orientation.y)
        pose.orientation.z = float(self.latest_ee.orientation.z)
        pose.orientation.w = float(self.latest_ee.orientation.w)
        self._publish_goal(pose, now)
        return True

    def _build_locked_grasp_target(self):
        base_grasp = self.locked_grasp_pose if self.locked_grasp_pose is not None else self.grasp
        if base_grasp is None:
            return None
        return copy.deepcopy(base_grasp)

    def _build_vertical_grasp_from_current_xy(self):
        base_grasp = self.locked_grasp_pose if self.locked_grasp_pose is not None else self.grasp
        if base_grasp is None:
            return None
        out = copy.deepcopy(base_grasp)
        if self.latest_ee is None:
            return out
        out.pose.position.x = float(self.latest_ee.position.x)
        out.pose.position.y = float(self.latest_ee.position.y)
        return out

    def _update_progress(self, target_pose, now):
        dist = float(np.linalg.norm(_as_np_pos(self.latest_ee) - _as_np_pos(target_pose)))
        if self.phase_best_distance is None or dist < (self.phase_best_distance - self.stall_progress_epsilon_m):
            self.phase_best_distance = dist
            self.phase_last_progress_time = now
        return dist

    def _stalled(self, now):
        if self.phase_last_progress_time is None:
            return False
        return (now - self.phase_last_progress_time).to_sec() >= self.stall_timeout_sec

    def _tick(self, _evt):
        if self.paused:
            self._publish_status("paused_for_home_motion")
            return

        now = rospy.Time.now()
        if self.latest_ee is None or self.pregrasp is None or self.grasp is None:
            self._publish_status(
                "waiting_for_targets ee={} pregrasp={} grasp={} endpoint_topic={} pre_topic={} grasp_topic={}".format(
                    int(self.latest_ee is not None),
                    int(self.pregrasp is not None),
                    int(self.grasp is not None),
                    self.endpoint_topic,
                    self.pregrasp_topic,
                    self.grasp_topic,
                )
            )
            return

        if self.last_joy_time is None:
            self._publish_status("waiting_for_joy topic={}".format(self.joy_topic))
            return

        if not self._selected_target_matches_phase():
            required_categories = sorted(self._required_categories_for_phase())
            self.exec_pregrasp = None
            self.exec_grasp = None
            self.last_cmd_time = None
            self._set_state("WAIT_PREGRASP_CONFIRM")
            self._publish_status(
                "waiting_for_{}_target current_label={}".format(
                    "_or_".join(required_categories) if required_categories else "valid",
                    self.grasp_complete_label or "none",
                )
            )
            return

        if self._pressed_edge(self.cancel_button_index):
            current_stage = "grasp" if self.state in ("WAIT_GRASP_CONFIRM", "EXEC_GRASP", "WAIT_CLOSE_A", "WAIT_OPEN_B") else "pregrasp"
            self._set_state("WAIT_PREGRASP_CONFIRM")
            self.exec_pregrasp = None
            self.exec_grasp = None
            self.locked_grasp_label = ""
            self.locked_grasp_pose = None
            self.last_cmd_time = None
            self._publish_current_pose_hold_goal(now)
            self._publish_study_event(
                "confirm_cancel",
                grasp_id=self.grasp_complete_label or "",
                stage=current_stage,
            )
            self._publish_status("cancelled_wait_pregrasp")
            return

        if self.state == "WAIT_PREGRASP_CONFIRM":
            if self._is_destination_phase():
                self._publish_status("Move above placement target? Press X to continue, Y to cancel.")
            else:
                self._publish_status("Execute pregrasp? Press X to continue, Y to cancel.")
            if self._pressed_edge(self.confirm_button_index):
                self._maybe_accept_loaded_target()
            return

        if self.state == "EXEC_PREGRASP":
            target_pre = self.exec_pregrasp.pose if self.exec_pregrasp is not None else self.pregrasp.pose
            cmd = self._build_cmd_pose(target_pre, now)
            self._publish_goal(cmd, now)
            dist = self._update_progress(target_pre, now)
            self._publish_status("Executing pregrasp... dist={:.3f}m".format(dist))
            if self._at_target(target_pre, pos_tol=self.pregrasp_pos_tol, rot_tol=self.pregrasp_rot_tol):
                self._publish_study_event(
                    "auto_complete",
                    grasp_id=self.grasp_complete_label or "",
                    stage="pregrasp",
                )
                if self._is_destination_phase():
                    self._set_state("WAIT_OPEN_B")
                else:
                    self._set_state("WAIT_GRASP_CONFIRM")
            elif self._stalled(now):
                self.exec_pregrasp = None
                self._set_state("WAIT_PREGRASP_CONFIRM")
                self.last_cmd_time = None
                self._publish_current_pose_hold_goal(now)
                self._publish_study_event(
                    "auto_stalled",
                    grasp_id=self.grasp_complete_label or "",
                    stage="pregrasp",
                )
                self._publish_status("pregrasp_stalled_move_joystick_or_press_x_to_retry")
            return

        if self.state == "WAIT_OPEN_B":
            self._publish_status(
                "At release pose. Make any slight joystick height adjustment now, then press B to open gripper and release object."
            )
            if self._pressed_edge(self.open_button_index):
                self._publish_study_event(
                    "release_complete",
                    grasp_id=self.grasp_complete_label or "",
                    stage="release",
                )
                self.exec_state_pub.publish(String(data="release_complete:{}".format(self.grasp_complete_label)))
                self._set_state("DONE")
                self.last_cmd_time = None
                self._publish_current_pose_hold_goal(now)
            return

        if self.state == "WAIT_GRASP_CONFIRM":
            self._publish_status(
                "At pregrasp. Make any slight joystick adjustment now, then press X to execute the grasp motion. Press Y to cancel."
            )
            if self._pressed_edge(self.confirm_button_index):
                self.exec_grasp = (
                    self._build_locked_grasp_target()
                    if self._use_tag_aligned_grasp_motion()
                    else self._build_vertical_grasp_from_current_xy()
                )
                self._set_state("EXEC_GRASP")
                self._publish_study_event(
                    "confirm_accept",
                    grasp_id=self.grasp_complete_label or "",
                    stage="grasp",
                )
                self._publish_study_event(
                    "auto_start",
                    grasp_id=self.grasp_complete_label or "",
                    stage="grasp",
                )
                self._publish_status("confirmed_grasp_start")
            return

        if self.state == "EXEC_GRASP":
            target_grasp = self.exec_grasp.pose if self.exec_grasp is not None else self.grasp.pose
            cmd = self._build_cmd_pose(target_grasp, now)
            self._publish_goal(cmd, now)
            dist = self._update_progress(target_grasp, now)
            self._publish_status("Executing grasp... dist={:.3f}m".format(dist))
            if self._at_target(target_grasp):
                self._publish_study_event(
                    "auto_complete",
                    grasp_id=self.grasp_complete_label or "",
                    stage="grasp",
                )
                self._set_state("WAIT_CLOSE_A")
            elif self._stalled(now):
                self.exec_grasp = None
                self._set_state("WAIT_GRASP_CONFIRM")
                self.last_cmd_time = None
                self._publish_current_pose_hold_goal(now)
                self._publish_study_event(
                    "auto_stalled",
                    grasp_id=self.grasp_complete_label or "",
                    stage="grasp",
                )
                self._publish_status("grasp_stalled_move_joystick_or_press_x_to_retry")
            return

        if self.state == "WAIT_CLOSE_A":
            self._publish_status("At grasp pose. Press A to close gripper.")
            if self._pressed_edge(self.close_button_index):
                self._publish_study_event(
                    "grasp_complete",
                    grasp_id=self.grasp_complete_label or "",
                    stage="grasp",
                )
                self.exec_state_pub.publish(String(data="grasp_complete:{}".format(self.grasp_complete_label)))
                self._set_state("DONE")
                self.last_cmd_time = None
                self._publish_current_pose_hold_goal(now)
            return

        self._publish_status("Done. Returning control to shared autonomy...")
        if (now - self.state_started_at).to_sec() >= self.done_reset_sec:
            self.exec_pregrasp = None
            self.exec_grasp = None
            self.locked_grasp_label = ""
            self.locked_grasp_pose = None
            self.last_cmd_time = None
            self._set_state("WAIT_PREGRASP_CONFIRM")
            self._publish_current_pose_hold_goal(now)
            self._publish_status("ready_for_next_target")


if __name__ == "__main__":
    AprilTagFilteredExecutor()
    rospy.spin()
