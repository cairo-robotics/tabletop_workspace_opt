#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execute filtered AprilTag pregrasp/grasp with joystick confirmation."""

import copy
import math
import os

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from intera_core_msgs.msg import EndpointState
from relaxed_ik_ros1.msg import EEPoseGoals
from sensor_msgs.msg import Joy
from std_msgs.msg import String
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
        self.task_phase_topic = str(rospy.get_param("~task_phase_topic", "/task_context/phase")).strip()
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
        self.max_speed_mps = float(rospy.get_param("~max_speed_mps", 0.10))
        self.max_angular_step = float(rospy.get_param("~max_angular_step", 0.12))
        self.pos_tol = float(rospy.get_param("~position_tolerance_m", 0.01))
        self.rot_tol = float(rospy.get_param("~orientation_tolerance_rad", 0.25))
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

        self.latest_ee = None
        self.latest_axes = []
        self.latest_buttons = []
        self.prev_buttons = []
        self.last_joy_time = None
        self.pregrasp = None
        self.grasp = None
        self.exec_pregrasp = None
        self.exec_grasp = None
        self.state = "WAIT_PREGRASP_CONFIRM"
        self.state_started_at = rospy.Time.now()
        self.phase_best_distance = None
        self.phase_last_progress_time = self.state_started_at
        self.last_cmd_time = None
        self.last_status = ""
        self._cv_window_initialized = False
        self.current_phase = "scan_workspace"
        self.label_to_meta = self._load_label_metadata()

        self.goal_pub = rospy.Publisher("/relaxed_ik/ee_pose_goals", EEPoseGoals, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.prompt_pub = rospy.Publisher(self.prompt_topic, String, queue_size=1, latch=True)
        self.exec_state_pub = rospy.Publisher(self.execution_state_topic, String, queue_size=1, latch=True)
        self.prompt_marker_pub = rospy.Publisher(self.prompt_marker_topic, Marker, queue_size=1, latch=True)

        rospy.Subscriber(self.endpoint_topic, EndpointState, self._ee_cb, queue_size=10)
        rospy.Subscriber(self.pregrasp_topic, PoseStamped, self._pre_cb, queue_size=1)
        rospy.Subscriber(self.grasp_topic, PoseStamped, self._grasp_cb, queue_size=1)
        rospy.Subscriber(self.joy_topic, Joy, self._joy_cb, queue_size=10)
        rospy.Subscriber(self.selected_grasp_label_topic, String, self._selected_grasp_label_cb, queue_size=1)
        rospy.Subscriber(self.task_phase_topic, String, self._task_phase_cb, queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self._tick)
        rospy.Timer(rospy.Duration(0.5), self._guard)
        rospy.Timer(rospy.Duration(0.1), self._ui_tick)
        rospy.on_shutdown(self._shutdown)

        self._init_status_window()
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
            self.state = new_state
            self.state_started_at = rospy.Time.now()
            self.phase_best_distance = None
            self.phase_last_progress_time = self.state_started_at

    def _load_label_metadata(self):
        if not os.path.exists(self.object_map_yaml):
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        tag_objects = data.get("tag_objects", {}) if isinstance(data, dict) else {}
        mapping = {}
        for meta in tag_objects.values():
            if not isinstance(meta, dict):
                continue
            label = str(meta.get("grasp_complete_label", "")).strip()
            if label:
                mapping[label] = meta
        return mapping

    def _selected_grasp_label_cb(self, msg):
        selected_label = str(msg.data).strip()
        if not selected_label or selected_label == self.grasp_complete_label:
            return
        self.grasp_complete_label = selected_label
        rospy.loginfo(
            "[apriltag_filtered_executor] updated grasp_complete_label to %s",
            self.grasp_complete_label,
        )

    def _task_phase_cb(self, msg):
        self.current_phase = str(msg.data).strip().lower() or "scan_workspace"

    def _required_categories_for_phase(self):
        if self.current_phase in ("grasp_milk", "select_milk", "milk"):
            return {"milk"}
        if self.current_phase in ("grasp_condiment", "select_condiment", "condiment"):
            return {"cereal", "chocolate"}
        if self.current_phase in ("grasp_cereal", "select_cereal", "cereal"):
            return {"cereal"}
        if self.current_phase in ("grasp_chocolate", "select_chocolate", "chocolate"):
            return {"chocolate"}
        return set()

    def _selected_target_matches_phase(self):
        required_categories = self._required_categories_for_phase()
        if not required_categories:
            return True
        meta = self.label_to_meta.get(self.grasp_complete_label, {})
        actual_category = str(meta.get("category", "")).strip().lower()
        return actual_category in required_categories

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
            "X:{}  Y:{}  A:{}".format(self.confirm_button_index, self.cancel_button_index, self.close_button_index),
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

    def _grasp_cb(self, msg):
        self.grasp = copy.deepcopy(msg)

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

        if not self.use_goal_orientation:
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

    def _at_target(self, target_pose):
        cur = self.latest_ee
        if cur is None:
            return False
        cur_p = _as_np_pos(cur)
        tgt_p = _as_np_pos(target_pose)
        dist = float(np.linalg.norm(cur_p - tgt_p))
        if not self.use_goal_orientation:
            return dist <= self.pos_tol
        ang = _quat_angle_rad(
            [cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w],
            [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w],
        )
        return dist <= self.pos_tol and ang <= self.rot_tol

    def _publish_goal(self, pose, now):
        msg = EEPoseGoals()
        msg.header.stamp = now
        msg.header.frame_id = self.base_frame
        msg.ee_poses.append(pose)
        self.goal_pub.publish(msg)

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
            self._set_state("WAIT_PREGRASP_CONFIRM")
            self.exec_pregrasp = None
            self.exec_grasp = None
            self._publish_status("cancelled_wait_pregrasp")
            return

        if self.state == "WAIT_PREGRASP_CONFIRM":
            self._publish_status("Execute pregrasp? Press X to continue, Y to cancel.")
            if self._pressed_edge(self.confirm_button_index):
                self.exec_pregrasp = copy.deepcopy(self.pregrasp)
                self.exec_grasp = None
                self._set_state("EXEC_PREGRASP")
                self._publish_status("confirmed_pregrasp_start")
            return

        if self.state == "EXEC_PREGRASP":
            target_pre = self.exec_pregrasp.pose if self.exec_pregrasp is not None else self.pregrasp.pose
            cmd = self._build_cmd_pose(target_pre, now)
            self._publish_goal(cmd, now)
            dist = self._update_progress(target_pre, now)
            self._publish_status("Executing pregrasp... dist={:.3f}m".format(dist))
            if self._at_target(target_pre):
                self._set_state("WAIT_GRASP_CONFIRM")
            elif self._stalled(now):
                self.exec_pregrasp = None
                self._set_state("WAIT_PREGRASP_CONFIRM")
                self._publish_status("pregrasp_stalled_move_joystick_or_press_x_to_retry")
            return

        if self.state == "WAIT_GRASP_CONFIRM":
            self._publish_status("Execute grasp? Press X to continue, Y to cancel.")
            if self._pressed_edge(self.confirm_button_index):
                self.exec_grasp = copy.deepcopy(self.grasp)
                self._set_state("EXEC_GRASP")
                self._publish_status("confirmed_grasp_start")
            return

        if self.state == "EXEC_GRASP":
            target_grasp = self.exec_grasp.pose if self.exec_grasp is not None else self.grasp.pose
            cmd = self._build_cmd_pose(target_grasp, now)
            self._publish_goal(cmd, now)
            dist = self._update_progress(target_grasp, now)
            self._publish_status("Executing grasp... dist={:.3f}m".format(dist))
            if self._at_target(target_grasp):
                self._set_state("WAIT_CLOSE_A")
            elif self._stalled(now):
                self.exec_grasp = None
                self._set_state("WAIT_GRASP_CONFIRM")
                self._publish_status("grasp_stalled_move_joystick_or_press_x_to_retry")
            return

        if self.state == "WAIT_CLOSE_A":
            self._publish_status("At grasp pose. Press A to close gripper.")
            if self._pressed_edge(self.close_button_index):
                self.exec_state_pub.publish(String(data="grasp_complete:{}".format(self.grasp_complete_label)))
                self._set_state("DONE")
            return

        self._publish_status("Done. Returning control to shared autonomy...")
        if (now - self.state_started_at).to_sec() >= self.done_reset_sec:
            self.exec_pregrasp = None
            self.exec_grasp = None
            self.last_cmd_time = None
            self._set_state("WAIT_PREGRASP_CONFIRM")
            self._publish_status("ready_for_next_target")


if __name__ == "__main__":
    AprilTagFilteredExecutor()
    rospy.spin()
