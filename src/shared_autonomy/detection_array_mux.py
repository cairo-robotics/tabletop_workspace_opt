#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge multiple Detection2DArray topics into a single candidate list."""

import copy

import rospy
from intera_core_msgs.msg import EndpointState
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


class DetectionArrayMux:
    def __init__(self):
        rospy.init_node("detection_array_mux")

        self.output_topic = str(rospy.get_param("~output_topic", "/hybrid_grasp_candidates/detections")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.stale_timeout_sec = float(rospy.get_param("~stale_timeout_sec", 1.0))
        self.task_phase_topic = str(rospy.get_param("~task_phase_topic", "/task_context/phase")).strip()
        self.command_topic = str(rospy.get_param("~command_topic", "/task_context/command")).strip()
        self.end_effector_topic = str(
            rospy.get_param("~end_effector_topic", "/robot/limb/right/endpoint_state")
        ).strip()
        self.auto_single_lego_pick = bool(rospy.get_param("~auto_single_lego_pick", True))
        raw_lego_ids = rospy.get_param("~lego_candidate_ids", [20, 21, 22, 23])
        if isinstance(raw_lego_ids, str):
            text = raw_lego_ids.strip().replace("[", "").replace("]", "").replace(",", " ")
            raw_lego_ids = [s for s in text.split() if s]
        self.lego_candidate_ids = set(int(v) for v in raw_lego_ids)
        self.lego_switch_margin_m = float(rospy.get_param("~lego_switch_margin_m", 0.04))

        raw_sources = rospy.get_param(
            "~input_topics",
            ["/apriltag_grasp_registry/detections", "/sam_lego_grasp/detections"],
        )
        if isinstance(raw_sources, str):
            text = raw_sources.strip().replace("[", "").replace("]", "").replace(",", " ")
            raw_sources = [s for s in text.split() if s]
        self.sources = [str(s).strip() for s in raw_sources if str(s).strip()]
        self.latest_msgs = {}
        self.latest_times = {}
        self.last_status = ""
        self.current_phase = "scan_workspace"
        self.latest_ee = None
        self.selected_lego_id = None
        self.excluded_ids = set()
        raw_clear_commands = rospy.get_param("~clear_commands", ["scan_workspace", "scan", "reset_task", "reset"])
        if isinstance(raw_clear_commands, str):
            text = raw_clear_commands.strip().replace("[", "").replace("]", "").replace(",", " ")
            raw_clear_commands = [s for s in text.split() if s]
        self.clear_commands = set(str(s).strip().lower() for s in raw_clear_commands if str(s).strip())

        self.pub = rospy.Publisher(self.output_topic, Detection2DArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        for topic in self.sources:
            rospy.Subscriber(topic, Detection2DArray, self._cb, callback_args=topic, queue_size=1)
        rospy.Subscriber(self.task_phase_topic, String, self._phase_cb, queue_size=1)
        rospy.Subscriber(self.command_topic, String, self._command_cb, queue_size=10)
        rospy.Subscriber(self.end_effector_topic, EndpointState, self._ee_cb, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._tick)
        self._publish_status("waiting_for_sources topics={}".format(self.sources))

    def _cb(self, msg, topic):
        self.latest_msgs[topic] = copy.deepcopy(msg)
        self.latest_times[topic] = rospy.Time.now()

    def _phase_cb(self, msg):
        phase = str(msg.data).strip().lower() or "scan_workspace"
        if phase != self.current_phase:
            self.current_phase = phase
            if phase not in ("select_lego_brick", "lego_brick"):
                self.selected_lego_id = None

    def _ee_cb(self, msg):
        self.latest_ee = msg.pose

    def _command_cb(self, msg):
        cmd = str(msg.data).strip().lower()
        if not cmd:
            return
        if cmd in self.clear_commands:
            self.excluded_ids.clear()
            return
        if cmd.startswith("remove_tag:"):
            try:
                tag_id = int(cmd.split(":", 1)[1].strip())
            except Exception:
                return
            self.excluded_ids.add(tag_id)

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[detection_array_mux] %s", text)
        self.status_pub.publish(String(data=text))

    def _tick(self, _evt):
        now = rospy.Time.now()
        merged = []
        active_topics = []
        header = None
        seen_ids = set()

        for topic in self.sources:
            msg = self.latest_msgs.get(topic)
            updated = self.latest_times.get(topic, rospy.Time(0))
            if msg is None or updated == rospy.Time(0):
                continue
            if (now - updated).to_sec() > self.stale_timeout_sec:
                continue
            active_topics.append(topic)
            if header is None:
                header = copy.deepcopy(msg.header)
            for det in msg.detections:
                if not det.results:
                    continue
                try:
                    det_id = int(det.results[0].id)
                except Exception:
                    continue
                if det_id in self.excluded_ids:
                    continue
                if det_id in seen_ids:
                    continue
                seen_ids.add(det_id)
                merged.append(copy.deepcopy(det))

        merged = self._filter_for_phase(merged)
        merged.sort(key=lambda det: int(det.results[0].id) if det.results else 0)

        out = Detection2DArray()
        if header is not None:
            out.header = header
        else:
            out.header.stamp = now
        out.detections = merged
        self.pub.publish(out)
        published_ids = []
        for det in merged:
            if det.results:
                try:
                    published_ids.append(int(det.results[0].id))
                except Exception:
                    pass
        self._publish_status(
            "active_sources={} phase={} merged_ids={} selected_lego={}".format(
                active_topics,
                self.current_phase,
                published_ids,
                self.selected_lego_id,
            )
        )

    def _filter_for_phase(self, detections):
        if not self.auto_single_lego_pick:
            return detections
        if self.current_phase not in ("select_lego_brick", "lego_brick"):
            return detections
        lego_dets = []
        other_dets = []
        for det in detections:
            if not det.results:
                continue
            try:
                det_id = int(det.results[0].id)
            except Exception:
                other_dets.append(det)
                continue
            if det_id in self.lego_candidate_ids:
                lego_dets.append(det)
            else:
                other_dets.append(det)
        if not lego_dets:
            self.selected_lego_id = None
            return other_dets
        chosen = self._choose_single_lego_detection(lego_dets)
        if chosen is None:
            self.selected_lego_id = None
            return other_dets
        try:
            self.selected_lego_id = int(chosen.results[0].id)
        except Exception:
            self.selected_lego_id = None
        return other_dets + [chosen]

    def _choose_single_lego_detection(self, lego_dets):
        if len(lego_dets) == 1 or self.latest_ee is None:
            return copy.deepcopy(lego_dets[0])

        ee = self.latest_ee.position
        distances = {}
        by_id = {}
        for det in lego_dets:
            try:
                det_id = int(det.results[0].id)
            except Exception:
                continue
            pose = det.results[0].pose.pose.position
            dx = float(pose.x - ee.x)
            dy = float(pose.y - ee.y)
            dz = float(pose.z - ee.z)
            distances[det_id] = (dx * dx + dy * dy + dz * dz) ** 0.5
            by_id[det_id] = det
        if not distances:
            return copy.deepcopy(lego_dets[0])

        best_id = min(distances, key=distances.get)
        if self.selected_lego_id in distances:
            current_dist = distances[self.selected_lego_id]
            best_dist = distances[best_id]
            if current_dist <= best_dist + self.lego_switch_margin_m:
                best_id = self.selected_lego_id
        return copy.deepcopy(by_id[best_id])


if __name__ == "__main__":
    DetectionArrayMux()
    rospy.spin()
