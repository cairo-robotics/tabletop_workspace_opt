#!/usr/bin/env python3
"""Publish a small fake visual grasp correction for executor integration demos."""

import math

import rospy
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String


def _parse_xyz(text, default):
    parts = [chunk.strip() for chunk in str(text).split(",") if chunk.strip()]
    if len(parts) != 3:
        return default
    try:
        return tuple(float(v) for v in parts)
    except Exception:
        return default


class FakeVisualGraspRefineNode:
    def __init__(self):
        rospy.init_node("fake_visual_grasp_refine")
        self.offset_topic = str(rospy.get_param("~offset_topic", "/visual_grasp_refine/offset")).strip()
        self.execution_state_topic = str(
            rospy.get_param("~execution_state_topic", "/intent_inference/execution_state")
        ).strip()
        self.selected_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.initial_offset = _parse_xyz(
            rospy.get_param("~initial_offset_xyz", "0.015,-0.010,0.0"),
            (0.015, -0.010, 0.0),
        )
        self.duration_sec = max(0.1, float(rospy.get_param("~duration_sec", 2.0)))
        self.rate_hz = max(1.0, float(rospy.get_param("~rate_hz", 20.0)))
        self.active_states = {
            chunk.strip().lower()
            for chunk in str(rospy.get_param("~active_states", "exec_pregrasp,visual_align")).split(",")
            if chunk.strip()
        }

        self.execution_state = ""
        self.selected_label = ""
        self.sequence_label = ""
        self.sequence_start = None
        self.pub = rospy.Publisher(self.offset_topic, Vector3Stamped, queue_size=1)
        rospy.Subscriber(self.execution_state_topic, String, self._state_cb, queue_size=1)
        rospy.Subscriber(self.selected_label_topic, String, self._label_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)
        rospy.loginfo(
            "[fake_visual_grasp_refine] publishing %s from state=%s label=%s",
            self.offset_topic,
            self.execution_state_topic,
            self.selected_label_topic,
        )

    def _state_cb(self, msg):
        self.execution_state = str(msg.data).strip().lower()
        if self.execution_state not in self.active_states:
            self.sequence_start = None

    def _label_cb(self, msg):
        next_label = str(msg.data).strip()
        if next_label != self.selected_label:
            self.sequence_start = None
            self.sequence_label = ""
        self.selected_label = next_label

    def _tick(self, _evt):
        if self.execution_state not in self.active_states or not self.selected_label:
            return
        now = rospy.Time.now()
        if self.execution_state == "exec_pregrasp":
            scale = 1.0
            self.sequence_start = None
            self.sequence_label = self.selected_label
        else:
            if self.sequence_start is None or self.sequence_label != self.selected_label:
                self.sequence_start = now
                self.sequence_label = self.selected_label
            elapsed = max(0.0, (now - self.sequence_start).to_sec())
            scale = max(0.0, 1.0 - elapsed / self.duration_sec)
            if scale > 0.0:
                scale = 0.5 * (1.0 + math.cos(math.pi * (1.0 - scale)))
        if self.sequence_start is None and self.execution_state != "exec_pregrasp":
            self.sequence_start = now
            self.sequence_label = self.selected_label
        msg = Vector3Stamped()
        msg.header.stamp = now
        msg.header.frame_id = self.base_frame
        msg.vector.x = float(self.initial_offset[0] * scale)
        msg.vector.y = float(self.initial_offset[1] * scale)
        msg.vector.z = float(self.initial_offset[2] * scale)
        self.pub.publish(msg)


if __name__ == "__main__":
    FakeVisualGraspRefineNode()
    rospy.spin()
