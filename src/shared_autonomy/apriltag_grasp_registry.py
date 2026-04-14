#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record and retain AprilTag grasp candidates discovered during scanning."""

import copy
import math

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def _quat_angle_deg(a, b):
    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
    dot = max(0.0, min(1.0, float(dot)))
    return math.degrees(2.0 * math.acos(dot))


def _pose_distance(a, b):
    dx = float(a.position.x - b.position.x)
    dy = float(a.position.y - b.position.y)
    dz = float(a.position.z - b.position.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class AprilTagGraspRegistry:
    def __init__(self):
        rospy.init_node("apriltag_grasp_registry")

        self.base_frame = str(rospy.get_param("~base_frame", "base")).strip()
        self.input_topic = str(rospy.get_param("~input_topic", "/apriltag_candidate_manager/detections")).strip()
        self.output_topic = str(rospy.get_param("~output_topic", "/apriltag_grasp_registry/detections")).strip()
        self.status_topic = str(rospy.get_param("~status_topic", "~status")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))
        self.stable_count_required = int(rospy.get_param("~stable_count_required", 3))
        self.max_pos_jump_m = float(rospy.get_param("~max_pos_jump_m", 0.05))
        self.max_rot_jump_deg = float(rospy.get_param("~max_rot_jump_deg", 20.0))
        self.overwrite_recorded = bool(rospy.get_param("~overwrite_recorded", True))

        self.observed = {}
        self.recorded = {}
        self.last_status = ""

        self.pub = rospy.Publisher(self.output_topic, Detection2DArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)

        rospy.Subscriber(self.input_topic, Detection2DArray, self._input_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(1.0, self.publish_rate_hz)), self._timer_cb)
        self._publish_status("waiting_for_scan_candidates")

    def _publish_status(self, text):
        if text == self.last_status:
            return
        self.last_status = text
        rospy.loginfo("[apriltag_grasp_registry] %s", text)
        self.status_pub.publish(String(data=text))

    def _input_cb(self, msg):
        seen_ids = []
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            tag_id = int(hyp.id)
            seen_ids.append(tag_id)
            pose = hyp.pose.pose

            obs = self.observed.get(tag_id)
            if obs is None:
                self.observed[tag_id] = {
                    "pose": copy.deepcopy(pose),
                    "stable_count": 1,
                }
            else:
                pos_jump = _pose_distance(obs["pose"], pose)
                rot_jump = _quat_angle_deg(obs["pose"].orientation, pose.orientation)
                if pos_jump <= self.max_pos_jump_m and rot_jump <= self.max_rot_jump_deg:
                    obs["stable_count"] += 1
                else:
                    obs["stable_count"] = 1
                obs["pose"] = copy.deepcopy(pose)

            obs = self.observed[tag_id]
            if obs["stable_count"] >= self.stable_count_required:
                if tag_id not in self.recorded or self.overwrite_recorded:
                    self.recorded[tag_id] = copy.deepcopy(obs["pose"])

        self._publish_status(
            "scanning seen={} recorded={}".format(
                sorted(seen_ids),
                sorted(self.recorded.keys()),
            )
        )

    def _timer_cb(self, _evt):
        now = rospy.Time.now()
        out = Detection2DArray()
        out.header.stamp = now
        out.header.frame_id = self.base_frame

        for tag_id in sorted(self.recorded.keys()):
            pose = self.recorded[tag_id]
            det = Detection2D()
            det.header = out.header
            hyp = ObjectHypothesisWithPose()
            hyp.id = int(tag_id)
            hyp.score = 1.0
            hyp.pose.pose = copy.deepcopy(pose)
            det.results.append(hyp)
            out.detections.append(det)

        self.pub.publish(out)


if __name__ == "__main__":
    AprilTagGraspRegistry()
    rospy.spin()
