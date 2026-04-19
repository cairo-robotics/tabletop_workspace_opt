#!/usr/bin/env python3
"""Log synchronized intent + grasp-selection samples for offline training.

This logger is designed for the MuJoCo shared-autonomy pipeline. It records the
latest RGB image and the most recent object-intent, candidate-grasp, and
selector outputs whenever a new selected grasp arrives, with a configurable
minimum period between samples.
"""

import json
import os
from typing import Dict, List, Optional

import cv2
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from tabletop_workspace_opt.msg import GraspCandidate, GraspCandidateArray, GraspScoreArray
from vision_msgs.msg import Detection2DArray

try:
    from relaxed_ik_ros1.msg import EEVelGoals
except ImportError:
    EEVelGoals = None


def _time_to_nsecs(stamp: rospy.Time) -> int:
    return int(stamp.to_nsec()) if stamp != rospy.Time() else int(rospy.Time.now().to_nsec())


def _point_dict(point) -> Dict[str, float]:
    return {
        "x": float(point.x),
        "y": float(point.y),
        "z": float(point.z),
    }


def _quat_dict(quat) -> Dict[str, float]:
    return {
        "x": float(quat.x),
        "y": float(quat.y),
        "z": float(quat.z),
        "w": float(quat.w),
    }


def _pose_dict(pose) -> Dict[str, Dict[str, float]]:
    return {
        "position": _point_dict(pose.position),
        "orientation": _quat_dict(pose.orientation),
    }


class IntentGraspDatasetLogger:
    def __init__(self):
        rospy.init_node("intent_grasp_dataset_logger")

        self.output_dir = os.path.abspath(os.path.expanduser(rospy.get_param("~output_dir", "/tmp/intent_grasp_dataset")))
        self.images_dir = os.path.join(self.output_dir, "images")
        self.records_path = os.path.join(self.output_dir, "records.jsonl")
        self.task_instruction = str(rospy.get_param("~task_instruction", "")).strip()
        self.image_format = str(rospy.get_param("~image_format", "png")).strip().lower() or "png"
        self.min_sample_period_sec = float(rospy.get_param("~min_sample_period_sec", 1.0))
        self.write_on_selected_grasp = bool(rospy.get_param("~write_on_selected_grasp", True))

        self.image_topic = str(rospy.get_param("~image_topic", "/realsense/color/image_raw")).strip()
        self.detections_topic = str(rospy.get_param("~detections_topic", "/mujoco_sim/detections")).strip()
        self.intent_distribution_topic = str(
            rospy.get_param("~intent_distribution_topic", "/intent_inference/distribution")
        ).strip()
        self.intent_top_goal_topic = str(rospy.get_param("~intent_top_goal_topic", "/intent_inference/top_goal")).strip()
        self.intent_top_pose_topic = str(rospy.get_param("~intent_top_pose_topic", "/intent_inference/top_pose")).strip()
        self.candidate_grasps_topic = str(
            rospy.get_param("~candidate_grasps_topic", "/shared_autonomy/candidate_grasps")
        ).strip()
        self.grasp_scores_topic = str(rospy.get_param("~grasp_scores_topic", "/shared_autonomy/grasp_scores")).strip()
        self.selected_grasp_topic = str(rospy.get_param("~selected_grasp_topic", "/shared_autonomy/selected_grasp")).strip()
        self.ee_vel_goals_topic = str(rospy.get_param("~ee_vel_goals_topic", "")).strip()

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self._records_handle = open(self.records_path, "a", encoding="utf-8")

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_image_stamp = rospy.Time()
        self.latest_image_frame_id = ""
        self.latest_detections: Optional[Detection2DArray] = None
        self.latest_intent_distribution: List[float] = []
        self.latest_intent_stamp = rospy.Time()
        self.latest_top_goal = ""
        self.latest_top_pose: Optional[PoseStamped] = None
        self.latest_candidates: Optional[GraspCandidateArray] = None
        self.latest_scores: Optional[GraspScoreArray] = None
        self.latest_input_vector: Optional[Dict[str, float]] = None
        self.last_write_time = rospy.Time(0)

        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.detections_topic, Detection2DArray, self.detections_cb, queue_size=1)
        rospy.Subscriber(self.intent_distribution_topic, Float32MultiArray, self.intent_distribution_cb, queue_size=1)
        rospy.Subscriber(self.intent_top_goal_topic, String, self.intent_top_goal_cb, queue_size=1)
        rospy.Subscriber(self.intent_top_pose_topic, PoseStamped, self.intent_top_pose_cb, queue_size=1)
        rospy.Subscriber(self.candidate_grasps_topic, GraspCandidateArray, self.candidate_cb, queue_size=1)
        rospy.Subscriber(self.grasp_scores_topic, GraspScoreArray, self.score_cb, queue_size=1)
        rospy.Subscriber(self.selected_grasp_topic, GraspCandidate, self.selected_grasp_cb, queue_size=1)
        if self.ee_vel_goals_topic and EEVelGoals is not None:
            rospy.Subscriber(self.ee_vel_goals_topic, EEVelGoals, self.ee_vel_goals_cb, queue_size=1)

        rospy.on_shutdown(self._close_records)

        rospy.loginfo(
            "Intent grasp dataset logger ready. output_dir=%s image=%s detections=%s intent=%s candidates=%s selected=%s",
            self.output_dir,
            self.image_topic,
            self.detections_topic,
            self.intent_distribution_topic,
            self.candidate_grasps_topic,
            self.selected_grasp_topic,
        )

    def _close_records(self):
        if not self._records_handle.closed:
            self._records_handle.close()

    def image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
            self.latest_image_frame_id = str(msg.header.frame_id)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to decode logger image: %s", exc)

    def detections_cb(self, msg: Detection2DArray):
        self.latest_detections = msg

    def intent_distribution_cb(self, msg: Float32MultiArray):
        self.latest_intent_distribution = [float(v) for v in msg.data]
        self.latest_intent_stamp = rospy.Time.now()

    def intent_top_goal_cb(self, msg: String):
        self.latest_top_goal = str(msg.data)

    def intent_top_pose_cb(self, msg: PoseStamped):
        self.latest_top_pose = msg

    def candidate_cb(self, msg: GraspCandidateArray):
        self.latest_candidates = msg

    def score_cb(self, msg: GraspScoreArray):
        self.latest_scores = msg

    def ee_vel_goals_cb(self, msg):
        if not msg.ee_vels:
            return
        twist = msg.ee_vels[0]
        self.latest_input_vector = {
            "x": float(twist.linear.x),
            "y": float(twist.linear.y),
            "z": float(twist.linear.z),
        }

    def selected_grasp_cb(self, msg: GraspCandidate):
        if not self.write_on_selected_grasp:
            return

        now = rospy.Time.now()
        if self.last_write_time != rospy.Time(0):
            elapsed = (now - self.last_write_time).to_sec()
            if elapsed < self.min_sample_period_sec:
                return

        if self.latest_image is None or self.latest_candidates is None:
            rospy.logwarn_throttle(
                2.0,
                "Skipping dataset sample: waiting for image and candidate grasp messages.",
            )
            return

        record = self._build_record(msg, now)
        if record is None:
            return

        self._records_handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._records_handle.flush()
        self.last_write_time = now
        rospy.loginfo(
            "Logged dataset sample %s selected_grasp=%s top_goal=%s candidates=%d",
            record["sample_id"],
            record["selected_grasp_id"],
            record["intent_top_goal"],
            len(record["candidate_grasps"]),
        )

    def _build_record(self, selected_grasp: GraspCandidate, stamp: rospy.Time) -> Optional[Dict[str, object]]:
        sample_id = str(_time_to_nsecs(stamp))
        image_filename = f"{sample_id}.{self.image_format}"
        image_path = os.path.join(self.images_dir, image_filename)

        if not cv2.imwrite(image_path, self.latest_image):
            rospy.logwarn("Failed to write dataset image to %s", image_path)
            return None

        detections = self._serialize_detections()
        scores_by_id = self._score_map()

        candidate_records = []
        for candidate in self.latest_candidates.grasps:
            candidate_record = {
                "grasp_id": str(candidate.grasp_id),
                "pose": _pose_dict(candidate.pose),
                "approach_direction": _point_dict(candidate.approach_direction),
                "grasp_score": float(candidate.grasp_score),
                "feasible": bool(candidate.feasible),
                "label_selected": bool(candidate.grasp_id == selected_grasp.grasp_id),
            }
            if candidate.grasp_id in scores_by_id:
                candidate_record["selector_scores"] = scores_by_id[candidate.grasp_id]
            candidate_records.append(candidate_record)

        record = {
            "sample_id": sample_id,
            "stamp_ns": _time_to_nsecs(stamp),
            "task_instruction": self.task_instruction,
            "image_path": os.path.relpath(image_path, self.output_dir),
            "image_frame_id": self.latest_image_frame_id,
            "image_stamp_ns": _time_to_nsecs(self.latest_image_stamp),
            "selected_grasp_id": str(selected_grasp.grasp_id),
            "selected_grasp_pose": _pose_dict(selected_grasp.pose),
            "detections": detections,
            "intent_distribution": self.latest_intent_distribution,
            "intent_top_goal": self.latest_top_goal,
            "intent_top_pose": _pose_dict(self.latest_top_pose.pose) if self.latest_top_pose is not None else None,
            "input_vector": self.latest_input_vector,
            "candidate_grasps": candidate_records,
        }
        return record

    def _serialize_detections(self) -> List[Dict[str, object]]:
        if self.latest_detections is None:
            return []

        serialized = []
        for detection in self.latest_detections.detections:
            if not detection.results:
                continue
            result = detection.results[0]
            serialized.append(
                {
                    "label": str(result.id),
                    "score": float(getattr(result, "score", 0.0)),
                    "pose": _pose_dict(result.pose.pose),
                }
            )
        if self.latest_intent_distribution:
            for index, prob in enumerate(self.latest_intent_distribution):
                if index < len(serialized):
                    serialized[index]["intent_probability"] = float(prob)
        return serialized

    def _score_map(self) -> Dict[str, Dict[str, object]]:
        if self.latest_scores is None:
            return {}

        score_map = {}
        for item in self.latest_scores.scores:
            score_map[str(item.grasp_id)] = {
                "candidate_index": int(item.candidate_index),
                "total_score": float(item.total_score),
                "alignment_score": float(item.alignment_score),
                "grasp_score": float(item.grasp_score),
                "feasibility_score": float(item.feasibility_score),
                "distance_to_grasp": float(item.distance_to_grasp),
                "feasible": bool(item.feasible),
            }
        return score_map


def main():
    IntentGraspDatasetLogger()
    rospy.spin()


if __name__ == "__main__":
    main()
