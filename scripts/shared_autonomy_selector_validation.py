#!/usr/bin/env python3
"""Validation harness for the shared autonomy grasp selector.

Publishes synthetic end-effector pose, grasp candidates, and operator input
signals, then checks whether the selector picks the expected grasp.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, TwistStamped, Vector3
from tabletop_workspace_opt.msg import (
    GraspCandidate,
    GraspCandidateArray,
    GraspScoreArray,
)


@dataclass(frozen=True)
class CandidateSpec:
    grasp_id: str
    position: Tuple[float, float, float]
    grasp_score: float
    feasible: bool
    approach_direction: Tuple[float, float, float]


@dataclass(frozen=True)
class TestCase:
    name: str
    input_direction: Tuple[float, float, float]
    expected_grasp_id: str
    duration_sec: float = 1.0


@dataclass(frozen=True)
class Scenario:
    ee_position: Tuple[float, float, float]
    candidates: Tuple[CandidateSpec, ...]
    tests: Tuple[TestCase, ...]


SCENARIOS: Dict[str, Scenario] = {
    "basic_directional": Scenario(
        ee_position=(0.0, 0.0, 0.5),
        candidates=(
            CandidateSpec(
                grasp_id="milk_side_pour",
                position=(0.45, 0.18, 0.18),
                grasp_score=0.90,
                feasible=True,
                approach_direction=(0.0, -1.0, 0.0),
            ),
            CandidateSpec(
                grasp_id="milk_top_pick",
                position=(0.42, 0.18, 0.32),
                grasp_score=0.76,
                feasible=True,
                approach_direction=(0.0, 0.0, -1.0),
            ),
            CandidateSpec(
                grasp_id="cup_side_grasp",
                position=(0.26, -0.28, 0.16),
                grasp_score=0.82,
                feasible=True,
                approach_direction=(-1.0, 0.0, 0.0),
            ),
            CandidateSpec(
                grasp_id="rear_safe",
                position=(0.16, 0.40, 0.16),
                grasp_score=0.78,
                feasible=True,
                approach_direction=(0.0, -1.0, 0.0),
            ),
            CandidateSpec(
                grasp_id="rear_infeasible",
                position=(0.18, 0.45, 0.15),
                grasp_score=0.98,
                feasible=False,
                approach_direction=(0.0, -1.0, 0.0),
            ),
        ),
        tests=(
            TestCase(
                name="prefer_milk_side",
                input_direction=(0.83, 0.33, -0.45),
                expected_grasp_id="milk_side_pour",
            ),
            TestCase(
                name="prefer_milk_top",
                input_direction=(0.91, 0.39, -0.06),
                expected_grasp_id="milk_top_pick",
            ),
            TestCase(
                name="prefer_cup",
                input_direction=(0.58, -0.58, -0.58),
                expected_grasp_id="cup_side_grasp",
            ),
            TestCase(
                name="respect_feasibility",
                input_direction=(0.33, 0.86, -0.39),
                expected_grasp_id="rear_safe",
            ),
        ),
    ),
}


def _vector3(values: Tuple[float, float, float]) -> Vector3:
    return Vector3(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _identity_pose(position: Tuple[float, float, float]) -> Pose:
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return pose


class SharedAutonomySelectorValidation:
    def __init__(self):
        rospy.init_node("shared_autonomy_selector_validation")

        scenario_name = rospy.get_param("~scenario", "basic_directional")
        if scenario_name not in SCENARIOS:
            raise ValueError(
                "Unknown scenario '%s'. Available: %s"
                % (scenario_name, ", ".join(sorted(SCENARIOS.keys())))
            )
        self.scenario = SCENARIOS[scenario_name]

        self.frame_id = rospy.get_param("~frame_id", "world")
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))
        self.case_gap_sec = float(rospy.get_param("~case_gap_sec", 0.4))
        self.startup_delay_sec = float(rospy.get_param("~startup_delay_sec", 1.0))

        self.ee_pose_topic = rospy.get_param("~ee_pose_topic", "/selector_validation/ee_pose")
        self.candidate_topic = rospy.get_param("~candidate_grasps_topic", "/selector_validation/candidate_grasps")
        self.input_topic = rospy.get_param("~input_twist_topic", "/selector_validation/input_twist")
        self.selected_grasp_topic = rospy.get_param("~selected_grasp_topic", "/selector_validation/selected_grasp")
        self.score_topic = rospy.get_param("~grasp_scores_topic", "/selector_validation/grasp_scores")

        self.ee_pub = rospy.Publisher(self.ee_pose_topic, PoseStamped, queue_size=1)
        self.candidate_pub = rospy.Publisher(self.candidate_topic, GraspCandidateArray, queue_size=1)
        self.input_pub = rospy.Publisher(self.input_topic, TwistStamped, queue_size=1)

        rospy.Subscriber(self.selected_grasp_topic, GraspCandidate, self.selected_grasp_cb, queue_size=1)
        rospy.Subscriber(self.score_topic, GraspScoreArray, self.score_cb, queue_size=1)

        self.last_selected_grasp_id: Optional[str] = None
        self.last_score_msg: Optional[GraspScoreArray] = None

        self.ee_pose_msg = PoseStamped()
        self.ee_pose_msg.header.frame_id = self.frame_id
        self.ee_pose_msg.pose = _identity_pose(self.scenario.ee_position)

        self.candidate_msg = GraspCandidateArray()
        self.candidate_msg.header.frame_id = self.frame_id
        self.candidate_msg.grasps = [self._make_candidate_msg(spec) for spec in self.scenario.candidates]

    def _make_candidate_msg(self, spec: CandidateSpec) -> GraspCandidate:
        msg = GraspCandidate()
        msg.grasp_id = spec.grasp_id
        msg.pose = _identity_pose(spec.position)
        msg.approach_direction = _vector3(spec.approach_direction)
        msg.grasp_score = float(spec.grasp_score)
        msg.feasible = bool(spec.feasible)
        return msg

    def selected_grasp_cb(self, msg: GraspCandidate):
        self.last_selected_grasp_id = msg.grasp_id

    def score_cb(self, msg: GraspScoreArray):
        self.last_score_msg = msg

    def _publish_static_state(self):
        now = rospy.Time.now()
        self.ee_pose_msg.header.stamp = now
        self.candidate_msg.header.stamp = now
        self.ee_pub.publish(self.ee_pose_msg)
        self.candidate_pub.publish(self.candidate_msg)

    def _publish_input(self, input_direction: Tuple[float, float, float]):
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.twist.linear.x = float(input_direction[0])
        msg.twist.linear.y = float(input_direction[1])
        msg.twist.linear.z = float(input_direction[2])
        self.input_pub.publish(msg)

    def _score_summary(self) -> str:
        if self.last_score_msg is None or not self.last_score_msg.scores:
            return "no score message"

        sorted_scores = sorted(
            self.last_score_msg.scores,
            key=lambda item: item.total_score,
            reverse=True,
        )
        top_items = sorted_scores[:3]
        return ", ".join("%s=%.3f" % (item.grasp_id, item.total_score) for item in top_items)

    def run(self) -> int:
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("Validation scenario: %s", rospy.get_param("~scenario", "basic_directional"))
        rospy.loginfo("Publishing candidates on %s", self.candidate_topic)
        rospy.loginfo("Publishing synthetic operator input on %s", self.input_topic)

        start = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < self.startup_delay_sec:
            self._publish_static_state()
            rate.sleep()

        passed = 0
        total = len(self.scenario.tests)

        for test in self.scenario.tests:
            if rospy.is_shutdown():
                return 1

            self.last_selected_grasp_id = None
            self.last_score_msg = None

            rospy.loginfo("Running case '%s' expecting '%s'", test.name, test.expected_grasp_id)
            case_start = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - case_start).to_sec() < test.duration_sec:
                self._publish_static_state()
                self._publish_input(test.input_direction)
                rate.sleep()

            actual = self.last_selected_grasp_id
            success = actual == test.expected_grasp_id
            if success:
                passed += 1
                rospy.loginfo(
                    "PASS  case='%s' selected='%s' scores=[%s]",
                    test.name,
                    actual,
                    self._score_summary(),
                )
            else:
                rospy.logerr(
                    "FAIL  case='%s' expected='%s' actual='%s' scores=[%s]",
                    test.name,
                    test.expected_grasp_id,
                    actual,
                    self._score_summary(),
                )

            gap_start = rospy.Time.now()
            while not rospy.is_shutdown() and (rospy.Time.now() - gap_start).to_sec() < self.case_gap_sec:
                self._publish_static_state()
                self._publish_input((0.0, 0.0, 0.0))
                rate.sleep()

        rospy.loginfo("Validation summary: %d/%d cases passed", passed, total)
        return 0 if passed == total else 1


if __name__ == "__main__":
    validator = SharedAutonomySelectorValidation()
    exit_code = validator.run()
    rospy.signal_shutdown("validation complete")
    raise SystemExit(exit_code)
