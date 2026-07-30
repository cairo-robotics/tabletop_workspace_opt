#!/usr/bin/env python3
"""CASPER shared-autonomy coordinator.

Owns the CASPER interaction state machine:

    TELEOP --(vote >= eta)--> OFFER --y--> AUTONOMOUS --done--> TELEOP
                                 \--n / timeout / keep-driving--> TELEOP

Responsibilities: the task state machine (candidates = ValidGoals, identical
to the Bayesian baselines), the teleop gate, offer UX (terminal banner +
topics, y/n keys), sim gripper keys (o/p -> /operate_gripper), skill
execution via the existing TaskExecutor, RelaxedIK re-sync before returning
control, generation-based staleness, and session logging.
"""
import json
import os
import sys
import threading
import time

import rospy
import yaml
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String
from vision_msgs.msg import Detection2DArray

from tabletop_workspace_opt.msg import ValidGoal, ValidGoals
from tabletop_workspace_opt.srv import OperateGripper

_PKG_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")
if _PKG_SRC not in sys.path:
    sys.path.insert(0, _PKG_SRC)

from shared_autonomy import shared_autonomy_runner as sar
from shared_autonomy.vlm_intent.prompt_builder import (
    build_candidates,
    goal_description,
)
from shared_autonomy.vlm_intent.session_logger import SessionLogger
from shared_autonomy.vlm_intent.skill_executor import SkillExecutor

RELAXED_IK_JOINTS = ["right_j0", "right_j1", "right_j2", "right_j3",
                     "right_j4", "right_j5", "right_j6"]


class CasperCoordinator:
    def __init__(self):
        cfg = rospy.get_param("~casper", rospy.get_param("casper", {}))
        self.offer_cfg = cfg.get("offer", {})
        self.accept_key = self.offer_cfg.get("accept_key", "y")
        self.reject_key = self.offer_cfg.get("reject_key", "n")
        self.offer_timeout_s = float(self.offer_cfg.get("timeout_s", 8.0))
        self.reject_suppress_s = float(
            self.offer_cfg.get("reject_suppress_s", 15.0))
        gating = cfg.get("gating", {})
        self.override_vel_thresh = float(
            gating.get("override_vel_thresh", 0.02))
        self.override_hold_s = float(gating.get("override_hold_s", 1.5))

        # ---- task + scene -------------------------------------------------
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        task_config = rospy.get_param("~task_config",
                                      "config/tasks/full_breakfast_sa.yaml")
        if not os.path.isabs(task_config):
            task_config = os.path.join(pkg_root, task_config)
        with open(task_config) as handle:
            self.task = yaml.safe_load(handle)["task"]
        self.scene = rospy.get_param("~scene", "") or \
            self.task.get("scene", "scene_breakfast")

        rospy.loginfo("CASPER: waiting for skill services (MoveIt, gripper)…")
        self.skills = SkillExecutor(
            self.scene,
            cfg.get("execution", {}).get("grasp_library",
                                         "config/grasp_poses.yaml"))
        det_ids = self.skills._run_task.OBJECT_DET_IDS
        self.state_machine = sar.TaskStateMachine(self.task, det_ids)

        # ---- session logging ----------------------------------------------
        log_cfg = cfg.get("logging", {})
        out_dir = log_cfg.get("out_dir", "results/casper_runs")
        if not os.path.isabs(os.path.expanduser(out_dir)):
            out_dir = os.path.join(pkg_root, out_dir)
        self.logger = SessionLogger(
            out_dir=out_dir,
            scene=self.scene,
            task_name=self.task.get("name", "task"),
            meta={"config": cfg, "task_config": task_config},
            save_images=bool(log_cfg.get("save_annotated_images", True)),
        )

        # ---- runtime state -------------------------------------------------
        self.state = "TELEOP"
        self.generation = 0
        self.det_msg = None
        self.joint_state = None
        self.gripper_state = "open"
        self.pending_offer = None          # (letter, goal_spec)
        self.offer_deadline = 0.0
        self.offer_started = 0.0
        self.override_since = None
        self.suppressed = {}               # goal_id -> expiry time
        self.candidates = []               # [(letter, goal_spec, pos)]
        self._lock = threading.Lock()

        # ---- ROS I/O --------------------------------------------------------
        self.pub_enable = rospy.Publisher("/casper/teleop_enabled", Bool,
                                          queue_size=1, latch=True)
        self.pub_ctl = rospy.Publisher("/casper/inference_ctl", String,
                                       queue_size=1, latch=True)
        self.pub_offer = rospy.Publisher("/casper/offer", String,
                                         queue_size=2)
        self.pub_status = rospy.Publisher("/casper/status", String,
                                          queue_size=2, latch=True)
        self.pub_goals = rospy.Publisher("/shared_autonomy/valid_goals",
                                         ValidGoals, queue_size=1)
        self.pub_gripper = rospy.Publisher("/casper/gripper_state", String,
                                           queue_size=1, latch=True)
        self.pub_ik_reset = rospy.Publisher("/relaxed_ik/reset", JointState,
                                            queue_size=1, latch=True)

        self.gripper_srv = rospy.ServiceProxy("/operate_gripper",
                                              OperateGripper)
        rospy.Subscriber("/mujoco_sim/detections", Detection2DArray,
                         self._det_cb, queue_size=1)
        rospy.Subscriber("/joint_states", JointState, self._joints_cb,
                         queue_size=1)
        rospy.Subscriber("/casper/keys", String, self._keys_cb, queue_size=8)
        rospy.Subscriber("/casper/intent_vote", String, self._vote_cb,
                         queue_size=4)
        rospy.Subscriber("/casper/teleop_activity", Float32,
                         self._activity_cb, queue_size=1)

        self.pub_enable.publish(Bool(data=True))
        self.pub_gripper.publish(String(data=self.gripper_state))
        rospy.Timer(rospy.Duration(1.0), self._tick)
        self._set_status("TELEOP: drive with w/a/s/d/r/f; o/p = gripper. "
                         "The VLM will offer help when your intent is clear.")
        rospy.loginfo("CASPER coordinator ready (task=%s, scene=%s)",
                      self.task.get("name"), self.scene)

    # ---- callbacks -----------------------------------------------------------

    def _det_cb(self, msg):
        self.det_msg = msg

    def _joints_cb(self, msg):
        self.joint_state = msg

    def _activity_cb(self, msg):
        if self.state != "OFFER":
            self.override_since = None
            return
        if msg.data > self.override_vel_thresh:
            now = time.time()
            if self.override_since is None:
                self.override_since = now
            elif now - self.override_since >= self.override_hold_s:
                self._reject("kept_driving")
        else:
            self.override_since = None

    def _keys_cb(self, msg):
        key = (msg.data or "").lower()
        if key in ("o", "p") and self.state != "AUTONOMOUS":
            self._operate_gripper(open_=(key == "o"))
        elif self.state == "OFFER":
            if key == self.accept_key:
                self._accept()
            elif key == self.reject_key:
                self._reject("key")

    def _vote_cb(self, msg):
        try:
            vote = json.loads(msg.data)
        except ValueError:
            return
        stale = vote.get("generation") != self.generation
        self.logger.log("inference", state=self.state, stale=stale, **vote)
        if stale or self.state != "TELEOP" or not vote.get("winner"):
            return
        with self._lock:
            spec = dict((letter, s) for letter, s, _ in self.candidates).get(
                vote["winner"])
        if spec is None:
            return
        self._start_offer(vote["winner"], spec, vote)

    # ---- periodic tick ---------------------------------------------------------

    def _tick(self, _evt):
        now = time.time()
        self.suppressed = {gid: exp for gid, exp in self.suppressed.items()
                           if exp > now}
        if self.state == "OFFER" and now > self.offer_deadline:
            self._reject("timeout")
        if self.state == "TELEOP":
            self._refresh_candidates()
        self._publish_ctl()

    def _refresh_candidates(self):
        if self.det_msg is None:
            return
        if self.state_machine.is_done():
            if self.candidates:
                self.candidates = []
                self._set_status("TASK COMPLETE — all subtasks done.")
                self.logger.log("state_change", state="DONE")
            return
        goals = self.state_machine.get_valid_goals(self.det_msg)
        specs = [spec for spec, _pos in goals]
        letter_specs = build_candidates(specs)
        positions = {id(spec): pos for spec, pos in goals}
        with self._lock:
            self.candidates = [
                (letter, spec, list(map(float, positions[id(spec)])))
                for (letter, spec) in letter_specs]
        goals_msg = ValidGoals()
        goals_msg.header.stamp = rospy.Time.now()
        goals_msg.current_state = self.state_machine.current_state
        for letter, spec, pos in self.candidates:
            goal = ValidGoal()
            goal.goal_id = spec.get("id", letter)
            goal.action_type = spec.get("action", "")
            goal.object_name = spec.get("object", "") or ""
            goal.target_position.x, goal.target_position.y, \
                goal.target_position.z = pos
            goals_msg.goals.append(goal)
        self.pub_goals.publish(goals_msg)

    def _publish_ctl(self):
        now = time.time()
        with self._lock:
            # Intent inference is only for 'pick': the VLM chooses which
            # object the user is reaching for. place/pour are teleoperated
            # by the human (they need destination poses, not object choice),
            # so they are never marked or offered. Letters stay consistent
            # with self.candidates for the vote->spec lookup in _vote_cb.
            candidates = [
                {"letter": letter, "goal_spec": spec, "position_world": pos}
                for letter, spec, pos in self.candidates
                if spec.get("action") == "pick"]
        rejections = []
        for letter, spec, _pos in self.candidates:
            if spec.get("id") in self.suppressed:
                rejections.append(goal_description(spec))
        self.pub_ctl.publish(String(data=json.dumps({
            "state": self.state,
            "generation": self.generation,
            "current_state": self.state_machine.current_state,
            "holding": self.state_machine.holding,
            "session_dir": self.logger.directory,
            "rejections": rejections,
            "candidates": candidates,
        })))

    # ---- offer lifecycle ----------------------------------------------------------

    def _start_offer(self, letter, spec, vote):
        description = goal_description(spec)
        self.state = "OFFER"
        self.pending_offer = (letter, spec)
        self.offer_started = time.time()
        self.offer_deadline = self.offer_started + self.offer_timeout_s
        self.override_since = None
        self.logger.log("offer", candidate=letter, goal_id=spec.get("id"),
                        description=description,
                        confidence=vote.get("confidence"))
        self.pub_offer.publish(String(data=json.dumps(
            {"candidate": letter, "goal_id": spec.get("id"),
             "description": description,
             "confidence": vote.get("confidence")})))
        banner = ("\n" + "=" * 62 +
                  "\n>>> OFFER: %s?  [%s]=accept  [%s]=reject "
                  "(or keep driving)\n" % (description, self.accept_key,
                                           self.reject_key) +
                  "=" * 62)
        print(banner)
        self._set_status("OFFER: %s? press %s/%s"
                         % (description, self.accept_key, self.reject_key))
        self._publish_ctl()

    def _accept(self):
        letter, spec = self.pending_offer
        self.logger.log("decision", decision="accept", goal_id=spec.get("id"),
                        latency_s=round(time.time() - self.offer_started, 2))
        self.state = "AUTONOMOUS"
        self.pending_offer = None
        self.pub_enable.publish(Bool(data=False))
        self._set_status("AUTONOMOUS: executing '%s' — teleop disabled."
                         % goal_description(spec))
        print(">>> Taking over: %s" % goal_description(spec))
        threading.Thread(target=self._execute, args=(spec,),
                         daemon=True).start()

    def _reject(self, reason):
        if self.pending_offer is None:
            return
        letter, spec = self.pending_offer
        self.pending_offer = None
        self.suppressed[spec.get("id")] = time.time() + self.reject_suppress_s
        self.logger.log("decision", decision="reject", reason=reason,
                        goal_id=spec.get("id"),
                        latency_s=round(time.time() - self.offer_started, 2))
        self.generation += 1
        self.state = "TELEOP"
        print(">>> Offer %s (%s). You have control." %
              ("rejected" if reason == "key" else "dismissed", reason))
        self._set_status("TELEOP (offer %s)" % reason)
        self._publish_ctl()

    # ---- execution ---------------------------------------------------------------

    def _execute(self, spec):
        started = time.time()
        success = self.skills.execute(spec)
        self.logger.log("skill_outcome", goal_id=spec.get("id"),
                        action=spec.get("action"), success=success,
                        duration_s=round(time.time() - started, 1))
        if success:
            self.state_machine.transition(spec)
            self.logger.log("state_change",
                            state=self.state_machine.current_state,
                            holding=self.state_machine.holding)
        self.gripper_state = "closed" if self.state_machine.holding else "open"
        self.pub_gripper.publish(String(data=self.gripper_state))

        # MoveIt drove the arm without RelaxedIK seeing it; re-sync before
        # giving velocity control back (otherwise the next teleop command
        # snaps the arm toward RelaxedIK's stale internal goal).
        self._reset_relaxed_ik()

        self.generation += 1
        self.state = "TELEOP"
        self.pub_enable.publish(Bool(data=True))
        done = self.state_machine.is_done()
        outcome = "done" if success else "FAILED — try teleoperating"
        print(">>> Skill %s. You have control.%s"
              % (outcome, "  TASK COMPLETE!" if done else ""))
        self._set_status("TELEOP (skill %s)%s"
                         % (outcome, " — TASK COMPLETE" if done else ""))
        self._refresh_candidates()
        self._publish_ctl()

    def _reset_relaxed_ik(self):
        if self.joint_state is None:
            rospy.logwarn("no joint states; skipping RelaxedIK re-sync")
            return
        by_name = dict(zip(self.joint_state.name, self.joint_state.position))
        if any(name not in by_name for name in RELAXED_IK_JOINTS):
            rospy.logwarn("joint states incomplete; skipping RelaxedIK re-sync")
            return
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(RELAXED_IK_JOINTS)
        msg.position = [by_name[name] for name in RELAXED_IK_JOINTS]
        self.pub_ik_reset.publish(msg)
        rospy.sleep(0.3)

    # ---- misc -----------------------------------------------------------------------

    def _operate_gripper(self, open_):
        try:
            self.gripper_srv(open_)
            self.gripper_state = "open" if open_ else "closed"
            self.pub_gripper.publish(String(data=self.gripper_state))
            self.logger.log("gripper", state=self.gripper_state, source="user")
        except rospy.ServiceException as exc:
            rospy.logwarn("gripper service failed: %s", exc)

    def _set_status(self, text):
        self.pub_status.publish(String(data=text))
        rospy.loginfo("[CASPER] %s", text)


if __name__ == "__main__":
    rospy.init_node("casper_coordinator")
    CasperCoordinator()
    rospy.spin()
