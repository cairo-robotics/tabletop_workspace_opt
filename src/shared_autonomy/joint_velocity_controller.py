#!/usr/bin/env python3

import time
import threading
import rospy
from sensor_msgs.msg import JointState
import intera_interface
from intera_interface import CHECK_VERSION

def _name_map(msg):
    if msg.name and len(msg.name) == len(msg.position):
        return dict(zip(msg.name, msg.position))
    return None

class JointVelController(object):
    def __init__(self):
        # -------- Params --------
        self.arm         = rospy.get_param("~arm", "right")
        self.ik_topic    = rospy.get_param("~ik_topic", "/relaxed_ik/joint_angle_solutions")
        self.rate_hz     = float(rospy.get_param("~rate_hz", 200.0))
        self.kp          = float(rospy.get_param("~kp", 2.0))
        self.kd          = float(rospy.get_param("~kd", 0.1))
        self.max_abs_vel = float(rospy.get_param("~max_abs_vel", 0.5))      # rad/s
        self.pos_tol     = float(rospy.get_param("~pos_tol", 0.01))         # rad per joint
        self.settle_time = float(rospy.get_param("~settle_time", 0.05))     # sec within tol to declare done
        self.enable_on_start = bool(rospy.get_param("~enable_on_start", True))
        self.dry_run     = bool(rospy.get_param("~dry_run", False))

        # -------- Enable & interfaces --------
        rospy.loginfo("Initializing Sawyer…")
        self.enabler = intera_interface.RobotEnable(CHECK_VERSION)
        if self.enable_on_start and not self.dry_run:
            self.enabler.enable()

        self.limb = intera_interface.Limb(self.arm)
        self.joint_names = self.limb.joint_names()
        rospy.loginfo("Controlling joints: %s", self.joint_names)

        # -------- Controller state --------
        self.lock = threading.Lock()
        self.active = False                 # true while pursuing current target
        self.target = {j: 0.0 for j in self.joint_names}
        self.prev_err = {j: 0.0 for j in self.joint_names}
        self.prev_t = None
        self.settle_start = None            # time when all joints first within tol

        # -------- Subscriber & loop --------
        self.sub = rospy.Subscriber(self.ik_topic, JointState, self._target_cb, queue_size=1)
        self.dt_nom = 1.0 / self.rate_hz
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        rospy.loginfo("Single-target PD velocity controller running at %.1f Hz", self.rate_hz)

    def _target_cb(self, msg: JointState):
        """New target: reset PD memory so derivative starts at 0 (no kick)."""
        name_map = _name_map(msg)
        with self.lock:
            if name_map:
                # Align by name; ignore any unknown joints
                missing = [j for j in self.joint_names if j not in name_map]
                if missing:
                    rospy.logwarn_throttle(2.0, "Incoming target missing joints: %s", missing)
                for j in self.joint_names:
                    if j in name_map:
                        self.target[j] = float(name_map[j])
            else:
                if not msg.position or len(msg.position) < len(self.joint_names):
                    rospy.logwarn("Target without names and insufficient positions; ignoring.")
                    return
                for j, qstar in zip(self.joint_names, msg.position[:len(self.joint_names)]):
                    self.target[j] = float(qstar)

            # Reset PD state w.r.t current q so initial derivative = 0
            q_now = self.limb.joint_angles()
            for j in self.joint_names:
                self.prev_err[j] = self.target[j] - q_now[j]
            self.prev_t = time.time()
            self.settle_start = None
            self.active = True

    def _loop(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if not self.active:
                # Keep streaming zeros to be safe
                self._send_zero_vel()
                rate.sleep()
                continue

            now = time.time()
            q = self.limb.joint_angles()  # dict {joint: pos}

            # Compute errors
            err = {}
            max_abs_err = 0.0
            for j in self.joint_names:
                e = self.target[j] - q[j]
                err[j] = e
                ae = abs(e)
                if ae > max_abs_err:
                    max_abs_err = ae

            # Check convergence: within pos_tol on all joints for settle_time
            if max_abs_err <= self.pos_tol:
                if self.settle_start is None:
                    self.settle_start = now
                if (now - self.settle_start) >= self.settle_time:
                    # Reached target: stop and go inactive
                    self._send_zero_vel()
                    self.active = False
                    rate.sleep()
                    continue
            else:
                self.settle_start = None

            # PD step
            if self.prev_t is None:
                dt = self.dt_nom
            else:
                dt = max(1e-3, now - self.prev_t)

            v_cmd = {}
            for j in self.joint_names:
                de = (err[j] - self.prev_err[j]) / dt
                v = self.kp * err[j] + self.kd * de
                if v > self.max_abs_vel:
                    v = self.max_abs_vel
                elif v < -self.max_abs_vel:
                    v = -self.max_abs_vel
                v_cmd[j] = v
                self.prev_err[j] = err[j]

            self.prev_t = now

            if not self.dry_run:
                self.limb.set_joint_velocities(v_cmd)
            else:
                rospy.loginfo_throttle(1.0, "Dry-run velocities: %s", v_cmd)

            rate.sleep()

        # Shutdown: ensure stop
        self._send_zero_vel()

    def _send_zero_vel(self):
        zeros = {j: 0.0 for j in self.joint_names}
        if not self.dry_run:
            self.limb.set_joint_velocities(zeros)

def main():
    rospy.init_node("sawyer_pd_vel_single_target")
    JointVelController()
    rospy.spin()

if __name__ == "__main__":
    main()
