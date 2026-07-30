#!/usr/bin/env python3
"""Teleop command gate for CASPER shared autonomy.

Relays /teleop/ee_vel_goals -> relaxed_ik/ee_vel_goals only while teleop is
enabled (/casper/teleop_enabled, latched Bool). Also publishes the current
commanded speed on /casper/teleop_activity so the coordinator can detect
manual override during a pending offer.

Teleop nodes are launch-remapped to publish on /teleop/ee_vel_goals; their
control logic is untouched.
"""
import rospy
from relaxed_ik_ros1.msg import EEVelGoals
from std_msgs.msg import Bool, Float32


class TeleopGate:
    def __init__(self):
        self.enabled = bool(rospy.get_param("~start_enabled", True))
        self.pub_out = rospy.Publisher("relaxed_ik/ee_vel_goals",
                                       EEVelGoals, queue_size=1)
        self.pub_activity = rospy.Publisher("/casper/teleop_activity",
                                            Float32, queue_size=1)
        rospy.Subscriber("/casper/teleop_enabled", Bool,
                         self._enable_cb, queue_size=1)
        rospy.Subscriber("/teleop/ee_vel_goals", EEVelGoals,
                         self._teleop_cb, queue_size=1)
        rospy.loginfo("teleop gate ready (enabled=%s)", self.enabled)

    def _enable_cb(self, msg):
        if msg.data != self.enabled:
            rospy.loginfo("teleop %s", "ENABLED" if msg.data else "DISABLED")
        self.enabled = msg.data

    def _teleop_cb(self, msg):
        speed = 0.0
        if msg.ee_vels:
            v = msg.ee_vels[0].linear
            speed = (v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5
        self.pub_activity.publish(Float32(data=speed))
        if self.enabled:
            self.pub_out.publish(msg)


if __name__ == "__main__":
    rospy.init_node("teleop_gate")
    TeleopGate()
    rospy.spin()
