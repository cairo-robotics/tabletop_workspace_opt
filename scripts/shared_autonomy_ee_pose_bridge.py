#!/usr/bin/env python3
"""Bridge MuJoCo endpoint state to PoseStamped for shared autonomy nodes."""

import rospy
from geometry_msgs.msg import PoseStamped
from intera_core_msgs.msg import EndpointState


class EndpointPoseBridge:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/mujoco_sim/endpoint_state")
        self.output_topic = rospy.get_param("~output_topic", "/shared_autonomy/ee_pose")
        self.default_frame = rospy.get_param("~default_frame", "world")
        self.output_frame = rospy.get_param("~output_frame", "")

        self.pub = rospy.Publisher(self.output_topic, PoseStamped, queue_size=1)
        rospy.Subscriber(self.input_topic, EndpointState, self.callback, queue_size=1)

        rospy.loginfo(
            "Endpoint pose bridge ready. input=%s output=%s frame=%s",
            self.input_topic,
            self.output_topic,
            self.output_frame or self.default_frame,
        )

    def callback(self, msg: EndpointState):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        pose_msg.header.frame_id = self.output_frame or msg.header.frame_id or self.default_frame
        pose_msg.pose = msg.pose
        self.pub.publish(pose_msg)


def main():
    rospy.init_node("shared_autonomy_ee_pose_bridge")
    EndpointPoseBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
