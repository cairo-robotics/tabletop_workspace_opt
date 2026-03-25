#!/usr/bin/env python3
"""
publish_tf_from_jointstate_fk.py  (enhanced "JointState Hub")

Input:
  - /relaxed_ik/joint_angle_solutions  (sensor_msgs/JointState)

Outputs:
  - /joint_states                      (sensor_msgs/JointState)  [relay for MoveIt/robot_state_publisher]
  - /tf                                (dynamic TF: base_link -> tip_link)  [FK computed from URDF+KDL]
  - /tf_static                         (static TF: world -> base_link, tip_link -> cam_frame)
  - /ee_pose                           (geometry_msgs/PoseStamped) [optional debug]

Notes:
  - Uses URDF from parameter server (robot_description).
  - KDL FK is computed for chain base_link -> tip_link.
  - Joint order is derived from the KDL chain joint sequence.
"""

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
from sensor_msgs.msg import JointState

# urdf + kdl
from urdf_parser_py.urdf import URDF
import PyKDL
from kdl_parser_py.urdf import treeFromUrdfModel


def kdl_frame_to_trans_quat(F: PyKDL.Frame):
    p = F.p
    M = F.M
    qx, qy, qz, qw = M.GetQuaternion()  # x y z w
    return (p[0], p[1], p[2], qx, qy, qz, qw)


class FKTFPublisher:
    def __init__(self):
        # ---------------- Parameters ----------------
        self.joint_topic = rospy.get_param("~joint_topic", "/relaxed_ik/joint_angle_solutions")

        self.world_frame = rospy.get_param("~world_frame", "world")
        self.base_link = rospy.get_param("~base_link", "base_link")    # URDF root or your base
        self.tip_link = rospy.get_param("~tip_link", "right_hand")     # end effector link

        # camera extrinsic wrt tip_link (static)
        self.cam_frame = rospy.get_param("~cam_frame", "realsense_color_optical_frame")
        self.tip_to_cam_xyz = rospy.get_param("~tip_to_cam_xyz", [0.0, 0.0, 0.0])
        self.tip_to_cam_quat = rospy.get_param("~tip_to_cam_quat", [0.0, 0.0, 0.0, 1.0])  # x y z w

        # fixed world->base (static)
        self.world_to_base_xyz = rospy.get_param("~world_to_base_xyz", [0.0, 0.0, 0.0])
        self.world_to_base_quat = rospy.get_param("~world_to_base_quat", [0.0, 0.0, 0.0, 1.0])

        # publish toggles
        self.publish_joint_states = rospy.get_param("~publish_joint_states", True)
        self.publish_ee_pose = rospy.get_param("~publish_ee_pose", True)

        # ---------------- Load URDF / KDL ----------------
        robot = URDF.from_parameter_server()
        ok, tree = treeFromUrdfModel(robot)
        if not ok:
            raise RuntimeError("Failed to parse URDF into KDL tree")
        self.tree = tree

        self.chain = self.tree.getChain(self.base_link, self.tip_link)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)

        # Build joint name list in EXACT KDL joint order
        none_joint_type = getattr(PyKDL.Joint, "None", None)  # some bindings expose Joint.None
        self.kdl_joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            j = self.chain.getSegment(i).getJoint()

            # If binding provides Joint.None, filter by joint type; otherwise fallback to name check.
            if none_joint_type is not None:
                if j.getType() == none_joint_type:
                    continue
            else:
                # Fallback: "None" joints usually have empty name in KDL
                if j.getName() == "":
                    continue

            self.kdl_joint_names.append(j.getName())
            
        self.num_joints = self.chain.getNrOfJoints()
        if len(self.kdl_joint_names) != self.num_joints:
            rospy.logwarn(
                "[fk_tf] KDL joint name count (%d) != chain.getNrOfJoints() (%d). "
                "Proceeding, but verify URDF/joints.",
                len(self.kdl_joint_names), self.num_joints
            )

        rospy.loginfo("[fk_tf] chain joints (%d): %s", len(self.kdl_joint_names), self.kdl_joint_names)

        # index lookup for incoming joint states
        self.kdl_joint_set = set(self.kdl_joint_names)

        # ---------------- Publishers/Subscribers ----------------
        self.br = tf2_ros.TransformBroadcaster()
        self.static_br = tf2_ros.StaticTransformBroadcaster()

        if self.publish_joint_states:
            self.pub_joint_states = rospy.Publisher("/joint_states", JointState, queue_size=10)
        else:
            self.pub_joint_states = None

        if self.publish_ee_pose:
            self.pub_ee_pose = rospy.Publisher("/ee_pose", PoseStamped, queue_size=10)
        else:
            self.pub_ee_pose = None

        self._publish_static_transforms()

        self.sub = rospy.Subscriber(self.joint_topic, JointState, self.cb, queue_size=1)

        rospy.loginfo("[fk_tf] listening: %s", self.joint_topic)
        rospy.loginfo("[fk_tf] outputs: /tf (dynamic %s->%s), /tf_static (world/base + tip/cam), /joint_states relay=%s, /ee_pose=%s",
                      self.base_link, self.tip_link, str(self.publish_joint_states), str(self.publish_ee_pose))

    def _publish_static_transforms(self):
        stamp = rospy.Time.now()

        # world -> base_link
        t_wb = TransformStamped()
        t_wb.header.stamp = stamp
        t_wb.header.frame_id = self.world_frame
        t_wb.child_frame_id = self.base_link
        t_wb.transform.translation.x = float(self.world_to_base_xyz[0])
        t_wb.transform.translation.y = float(self.world_to_base_xyz[1])
        t_wb.transform.translation.z = float(self.world_to_base_xyz[2])
        t_wb.transform.rotation.x = float(self.world_to_base_quat[0])
        t_wb.transform.rotation.y = float(self.world_to_base_quat[1])
        t_wb.transform.rotation.z = float(self.world_to_base_quat[2])
        t_wb.transform.rotation.w = float(self.world_to_base_quat[3])

        # tip_link -> camera
        t_tc = TransformStamped()
        t_tc.header.stamp = stamp
        t_tc.header.frame_id = self.tip_link
        t_tc.child_frame_id = self.cam_frame
        t_tc.transform.translation.x = float(self.tip_to_cam_xyz[0])
        t_tc.transform.translation.y = float(self.tip_to_cam_xyz[1])
        t_tc.transform.translation.z = float(self.tip_to_cam_xyz[2])
        t_tc.transform.rotation.x = float(self.tip_to_cam_quat[0])
        t_tc.transform.rotation.y = float(self.tip_to_cam_quat[1])
        t_tc.transform.rotation.z = float(self.tip_to_cam_quat[2])
        t_tc.transform.rotation.w = float(self.tip_to_cam_quat[3])

        self.static_br.sendTransform([t_wb, t_tc])

    def cb(self, msg: JointState):
        # Choose a sensible timestamp
        stamp = msg.header.stamp if msg.header.stamp and msg.header.stamp.to_sec() > 0 else rospy.Time.now()

        # 0) relay /joint_states for MoveIt / robot_state_publisher
        if self.pub_joint_states is not None:
            out = JointState()
            out.header.stamp = stamp
            out.name = list(msg.name)
            out.position = list(msg.position)
            out.velocity = list(msg.velocity) if msg.velocity else []
            out.effort = list(msg.effort) if msg.effort else []
            self.pub_joint_states.publish(out)

        # 1) Build KDL JntArray in KDL chain joint order
        name_to_pos = dict(zip(msg.name, msg.position))
        missing = [jn for jn in self.kdl_joint_names if jn not in name_to_pos]
        if missing:
            rospy.logwarn_throttle(2.0, "[fk_tf] missing joints %s (have %d joints in msg), skip FK",
                                   missing, len(msg.name))
            return

        q = PyKDL.JntArray(self.num_joints)
        for i, jn in enumerate(self.kdl_joint_names[:self.num_joints]):
            q[i] = float(name_to_pos[jn])

        out_frame = PyKDL.Frame()
        ret = self.fk_solver.JntToCart(q, out_frame)
        if ret < 0:
            rospy.logwarn_throttle(2.0, "[fk_tf] FK solver failed: %d", ret)
            return

        # 2) base_link -> tip_link (dynamic from FK)
        x, y, z, qx, qy, qz, qw = kdl_frame_to_trans_quat(out_frame)

        t_bt = TransformStamped()
        t_bt.header.stamp = stamp
        t_bt.header.frame_id = self.base_link
        t_bt.child_frame_id = self.tip_link
        t_bt.transform.translation.x = float(x)
        t_bt.transform.translation.y = float(y)
        t_bt.transform.translation.z = float(z)
        t_bt.transform.rotation.x = float(qx)
        t_bt.transform.rotation.y = float(qy)
        t_bt.transform.rotation.z = float(qz)
        t_bt.transform.rotation.w = float(qw)

        self.br.sendTransform(t_bt)

        # 3) Optional EE pose debug topic
        if self.pub_ee_pose is not None:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.base_link
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            self.pub_ee_pose.publish(ps)


def main():
    rospy.init_node("publish_tf_from_jointstate_fk")
    FKTFPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()