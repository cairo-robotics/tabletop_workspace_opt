#!/usr/bin/env python3
import rospy
from sawyer import Sawyer, Pose
import numpy as np
from tabletop_workspace_opt.srv import MoveToCartesianPose, MoveToCartesianPoseResponse, OperateGripper, OperateGripperResponse


def move_to_cartesian_pose(position, orientation=None):
    joint_angles = sawyer_robot.ik_service_client(position, orientation)
    
    if joint_angles is None:
        rospy.logerr("Cannot get IK solution")
        return "Cannot get IK solution", False

    return move_to_joint_angles(joint_angles)


def move_to_joint_angles(joint_angles):
    # success = sawyer_robot.move_to_joint_angles(Pose(joint_angles=joint_angles))
    current_joint_angles = sawyer_robot.get_joint_angles()
    print("joint error: %0.5f" % np.linalg.norm(current_joint_angles - np.array(joint_angles)))
    max_attempts = 2
    attempt = 1
    while np.linalg.norm(current_joint_angles - np.array(joint_angles)) > 0.06:
        success = sawyer_robot.move_to_joint_angles(Pose(joint_angles=joint_angles))
        current_joint_angles = sawyer_robot.get_joint_angles()
        print("joint error: %0.5f" % np.linalg.norm(current_joint_angles - np.array(joint_angles)))
        attempt += 1
        if attempt >= max_attempts:
            break
    return "", success


def handle_move_to_cartesian_pose(req):
    position = (req.x, req.y, req.z)
    orientation = (req.a, req.b, req.c, req.d)
    success, output = move_to_cartesian_pose(position, orientation)
    return MoveToCartesianPoseResponse(success, output)


def handle_operate_gripper(req):
    if req.open:
        sawyer_robot.open_gripper()
    else:
        sawyer_robot.close_gripper()
    return OperateGripperResponse("success")
    


def main():
    rospy.Service('move_to_cartesian_pose', MoveToCartesianPose, handle_move_to_cartesian_pose)
    rospy.Service('operate_gripper', OperateGripper, handle_operate_gripper)
    rospy.spin()


if __name__ == '__main__':
    rospy.init_node('move_to_cartesian_pose')
    sawyer_robot = Sawyer()
    main()
