#!/usr/bin/python3

import argparse
import rospy
import intera_interface
import intera_external_devices
from geometry_msgs.msg import Twist
from relaxed_ik_ros1.msg import EEVelGoals
from sensor_msgs.msg import Joy
from intera_interface import CHECK_VERSION, RobotEnable
import rospkg
import roslib
import sys
import os
roslib.load_manifest('relaxed_ik_ros1')
pkg_path = rospkg.RosPack().get_path('relaxed_ik_ros1')
sys.path.insert(0, os.path.join(pkg_path, 'scripts')) 
from robot import Robot

def apply_deadzone(val, threshold=0.1):
    """Filter joystick noise."""
    return val if abs(val) > threshold else 0.0

def cubic_scale(value, max_val):
    """Cubic scaling for fine control near center."""
    return (value ** 3) * max_val

class JoystickInput:
    def __init__(self, limb):
        # Robot setup
        self.robot = Robot(rospy.get_param('setting_file_path'))
        self.ee_vel_goals_pub = rospy.Publisher('relaxed_ik/ee_vel_goals', EEVelGoals, queue_size=5)

        # Velocity parameters
        self.max_lin_vel = 0.01  # m/s
        self.max_ang_vel = 0.01  # rad/s
        self.alpha = 0.2         # smoothing factor

        # State
        self.linear = [0.0, 0.0, 0.0]
        self.angular = [0.0, 0.0, 0.0]
        self.smoothed_linear = [0.0, 0.0, 0.0]
        self.smoothed_angular = [0.0, 0.0, 0.0]

        self.limb = limb
        self.gripper = None
        try:
            self.gripper = intera_interface.Gripper(limb + '_gripper')
            self.gripper.set_dead_zone(0.001)
        except:
            rospy.loginfo("Could not detect a connected electric gripper.")

        # Subscribers
        rospy.Subscriber("joy", Joy, self.joy_callback)
        rospy.Timer(rospy.Duration(0.033), self.timer_callback)  # ~30 Hz

    def joy_callback(self, joy_msg):
        """Map joystick input to end-effector velocities with LT modifier for yaw."""

        # Check LT pressed
        lt_pressed = joy_msg.axes[2] < 0.0

        # Linear velocities (left stick)
        self.linear[0] = apply_deadzone(joy_msg.axes[1]) * self.max_lin_vel  # forward/back
        self.linear[1] = apply_deadzone(joy_msg.axes[0]) * self.max_lin_vel  # left/right
        self.linear[2] = apply_deadzone(joy_msg.axes[4]) * self.max_lin_vel  # up/down

        # Angular velocities
        if lt_pressed:
            # LT pressed → right stick horizontal controls yaw
            roll_input = 0.0
            pitch_input = 0.0
            yaw_input = apply_deadzone(joy_msg.axes[3])
        else:
            # Normal: right stick controls roll/pitch
            roll_input = apply_deadzone(joy_msg.axes[3])
            pitch_input = apply_deadzone(joy_msg.axes[4])
            yaw_input = 0.0

        # Cubic scaling
        self.angular[0] = cubic_scale(roll_input, self.max_ang_vel)
        self.angular[1] = cubic_scale(pitch_input, self.max_ang_vel)
        self.angular[2] = cubic_scale(yaw_input, self.max_ang_vel)

        # Gripper control
        if joy_msg.buttons[0] and self.gripper:  # close
            self.gripper.close()
        elif joy_msg.buttons[1] and self.gripper:  # open
            self.gripper.open()

        # Adjust linear velocity live
        # if joy_msg.buttons[0]:  # A / Cross → slower
        #     self.max_lin_vel = max(0.001, self.max_lin_vel - 0.005)
        #     rospy.loginfo(f"Decreased max linear vel to {self.max_lin_vel:.3f}")
        # if joy_msg.buttons[1]:  # B / Circle → faster
        #     self.max_lin_vel += 0.005
        #     rospy.loginfo(f"Increased max linear vel to {self.max_lin_vel:.3f}")

    def timer_callback(self, event):
        """Publish smoothed EE velocities."""
        msg = EEVelGoals()

        # Smoothing
        for i in range(3):
            self.smoothed_linear[i] = (1 - self.alpha) * self.smoothed_linear[i] + self.alpha * self.linear[i]
            self.smoothed_angular[i] = (1 - self.alpha) * self.smoothed_angular[i] + self.alpha * self.angular[i]

        # Build message
        for _ in range(self.robot.num_chain):
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.linear.z = self.smoothed_linear
            twist.angular.x, twist.angular.y, twist.angular.z = self.smoothed_angular
            msg.ee_vels.append(twist)
            msg.tolerances.append(Twist())

        self.ee_vel_goals_pub.publish(msg)

def main():
    rospy.init_node("joystick_control_node")

    parser = argparse.ArgumentParser(description="Sawyer Joystick Teleop with LT-based yaw control")
    parser.add_argument('-j', '--joystick', required=True, choices=['xbox', 'logitech', 'ps3'], help='Joystick type')
    parser.add_argument("-l", "--limb", dest="limb", default="right", choices=['right', 'left'], help="Limb to control")
    args = parser.parse_args(rospy.myargv()[1:])

    # Setup joystick
    if args.joystick == 'xbox':
        joystick = intera_external_devices.joystick.XboxController()
    elif args.joystick == 'logitech':
        joystick = intera_external_devices.joystick.LogitechController()
    elif args.joystick == 'ps3':
        joystick = intera_external_devices.joystick.PS3Controller()

    # Enable robot
    rs = RobotEnable(CHECK_VERSION)
    rs.enable()

    # Start teleop
    JoystickInput(args.limb)
    rospy.spin()


if __name__ == '__main__':
    main()
