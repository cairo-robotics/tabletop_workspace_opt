#!/home/heyang/catkin_ws/.venv/bin/python3
"""Keyboard Teleoperation Node

Maps keyboard axes/buttons to end-effector Cartesian velocity commands and
gripper open/close actions. Publishes smoothed velocity goals for RelaxedIK.
"""

import readchar
import rospy
import rospkg
from geometry_msgs.msg import Twist
from relaxed_ik_ros1.msg import EEVelGoals
import transformations as T
from pynput import keyboard
from intera_interface import CHECK_VERSION, RobotEnable, Gripper
import argparse
import sys
import os
import roslib
roslib.load_manifest('relaxed_ik_ros1')
pkg_path = rospkg.RosPack().get_path('relaxed_ik_ros1')
sys.path.insert(0, os.path.join(pkg_path, 'scripts'))
from robot import Robot
from std_msgs.msg import Bool

def apply_deadzone(val, threshold=0.1):
    """Filter joystick noise."""
    return val if abs(val) > threshold else 0.0

def cubic_scale(value, max_val):
    """Cubic scaling for fine control near center."""
    return (value ** 3) * max_val

class KeyboardInput:
    def __init__(self, limb):
        # Robot setup
        self.robot = Robot(rospy.get_param('setting_file_path'))
        self.ee_vel_goals_pub = rospy.Publisher('relaxed_ik/ee_vel_goals', EEVelGoals, queue_size=1)

        self.pos_stride = 0.01
        self.rot_stride = 0.01
        
        # State
        self.linear = [0.0, 0.0, 0.0]
        self.angular = [0.0, 0.0, 0.0]
        self.smoothed_linear = [0.0, 0.0, 0.0]
        self.smoothed_angular = [0.0, 0.0, 0.0]
        self.alpha = 0.2 # smoothing factor

        keyboard_listener = keyboard.Listener(
            on_press = self.on_press,
            on_release = self.on_release)
        
        self.limb = limb
        self.gripper = None
        try:
            self.gripper = Gripper(limb + '_gripper')
            self.gripper.set_dead_zone(0.001)
        except:
            rospy.loginfo("Could not detect a connected electric gripper.")

        keyboard_listener.start()
        self.mujoco_gripper_pub = rospy.Publisher("/mujoco_sim/gripper_open", Bool, queue_size=1)
        self.print_usage()
        rospy.Timer(rospy.Duration(0.033), self.timer_callback)  # ~30 Hz

    def print_usage(self):
        msg = """
    Keyboard Teleop Controls:
    -------------------------
    Linear Motion:
        w/s : +/- X
        a/d : +/- Y
        r/f : +/- Z
    
    Angular Motion:
        1/2 : +/- Roll
        3/4 : +/- Pitch
        5/6 : +/- Yaw

    Gripper:
        o : Open
        p : Close

    Stride:
        . : Increase stride
        , : Decrease stride

    c : Quit
    """
        print(msg)

    def on_press(self, key):
        ch = getattr(key, 'char', None)
        if ch is None:
            return

        self.linear = [0,0,0]
        self.angular = [0,0,0]

        if ch == 'w':
            self.linear[0] += self.pos_stride
        elif ch == 's':
            self.linear[0] -= self.pos_stride
        elif ch == 'a':
            self.linear[1] += self.pos_stride
        elif ch == 'd':
            self.linear[1] -= self.pos_stride
        elif ch == 'r':
            self.linear[2] += self.pos_stride
        elif ch == 'f':
            self.linear[2] -= self.pos_stride
        elif ch == '1':
            self.angular[0] += self.rot_stride
        elif ch == '2':
            self.angular[0] -= self.rot_stride
        elif ch == '3':
            self.angular[1] += self.rot_stride
        elif ch == '4':
            self.angular[1] -= self.rot_stride
        elif ch == '5':
            self.angular[2] += self.rot_stride
        elif ch == '6':
            self.angular[2] -= self.rot_stride
        elif ch == 'c':
            rospy.signal_shutdown("quit")
        elif ch == '.':
            self.pos_stride += 0.01
            print(f"Increased position stride to {self.pos_stride}")
        elif ch == ',':
            self.pos_stride = max(0.0, self.pos_stride - 0.01)  # 防止负数
            print(f"Decreased position stride to {self.pos_stride}")
        elif ch == 'o':
            if self.gripper:
                self.gripper.open()
            else:
                self.mujoco_gripper_pub.publish(Bool(True))
        elif ch == 'p':
            if self.gripper:
                self.gripper.close()
            else:
                self.mujoco_gripper_pub.publish(Bool(False))

    def on_release(self, key):
        ch = getattr(key, 'char', None)
        if ch is None:
            return
        self.linear = [0,0,0]
        self.angular = [0,0,0]

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
        
        
if __name__ == '__main__':
    rospy.init_node('keyboard_input')
    
    parser = argparse.ArgumentParser(description="Sawyer Keyboard Teleop")
    parser.add_argument("-l", "--limb", dest="limb", default="right", choices=['right', 'left'], help="Limb to control")
    args = parser.parse_args(rospy.myargv()[1:])

    # Start teleop
    keyboard = KeyboardInput(args.limb)
    rospy.spin()
