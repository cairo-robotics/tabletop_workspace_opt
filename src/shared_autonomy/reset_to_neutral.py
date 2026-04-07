#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Move Sawyer back to the built-in neutral pose."""

import argparse

import rospy
from intera_interface import CHECK_VERSION, RobotEnable, Limb


def main():
    rospy.init_node("reset_to_neutral")

    parser = argparse.ArgumentParser(description="Move Sawyer to the built-in neutral pose")
    parser.add_argument("-l", "--limb", default="right", choices=["right", "left"])
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=0.2)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.loginfo(
        "[reset_to_neutral] enabling robot and moving limb=%s to neutral (timeout=%.1f, speed=%.2f)",
        args.limb,
        args.timeout,
        args.speed,
    )

    rs = RobotEnable(CHECK_VERSION)
    rs.enable()

    limb = Limb(args.limb)
    ok = limb.move_to_neutral(timeout=args.timeout, speed=args.speed)

    if ok:
        rospy.loginfo("[reset_to_neutral] limb=%s reached neutral.", args.limb)
    else:
        rospy.logwarn("[reset_to_neutral] limb=%s failed to reach neutral.", args.limb)


if __name__ == "__main__":
    main()
