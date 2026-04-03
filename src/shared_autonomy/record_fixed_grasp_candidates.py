#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record fixed grasp candidates from the current end-effector pose.

Typical workflow:
- move the robot with teleop to a desired pre-grasp pose
- press `p` and enter the grasp id
- move to the actual grasp pose
- press `g` and enter the same grasp id

The script stores poses in a YAML file so they can be reused later by the
shared-autonomy grasp chooser.
"""

import os
import select
import sys
import termios
import time
import tty

import rospy
import yaml
from intera_core_msgs.msg import EndpointState


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pose_to_dict(pose_msg):
    return {
        "position": [
            float(pose_msg.position.x),
            float(pose_msg.position.y),
            float(pose_msg.position.z),
        ],
        "orientation": [
            float(pose_msg.orientation.x),
            float(pose_msg.orientation.y),
            float(pose_msg.orientation.z),
            float(pose_msg.orientation.w),
        ],
    }


class _TerminalInput:
    def __init__(self):
        self.enabled = False
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        else:
            rospy.logwarn(
                "[record_grasps] stdin is not a TTY; interactive prompts may not work as expected."
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disable_raw()

    def disable_raw(self):
        if self.enabled and self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            self.enabled = False

    def enable_raw(self):
        if self.fd is not None and self.old_settings is not None and not self.enabled:
            tty.setcbreak(self.fd)
            self.enabled = True

    def read_key(self, timeout=0.05):
        if self.fd is None:
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except Exception:
            return None
        if not ready:
            return None
        try:
            if self.enabled:
                return sys.stdin.read(1).strip().lower() or None
            line = sys.stdin.readline()
            return line[:1].strip().lower() if line else None
        except Exception:
            return None

    def prompt(self, text):
        self.disable_raw()
        try:
            return input(text).strip()
        finally:
            if self.fd is not None:
                self.enable_raw()


class FixedGraspRecorder:
    def __init__(self):
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_out = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")

        self.output_path = os.path.expanduser(rospy.get_param("~output_path", default_out))
        self.endpoint_topic = rospy.get_param(
            "~end_effector_topic", "/robot/limb/right/endpoint_state"
        )
        self.frame_id = rospy.get_param("~frame_id", "base")
        self.require_confirmation = bool(rospy.get_param("~require_confirmation", True))

        ensure_dir(os.path.dirname(self.output_path) or ".")

        self.latest_pose = None
        self.latest_stamp = rospy.Time(0)
        self.grasps = self._load_yaml()

        rospy.Subscriber(self.endpoint_topic, EndpointState, self._endpoint_cb, queue_size=5)

        rospy.loginfo(
            "[record_grasps] ready. endpoint_topic=%s output_path=%s loaded_candidates=%d",
            self.endpoint_topic,
            self.output_path,
            len(self.grasps),
        )

    def _endpoint_cb(self, msg):
        self.latest_pose = msg.pose
        self.latest_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    def _load_yaml(self):
        if not os.path.exists(self.output_path):
            return []
        try:
            with open(self.output_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception as exc:
            rospy.logwarn("[record_grasps] failed to load %s: %s", self.output_path, str(exc))
            return []

        grasps = data.get("grasps", [])
        if not isinstance(grasps, list):
            rospy.logwarn("[record_grasps] invalid YAML format in %s; starting empty.", self.output_path)
            return []
        return grasps

    def _save_yaml(self):
        payload = {
            "frame_id": self.frame_id,
            "grasps": self.grasps,
        }
        with open(self.output_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

    def _find_grasp(self, grasp_id):
        for grasp in self.grasps:
            if str(grasp.get("grasp_id", "")).strip() == grasp_id:
                return grasp
        return None

    def _ensure_latest_pose(self):
        if self.latest_pose is None:
            raise RuntimeError("No end-effector pose received yet.")
        return pose_to_dict(self.latest_pose)

    def _upsert_pose(self, grasp_id, stage_name, pose_dict, object_name=""):
        grasp = self._find_grasp(grasp_id)
        created = False
        if grasp is None:
            grasp = {
                "grasp_id": grasp_id,
                "object_name": object_name or grasp_id,
                "frame_id": self.frame_id,
            }
            self.grasps.append(grasp)
            created = True

        if object_name:
            grasp["object_name"] = object_name

        grasp["frame_id"] = self.frame_id
        grasp[stage_name] = pose_dict
        self._save_yaml()
        return created

    def _print_pose(self, pose_dict):
        pos = pose_dict["position"]
        ori = pose_dict["orientation"]
        print(
            "  position    : [{:.4f}, {:.4f}, {:.4f}]".format(pos[0], pos[1], pos[2])
        )
        print(
            "  orientation : [{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                ori[0], ori[1], ori[2], ori[3]
            )
        )

    def save_stage_interactive(self, terminal, stage_name):
        pose_dict = self._ensure_latest_pose()
        print("")
        print(f"Current pose for `{stage_name}`:")
        self._print_pose(pose_dict)

        grasp_id = terminal.prompt("grasp_id: ").strip()
        if not grasp_id:
            print("Cancelled: grasp_id is required.")
            return

        existing = self._find_grasp(grasp_id)
        default_object_name = existing.get("object_name", grasp_id) if existing else grasp_id
        object_name = terminal.prompt(f"object_name [{default_object_name}]: ").strip()
        object_name = object_name or default_object_name

        if self.require_confirmation:
            confirm = terminal.prompt(
                f"Save `{stage_name}` for `{grasp_id}` to {self.output_path}? [y/N]: "
            ).strip().lower()
            if confirm not in ("y", "yes"):
                print("Cancelled.")
                return

        created = self._upsert_pose(grasp_id, stage_name, pose_dict, object_name=object_name)
        action = "Created" if created else "Updated"
        print(f"{action} `{grasp_id}` with `{stage_name}`.")

    def delete_interactive(self, terminal):
        grasp_id = terminal.prompt("Delete grasp_id: ").strip()
        if not grasp_id:
            print("Cancelled.")
            return

        before = len(self.grasps)
        self.grasps = [grasp for grasp in self.grasps if grasp.get("grasp_id") != grasp_id]
        if len(self.grasps) == before:
            print(f"No candidate named `{grasp_id}`.")
            return
        self._save_yaml()
        print(f"Deleted `{grasp_id}`.")

    def print_candidates(self):
        print("")
        print(f"Saved candidates: {len(self.grasps)}")
        for grasp in self.grasps:
            stages = []
            if "pregrasp_pose" in grasp:
                stages.append("pregrasp")
            if "grasp_pose" in grasp:
                stages.append("grasp")
            stage_text = ", ".join(stages) if stages else "none"
            print(
                "- {} | object={} | stages={}".format(
                    grasp.get("grasp_id", "<unnamed>"),
                    grasp.get("object_name", ""),
                    stage_text,
                )
            )
        print("")

    @staticmethod
    def print_help():
        print(
            "\nFixed Grasp Recorder Controls\n"
            "  p : save current EE pose as pregrasp_pose\n"
            "  g : save current EE pose as grasp_pose\n"
            "  l : list saved candidates\n"
            "  d : delete one candidate\n"
            "  h : print this help\n"
            "  q : quit\n"
        )

    def run(self):
        self.print_help()
        print(f"Listening on {self.endpoint_topic}")
        print(f"Writing grasp candidates to {self.output_path}")

        with _TerminalInput() as terminal:
            while not rospy.is_shutdown():
                key = terminal.read_key(timeout=0.05)
                if key is None:
                    time.sleep(0.05)
                    continue

                if key == "p":
                    try:
                        self.save_stage_interactive(terminal, "pregrasp_pose")
                    except Exception as exc:
                        rospy.logwarn("[record_grasps] failed to save pregrasp pose: %s", str(exc))
                elif key == "g":
                    try:
                        self.save_stage_interactive(terminal, "grasp_pose")
                    except Exception as exc:
                        rospy.logwarn("[record_grasps] failed to save grasp pose: %s", str(exc))
                elif key == "l":
                    self.print_candidates()
                elif key == "d":
                    self.delete_interactive(terminal)
                elif key == "h":
                    self.print_help()
                elif key == "q":
                    rospy.loginfo("[record_grasps] quitting.")
                    return


def main():
    node = FixedGraspRecorder()
    node.run()


if __name__ == "__main__":
    rospy.init_node("record_fixed_grasp_candidates")
    main()
