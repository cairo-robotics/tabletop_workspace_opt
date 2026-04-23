#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive recorder for fixed-scene candidate-classification datasets.

Two recording modes are supported:
1. `image_labels`: save the latest image plus instruction-conditioned labels
2. `bag_labels`: assume a rosbag is running, and only save labels plus the
   latest image timestamp so frames can be extracted later

Typical rosbag workflow:
1. Start `rosbag record` for camera + robot state topics.
2. Arrange the milk cartons in a scene.
3. Press `c` to mark the current scene/view.
4. Enter slot assignments and one or more labeled instructions.
5. Repeat for the next scene permutation or view.

The goal is to create data for:
    image + instruction -> correct predefined candidate
"""

import json
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
            rospy.logwarn("[record_candidate_dataset] stdin is not a TTY.")
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


class CandidateDatasetRecorder:
    DEFAULT_CANDIDATES = [
        "whole_top",
        "whole_side",
        "oat_top",
        "oat_side",
        "soy_top",
        "soy_side",
    ]
    VALID_OBJECTS = ["whole_milk", "oat_milk", "soy_milk"]

    def __init__(self):
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_root = os.path.join(package_root, "data", "milk_candidate_cls")

        self.dataset_root = os.path.expanduser(rospy.get_param("~dataset_root", default_root))
        self.images_dir = os.path.join(self.dataset_root, "images")
        self.episodes_path = os.path.join(self.dataset_root, "episodes.jsonl")
        self.record_mode = str(rospy.get_param("~record_mode", "image_labels")).strip().lower()
        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw")).strip()
        self.image_encoding = str(rospy.get_param("~image_encoding", "bgr8")).strip()
        self.scene_prefix = str(rospy.get_param("~scene_prefix", "scene")).strip()
        self.episode_prefix = str(rospy.get_param("~episode_prefix", "ep")).strip()
        self.allowed_candidates = list(rospy.get_param("~allowed_candidates", self.DEFAULT_CANDIDATES))
        self.default_task_type = str(rospy.get_param("~default_task_type", "")).strip()
        self.user_name = str(rospy.get_param("~user_name", "")).strip()
        self.default_bag_name = str(rospy.get_param("~bag_name", "")).strip()
        self.default_view_id = str(rospy.get_param("~default_view_id", "")).strip()

        if self.record_mode == "image_labels":
            ensure_dir(self.images_dir)
        ensure_dir(self.dataset_root)

        self.bridge = CvBridge()
        self.latest_cv_image = None
        self.latest_stamp = None
        self.last_scene_id = None
        self.last_image_relpath = None
        self.last_bag_name = self.default_bag_name
        self.last_view_id = self.default_view_id
        self.last_slot_assignment = None
        self.last_scene_notes = ""

        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2 ** 24)

        rospy.loginfo(
            "[record_candidate_dataset] ready. mode=%s image_topic=%s dataset_root=%s",
            self.record_mode,
            self.image_topic,
            self.dataset_root,
        )

    def _image_cb(self, msg):
        try:
            self.latest_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
            self.latest_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[record_candidate_dataset] cv_bridge failed: %s", str(exc))

    def _next_scene_id(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "{}_{}".format(self.scene_prefix, stamp)

    def _make_episode_id(self, scene_id, idx):
        return "{}_{}_{}".format(self.episode_prefix, scene_id, idx)

    def _save_current_image(self, scene_id):
        if self.latest_cv_image is None:
            raise RuntimeError("No image received yet.")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV (cv2) is required for image_labels mode.") from exc
        filename = "{}.png".format(scene_id)
        abs_path = os.path.join(self.images_dir, filename)
        if not cv2.imwrite(abs_path, self.latest_cv_image):
            raise RuntimeError("Failed to write image to {}".format(abs_path))
        rel_path = os.path.join("images", filename)
        self.last_scene_id = scene_id
        self.last_image_relpath = rel_path
        return rel_path

    def _latest_stamp_string(self):
        if self.latest_stamp is None:
            raise RuntimeError("No image timestamp received yet.")
        return "{:.9f}".format(self.latest_stamp.to_sec())

    def _normalize_slot_value(self, value):
        txt = str(value).strip().lower().replace(" ", "_")
        aliases = {
            "whole": "whole_milk",
            "wholemilk": "whole_milk",
            "whole_milk": "whole_milk",
            "oat": "oat_milk",
            "oatmilk": "oat_milk",
            "oat_milk": "oat_milk",
            "soy": "soy_milk",
            "soymilk": "soy_milk",
            "soy_milk": "soy_milk",
        }
        return aliases.get(txt, txt)

    def _prompt_slot_value(self, terminal, slot_name):
        while not rospy.is_shutdown():
            raw = terminal.prompt(
                "  {} slot object [whole_milk/oat_milk/soy_milk]: ".format(slot_name)
            ).strip()
            normalized = self._normalize_slot_value(raw)
            if normalized in self.VALID_OBJECTS:
                return normalized
            print("Invalid slot value. Use one of: {}".format(", ".join(self.VALID_OBJECTS)))

    def _prompt_slot_assignment(self, terminal):
        print("")
        print("Slot assignment for current scene.")
        left = self._prompt_slot_value(terminal, "left")
        center = self._prompt_slot_value(terminal, "center")
        right = self._prompt_slot_value(terminal, "right")
        return {"left": left, "center": center, "right": right}

    def _prompt_notes(self, terminal):
        return terminal.prompt("Scene notes [optional]: ").strip()

    def _prompt_view_id(self, terminal):
        default = self.last_view_id or self.default_view_id or "top/side/lean"
        value = terminal.prompt("view_id [{}]: ".format(default)).strip()
        self.last_view_id = value or self.last_view_id or self.default_view_id
        return self.last_view_id

    def _prompt_bag_name(self, terminal):
        default = self.last_bag_name or self.default_bag_name or "milk_day1.bag"
        value = terminal.prompt("bag_name [{}]: ".format(default)).strip()
        self.last_bag_name = value or self.last_bag_name or self.default_bag_name
        return self.last_bag_name

    def _prompt_task_type(self, terminal):
        default = self.default_task_type or "pickup/pour/etc"
        value = terminal.prompt("task_type [{}]: ".format(default)).strip()
        return value or self.default_task_type

    def _collect_instruction_entries(self, terminal):
        print("")
        print("Paste one line per label using:")
        print("  instruction | correct_candidate_id | task_type | episode_notes")
        print("Example:")
        print("  Pick up the whole milk. | whole_top | pickup |")
        print("Type END on its own line to finish this scene.")

        entries = []
        while not rospy.is_shutdown():
            line = terminal.prompt("label line: ").strip()
            if not line:
                print("Empty line ignored. Use END to finish.")
                continue
            if line.upper() == "END":
                break
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 3:
                print("Invalid line. Need at least 3 fields separated by '|'.")
                continue
            instruction = parts[0]
            correct_candidate_id = parts[1]
            task_type = parts[2]
            episode_notes = parts[3] if len(parts) > 3 else ""
            if correct_candidate_id not in self.allowed_candidates:
                print("Invalid candidate. Expected one of: {}".format(", ".join(self.allowed_candidates)))
                continue
            entries.append(
                {
                    "instruction": instruction,
                    "correct_candidate_id": correct_candidate_id,
                    "task_type": task_type,
                    "episode_notes": episode_notes,
                }
            )
        return entries

    def _prompt_instruction_block(
        self,
        terminal,
        scene_id,
        image_relpath,
        image_stamp,
        slot_assignment,
        scene_notes,
        bag_name,
        view_id,
    ):
        print("")
        print("Enter instruction labels for {}.".format(scene_id))
        print("Allowed candidates: {}".format(", ".join(self.allowed_candidates)))
        entries = self._collect_instruction_entries(terminal)

        count = 0
        for entry in entries:
            count += 1

            payload = {
                "episode_id": self._make_episode_id(scene_id, count),
                "scene_id": scene_id,
                "instruction": entry["instruction"],
                "correct_candidate_id": entry["correct_candidate_id"],
                "task_type": entry["task_type"],
                "slot_assignment": slot_assignment,
                "view_id": view_id,
                "scene_notes": scene_notes,
                "episode_notes": entry["episode_notes"],
                "allowed_candidates": self.allowed_candidates,
                "image_topic": self.image_topic,
                "image_stamp": image_stamp,
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
            }
            if image_relpath:
                payload["image_path"] = image_relpath
            if bag_name:
                payload["bag_name"] = bag_name
            if self.user_name:
                payload["recorded_by"] = self.user_name

            append_jsonl(self.episodes_path, payload)
            print("Saved {} -> {}".format(payload["episode_id"], entry["correct_candidate_id"]))

        if count == 0:
            print("No instruction labels saved for this scene.")

    @staticmethod
    def print_help():
        print(
            "\nCandidate Dataset Recorder Controls\n"
            "  c : capture/mark a new scene and label one or more instructions\n"
            "  a : add more instruction labels for the last captured scene\n"
            "  l : print last captured scene info\n"
            "  h : print this help\n"
            "  q : quit\n"
        )

    def run(self):
        self.print_help()
        print("Listening on {}".format(self.image_topic))
        print("Writing dataset to {}".format(self.dataset_root))
        print("Episodes file: {}".format(self.episodes_path))
        print("Record mode: {}".format(self.record_mode))

        with _TerminalInput() as terminal:
            while not rospy.is_shutdown():
                key = terminal.read_key(timeout=0.05)
                if key is None:
                    time.sleep(0.05)
                    continue

                if key == "c":
                    try:
                        scene_id = terminal.prompt("scene_id [auto]: ").strip() or self._next_scene_id()
                        slot_assignment = self._prompt_slot_assignment(terminal)
                        scene_notes = self._prompt_notes(terminal)
                        view_id = self._prompt_view_id(terminal)
                        bag_name = self._prompt_bag_name(terminal) if self.record_mode == "bag_labels" else ""
                        image_stamp = self._latest_stamp_string()
                        image_relpath = ""
                        if self.record_mode == "image_labels":
                            image_relpath = self._save_current_image(scene_id)
                        self.last_scene_id = scene_id
                        self.last_slot_assignment = slot_assignment
                        self.last_scene_notes = scene_notes
                        self._prompt_instruction_block(
                            terminal=terminal,
                            scene_id=scene_id,
                            image_relpath=image_relpath,
                            image_stamp=image_stamp,
                            slot_assignment=slot_assignment,
                            scene_notes=scene_notes,
                            bag_name=bag_name,
                            view_id=view_id,
                        )
                    except Exception as exc:
                        rospy.logwarn("[record_candidate_dataset] capture failed: %s", str(exc))
                elif key == "a":
                    if not self.last_scene_id:
                        print("No previously captured scene. Use `c` first.")
                        continue
                    try:
                        print("")
                        print("Appending labels to {}".format(self.last_scene_id))
                        image_stamp = self._latest_stamp_string()
                        view_id = self._prompt_view_id(terminal)
                        bag_name = self._prompt_bag_name(terminal) if self.record_mode == "bag_labels" else ""
                        self._prompt_instruction_block(
                            terminal=terminal,
                            scene_id=self.last_scene_id,
                            image_relpath=self.last_image_relpath or "",
                            image_stamp=image_stamp,
                            slot_assignment=self.last_slot_assignment or {},
                            scene_notes=self.last_scene_notes,
                            bag_name=bag_name,
                            view_id=view_id,
                        )
                    except Exception as exc:
                        rospy.logwarn("[record_candidate_dataset] append failed: %s", str(exc))
                elif key == "l":
                    print("")
                    print("Last scene: {}".format(self.last_scene_id or "<none>"))
                    print("Last image: {}".format(self.last_image_relpath or "<none>"))
                    print("Last bag: {}".format(self.last_bag_name or "<none>"))
                    print("Last view: {}".format(self.last_view_id or "<none>"))
                    print("Last stamp: {}".format(self._latest_stamp_string() if self.latest_stamp else "<none>"))
                    print("Allowed candidates: {}".format(", ".join(self.allowed_candidates)))
                    print("")
                elif key == "h":
                    self.print_help()
                elif key == "q":
                    rospy.loginfo("[record_candidate_dataset] quitting.")
                    return


def main():
    rospy.init_node("record_candidate_dataset")
    CandidateDatasetRecorder().run()


if __name__ == "__main__":
    main()
