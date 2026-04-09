#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small GUI for choosing the active side-grasp target at runtime."""

import os
import tkinter as tk

import rospy
import yaml
from std_msgs.msg import String


class GraspTargetSelectorGUI:
    def __init__(self):
        rospy.init_node("grasp_target_selector_gui")

        self.fixed_grasp_yaml = self._resolve_yaml_path()
        self.selected_grasp_label_topic = str(
            rospy.get_param("~selected_grasp_label_topic", "/shared_autonomy/selected_grasp_label")
        ).strip()
        self.default_label = str(rospy.get_param("~default_grasp_label", "side_grasp_milk")).strip()
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 5.0))

        self.publisher = rospy.Publisher(self.selected_grasp_label_topic, String, queue_size=1, latch=True)
        self.options = self._load_side_grasp_options()
        self.selected_label = self._choose_initial_label()

        self.root = tk.Tk()
        self.root.title("Pour Target Selector")
        self.root.geometry("360x240")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar()
        self._build_ui()
        self._publish_selection()

        rospy.loginfo(
            "[grasp_target_selector_gui] ready. topic=%s default=%s options=%s",
            self.selected_grasp_label_topic,
            self.selected_label,
            ",".join(label for label, _ in self.options),
        )

    def _resolve_yaml_path(self):
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_yaml = os.path.join(package_root, "config", "fixed_grasp_candidates.yaml")
        return os.path.expanduser(rospy.get_param("~fixed_grasp_yaml", default_yaml))

    def _load_side_grasp_options(self):
        with open(self.fixed_grasp_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        options = []
        for grasp in data.get("grasps", []):
            grasp_id = str(grasp.get("grasp_id", "")).strip()
            if not grasp_id.startswith("side_grasp_"):
                continue
            object_name = str(grasp.get("object_name", "")).strip()
            object_suffix = grasp_id.replace("side_grasp_", "", 1).replace("_", " ").title()
            button_text = object_name.replace("_", " ").title() if object_name else object_suffix
            options.append((grasp_id, button_text))

        if not options:
            raise RuntimeError(f"No side_grasp_* candidates found in {self.fixed_grasp_yaml}")
        return options

    def _choose_initial_label(self):
        valid_labels = {label for label, _ in self.options}
        if self.default_label in valid_labels:
            return self.default_label
        return self.options[0][0]

    def _build_ui(self):
        container = tk.Frame(self.root, padx=18, pady=18)
        container.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(container, text="Select Pour Target", font=("Helvetica", 16, "bold"))
        title.pack(anchor="w")

        subtitle = tk.Label(
            container,
            text="Choosing a target updates the active grasp-complete label at runtime.",
            justify=tk.LEFT,
            wraplength=320,
        )
        subtitle.pack(anchor="w", pady=(6, 12))

        for grasp_id, button_text in self.options:
            button = tk.Button(
                container,
                text=button_text,
                width=24,
                command=lambda label=grasp_id: self._set_selection(label),
            )
            button.pack(anchor="w", pady=3)

        status_label = tk.Label(container, textvariable=self.status_var, fg="#0B5")
        status_label.pack(anchor="w", pady=(14, 0))
        self._refresh_status_text()

    def _refresh_status_text(self):
        self.status_var.set(f"Current target: {self.selected_label}")

    def _set_selection(self, grasp_label):
        self.selected_label = grasp_label
        self._refresh_status_text()
        self._publish_selection()

    def _publish_selection(self):
        self.publisher.publish(String(data=self.selected_label))
        rospy.loginfo("[grasp_target_selector_gui] selected target -> %s", self.selected_label)

    def _on_close(self):
        rospy.signal_shutdown("selector window closed")
        self.root.destroy()

    def run(self):
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            self.root.update_idletasks()
            self.root.update()
            rate.sleep()


def main():
    GraspTargetSelectorGUI().run()


if __name__ == "__main__":
    main()
