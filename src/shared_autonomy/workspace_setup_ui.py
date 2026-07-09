#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment-side workspace setup UI for AprilTag scanning."""

import math
import os
import tkinter as tk
from tkinter import ttk

import rospy
import yaml
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


CANVAS_BG = "#111827"
GRID_COLOR = "#2B3648"
LIVE_COLOR = "#F59E0B"
RECORDED_COLOR = "#22C55E"
TEXT_COLOR = "#E5E7EB"
SUBTEXT_COLOR = "#94A3B8"


def _quat_to_matrix(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _quat_to_rpy_deg(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


class WorkspaceSetupUI:
    def __init__(self):
        rospy.init_node("workspace_setup_ui")

        self.live_topic = str(
            rospy.get_param("~live_candidates_topic", "/apriltag_candidate_manager/detections")
        ).strip()
        self.recorded_topic = str(
            rospy.get_param("~recorded_candidates_topic", "/apriltag_grasp_registry/detections")
        ).strip()
        self.live_status_topic = str(
            rospy.get_param("~live_status_topic", "/apriltag_candidate_manager/status")
        ).strip()
        self.recorded_status_topic = str(
            rospy.get_param("~recorded_status_topic", "/apriltag_grasp_registry/status")
        ).strip()
        self.object_map_yaml = os.path.expanduser(
            rospy.get_param(
                "~object_map_yaml",
                os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                    "config",
                    "apriltag_object_map.yaml",
                ),
            )
        )

        self.object_map = self._load_object_map()
        self.live_candidates = {}
        self.recorded_candidates = {}
        self.live_status = "waiting_for_live_candidates"
        self.recorded_status = "waiting_for_recorded_candidates"

        rospy.Subscriber(self.live_topic, Detection2DArray, self._live_cb, queue_size=1)
        rospy.Subscriber(self.recorded_topic, Detection2DArray, self._recorded_cb, queue_size=1)
        rospy.Subscriber(self.live_status_topic, String, self._live_status_cb, queue_size=1)
        rospy.Subscriber(self.recorded_status_topic, String, self._recorded_status_cb, queue_size=1)

        self.root = tk.Tk()
        self.root.title("Workspace Setup Monitor")
        self.root.geometry("1540x920")
        self.root.configure(bg="#0F172A")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.summary_var = tk.StringVar(value="Live: 0 | Recorded: 0")
        self.live_status_var = tk.StringVar(value=self.live_status)
        self.recorded_status_var = tk.StringVar(value=self.recorded_status)

        self._build_ui()

    def _load_object_map(self):
        if not os.path.exists(self.object_map_yaml):
            return {}
        with open(self.object_map_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("tag_objects", {}) if isinstance(data, dict) else {}

    def _tag_meta(self, tag_id):
        return self.object_map.get(tag_id, self.object_map.get(str(tag_id), {}))

    def _parse_detections(self, msg):
        parsed = {}
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            tag_id = int(hyp.id)
            pose = hyp.pose.pose
            meta = self._tag_meta(tag_id)
            parsed[tag_id] = {
                "tag_id": tag_id,
                "object_name": str(meta.get("object_name", f"tag_{tag_id}")).strip(),
                "category": str(meta.get("category", "")).strip(),
                "pose": pose,
            }
        return parsed

    def _live_cb(self, msg):
        self.live_candidates = self._parse_detections(msg)

    def _recorded_cb(self, msg):
        self.recorded_candidates = self._parse_detections(msg)

    def _live_status_cb(self, msg):
        self.live_status = str(msg.data).strip()

    def _recorded_status_cb(self, msg):
        self.recorded_status = str(msg.data).strip()

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#0F172A", padx=18, pady=14)
        header.pack(fill=tk.X)

        title = tk.Label(
            header,
            text="Workspace Setup Monitor",
            fg=TEXT_COLOR,
            bg="#0F172A",
            font=("Helvetica", 20, "bold"),
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text="Experiment-side view of live AprilTag candidates and recorded grasp poses.",
            fg=SUBTEXT_COLOR,
            bg="#0F172A",
            font=("Helvetica", 11),
        )
        subtitle.pack(anchor="w", pady=(4, 8))

        summary = tk.Label(
            header,
            textvariable=self.summary_var,
            fg="#BFDBFE",
            bg="#0F172A",
            font=("Helvetica", 13, "bold"),
        )
        summary.pack(anchor="w")

        status_row = tk.Frame(self.root, bg="#0F172A", padx=18, pady=4)
        status_row.pack(fill=tk.X)
        tk.Label(status_row, text="Live:", fg="#FCD34D", bg="#0F172A", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self.live_status_var, fg=TEXT_COLOR, bg="#0F172A", font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(6, 18))
        tk.Label(status_row, text="Recorded:", fg="#86EFAC", bg="#0F172A", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(status_row, textvariable=self.recorded_status_var, fg=TEXT_COLOR, bg="#0F172A", font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(6, 0))

        body = tk.Frame(self.root, bg="#0F172A", padx=18, pady=12)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg="#0F172A")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(body, bg="#0F172A")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        self.xy_canvas = self._make_projection_panel(left, "Top View (X-Y, orientation yaw)")
        self.xz_canvas = self._make_projection_panel(left, "Side View (X-Z)")
        self.yz_canvas = self._make_projection_panel(left, "Front View (Y-Z)")

        legend = tk.Label(
            left,
            text="Orange = live candidate, Green = recorded scan result",
            fg=SUBTEXT_COLOR,
            bg="#0F172A",
            font=("Helvetica", 10),
        )
        legend.pack(anchor="w", pady=(10, 0))

        table_title = tk.Label(
            right,
            text="Recorded Poses",
            fg=TEXT_COLOR,
            bg="#0F172A",
            font=("Helvetica", 16, "bold"),
        )
        table_title.pack(anchor="w", pady=(0, 8))

        columns = ("tag", "object", "x", "y", "z", "roll", "pitch", "yaw")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=28)
        for key, label, width in (
            ("tag", "Tag", 60),
            ("object", "Object", 170),
            ("x", "X", 90),
            ("y", "Y", 90),
            ("z", "Z", 90),
            ("roll", "Roll", 80),
            ("pitch", "Pitch", 80),
            ("yaw", "Yaw", 80),
        ):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=tk.CENTER)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#111827", foreground=TEXT_COLOR, fieldbackground="#111827", rowheight=24)
        style.configure("Treeview.Heading", background="#1F2937", foreground=TEXT_COLOR)

        scrollbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _make_projection_panel(self, parent, title):
        frame = tk.Frame(parent, bg="#0F172A")
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        tk.Label(frame, text=title, fg=TEXT_COLOR, bg="#0F172A", font=("Helvetica", 13, "bold")).pack(anchor="w", pady=(0, 6))
        canvas = tk.Canvas(frame, width=760, height=220, bg=CANVAS_BG, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas

    def _world_bounds(self):
        entries = list(self.live_candidates.values()) + list(self.recorded_candidates.values())
        if not entries:
            return (-0.2, 1.0), (-0.8, 0.8), (-0.1, 0.8)

        xs = [e["pose"].position.x for e in entries]
        ys = [e["pose"].position.y for e in entries]
        zs = [e["pose"].position.z for e in entries]

        def pad(lo, hi, margin):
            if abs(hi - lo) < 1e-3:
                lo -= margin
                hi += margin
            return lo - margin, hi + margin

        return pad(min(xs), max(xs), 0.08), pad(min(ys), max(ys), 0.08), pad(min(zs), max(zs), 0.05)

    def _draw_projection(self, canvas, x_range, y_range, items, axis_a, axis_b, show_heading=False):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 50)
        height = max(canvas.winfo_height(), 50)
        margin = 24

        canvas.create_rectangle(0, 0, width, height, fill=CANVAS_BG, outline="")
        canvas.create_rectangle(margin, margin, width - margin, height - margin, outline=GRID_COLOR, width=1)

        for i in range(1, 4):
            x = margin + i * (width - 2 * margin) / 4.0
            y = margin + i * (height - 2 * margin) / 4.0
            canvas.create_line(x, margin, x, height - margin, fill=GRID_COLOR)
            canvas.create_line(margin, y, width - margin, y, fill=GRID_COLOR)

        xmin, xmax = x_range
        ymin, ymax = y_range
        xspan = max(xmax - xmin, 1e-6)
        yspan = max(ymax - ymin, 1e-6)

        def project(a, b):
            px = margin + (a - xmin) / xspan * (width - 2 * margin)
            py = height - margin - (b - ymin) / yspan * (height - 2 * margin)
            return px, py

        for source_name, values, color in items:
            for entry in values:
                pose = entry["pose"]
                pos = pose.position
                ori = pose.orientation

                coords = {"x": pos.x, "y": pos.y, "z": pos.z}
                px, py = project(coords[axis_a], coords[axis_b])
                radius = 6 if source_name == "recorded" else 4
                canvas.create_oval(px - radius, py - radius, px + radius, py + radius, fill=color, outline="")
                canvas.create_text(px + 10, py - 10, text=str(entry["tag_id"]), fill=TEXT_COLOR, anchor="w", font=("Helvetica", 9, "bold"))

                if show_heading:
                    rot = _quat_to_matrix(ori.x, ori.y, ori.z, ori.w)
                    heading = {"x": rot[0][0], "y": rot[1][0], "z": rot[2][0]}
                    tip_a = coords[axis_a] + 0.06 * heading[axis_a]
                    tip_b = coords[axis_b] + 0.06 * heading[axis_b]
                    tx, ty = project(tip_a, tip_b)
                    canvas.create_line(px, py, tx, ty, fill=color, width=2, arrow=tk.LAST)

        canvas.create_text(margin + 4, height - 8, text=f"{axis_a.upper()} range: [{xmin:.2f}, {xmax:.2f}]", fill=SUBTEXT_COLOR, anchor="sw", font=("Helvetica", 9))
        canvas.create_text(width - margin - 4, height - 8, text=f"{axis_b.upper()} range: [{ymin:.2f}, {ymax:.2f}]", fill=SUBTEXT_COLOR, anchor="se", font=("Helvetica", 9))

    def _refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for tag_id in sorted(self.recorded_candidates.keys()):
            entry = self.recorded_candidates[tag_id]
            pose = entry["pose"]
            roll, pitch, yaw = _quat_to_rpy_deg(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            self.tree.insert(
                "",
                tk.END,
                values=(
                    tag_id,
                    entry["object_name"],
                    f"{pose.position.x:.3f}",
                    f"{pose.position.y:.3f}",
                    f"{pose.position.z:.3f}",
                    f"{roll:.1f}",
                    f"{pitch:.1f}",
                    f"{yaw:.1f}",
                ),
            )

    def _tick(self):
        self.summary_var.set(
            f"Live: {len(self.live_candidates)} | Recorded: {len(self.recorded_candidates)}"
        )
        self.live_status_var.set(self.live_status)
        self.recorded_status_var.set(self.recorded_status)

        x_range, y_range, z_range = self._world_bounds()
        self._draw_projection(
            self.xy_canvas,
            x_range,
            y_range,
            (
                ("live", self.live_candidates.values(), LIVE_COLOR),
                ("recorded", self.recorded_candidates.values(), RECORDED_COLOR),
            ),
            "x",
            "y",
            show_heading=True,
        )
        self._draw_projection(
            self.xz_canvas,
            x_range,
            z_range,
            (
                ("live", self.live_candidates.values(), LIVE_COLOR),
                ("recorded", self.recorded_candidates.values(), RECORDED_COLOR),
            ),
            "x",
            "z",
        )
        self._draw_projection(
            self.yz_canvas,
            y_range,
            z_range,
            (
                ("live", self.live_candidates.values(), LIVE_COLOR),
                ("recorded", self.recorded_candidates.values(), RECORDED_COLOR),
            ),
            "y",
            "z",
        )
        self._refresh_table()

    def _on_close(self):
        rospy.signal_shutdown("workspace setup UI closed")
        self.root.destroy()

    def run(self):
        rate_hz = float(rospy.get_param("~ui_rate_hz", 5.0))
        rate = rospy.Rate(rate_hz)
        while not rospy.is_shutdown():
            self._tick()
            self.root.update_idletasks()
            self.root.update()
            rate.sleep()


def main():
    WorkspaceSetupUI().run()


if __name__ == "__main__":
    main()
