#!/usr/bin/env python3
"""Render CASPER-style visual prompts onto recorded observation images.

The logger records world-frame candidate and end-effector positions but not
pixel masks. This renderer creates a lightweight Set-of-Marks approximation by
mapping table XY positions into the saved image, then drawing candidate marks,
the gripper/end-effector position, and trajectory arrows. It writes a new JSONL
that points `image_path` at the annotated image.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np


PALETTE = [
    (0, 255, 255),
    (255, 180, 0),
    (0, 220, 0),
    (255, 80, 80),
    (220, 80, 255),
    (80, 180, 255),
    (180, 255, 80),
    (255, 255, 255),
]


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _xy_from(value):
    if not isinstance(value, dict):
        return None
    if "position" in value and isinstance(value["position"], dict):
        value = value["position"]
    if "ee_position" in value and isinstance(value["ee_position"], dict):
        value = value["ee_position"]
    if "x" in value and "y" in value:
        return np.array([float(value["x"]), float(value["y"])], dtype=np.float64)
    return None


def _uv_from(value):
    if not isinstance(value, dict):
        return None
    center_uv = value.get("center_uv") or value.get("uv")
    if isinstance(center_uv, (list, tuple)) and len(center_uv) >= 2:
        return np.array([float(center_uv[0]), float(center_uv[1])], dtype=np.float64)
    return None


def _collect_xy(row):
    points = []
    for candidate in row.get("candidates") or []:
        xy = _xy_from(candidate.get("position") or candidate)
        if xy is not None:
            points.append(xy)
    for sample in row.get("trajectory_history") or []:
        xy = _xy_from(sample)
        if xy is not None:
            points.append(xy)
    return points


def _make_mapper(row, width, height, margin_px):
    points = _collect_xy(row)
    if not points:
        return None
    stacked = np.vstack(points)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = np.maximum(span * 0.15, np.array([0.04, 0.04], dtype=np.float64))
    lo = lo - pad
    hi = hi + pad
    span = np.maximum(hi - lo, 1e-3)
    drawable_w = max(1.0, float(width - 2 * margin_px))
    drawable_h = max(1.0, float(height - 2 * margin_px))

    def map_xy(xy):
        xy = np.array(xy, dtype=np.float64)
        u = float(margin_px) + ((xy[0] - lo[0]) / span[0]) * drawable_w
        v = float(height - margin_px) - ((xy[1] - lo[1]) / span[1]) * drawable_h
        return int(round(np.clip(u, 0, width - 1))), int(round(np.clip(v, 0, height - 1)))

    return map_xy


def _make_mapper_from_points(points, width, height, margin_px):
    if not points:
        return None
    stacked = np.vstack(points)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = np.maximum(span * 0.18, np.array([0.05, 0.05], dtype=np.float64))
    lo = lo - pad
    hi = hi + pad
    span = np.maximum(hi - lo, 1e-3)
    drawable_w = max(1.0, float(width - 2 * margin_px))
    drawable_h = max(1.0, float(height - 2 * margin_px))

    def map_xy(xy):
        xy = np.array(xy, dtype=np.float64)
        u = float(margin_px) + ((xy[0] - lo[0]) / span[0]) * drawable_w
        v = float(height - margin_px) - ((xy[1] - lo[1]) / span[1]) * drawable_h
        return int(round(np.clip(u, 0, width - 1))), int(round(np.clip(v, 0, height - 1)))

    return map_xy


def _label_for_candidate(candidate):
    candidate_id = str(candidate.get("candidate_id") or candidate.get("label") or "").strip()
    object_name = str(candidate.get("object_name") or "").strip()
    if object_name:
        return "{}:{}".format(candidate_id, object_name)
    return candidate_id


def _short_label_for_candidate(candidate):
    candidate_id = str(candidate.get("candidate_id") or candidate.get("label") or "").strip()
    object_name = str(candidate.get("object_name") or "").strip()
    if object_name:
        return "{} {}".format(candidate_id, object_name.replace("_", " "))
    return candidate_id


def _candidate_display_label(candidate, mode):
    mode = str(mode or "id").strip().lower()
    candidate_id = str(candidate.get("candidate_id") or candidate.get("label") or "").strip()
    object_name = str(candidate.get("object_name") or "").strip()
    if mode in ("none", "off", "false"):
        return ""
    if mode in ("object", "object_name", "name"):
        return object_name.replace("_", " ") if object_name else candidate_id
    if mode in ("full", "id_object", "id_name"):
        return _short_label_for_candidate(candidate)
    return candidate_id


def _draw_label(image, text, xy, color):
    if not str(text or "").strip():
        return
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0 = min(max(0, x + 7), max(0, image.shape[1] - tw - 6))
    y0 = min(max(th + 6, y - 8), max(th + 6, image.shape[0] - 4))
    cv2.rectangle(image, (x0 - 3, y0 - th - 4), (x0 + tw + 3, y0 + baseline + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x0, y0), font, scale, color, thickness, cv2.LINE_AA)


def _candidate_region_radius(candidate, default_px=26):
    rel = candidate.get("relative_to_gripper") or {}
    category = str(candidate.get("category") or "").strip().lower()
    if category == "destination":
        return int(default_px * 1.35)
    try:
        distance = float(rel.get("distance_xy_m"))
        if distance < 0.06:
            return int(default_px * 1.15)
    except Exception:
        pass
    return int(default_px)


def _draw_candidate_region(image, uv, color, radius_px, mode):
    mode = str(mode or "ellipse").strip().lower()
    if mode in ("none", "off", "false"):
        return
    overlay = image.copy()
    if mode in ("disk", "circle"):
        cv2.circle(overlay, uv, int(radius_px), color, -1, cv2.LINE_AA)
    else:
        axes = (int(radius_px * 1.35), int(radius_px * 0.9))
        cv2.ellipse(overlay, uv, axes, 0.0, 0.0, 360.0, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, image, 0.72, 0, dst=image)


def render_semantic_map(row, output_dir, size_px=512, margin_px=48, label_mode="id", region_mode="ellipse"):
    points = _collect_xy(row)
    if not points:
        return None
    size_px = int(max(256, size_px))
    image = np.full((size_px, size_px, 3), 242, dtype=np.uint8)
    mapper = _make_mapper_from_points(points, size_px, size_px, margin_px)
    if mapper is None:
        return None

    cv2.rectangle(image, (margin_px, margin_px), (size_px - margin_px, size_px - margin_px), (185, 185, 185), 1)
    cv2.putText(
        image,
        "top-down semantic map",
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )

    traj_uv = []
    for sample in row.get("trajectory_history") or []:
        xy = _xy_from(sample)
        if xy is not None:
            traj_uv.append(mapper(xy))
    for p0, p1 in zip(traj_uv, traj_uv[1:]):
        if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < 2:
            continue
        cv2.arrowedLine(image, p0, p1, (60, 60, 60), 2, cv2.LINE_AA, tipLength=0.28)
    if traj_uv:
        cv2.circle(image, traj_uv[0], 6, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.circle(image, traj_uv[-1], 13, (0, 0, 220), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            "G",
            (min(size_px - 80, traj_uv[-1][0] + 9), max(42, traj_uv[-1][1] - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 180),
            2,
            cv2.LINE_AA,
        )

    top_goal_label = str(row.get("top_goal_label") or row.get("top_goal") or "").strip()
    selected_candidate_id = str(row.get("selected_candidate_id") or "").strip()
    candidates = list(row.get("candidates") or [])
    candidates.sort(
        key=lambda candidate: int((candidate.get("relative_to_gripper") or {}).get("rank_by_distance", 999999)),
        reverse=True,
    )
    for idx, candidate in enumerate(candidates):
        xy = _xy_from(candidate.get("position") or candidate)
        if xy is None:
            continue
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        color = PALETTE[idx % len(PALETTE)]
        uv = mapper(xy)
        _draw_candidate_region(image, uv, color, _candidate_region_radius(candidate), region_mode)
        radius = 18 if candidate_id and candidate_id in (top_goal_label, selected_candidate_id) else 14
        thickness = 4 if candidate_id and candidate_id in (top_goal_label, selected_candidate_id) else 2
        cv2.circle(image, uv, radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, uv, 4, color, -1, cv2.LINE_AA)
        _draw_label(image, _candidate_display_label(candidate, label_mode), (uv[0] + 12, uv[1]), color)

    line_no = int(row.get("_line_no") or 0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.splitext(os.path.basename(str(row.get("image_path") or "semantic")))[0]
    out_path = os.path.join(output_dir, "{}_line_{:06d}_{}_semantic_map.jpg".format(base, line_no, stamp))
    ok = cv2.imwrite(out_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        return None
    return out_path


def render_row(row, output_dir, margin_px=28, label_mode="none", draw_candidates=False):
    image_path = str(row.get("image_path") or "").strip()
    if not image_path or not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    mapper = _make_mapper(row, width, height, margin_px)
    if mapper is None:
        return None

    overlay = image.copy()
    if draw_candidates:
        for idx, candidate in enumerate(row.get("candidates") or []):
            direct_uv = _uv_from(candidate)
            xy = _xy_from(candidate.get("position") or candidate)
            if direct_uv is None and xy is None:
                continue
            color = PALETTE[idx % len(PALETTE)]
            if direct_uv is not None:
                uv = (
                    int(round(np.clip(direct_uv[0], 0, width - 1))),
                    int(round(np.clip(direct_uv[1], 0, height - 1))),
                )
            else:
                uv = mapper(xy)
            cv2.circle(overlay, uv, 13, color, 2, cv2.LINE_AA)
            cv2.circle(overlay, uv, 3, color, -1, cv2.LINE_AA)
            _draw_label(overlay, _candidate_display_label(candidate, label_mode), (uv[0] + 10, uv[1]), color)

    traj_uv = []
    for sample in row.get("trajectory_history") or []:
        xy = _xy_from(sample)
        if xy is not None:
            traj_uv.append(mapper(xy))
    for p0, p1 in zip(traj_uv, traj_uv[1:]):
        if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < 3:
            continue
        cv2.arrowedLine(overlay, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
    if traj_uv:
        cv2.circle(overlay, traj_uv[0], 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, traj_uv[-1], 11, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            "G",
            (min(width - 100, traj_uv[-1][0] + 8), max(14, traj_uv[-1][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    alpha = 0.82
    image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    line_no = int(row.get("_line_no") or 0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, "{}_line_{:06d}_{}.jpg".format(base, line_no, stamp))
    ok = cv2.imwrite(out_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input casper_observations JSONL.")
    parser.add_argument("--output", required=True, help="Output JSONL with image_path rewritten to annotated images.")
    parser.add_argument(
        "--frame-dir",
        default="",
        help="Directory for annotated frames. Default: logs/casper_visual_prompts next to input log.",
    )
    parser.add_argument("--margin-px", type=int, default=28, help="2D table-map margin inside the image.")
    parser.add_argument(
        "--label-mode",
        choices=("id", "full", "object", "none"),
        default="none",
        help="Text drawn next to candidate marks in the original camera image.",
    )
    parser.add_argument(
        "--wrist-candidate-marks",
        action="store_true",
        help="Draw candidate marks on the original wrist-camera image. Default: only gripper/trajectory cues.",
    )
    parser.add_argument("--semantic-map", action="store_true", help="Also render a lightweight top-down semantic map.")
    parser.add_argument("--semantic-map-size", type=int, default=512, help="Semantic map image size in pixels.")
    parser.add_argument(
        "--semantic-region-mode",
        choices=("ellipse", "disk", "none"),
        default="ellipse",
        help="Pseudo-mask region style for candidates in the top-down semantic map.",
    )
    parser.add_argument(
        "--semantic-label-mode",
        choices=("id", "full", "object", "none"),
        default="id",
        help="Text drawn next to candidate marks in the top-down semantic map.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input))
    if args.frame_dir:
        frame_dir = os.path.abspath(os.path.expanduser(args.frame_dir))
    else:
        frame_dir = os.path.join(os.path.dirname(input_path), "casper_visual_prompts")
    os.makedirs(frame_dir, exist_ok=True)

    out_rows = []
    rendered = 0
    for row in _read_jsonl(input_path):
        next_row = dict(row)
        original_image_path = str(row.get("image_path") or "")
        rendered_path = render_row(
            row,
            frame_dir,
            margin_px=args.margin_px,
            label_mode=args.label_mode,
            draw_candidates=args.wrist_candidate_marks,
        )
        if rendered_path:
            next_row["original_image_path"] = original_image_path
            next_row["image_path"] = rendered_path
            next_row["visual_prompting"] = {
                "type": "casper_lite_overlay",
                "candidate_marks": bool(args.wrist_candidate_marks),
                "gripper_end_marker": True,
                "trajectory_arrows": True,
                "projection": "table_xy_affine_to_image",
                "uses_center_uv_when_available": True,
            }
            rendered += 1
        if args.semantic_map:
            semantic_map_path = render_semantic_map(
                row,
                frame_dir,
                size_px=args.semantic_map_size,
                label_mode=args.semantic_label_mode,
                region_mode=args.semantic_region_mode,
            )
            if semantic_map_path:
                next_row["semantic_map_path"] = semantic_map_path
                next_row.setdefault("visual_prompting", {})
                next_row["visual_prompting"]["semantic_topdown_map"] = True
                next_row["visual_prompting"]["semantic_region_mode"] = args.semantic_region_mode
        out_rows.append(next_row)

    _write_jsonl(os.path.abspath(os.path.expanduser(args.output)), out_rows)
    print(json.dumps({"rows": len(out_rows), "rendered": rendered, "output": args.output, "frame_dir": frame_dir}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
