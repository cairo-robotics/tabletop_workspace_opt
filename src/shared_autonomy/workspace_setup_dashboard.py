#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight web dashboard for experiment-side workspace setup."""

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
import yaml
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection2DArray


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Workspace Setup Dashboard</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #172033;
      --line: #243047;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --live: #f59e0b;
      --recorded: #22c55e;
      --accent: #60a5fa;
      --danger: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at top, #172554 0%, var(--bg) 42%);
      color: var(--text);
    }
    .page {
      max-width: 1520px;
      margin: 0 auto;
      padding: 22px;
    }
    .title {
      font-size: 30px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .subtitle {
      color: var(--muted);
      margin-bottom: 18px;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }
    .card {
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 35px rgba(0,0,0,0.18);
    }
    .metric-label {
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .metric-value {
      font-size: 34px;
      font-weight: 700;
      margin-top: 6px;
    }
    .status-line {
      margin-top: 8px;
      color: var(--text);
      font-size: 14px;
      line-height: 1.4;
      min-height: 40px;
    }
    .layout {
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 18px;
    }
    .panel-title {
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .panel-subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 12px;
    }
    .canvas-wrap {
      background: linear-gradient(180deg, #0b1220 0%, #101a2f 100%);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      margin-bottom: 10px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 560px;
      cursor: grab;
    }
    canvas.dragging { cursor: grabbing; }
    .legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }
    .legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }
    .dot {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }
    .sidebar {
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 18px;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }
    .detail-grid strong {
      display: block;
      color: var(--text);
      font-size: 16px;
      margin-top: 4px;
    }
    .list-toolbar {
      display: flex;
      gap: 10px;
      margin-bottom: 12px;
    }
    .list-toolbar input {
      width: 100%;
      background: var(--panel-2);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      outline: none;
    }
    .rows {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 560px;
      overflow: auto;
      padding-right: 2px;
    }
    .row {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: rgba(23, 32, 51, 0.72);
      cursor: pointer;
    }
    .row.active {
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.45);
    }
    .row-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }
    .row-title {
      font-size: 15px;
      font-weight: 700;
    }
    .row-meta {
      color: var(--muted);
      font-size: 12px;
    }
    .badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .badge {
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .badge.live { background: rgba(245, 158, 11, 0.18); color: #fcd34d; }
    .badge.recorded { background: rgba(34, 197, 94, 0.16); color: #86efac; }
    .coords {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .coords span {
      display: block;
      color: var(--text);
      font-size: 14px;
      margin-top: 2px;
    }
    .empty {
      color: var(--muted);
      padding: 20px 0;
    }
    @media (max-width: 1200px) {
      .layout { grid-template-columns: 1fr; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      .summary { grid-template-columns: 1fr; }
      .detail-grid { grid-template-columns: 1fr; }
      canvas { height: 420px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="title">Workspace Setup Dashboard</div>
    <div class="subtitle">Experiment-side scan monitor for live AprilTag candidates and recorded grasp poses.</div>

    <div class="summary">
      <div class="card">
        <div class="metric-label">Live Candidates</div>
        <div class="metric-value" id="live-count">0</div>
      </div>
      <div class="card">
        <div class="metric-label">Recorded Poses</div>
        <div class="metric-value" id="recorded-count">0</div>
      </div>
      <div class="card">
        <div class="metric-label">Live Status</div>
        <div class="status-line" id="live-status">waiting_for_live_candidates</div>
      </div>
      <div class="card">
        <div class="metric-label">Recorded Status</div>
        <div class="status-line" id="recorded-status">waiting_for_recorded_candidates</div>
      </div>
    </div>

    <div class="layout">
      <div class="card">
        <div class="panel-title">3D Grasp Scene</div>
        <div class="panel-subtitle">Drag to orbit. Mouse wheel zooms. MarkerArray geometry is rendered directly from the AprilTag grasp topics.</div>
        <div class="canvas-wrap">
          <canvas id="scene" width="920" height="560"></canvas>
        </div>
        <div class="legend">
          <div class="legend-chip"><span class="dot" style="background: var(--live)"></span> Live candidate</div>
          <div class="legend-chip"><span class="dot" style="background: var(--recorded)"></span> Recorded scan result</div>
          <div class="legend-chip">ROS markers: spheres, cubes, arrows, labels</div>
        </div>
      </div>

      <div class="sidebar">
        <div class="card">
          <div class="panel-title">Selected Object</div>
          <div class="panel-subtitle">Pose details for the highlighted candidate.</div>
          <div class="detail-grid" id="detail-grid"></div>
        </div>

        <div class="card">
          <div class="panel-title">Scanned Objects</div>
          <div class="panel-subtitle">Browse by tag, object name, or category.</div>
          <div class="list-toolbar">
            <input id="filter" placeholder="Filter objects...">
          </div>
          <div class="rows" id="rows"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const state = {
      data: null,
      selectedTag: null,
      filterText: "",
      cameraYaw: -0.9,
      cameraPitch: 0.55,
      cameraDistance: 1.5,
      drag: null
    };

    function fmt(n, digits=3) {
      return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
    }

    function poseSummary(entry) {
      const p = entry.pose.position;
      return `${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)}`;
    }

    function buildMergedRows(data) {
      const map = new Map();
      for (const entry of data.live_candidates) {
        map.set(entry.tag_id, {
          tag_id: entry.tag_id,
          object_name: entry.object_name,
          category: entry.category,
          live: entry,
          recorded: null
        });
      }
      for (const entry of data.recorded_candidates) {
        const row = map.get(entry.tag_id) || {
          tag_id: entry.tag_id,
          object_name: entry.object_name,
          category: entry.category,
          live: null,
          recorded: null
        };
        row.object_name = entry.object_name || row.object_name;
        row.category = entry.category || row.category;
        row.recorded = entry;
        map.set(entry.tag_id, row);
      }
      return Array.from(map.values()).sort((a, b) => a.tag_id - b.tag_id);
    }

    function filteredRows(rows) {
      const q = state.filterText.trim().toLowerCase();
      if (!q) return rows;
      return rows.filter((row) => {
        const hay = `${row.tag_id} ${row.object_name} ${row.category}`.toLowerCase();
        return hay.includes(q);
      });
    }

    function ensureSelection(rows) {
      if (!rows.length) {
        state.selectedTag = null;
        return;
      }
      const exists = rows.some((row) => row.tag_id === state.selectedTag);
      if (!exists) {
        const recordedFirst = rows.find((row) => row.recorded) || rows[0];
        state.selectedTag = recordedFirst.tag_id;
      }
    }

    function renderDetails(selected) {
      const root = document.getElementById("detail-grid");
      if (!selected) {
        root.innerHTML = '<div class="empty">No scanned object selected yet.</div>';
        return;
      }
      const entry = selected.recorded || selected.live;
      const p = entry.pose.position;
      const rpy = entry.pose.rpy_deg;
      root.innerHTML = `
        <div>Tag ID<strong>${selected.tag_id}</strong></div>
        <div>Object<strong>${selected.object_name || "unknown"}</strong></div>
        <div>Category<strong>${selected.category || "uncategorized"}</strong></div>
        <div>Status<strong>${selected.recorded ? "recorded" : "live only"}</strong></div>
        <div>X / Y / Z<strong>${fmt(p.x)} / ${fmt(p.y)} / ${fmt(p.z)}</strong></div>
        <div>Roll / Pitch / Yaw<strong>${fmt(rpy.roll, 1)} / ${fmt(rpy.pitch, 1)} / ${fmt(rpy.yaw, 1)}</strong></div>
      `;
    }

    function renderRows(rows) {
      const root = document.getElementById("rows");
      if (!rows.length) {
        root.innerHTML = '<div class="empty">No candidates visible.</div>';
        return;
      }
      root.innerHTML = rows.map((row) => {
        const active = row.tag_id === state.selectedTag ? "active" : "";
        const source = row.recorded || row.live;
        const p = source.pose.position;
        return `
          <div class="row ${active}" data-tag="${row.tag_id}">
            <div class="row-top">
              <div>
                <div class="row-title">Tag ${row.tag_id} · ${row.object_name || "unknown"}</div>
                <div class="row-meta">${row.category || "uncategorized"}</div>
              </div>
              <div class="badges">
                ${row.live ? '<span class="badge live">live</span>' : ''}
                ${row.recorded ? '<span class="badge recorded">recorded</span>' : ''}
              </div>
            </div>
            <div class="coords">
              <div>X<span>${fmt(p.x)}</span></div>
              <div>Y<span>${fmt(p.y)}</span></div>
              <div>Z<span>${fmt(p.z)}</span></div>
            </div>
          </div>
        `;
      }).join("");
      root.querySelectorAll(".row").forEach((el) => {
        el.addEventListener("click", () => {
          state.selectedTag = Number(el.dataset.tag);
          render();
        });
      });
    }

    function quatToMatrix(q) {
      const {x, y, z, w} = q;
      return [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
      ];
    }

    function rotatePoint(p, center) {
      const dx = p.x - center.x;
      const dy = p.y - center.y;
      const dz = p.z - center.z;
      const cy = Math.cos(state.cameraYaw), sy = Math.sin(state.cameraYaw);
      const cp = Math.cos(state.cameraPitch), sp = Math.sin(state.cameraPitch);
      const x1 = cy * dx - sy * dy;
      const y1 = sy * dx + cy * dy;
      const z1 = dz;
      const x2 = x1;
      const y2 = cp * y1 - sp * z1;
      const z2 = sp * y1 + cp * z1;
      return {x: x2, y: y2, z: z2};
    }

    function project3D(p, center, width, height) {
      const cam = rotatePoint(p, center);
      const z = cam.z + state.cameraDistance;
      const scale = 320 / Math.max(z, 0.15);
      return {
        x: width / 2 + cam.x * scale,
        y: height / 2 - cam.y * scale,
        scale,
        depth: z
      };
    }

    function transformPoint(local, pose) {
      const R = quatToMatrix(pose.orientation);
      return {
        x: pose.position.x + R[0][0] * local.x + R[0][1] * local.y + R[0][2] * local.z,
        y: pose.position.y + R[1][0] * local.x + R[1][1] * local.y + R[1][2] * local.z,
        z: pose.position.z + R[2][0] * local.x + R[2][1] * local.y + R[2][2] * local.z,
      };
    }

    function sceneCenter(data) {
      const points = [];
      for (const e of [...data.live_candidates, ...data.recorded_candidates]) {
        points.push(e.pose.position);
      }
      if (!points.length) return {x: 0.6, y: 0.0, z: 0.15};
      return {
        x: points.reduce((a, p) => a + p.x, 0) / points.length,
        y: points.reduce((a, p) => a + p.y, 0) / points.length,
        z: points.reduce((a, p) => a + p.z, 0) / points.length,
      };
    }

    function colorStr(c) {
      const r = Math.round((c.r ?? 1) * 255);
      const g = Math.round((c.g ?? 1) * 255);
      const b = Math.round((c.b ?? 1) * 255);
      const a = c.a ?? 1;
      return `rgba(${r},${g},${b},${a})`;
    }

    function drawArrow(ctx, a, b, color, width=2) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      const angle = Math.atan2(b.y - a.y, b.x - a.x);
      const head = 8 + width;
      ctx.beginPath();
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(b.x - head * Math.cos(angle - Math.PI/6), b.y - head * Math.sin(angle - Math.PI/6));
      ctx.lineTo(b.x - head * Math.cos(angle + Math.PI/6), b.y - head * Math.sin(angle + Math.PI/6));
      ctx.closePath();
      ctx.fill();
    }

    function render3D(rows, data) {
      const canvas = document.getElementById("scene");
      const ctx = canvas.getContext("2d");
      const width = canvas.width, height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#0b1220";
      ctx.fillRect(0, 0, width, height);

      const center = sceneCenter(data);
      const drawables = [];

      for (let gx = -3; gx <= 3; gx++) {
        drawables.push({
          depthRef: {x: center.x + gx * 0.15, y: center.y, z: 0},
          draw() {
            const p0 = project3D({x: center.x + gx * 0.15, y: center.y - 0.45, z: 0}, center, width, height);
            const p1 = project3D({x: center.x + gx * 0.15, y: center.y + 0.45, z: 0}, center, width, height);
            ctx.strokeStyle = "#22304a";
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
          }
        });
      }
      for (let gy = -3; gy <= 3; gy++) {
        drawables.push({
          depthRef: {x: center.x, y: center.y + gy * 0.15, z: 0},
          draw() {
            const p0 = project3D({x: center.x - 0.45, y: center.y + gy * 0.15, z: 0}, center, width, height);
            const p1 = project3D({x: center.x + 0.45, y: center.y + gy * 0.15, z: 0}, center, width, height);
            ctx.strokeStyle = "#22304a";
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
          }
        });
      }

      for (const row of rows) {
        const entry = row.recorded || row.live;
        const proj = project3D(entry.pose.position, center, width, height);
        const color = row.recorded ? "#22c55e" : "#f59e0b";
        drawables.push({
          depthRef: entry.pose.position,
          draw() {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(proj.x, proj.y, row.tag_id === state.selectedTag ? 8 : 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = "#e5e7eb";
            ctx.font = "bold 13px Helvetica";
            ctx.fillText(String(row.tag_id), proj.x + 10, proj.y - 10);
          }
        });
      }

      for (const marker of data.markers || []) {
        const c = colorStr(marker.color || {});
        if (marker.type === 0 && marker.points && marker.points.length >= 2) {
          drawables.push({
            depthRef: marker.points[0],
            draw() {
              const a = project3D(marker.points[0], center, width, height);
              const b = project3D(marker.points[1], center, width, height);
              drawArrow(ctx, a, b, c, 2 + (marker.scale?.x || 0.004) * 120);
            }
          });
        } else if (marker.type === 2) {
          drawables.push({
            depthRef: marker.pose.position,
            draw() {
              const p = project3D(marker.pose.position, center, width, height);
              const r = Math.max(3, (marker.scale?.x || 0.03) * p.scale * 0.6);
              ctx.fillStyle = c;
              ctx.beginPath();
              ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
              ctx.fill();
            }
          });
        } else if (marker.type === 1) {
          drawables.push({
            depthRef: marker.pose.position,
            draw() {
              const sx = (marker.scale?.x || 0.03) / 2;
              const sy = (marker.scale?.y || 0.02) / 2;
              const sz = (marker.scale?.z || 0.02) / 2;
              const verts = [
                {x:-sx,y:-sy,z:-sz},{x:sx,y:-sy,z:-sz},{x:sx,y:sy,z:-sz},{x:-sx,y:sy,z:-sz},
                {x:-sx,y:-sy,z:sz},{x:sx,y:-sy,z:sz},{x:sx,y:sy,z:sz},{x:-sx,y:sy,z:sz},
              ].map(v => transformPoint(v, marker.pose));
              const p = verts.map(v => project3D(v, center, width, height));
              const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
              ctx.strokeStyle = c;
              ctx.lineWidth = 2;
              for (const [i,j] of edges) {
                ctx.beginPath();
                ctx.moveTo(p[i].x, p[i].y);
                ctx.lineTo(p[j].x, p[j].y);
                ctx.stroke();
              }
            }
          });
        } else if (marker.type === 9) {
          drawables.push({
            depthRef: marker.pose.position,
            draw() {
              const p = project3D(marker.pose.position, center, width, height);
              ctx.fillStyle = c;
              ctx.font = "13px Helvetica";
              ctx.fillText(marker.text || "", p.x + 6, p.y - 4);
            }
          });
        }
      }

      drawables.sort((a, b) => project3D(b.depthRef, center, width, height).depth - project3D(a.depthRef, center, width, height).depth);
      for (const item of drawables) item.draw();
      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px Helvetica";
      ctx.fillText("Base-frame 3D scene. Drag to orbit, wheel to zoom.", 20, height - 18);
    }

    function render() {
      if (!state.data) return;
      const rows = filteredRows(buildMergedRows(state.data));
      ensureSelection(rows);
      document.getElementById("live-count").textContent = state.data.live_candidates.length;
      document.getElementById("recorded-count").textContent = state.data.recorded_candidates.length;
      document.getElementById("live-status").textContent = state.data.live_status;
      document.getElementById("recorded-status").textContent = state.data.recorded_status;
      renderRows(rows);
      render3D(rows, state.data);
      renderDetails(rows.find((row) => row.tag_id === state.selectedTag));
    }

    async function refreshState() {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        state.data = await res.json();
        render();
      } catch (err) {
        console.error(err);
      }
    }

    document.getElementById("filter").addEventListener("input", (ev) => {
      state.filterText = ev.target.value || "";
      render();
    });

    const canvas = document.getElementById("scene");
    canvas.addEventListener("mousedown", (ev) => {
      state.drag = {x: ev.clientX, y: ev.clientY, yaw: state.cameraYaw, pitch: state.cameraPitch};
      canvas.classList.add("dragging");
    });
    window.addEventListener("mouseup", () => {
      state.drag = null;
      canvas.classList.remove("dragging");
    });
    window.addEventListener("mousemove", (ev) => {
      if (!state.drag) return;
      const dx = ev.clientX - state.drag.x;
      const dy = ev.clientY - state.drag.y;
      state.cameraYaw = state.drag.yaw - dx * 0.01;
      state.cameraPitch = Math.max(-1.2, Math.min(1.2, state.drag.pitch + dy * 0.01));
      render();
    });
    canvas.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      state.cameraDistance = Math.max(0.4, Math.min(4.0, state.cameraDistance + ev.deltaY * 0.001));
      render();
    }, {passive:false});

    refreshState();
    setInterval(refreshState, 800);
  </script>
</body>
</html>
"""


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

    return {
        "roll": math.degrees(roll),
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
    }


class WorkspaceSetupDashboard:
    def __init__(self):
        rospy.init_node("workspace_setup_dashboard")

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
        self.tag_ids = self._parse_int_list_param("~tag_ids", [0, 1, 2, 3, 4])
        self.markers_namespace_prefix = str(
            rospy.get_param("~markers_namespace_prefix", "/apriltag_candidates/tag_")
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
        self.host = str(rospy.get_param("~host", "127.0.0.1")).strip()
        self.port = int(rospy.get_param("~port", 8765))

        self.lock = threading.Lock()
        self.object_map = self._load_object_map()
        self.live_candidates = {}
        self.recorded_candidates = {}
        self.marker_arrays = {}
        self.live_status = "waiting_for_live_candidates"
        self.recorded_status = "waiting_for_recorded_candidates"

        rospy.Subscriber(self.live_topic, Detection2DArray, self._live_cb, queue_size=1)
        rospy.Subscriber(self.recorded_topic, Detection2DArray, self._recorded_cb, queue_size=1)
        rospy.Subscriber(self.live_status_topic, String, self._live_status_cb, queue_size=1)
        rospy.Subscriber(self.recorded_status_topic, String, self._recorded_status_cb, queue_size=1)
        for tag_id in self.tag_ids:
            rospy.Subscriber(
                "{}/markers".format("{}{}".format(self.markers_namespace_prefix, tag_id)),
                MarkerArray,
                self._marker_cb,
                callback_args=tag_id,
                queue_size=1,
            )

        self.httpd = self._make_server()
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()
        rospy.loginfo(
            "[workspace_setup_dashboard] ready at http://%s:%d",
            self.host,
            self.port,
        )

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
                "pose": {
                    "position": {
                        "x": float(pose.position.x),
                        "y": float(pose.position.y),
                        "z": float(pose.position.z),
                    },
                    "orientation": {
                        "x": float(pose.orientation.x),
                        "y": float(pose.orientation.y),
                        "z": float(pose.orientation.z),
                        "w": float(pose.orientation.w),
                    },
                    "rpy_deg": _quat_to_rpy_deg(
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ),
                },
            }
        return parsed

    def _live_cb(self, msg):
        with self.lock:
            self.live_candidates = self._parse_detections(msg)

    def _recorded_cb(self, msg):
        with self.lock:
            self.recorded_candidates = self._parse_detections(msg)

    def _live_status_cb(self, msg):
        with self.lock:
            self.live_status = str(msg.data).strip()

    def _recorded_status_cb(self, msg):
        with self.lock:
            self.recorded_status = str(msg.data).strip()

    def _parse_int_list_param(self, name, default):
        raw = rospy.get_param(name, default)
        if isinstance(raw, (list, tuple)):
            return [int(v) for v in raw]
        if isinstance(raw, str):
            txt = raw.strip().replace("[", "").replace("]", "").replace(",", " ")
            return [int(v) for v in txt.split() if v]
        return [int(v) for v in default]

    def _serialize_marker(self, marker):
        return {
            "ns": str(marker.ns),
            "id": int(marker.id),
            "type": int(marker.type),
            "action": int(marker.action),
            "pose": {
                "position": {
                    "x": float(marker.pose.position.x),
                    "y": float(marker.pose.position.y),
                    "z": float(marker.pose.position.z),
                },
                "orientation": {
                    "x": float(marker.pose.orientation.x),
                    "y": float(marker.pose.orientation.y),
                    "z": float(marker.pose.orientation.z),
                    "w": float(marker.pose.orientation.w),
                },
            },
            "scale": {
                "x": float(marker.scale.x),
                "y": float(marker.scale.y),
                "z": float(marker.scale.z),
            },
            "color": {
                "r": float(marker.color.r),
                "g": float(marker.color.g),
                "b": float(marker.color.b),
                "a": float(marker.color.a),
            },
            "points": [
                {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
                for p in marker.points
            ],
            "text": str(marker.text),
        }

    def _marker_cb(self, msg, tag_id):
        serialized = []
        for marker in msg.markers:
            if int(marker.action) == 3:  # DELETEALL
                serialized = []
                break
            if int(marker.action) != 0:
                continue
            serialized.append(self._serialize_marker(marker))
        with self.lock:
            self.marker_arrays[tag_id] = serialized

    def _bounds(self, live_vals, recorded_vals):
        entries = list(live_vals) + list(recorded_vals)
        if not entries:
            return {
                "xmin": -0.2,
                "xmax": 1.0,
                "ymin": -0.8,
                "ymax": 0.8,
            }

        xs = [entry["pose"]["position"]["x"] for entry in entries]
        ys = [entry["pose"]["position"]["y"] for entry in entries]
        margin = 0.08
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)
        if abs(xmax - xmin) < 1e-3:
            xmin -= margin
            xmax += margin
        if abs(ymax - ymin) < 1e-3:
            ymin -= margin
            ymax += margin
        return {
            "xmin": xmin - margin,
            "xmax": xmax + margin,
            "ymin": ymin - margin,
            "ymax": ymax + margin,
        }

    def state_payload(self):
        with self.lock:
            live_vals = [self.live_candidates[key] for key in sorted(self.live_candidates.keys())]
            recorded_vals = [self.recorded_candidates[key] for key in sorted(self.recorded_candidates.keys())]
            markers = []
            for tag_id in sorted(self.marker_arrays.keys()):
                markers.extend(self.marker_arrays[tag_id])
            return {
                "live_status": self.live_status,
                "recorded_status": self.recorded_status,
                "live_candidates": live_vals,
                "recorded_candidates": recorded_vals,
                "markers": markers,
                "bounds": self._bounds(live_vals, recorded_vals),
            }

    def _make_server(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = HTML_PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/api/state":
                    body = json.dumps(dashboard.state_payload()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_error(404)

            def log_message(self, fmt, *args):
                rospy.logdebug("[workspace_setup_dashboard] " + fmt, *args)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.daemon_threads = True
        return server

    def run(self):
        rospy.on_shutdown(self.shutdown)
        rospy.spin()

    def shutdown(self):
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


def main():
    WorkspaceSetupDashboard().run()


if __name__ == "__main__":
    main()
